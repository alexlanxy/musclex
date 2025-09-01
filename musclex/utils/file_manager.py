"""
Copyright 1999 Illinois Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL ILLINOIS INSTITUTE OF TECHNOLOGY BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of Illinois Institute
of Technology shall not be used in advertising or otherwise to promote
the sale, use or other dealings in this Software without prior written
authorization from Illinois Institute of Technology.
"""

import os
from os.path import split, exists, join
import numpy as np
import fabio
#from ..ui.pyqt_utils import *
from .hdf5_manager import loadFile
from PySide6.QtWidgets import QMessageBox

input_types = ['adsc', 'cbf', 'edf', 'fit2d', 'mar345', 'marccd', 'hdf5', 'h5', 'pilatus', 'tif', 'tiff', 'smv']

def getFilesAndHdf(dir_path):
    """
    Give the image files and hdf files in a folder selected
    :param dir_path: directory path
    :return: image list, hdf list
    """
    fileList = os.listdir(dir_path)
    imgList = []
    hdfList = []

    for f in fileList:
        full_file_name = fullPath(dir_path, f)
        if isImg(full_file_name):
            imgList.append(f)
        else:
            toks = f.split('.')
            if toks[-1] == 'hdf':
                hdfList.append(f)

    return imgList, hdfList

def getBlankImageAndMask(path):
    """
    Give the blank image and the mask threshold saved in settings
    :return: blankImage, mask threshold
    """
    mask_file = join(join(path, 'settings'),'mask.tif')
    blank_file = join(join(path, 'settings'),'blank.tif')
    mask = None
    blank_img = None
    if exists(mask_file):
        mask = fabio.open(mask_file).data
    if exists(blank_file):
        blank_img = fabio.open(blank_file).data
    return blank_img, mask

def getMaskOnly(path):
    """
    Give only the mask threshold
    :param path: file path
    :return: mask threshold
    """
    maskonly_file = join(join(path, 'settings'),'maskonly.tif')
    if exists(maskonly_file):
        return fabio.open(maskonly_file).data
    return None

def getImgFiles(fullname, headless=False):
    """
    Get directory, all image-like entries in the same directory and current file index.
    Directory may contain TIFFs, single-image HDF5, and multi-image HDF5.
    Returns a unified list of display names and a parallel list of lazy loader specs.
    :param fullname: absolute path to a file selected by user
    :return: (dir_path, imgList, current, fileList, ext)
             - dir_path: directory string
             - imgList: sorted list of display names (strings)
             - current: index of the selected entry in imgList
             - fileList: [imgList, loader_specs]
                   loader_specs contains tuples describing how to load on demand:
                     ("tiff", abs_path)
                     ("h5", abs_path, frame_index)
             - ext: '.mixed' to indicate unified mixed-mode
    """
    dir_path, filename = split(str(fullname))
    dir_path = str(dir_path)
    filename = str(filename)
    _, selected_ext = os.path.splitext(str(filename))

    # Collect optional filter from a .txt list
    failedcases = [] if selected_ext == ".txt" else None
    if failedcases is not None:
        for line in open(fullname, "r"):
            failedcases.append(line.rstrip('\n'))

    # Build unified entries: list of (display_name, loader_spec)
    entries = []
    try:
        dir_list = os.listdir(dir_path)
    except Exception:
        return None, None, None, None, None

    for f in dir_list:
        if failedcases is not None and f not in failedcases:
            continue
        full_file_name = fullPath(dir_path, f)
        base, ext = os.path.splitext(f)

        # Skip calibration artifact
        if f == "calibration.tif":
            continue

        # Standard images (non-HDF5)
        if isImg(full_file_name) and ext.lower() not in ('.hdf5', '.h5'):
            entries.append((f, ("tiff", full_file_name)))
            continue

        # HDF5 images: enumerate frames lazily
        if ext.lower() in ('.hdf5', '.h5'):
            try:
                fab = fabio.open(full_file_name)
                nframes = getattr(fab, 'nframes', 1)
                # Always create a pseudo-name per frame to unify stepping
                if nframes <= 1:
                    disp = f"{base}_00001{ext}"
                    entries.append((disp, ("h5", full_file_name, 0)))
                else:
                    # enumerate all frames
                    for i in range(nframes):
                        disp = f"{base}_{i+1:05d}{ext}"
                        entries.append((disp, ("h5", full_file_name, i)))
            except Exception:
                # Invalid/corrupt HDF5 → skip silently for fast stepping
                continue
            finally:
                try:
                    fab.close()
                except Exception:
                    pass

    # Sort by display name for stable stepping
    entries.sort(key=lambda x: x[0])

    imgList = [name for name, _ in entries]
    loader_specs = [spec for _, spec in entries]

    # Determine current index based on the selected file
    current = 0
    if imgList:
        if selected_ext.lower() in ('.hdf5', '.h5'):
            # If user picked an HDF5, default to its first frame pseudo-name
            base, ext = os.path.splitext(filename)
            preferred = f"{base}_00001{ext}"
            if preferred in imgList:
                current = imgList.index(preferred)
            else:
                # fallback: first entry with same base
                same = [i for i, n in enumerate(imgList) if n.startswith(base + '_') and n.endswith(ext)]
                current = same[0] if same else 0
        else:
            # Plain images match by original file name
            if filename in imgList:
                current = imgList.index(filename)
            else:
                current = 0

    # Return unified structure; ext is mixed to disable H5-only GUI affordances
    fileList = [imgList, loader_specs]
    return dir_path, imgList, current, fileList, '.mixed'

def fullPath(filePath, fileName):
    """
    Combine a path and file name to get full file name
    :param filePath: directory (string)
    :param fileName: file name (string)
    :return: filePath/filename (string)
    """
    # if filePath[-1] == '/':
    #     return filePath+fileName
    # else:
    #     return filePath+"/"+fileName
    return os.path.join(filePath, fileName)

def isImg(fileName):
    """
    Check if a file name is an image file
    :param fileName: (str)
    :return: True or False
    """
    nameList = fileName.split('.')
    return nameList[-1] in input_types

def validateImage(fileName, showDialog=True):
    try:
        test = fabio.open(fileName).data
        return True
    except Exception:
        if showDialog:
            infMsg = QMessageBox()
            infMsg.setText('Error opening file: ' + fileName)
            infMsg.setInformativeText("Fabio could not open .TIFF File. File is either corrupt or invalid.")
            infMsg.setStandardButtons(QMessageBox.Ok)
            infMsg.setIcon(QMessageBox.Information)
            infMsg.exec_()
        return False

def isHdf5(fileName):
    """
    Check if a file name is an hdf5 file
    :param fileName: (str)
    :return: True or False
    """
    nameList = fileName.split('.')
    return nameList[-1] in ('hdf5', 'h5')

def ifHdfReadConvertless(fileName, img):
    """
    Check if a file name is an hdf5 file
    and convert it to be directly readable without converting to tiff
    :param fileName, img: (str), (array)
    :return: img converted
    """
    if isHdf5(fileName):
        img = img.astype(np.int32)
        img[img==4294967295] = -1
    return img

def createFolder(path):
    """
    Create a folder if it doesn't exist
    :param path: full path of creating directory
    :return:
    """
    if not exists(path):
        os.makedirs(path)
