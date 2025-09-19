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
from concurrent.futures import ProcessPoolExecutor
import hashlib
import time

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

    # Fast path: immediate imgList/specs from cache+parallel scan
    imgList, loader_specs = scan_directory_images_cached(dir_path, failedcases)

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

# --------------------- Fast, cached, multiprocessing directory scan ---------------------
_SCAN_CACHE = {}

def _dir_signature(dir_path):
    try:
        entries = []
        with os.scandir(dir_path) as it:
            for e in it:
                if e.is_file():
                    try:
                        stat = e.stat()
                        entries.append((e.name, stat.st_size, int(stat.st_mtime)))
                    except Exception:
                        # best-effort; skip entries we cannot stat
                        continue
        entries.sort()
        h = hashlib.sha256()
        for name, sz, mt in entries:
            h.update(name.encode('utf-8', errors='ignore'))
            h.update(str(sz).encode())
            h.update(str(mt).encode())
        return h.hexdigest()
    except Exception:
        return None

def _h5_nframes(path):
    try:
        f = fabio.open(path)
        n = getattr(f, 'nframes', 1)
        try:
            f.close()
        except Exception:
            pass
        return n
    except Exception:
        return 0

def scan_directory_images_cached(dir_path, failedcases=None, max_workers=None):
    """
    Scan a directory for TIFF and HDF5 images and return unified (imgList, loader_specs).
    Uses a cache keyed by directory content signature. HDF5 frame counts are computed
    in parallel using processes. Frames are NOT loaded.
    """
    sig = _dir_signature(dir_path)
    if sig is not None and dir_path in _SCAN_CACHE and _SCAN_CACHE[dir_path][0] == sig:
        return _SCAN_CACHE[dir_path][1]

    entries = []
    h5_files = []

    try:
        file_names = os.listdir(dir_path)
    except Exception:
        return [], []

    for f in file_names:
        if failedcases is not None and f not in failedcases:
            continue
        full_file_name = fullPath(dir_path, f)
        base, ext = os.path.splitext(f)
        if f == "calibration.tif":
            continue
        if ext.lower() in ('.hdf5', '.h5'):
            h5_files.append((base, ext, full_file_name))
        elif isImg(full_file_name) and ext.lower() not in ('.hdf5', '.h5'):
            entries.append((f, ("tiff", full_file_name)))

    # Count HDF5 frames in parallel
    if h5_files:
        if max_workers is None:
            try:
                max_workers = max(2, min(8, os.cpu_count() or 2))
            except Exception:
                max_workers = 2
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            paths = [p for _, _, p in h5_files]
            nframes_list = list(pool.map(_h5_nframes, paths))
        for (base, ext, path), nframes in zip(h5_files, nframes_list):
            if nframes <= 0:
                continue
            if nframes == 1:
                disp = f"{base}_00001{ext}"
                entries.append((disp, ("h5", path, 0)))
            else:
                for i in range(nframes):
                    disp = f"{base}_{i+1:05d}{ext}"
                    entries.append((disp, ("h5", path, i)))

    entries.sort(key=lambda x: x[0])
    imgList = [n for n, _ in entries]
    specs = [s for _, s in entries]

    if sig is not None:
        _SCAN_CACHE[dir_path] = (sig, (imgList, specs))

    return imgList, specs

# --------------------- Reusable helpers for GUI provisional selection ---------------------
def build_provisional_selection(fullname):
    """
    Build a provisional selection from a single chosen file for immediate GUI display.
    Returns a tuple compatible with getImgFiles, but without scanning the whole directory:
      (dir_path, imgList, current_index, fileList, ext)

    - For HDF5, creates a pseudo-display name like base_00001.ext and a loader spec ("h5", path, 0)
    - For TIFF and other supported images, uses the filename and a loader spec ("tiff", path)
    - ext is set to '.mixed' to indicate unified mixed-mode
    """
    dir_path, sel_name = split(str(fullname))
    dir_path = str(dir_path)
    sel_name = str(sel_name)
    base, ext = os.path.splitext(sel_name)

    if ext.lower() in ('.h5', '.hdf5'):
        disp = f"{base}_00001{ext}"
        imgList = [disp]
        loader_specs = [("h5", os.path.join(dir_path, sel_name), 0)]
    else:
        imgList = [sel_name]
        loader_specs = [("tiff", os.path.join(dir_path, sel_name))]

    current = 0
    fileList = [imgList, loader_specs]
    return dir_path, imgList, current, fileList, '.mixed'

def async_scan_directory(dir_path, on_done):
    """
    Start a background scan of a directory using scan_directory_images_cached and invoke
    on_done(imgList, specs) when finished. Returns the Thread object.

    Note: on_done may be executed on a non-GUI thread; if using Qt, marshal back to the main
    thread (e.g., via signals/QTimer) before touching widgets.
    """
    import threading

    def _worker():
        imgList, specs = scan_directory_images_cached(dir_path)
        try:
            on_done(imgList, specs)
        except Exception:
            pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
