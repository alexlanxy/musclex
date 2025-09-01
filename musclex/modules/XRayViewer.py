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

import fabio
import numpy as np
try:
    from ..utils.file_manager import fullPath, ifHdfReadConvertless
    from ..utils.image_processor import *
except: # for coverage
    from utils.file_manager import fullPath, ifHdfReadConvertless
    from utils.image_processor import *

class XRayViewer:
    """
    A class for Quadrant Folding processing - go to process() to see all processing steps
    """
    def __init__(self, img_path, img_name, file_list=None, extension=''):
        """
        Initial value for QuadrantFolder object
        :param img_path: directory path of input image
        :param img_name: image file name
        """
        self.img_name = img_name
        # Locate corresponding loader spec by display name
        selected_index = 0
        if isinstance(file_list, list) and len(file_list) >= 2 and isinstance(file_list[0], list):
            try:
                selected_index = next((i for i, item in enumerate(file_list[0]) if item == img_name), 0)
            except Exception:
                selected_index = 0

        source = None
        if isinstance(file_list, list) and len(file_list) >= 2 and isinstance(file_list[1], list) and selected_index < len(file_list[1]):
            source = file_list[1][selected_index]

        # Lazy load based on loader spec
        if isinstance(source, np.ndarray):
            self.orig_img = source
        elif isinstance(source, tuple):
            # Tuple forms:
            # ("tiff", abs_path) or ("h5", abs_path, frame_idx)
            kind = source[0]
            if kind == "tiff" and len(source) == 2:
                try:
                    self.orig_img = fabio.open(source[1]).data
                except Exception:
                    exit
            elif kind == "h5" and len(source) == 3:
                abs_path, frame_idx = source[1], int(source[2])
                try:
                    fab = fabio.open(abs_path)
                    # Single frame fast path
                    if getattr(fab, 'nframes', 1) == 1 or frame_idx == 0:
                        data = fab.data if frame_idx == 0 else fab.get_frame(frame_idx).data
                    else:
                        frame = fab.get_frame(frame_idx)
                        data = frame.data
                    self.orig_img = data
                except Exception:
                    exit
                finally:
                    try:
                        fab.close()
                    except Exception:
                        pass
            else:
                # Unexpected spec; fall back to direct open
                try:
                    self.orig_img = fabio.open(fullPath(img_path, img_name)).data
                except Exception:
                    exit
        else:
            # Backward compatibility: extension-based or direct
            if extension in ('.hdf5', '.h5'):
                try:
                    # Old path used to pass ndarray in file_list[1]
                    self.orig_img = source if source is not None else fabio.open(fullPath(img_path, img_name)).data
                except Exception:
                    exit
            else:
                try:
                    self.orig_img = fabio.open(fullPath(img_path, img_name)).data
                except Exception:
                    exit

        self.orig_img = ifHdfReadConvertless(img_name, self.orig_img)
        self.orig_img = self.orig_img.astype("float32")
        self.orig_image_center = None
        self.hist = []
        self.dl, self.db = 0, 0

        self.info = {}

    def getRotatedImage(self, angle, center):
        """
        Get rotated image by angle while image = original input image, and angle = self.info["rotationAngle"]
        """
        img = np.array(self.orig_img, dtype="float32")
        b, l = img.shape
        rotImg, _, _ = rotateImage(img, center, angle)

        # Cropping off the surrounding part since we had already expanded the image to maximum possible extent in centerize image
        bnew, lnew = rotImg.shape
        db, dl = (bnew - b)//2, (lnew-l)//2
        final_rotImg = rotImg[db:bnew-db, dl:lnew-dl]
        self.dl, self.db = dl, db # storing the cropped off section to recalculate coordinates when manual center is given

        return final_rotImg

    def findCenter(self):
        if 'center' in self.info:
            return
        print("Center is being calculated ... ")
        self.orig_image_center = getCenter(self.orig_img)
        self.orig_img, self.info['center'] = processImageForIntCenter(self.orig_img, self.orig_image_center)
        print("Done. Center = "+str(self.info['center']))