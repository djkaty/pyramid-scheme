# Single-threaded: ~9.15 seconds

import sys
import os.path
import glob
import time
import cv2
import numpy as np
from multiprocessing import Pool
from python.fs_pyramid import get_pyramid_fusion


def stack_filter(images, threshold):
    def compute_fv(image, gaussian_blur_kernel_size=5, laplacian_kernel_size=5):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), 0
        )
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=laplacian_kernel_size)
        mean, stddev = cv2.meanStdDev(laplacian)
        stddev = stddev[0][0] / 25.0
        return stddev * stddev

    focus = [compute_fv(image) for image in images]
    print(["%.2f" % x for x in focus])
    max_index = focus.index(max(focus))
    print(
        "Max at %d/%d (%d%%)" % (max_index, len(focus), 100 * max_index // len(focus))
    )
    imageset = [image for image, fv in zip(images, focus) if fv >= threshold]
    if len(imageset) < 5:
        max_fv = max(focus) * 0.66
        imageset = [image for image, fv in zip(images, focus) if fv >= max_fv]
    return imageset


def stack_images(images):
    return get_pyramid_fusion(np.asarray(images))


def stack_dir(dirpath, outdirpath=None, overwrite=False):
    if outdirpath is None:
        dest = dirpath + ".jpg"
    else:
        dest = os.path.join(outdirpath, os.path.basename(dirpath) + ".jpg")
    if not overwrite and os.path.exists(dest):
        return
    print("Reading %s" % dirpath)
    images = [
        cv2.imread(file, cv2.IMREAD_UNCHANGED)
        for file in glob.glob(os.path.join(dirpath, "*.*"))
    ]
    images = stack_filter(images, 1.0)
    print("Stacking %d images in %s" % (len(images), dirpath))
    stacked = stack_images(images)
    print("Writing %s" % dest)
    cv2.imwrite(dest, stacked)


if __name__ == "__main__":
    inputspec = sys.argv[1]
    try:
        outdirpath = sys.argv[2]
    except:
        outdirpath = None

    if "*" in inputspec:
        try:
            processes = int(sys.argv[3])
        except:
            processes = 4

        with Pool(processes=processes) as tp:
            ars = []
            for dirpath in glob.glob(inputspec):
                if os.path.isdir(dirpath):
                    ars.append(tp.apply_async(stack_dir, (dirpath, outdirpath, True)))
            for ar in ars:
                ar.wait()
    else:
        stack_dir(inputspec, outdirpath, True)
