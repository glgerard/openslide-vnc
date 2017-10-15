"""
Copyright (c) 2017 Gianluca Gerard

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys, getopt
import openslide
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_region(wsi, level, ds, top_left_x, top_left_y, size):
    # read a downsampled squared region of wsi
    # with the top_left corner relative
    # position (pw, ph) and side of size pixels

    wl, hl = wsi.level_dimensions[level]
    print("Downsample dimensions ({},{})".format(wl, hl))

    # Avoid negative or zero sides and sizes
    # exceeding the dimensions of the image
    # at the selected level
    if size <= 0:
        return None
    elif size > hl:
        size = hl
    elif size > wl:
        size = wl

    # Avoid out of image top left corner of
    # the wsi region
    w, h = wsi.dimensions

    if top_left_x < 0:
        top_left_x = 0
    elif top_left_x > w:
        top_left_x = w

    if top_left_y < 0:
        top_left_y = 0
    elif top_left_y > h:
        top_left_y = h

    # Avoid out of image bottom down corner of
    # the wsi region
    if top_left_x/ds + int(size) > wl:
        width = wl - int(top_left_x/ds)
    else:
        width = int(size)

    if top_left_y/ds + int(size) > hl:
        height = hl - int(top_left_y/ds)
    else:
        height = int(size)

    #print(top_left_x, top_left_y)
    #print(width, height)

    return wsi.read_region((top_left_x, top_left_y),
                           level,
                           (width, height))


def read_args(argv):
    # read the CLI arguments and return a dictionary with
    # the parameters or None
    def usage():
        print('read-image.py -f <wsi> [-m <wsi_mask>] [-i] [-t]', file=sys.stderr)
        return {}

    try:
        opts, args = getopt.getopt(argv, "hitf:m:", ["help", "info", "thumbnail",
                                                    "file=", "mask="])
    except getopt.GetoptError:
        return usage()

    params_dict = {}
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            return usage()
        elif opt in ("-f", "--file"):
            params_dict["imgfile"] = arg
        elif opt in ("-m", "--mask"):
            params_dict["maskfile"] = arg
        elif opt in ("-i", "--info"):
            params_dict["info"] = True
        elif opt in ("-t", "--thumbnail"):
            params_dict["thumbnail"] = True

    return params_dict


def main(argv):
    # Main
    params_dict = read_args(argv)
    if not params_dict:
        exit(-1)

    if "imgfile" in params_dict.keys():
        try:
            wsi = openslide.OpenSlide(params_dict["imgfile"])
            if "info" in params_dict.keys():
                print_wsi_info(wsi)
            if "thumbnail" in params_dict.keys():
                thumbnail = wsi.get_thumbnail((1024, 1024))
                thumbnail.show()
        except openslide.OpenSlideError:
            print('Error opening file {}'.format(params_dict["imgfile"]), file=sys.stderr)
            exit(-1)
    else:
        print("Error: No image file provided!", file=sys.stderr)
        exit(-1)

    wsi_mask = None
    if "maskfile" in params_dict.keys():
        try:
            wsi_mask = openslide.OpenSlide(params_dict["maskfile"])
            if "info" in params_dict.keys():
                print_wsi_info(wsi_mask)
        except openslide.OpenSlideError:
            print('Error opening file {}'.format(params_dict["maskfile"]), file=sys.stderr)

    max_downsample = np.exp2(wsi.level_count - 1)
    cont = 'y'
    while cont in ('Y', 'y'):
        downsample = get_float("Downsample", 1.0, max_downsample)
        level = wsi.get_best_level_for_downsample(downsample)
        width_pos = get_float("Width", 0.0, 1.0)
        height_pos = get_float("Height", 0.0, 1.0)
        max_size = np.minimum(wsi.level_dimensions[level][0],
                              wsi.level_dimensions[level][1])
        size = get_float("Size", 1, max_size)

        img = None
        img_mask = None
        if size > 0:
            w, h = wsi.dimensions
            level = wsi.get_best_level_for_downsample(downsample)
            print("Level", level)
            ds = wsi.level_downsamples[level]
            x = int(w*width_pos)
            y = int(h*height_pos)
            img = get_region(wsi, level, ds, x, y, size)
            if wsi_mask:
                tmp_img_mask = get_region(wsi_mask, level, ds, x, y, size)
                img_mask = cv2.cvtColor(np.array(tmp_img_mask.convert('RGB')), cv2.COLOR_RGB2BGR)

        if img:
            pil_image = np.array(img.convert('RGB'))
            open_cv_image = cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR)
            hsv_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV)
            h = hsv_cv_image[:, :, 0]
            s = hsv_cv_image[:, :, 1]
            retH, threshH = cv2.threshold(h, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            retS, threshS = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print('Threshold levels: {}, {}'.format(retH, retS))
            threshHS = cv2.bitwise_or(threshH, threshS)
            fbr = foreground_ratio(threshHS)
            if fbr:
                print("The percentage of foreground is {:.2f}%".format(fbr*100.0))
            pil_image_blk_bg = cv2.bitwise_and(pil_image, pil_image, mask=threshHS)
            f, axarr = plt.subplots(2, 2, sharey=True)
            axarr[0, 0].imshow(pil_image)
            axarr[0, 1].imshow(threshHS, 'gray')
            axarr[1, 0].imshow(pil_image_blk_bg)
            if img_mask is not None:
                cv_image_msk_bg = np.zeros_like(pil_image_blk_bg)
                cv_image_msk_bg[:img_mask.shape[0], :img_mask.shape[1], 1] = img_mask[:, :, 1]
                cv_image_msk_bg = cv2.add(pil_image_blk_bg,cv_image_msk_bg)
                axarr[1, 1].imshow(cv_image_msk_bg)
            plt.show()

        cont = input("Continue? [Y/N] ")


def print_wsi_info(wsi):
    # Print basic info on wsi
    level_count = wsi.level_count
    max_downsample = np.exp2(level_count - 1)
    print("Level count", level_count)
    print("Max downsample", max_downsample)
    for l in range(level_count):
        print("Level-{} dimensions {}".format(l,
                                              wsi.level_dimensions[l]))


def get_float(param_name, min, max):
    # Simple function to read a number with defined constraints
    value = float('NaN')
    while math.isnan(value):
        try:
            value = float(input('{} [{}, {}]: '.format(param_name, min, max)))
        except ValueError:
            print('Only numbers accepted', file=sys.stderr)
            value = float('NaN')

        if (value < min) or (value > max):
            print('{} must be between {} and {}.'.format(param_name, min, max),
                  file=sys.stderr)
            value = float('NaN')

    return value

def foreground_ratio(img, fg=255, bg=0):
    # For a two-level monochromatic image computes the ratio of foreground
    # color, fg, vs background color, bg
    if img.ndim > 2:
        print('Only monochromatic images are accepts', file=sys.stderr)
        return None

    tmp = 1*(img == fg) + 0*(img == bg)
    return np.mean(tmp)

if __name__ == '__main__':
    main(sys.argv[1:])