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
import os

def get_region(wsi, level, ds, top_left_x, top_left_y, size):
    # read a downsampled squared region of wsi
    # with the top_left corner relative
    # position (pw, ph) and side of size pixels

    wl, hl = wsi.level_dimensions[level]
    # print("Downsample dimensions ({},{})".format(wl, hl))

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
        print('gen-patches.py -f <wsi> [-m <wsi_mask>] [-d downsample] '
              '[-s size] [-b threshold] [-i] [-t] [-v] [-w]', file=sys.stderr)
        return {}

    try:
        opts, args = getopt.getopt(argv, "hitf:m:d:s:wvb:", ["help", "info", "thumbnail",
                                                    "file=", "mask=", "downsample=",
                                                    "size=", "view", "write",
                                                    "background"])
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
        elif opt in ("-d", "--downsample"):
            params_dict["downsample"] = arg
        elif opt in ("-s", "--size"):
            params_dict["size"] = arg
        elif opt in ("-w", "--write"):
            params_dict["write"] = True
        elif opt in ("-v", "--view"):
            params_dict["view"] = True
        elif opt in ("-b", "--background"):
            params_dict["background"] = arg

    return params_dict


def main(argv):
    # Main

    bg_downsample = 32.0
    downsample = 4.0 # default downsampling
    size = 2048 # default patch size
    background = .995

    MAX_FIG = 4

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
            print('ERROR: Couldn\'t open file {}'.format(params_dict["imgfile"]), file=sys.stderr)
            exit(-1)
        basename, _ = os.path.splitext(os.path.basename(params_dict["imgfile"]))
    else:
        print("ERROR: No image file provided!", file=sys.stderr)
        exit(-1)

    wsi_mask = None
    if "maskfile" in params_dict.keys():
        try:
            wsi_mask = openslide.OpenSlide(params_dict["maskfile"])
            if "info" in params_dict.keys():
                print_wsi_info(wsi_mask)
        except openslide.OpenSlideError:
            print('ERROR: Couldn\'t open file {}'.format(params_dict["maskfile"]), file=sys.stderr)

    max_downsample = np.exp2(wsi.level_count - 1)

    if "downsample" in params_dict.keys():
        try:
            downsample = float(params_dict["downsample"])
        except ValueError:
            print('ERROR: Not a valid downsample', file=sys.stderr)
            exit(-1)

    if ( downsample < 1 ) or ( downsample > max_downsample):
            print('ERROR: Downsample must be between {} and {}'.format(1, max_downsample),
                file=sys.stderr)
            exit(-1)

    level = wsi.get_best_level_for_downsample(downsample)
    max_width, max_height = wsi.level_dimensions[level]
    max_size = np.minimum(max_width, max_height)

    if "size" in params_dict.keys():
        try:
            size = int(params_dict["size"])
        except ValueError:
            print('ERROR: Not a valid size', file=sys.stderr)
            exit(-1)

    if ( size < 1 ) or ( size > max_size):
            print('ERROR: Size must be between {} and {}'.format(1, max_size),
                file=sys.stderr)
            exit(-1)

    if "background" in params_dict.keys():
        try:
            background = float(params_dict["background"])/100.0
        except ValueError:
            print('ERROR: Not a valid background threshold', file=sys.stderr)
            exit(-1)

    if ( background < 0 ) or ( background > 1):
        print('ERROR: Background threshold must be between 0 and 1',
               file=sys.stderr)
        exit(-1)

    min_fg = 1 - background

    view_plt = "view" in params_dict.keys()
    write_img = "write" in params_dict.keys()

    # Compute the background thresholding values

    bg_level = wsi.get_best_level_for_downsample(bg_downsample)
    img = wsi.read_region((0, 0), bg_level, wsi.level_dimensions[bg_level])
    hsv_img = cv2.cvtColor(np.array(img.convert('RGB')),
              cv2.COLOR_RGB2HSV)
    retH, _ = cv2.threshold(hsv_img[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    retS, _ = cv2.threshold(hsv_img[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print('Threshold levels: {}, {}'.format(retH, retS))

    # Create patches

    ds = int(wsi.level_downsamples[level])
    n_grids = int(np.sqrt(int(max_width / size) + 1))+1
    print("x,y,fg")
    j = 0
    for y in range(0, max_height, size):
        if "display" in params_dict.keys():
            fig = plt.figure(num=j%MAX_FIG)
            plt.clf()
        i = 1
        for x in range(0, max_width, size):
            img_mask = None
            img = get_region(wsi, level, ds, x*ds, y*ds, size)
            if wsi_mask:
                mask = get_region(wsi_mask, level, ds, x*ds, y*ds, size)
                img_mask = mask.convert('RGB')
            if img:
                pil_img = img.conver('RGB')
                hsv_cv_img = cv2.cvtColor(pil_img,
                                cv2.COLOR_RGB2HSV)
                h = hsv_cv_img[:, :, 0]
                s = hsv_cv_img[:, :, 1]
                _, threshH = cv2.threshold(h, retH, 255, cv2.THRESH_BINARY)
                _, threshS = cv2.threshold(s, retS, 255, cv2.THRESH_BINARY)
                threshHS = cv2.bitwise_or(threshH, threshS)
                fbr = foreground_ratio(threshHS)
                print(x*ds,y*ds, sep=',', end='')
                if fbr is not None:
                    print(",{:.2f}%".format(fbr*100.0), end='')
                print('') 
                if fbr > min_fg:
                    process_img(pil_img, threshHS, view_plt, fig, n_grids,
                                i, write_img, basename, x, y, img_mask)
                    if view_plt:
                        plt.show(block=False)
                        fig.canvas.draw()
            i += 1
        j += 1
        if (j%MAX_FIG == 0) and ("display" in params_dict.keys()):
            input("Press a key to continue...")

def process_img(pil_img, threshHS, view_plt, fig, n_grids, i, write_img, basename, x, y, img_mask):
    pil_img_w_bg = cv2.bitwise_and(pil_img, pil_img, mask=threshHS)
    if view_plt:
        fig.add_subplot(n_grids, n_grids, i)
        plt.imshow(pil_img_w_bg)
    if write_img:
        try:
            cv2.imwrite('{}_p{}_{}.jpg'.format(basename,x,y),
                    cv2.cvtColor(pil_img_w_bg, cv2.COLOR_RGB2BGR))
        except cv2.error as e:
            print("ERROR: could not write image", file=sys.stderr)
            print(e, file=sys.stderr)
    if img_mask is not None:
        img_mask_w_bg = np.zeros_like(img_mask)
        img_mask_w_bg[:img_mask.shape[0], :img_mask.shape[1], 1] = img_mask[:, :, 1]
        img_mask_w_bg = cv2.bitwise_and(img_mask_w_bg,img_mask_w_bg, mask=threshHS)
        if view_plt:
            plt.imshow(img_mask_w_bg)
        if write_img:
            try:
                cv2.imwrite('{}_p{}_{}m.jpg'.format(basename,x,y),
                    cv2.cvtColor(img_mask_w_bg, cv2.COLOR_RGB2BGR))
            except cv2.error as e:
                print("ERROR: could not write image", file=sys.stderr)
                print(e, file=sys.stderr)


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
            print('WARNING: Only numbers accepted', file=sys.stderr)
            value = float('NaN')

        if (value < min) or (value > max):
            print('WARNING: {} must be between {} and {}.'.format(param_name, min, max),
                  file=sys.stderr)
            value = float('NaN')

    return value

def foreground_ratio(img, fg=255, bg=0):
    # For a two-level monochromatic image computes the ratio of foreground
    # color, fg, vs background color, bg
    if img.ndim > 2:
        print('ERROR: Only monochromatic images are accepts', file=sys.stderr)
        return None

    tmp = 1*(img == fg) + 0*(img == bg)
    return np.mean(tmp)

if __name__ == '__main__':
    main(sys.argv[1:])