# Image ditherer
# Copyright 2024 Persune
# MIT-0

import numpy as np
import sys, argparse, os
from PIL import Image, ImageColor, ImagePalette

VERSION = "0.2.0"

# http://brucelindbloom.com/index.html?Math.html
def srgb_to_xyz(rgb_array):
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]])
    return np.linalg.matmul(M,rgb_array)

def xyz_to_lab(xyz_array):
    X = xyz_array[0]
    Y = xyz_array[1]
    Z = xyz_array[2]

    # https://en.wikipedia.org/wiki/Standard_illuminant#D65_values
    d65_x = 0.31272
    d65_y = 0.32903
    d65_Y = 100

    X_r = (d65_x*d65_Y) / d65_y
    Y_r = d65_Y
    Z_r = ((1-d65_x-d65_y) * d65_Y)/d65_y

    x_r = X/X_r
    y_r = Y/Y_r
    z_r = Z/Z_r
    
    epsilon = 216/24389
    kappa = 24389/27

    f_x = np.cbrt(x_r) if x_r > epsilon else (kappa*x_r+16)/116
    f_y = np.cbrt(y_r) if y_r > epsilon else (kappa*y_r+16)/116
    f_z = np.cbrt(z_r) if z_r > epsilon else (kappa*z_r+16)/116

    L=116*f_y-16
    a=500*(f_x-f_y)
    b=200*(f_y-f_z)

    return np.array([L, a, b])

def lab_to_lch(lab_array):
    L = lab_array[0]
    a = lab_array[1]
    b = lab_array[2]
    
    C = np.sqrt(a**2+b**2)
    h = np.atan2(b,a)
    
    return np.array([L, C, h])

# finds index with shortest CIE94 distance
# https://bisqwit.iki.fi/story/howto/dither/jy/#Appendix%203ColorComparisons
# http://brucelindbloom.com/index.html?Math.html
def closest_CIE94_index(px_in, palette, textile=False):
    px_in = np.array(px_in)
    palette = np.array(palette)
    # convert colors to CIELAB
    px_Lab = xyz_to_lab(srgb_to_xyz(px_in))
    pal_Lab = [xyz_to_lab(srgb_to_xyz(pal)) for pal in palette]
    distances = []

    for pal in pal_Lab:
        delta_L = pal[0] - px_Lab[0]
        delta_a = pal[1] - px_Lab[1]
        delta_b = pal[2] - px_Lab[2]
        C_1 = np.sqrt(pal[1]**2+pal[2]**2)
        C_2 = np.sqrt(px_Lab[1]**2+px_Lab[2]**2)
        C_12 = np.sqrt(C_1*C_2)
        delta_C = C_1 - C_2
        delta_Hsqr = delta_a**2 + delta_b**2 - delta_C**2
        K_1 = 0.048 if textile else 0.045
        K_2 = 0.014 if textile else 0.015
        k_L = 2 if textile else 1
        S_C = 1 + K_1*C_12
        S_H = 1 + K_2*C_12
        delta_E = np.sqrt(
            (delta_L**2/k_L**2)+(delta_C**2/S_C**2)+(delta_Hsqr/S_H**2)
        )
        distances.append(delta_E)

    distances = np.array(distances)
    index_of_smallest = np.where(distances==np.amin(distances))[0][0]
    return index_of_smallest

# finds index of closest color
def closest_index(px_in, palette):
    px_in = np.array(px_in)
    palette = np.array(palette)

    distances = np.sqrt(np.sum((palette-px_in)**2,axis=1))
    index_of_smallest = np.where(distances==np.amin(distances))[0][0]
    return index_of_smallest

def main(argv=None):
    def parse_argv(argv):
        parser=argparse.ArgumentParser(
            description="Image ditherer",
            epilog="version " + VERSION)
        parser.add_argument(
            "-CIE94",
            help="compares colors in CIE94 Delta-E. default is False",
            action="store_true")
        parser.add_argument(
            "-gam",
            "--gamma",
            help="EOTF gamma setting. default is 2.2",
            type=np.float64,
            default=2.2)
        parser.add_argument(
            "--CIE94-tex",
            help="sets application constants for CIE94 Delta-E to textile mode. default is False.",
            action="store_true")
        parser.add_argument(
            "--threshold",
            help="adjusts dither threshold. default is 0.5",
            type=np.float64,
            default=0.5)
        parser.add_argument(
            "--palette",
            "-pal",
            nargs="+",
            help="palette array, in #rrggbb hex")
        parser.add_argument(
            "mask",
            help="input .png dither mask")
        parser.add_argument(
            "input",
            help="input .png file")
        parser.add_argument(
            "output",
            help="output .png file")    
        return parser.parse_args(argv[1:])
    args = parse_argv(argv or sys.argv)
    
    palette = np.array([ImageColor.getrgb(color) for color in args.palette], dtype=np.float64)
    # flatten palette?
    # palette = [value for sublist in palette for value in sublist]

    with Image.open(args.mask) as immask:
        dithermask = np.array(immask, dtype=np.float64)
        with Image.open(args.input) as im:
            imarr = np.array(im, dtype=np.float64)
            imout = np.zeros(imarr.shape, dtype=np.float64)
            # rescale to 0-1
            imarr /= 255.0
            palette /= 255.0
            dithermask /= 255.0
            n = dithermask.shape[0]
            # apply 2.2 gamma eotf to image before dither
            for y in range(imarr.shape[0]):
                for x in range(imarr.shape[1]):
                    for c in range(imarr.shape[2]):
                        imarr[y,x,c] = min(max(imarr[y,x,c]**(2.2), 0.0), 1.0)
            # standard ordered dithering
            # https://bisqwit.iki.fi/story/howto/dither/jy/#Algorithms
            for y in range(imarr.shape[0]):
                for x in range(imarr.shape[1]):
                    px_in = imarr[y,x]
                    px_mask = dithermask[(y%n),(x%n)]
                    px_attempt = px_in + px_mask * args.threshold
                    index = closest_CIE94_index(px_attempt, palette, args.CIE94_tex) if args.CIE94 else closest_index(px_attempt, palette)
                    imout[y,x] = palette[index]
                    
            # closest_color([0.5,0.5,0], palette)
            imout = np.uint8(np.around(imout*255))
            pil_image = Image.fromarray(imout)
            pil_image.save(args.output, format="PNG", optimize=True)

if __name__ == '__main__':
    main(sys.argv)
