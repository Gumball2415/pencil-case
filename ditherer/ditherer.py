# Image ditherer
# Copyright 2024 Persune
# MIT-0

import numpy as np
import sys, argparse
from os import path
from PIL import Image, ImageColor, ImagePalette

VERSION = "0.4.1"

def parse_argv(argv):
    parser=argparse.ArgumentParser(
        description="Image ditherer",
        epilog="version " + VERSION)
    parser.add_argument(
        "--deltae",
        type = str,
        help = "Method of delta E, or finding closest color. default is \"sRGB\".",
        choices=[
            "sRGB",
            "CIE94"
        ],
        default = "sRGB")
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
        "--scale-threshold",
        type=str,
        help="method for scaling the threshold according to the dither matrix's characteristics (experimental).",
        choices=[
            "no",
            "mask-size",
            "unique-count"
        ],
        default = "no")
    parser.add_argument(
        "--color-threshold",
        help="colors the dither mask locally according to the pixel color",
        action="store_true")
    parser.add_argument(
        "-pal",
        "--palette",
        nargs="+",
        help="palette array, in #rrggbb hex")
    parser.add_argument(
        "mask",
        help="input .png dither mask")
    parser.add_argument(
        "input",
        help="input .png file. saved as input_dither_mask.png")
    return parser.parse_args(argv[1:])

# http://brucelindbloom.com/index.html?Math.html
srgb_to_xyz_M = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]])

def srgb_to_xyz(rgb_array):
    return np.linalg.matmul(srgb_to_xyz_M,rgb_array)

# https://en.wikipedia.org/wiki/Standard_illuminant#D65_values
d65_x = 0.31272
d65_y = 0.32903
d65_Y = 100

X_r = (d65_x*d65_Y) / d65_y
Y_r = d65_Y
Z_r = ((1-d65_x-d65_y) * d65_Y)/d65_y
epsilon = 216/24389
kappa = 24389/27

def xyz_to_lab(xyz_array):
    x_r = xyz_array[0]/X_r
    y_r = xyz_array[1]/Y_r
    z_r = xyz_array[2]/Z_r

    f_x = np.cbrt(x_r) if x_r > epsilon else (kappa*x_r+16)/116
    f_y = np.cbrt(y_r) if y_r > epsilon else (kappa*y_r+16)/116
    f_z = np.cbrt(z_r) if z_r > epsilon else (kappa*z_r+16)/116

    L=116*f_y-16
    a=500*(f_x-f_y)
    b=200*(f_y-f_z)

    return np.array([L, a, b])

# finds index with shortest CIE94 distance
# https://bisqwit.iki.fi/story/howto/dither/jy/#Appendix%203ColorComparisons
# http://brucelindbloom.com/Eqn_DeltaE_CIE94.html
def closest_CIE94_index(px_in, palette, args=None):
    px_in = np.array(xyz_to_lab(srgb_to_xyz(px_in)))
    palette = np.array([xyz_to_lab(srgb_to_xyz(pal)) for pal in palette])
    distances = []

    K_L = 2 if args.CIE94_tex else 1
    # K_C and K_H is 1 by default
    K_1 = 0.048 if args.CIE94_tex else 0.045
    K_2 = 0.014 if args.CIE94_tex else 0.015

    for pal in palette:
        # convert colors to LC
        delta_L = pal[0] - px_in[0]
        delta_a = pal[1] - px_in[1]
        delta_b = pal[2] - px_in[2]

        C_1 = np.sqrt(pal[1]**2     +   pal[2]**2)
        C_2 = np.sqrt(px_in[1]**2   +   px_in[2]**2)
        delta_C = C_1 - C_2
    
        # using C_12 because of bisqwit
        C_12 = np.sqrt(C_1*C_2)
        # S_L = 1
        S_C = 1 + K_1*C_12
        S_H = 1 + K_2*C_12

        # H is compared in a different way than just euclidean distance
        # removed sqrt for optimization
        delta_Hsqr = delta_a**2 + delta_b**2 - delta_C**2

        delta_E = np.sqrt(
            (delta_L**2/K_L**2)+(delta_C**2/S_C**2)+(delta_Hsqr/S_H**2)
        )

        distances.append(delta_E)

    distances = np.array(distances)
    index_of_smallest = np.where(distances==np.amin(distances))
    return index_of_smallest

def closest_index(px_in, palette, args=None):
    distances = np.sqrt(np.sum((palette-px_in)**2,axis=1))
    index_of_smallest = np.where(distances==np.amin(distances))
    return index_of_smallest

deltae_func = {
    "sRGB": closest_index,
    "CIE94": closest_CIE94_index,
}

# Threshold scaling works by reducing the threshold on smaller matrices using an exponential polynomial law.
# The higher the denominator, the less the threshold will be increased for lower metric values.
SCALE_THRESHOLD_FACTOR = 1/5

def main(argv=None):
    args = parse_argv(argv or sys.argv)

    deltae = deltae_func[args.deltae]
    
    palette = np.array([ImageColor.getrgb(color) for color in args.palette], dtype=np.float64)
    # flatten palette?
    # palette = [value for sublist in palette for value in sublist]

    with Image.open(args.mask) as immask:
        if immask.mode != "L":
            print("dither mask is not 8-bit grayscale! converting...")
            immask = immask.convert(mode="L")

        dithermask = np.array(immask, dtype=np.float64)
    d_h, d_w = dithermask.shape

    threshold: np.float64
    match args.scale_threshold:
        case "no":
            metric = 1
        case "size":
            metric = np.float64(d_w * d_h) / 2
        case "unique-count":
            metric = np.float64(len(np.unique_values(dithermask))) / 2

    scale_factor = np.pow(metric, SCALE_THRESHOLD_FACTOR)
    
    threshold = args.threshold / scale_factor
    #print(f"metric: {metric}, scalefactor: {scale_factor}, threshold {args.threshold}->{threshold}")

    with Image.open(args.input) as im:
        # convert im to RGB only
        im = im.convert(mode="RGB")
        imarr = np.array(im, dtype=np.float64)
        imout = np.zeros(imarr.shape[0:2], dtype=np.uint8)
        i_h, i_w, _ = imarr.shape

    # rescale to 0-1
    imarr /= 255.0
    palette /= 255.0
    dithermask /= 255.0

    # rescale dithermask 
    threshold_clamped = min(threshold, 1.0)
    dithermask -= np.average(dithermask) * threshold_clamped

    # normalize individual pixels of image for color thresholding, if specified
    # this is done in the nonlinear domain to preserve chroma
    imarr_normalized: np.array
    if args.color_threshold:
        imarr_normalized = np.apply_along_axis(lambda x: x / np.max(x) if np.max(x) > 0. else 1., 2, imarr)

    # convert to linear light
    # apply eotf gamma to image before dither
    for y in range(imarr.shape[0]):
        for x in range(imarr.shape[1]):
            for c in range(imarr.shape[2]):
                imarr[y,x,c] = min(max(imarr[y,x,c]**(args.gamma), 0.0), 1.0)

    # extend dithermask to cover the whole image
    full_dithermask = np.tile(dithermask, [int(np.ceil(i_h/d_h)), int(np.ceil(i_w/d_w))])[0:i_h, 0:i_w]
    if args.color_threshold:
        full_dithermask = full_dithermask[..., None] * (imarr_normalized * threshold_clamped + (1 - threshold_clamped))

    # modified ordered dithering
    # NOTE: This is not Yliluoma dithering. ---- Kagamiin~
    # https://bisqwit.iki.fi/story/howto/dither/jy/#Algorithms
    for y in range(imout.shape[0]):
        for x in range(imout.shape[1]):
            px_in = imarr[y,x]
            px_mask = full_dithermask[y,x]
            px_attempt = px_in + (px_mask * threshold)
            index = deltae(px_attempt, palette, args)
            imout[y,x] = index[0][0]

    pil_image = Image.fromarray(imout, "P")
    pil_image.putpalette(np.concatenate([[int(c[0]), int(c[1]), int(c[2])] for c in palette * 255]).tolist(), rawmode="RGB")
    pil_image.save(f"{path.splitext(args.input)[0]}_{path.splitext(path.basename(args.mask))[0]}_dither.png", format="PNG", optimize=True)

if __name__ == '__main__':
    main(sys.argv)
