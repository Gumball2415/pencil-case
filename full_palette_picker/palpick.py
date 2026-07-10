#!/usr/bin/env python
# full_palette palette picker

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import argparse
from PIL import Image
import numpy as np

VERSION = "0.2.0"

def parse_argv(argv):
    parser=argparse.ArgumentParser(
        description="full_palette palette picker",
        epilog="version " + VERSION)
    # output options
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="debug messages")
    parser.add_argument(
        "input",
        help="Input full_palette screenshot, index matched by input palette.\
            Output will be saved as \"input.pal\".",
        type=str)
    parser.add_argument(
        "grid_start",
        help="Top-left coordinate of starting grid cell (X, Y).",
        nargs=2,
        type=int)
    parser.add_argument(
        "grid_size",
        help="Grid dimensions (X, Y).",
        nargs=2,
        type=int)
    parser.add_argument(
        "cell_size",
        help="Cell dimensions (X, Y).",
        nargs=2,
        type=int)
    parser.add_argument(
        "crop",
        help="How much to crop into the cell (Top, Left, Right, Bottom).",
        nargs=4,
        type=int)
    return parser.parse_args(argv[1:])

def crop_swatch(
    fulpal_im: Image.Image,
    index: int,
    grid_start: tuple[int, int],
    grid_size: tuple[int, int],
    cell_size: tuple[int, int],
    crop: tuple[int, int, int, int],
    debug: bool
):
    """
    Given an index and grid attributes, crop a solid swatch of color and return
    its color in 64-bit floating point.

    :param fulpal_im: The full_palette screenshot
    :type fulpal_im: Image.Image
    :param index: Cell index to access,
    :type index: intgrid_size
    :param grid_start: Top-left coordinate of starting grid cell
    :type grid_start: tuple[int, int]
    :param grid_size: Grid dimensions
    :type grid_size: tuple[int, int]
    :param cell_size: Cell dimensions
    :type cell_size: tuple[int, int]
    :param crop: How much to crop into the cell (Top, Left, Right, Bottom)
    :type crop: tuple[int, int, int, int]
    :param debug: Enable debug output
    :type debug: bool

    :return: Cropped swatch image
    :rtype: Image.Image
    """
    index_x = index % grid_size[0]
    index_y = int(index / grid_size[0])
    left = index_x*cell_size[0] + grid_start[0] + crop[1]
    top = index_y*cell_size[1] + grid_start[1] + crop[0]
    right = ((index_x+1)*cell_size[0]) + grid_start[0] - crop[2]
    bottom = ((index_y+1)*cell_size[1]) + grid_start[1] - crop[3]

    swatch = fulpal_im.crop((left, top, right, bottom))
    if debug:
        swatch.save(os.path.join("debug", f"swatch_{index}.png"))

    swatch = np.asarray(swatch, dtype=np.float64)
    swatch /= 255.
    swatch = np.average(swatch, axis=(0, 1))

    return swatch

def main(argv=None):
    args = parse_argv(argv or sys.argv)
    fulpal_im = Image.open(args.input)

    colors = np.zeros((args.grid_size[0], args.grid_size[1], 3), dtype=np.float64)

    for index in range(args.grid_size[0]*args.grid_size[1]):
        # crop a grid square and get the average color
        swatch = crop_swatch(
            fulpal_im,
            index,
            (args.grid_start[0], args.grid_start[1]),
            (args.grid_size[0], args.grid_size[1]),
            (args.cell_size[0], args.cell_size[1]),
            (args.crop[0], args.crop[1], args.crop[2], args.crop[3]),
            args.debug
        )

        index_x = index % args.grid_size[0]
        index_y = int(index / args.grid_size[0])
        colors[index_x, index_y] = swatch

    # TODO: different sort modes depending on full_palette
    EMPHASIS = 8
    HUE = 16
    VALUE = 4
    fulpal = np.zeros((EMPHASIS, VALUE, HUE, 3), dtype=np.float64)

    black = colors[13, 2]

    for emph in range(EMPHASIS):
        for value in range(VALUE):
            for hue in range(HUE):
                index_x = hue
                index_y = VALUE-(value+1) + emph*(VALUE)
                if hue >= 0xE:
                    fulpal[emph, value, hue] = black
                else:
                    fulpal[emph, value, hue] = colors[index_x, index_y]

    with open((os.path.splitext(args.input)[0] + ".pal"), mode="wb") as palfile:
        palfile.write(np.uint8(np.around(fulpal*0xFF)))

    with open((os.path.splitext(args.input)[0] + ".fpal"), mode="wb") as fpalfile:
        fpalfile.write(fulpal)

if __name__=='__main__':
    main(sys.argv)
