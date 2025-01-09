# Z1 dither mask generator
# based on Constrained Image Binarization for Jacquard Fabric Pattern Generation
# https://www.jstage.jst.go.jp/article/artsci/13/3/13_124/_pdf/-char/ja
# Copyright 2024 Persune
# MIT-0

import numpy as np
import sys, argparse, os
from PIL import Image

VERSION = "0.3.0"

def main(argv=None):
    def parse_argv(argv):
        parser=argparse.ArgumentParser(
            description="Z1 dither mask generator",
            epilog="version " + VERSION)
        parser.add_argument(
            "n",
            help="value of n, which is the unit size of the tile. default is 8",
            type=int,
            default=8)
        parser.add_argument(
            "m",
            help="value of m, which is the line offset. default is 5",
            type=int,
            default=5)
        parser.add_argument(
            "-saw",
            help="generates a sawtooth sequence instead of triangle. default is False",
            action="store_true")
        parser.add_argument(
            "output",
            help="output .png file")
        return parser.parse_args(argv[1:])
    args = parse_argv(argv or sys.argv)

    # 4.1 ジャカード織物用ディザマスクの生成
    n = args.n
    m = args.m

    z1_tile = np.zeros((n,n), np.uint8)

    # we need to select n(n-2) values to fill the rest of the slots
    # we have wiggle room of 1 to 254
    # so, distribute these values among the free slots
    v = np.arange(1,((n*(n-2))+1)).reshape(((n-2),n)).T
    v = np.around((254/(n*(n-2)))*v)

    for i in range(n):
        for j in range(n):
            # eq 7
            if args.saw:
                t = (m*i+j)%n
            else:
                threshold = np.floor((n-1)/2)
                k = (m*i+j)%n
                u = k - threshold
                if u <= 0: t = -2*u
                if 0 < u <= threshold: t = 2*u - 1
                if u > threshold: t = n-1

            if t == n-1:
                z1_tile[i,j] = 255
            elif t == 0:
                z1_tile[i,j] = 0
            else:
                z1_tile[i,j] = v[i,int(t-1)]

    pil_image = Image.fromarray(z1_tile)
    pil_image.save(args.output, format="PNG", optimize=True)

if __name__ == '__main__':
    main(sys.argv)
