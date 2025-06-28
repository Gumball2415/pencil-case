#!/usr/bin/env python
# NES PPU composite video encoder
# Copyright (C) 2025 Persune

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

import argparse
import sys
import numpy as np

VERSION = "0.1.4"

# signal LUTs
# voltage highs and lows
# from https://forums.nesdev.org/viewtopic.php?p=159266#p159266
# signal[5][2][2] $0x-$3x, $x0/$xD, no emphasis/emphasis
# 5th index is purely colorburst
signal_table_composite = np.array([
    [
        [ 0.616, 0.500 ],
        [ 0.228, 0.192 ]
    ],
    [
        [ 0.840, 0.676 ],
        [ 0.312, 0.256 ]
    ],
    [
        [ 1.100, 0.896 ],
        [ 0.552, 0.448 ]
    ],
    [
        [ 1.100, 0.896 ],
        [ 0.880, 0.712 ]
    ],
    # colorburst high, colorburst low
    [
        [ 0.524, 0.524 ],
        [ 0.148, 0.148 ]
    ],
    # sync level, blank level
    [
        [ 0.048, 0.048 ],
        [ 0.312, 0.312 ]
    ]
], np.float64)

# first 512 entries are exclusively for the 9-bit PPU pixel format: "eeellcccc".
# these additional entries are bitshifted to avoid collisions
SYNC_LEVEL = 1 << 9
BLANK_LEVEL = 2 << 9
COLORBURST = 3 << 9

composite_white = signal_table_composite[3, 0, 0]
composite_black = signal_table_composite[1, 1, 0]

def parse_argv(argv):
    parser=argparse.ArgumentParser(
        description="NES PPU composite video generator",
        epilog="version " + VERSION)
    # output options
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="debug messages")

    return parser.parse_args(argv[1:])

ALTERNATE_PHASE = {
    1: 4,
    2: 3,
    3: 2,
    4: 1,
    5: 12,
    6: 11,
    7: 10,
    8: 9,
    9: 8,
    10: 7,
    11: 6,
    12: 5,
}

# 2C07 phase alternation
def pal_phase(hue: int, alternate_line=False):
    if (hue >= 1 and hue <= 12) and alternate_line:
        return ALTERNATE_PHASE[hue]
    else:
        return hue

# waveform generation
def in_color_phase(hue: int, phase: int, alternate_line=False):
    return ((pal_phase(hue, alternate_line) + phase) % 12) < 6

# encodes a composite sample from a given PPU pixel and a given phase
def encode_composite_sample(
    pixel: int,
    wave_phase: int,
    sinusoidal_peak_generation: bool,
    cburst_phase: int,
    alternate_line=False):

    if pixel >= SYNC_LEVEL:
        if pixel == SYNC_LEVEL:
            return signal_table_composite[5, 0, 0]
        elif pixel == BLANK_LEVEL:
            return signal_table_composite[5, 1, 0]
        elif pixel == COLORBURST:
            return signal_table_composite[4, int(in_color_phase(cburst_phase, wave_phase, alternate_line)), 0]
        else:
            sys.exit(f"invalid PPU pixel. got {pixel:04X}")

    emphasis = (pixel & 0b111000000) >> 6
    luma =     (pixel & 0b000110000) >> 4
    hue =       pixel & 0b000001111
    # 1 = emphasis activate
    if emphasis != 0:
        attenuate = int(
            (((emphasis & 1) and in_color_phase(0xC, wave_phase, alternate_line)) or
            ((emphasis & 2) and in_color_phase(0x4, wave_phase, alternate_line)) or
            ((emphasis & 4) and in_color_phase(0x8, wave_phase, alternate_line))) and
            (hue < 0xE)
        )
    else: attenuate = 0

    #columns $xE-$xF
    if (hue >= 0xE):
        luma = 0x1

    # generate sinusoidal waveforms with matching p-p amplitudes
    if (sinusoidal_peak_generation):
        # rows $x0 and $xD
        if (hue == 0x00):
            return signal_table_composite[luma, 0, attenuate]
        
        if (hue >= 0x0D):
            return signal_table_composite[luma, 1, attenuate]

        wave_amp = (signal_table_composite[luma, 0, attenuate] - signal_table_composite[luma, 1, attenuate]) / 2
        wave_dc = (signal_table_composite[luma, 0, attenuate] + signal_table_composite[luma, 1, attenuate]) / 2

        return wave_dc + (np.sin((2 * np.pi * (hue+0.5)/12) + (2 * np.pi / 12 * (wave_phase))) * wave_amp)

    # 0 = waveform high; 1 = waveform low
    n_wave_level = int(not in_color_phase(hue, wave_phase, alternate_line))

    # rows $x0 and $xD
    if (hue == 0x0): n_wave_level = 0
    elif (hue >= 0xD): n_wave_level = 1

    return signal_table_composite[luma, n_wave_level, attenuate]

def main(argv=None):
    return

if __name__=='__main__':
    main(sys.argv)