#!/usr/bin/env python
# NES PPU composite video filter for input images
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

import ppu_composite as ppu
import argparse
import sys, os
import numpy as np
from scipy import signal
from PIL import Image, ImagePalette
from dataclasses import dataclass, field

VERSION = "0.2.3"

def parse_argv(argv):
    parser=argparse.ArgumentParser(
        description="NES PPU composite video shader previewer",
        epilog="version " + VERSION)
    # output options
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="debug messages")
    parser.add_argument(
        "-cxp",
        "--color_clock_phase",
        type=int,
        default=0,
        help="starting clock phase of first frame. will affect all other phases. range: 0-11")
    parser.add_argument(
        "input",
        help="input 256x240 indexed .png screenshot. will save output as input_ppucvbs_ph_x.png",
        type=str)
    parser.add_argument(
        "-raw",
        "--raw_ppu_px",
        action="store_true",
        help="input image is instead a 256x240 uint16_t array of raw 9-bit PPU pixels. format: EEELLCCCC")
    parser.add_argument(
        "-pal",
        "--palette",
        help="input 192-byte .pal file",
        type=str)
    parser.add_argument(
        "-ppu",
        "--ppu",
        type=str,
        help="PPU chip used for generating colors. default = 2C02",
        choices=[
            "2C02",
            "2C07",
        ],
        default = "2C02")
    parser.add_argument(
        "-phd",
        "--phase_distortion",
        type = np.float64,
        help = "amount of voltage-dependent impedance for RC lowpass, where RC = \"amount * (level/composite_white) * 1e-8\". this will also desaturate and hue shift the resulting colors nonlinearly. a value of 4 very roughly corresponds to a -5 degree delta per luma row. default = 4",
        default = 4)
    parser.add_argument(
        "-filt",
        "--decoding_filter",
        choices=[
            "notch",
            "box",
            "2-line",
            "3-line",
        ],
        default="notch",
        help = "choose decoding method to filter chroma and luma. default = notch.")
    parser.add_argument(
        "-full",
        "--full_resolution",
        action="store_true",
        help="store the full framebuffer")
    parser.add_argument(
        "-frames",
        type = int,
        help="render x consecutive frames. range: 1-3. default = 1",
        default=1)
    parser.add_argument(
        "-avg",
        "--average",
        action="store_true",
        help="use with -frames argument. averages all rendered frames into one. will save output as input_ppucvbs_ph_avg_x.png")
    parser.add_argument(
        "-noskipdot",
        action="store_true",
        help="turns off skipped dot rendering. equivalent to rendering on 2C02s")

    return parser.parse_args(argv[1:])

# B-Y and R-Y reduction factors
BY_rf = 0.492111
RY_rf = 0.877283

# derived from the NTSC base matrix of luminance and color-difference
RGB_to_YUV = np.array([
    [ 0.299,        0.587,        0.114],
    [-0.299*BY_rf, -0.587*BY_rf,  0.886*BY_rf],
    [ 0.701*RY_rf, -0.587*RY_rf, -0.114*RY_rf]
], np.float64)


# get starting phase of previous frame and increment it
# def starting_phase(phase: int, cycles_per_frame: int, cycles_per_px: int, rendering: bool = True):
#     dotskip = cycles_per_px if rendering else 0
#     phase = (phase + cycles_per_frame + dotskip)
#     return phase

# open raw ppu pixels and parse as ndarray
def parse_raw_ppu_px(file: str):
    with open(file, mode="rb") as input:
        array = np.array(np.frombuffer(input.read(), dtype=np.uint16), dtype=int)
        if array.size != 240*256:
            sys.exit("raw ppu pixel array is not 256x240.")
        return np.reshape(array, (240, 256))

# open image
def parse_indexed_png(file: str, palfile: str):
    with Image.open(file) as im:
        if im.size != (256, 240):
            sys.exit("image is not 256x240.")

        if im.mode != "P" or palfile is not None:
            # try to convert with input palette
            if palfile is None:
                sys.exit("image is not indexed and palette is not provided.")

            # convert .pal to an indexed Image object
            with open(palfile, mode="rb") as master_pal:
                palette_buf = np.transpose(np.frombuffer(master_pal.read(), dtype=np.uint8))
                
                # ideally the palette would be 512 entries (1536 bytes) but .png don't support shit like that
                # so in practice the palette is 64 entries (192 bytes)
                if len(palette_buf) != 192:
                    sys.exit(f"palette size is not 192: got {len(palette_buf)} bytes.")
                
                nespal = ImagePalette.ImagePalette(mode="RGB", palette=list(palette_buf))
                imgpal = Image.new('P',(0,0))
                imgpal.putpalette(nespal)
                # quantize .png to become indexed
                im = im.convert(mode="RGB").quantize(colors=256, palette=imgpal, dither=Image.Dither.NONE)
        else:
            # check if index is correct
            if len(im.getpalette()) != 192:
                sys.exit(f"image's index is not expected length of 192: got {len(im.getpalette())} bytes.")

        # at this point, image should be indexed .png
        # return image with raw indices of $00-$3F
        return np.array(im, dtype=int)

# phase shift the composite using a simple lowpass
# for differential phase distortion
# https://en.wikipedia.org/wiki/Low-pass_filter#Simple_infinite_impulse_response_filter
def RC_lowpass(signal, amount, dt):
    # dt = 1/PPU_Fs
    v_out = np.zeros(signal.shape, np.float64)
    v_prev = signal[0]
    for i in range(len(signal)):
        # impedance changes depending on DAC tap
        # we approximate this by using the raw signal's voltage
        # https://forums.nesdev.org/viewtopic.php?p=287241#p287241
        # the phase shifts negative on higher levels according to https://forums.nesdev.org/viewtopic.php?p=186297#p186297
        v_prev_norm = signal[i] / ppu.composite_white

        # RC constant tuned so that 0x and 3x colors have around 14 degree delta when phd = 3
        # https://forums.nesdev.org/viewtopic.php?p=186297#p186297
        alpha = dt / (v_prev_norm * amount * 1e-8 + dt)
        v_prev = alpha * signal[i] + (1-alpha) * v_prev
        v_out[i] = v_prev
    return v_out

def QAM_phase(signal):
    buf_size = len(signal)
    t = np.arange(12)
    U = np.tile(np.sin(2 * np.pi / 12 * t)*2, buf_size//12)
    V = np.tile(np.cos(2 * np.pi / 12 * t)*2, buf_size//12)
    signal_U = np.average(signal * U)
    signal_V = np.average(signal * V)
    return np.atan2(signal_U, signal_V)

@dataclass
class RasterTimings:
    """scanline timing constants"""
    HSYNC: int
    B_PORCH_A: int
    CBURST: int
    B_PORCH_B: int
    PULSE: int
    L_BORDER: int
    ACTIVE: int
    R_BORDER: int
    F_PORCH: int
    CBURST_PHASE: int

    # todo: change to use more complete ppu buffer
    BACKDROP: int

    PIXEL_SIZE: int

    SCANLINES: int
    NEXT_SHIFT: int = field(init=False)

    BEFORE_ACTIVE: int = field(init=False)# = HSYNC + B_PORCH_A + CBURST + B_PORCH_B + PULSE + L_BORDER
    AFTER_ACTIVE: int = field(init=False)# = BEFORE_ACTIVE + ACTIVE

    BEFORE_VID: int = field(init=False)# = HSYNC + B_PORCH_A + CBURST + B_PORCH_B
    AFTER_VID: int = field(init=False)# = BEFORE_VID + PULSE + L_BORDER + ACTIVE + R_BORDER

    BEFORE_CBURST: int = field(init=False)# = HSYNC + B_PORCH_A
    AFTER_CBURST: int = field(init=False)# = BEFORE_CBURST + CBURST

    PULSE_INDEX: int = field(init=False)# = HSYNC + B_PORCH_A + CBURST + B_PORCH_B

    PIXELS_PER_SCANLINE: int = field(init=False)# = HSYNC + B_PORCH_A + CBURST + B_PORCH_B + PULSE + L_BORDER + ACTIVE + R_BORDER + F_PORCH
    SAMPLES_PER_SCANLINE: int = field(init=False)# = PIXELS_PER_SCANLINE * PIXEL_SIZE

    def __post_init__(self):
        self.BEFORE_ACTIVE = self.HSYNC + self.B_PORCH_A + self.CBURST + self.B_PORCH_B + self.PULSE + self.L_BORDER
        self.AFTER_ACTIVE = self.BEFORE_ACTIVE + self.ACTIVE
        self.BEFORE_VID = self.HSYNC + self.B_PORCH_A + self.CBURST + self.B_PORCH_B
        self.AFTER_VID = self.BEFORE_VID + self.PULSE + self.L_BORDER + self.ACTIVE + self.R_BORDER

        self.BEFORE_CBURST = self.HSYNC + self.B_PORCH_A
        self.AFTER_CBURST = self.BEFORE_CBURST + self.CBURST

        self.PULSE_INDEX = self.HSYNC + self.B_PORCH_A + self.CBURST + self.B_PORCH_B

        self.PIXELS_PER_SCANLINE = self.HSYNC + self.B_PORCH_A + self.CBURST + self.B_PORCH_B + self.PULSE + self.L_BORDER + self.ACTIVE + self.R_BORDER + self.F_PORCH
        self.SAMPLES_PER_SCANLINE = self.PIXELS_PER_SCANLINE * self.PIXEL_SIZE
        # scanline shift to account for phase
        self.NEXT_SHIFT = (self.SAMPLES_PER_SCANLINE) % 12

# filters one line of raw ppu pixels
def encode_scanline(data,
    starting_phase: int,
    phd: int,
    scanline: int,
    Fs_dt: float,
    r: RasterTimings,
    alternate_line = False
    ):

    raw_ppu = np.full(r.PIXELS_PER_SCANLINE, ppu.BLANK_LEVEL, dtype=np.int16)
    cvbs_ppu = np.zeros(r.SAMPLES_PER_SCANLINE, dtype=np.float64)
    
    raw_ppu[0:r.HSYNC] = ppu.SYNC_LEVEL
    raw_ppu[r.HSYNC:] = ppu.BLANK_LEVEL
    raw_ppu[r.BEFORE_CBURST:r.AFTER_CBURST] = ppu.COLORBURST
    raw_ppu[r.BEFORE_ACTIVE:r.AFTER_ACTIVE] = data
    if r.PULSE != 0:
        raw_ppu[r.PULSE_INDEX] &= 0xF0

    scanline_phase = (starting_phase + scanline*r.SAMPLES_PER_SCANLINE) % 12

    # encode one scanline, then lowpass it accordingly
    for pixel in range(raw_ppu.shape[0]):
        for sample in range(r.PIXEL_SIZE):
            # todo: concurrency
            phase = (scanline_phase + pixel*r.PIXEL_SIZE + sample) % 12
            cvbs_ppu[pixel*r.PIXEL_SIZE + sample] = ppu.encode_composite_sample(raw_ppu[pixel], phase, False, r.CBURST_PHASE, alternate_line)
    # lowpass
    if phd != 0:
        cvbs_ppu = RC_lowpass(cvbs_ppu, phd, Fs_dt)

    return cvbs_ppu

# TODO: custom kernels
box_kernel = np.full(12, 1, dtype=np.float64)
box_kernel /= np.sum(box_kernel)

def yc_box(cvbs_ppu):
    y_line = signal.convolve(cvbs_ppu, box_kernel, mode="same")
    c_line = cvbs_ppu - y_line
    return y_line, c_line, c_line

def yc_notch(cvbs_ppu, b_notch, a_notch):
    y_line = signal.filtfilt(b_notch, a_notch, cvbs_ppu)
    c_line = cvbs_ppu - y_line
    return y_line, c_line, c_line

def yc_comb_2line(cvbs_ppu, prev_line, r: RasterTimings, b_notch, a_notch, pal_comb: bool, scanline_0: bool):
    if pal_comb:
        # bandpass and combine lines in specific way to retrieve U and V
        # based on Single Delay Line PAL Y/C Separator
        # Jack, K. (2007). NTSC and PAL digital encoding and decoding. In Video
        # Demystified (5th ed., p. 450). Elsevier.
        # https://archive.org/details/video-demystified-5th-edition/
        if scanline_0:
            prev_line = np.zeros(r.SAMPLES_PER_SCANLINE, dtype=np.float64)

        u_line = (cvbs_ppu + prev_line) / 2
        v_line = (cvbs_ppu - prev_line) / 2

        # rotate chroma line to account for next line's phase shift
        shift = r.NEXT_SHIFT
    else:
        if scanline_0:
            prev_line = np.zeros(r.SAMPLES_PER_SCANLINE, dtype=np.float64)
        u_line = v_line = np.array(cvbs_ppu - prev_line) / 2
        # rotate chroma line to account for next line's phase shift
        shift = -2
    prev_line = np.append(cvbs_ppu[shift:],cvbs_ppu[:shift])
    # notch filter the luma
    y_line = signal.filtfilt(b_notch, a_notch, cvbs_ppu)
    return y_line, u_line, v_line, prev_line

def yc_comb_3line(cvbs_ppu, prev_line, r: RasterTimings, b_peak, a_peak, pal_comb: bool, scanline: int):
    # Figure 9.45, Figure 9.47
    # Two-Line Delay PAL Y/C Separator Optimized for Standards Conversion and Video Processing
    # Jack, K. (2007). NTSC and PAL digital encoding and decoding. In Video
    # Demystified (5th ed., p. 456). Elsevier.
    # https://archive.org/details/video-demystified-5th-edition/
    if scanline == 0:
        prev_line = np.zeros((r.SAMPLES_PER_SCANLINE, 2), dtype=np.float64)

    c_line = (cvbs_ppu + prev_line[:, 1]) / 2
    c_line = prev_line[:, 0] - c_line
    if not pal_comb: c_line /= 2

    c_line = signal.filtfilt(b_peak, a_peak, c_line)

    y_line = prev_line[:, 0] - c_line

    c_line = prev_line[:, 0]

    # rotate chroma line to account for next line's phase shift
    if pal_comb:
        shift = 1
    else:
        shift = -2
    prev_line[:, 1] = np.append(prev_line[shift:, 0],prev_line[:shift, 0])
    prev_line[:, 0] = np.append(cvbs_ppu[shift:],cvbs_ppu[:shift])
    
    return y_line, c_line, c_line, prev_line

def decode_scanline(
    cvbs_ppu,
    starting_phase: int,
    ppu_type: str,
    scanline: int,
    r: RasterTimings,
    PPU_Fs: float,
    PPU_Cb: float,
    decoding_filter: str,
    prev_line,
    alternate_line = False,
    debug = False
    ):
    # chroma filters
    Q_FACTOR = 1
    chroma_N, chroma_Wn = signal.buttord(1300000, 3600000, 2, 20, fs=PPU_Fs, analog=False)
    b_chroma, a_chroma = signal.butter(chroma_N, chroma_Wn, 'low', fs=PPU_Fs, analog=False)
    b_notch, a_notch = signal.iirnotch(PPU_Cb, Q_FACTOR, PPU_Fs)
    b_peak, a_peak = signal.iirpeak(PPU_Cb, Q_FACTOR, PPU_Fs)

    # normalize blank/black
    cvbs_ppu -= ppu.composite_black
    YUV_line = np.zeros((r.SAMPLES_PER_SCANLINE, 3), dtype=np.float64)

    # separate luma and chroma
    match decoding_filter:
        case "box":
            YUV_line[:, 0], u_line, v_line = yc_box(cvbs_ppu)
        case "2-line":
            YUV_line[:, 0], u_line, v_line, prev_line = yc_comb_2line(cvbs_ppu, prev_line, r, b_notch, a_notch, ppu_type == "2C07", scanline == 0)
        case "3-line":
            YUV_line[:, 0], u_line, v_line, prev_line = yc_comb_3line(cvbs_ppu, prev_line, r, b_peak, a_peak, ppu_type == "2C07", scanline)
        case "notch":
            YUV_line[:, 0], u_line, v_line = yc_notch(cvbs_ppu, b_notch, a_notch)
        case _:
            sys.exit(f"unknown decoding filter option: {decoding_filter}")


    # get UV decoding phases
    # correct for weird 90 degree offset
    cburst_phase = QAM_phase(u_line[(r.BEFORE_CBURST+3)*r.PIXEL_SIZE:(r.AFTER_CBURST-3)*r.PIXEL_SIZE]) - np.pi/2

    # generate UV decoding sines
    cburst_shift = (-r.BEFORE_CBURST+3 + r.NEXT_SHIFT) % 12

    t = np.arange(12, dtype=np.float64) + cburst_shift

    # adjust with colorburst = 135 degrees
    # with this setting, hues match this image: https://forums.nesdev.org/viewtopic.php?p=133638#p133638
    if ppu_type == "2C07":
        t -= 1
        # 2-line comb filter adjusts the phase for us for some reason
        if decoding_filter != "2-line":
            # if not comb filtering, U would align to 180 degrees instead of 135
            cburst_phase += np.pi/4 * (1 if alternate_line else -1)

    U_decode = np.sin((2 * np.pi / 12 * t) - cburst_phase) * 2
    V_decode = np.cos((2 * np.pi / 12 * t) - cburst_phase) * 2

    # switch V
    if ppu_type == "2C07" and alternate_line: V_decode *= -1

    tilecount = int(np.ceil(r.SAMPLES_PER_SCANLINE/12))
    U_decode = np.tile(U_decode, tilecount)[:r.SAMPLES_PER_SCANLINE]
    V_decode = np.tile(V_decode, tilecount)[:r.SAMPLES_PER_SCANLINE]
    # qam
    YUV_line[:, 1] = u_line * U_decode
    YUV_line[:, 2] = v_line * V_decode


    # filter chroma
    if debug:
        YUV_line = np.array([cvbs_ppu, YUV_line[:, 1], YUV_line[:, 2]]).T
    else:
        if decoding_filter == "box":
            YUV_line[:, 1] = signal.convolve(YUV_line[:, 1], box_kernel, mode="same")
            YUV_line[:, 2] = signal.convolve(YUV_line[:, 2], box_kernel, mode="same")
        else:
            YUV_line[:, 1] = signal.filtfilt(b_chroma, a_chroma, YUV_line[:, 1])
            YUV_line[:, 2] = signal.filtfilt(b_chroma, a_chroma, YUV_line[:, 2])
    return YUV_line, prev_line

# keeps track of odd-even frame render parity
is_odd_frame = True

# encodes a (240,256) array of uint16 raw PPU pixels into an RGB output image
# also outputs the next starting phase for the next frame
def encode_frame(raw_ppu,
    ppu_type: str,
    starting_phase: int,
    phd: int,
    decoding_filter: str,
    skipdot = False,
    full_resolution = False,
    debug = False
):
    X_main = 26601712.5 if ppu_type == "2C07" else (236250000 / 11)
    # samplerate and colorburst freqs
    # used for phase shift
    PPU_Fs = X_main * 2
    PPU_Cb = X_main / 6
    Fs_dt = 1/PPU_Fs

    if ppu_type == "2C07":
        r = RasterTimings(
            HSYNC = 25,
            B_PORCH_A = 4,
            CBURST = 15,
            B_PORCH_B = 5,
            PULSE = 0,
            L_BORDER = 18,
            ACTIVE = 256,
            R_BORDER = 9,
            F_PORCH = 9,
            CBURST_PHASE = (7-6)%12,
            PIXEL_SIZE = 8,
            BACKDROP = 0x0F,
            SCANLINES = 312
        )
    else:
        r = RasterTimings(
            HSYNC = 25,
            B_PORCH_A = 4,
            CBURST = 15,
            B_PORCH_B = 5,
            PULSE = 1,
            L_BORDER = 15,
            ACTIVE = 256,
            R_BORDER = 11,
            F_PORCH = 9,
            CBURST_PHASE = (8-6)%12,
            PIXEL_SIZE = 8,
            BACKDROP = 0x0F,
            SCANLINES = 262
        )

    out = np.zeros((raw_ppu.shape[0], r.SAMPLES_PER_SCANLINE, 3), np.float64)

    # dummy; will be overwritten with an array
    prev_line = 0

    for scanline in range(raw_ppu.shape[0]):
        # todo: concurrency for non-comb filter decoding
        alternate_line = ppu_type == "2C07" and (scanline % 2 == 0)
        # encode scanline
        cvbs_ppu = encode_scanline(raw_ppu[scanline], starting_phase, phd, scanline, Fs_dt, r, alternate_line)

        # deccode scanline
        out[scanline], prev_line = decode_scanline(
            cvbs_ppu,
            starting_phase,
            ppu_type,
            scanline,
            r,
            PPU_Fs,
            PPU_Cb,
            decoding_filter,
            prev_line,
            alternate_line,
            debug
        )

    # crop image
    if not (full_resolution or debug):
        out = out[:, r.BEFORE_VID*r.PIXEL_SIZE:r.AFTER_VID*r.PIXEL_SIZE, :]

    # rescale to 0-100 IRE
    out *= 140
    out /= 100
    # convert to RGB
    if not debug:
        out = np.einsum('ij,klj->kli', np.linalg.inv(RGB_to_YUV), out, dtype=np.float64)
    # fit RGB within range of 0.0-1.0
    np.clip(out, 0, 1, out=out)

    global is_odd_frame
    is_odd_frame ^= is_odd_frame

    if ppu_type == "2C07": skipdot = False
    dot = r.PIXEL_SIZE if skipdot and is_odd_frame else 0
    next_phase = (starting_phase + r.SAMPLES_PER_SCANLINE*r.SCANLINES - dot) % 12

    return out, next_phase

def save_image(input: str, out, phases, full_resolution = False, debug = False):
    with Image.fromarray(np.ubyte(np.around(out * 255))) as imageout:
    # scale image
        if (full_resolution or debug):
            imageout = imageout.resize((imageout.size[0], int(imageout.size[1]*8)), Image.Resampling.NEAREST)
        else:
            imageout = imageout.resize((640, imageout.size[1]), resample=Image.Resampling.LANCZOS)
            imageout = imageout.resize((imageout.size[0], int(imageout.size[1]*2)), Image.Resampling.NEAREST)
        imageout.save(f"{os.path.splitext(input)[0]}_ppucvbs_ph_{phases}.png")

def main(argv=None):
    args = parse_argv(argv or sys.argv)
    # load input image and parse it for indices
    raw_ppu = np.zeros((240, 256), dtype=np.int16)
    if args.raw_ppu_px: raw_ppu = parse_raw_ppu_px(args.input)
    else: raw_ppu = parse_indexed_png(args.input, args.palette)

    phase = args.color_clock_phase

    avg = args.average and args.frames > 1

    frames = []

    for frame in range(args.frames):
        out, nextphase = encode_frame(raw_ppu,
            args.ppu,
            phase,
            args.phase_distortion,
            args.decoding_filter,
            (not args.noskipdot),
            args.full_resolution,
            args.debug
        )
        frames.append((out, phase))

        if not avg:
            save_image(args.input, out, phase, args.full_resolution, args.debug)
        phase = nextphase

    if avg:
        images, phases = zip(*frames)

        images = np.array(images)
        out = np.average(images, axis=0)
        save_image(args.input, out, phases, args.full_resolution, args.debug)


    return

if __name__=='__main__':
    main(sys.argv)