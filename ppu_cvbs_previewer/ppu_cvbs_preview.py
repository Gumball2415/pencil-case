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

VERSION = "0.6.0"

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
        "--plot_filters",
        action="store_true",
        help="plot chroma/luma filters. does not include filter response of comb filters.")
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
        help="input 192-byte .pal file. default = \"2C02.pal\"",
        type=str,
        default="2C02.pal"
        )
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
            "fir",
            "notch",
            "2-line",
            "3-line",
        ],
        default="notch",
        help = "method for separating luma and chroma. default = notch.")
    parser.add_argument(
        "-ftype",
        "--fir_filter_type",
        choices=[
            "lanczos",
            "gauss",
            "box",
            "kaiser",
            "firls",
        ],
        nargs="+",
        default=None,
        help = "1-2 FIR kernels for separating luma and chroma. Decoding filter used will automatically be FIR if this is specified. If one kernel is specified, it will be used for both luma and chroma separation.")
    parser.add_argument(
        "-full",
        "--full_resolution",
        action="store_true",
        help="store the full framebuffer")
    parser.add_argument(
        "-frames",
        type = int,
        help="render x consecutive frames. usable range: 1-3. default = 1",
        default=1)
    parser.add_argument(
        "-avg",
        "--average",
        action="store_true",
        help="use with -frames argument. averages all rendered frames into one. will save output as input_ppucvbs_ph_avg_x.png")
    parser.add_argument(
        "-diff",
        "--difference",
        action="store_true",
        help="use with -frames argument. calculates the absolute difference of the first two frames. will save output as input_ppucvbs_ph_avg_x.png")
    parser.add_argument(
        "-noskipdot",
        action="store_true",
        help="turns off skipped dot rendering. equivalent to rendering on 2C02s")

    return parser.parse_args(argv[1:])

def print_err_quit(message: str):
    print("Error: "+message, file=sys.stderr)
    sys.exit()

# B-Y and R-Y reduction factors
BY_rf = 0.492111
RY_rf = 0.877283

# derived from the NTSC base matrix of luminance and color-difference
RGB_to_YUV = np.array([
    [ 0.299,        0.587,        0.114],
    [-0.299*BY_rf, -0.587*BY_rf,  0.886*BY_rf],
    [ 0.701*RY_rf, -0.587*RY_rf, -0.114*RY_rf]
], np.float64)




### image input/output

# open raw ppu pixels and parse as ndarray
def parse_raw_ppu_px(file: str):
    with open(file, mode="rb") as input:
        array = np.array(np.frombuffer(input.read(), dtype=np.uint16), dtype=int)
        if array.size != 240*256:
            print_err_quit("raw ppu pixel array is not 256x240.")
        return np.reshape(array, (240, 256))

# open image
def parse_indexed_png(file: str, palfile: str):
    with Image.open(file) as im:
        if im.size != (256, 240):
            print_err_quit("image is not 256x240.")

        if im.mode != "P" or palfile is not None:
            # try to convert with input palette
            if palfile is None:
                print_err_quit("image is not indexed and palette is not provided.")

            # convert .pal to an indexed Image object
            with open(palfile, mode="rb") as master_pal:
                palette_buf = np.transpose(np.frombuffer(master_pal.read(), dtype=np.uint8))
                
                # ideally the palette would be 512 entries (1536 bytes) but .png don't support shit like that
                # so in practice the palette is 64 entries (192 bytes)
                if len(palette_buf) != 192:
                    print_err_quit(f"palette size is not 192: got {len(palette_buf)} bytes.")
                
                nespal = ImagePalette.ImagePalette(mode="RGB", palette=list(palette_buf))
                imgpal = Image.new('P',(0,0))
                imgpal.putpalette(nespal)
                # quantize .png to become indexed
                im = im.convert(mode="RGB").quantize(colors=256, palette=imgpal, dither=Image.Dither.NONE)
        else:
            # check if index is correct
            if len(im.getpalette()) != 192:
                print_err_quit(f"image's index is not expected length of 192: got {len(im.getpalette())} bytes.")

        # at this point, image should be indexed .png
        # return image with raw indices of $00-$3F
        return np.array(im, dtype=int)

# save image
def save_image(input: str, out, phases, full_resolution = False, debug = False):
    with Image.fromarray(np.ubyte(np.around(out * 255))) as imageout:
    # scale image
        if (full_resolution or debug):
            imageout = imageout.resize((imageout.size[0], int(imageout.size[1]*8)), Image.Resampling.NEAREST)
        else:
            imageout = imageout.resize((640, imageout.size[1]), resample=Image.Resampling.LANCZOS)
            imageout = imageout.resize((imageout.size[0], int(imageout.size[1]*2)), Image.Resampling.NEAREST)
        imageout.save(f"{os.path.splitext(input)[0]}_ppucvbs_ph_{phases}.png")



### filtering and raster configuration

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

    BEFORE_ACTIVE: int = field(init=False)
    AFTER_ACTIVE: int = field(init=False)

    BEFORE_VID: int = field(init=False)
    AFTER_VID: int = field(init=False)

    BEFORE_CBURST: int = field(init=False)
    AFTER_CBURST: int = field(init=False)

    PULSE_INDEX: int = field(init=False)

    PIXELS_PER_SCANLINE: int = field(init=False)
    SAMPLES_PER_SCANLINE: int = field(init=False)

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

# phase shift the composite using a simple lowpass
# for differential phase distortion
# https://en.wikipedia.org/wiki/Low-pass_filter#Simple_infinite_impulse_response_filter
def RC_lowpass(signal, amount, dt):
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

filter_config = []

def configure_filters(
        ppu_type: str,
        fir_filter_type,
        decoding_filter: str,
        plot_filters: bool = False
    ):
    
    X_main = 26601712.5 if ppu_type == "2C07" else (236.25e6 / 11)
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

    # dummy, will be overwritten
    luma_kernel = chroma_kernel = b_luma = a_luma = 0

    MAX_DB_LOSS = 2
    MIN_DB_GAIN = 65

    # chroma filters
    # SMPTE 170M-2004, page 5, section 7.2
    passband = 2
    stopband = 20
    b_chroma, a_chroma = signal.iirdesign(
        1.3e6,
        3.6e6,
        gpass=passband,
        gstop=stopband,
        analog=False,
        ftype="butter",
        fs=PPU_Fs
    )

    # kernels
    box_kernel = np.full(12, 1, dtype=np.float64)
    box_kernel /= np.sum(box_kernel)

    # chosen to have not more than -12db gain at pixel frequency
    # while at the same time minimizing tap length
    # for px = -6db: 96+1 taps
    kernel_taps = 48+1 # M
    # Equation 16-3
    # https://www.dspguide.com/ch16/2.htm
    bandwidth = 4/kernel_taps*PPU_Fs

    cutoff = PPU_Cb-bandwidth

    # magic number! align first notch to colorburst
    lz_cutoff = PPU_Cb-(bandwidth/3)

    if PPU_Cb <= bandwidth:
        print_err_quit(f"tap size {kernel_taps} too small! bandwidth {bandwidth} larger than cutoff {PPU_Cb}")

    # sinc kernel
    # Equation 16-1
    # https://www.dspguide.com/ch16/1.htm
    # sin(2*pi*fc*i) / i*pi
    # 2*fc*sinc(2*fci)

    lanczos_kernel = 2 * (lz_cutoff/PPU_Fs) * np.sinc(
        2 * (lz_cutoff/PPU_Fs) *
        # shifted to range 0 to M
        (np.arange(kernel_taps) - (kernel_taps/2))
    ) * signal.windows.lanczos(kernel_taps)

    lanczos_kernel /= np.sum(lanczos_kernel)

    gauss_kernel = signal.windows.gaussian(kernel_taps, 12)
    gauss_kernel *= signal.windows.kaiser(kernel_taps, signal.kaiser_beta(MIN_DB_GAIN))
    gauss_kernel /= np.sum(gauss_kernel)

    taps, beta = signal.kaiserord(MIN_DB_GAIN, 5e6/(PPU_Fs))

    kaiser_kernel = signal.firwin(
        taps,
        cutoff=PPU_Cb-(2/taps*PPU_Fs),
        window=("kaiser", beta),
        pass_zero=True,
        fs=PPU_Fs
    )

    firls_kernel = signal.firls(
        kernel_taps,
        [0, cutoff, PPU_Cb, PPU_Fs/2],
        [1, 1, 0, 0],
        fs=PPU_Fs
    )

    kernel_dict = {
        "lanczos": lanczos_kernel,
        "gauss": gauss_kernel,
        "box": box_kernel,
        "kaiser": kaiser_kernel,
        "firls": firls_kernel,
    }

    # bandwidth: from 2px freq to colorburst frequency
    q = PPU_Cb/abs(PPU_Cb-(PPU_Fs/32))
    match decoding_filter:
        case "fir":
            luma_kernel = kernel_dict[fir_filter_type[0]]
            if len(fir_filter_type) == 1:
                chroma_kernel = kernel_dict[fir_filter_type[0]]
            else:
                chroma_kernel = kernel_dict[fir_filter_type[1]]
        case "3-line":
            b_luma, a_luma = signal.iirpeak(PPU_Cb, q, PPU_Fs)
        case _:
            b_luma, a_luma = signal.iirnotch(PPU_Cb, q, PPU_Fs)

    if plot_filters:
        import matplotlib.pyplot as plt
        if decoding_filter == "fir":
            plt.plot(luma_kernel)
            plt.plot(chroma_kernel)
            plt.show()
            w, h = signal.freqz(luma_kernel)
            x = w * PPU_Fs * 1.0 / (2 * np.pi)
            y = 20 * np.log10(abs(h))
            plt.figure(figsize=(10,5))
            plt.semilogx(x, y)
            w, h = signal.freqz(chroma_kernel)
            x = w * PPU_Fs * 1.0 / (2 * np.pi)
            y = 20 * np.log10(abs(h))
            plt.semilogx(x, y)
        else:
            w, h = signal.freqz(b_luma, a_luma, fs=PPU_Fs)
            x = w
            y = 20 * np.log10(abs(h))
            plt.semilogx(x, y)
            w, h = signal.freqz(b_chroma, a_chroma, fs=PPU_Fs)
            x = w
            y = 20 * np.log10(abs(h))
            plt.semilogx(x, y)
        plt.ylabel('Amplitude [dB]')
        plt.xlabel('Frequency [Hz]')
        plt.title('Frequency response')
        plt.grid(which='both', linestyle='-', color='grey')
        plt.xticks([PPU_Fs/32, PPU_Fs/16, cutoff, PPU_Cb, PPU_Fs/2], ["px/2", "px", "cut", "cb", "nyquist"])
        plt.show()

    return (luma_kernel, chroma_kernel, b_luma, a_luma, b_chroma, a_chroma, Fs_dt, r)

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

def yc_kernel(cvbs_ppu, kernel):
    y_line = signal.convolve(cvbs_ppu, kernel, mode="same")
    c_line = cvbs_ppu - y_line
    return y_line, c_line

def yc_notch(cvbs_ppu, b_notch, a_notch):
    y_line = signal.filtfilt(b_notch, a_notch, cvbs_ppu)
    c_line = cvbs_ppu - y_line
    return y_line, c_line, c_line

def yc_pal_delayline(cvbs_ppu, prev_line, r: RasterTimings, b_notch, a_notch, scanline_0: bool):
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
    prev_line = np.append(cvbs_ppu[shift:],cvbs_ppu[:shift])

    # notch filter the luma
    y_line = signal.filtfilt(b_notch, a_notch, cvbs_ppu)
    return y_line, u_line, v_line, prev_line

def yc_comb_2line(cvbs_ppu, prev_line, r: RasterTimings, b_notch, a_notch, scanline_0: bool):
    if scanline_0:
        prev_line = np.zeros(r.SAMPLES_PER_SCANLINE, dtype=np.float64)

    u_line = v_line = np.array(cvbs_ppu - prev_line) / 2

    # rotate chroma line to account for next line's phase shift
    shift = -2
    prev_line = np.append(cvbs_ppu[shift:],cvbs_ppu[:shift])

    # notch filter the luma
    y_line = signal.filtfilt(b_notch, a_notch, cvbs_ppu)
    return y_line, u_line, v_line, prev_line

def yc_comb_3line(cvbs_ppu, prev_line, r: RasterTimings, b_peak, a_peak, pal_comb: bool, scanline_0: bool):
    # Figure 9.45, Figure 9.47
    # Two-Line Delay PAL Y/C Separator Optimized for Standards Conversion and Video Processing
    # Jack, K. (2007). NTSC and PAL digital encoding and decoding. In Video
    # Demystified (5th ed., p. 456). Elsevier.
    # https://archive.org/details/video-demystified-5th-edition/
    if scanline_0:
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
    ppu_type: str,
    scanline: int,
    r: RasterTimings,
    b_luma, a_luma,
    b_chroma, a_chroma,
    decoding_filter: str,
    luma_kernel,
    chroma_kernel,
    prev_line,
    alternate_line = False,
    debug = False
    ):

    # normalize blank/black
    cvbs_ppu -= ppu.composite_black
    YUV_line = np.zeros((r.SAMPLES_PER_SCANLINE, 3), dtype=np.float64)

    # separate luma and chroma
    match decoding_filter:
        case "fir":
            YUV_line[:, 0], u_line = yc_kernel(cvbs_ppu, luma_kernel)
            v_line = u_line
        case "2-line":
            if ppu_type == "2C07":
                YUV_line[:, 0], u_line, v_line, prev_line = yc_pal_delayline(cvbs_ppu, prev_line, r, b_luma, a_luma, scanline == 0)
            else:
                YUV_line[:, 0], u_line, v_line, prev_line = yc_comb_2line(cvbs_ppu, prev_line, r, b_luma, a_luma, scanline == 0)
        case "3-line":
            YUV_line[:, 0], u_line, v_line, prev_line = yc_comb_3line(cvbs_ppu, prev_line, r, b_luma, a_luma, ppu_type == "2C07", scanline == 0)
        case "notch":
            YUV_line[:, 0], u_line, v_line = yc_notch(cvbs_ppu, b_luma, a_luma)
        case _:
            print_err_quit(f"unknown decoding filter option: {decoding_filter}")

    DISABLE_CHROMA = False

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
    if not DISABLE_CHROMA:
        YUV_line[:, 1] = u_line * U_decode
        YUV_line[:, 2] = v_line * V_decode


    # filter chroma
    if debug:
        YUV_line = np.array([cvbs_ppu, YUV_line[:, 1], YUV_line[:, 2]]).T
    elif decoding_filter == "fir":
        YUV_line[:, 1], _ = yc_kernel(YUV_line[:, 1], chroma_kernel)
        YUV_line[:, 2], _ = yc_kernel(YUV_line[:, 2], chroma_kernel)
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
        filter_config,
        decoding_filter,
        skipdot = False,
        full_resolution = False,
        debug = False
    ):

    (luma_kernel, chroma_kernel, b_luma, a_luma, b_chroma, a_chroma, Fs_dt, r) = filter_config

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
            ppu_type,
            scanline,
            r,
            b_luma, a_luma,
            b_chroma, a_chroma,
            decoding_filter,
            luma_kernel,
            chroma_kernel,
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

def main(argv=None):
    args = parse_argv(argv or sys.argv)
    # load input image and parse it for indices
    raw_ppu = np.zeros((240, 256), dtype=np.int16)
    if args.raw_ppu_px: raw_ppu = parse_raw_ppu_px(args.input)
    else: raw_ppu = parse_indexed_png(args.input, args.palette)

    phase = args.color_clock_phase

    avg = args.average and args.frames > 1

    frames = []

    # if FIR filter type is mentioned, it implies FIR decoding filter
    if args.fir_filter_type is not None:
        args.decoding_filter = "fir"

    filter_config = configure_filters(
        args.ppu,
        args.fir_filter_type,
        args.decoding_filter,
        args.plot_filters
    )

    for _ in range(args.frames):
        out, nextphase = encode_frame(raw_ppu,
            args.ppu,
            phase,
            args.phase_distortion,
            filter_config,
            args.decoding_filter,
            (not args.noskipdot),
            args.full_resolution,
            args.debug
        )

        frames.append((out, phase))

        if not (avg or args.difference):
            print(nextphase)
            save_image(args.input, out, phase, args.full_resolution, args.debug)
        phase = nextphase

    if avg or args.difference:
        images, phases = zip(*frames)
        images = np.array(images)

        if avg:
            out = np.average(images, axis=0)
        elif args.difference:
            out = images[0] - images[1]
            out = np.absolute(out)
        save_image(args.input, out, phases, args.full_resolution, args.debug)

if __name__=='__main__':
    main(sys.argv)
