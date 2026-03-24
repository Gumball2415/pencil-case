#!/usr/bin/env python
# NES PPU composite video filter for input images
# Copyright (C) 2026 Persune

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

import sys, os
import argparse
import numpy as np

VERSION = "0.16.0"

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
        help="Plot chroma / luma filters. Not including the filter response\
            of comb filters.")
    parser.add_argument(
        "-cxp",
        "--color_clock_phase",
        type=int,
        default=0,
        help="Starting clock phase of first frame. This will affect all\
        proceeding phases. default = 0, range: 0-11")
    parser.add_argument(
        "input",
        help="Input 256x240 .png screenshot, index matched by input palette.\
            Output will be saved as \"input_ppucvbs_ph_x.png\".",
        type=str)
    parser.add_argument(
        "-raw",
        "--raw_ppu_px",
        action="store_true",
        help="Indicates input is a 256x240 uint16_t array of raw 9-bit PPU\
            pixels. Bit format: xxxxxxxEEELLCCCC")
    parser.add_argument(
        "-b",
        "--backdrop",
        type=str,
        default="0F",
        help="Backdrop color index in hexadecimal (00-3F), used for NTSC \
            active video border. Default= \"0F\"")
    parser.add_argument(
        "-pal",
        "--palette",
        help="Input 192-byte .pal file to index the input .png.\
            Default = \"2C02.pal\"",
        type=str,
        default="2C02.pal"
        )
    parser.add_argument(
        "-ppu",
        "--ppu",
        type=str,
        help="Composite PPU chip used for generating colors. Default =\
            \"2C02\"",
        choices=[
            "2C02",
            "2C07",
        ],
        default = "2C02")
    parser.add_argument(
        "-phd",
        "--phase_distortion",
        type = np.float64,
        help = "The amount of voltage-dependent impedance for RC lowpass,\
            where RC = \"phd * (level/composite_white) * 1e-8\". See\
            https://www.nesdev.org/wiki/NTSC_video#Simulating_differential_phase_distortion\
            for more details.\
            Default = 3",
        default = 3)
    parser.add_argument(
        "-x",
        "--disable",
        choices=[
            "chroma",
            "luma",
        ],
        help = "Disables chroma by setting UV to 0. Disables luma by setting Y\
            to 0.5.")
    parser.add_argument(
        "-filt",
        "--decoding_filter",
        choices=[
            "compl",
            "1-line",
            "2-line",
        ],
        default="compl",
        help = "Method for luma and chroma decoding.\
                Default = \"compl\".")
    parser.add_argument(
        "-ftype",
        "--filter_type",
        choices=[
            "iir",
            "lanczos",
            "lanczos_notch",
            "gauss",
            "box",
            "kaiser",
            "leastsquares",
            "none",
        ],
        nargs="+",
        default=["iir"],
        help = "One filter for complementary luma-chroma filtering and\
            another filter for lowpassing quadrature demodulated chroma.\
            If one option is specified, it will be used for both filters.\
            Only up to two filters are recognized.\
            Default =\"iir\".")
    parser.add_argument(
        "-full",
        "--full_resolution",
        action="store_true",
        help="Saves full scanlines instead of a cropped output.\
            Scanlines are scaled to 8x to preserve 1:1 pixel aspect ratio.")
    parser.add_argument(
        "-frames",
        type = int,
        help="Render x consecutive frames. Usable range: 1-3. default = 1",
        default=1)
    parser.add_argument(
        "-avg",
        "--average",
        action="store_true",
        help="To be used with \"-frames\" argument. Averages all rendered\
            frames into one.\
            Output will be saved as input_ppucvbs_ph_avg_x.png")
    parser.add_argument(
        "-diff",
        "--difference",
        action="store_true",
        help="To be used with \"-frames\" argument. Calculates the absolute\
            difference of the first two frames.\
            Output will be saved as input_ppucvbs_ph_avg_x.png")
    parser.add_argument(
        "-noskipdot",
        action="store_true",
        help="Turns off skipped dot rendering, generating three chroma dot\
            phases. Equivalent to rendering on 2C02s")
    return parser.parse_args(argv[1:])

def print_err_quit(message: str):
    """
    Prints an error message and quits.
    """
    sys.exit("Error: "+message)

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
from PIL import Image, ImagePalette

def parse_raw_ppu_px(file: str):
    """
    Open raw PPU pixels and parse as Numpy array for processing.

    :param file: File path to raw uint16 PPU pixel binary
    :type file: str



    :return: 256x240 Numpy uint16 array
    :rtype: np.ndarray
    """
    with open(file, mode="rb") as input:
        array = np.array(
            np.frombuffer(input.read(), dtype=np.uint16), dtype=int)
        if array.size != 240*256:
            print_err_quit("raw ppu pixel array is not 256x240.")
        return np.reshape(array, (240, 256))

def parse_indexed_png(file: str, palfile: str):
    """
    Open .png, quantize to palette, and parse as Numpy array for processing.

    _extended_summary_

    :param file: File path to .png. Must be 256x240.
    :type file: str

    :param palfile: File path to palette to quantize with.
        Must be 192 bytes, or 64 RGB24 entries.
    :type palfile: str



    :return: 256x240 Numpy uint16 array
    :rtype: np.ndarray
    """
    with Image.open(file) as im:
        if im.size != (256, 240):
            print_err_quit("image is not 256x240.")

        if im.mode != "P" or palfile is not None:
            # try to convert with input palette
            if palfile is None:
                print_err_quit(
                    "image is not indexed and palette is not provided.")

            # convert .pal to an indexed Image object
            with open(palfile, mode="rb") as master_pal:
                palette_buf = np.transpose(
                    np.frombuffer(master_pal.read(), dtype=np.uint8))

                # ideally the palette would be 512 entries (1536 bytes)
                # but .png only supports up to 256.
                # so in practice the palette is 64 entries (192 bytes)
                if len(palette_buf) != 192:
                    print_err_quit(
                        f"palette size is not 192: got \
                            {len(palette_buf)} bytes.")

                nespal = ImagePalette.ImagePalette(
                    mode="RGB", palette=list(palette_buf))
                imgpal = Image.new('P',(0,0))
                imgpal.putpalette(nespal)
                # quantize .png to become indexed
                im = im.convert(mode="RGB").quantize(
                    colors=256, palette=imgpal, dither=Image.Dither.NONE)
        else:
            # check if index is correct
            if len(im.getpalette()) != 192:
                print_err_quit(
                    f"image's index is not expected length of 192: got \
                        {len(im.getpalette())} bytes.")

        # at this point, image should be indexed .png
        # return image with raw indices of $00-$3F
        return np.array(im, dtype=int)



# filtering and raster configuration

from dataclasses import dataclass, field
@dataclass
class RasterTimings:
    """
    Scanline and raster timing constants. Used for generation logic.
    """
    # scanline timing constants
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
    VBLANK_PULSE: int

    # in NTSC, this fills active video
    BACKDROP: int

    PX_SAMPLES: int
    SC_SAMPLES: int

    NEXT_SHIFT: int = field(init=False)

    BEFORE_RENDER: int = field(init=False)
    AFTER_RENDER: int = field(init=False)

    BEFORE_VID: int = field(init=False)
    AFTER_VID: int = field(init=False)

    BEFORE_CBURST: int = field(init=False)
    AFTER_CBURST: int = field(init=False)

    PULSE_INDEX: int = field(init=False)

    SCANLINE_PIXELS: int = field(init=False)
    SCANLINE_SAMPLES: int = field(init=False)

    # vertical timing constants
    VSYNC_LINES: int
    PRERENDER_BLANK_LINES: int
    RENDER_LINES: int
    POSTRENDER_BD_LINES: int
    POSTRENDER_BLANK_LINES: int

    VSYNC_AREA: int = field(init=False)
    PRE_BLANK_AREA: int = field(init=False)
    ACTIVE_AREA: int = field(init=False)
    POST_RENDER_AREA: int = field(init=False)
    POST_BLANK_AREA: int = field(init=False)

    FIELD_SCANLINES: int = field(init=False)

    # for cropping, in final samples, NOT PPU pixels
    BEFORE_ACTIVE: int = field(init=False)
    ACTIVE_WIDTH: int
    AFTER_ACTIVE: int = field(init=False)

    ACTIVE_LINE_START: int
    ACTIVE_LINES: int
    ACTIVE_LINE_END: int = field(init=False)

    def __post_init__(s):
        s.BEFORE_RENDER = s.HSYNC + s.B_PORCH_A + s.CBURST + s.B_PORCH_B +\
            s.PULSE + s.L_BORDER
        s.AFTER_RENDER = s.BEFORE_RENDER + s.ACTIVE
        s.BEFORE_VID = s.HSYNC + s.B_PORCH_A + s.CBURST + s.B_PORCH_B
        s.AFTER_VID = s.BEFORE_VID + s.PULSE + s.L_BORDER + s.ACTIVE +\
            s.R_BORDER

        s.BEFORE_CBURST = s.HSYNC + s.B_PORCH_A
        s.AFTER_CBURST = s.BEFORE_CBURST + s.CBURST

        s.PULSE_INDEX = s.HSYNC + s.B_PORCH_A + s.CBURST + s.B_PORCH_B

        s.SCANLINE_PIXELS = s.HSYNC + s.B_PORCH_A + s.CBURST + s.B_PORCH_B +\
            s.PULSE + s.L_BORDER + s.ACTIVE + s.R_BORDER + s.F_PORCH
        s.SCANLINE_SAMPLES = s.SCANLINE_PIXELS * s.PX_SAMPLES

        # scanline shift to account for phase
        s.NEXT_SHIFT = (s.SCANLINE_SAMPLES) % s.SC_SAMPLES

        # for quicker if statement declaration
        s.VSYNC_AREA = s.VSYNC_LINES
        s.PRE_BLANK_AREA = s.VSYNC_AREA + s.PRERENDER_BLANK_LINES
        s.ACTIVE_AREA = s.PRE_BLANK_AREA + s.RENDER_LINES
        s.POST_RENDER_AREA = s.ACTIVE_AREA + s.POSTRENDER_BD_LINES
        s.POST_BLANK_AREA = s.POST_RENDER_AREA+s.POSTRENDER_BLANK_LINES
        s.FIELD_SCANLINES = s.POST_BLANK_AREA

        s.BEFORE_ACTIVE = s.BEFORE_VID * s.PX_SAMPLES
        s.AFTER_ACTIVE = s.BEFORE_ACTIVE + s.ACTIVE_WIDTH

        s.ACTIVE_LINE_END = s.ACTIVE_LINE_START + s.ACTIVE_LINES

def save_image(
        ppu_type: str,
        input_name: str,
        out: np.ndarray,
        phases: tuple[int],
        r: RasterTimings,
        full_resolution = False,
        debug = False
    ):
    """
    Save output to .png

    :param ppu_type: Composite PPU chip used for generating colors. See
        argparse arguments.
    :type ppu_type: str

    :param input_name: Filename of output image, excluding file extension.
    :type input_name: str

    :param out: Rendered RGB image.
    :type out: np.ndarray

    :param phases: List of phases rendered to output.
    :type phases: tuple[int]

    :param r: Raster timings.
    :type r: RasterTimings

    :param full_resolution: Render output without downscaling,
        defaults to False
    :type full_resolution: bool, optional

    :param debug: Enable debug messages and options, defaults to False
    :type debug: bool, optional
    """
    out_start = r.ACTIVE_LINE_START
    out_height = r.ACTIVE_LINES
    if ppu_type == "2C07":
        out_width=768
    else:
        out_width=640
    with Image.fromarray(np.ubyte(np.around(out * 255))) as imageout:
    # scale image
        if (full_resolution or debug):
            imageout = imageout.resize(
                (imageout.size[0], int(imageout.size[1]*r.PX_SAMPLES)),
                Image.Resampling.NEAREST)
        else:
            # crop to active video
            imageout = imageout.crop((
                r.BEFORE_ACTIVE,
                out_start,
                r.AFTER_ACTIVE,
                (out_start+out_height)
            ))
            imageout = imageout.resize(
                (out_width, imageout.size[1]),
                resample=Image.Resampling.LANCZOS)
            imageout = imageout.resize(
                (imageout.size[0], out_height*2), Image.Resampling.NEAREST)
        imageout.save(f"{input_name}_ppucvbs_ph_{phases}.png")

import ppu_composite as ppu

def RC_lowpass(
        signal: np.ndarray,
        phd: float,
        dt: float):
    """
    phase shift the composite using a simple lowpass
    for differential phase distortion.

    https://en.wikipedia.org/wiki/Low-pass_filter#Simple_infinite_impulse_response_filter

    :param signal: Input composite signal
    :type signal: np.ndarray

    :param phd: Phase distortion amount. See argparse arguments.
    :type phd: int

    :param dt: Sample time of signal (inverse of samplerate).
    :type dt: float



    :return: Phase distorted signal.
    :rtype: np.ndarray
    """
    v_out = np.zeros(signal.shape, np.float64)
    v_prev = signal[0]
    for i in range(len(signal)):
        # impedance changes depending on DAC tap
        # we approximate this by using the raw signal's voltage
        # https://forums.nesdev.org/viewtopic.php?p=287241#p287241
        # the phase shifts negative on higher levels according to
        # https://forums.nesdev.org/viewtopic.php?p=186297#p186297
        v_prev_norm = signal[i] / ppu.WHITE_LEVEL

        # RC constant tuned so that 0x and 3x colors have around 14 degree
        # delta when phd = 3
        # https://forums.nesdev.org/viewtopic.php?p=186297#p186297
        alpha = dt / (v_prev_norm * phd * 1e-8 + dt)
        v_prev = alpha * signal[i] + (1-alpha) * v_prev
        v_out[i] = v_prev
    return v_out

def QAM_phase(signal: np.ndarray, sc_samples: int):
    """Checks the phase for a sinusoid whose period is `sc_samples` samples.

    :param signal: Input sinusoidal signal
    :type signal: np.ndarray

    :param sc_samples: Period length of cycle, in samples
    :type sc_samples: int



    :return: Angle, in radians.
    :rtype: float
    """
    buf_size = len(signal)
    t = np.arange(sc_samples)
    u = np.sin(2*np.pi*t / sc_samples)
    v = np.cos(2*np.pi*t / sc_samples)

    tilecount = int(np.ceil(buf_size/sc_samples))
    u = np.tile(u, tilecount)[:buf_size]
    v = np.tile(v, tilecount)[:buf_size]
    signal_U = np.average(signal * u)
    signal_V = np.average(signal * v)
    return np.arctan2(signal_U, signal_V)

from scipy import signal

def configure_filters(
        ppu_type: str,
        filter_type: list[str],
        backdrop: int,
        plot_filters: bool = False
    ):
    """
    Configures the RasterTimings object for encoding and filter parameters
    for decoding.

    Depending on the PPU type, the RasterTimings will be initialized with the
    proper raster timing constants and optionally the backdrop color.

    Depending on the filter type, an FIR kernel or IIR filter
    numerator/denominator will be generated for complementary luma-chroma
    separation and QAM demodulation lowpassing. Otherwise they are initialized with `None`.

    Optionally, for debugging and developing filters, the filter's kernels (if
    FIR) and frequency response may be plotted with Matplotlib.

    :param ppu_type: Composite PPU chip used for generating colors. See
        argparse arguments.
    :type ppu_type: str

    :param filter_type: Filter types for LC separation and QAM filtering.
        See argparse arguments.
    :type filter_type: list[str]

    :param backdrop: Backdrop index color
    :type backdrop: int

    :param plot_filters: Enable plotting of filters using matplotlib,
        defaults to False
    :type plot_filters: bool, optional



    :return: A tuple of filtering parameters, the sampling time, and the
        RasterTimings object. You may choose to destructure the tuple for
        convenience.
        `(
            luma_fir,
            chroma_fir,
            b_luma_iir, a_luma_iir,
            b_chroma_iir, a_chroma_iir,
            Fs_dt,
            r,
        )`
    :rtype: (np.nparray optional,
        np.nparray optional,
        np.nparray optional, np.nparray optional,
        np.nparray optional, np.nparray optional,
        float,
        RasterTimings,
    )
    """

    # dummy, will be overwritten
    luma_fir = chroma_fir =\
    b_luma_iir = a_luma_iir =\
    b_chroma_iir = a_chroma_iir = None

    # https://www.nesdev.org/wiki/Cycle_reference_chart#Clock_rates

    X_main = 0
    r = []
    match ppu_type:
        case "2C07":
            X_main = 26601712.5
            r = RasterTimings(
                HSYNC = 25,
                B_PORCH_A = 4,
                CBURST = 15,
                B_PORCH_B = 12,
                PULSE = 0,
                L_BORDER = 11,
                ACTIVE = 256,
                R_BORDER = 9,
                F_PORCH = 9,
                CBURST_PHASE = 7,
                PX_SAMPLES = 10,
                SC_SAMPLES = 12,
                BACKDROP = ppu.BLANK_INDEX,
                VBLANK_PULSE=320,
                VSYNC_LINES=3,
                PRERENDER_BLANK_LINES=39,
                RENDER_LINES=240,
                POSTRENDER_BD_LINES=0,
                POSTRENDER_BLANK_LINES=30,
                # based on Rec. ITU-R BT.1700, Table 2 for 625 PAL
                #   f_H - (b + c)
                # = 64us - (10.5+1.2)us
                # approx. 2783 samples
                ACTIVE_WIDTH=int(round(52.3e-6*X_main*2)),
                ACTIVE_LINE_START=25-2, # symbol j - symbol l rounded down
                ACTIVE_LINES=287,
            )
        case "2C02":
            X_main = 236.25e6 / 11
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
                CBURST_PHASE = 8,
                PX_SAMPLES = 8,
                SC_SAMPLES = 12,
                BACKDROP = backdrop,
                VBLANK_PULSE=318,
                VSYNC_LINES=3,
                PRERENDER_BLANK_LINES=14,
                RENDER_LINES=240,
                POSTRENDER_BD_LINES=2,
                POSTRENDER_BLANK_LINES=3,
                # based on SMPTE 170M 2004, Figure 7
                # f_H - (h.ref to blank end - blank start to h.ref)
                # (1/(2*315e6/88/455)-(9.2+1.5)*1e-6)*PPU_Fs
                # = approx. 2270 samples
                ACTIVE_WIDTH=int(round(
                    (1/((2*315e6/88)/455)-(9.2+1.5)*1e-6)*X_main*2
                )),
                # start on first rendered scanline instead
                ACTIVE_LINE_START=17,
                ACTIVE_LINES=240,
            )
        case _:
            print_err_quit("Unknown PPU type")

    # samplerate and colorburst freqs
    # used for phase shift
    PPU_Fs = X_main * 2
    PPU_Cb = X_main / 6
    Fs_dt = 1/PPU_Fs

    MAX_DB_LOSS = 2
    MIN_DB_GAIN = 20

    # notch filter complementary decoding
    bw = 1.3e6
    b_luma_iir, a_luma_iir = signal.iirdesign(
        [PPU_Cb-bw, PPU_Cb+bw],
        [PPU_Cb, PPU_Cb],
        gpass=MAX_DB_LOSS,
        gstop=MIN_DB_GAIN,
        analog=False,
        ftype="butter",
        fs=PPU_Fs
    )
    # chroma filters, used both in notch and comb filtering
    # SMPTE 170M-2004, page 5, section 7.2
    b_chroma_iir, a_chroma_iir = signal.iirdesign(
        1.3e6,
        3.6e6,
        gpass=MAX_DB_LOSS,
        gstop=MIN_DB_GAIN,
        analog=False,
        ftype="butter",
        fs=PPU_Fs
    )

    # kernels

    box_kernel = np.full(r.SC_SAMPLES, 1, dtype=np.float64)
    box_kernel /= np.sum(box_kernel)

    kernel_taps = r.SC_SAMPLES*4 # M
    kernel_odd = kernel_taps+1
    # Equation 16-3
    # https://www.dspguide.com/ch16/2.htm
    bandwidth =  4/(kernel_odd)

    # MAGIC: align first notch to colorburst
    band_lo_cutoff = PPU_Cb-(bandwidth*PPU_Fs/2.9)
    band_hi_cutoff = PPU_Cb+(bandwidth*PPU_Fs/2.8)

    if PPU_Cb/PPU_Fs <= bandwidth:
        print_err_quit(
            f"tap size {kernel_taps} too small! bandwidth {bandwidth}\
            larger than cutoff {PPU_Cb/PPU_Fs}")

    # sinc kernel
    # Equation 16-1
    # https://www.dspguide.com/ch16/1.htm
    # sin(2*pi*fc*i) / i*pi
    # 2*fc* np.sinc(2*fc*i)

    lanczos_kernel = 2 * (band_lo_cutoff/PPU_Fs) * np.sinc(
        2 * (band_lo_cutoff/PPU_Fs) *
        # shifted to range 0 to M
        (np.arange(kernel_taps) - (kernel_taps/2))
    )

    lanczos_kernel *= signal.windows.lanczos(kernel_taps)
    lanczos_kernel /= np.sum(lanczos_kernel)

    # improvement by notching/bandpassing instead
    lanczos_notch_kernel = 2 * (band_hi_cutoff/PPU_Fs) * np.sinc(
        2 * (band_hi_cutoff/PPU_Fs) *
        # shifted to range 0 to M
        (np.arange(kernel_taps) - (kernel_taps/2))
    )
    # spectral invert to highpass
    lanczos_notch_kernel /= np.sum(lanczos_notch_kernel)
    lanczos_notch_kernel *= -1.0
    lanczos_notch_kernel[(kernel_taps)//2] += 1

    # add lowpass to turn it into bandlimit
    lanczos_notch_kernel += 2 * (band_lo_cutoff/PPU_Fs) * np.sinc(
        2 * (band_lo_cutoff/PPU_Fs) *
        # shifted to range 0 to M
        (np.arange(kernel_taps) - (kernel_taps/2))
    )

    lanczos_notch_kernel *= signal.windows.lanczos(kernel_taps)
    lanczos_notch_kernel /= np.sum(lanczos_notch_kernel)

    # align notch to cb
    # to align to -20dB from SMPTE 170M-2004, page 5, section 7.2, use std=3.5
    gauss_kernel = signal.windows.gaussian(kernel_odd, 9.4)
    gauss_kernel *= signal.windows.lanczos(kernel_odd)
    gauss_kernel /= np.sum(gauss_kernel)

    bw = 1e6
    taps, beta = signal.kaiserord(MIN_DB_GAIN, bw/(PPU_Fs))

    # Kaiser window method
    kaiser_kernel = signal.firwin(
        taps,
        cutoff=PPU_Cb-bw/2,
        window=("kaiser", beta),
        pass_zero=True,
        fs=PPU_Fs
    )

    # least squares error minimization
    leastsquares_kernel = signal.firls(
        kernel_odd,
        [0, PPU_Cb-1.3e6, PPU_Cb, PPU_Fs/2],
        [1, 1, 0, 1],
        fs=PPU_Fs
    )

    nofilt = np.array([1.0], dtype=np.float64)

    kernel_dict = {
        "lanczos": lanczos_kernel,
        "lanczos_notch": lanczos_notch_kernel,
        "gauss": gauss_kernel,
        "box": box_kernel,
        "kaiser": kaiser_kernel,
        "leastsquares": leastsquares_kernel,
        "none": nofilt,
        "iir": None,
    }

    luma_fir = kernel_dict[filter_type[0]]
    if len(filter_type) == 1:
        chroma_fir = luma_fir
    elif len(filter_type) == 2:
        chroma_fir = kernel_dict[filter_type[1]]
    else:
        print_err_quit(f"Unknown filter type count: {len(filter_type)}.")

    if plot_filters:
        import matplotlib.pyplot as plt
        # luma
        plt.figure(figsize=(10,5))
        if luma_fir is None:
            w, h = signal.freqz(b_luma_iir, a_luma_iir, fs=PPU_Fs)
            x = w
            y = 20 * np.log10(abs(h))
            plt.subplot(2, 1, (1, 2))
            plt.semilogx(x, y)
        else:
            w, h = signal.freqz(luma_fir)
            x = w * PPU_Fs * 1.0 / (2 * np.pi)
            y = 20 * np.log10(abs(h))
            plt.subplot(2, 2, 3)
            plt.plot(luma_fir, "o")
            plt.subplot(2, 2, (1, 2))
            plt.semilogx(x, y)
        # chroma
        if chroma_fir is None:
            w, h = signal.freqz(b_chroma_iir, a_chroma_iir, fs=PPU_Fs)
            x = w
            y = 20 * np.log10(abs(h))
            plt.subplot(2, 2, (1, 2))
            plt.semilogx(x, y)
        else:
            w, h = signal.freqz(chroma_fir)
            x = w * PPU_Fs * 1.0 / (2 * np.pi)
            y = 20 * np.log10(abs(h))
            plt.subplot(2, 2, 4)
            plt.plot(chroma_fir, "o")
            plt.subplot(2, 2, (1, 2))
            plt.semilogx(x, y)
        plt.ylabel('Amplitude [dB]')
        plt.xlabel('Frequency [Hz]')
        plt.title('Frequency response')
        plt.grid(which='both', linestyle='-', color='grey')
        plt.xticks(
            [PPU_Fs/32, PPU_Fs/16, PPU_Cb, PPU_Fs/2],
            ["px/2", "px","cb", "nyquist"])
        plt.show()

    return (
        luma_fir,
        chroma_fir,
        b_luma_iir, a_luma_iir,
        b_chroma_iir, a_chroma_iir,
        Fs_dt,
        r
    )


# encoding / decoding

def encode_blank_scanline(
        starting_phase: int,
        blank: np.ndarray,
        r: RasterTimings,
        alternate_line = False,
    ):
    cvbs_ppu = np.full(
        r.SCANLINE_SAMPLES, ppu.SYNC_LEVEL, dtype=np.float64)
    scanline_phase = starting_phase
    for pixel in range(blank.shape[0]):
        for sample in range(r.PX_SAMPLES):
            cvbs_ppu[pixel*r.PX_SAMPLES + sample] =\
                ppu.encode_composite_sample(
                    blank[pixel],
                    scanline_phase,
                    False,
                    r.CBURST_PHASE,
                    alternate_line
                )
            scanline_phase = (scanline_phase + 1) % r.SC_SAMPLES
    return cvbs_ppu, scanline_phase

def encode_blank_syncs(
        starting_phase: int,
        scanline: int,
        r: RasterTimings,
        alternate_line = False,
):
    """
    Encode a scanline that does not have PPU data.

    :param starting_phase: Phase of color generator clock. Ranges from 0-11.
    :type starting_phase: int

    :param scanline: Scanline number.
    :type scanline: int

    :param r: Raster timings.
    :type r: RasterTimings

    :param alternate_line: Alternate the line phase for PAL, defaults to False
    :type alternate_line: bool, optional



    :return: A Numpy array of voltage samples in a single scanline, along with
        the next accumulated phase.
    :rtype: (np.ndarray, int)
    """
    cvbs_ppu = np.full(
        r.SCANLINE_SAMPLES, ppu.SYNC_LEVEL, dtype=np.float64)
    scanline_phase = starting_phase

    # vertical sync
    if scanline<r.VSYNC_AREA:
        scanline_phase = (scanline_phase + r.SCANLINE_SAMPLES) %\
            r.SC_SAMPLES

        cvbs_ppu[r.VBLANK_PULSE*r.PX_SAMPLES:] = ppu.BLANK_LEVEL

    # prerender render blanking lines
    elif scanline<r.PRE_BLANK_AREA:
        blank = np.full(r.SCANLINE_PIXELS, ppu.BLANK_INDEX, dtype=np.int16)
        blank[:r.HSYNC] = ppu.SYNC_INDEX
        blank[r.BEFORE_CBURST:r.AFTER_CBURST] = ppu.COLORBURST_INDEX
        blank[r.BEFORE_VID:r.BEFORE_RENDER] = ppu.BLANK_INDEX

        cvbs_ppu, scanline_phase = encode_blank_scanline(
            scanline_phase, blank, r, alternate_line)

    elif scanline<r.ACTIVE_AREA:
        # this should be handled by the parent function, return error
        print_err_quit("Trying to render scanline in blank sync function")

    elif scanline<r.POST_RENDER_AREA:
        blank = np.full(r.SCANLINE_PIXELS, ppu.BLANK_INDEX, dtype=np.int16)
        blank[:r.HSYNC] = ppu.SYNC_INDEX
        blank[r.BEFORE_CBURST:r.AFTER_CBURST] = ppu.COLORBURST_INDEX
        blank[r.BEFORE_VID:r.AFTER_VID] = r.BACKDROP
        if r.PULSE != 0:
            blank[r.PULSE_INDEX] &= 0xF0

        cvbs_ppu, scanline_phase = encode_blank_scanline(
            scanline_phase, blank, r, alternate_line)

    # misc. blank lines
    elif scanline<r.POST_BLANK_AREA:
        blank = np.full(r.SCANLINE_PIXELS, ppu.BLANK_INDEX, dtype=np.int16)
        blank[:r.HSYNC] = ppu.SYNC_INDEX
        blank[r.BEFORE_CBURST:r.AFTER_CBURST] = ppu.COLORBURST_INDEX
        blank[r.BEFORE_VID:r.BEFORE_RENDER] = ppu.BLANK_INDEX

        cvbs_ppu, scanline_phase = encode_blank_scanline(
            scanline_phase, blank, r, alternate_line)

    return cvbs_ppu, scanline_phase

def encode_scanline(
        ppu_type: str,
        data: np.ndarray,
        starting_phase: int,
        scanline: int,
        r: RasterTimings,
        alternate_line = False,
        skip_dot = False,
    ):
    """
    Encodes one line of raw ppu pixels. returns raw voltage and next phase.

    :param ppu_type: Composite PPU chip used for generating colors. See
        argparse arguments.
    :type ppu_type: str

    :param data: Raw PPU pixels, in a 1D uint16 Numpy array.
    :type data: np.ndarray

    :param starting_phase: Phase of color generator clock. Ranges from 0-11.
    :type starting_phase: int

    :param scanline: Scanline number.
    :type scanline: int

    :param r: Raster timings.
    :type r: RasterTimings

    :param alternate_line: Alternate the line phase for PAL, defaults to False
    :type alternate_line: bool, optional

    :param skip_dot: Enable skipped dot behavior in 2C02s, defaults to False
    :type skip_dot: bool, optional



    :return: A Numpy array of voltage samples in a single scanline, along with
        the next accumulated phase.
    :rtype: (np.ndarray, int)
    """

    raw_ppu = np.full(r.SCANLINE_PIXELS, ppu.BLANK_INDEX, dtype=np.int16)
    cvbs_ppu = np.full(
        r.SCANLINE_SAMPLES-int(skip_dot), ppu.SYNC_LEVEL, dtype=np.float64)

    raw_ppu[0:r.HSYNC] = ppu.SYNC_INDEX
    raw_ppu[r.HSYNC:] = ppu.BLANK_INDEX
    raw_ppu[r.BEFORE_CBURST:r.AFTER_CBURST] = ppu.COLORBURST_INDEX
    raw_ppu[r.BEFORE_VID:r.BEFORE_RENDER] = r.BACKDROP

    # PAL crop
    if ppu_type =="2C07":
        if scanline != 0:
            raw_ppu[r.BEFORE_RENDER+2:r.AFTER_RENDER-2] = data[2 :-2]
    else:
        raw_ppu[r.BEFORE_RENDER:r.AFTER_RENDER] = data
    raw_ppu[r.AFTER_RENDER:r.AFTER_VID] = r.BACKDROP
    if r.PULSE != 0:
        raw_ppu[r.PULSE_INDEX] &= 0xF0
    scanline_phase = starting_phase

    # implement skipped dot by rolling the portion of video

    # due to skipped dot, the raw PPU index
    # and the encoded index is unsynced
    px_index = 0

    # encode one scanline, then lowpass it accordingly
    for pixel in range(raw_ppu.shape[0]):
        # dot skip
        # dot 340, scanline 0
        if pixel == (r.BEFORE_RENDER - 1) and skip_dot:
            continue

        for sample in range(r.PX_SAMPLES):
            # todo: concurrency
            cvbs_ppu[px_index*r.PX_SAMPLES + sample] =\
                ppu.encode_composite_sample(
                    raw_ppu[pixel],
                    scanline_phase,
                    False,
                    r.CBURST_PHASE,
                    alternate_line
                )
            scanline_phase = (scanline_phase + 1) % r.SC_SAMPLES
        px_index += 1

    return cvbs_ppu, scanline_phase

def yc_fir(cvbs_ppu: np.ndarray, kernel: np.ndarray):
    """Complementary luma-chroma separation using FIR filtering

    :param cvbs_ppu: Composite signal scanline input
    :type cvbs_ppu: np.ndarray

    :param kernel: FIR kernel for YC separation.
    :type kernel: np.ndarray



    :return: Luma and chroma buffer
    :rtype: (np.ndarray, np.ndarray)
    """
    y_line = signal.convolve(cvbs_ppu, kernel, mode="same")
    c_line = cvbs_ppu - y_line
    return y_line, c_line

def yc_iir(cvbs_ppu: np.ndarray, b_luma: np.ndarray, a_luma: np.ndarray):
    """Complementary luma-chroma separation using IIR filtering

    :param cvbs_ppu: Composite signal scanline input
    :type cvbs_ppu: np.ndarray

    :param b_luma: IIR filter numerator for YC separation.
    :type b_luma: np.ndarray

    :param a_luma: IIR filter denominator for YC separation.
    :type a_luma: np.ndarray


    :return: Luma and chroma buffer
    :rtype: (np.ndarray, np.ndarray)
    """
    y_line = signal.filtfilt(b_luma, a_luma, cvbs_ppu)
    c_line = cvbs_ppu - y_line
    return y_line, c_line

def yc_pal_delayline(
        cvbs_ppu: np.ndarray,
        prev_line: np.ndarray,
        r: RasterTimings,
        b_luma: np.ndarray,
        a_luma: np.ndarray,
        luma_kernel: np.ndarray,
        first_line: bool
    ):
    """
    bandpass and combine lines in specific way to retrieve U and V.

    based on Single Delay Line PAL Y/C Separator
    Jack, K. (2007). NTSC and PAL digital encoding and decoding. In Video
    Demystified (5th ed., p. 450). Elsevier.
    https://archive.org/details/video-demystified-5th-edition/

    :param cvbs_ppu: Composite signal scanline input
    :type cvbs_ppu: np.ndarray

    :param prev_line: Previous composite signal scanlines for comb filtering.
    :type prev_line: np.ndarray

    :param r: Raster timings.
    :type r: RasterTimings

    :param b_luma: IIR filter numerator for YC separation. May be `None`.
    :type b_luma: np.ndarray, optional

    :param a_luma: IIR filter denominator for YC separation. May be `None`.
    :type a_luma: np.ndarray, optional

    :param luma_kernel: FIR kernel for YC separation. May be `None`.
    :type luma_kernel: np.ndarray, optional

    :param first_line: If this is the first scanline being sent.
    :type first_line: bool



    :return: Y buffer, U buffer, V buffer, and the previous line
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    if first_line:
        prev_line = np.zeros(r.SCANLINE_SAMPLES, dtype=np.float64)

    u_line = (cvbs_ppu + prev_line) / 2
    v_line = (cvbs_ppu - prev_line) / 2

    # rotate chroma line to account for next line's phase shift
    prev_line = np.roll(cvbs_ppu, -r.NEXT_SHIFT)

    # filter luma
    if luma_kernel is None:
        y_line, _ = yc_iir(cvbs_ppu, b_luma, a_luma)
    else:
        y_line, _ = yc_fir(cvbs_ppu, luma_kernel)
    return y_line, u_line, v_line, prev_line

def yc_comb_1line(
        cvbs_ppu: np.ndarray,
        prev_line: np.ndarray,
        r: RasterTimings,
        b_luma: np.ndarray,
        a_luma: np.ndarray,
        luma_kernel: np.ndarray,
        first_line: bool
    ):
    """
    Single-line comb filter for non-PAL composite

    :param cvbs_ppu: Composite signal scanline input
    :type cvbs_ppu: np.ndarray

    :param prev_line: Previous composite signal scanlines for comb filtering.
    :type prev_line: np.ndarray

    :param r: Raster timings.
    :type r: RasterTimings

    :param b_luma: IIR filter numerator for YC separation. May be `None`.
    :type b_luma: np.ndarray, optional

    :param a_luma: IIR filter denominator for YC separation. May be `None`.
    :type a_luma: np.ndarray, optional

    :param luma_kernel: FIR kernel for YC separation. May be `None`.
    :type luma_kernel: np.ndarray, optional

    :param first_line: If this is the first scanline being sent.
    :type first_line: bool



    :return: Y buffer, U buffer, V buffer, and the previous line
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """

    if first_line:
        prev_line = np.zeros(r.SCANLINE_SAMPLES, dtype=np.float64)

    u_line = v_line = np.array(cvbs_ppu - prev_line) / 2

    # align chroma phase of next line to be out-of-phase
    # to avoid any funny business
    prev_line = np.roll(cvbs_ppu, r.NEXT_SHIFT//2)

    # notch filter the luma
    if luma_kernel is None:
        y_line, _ = yc_iir(cvbs_ppu, b_luma, a_luma)
    else:
        y_line, _ = yc_fir(cvbs_ppu, luma_kernel)
    return y_line, u_line, v_line, prev_line


# Experiment: use 1d comb filtering to process chroma
CHROMA_COMB_IN_2D_COMB = True

def yc_comb_2line(
        cvbs_ppu,
        prev_line,
        r: RasterTimings,
        b_luma: np.ndarray,
        a_luma: np.ndarray,
        luma_kernel: np.ndarray,
        pal_comb: bool,
        first_line: bool
    ):
    """
    Two-line comb filtering based on two circuits:

    Figure 9.45, Figure 9.47
    Two-Line Delay PAL Y/C Separator Optimized for Standards Conversion and
    Video Processing,
    Two-Line Delay NTSC Y/C Separator for Standards Conversion and Video
    Processing.
    Jack, K. (2007). NTSC and PAL digital encoding and decoding. In Video
    Demystified (5th ed., p. 456). Elsevier.
    https://archive.org/details/video-demystified-5th-edition/

    :param cvbs_ppu: Composite signal scanline input
    :type cvbs_ppu: np.ndarray

    :param prev_line: Previous composite signal scanlines for comb filtering.
    :type prev_line: np.ndarray

    :param r: Raster timings.
    :type r: RasterTimings

    :param b_luma: IIR filter numerator for YC separation. May be `None`.
    :type b_luma: np.ndarray, optional

    :param a_luma: IIR filter denominator for YC separation. May be `None`.
    :type a_luma: np.ndarray, optional

    :param luma_kernel: FIR kernel for YC separation. May be `None`.
    :type luma_kernel: np.ndarray, optional

    :param first_line: If this is the first scanline being sent.
    :type first_line: bool



    :return: Y buffer, U buffer, V buffer, and the previous lines (stored as
    one NDarray)
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    if first_line:
        prev_line = np.zeros((r.SCANLINE_SAMPLES, 2), dtype=np.float64)

    # For standards conversion (Figure 9.45), the chrominance signal
    # is just the full-bandwidth composite video signal.
    # We use the delayed line to sync with luma
    c_line = prev_line[:, 1]

    if CHROMA_COMB_IN_2D_COMB:
        if pal_comb:
            # ugly: roll this THE OTHER WAY for proper chroma extraction
            prev_0 = np.roll(prev_line[:,0], -r.NEXT_SHIFT//2 - r.NEXT_SHIFT)
            prev_1 = np.roll(prev_line[:,1], -r.NEXT_SHIFT*3)
            u_line = np.array(prev_0+prev_1) / 2
            v_line = np.array(prev_0-prev_1) / 2
        else:
            u_line = v_line = np.array(prev_line[:,0]-prev_line[:,1]) / 2

    y_line = (cvbs_ppu + prev_line[:, 1])/2
    y_line = prev_line[:, 0] - y_line

    if not pal_comb:
        y_line /= 2

    # this is sharper than using an iirpeak
    # recommended to use a notch filter
    if luma_kernel is None:
        _, y_line = yc_iir(y_line, b_luma, a_luma)
    else:
        _, y_line = yc_fir(y_line, luma_kernel)

    if not CHROMA_COMB_IN_2D_COMB: u_line = v_line = c_line

    y_line = prev_line[:, 0] - y_line

    # rotate chroma line to account for next line's phase shift
    prev_line[:, 1] = np.roll(prev_line[:,0], r.NEXT_SHIFT//2)
    prev_line[:, 0] = np.roll(cvbs_ppu, r.NEXT_SHIFT//2)

    return y_line, u_line, v_line, prev_line

def decode_scanline(
        cvbs_ppu: np.ndarray,
        ppu_type: str,
        scanline: int,
        r: RasterTimings,
        b_luma: np.ndarray, a_luma: np.ndarray,
        b_chroma: np.ndarray, a_chroma: np.ndarray,
        decoding_filter: str,
        luma_kernel: np.ndarray,
        chroma_kernel: np.ndarray,
        prev_line: np.ndarray,
        alternate_line = False
    ):
    """
    Decodes a composite scanline into YUV.

    :param cvbs_ppu: Composite signal scanline input
    :type cvbs_ppu: np.ndarray

    :param ppu_type: Composite PPU chip used for generating colors. See
        argparse arguments.
    :type ppu_type: str

    :param scanline: Scanline number
    :type scanline: int

    :param r: Raster timings.
    :type r: RasterTimings

    :param b_luma: IIR filter numerator for YC separation. May be `None`.
    :type b_luma: np.ndarray, optional

    :param a_luma: IIR filter denominator for YC separation. May be `None`.
    :type a_luma: np.ndarray, optional

    :param b_chroma: IIR filter numerator for QAM demodulation filtering.
        May be `None`.
    :type b_chroma: np.ndarray, optional

    :param a_chroma: IIR filter denominator for QAM demodulation filtering.
        May be `None`.
    :type a_chroma: np.ndarray, optional

    :param decoding_filter: Method for luma and chroma decoding. See argparse
        arguments.
    :type decoding_filter: str

    :param luma_kernel: FIR kernel for YC separation. May be `None`.
    :type luma_kernel: np.ndarray, optional

    :param chroma_kernel: FIR kernel for QAM demodulation filtering.
        May be `None`.
    :type chroma_kernel: np.ndarray, optional

    :param prev_line: Previous composite signal scanlines for comb filtering.
    :type prev_line: np.ndarray

    :param alternate_line: Alternate the line phase for PAL, defaults to False
    :type alternate_line: bool, optional



    :return: The current decoded scanline in YUV,
        and the previous composite scanline.
    :rtype: np.ndarray, np.ndarray
    """
    YUV_line = np.zeros((r.SCANLINE_SAMPLES, 3), dtype=np.float64)

    # separate luma and chroma
    # TODO: provide non-compelementary filtering for luma and chroma
    # ex: 2-line comb for luma and 1-line comb for chroma
    first_line = scanline == 0
    match decoding_filter:
        case "1-line":
            if ppu_type == "2C07":
                YUV_line[:, 0], u_line, v_line, prev_line = yc_pal_delayline(
                        cvbs_ppu,
                        prev_line,
                        r,
                        b_luma, a_luma,
                        luma_kernel,
                        first_line
                    )
            else:
                YUV_line[:, 0], u_line, v_line, prev_line = yc_comb_1line(
                    cvbs_ppu,
                    prev_line,
                    r,
                    b_luma, a_luma,
                    luma_kernel,
                    first_line
                )
        case "2-line":
            YUV_line[:, 0], u_line, v_line, prev_line = yc_comb_2line(
                cvbs_ppu,
                prev_line,
                r,
                b_luma, a_luma,
                luma_kernel,
                ppu_type == "2C07",
                first_line
            )
        case "compl":
            if luma_kernel is not None:
                YUV_line[:, 0], u_line = yc_fir(cvbs_ppu, luma_kernel)
            else:
                YUV_line[:, 0], u_line = yc_iir(cvbs_ppu, b_luma, a_luma)
            v_line = u_line
        case _:
            print_err_quit(
                f"unknown decoding filter option: {decoding_filter}")

    DISABLE_CHROMA = False

    # this is easily the most complicated part of decoding, especially in PAL,
    # and 1-line comb filtering.
    if not DISABLE_CHROMA:
        # measure actual angle of colorburst
        start = (r.BEFORE_CBURST)*r.PX_SAMPLES
        cburst_phase = QAM_phase(
            u_line[
                start:start+r.SC_SAMPLES*12],
            r.SC_SAMPLES
        )
        cburst_shift = (r.NEXT_SHIFT-start) % r.SC_SAMPLES

        t = np.arange(r.SCANLINE_SAMPLES, dtype=np.float64)-cburst_shift

        # set phase to 180 degrees
        cburst_phase -= -np.pi

        v_flip = 1

        # adjust with colorburst = 135 degrees
        # with this setting, hues match these images:
        # https://forums.nesdev.org/viewtopic.php?p=133638#p133638
        # https://forums.nesdev.org/viewtopic.php?p=187338#p187338
        if ppu_type == "2C07":
            # in 2-line comb, chroma is off by one
            if decoding_filter == "2-line":
                alternate_line = not alternate_line

            if alternate_line: v_flip = -1

            # nudge reference from 180 degrees to 135 degrees
            if decoding_filter == "1-line"\
                or decoding_filter == "2-line"\
                and CHROMA_COMB_IN_2D_COMB:
                # delay line handles the colorburst swing for us
                cburst_phase -= np.pi/4
            else:
                cburst_phase += np.pi/4*v_flip*-1
        # MAGIC: off by one?
        else: t -= 1

        # generate one period of UV decoding sines
        U_decode = np.sin((2*np.pi*t/r.SC_SAMPLES)-cburst_phase)*2
        V_decode = np.cos((2*np.pi*t/r.SC_SAMPLES)-cburst_phase)*2*v_flip

        # qam
        if not DISABLE_CHROMA:
            YUV_line[:, 1] = u_line * U_decode
            YUV_line[:, 2] = v_line * V_decode

    if chroma_kernel is not None:
    # filter chroma
        YUV_line[:, 1], _ = yc_fir(YUV_line[:, 1], chroma_kernel)
        YUV_line[:, 2], _ = yc_fir(YUV_line[:, 2], chroma_kernel)
    else:
        YUV_line[:, 1] = signal.filtfilt(b_chroma, a_chroma, YUV_line[:, 1])
        YUV_line[:, 2] = signal.filtfilt(b_chroma, a_chroma, YUV_line[:, 2])

    return YUV_line, prev_line

# keeps track of odd-even frame render parity
is_odd_frame = True


def encode_frame(raw_ppu,
        ppu_type: str,
        starting_phase: int,
        phd: float,
        filter_config: tuple,
        decoding_filter: str,
        skip_dot = False,
        debug = False,
        disable_yc: str = None,
        plot_raster = False
    ):

    """
    Encodes a (240,256) array of uint16 raw PPU pixels into an composite
    filtered RGB output image. Also outputs the next starting phase for the
    next frame.

    :param ppu_type: Composite PPU chip used for generating colors. See
        argparse arguments.
    :type ppu_type: str

    :param starting_phase: Phase of color generator clock. Ranges from 0-11.
    :type starting_phase: int

    :param phd: Phase distortion amount. See argparse arguments.
    :type phd: float

    :param filter_config: Filter configuration generated by
        `configure_filters()`.
        `(
            luma_fir,
            chroma_fir,
            b_luma_iir, a_luma_iir,
            b_chroma_iir, a_chroma_iir,
            Fs_dt,
            r
        )`
    :type filter_config: tuple(np.nparray optional,
        np.nparray optional,
        np.nparray optional, np.nparray optional,
        np.nparray optional, np.nparray optional,
        float,
        RasterTimings,
        float
    )

    :param decoding_filter: Method for luma and chroma decoding. See argparse
        arguments.
    :type decoding_filter: str

    :param skip_dot: Enable skipped dot behavior in 2C02s, defaults to False
    :type skip_dot: bool, optional

    :param debug: Enable debug messages and options, defaults to False
    :type debug: bool, optional

    :param disable_yc: Disables chroma by setting UV to 0. Disables luma by
        setting Y to 0.5. Defaults to `None`.
    :type disable_yc: str, optional

    :param plot_raster: Enable debug messages and options, defaults to False
    :type plot_raster: bool, optional



    :return: The output RGB image, and the beginning phase of the next field.
    :rtype: np.ndarray, int
    """

    (
        luma_kernel,
        chroma_kernel,
        b_luma, a_luma,
        b_chroma, a_chroma,
        Fs_dt,
        r
    ) = filter_config

    out = np.zeros((r.FIELD_SCANLINES , r.SCANLINE_SAMPLES, 3), np.float64)

    # 2-line comb filtering needs an extra scanline
    delay_lines = 0
    if decoding_filter == "2-line":
        delay_lines = 1
        raw_ppu = np.append(
            raw_ppu,
            np.full((delay_lines, raw_ppu.shape[1]), 0x0F, dtype=np.int16),
            axis=0
        )

    # dummy; will be overwritten with an array
    prev_line = 0

    # hack! keep track of oddness of rendered frames overall
    global is_odd_frame
    is_odd_frame = not is_odd_frame

    skip = skip_dot and is_odd_frame
    if ppu_type == "2C07": skip_dot = False

    next_phase = starting_phase % r.SC_SAMPLES

    full_signal = []

    for scanline in range(out.shape[0]):
        # todo: concurrency for non-comb filter decoding
        alternate_line = ppu_type == "2C07" and (
            (scanline-delay_lines) % 2 != 0)

        rawppu_i = scanline - r.PRE_BLANK_AREA
        skipdot_line = skip and rawppu_i==0
        if scanline<r.PRE_BLANK_AREA:
            # vsync, and blanking
            cvbs_ppu, next_phase = encode_blank_syncs(
                next_phase,
                scanline,
                r,
                alternate_line,
            )
        elif scanline<r.ACTIVE_AREA:
            # encode scanline
            cvbs_ppu, next_phase = encode_scanline(
                ppu_type,
                raw_ppu[rawppu_i],
                next_phase,
                rawppu_i,
                r,
                alternate_line,
                skipdot_line
            )
        else:
            # vsync, and blanking
            cvbs_ppu, next_phase = encode_blank_syncs(
                next_phase,
                scanline,
                r,
                alternate_line,
            )

        # lowpass
        # unfortunately, this only works on raw voltages, not normalized
        if phd > 0:
            cvbs_ppu = RC_lowpass(cvbs_ppu, phd, Fs_dt)
        full_signal += cvbs_ppu.tolist()

        # normalize
        cvbs_ppu -= ppu.BLANK_LEVEL
        cvbs_ppu /= ppu.WHITE_LEVEL - ppu.BLANK_LEVEL

        # decode scanline
        decoded_scanline, prev_line = decode_scanline(
            np.append(cvbs_ppu, [ppu.SYNC_LEVEL])
                if skipdot_line else cvbs_ppu,
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
        )

        line_index = scanline
        # delay scanline in case of comb filtering
        if decoding_filter == "2-line":
            line_index -= delay_lines
            # do not allow negative indexing
            if line_index < 0: continue

        out[line_index] = decoded_scanline

    # blank luma or chroma, if disabled
    if disable_yc == "chroma":
        out[:, :, 1] *= 0
        out[:, :, 2] *= 0
    elif disable_yc == "luma":
        out[:, :, 0] = 0.5

    # convert to RGB
    if not debug:
        out = np.einsum(
            'ij,klj->kli', np.linalg.inv(RGB_to_YUV), out, dtype=np.float64)
    # fit RGB within range of 0.0-1.0
    np.clip(out, 0, 1, out=out)

    return out, next_phase, full_signal

def main(argv=None):
    args = parse_argv(argv or sys.argv)
    # load input image and parse it for indices
    raw_ppu = np.zeros((240, 256), dtype=np.int16)
    if args.raw_ppu_px: raw_ppu = parse_raw_ppu_px(args.input)
    else: raw_ppu = parse_indexed_png(args.input, args.palette)

    phase = args.color_clock_phase

    avg = args.average and args.frames > 1

    frames = []

    filter_config = configure_filters(
        args.ppu,
        args.filter_type,
        int(args.backdrop, 16),
        args.plot_filters,
    )

    r = filter_config[7]

    # compensate for vsync and blank
    phase = (phase - r.PRE_BLANK_AREA*r.SCANLINE_SAMPLES) % r.SC_SAMPLES

    full_signal = []

    image_filename = os.path.splitext(args.input)[0]

    for _ in range(args.frames):
        # naming convention is the phase at the start of the active portion
        prev = (phase + r.PRE_BLANK_AREA*r.SCANLINE_SAMPLES) % r.SC_SAMPLES
        out, phase, signal = encode_frame(raw_ppu,
            args.ppu,
            phase,
            args.phase_distortion,
            filter_config,
            args.decoding_filter,
            (not args.noskipdot),
            args.debug,
            args.disable
        )

        full_signal += signal

        frames.append((out, prev))

        if not (avg or args.difference):
            print(phase)
            save_image(
                args.ppu,
                image_filename,
                out,
                prev,
                r,
                args.full_resolution,
                args.debug
            )

    if avg or args.difference:
        images, phases = zip(*frames)
        images = np.array(images)

        if avg:
            out = np.average(images, axis=0)
        elif args.difference:
            out = images[0] - images[1]
            out = np.absolute(out)
        save_image(
            args.ppu,
            image_filename,
            out,
            phases,
            r,
            args.full_resolution,
            args.debug
        )

    # # TODO: use pyflac and encode signal in realtime
    # import soundfile as sf

    # sig_min = np.min(full_signal)
    # sig_max = np.max(full_signal)
    # full_signal = np.asarray(full_signal, dtype=np.float64)
    # full_signal -= sig_min
    # full_signal /= (sig_max - sig_min)
    # full_signal -= 0.5
    # full_signal = np.clip(full_signal, -1, 1)
    # full_signal *= (2**15)
    # full_signal = np.int16(full_signal)

    # Fs_dt = filter_config[6]

    # sf.write(f"{image_filename}_{1/Fs_dt}MHz_16_bit.cvbs", full_signal,
    #     subtype="PCM_16", samplerate=96000, format="FLAC")

if __name__=='__main__':
    main(sys.argv)
