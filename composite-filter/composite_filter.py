# composite filter
# Copyright (C) 2023 Persune, MIT-0

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def parse_argv(argv):
    parser=argparse.ArgumentParser(
        description="Composite encoder-decoder",
        epilog="version 0.2.0")
    parser.add_argument("input_image", type=Path, help="input image")
    # output options
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="debug messages")
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
            "2-line",
            "3-line",
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
            Default =\"iir\".")
    parser.add_argument(
        "-full",
        "--full_resolution",
        action="store_true",
        help="Saves full scanlines instead of a cropped output.\
            Scanlines are scaled to 8x to preserve 1:1 pixel aspect ratio.")
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

### filtering and raster config

from dataclasses import dataclass, field

@dataclass
class RasterTimings:
    # definitions
    F_H: float      # horizontal line frequency, in Hertz
    F_V: float      # vertical line frequency, in Hertz
    F_SC: float     # color subcarrier frequency, in Hertz

    CB_BURST_START: int # h.ref to burst start, in subcarrier cycles
    CB_BURST_WIDTH: int # full amplitude colorburst width, in subcarrier cycles
    CB_BURST_ENV: int   # colorburst transition time, in nanoseconds

    # horizontal timings
    H_SYNC_LENGTH: float    # Horizontal sync length, in microseconds
    H_SYNC_ENV: float       # sync rise/fall time, in nanoseconds
    H_BLANK_ENV: float      # blank rise/fall time, in nanoseconds
    H_SYNC_TO_ACTIVE: float # h.ref to blanking end, in microseconds
    H_ACTIVE_TO_SYNC: float # "back porch" to h.ref, in microseconds; maybe not?
    H_ACTIVE: float = field(init=False)

    # vertical timings, see Figure 7 in SMPTE 170M-2004
    # start of
    V_BLANK_INT: int        # length of blanking interval, in lines
    # starting line of vsync pre-equalizing pulse interval
    V_PREQ: int
    # starting line of vsync pulse interval
    V_SERR: int
    # starting line of vsync post-equalizing pulse interval
    V_POEQ: int

    INTERLACED: bool

    FS: int             # samplerate
    ACTIVE_W_PX: int    # samples per active area width
    ACTIVE_H_PX: int    # scanlines per active area height
    # samplerate derived
    FULL_W_PX: int = field(init=False)      # samples per scanline
    FULL_H_PX: int = field(init=False)      # scanlines per frame
    HALF_W_PX: int = field(init=False)      # samples per half scanline rounded up
    HALF_H_PX: int = field(init=False)      # scanlines per field rounded up

    # sample-based timings
    H_SYNC_LENGTH_PX: int = field(init=False)
    H_ACTIVE_PX: int = field(init=False)
    # "center" h active video start offset
    # depends on actual active area in samples, and given ACTIVE_W_PX
    ACTIVE_W_PX_OFFS: int = field(init=False)

    # signal output
    SIG_WHITE: float    # white level, in IRE
    SIG_BLACK: float    # black (setup) level, in IRE
    SIG_BLANK: float    # blanking level, in IRE
    SIG_CBAMP: float    # colorburst pp amplitude, in IRE
    SIG_SYNC: float     # sync level, in IRE

    def __post_init__(s):
        s.FULL_W_PX = round(s.FS / s.F_H)
        s.FULL_H_PX = round(s.F_H / s.F_V * 2)
        s.HALF_W_PX = int(np.ceil(s.FULL_W_PX / 2))
        s.HALF_H_PX = int(np.ceil(s.FULL_H_PX / 2))
        s.H_SYNC_LENGTH_PX = round(s.FS * s.H_SYNC_LENGTH / 1e6)
        s.H_ACTIVE = 1/s.F_H*1e6 - (s.H_ACTIVE_TO_SYNC + s.H_SYNC_TO_ACTIVE)
        s.H_ACTIVE_PX = round(s.FS * s.H_ACTIVE / 1e6)
        s.ACTIVE_W_PX_OFFS = round((s.H_ACTIVE_PX - s.ACTIVE_W_PX) / 2)

# global raster timing
r = RasterTimings

def configure_filters():
    global r
    # NTSC definitions
    # from SMPTE 170M-2004
    # and Video Demystified, 4th ed.
    r = RasterTimings(
        F_SC = 5e6*63/88,
        F_H = 2/455 * 5e6*63/88,
        F_V = 2/525 * 2/455 * 5e6*63/88,
        FS = 13.5e6,

        H_SYNC_LENGTH = 4.7,
        H_SYNC_ENV = 140*4,
        H_BLANK_ENV = 140*4,
        H_SYNC_TO_ACTIVE = 9.2,
        H_ACTIVE_TO_SYNC = 1.5,

        # off-by-one, we count starting with 0
        # standards start with 1
        V_PREQ = 3,
        V_SERR = 6,
        V_POEQ = 9,
        V_BLANK_INT = 22,

        INTERLACED = True,

        CB_BURST_START = 19,
        CB_BURST_WIDTH = 9,
        CB_BURST_ENV = 300,

        SIG_WHITE = 100,
        SIG_BLACK = 7.5,
        SIG_BLANK = 0,
        SIG_CBAMP = 40,
        SIG_SYNC = -40,

        ACTIVE_W_PX = 720,
        ACTIVE_H_PX = 480,
    )
    print(r)

def encode_field(raster_buf, nd_img, field: int, phase_acc: int):
    global r
    v = 0 # v counter; field only
    while v < r.HALF_H_PX:
        # input index
        y = (v - r.V_BLANK_INT) * 2 - (field & 1)
        # buffer index
        b = v + (field & 1) * (r.HALF_H_PX - 1)
        # line buffer in IRE units
        line_buf = np.full(r.FULL_W_PX, r.SIG_BLANK, dtype=float)

        if v < r.V_PREQ:
            raster_buf[b], phase_acc = encode_eq_pulse(line_buf, phase_acc)
            raster_buf[b], phase_acc = encode_eq_pulse(line_buf, phase_acc)
        elif v < r.V_SERR:
            raster_buf[b], phase_acc = encode_vserr(line_buf, phase_acc)
            raster_buf[b], phase_acc = encode_vserr(line_buf, phase_acc)
        elif v < r.V_POEQ:
            raster_buf[b], phase_acc = encode_eq_pulse(line_buf, phase_acc)
            raster_buf[b], phase_acc = encode_eq_pulse(line_buf, phase_acc)
        elif v < r.V_BLANK_INT:
            raster_buf[b], phase_acc = encode_line(line_buf, phase_acc, None)
        elif v < r.V_BLANK_INT + r.ACTIVE_H_PX/2:
            # in interlaced, this first line is split on even fields
            raster_buf[b], phase_acc = encode_line(line_buf, phase_acc, nd_img[y])
        else:
            raster_buf[b], phase_acc = encode_line(line_buf, phase_acc, None)
            if field % 2 == 0:
                phase_acc -= r.HALF_W_PX
        # print(v, b, y, phase_acc)
        # plot_ndarray(raster_buf[b])
        v += 1
    field = (field+1)%4
    return raster_buf, phase_acc, field

def encode_eq_pulse(line_buf, phase_acc: int):
    global r
    # line buffer in IRE units
    h = 0 # h counter
    while h < r.HALF_W_PX:
        if h < r.H_SYNC_LENGTH_PX/2:
            line_buf[(h+phase_acc) % r.FULL_W_PX] = r.SIG_SYNC
        h += 1

    phase_acc = (phase_acc + h) % (r.FULL_W_PX*2)
    return line_buf, phase_acc

def encode_vserr(line_buf, phase_acc: int):
    global r
    # line buffer in IRE units
    h = 0 # h counter
    while h < r.HALF_W_PX:
        if h < r.HALF_W_PX - r.H_SYNC_LENGTH_PX:
            line_buf[(h+phase_acc) % r.FULL_W_PX] = r.SIG_SYNC
        h += 1

    phase_acc = (phase_acc + h) % (r.FULL_W_PX*2)
    return line_buf, phase_acc

def encode_line(line_buf, phase_acc: int, input_buf=None):
    global r
    h = 0 # h counter

    h_env_half = round(r.H_SYNC_ENV * 1e-9 * r.FS / 2)
    h_env = h_env_half * 2

    cv_env_half = round(r.CB_BURST_ENV * 1e-9 * r.FS / 2)
    cv_env = cv_env_half * 2
    cb_start = round(r.CB_BURST_START / r.F_SC * r.FS)
    cb_width = round((r.CB_BURST_WIDTH / r.F_SC)* r.FS)
    cb_end = cb_start + cb_width + (cv_env * 2)

    active_start = round(r.H_SYNC_TO_ACTIVE * 1e-6 * r.FS) + r.ACTIVE_W_PX_OFFS
    active_end = active_start + r.ACTIVE_W_PX

    while h < r.FULL_W_PX:
        s, c = generate_carrier(phase_acc + h)
        cb_sample = -s
        # hsync
        if h < r.H_SYNC_LENGTH_PX:
            # starting second half
            lvl = r.SIG_SYNC
            line_buf[h] = lvl
        # colorburst
        elif h >= cb_start and h < cb_end:
            # colorburst envelope
            # raised cosine shaping
            if (h < cb_start + cv_env):
                t = (h - cb_start - cv_env)
                cb_sample *= (np.cos(np.pi*t/cv_env)+1)/2
            elif (h >= cb_end - cv_env):
                t = (cb_end - cv_env) - h
                cb_sample *= (np.cos(np.pi*t/cv_env)+1)/2
            line_buf[h] += cb_sample * r.SIG_CBAMP / 2
        # active
        elif h >= active_start and h < active_end and input_buf is not None:
            # grab line
            t = h - active_start
            line_buf[h] = encode_active(s, c, input_buf[t])
        
        # line_buf[h] += s * -140
        h += 1
    phase_acc = (phase_acc + h) % (r.FULL_W_PX*2)
    return line_buf, phase_acc

# during active video, convert YUV input to composite
def encode_active(s: int, c: int, input):
    global r
    encoded = (
        0.925 * input[0] +
        r.SIG_BLACK +
        0.925 * input[1] * s +
        0.925 * input[2] * c)
    return encoded

def generate_carrier(sample: int):
    global r
    s = np.sin(2*np.pi*r.F_SC*(sample/r.FS))
    c = np.cos(2*np.pi*r.F_SC*(sample/r.FS))
    return s, c

def main(argv=None):
    args = parse_argv(argv or sys.argv)

    # load image
    image = Image.open(args.input_image)
    image_filename = args.input_image.parent.joinpath(args.input_image.stem)
    print(image_filename)
    global r
    configure_filters()
    # squeeze image into target resolution
    v_res = Image.Resampling.NEAREST
    h_res = Image.Resampling.NEAREST
    image = image.resize((image.size[0], r.ACTIVE_H_PX), resample=v_res)
    image = image.resize((r.ACTIVE_W_PX, r.ACTIVE_H_PX), resample=h_res)

    # convert to float
    nd_img = np.array(image.convert("RGB"))
    nd_img = nd_img / np.float64(255)
    nd_img = np.einsum('ij,klj->kli',RGB_to_YUV,nd_img, dtype=np.float64)
    # convert to IRE
    nd_img *= 100
    print(nd_img.shape)

    raster_buf = np.zeros((2, r.FULL_H_PX, r.FULL_W_PX), dtype=float)
    phase_acc = 0
    field = 0
    raster_buf[0], phase_acc, field = encode_field(raster_buf[0], nd_img, field, phase_acc)
    raster_buf[0], phase_acc, field = encode_field(raster_buf[0], nd_img, field, phase_acc)
    raster_buf[1], phase_acc, field = encode_field(raster_buf[1], nd_img, field, phase_acc)
    raster_buf[1], phase_acc, field = encode_field(raster_buf[1], nd_img, field, phase_acc)

    
    plot_2darr(raster_buf[1])
    raster_buf = raster_buf.flatten()
    raster_buf = np.tile(raster_buf, 2)
    raster_buf -= r.SIG_SYNC
    raster_buf /= 180
    raster_buf = (raster_buf * (2**16 - 1))
    raster_buf -= 2**15
    raster_buf = np.int16(raster_buf)
    import soundfile as sf
    
    sf.write(
        str(image_filename)+"_13-5msps_16_bit.flac", raster_buf, subtype="PCM_16", samplerate=96000)

def plot_2darr(arr):
    plt.figure(tight_layout=True, figsize=(10,5.625))
    plt.imshow(arr)
    plt.savefig("docs/example.png", dpi=300)
    plt.show()
    plt.close()

def plot_signal(arr):
    # upsample to 10x
    from scipy.interpolate import make_interp_spline
    x_size = arr.shape[0]
    x = np.linspace(0, x_size, num=x_size)
    x_new = np.linspace(0, x_size, num=round(x_size*2*r.FS/r.F_SC))
    plt.plot(x, arr, ".", color="gray")
    plt.plot(x_new, make_interp_spline(x, arr, bc_type="clamped")(x_new))
    plt.show()
    plt.close()

if __name__=='__main__':
    main(sys.argv)
