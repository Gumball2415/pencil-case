# composite filter
# Copyright (C) 2023 Persune, MIT-0

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import resample_poly, resample

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
        "-f",
        "--flip_field",
        action="store_true",
        help="Switch interlaced fields")
    parser.add_argument(
        "-pd",
        "--prefilter_disable",
        action="store_true",
        help="Disables filtering of chroma before encoding.")
    parser.add_argument(
        "-nl",
        "--notch_luma",
        action="store_true",
        help="Notch filters luma at colorburst to prevent crosstalk in simpler\
            Y/C decoders.")
    parser.add_argument(
        "-n",
        "--noise",
        type=float,
        default=0.0,
        help="Sigma of gaussian noise to add to the signal, in units of IRE. Default == 0.0")
    parser.add_argument(
        "--frames",
        type=int,
        default=2,
        help="Number of whole frames to render (or, x2 fields). Default == 2")
    parser.add_argument(
        "-x",
        "--disable",
        choices=[
            "chroma",
            "luma",
        ], 
        help = "Disables chroma by setting UV to 0. Disables luma by setting Y\
            to 0.5.")
    # parser.add_argument(
    #     "-filt",
    #     "--decoding_filter",
    #     choices=[
    #         "compl",
    #         "2-line",
    #         "3-line",
    #     ], 
    #     default="compl",
    #     help = "Method for luma and chroma decoding.\
    #             Default = \"compl\".")
    # parser.add_argument(
    #     "-ftype",
    #     "--filter_type",
    #     choices=[
    #         "iir",
    #         "lanczos",
    #         "lanczos_notch",
    #         "gauss",
    #         "box",
    #         "kaiser",
    #         "leastsquares",
    #         "none",
    #     ],
    #     nargs="+",
    #     default=["iir"],
    #     help = "One filter for complementary luma-chroma filtering and\
    #         another filter for lowpassing quadrature demodulated chroma.\
    #         If one option is specified, it will be used for both filters.\
    #         Default =\"iir\".")
    # parser.add_argument(
    #     "-full",
    #     "--full_resolution",
    #     action="store_true",
    #     help="Saves full scanlines instead of a cropped output.\
    #         Scanlines are scaled to 8x to preserve 1:1 pixel aspect ratio.")
    return parser.parse_args(argv[1:])

def print_err_quit(message: str):
    print("Error: "+message, file=sys.stderr)
    sys.exit()

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
    CB_ANGLE: float     # subcarrier color angle, in degrees
    CB_U: float = field(init=False) # subcarrier color angle, in UV
    CB_V: float = field(init=False)

    # horizontal timings
    H_SYNC_LENGTH: float    # Horizontal sync length, in microseconds
    H_SYNC_ENV: float       # sync rise/fall time, in nanoseconds
    H_BLANK_ENV: float      # blank rise/fall time, in nanoseconds
    H_SYNC_TO_ACTIVE: float # h.ref to blanking end, in microseconds
    H_ACTIVE_TO_SYNC: float # "back porch" to h.ref, in microseconds; maybe not?

    # vertical timings, see Figure 7 in SMPTE 170M-2004
    # start of
    V_BLANK_INT: int        # length of blanking interval, in lines
    V_ACTIVE_START: int     # line to start drawing active image
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
    H_ACTIVE_TO_SYNC_PX: int = field(init=False)
    H_SYNC_TO_ACTIVE_PX: int = field(init=False)
    H_SYNC_LENGTH_PX: int = field(init=False)
    # "center" h active video start offset
    # depends on actual active area in samples, and given ACTIVE_W_PX
    ACTIVE_W_START: int = field(init=False)
    ACTIVE_W_END: int = field(init=False)
    # the video may not necessarily fit active width, so calculate an offset
    ACTIVE_W_OFFS: int = field(init=False)

    # signal output
    SIG_EXCUR: float    # max excursion, in IRE
    SIG_WHITE: float    # white level, in IRE
    SIG_BLACK: float    # black (setup) level, in IRE
    SIG_BLANK: float    # blanking level, in IRE
    SIG_CBAMP: float    # colorburst pp amplitude, in IRE
    SIG_SYNC: float     # sync level, in IRE

    def __post_init__(s):
        s.FULL_W_PX = round(s.FS / s.F_H)
        s.FULL_H_PX = round(s.F_H / s.F_V * 2)
        s.HALF_W_PX = round(s.FULL_W_PX / 2)
        s.HALF_H_PX = round(s.FULL_H_PX / 2)
        s.H_ACTIVE_TO_SYNC_PX = round(s.FS * s.H_ACTIVE_TO_SYNC*1e-6)
        s.H_SYNC_TO_ACTIVE_PX = round(s.FS * s.H_SYNC_TO_ACTIVE*1e-6)
        s.H_SYNC_LENGTH_PX = round(s.FS * s.H_SYNC_LENGTH * 1e-6)
        s.ACTIVE_W_START = s.H_SYNC_TO_ACTIVE_PX
        s.ACTIVE_W_END = s.FULL_W_PX - s.H_ACTIVE_TO_SYNC_PX
        s.ACTIVE_W_OFFS = round(((s.ACTIVE_W_END - s.ACTIVE_W_START) - s.ACTIVE_W_PX) / 2)

        s.CB_U = np.cos(s.CB_ANGLE*np.pi/180)
        s.CB_V = np.sin(s.CB_ANGLE*np.pi/180)


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
        H_SYNC_ENV = 140,
        H_BLANK_ENV = 140,
        H_SYNC_TO_ACTIVE = 9.2,
        H_ACTIVE_TO_SYNC = 1.5,

        # off-by-one, we count starting with 0
        # standards start with 1
        V_PREQ = 3,
        V_SERR = 6,
        V_POEQ = 9,
        V_BLANK_INT = 20,
        V_ACTIVE_START = 22,

        INTERLACED = True,

        CB_ANGLE = -180,
        CB_BURST_START = 19,
        CB_BURST_WIDTH = 9,
        CB_BURST_ENV = 300,

        SIG_EXCUR = 140,
        SIG_WHITE = 100,
        SIG_BLACK = 7.5,
        SIG_BLANK = 0,
        SIG_CBAMP = 40,
        SIG_SYNC = -40,

        ACTIVE_W_PX = 720,
        ACTIVE_H_PX = 480,
    )
    print(r)

def mod_phase_acc(phase_acc: int):
    phase_acc = phase_acc % (r.FULL_W_PX*2)
    return phase_acc

def inc_line(b: int):
    b = (b + 1) % r.FULL_H_PX
    return b


def encode_field(
        raster_buf,
        nd_img,
        field: int,
        phase_acc: int,
        b: int,
        flip_field: bool = False,
        debug: bool = False,
        disable: str = None):
    """
    Encodes a field from a given YUV-encoded buffer. Returns an encoded raster buffer, and the phase state of the accumulator and the raster index.

    :param raster_buf:
        nd-array raster buffer to encode images in. Must be able to overwrite 
        lines for vsync generation.
    :param nd_img:
        input image buffer of active width and height. Must be YUV formatted 
        with range -1.0 to 1.0.
    :param field:
        Field type. range is 0-7.
    :type field: int
    :param phase_acc:
        Phase accumulator. Used for colorburst phase tracking, and vsync 
        alignment tracking.
    :type phase_acc: int
    :param b:
        Raster buffer line index. Used for interlaced indexing..
    :type b: int
    :param flip_field:
        Switches odd/even field to be encoded. default: bottom field first
    :type flip_field: bool
    :param debug:
        Print debug information.
    :type debug: bool
    :param disable:
        Disable "chroma" by setting saturation to 0, or "luma" by setting luma 
        to 50 IRE.
    :type debug: str
    """
    global r
    v = 0 # v counter; field only
    # buffer index
    b = b % r.FULL_H_PX
    active_start = r.V_ACTIVE_START
    active_end = active_start + r.ACTIVE_H_PX/2
    while v <= r.HALF_H_PX:
        # input index
        y = (v - active_start) * 2 + int(not (field & 1) ^ flip_field)
        # line buffer in IRE units
        if debug:
            print("f: {}\tv: {}\tb: {}\ty: {}\t ph: {}\t acc: {}".format(field, v, b, y, (phase_acc // r.HALF_W_PX), phase_acc))
        phase_acc = mod_phase_acc(phase_acc)

        # vsync, field 1/3
        if field & 1 == 0:
            if v < r.V_PREQ:
                raster_buf[b], phase_acc = encode_eq_pulse(raster_buf[b], phase_acc)
                raster_buf[b], phase_acc = encode_eq_pulse(raster_buf[b], phase_acc)
            elif v < r.V_SERR:
                raster_buf[b], phase_acc = encode_vserr(raster_buf[b], phase_acc)
                raster_buf[b], phase_acc = encode_vserr(raster_buf[b], phase_acc)
            elif v < r.V_POEQ:
                raster_buf[b], phase_acc = encode_eq_pulse(raster_buf[b], phase_acc)
                raster_buf[b], phase_acc = encode_eq_pulse(raster_buf[b], phase_acc)
            elif v >= active_start and v < active_end:
                raster_buf[b], phase_acc = encode_line(raster_buf[b], phase_acc, nd_img[y])
            else:
                raster_buf[b], phase_acc = encode_line(raster_buf[b], phase_acc, None)
            if v == r.HALF_H_PX:
                # in interlaced, this last line is split on even fields
                phase_acc -= r.HALF_W_PX
                phase_acc = mod_phase_acc(phase_acc)
                break
        # vsync, field 2/4
        else:
            if v < r.V_PREQ:
                raster_buf[b], phase_acc = encode_eq_pulse(raster_buf[b], phase_acc)
                b = inc_line(b)
                v += 1
                raster_buf[b], phase_acc = encode_eq_pulse(raster_buf[b], phase_acc)
                continue
            elif v < r.V_SERR:
                raster_buf[b], phase_acc = encode_vserr(raster_buf[b], phase_acc)
                b = inc_line(b)
                v += 1
                raster_buf[b], phase_acc = encode_vserr(raster_buf[b], phase_acc)
                continue
            elif v < r.V_POEQ:
                raster_buf[b], phase_acc = encode_eq_pulse(raster_buf[b], phase_acc)
                b = inc_line(b)
                v += 1
                raster_buf[b], phase_acc = encode_eq_pulse(raster_buf[b], phase_acc)
                continue
            elif v == r.V_POEQ:
                b = inc_line(b)
                # restore last line split from previous field
                phase_acc += r.HALF_W_PX
                phase_acc = mod_phase_acc(phase_acc)
                v += 1
                continue
            elif v >= active_start and v < active_end:
                raster_buf[b], phase_acc = encode_line(raster_buf[b], phase_acc, nd_img[y])
            else:
                raster_buf[b], phase_acc = encode_line(raster_buf[b], phase_acc, None, disable)

        # if debug:
        #     plot_signal(raster_buf[b])
        b = inc_line(b)
        v += 1
    field = (field+1)%8
    return raster_buf, phase_acc, field, b

def encode_eq_pulse(line_buf, phase_acc: int):
    global r
    # line buffer in IRE units
    h = 0 # h counter
    while h < r.HALF_W_PX:
        if h < r.H_SYNC_LENGTH_PX/2:
            line_buf[(h+phase_acc) % r.FULL_W_PX] = r.SIG_SYNC
        h += 1

    phase_acc += h
    phase_acc = mod_phase_acc(phase_acc)
    return line_buf, phase_acc

def encode_vserr(line_buf, phase_acc: int):
    global r
    # line buffer in IRE units
    h = 0 # h counter
    while h < r.HALF_W_PX:
        if h < r.HALF_W_PX - r.H_SYNC_LENGTH_PX:
            line_buf[(h+phase_acc) % r.FULL_W_PX] = r.SIG_SYNC
        h += 1

    phase_acc += h
    phase_acc = mod_phase_acc(phase_acc)
    return line_buf, phase_acc

def encode_line(line_buf, phase_acc: int, input_buf=None, disable: str = None):
    global r
    h = 0 # h counter

    cv_env = round(r.CB_BURST_ENV * 1e-9 * r.FS)
    cb_start = round(r.CB_BURST_START / r.F_SC * r.FS)
    cb_width = round((r.CB_BURST_WIDTH-1) / r.F_SC * r.FS)
    cb_end = cb_start + cb_width

    while h < r.FULL_W_PX:
        s, c = generate_carrier(phase_acc + h)
        # hack: vhs-decode cvbs decodes color 180 offset!
        # so, encode colorburst offset by 180
        cb_sample = s*r.CB_U + c*r.CB_V
        # hsync
        if h < r.H_SYNC_LENGTH_PX:
            # starting second half
            line_buf[h] = r.SIG_SYNC
        # colorburst
        elif h >= cb_start - cv_env and h < cb_end:
        # elif h >= r.H_SYNC_LENGTH_PX and h < r.ACTIVE_W_START:
            # colorburst envelope
            # raised cosine shaping
            # if (h < cb_start):
            #     t = h - cb_start
            #     cb_sample *= (np.cos(np.pi*t/cv_env)+1)/2
            # elif (h >= cb_end - cv_env):
            #     t = cb_end - cv_env - h
            #     cb_sample *= (np.cos(np.pi*t/cv_env)+1)/2
            line_buf[h] += cb_sample * r.SIG_CBAMP / 2
        # active
        elif h >= r.ACTIVE_W_START and h < r.ACTIVE_W_END and input_buf is not None:
            # grab line
            t = h - r.ACTIVE_W_START - r.ACTIVE_W_OFFS
            if t >= 0 and t < r.ACTIVE_W_PX:
                line_buf[h] = encode_active(s, c, input_buf[t], disable)
            else:
                line_buf[h] = encode_active(
                    s, c, np.array([0, 0, 0], dtype=np.float64), disable)
        h += 1
    phase_acc += h
    phase_acc = mod_phase_acc(phase_acc)
    return line_buf, phase_acc

# 
def encode_active(s: int, c: int, input, disable: str = None):
    """
    during active video, convert YUV input to composite
    """
    global r
    scale_factor = (r.SIG_WHITE - r.SIG_BLACK)/r.SIG_WHITE
    if disable == "luma":
        encoded = (
            scale_factor * 50 +
            r.SIG_BLACK +
            scale_factor * input[1] * s +
            scale_factor * input[2] * c)
    elif disable == "chroma":
        encoded = (
            scale_factor * input[0] +
            r.SIG_BLACK)
    else:
        encoded = (
            scale_factor * input[0] +
            r.SIG_BLACK +
            scale_factor * input[1] * s +
            scale_factor * input[2] * c)
    return encoded

def generate_carrier(sample: int):
    """
    generates a color subcarrier reference signal
    
    :param sample: time point, in samples
    :type sample: int
    """
    global r
    time_const = 2*np.pi*r.F_SC*(sample/r.FS)
    return -np.sin(time_const), -np.cos(time_const)

# B-Y and R-Y reduction factors
BY_rf = 0.492111
RY_rf = 0.877283

# derived from the NTSC base matrix of luminance and color-difference
RGB_to_YUV = np.array([
    [ 0.299,        0.587,        0.114],
    [-0.299*BY_rf, -0.587*BY_rf,  0.886*BY_rf],
    [ 0.701*RY_rf, -0.587*RY_rf, -0.114*RY_rf]
], np.float64)

def encode_image(image, args, field, phase_acc, b):
    """
    Encodes an image into a single frame (two fields.)
    
    :param image: Pillow Image to be encoded.
    :param args: argparse arguments. see -h.

        used arguments:
        - `prefilter_disable`
        - `notch_luma`
        - `flip_field`
        - `debug`
        - `disable`

    :param field: field counter.
    :param phase_acc: phase accumulator counter.
    :param b: Raster buffer line index.
    """
    res = Image.Resampling.LANCZOS
    image = image.resize((image.size[0], r.ACTIVE_H_PX), resample=res)
    image = image.resize((r.ACTIVE_W_PX, r.ACTIVE_H_PX), resample=res)
    # convert to float
    nd_img = np.array(image.convert("RGB"))
    nd_img = nd_img / np.float64(255)
    nd_img = np.einsum('ij,klj->kli',RGB_to_YUV,nd_img, dtype=np.float64)
    print(nd_img.shape)
    # squeeze image into target resolution
    # nd_img = resample(nd_img, r.ACTIVE_H_PX, axis=0)
    # nd_img = resample(nd_img, r.ACTIVE_W_PX, axis=1)
    # TODO: make this optional
    # downsample chroma channels by half, according to 4:2:2 subsampling
    nd_img[..., 1] = resample(
        resample(nd_img[..., 1], r.ACTIVE_W_PX//2, axis=1),
        r.ACTIVE_W_PX, axis=1)
    nd_img[..., 2] = resample(resample(
        nd_img[..., 2], r.ACTIVE_W_PX//2, axis=1),
        r.ACTIVE_W_PX, axis=1)
    print(nd_img.shape)

    # prefilter, if permitted
    if not args.prefilter_disable:
        from scipy import signal
        # bandlimiting chroma according to SMPTE 170M-2004, page 5, section 7.2
        b_chroma, a_chroma = signal.iirdesign(
            1.3e6,
            3.6e6,
            gpass=2,
            gstop=20,
            analog=False,
            ftype="butter",
            fs=r.FS
        )
        nd_img[..., 1] = signal.filtfilt(b_chroma, a_chroma, nd_img[..., 1])
        nd_img[..., 2] = signal.filtfilt(b_chroma, a_chroma, nd_img[..., 2])
    
    if args.notch_luma:
        # simple notch
        bw = 1.3e6
        b_luma, a_luma = signal.iirnotch(r.F_SC, r.F_SC/bw, fs=r.FS)
        nd_img[..., 0] = signal.filtfilt(b_luma, a_luma, nd_img[..., 0])

    # convert to IRE
    nd_img *= 100
    raster_buf = np.empty((1, r.FULL_H_PX, r.FULL_W_PX), dtype=np.float64)
    raster_buf[0], phase_acc, field, b = encode_field(
        raster_buf[0], nd_img, field, phase_acc, b, args.flip_field,
        args.debug, args.disable)
    raster_buf[0], phase_acc, field, b = encode_field(
        raster_buf[0], nd_img, field, phase_acc, b, args.flip_field,
        args.debug, args.disable)
    
    return raster_buf, field, phase_acc, b

def main(argv=None):
    args = parse_argv(argv or sys.argv)

    # load image
    image = Image.open(args.input_image)
    image_filename = args.input_image.parent.joinpath(args.input_image.stem)
    global r
    configure_filters()

    # phase_acc is used for colorburst phase, and also vsync serration phase
    phase_acc = 0
    b = 0
    field = 0

    # encode first frame

    raster_buf, field, phase_acc, b = encode_image(image, args, field, phase_acc, b)

    if args.frames-1 > 0:
        for _ in range(args.frames-1):
            buffer, field, phase_acc, b = encode_image(image, args, field, phase_acc, b)
            raster_buf = np.concatenate((raster_buf, buffer), dtype=np.float64)

    rng = np.random.default_rng()
    raster_buf += rng.standard_normal(size=raster_buf.shape) * args.noise
    if args.debug:
        plot_2darr(raster_buf[0])
    raster_buf -= r.SIG_SYNC
    raster_buf /= r.SIG_EXCUR - r.SIG_SYNC
    raster_buf -= 0.5
    raster_buf = np.clip(raster_buf, -1, 1)
    raster_buf *= (2**31)
    raster_buf = np.int32(raster_buf)
    raster_buf = raster_buf.flatten()
    # repeat signal in case of field dropping
    raster_buf = np.tile(raster_buf, 2)

    import soundfile as sf
    srate_name = str(round(r.FS/1e6, 4))
    
    sf.write(f"{image_filename}_{srate_name}MHz_32_bit.flac", raster_buf, subtype="PCM_24", samplerate=96000, format="FLAC")

def plot_2darr(arr):
    plt.figure(tight_layout=True, figsize=(10,5.625))
    plt.imshow(arr)
    plt.savefig("docs/example.png", dpi=300)
    plt.show()
    plt.close()

def plot_signal(arr):
    # upsample to 10x
    x_s = arr.shape[0]
    dws = 1
    ups = dws*8
    x_s2 = x_s * int(ups/dws)
    x = np.linspace(0, x_s, num=x_s)
    x_new = np.linspace(0, x_s, num=x_s2, endpoint=True)
    plt.plot(x, arr, ".", color="gray")
    plt.plot(x_new, resample_poly(arr, ups, dws))
    plt.show()
    plt.close()

if __name__=='__main__':
    main(sys.argv)
