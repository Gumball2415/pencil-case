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
        "-fl",
        "--fsc_limit",
        action="store_true",
        help="Limit frequency content to colorburst.")
    parser.add_argument(
        "-n",
        "--noise",
        type=float,
        default=0.0,
        help="Sigma of gaussian noise to add to the signal, in units of IRE. Default == 0.0")
    parser.add_argument(
        "-as",
        "--active_scale",
        type=int,
        default=None,
        help="Compensated active video width, in samples. Default = None.")
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
    CB_ANGLE: float     # subcarrier color angle, in radians
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

        s.CB_U = np.cos(s.CB_ANGLE)
        s.CB_V = np.sin(s.CB_ANGLE)


# global raster timing
r = RasterTimings

def configure_filters():
    global r
    # NTSC definitions
    # from SMPTE 170M-2004
    # and Video Demystified, 4th ed.
    r = RasterTimings(
        F_SC = 315e6/88,
        F_H = 2/455 * 315e6/88,
        F_V = 2/525 * 2/455 * 315e6/88,
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

        CB_ANGLE = -np.pi,
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

def inc_phase_acc(phase_acc: int, increment: int):
    phase_acc = (phase_acc+increment) % (r.FULL_W_PX*2)
    return phase_acc

def inc_line(b: int):
    b = (b + 1) % r.FULL_H_PX
    return b

def generate_t_pulse(width: int):
    """
    Generates a raised cosine impulse
    
    :param width: width of pulse, in samples
    :type width: int
    """
    t = np.linspace(-width/2, width/2, width, endpoint=True)
    pulse = (np.cos(np.pi*t/width)+1)/2
    pulse /= (np.sum(pulse))
    return pulse

def generate_sync(
        raster_buf: np.ndarray,
        phase_acc: int,
        b: int):
    global r
    transition = generate_t_pulse(round(r.H_SYNC_ENV*1e-9*r.FS))

    for field in range(4):
        v = 0
        while v <= r.HALF_H_PX:
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
                else:
                    raster_buf[b] = encode_hsync(raster_buf[b])
                    phase_acc = inc_phase_acc(phase_acc, r.FULL_W_PX)
                if v == r.HALF_H_PX:
                    # in interlaced, this last line is split on even fields
                    phase_acc = inc_phase_acc(phase_acc, -r.HALF_W_PX)
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
                    phase_acc = inc_phase_acc(phase_acc, r.HALF_W_PX)
                    v += 1
                    continue
                else:
                    raster_buf[b] = encode_hsync(raster_buf[b])
                    phase_acc = inc_phase_acc(phase_acc, r.FULL_W_PX)

            b = inc_line(b)
            v += 1

    for line in range(raster_buf.shape[0]):
        raster_buf[line] = np.convolve(raster_buf[line], transition, mode="same")

    return raster_buf

def gate_cb():
    """
    Returns a scanline of factor values to gate the colorburst, where the final
    colorburst signal can be calculated as `gate_cb() * carrier_arr`.
    """
    global r
    env_px = r.CB_BURST_ENV*1e-9*r.FS
    env_risetime = r.CB_BURST_ENV*1e-9*r.FS * 0.964

    transition = generate_t_pulse(round(env_px))
    gate = np.zeros(r.FULL_W_PX, dtype=np.float64)

    cb_start = round(r.CB_BURST_START / r.F_SC * r.FS - env_risetime)
    cb_width = round((r.CB_BURST_WIDTH-1) / r.F_SC * r.FS + env_risetime*2)
    cb_end = cb_start + cb_width

    gate[cb_start:cb_end] = 1
    gate = np.convolve(gate, transition, mode="same")
    return gate

def gate_active():
    """
    Returns a scanline of factor values to gate the active video, where the 
    final signal can be calculated as `gate_active() * signal_arr`.
    """
    global r
    transition = generate_t_pulse(round(r.H_BLANK_ENV*1e-9*r.FS))
    gate = np.zeros(r.FULL_W_PX, dtype=np.float64)
    gate[r.ACTIVE_W_START:r.ACTIVE_W_END] = 1
    gate = np.convolve(gate, transition, mode="same")
    return gate

def encode_field(
        raster_buf: np.ndarray,
        nd_img: np.ndarray,
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

    :type field: int
    :param field:
        Field type. range is 0-7.

    :type phase_acc: int
    :param phase_acc:
        Phase accumulator. Used for colorburst phase tracking, and vsync 
        alignment tracking.

    :type b: int
    :param b:
        Raster buffer line index. Used for interlaced indexing..

    :type flip_field: bool
    :param flip_field:
        Switches odd/even field to be encoded. default: bottom field first

    :type debug: bool
    :param debug:
        Print debug information.

    :type debug: str
    :param disable:
        Disable "chroma" by setting saturation to 0, or "luma" by setting luma 
        to 50 IRE.
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

        # vsync, field 1/3
        if field & 1 == 0:
            if v < r.V_POEQ:
                phase_acc = inc_phase_acc(phase_acc, r.FULL_W_PX)
            elif v >= active_start and v < active_end:
                raster_buf[b], phase_acc = encode_line(raster_buf[b], phase_acc, nd_img[y], disable)
            else:
                raster_buf[b], phase_acc = encode_line(raster_buf[b], phase_acc, None, disable)
            if v == r.HALF_H_PX:
                # in interlaced, this last line is split on even fields
                phase_acc = inc_phase_acc(phase_acc, -r.HALF_W_PX)
                break
        else:
            if v < r.V_POEQ:
                phase_acc = inc_phase_acc(phase_acc, r.FULL_W_PX)
            elif v == r.V_POEQ:
                b = inc_line(b)
                # restore last line split from previous field
                phase_acc = inc_phase_acc(phase_acc, r.HALF_W_PX)
                v += 1
                continue
            elif v >= active_start and v < active_end:
                raster_buf[b], phase_acc = encode_line(raster_buf[b], phase_acc, nd_img[y], disable)
            else:
                raster_buf[b], phase_acc = encode_line(raster_buf[b], phase_acc, None, disable)

        # if debug:
        #     plot_signal(raster_buf[b])
        b = inc_line(b)
        v += 1

    field = (field+1)%8
    return raster_buf, phase_acc, field, b

def encode_hsync(line_buf: np.ndarray):
    global r
    line_buf[:r.H_SYNC_LENGTH_PX] = r.SIG_SYNC
    return line_buf

def encode_eq_pulse(line_buf: np.ndarray, phase_acc: int):
    global r

    start = phase_acc % r.FULL_W_PX
    end = start + r.H_SYNC_LENGTH_PX//2
    line_buf[start:end] = r.SIG_SYNC

    phase_acc = inc_phase_acc(phase_acc, r.HALF_W_PX)
    return line_buf, phase_acc

def encode_vserr(line_buf: np.ndarray, phase_acc: int):
    global r
    # line buffer in IRE units
    start = phase_acc % r.FULL_W_PX
    end = start + r.HALF_W_PX - r.H_SYNC_LENGTH_PX
    line_buf[start:end] = r.SIG_SYNC

    phase_acc = inc_phase_acc(phase_acc, r.HALF_W_PX)
    return line_buf, phase_acc

def encode_line(
        line_buf: np.ndarray,
        phase_acc: int,
        input_buf: np.ndarray=None,
        disable: str = None):
    """
    Encodes a non-vsync scanline.
    """
    global r

    gate_cb_env = gate_cb()

    active_env = round(r.H_BLANK_ENV*1e-9*r.FS)
    gate_active_env = gate_active()

    # TODO: do this per-scanline as a ndarray operation
    t = np.arange(r.FULL_W_PX)
    s, c = generate_carrier(phase_acc, t)
    subcarrier = s*r.CB_U + c*r.CB_V
    
    # colorburst
    if disable != "chroma":
        line_buf += subcarrier * r.SIG_CBAMP / 2 * gate_cb_env
    # active
    start=r.ACTIVE_W_START - active_env
    end=r.ACTIVE_W_END + active_env
    active_start = round((r.ACTIVE_W_PX-(end-start)) / 2)
    active_end = active_start + end-start
    active = np.zeros((end-start), dtype=np.float64)
    if input_buf is not None:
        active = encode_active(
            s[start:end], c[start:end],
            input_buf[active_start:active_end], disable)
    line_buf[start:end] += active * gate_active_env[start:end]
    phase_acc = inc_phase_acc(phase_acc, r.FULL_W_PX)
    return line_buf, phase_acc

def encode_active(
        s: np.ndarray,
        c: np.ndarray,
        input: np.ndarray,
        disable: str = None):
    """
    during active video, convert YUV input to composite
    """
    global r
    scale_factor = (r.SIG_WHITE - r.SIG_BLACK)/r.SIG_WHITE
    if disable == "luma":
        encoded = (
            scale_factor * 50 +
            r.SIG_BLACK +
            scale_factor * input[..., 1] * s +
            scale_factor * input[..., 2] * c)
    elif disable == "chroma":
        encoded = (
            scale_factor * input[..., 0] +
            r.SIG_BLACK)
    else:
        encoded = (
            scale_factor * input[..., 0] +
            r.SIG_BLACK +
            scale_factor * input[..., 1] * s +
            scale_factor * input[..., 2] * c)
    return encoded

def generate_carrier(offset: int, time: np.ndarray):
    """
    generates a color subcarrier reference signal
    
    :param sample: time point, in samples
    :type sample: int
    """
    global r
    time_const = 2*np.pi*r.F_SC*(((time + offset) + 0.5)/r.FS)
    return np.sin(time_const), np.cos(time_const)

# B-Y and R-Y reduction factors
BY_rf = 0.492111
RY_rf = 0.877283

# derived from the NTSC base matrix of luminance and color-difference
RGB_to_YUV = np.array([
    [ 0.299,        0.587,        0.114],
    [-0.299*BY_rf, -0.587*BY_rf,  0.886*BY_rf],
    [ 0.701*RY_rf, -0.587*RY_rf, -0.114*RY_rf]
], np.float64)

def encode_image(
        image: Image.Image,
        active_scale: int,
        prefilter_disable: bool,
        notch_luma: bool,
        fsc_limit: bool,
        flip_field: bool,
        debug: bool,
        disable: str,
        field:int,
        phase_acc:int,
        b:int):
    """
    Encodes an image into a single frame (two fields.) No sync information is present.
    
    :param image: Pillow Image to be encoded.
    :param args: argparse arguments. see -h.

    :type active_scale: int
    :param active_scale:
        If specified, the final decoded active image width. May be `None`.

    :type prefilter_disable: bool
    :param prefilter_disable:
        Disables chroma lowpassing.

    :type notch_luma: bool
    :param notch_luma:
        Enables subcarrier notch filtering on luma.

    :type fsc_limit: bool
    :param fsc_limit:
        Limits frequency content to colorburst.

    :type flip_field: bool
    :param flip_field:
        Swaps the top and bottom fields of the image.

    :type debug: bool
    :param debug:
        Print/plot debugging information.

    :type disable: str
    :param disable:
        Disable chroma or luma.

    :type field: int
    :param field: field counter.

    :type phase_acc: int
    :param phase_acc: phase accumulator counter.

    :type b: int
    :param b: Raster buffer line index.
    """
    active_width = round(r.ACTIVE_W_PX**2/active_scale) \
        if active_scale is not None else r.ACTIVE_W_PX

    image = image.resize(
        (active_width, r.ACTIVE_H_PX),
        resample=Image.Resampling.LANCZOS,
        reducing_gap=3)
    if active_scale is not None:
        padding = Image.new(image.mode, (r.ACTIVE_W_PX, r.ACTIVE_H_PX))
        padding.paste(image, (round((r.ACTIVE_W_PX-active_width)/2), 0))
        image = padding 

    # convert to float
    nd_img = np.array(image.convert("RGB"))
    nd_img = nd_img / np.float64(255)
    nd_img = np.einsum('ij,klj->kli',RGB_to_YUV,nd_img, dtype=np.float64)

    # prefilter, if permitted
    if not prefilter_disable:
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
    
    if notch_luma:
        # simple notch
        bw = 1.3e6
        b_luma, a_luma = signal.iirnotch(r.F_SC, r.F_SC/bw, fs=r.FS)
        nd_img[..., 0] = signal.filtfilt(b_luma, a_luma, nd_img[..., 0])

    # resample luma to make sure there is no content beyond 2xfsc
    # NTSC
    if fsc_limit:
        factor=2
        down = round(factor*r.F_SC/(r.FS/r.FULL_W_PX))
        gauss_wnd = 100
        nd_img = resample_poly(
            resample_poly(nd_img,down ,r.FULL_W_PX, axis=1, window=("gaussian", gauss_wnd)),
            r.FULL_W_PX, down, axis=1, window=("gaussian", gauss_wnd))

    # convert to IRE
    nd_img *= 100
    raster_buf = np.zeros((1, r.FULL_H_PX, r.FULL_W_PX), dtype=np.float64)

    # insert active video inside sync
    raster_buf[0], phase_acc, field, b = encode_field(
        raster_buf[0], nd_img, field, phase_acc, b,
        flip_field, debug, disable)
    raster_buf[0], phase_acc, field, b = encode_field(
        raster_buf[0], nd_img, field, phase_acc, b,
        flip_field, debug, disable)
    
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
    frame = 0

    # TODO: per-field encoding
    # encode first frame
    raster_buf, frame, phase_acc, b = encode_image(
        image,
        args.active_scale,
        args.prefilter_disable,
        args.notch_luma,
        args.fsc_limit,
        args.flip_field,
        args.debug,
        args.disable,
        frame, phase_acc, b)

    if args.frames-1 > 0:
        for _ in range(args.frames-1):
            buffer, frame, phase_acc, b = encode_image(
                image,
                args.active_scale,
                args.prefilter_disable,
                args.notch_luma,
                args.fsc_limit,
                args.flip_field,
                args.debug,
                args.disable,
                frame, phase_acc, b)
            raster_buf = np.concatenate((raster_buf, buffer), dtype=np.float64)

    # pregenerate sync for full frame
    sync_buf = np.zeros((r.FULL_H_PX, r.FULL_W_PX), dtype=np.float64)
    sync_buf = generate_sync(sync_buf, phase_acc, b)
    for frame in range(raster_buf.shape[0]):
        raster_buf[frame] += sync_buf

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

    # TODO: use pyflac and encode signal in realtime
    import soundfile as sf
    srate_name = str(round(r.FS/1e6, 4))
    
    sf.write(f"{image_filename}_{srate_name}MHz_32_bit.cvbs", raster_buf, subtype="PCM_24", samplerate=96000, format="FLAC")

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
    x = np.linspace(0, x_s, num=x_s, endpoint=False)
    x_new = np.linspace(0, x_s, num=x_s2, endpoint=False)
    plt.plot(x, arr, ".", color="gray")
    plt.plot(x_new, resample_poly(arr, ups, dws))
    plt.show()
    plt.close()

if __name__=='__main__':
    main(sys.argv)
