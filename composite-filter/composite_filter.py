# composite filter
# Copyright (C) 2023 Persune
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

parser=argparse.ArgumentParser(
    description="yet another composite filter",
    epilog="version 0.1.0")
parser.add_argument("input_image", type=str, help="input image")
parser.add_argument("-dbg", "--debug", action="store_true", help="enable debug plot")
parser.add_argument("-skp", "--skip-plot", action="store_true", help="skip plot")
parser.add_argument("-nnv", "--neigbor-vertical", action="store_true", help="use nearest-neighbor scaling for vertical resolution")
parser.add_argument("-pfd", "--prefilter-enable", action="store_true", help="enable luma lowpassing before encoding to composite")
parser.add_argument("-prg", "--progressive", action="store_true", help="scale to 240 lines instead of 480")
parser.add_argument("-isf", "--interlace-separate-frames", action="store_true", help="output one file per field")
parser.add_argument("-irs", "--input-resolution-samplerate", action="store_true", help="use the image input's horizontal resolution as samplerate. minimum is 720 pixels")
parser.add_argument("-war", "--wide-aspect-ratio", action="store_true", help="use wide aspect ratio for output")
parser.add_argument("-pic", "--phase-invert-colorburst", action="store_true", help="invert phase of colorburst")
parser.add_argument("-rsm", "--resolution-multiplier", type=int, help="multiply the samplerate by x. makes the image sharper at the cost of filtering time", default=2)
parser.add_argument("-com", "--comb-filter", action="store_true", help="filter using 1d comb filter")

args = parser.parse_args()

fig = plt.figure(tight_layout=True, figsize=(10,5.625))

# load image
image = Image.open(args.input_image)
image_filename = os.path.splitext(image.filename)[0]

# B-Y and R-Y reduction factors
BY_rf = 0.492111
RY_rf = 0.877283

# derived from the NTSC base matrix of luminance and color-difference
RGB_to_YUV = np.array([
    [ 0.299,        0.587,        0.114],
    [-0.299*BY_rf, -0.587*BY_rf,  0.886*BY_rf],
    [ 0.701*RY_rf, -0.587*RY_rf, -0.114*RY_rf]
], np.float64)

if args.wide_aspect_ratio:
    raster_h_res = 960 * args.resolution_multiplier
    raster_frontporch_length = 21.5 * args.resolution_multiplier
elif args.input_resolution_samplerate and (image.size[0] > 720):
    raster_h_res = image.size[0] * 2
    raster_frontporch_length = np.around(raster_h_res / 44.6512 * 2) / 2 * args.resolution_multiplier
else:
    raster_h_res = 720 * args.resolution_multiplier
    raster_frontporch_length = 16 * args.resolution_multiplier

raster_v_res = 480
raster_samplerate = raster_h_res/((53 + (1/3)) / 1000000)
raster_subcarrier_freq = 315 / 88 * 1000000
raster_total_scanline_length = (raster_samplerate * 227.5 / (315/88))
raster_blanking_portion_length = raster_total_scanline_length - raster_h_res
raster_href_start = raster_blanking_portion_length - raster_frontporch_length

# convert to numpy array
# this switches the x and y axis, but that's ok
# it'll switch back when we convert back to an image
if args.progressive:
    image = image.resize((image.size[0], int(raster_v_res/2)), resample=Image.Resampling.NEAREST if args.neigbor_vertical else Image.Resampling.LANCZOS)
    image = image.resize((raster_h_res, int(raster_v_res/2)), resample=Image.Resampling.LANCZOS)
    image = image.resize((raster_h_res, raster_v_res), resample=Image.Resampling.NEAREST)
else:
    image = image.resize((image.size[0], raster_v_res), resample=Image.Resampling.NEAREST if args.neigbor_vertical else Image.Resampling.LANCZOS)
    image = image.resize((raster_h_res, raster_v_res), resample=Image.Resampling.LANCZOS)

print("buffer size: {0[0]} x {0[1]}".format(image.size))

print("converting to floating point...")
image_arr = np.array(image.convert("RGB"))
image_arr = image_arr / np.float64(255)
image_arr_copy = image_arr

print("converting RGB to YUV...")
image_arr = np.einsum('ij,klj->kli',RGB_to_YUV,image_arr, dtype=np.float64)

chroma_N, chroma_Wn = signal.buttord(1300000, 3600000, 2, 20, fs=raster_samplerate, analog=False)
b_chroma, a_chroma = signal.butter(chroma_N, chroma_Wn, 'low', fs=raster_samplerate, analog=False)

if args.prefilter_enable:
    print("bandlimiting luma...")
    image_arr[..., 0] = signal.filtfilt(b_chroma, a_chroma, image_arr[..., 0])

print("bandlimiting chroma according to SMPTE 170M-2004...")
image_arr[..., 1] = signal.filtfilt(b_chroma, a_chroma, image_arr[..., 1])
image_arr[..., 2] = signal.filtfilt(b_chroma, a_chroma, image_arr[..., 2])

luma_pedestal = 7.5
active_scanline_buffer = np.empty([raster_v_res, raster_h_res], np.float64)
# add padding for chroma
sample_pad = int(np.round(raster_frontporch_length))
active_scanline_buffer = np.pad(active_scanline_buffer, ((0, 0), (sample_pad, sample_pad)), mode='constant', constant_values=luma_pedestal)

print("encoding to composite signal...")
timepoint = lambda s, o: o + (2*np.pi*raster_subcarrier_freq*(((s + raster_href_start + 0.5)/raster_samplerate))) + (np.pi * int(args.phase_invert_colorburst))
for line in range(active_scanline_buffer.shape[0]):
    for sample in range(raster_h_res):
        offset = (np.pi * int(bool(line % 4 & 0b10)))
        active_scanline_buffer[line, (sample + sample_pad)] = (0.925*image_arr[line, sample, 0] + luma_pedestal
            + (0.925*image_arr[line, sample, 1] * np.sin(timepoint(sample, offset)))
            + (0.925*image_arr[line, sample, 2] * np.cos(timepoint(sample, offset)))
        )

YUV_buffer = np.empty((active_scanline_buffer.shape[0], active_scanline_buffer.shape[1], 3), np.float64)

# set up filters
q_factor = 0.5    # any lower than 1 and it'll become unstable

print("seperating chroma bandwidth...")
if (args.comb_filter):
    # the entire buffer has both fields in it, so skip every other line so we can comb filter the same field's previous line
    chroma_buffer = np.empty((active_scanline_buffer.shape[0], active_scanline_buffer.shape[1]), np.float64)
    for line in range(active_scanline_buffer.shape[-0]):
        if (line < 2):
            chroma_buffer[line, :] = (active_scanline_buffer[line, :] - luma_pedestal) / 2
        else:
            chroma_buffer[line, :] = (active_scanline_buffer[line, :] - active_scanline_buffer[line - 2, :]) / 2
else:
    b_peak, a_peak = signal.iirpeak(raster_subcarrier_freq, q_factor, raster_samplerate)
    chroma_buffer = signal.filtfilt(b_peak, a_peak, active_scanline_buffer)

print("seperating luma bandwidth...")

YUV_buffer[:, :, 0] = active_scanline_buffer - chroma_buffer
if (args.comb_filter):
    b_notch, a_notch = signal.iirnotch(raster_subcarrier_freq, q_factor, raster_samplerate)
    YUV_buffer[:, :, 0] = signal.filtfilt(b_notch, a_notch, YUV_buffer[:, :, 0])


print("normalizing luma and chroma...")
YUV_buffer[:, :, :] -= luma_pedestal
YUV_buffer /= 0.925

print("quadrature amplitude demodulation...")
for line in range(active_scanline_buffer.shape[-0]):
    for sample in range(active_scanline_buffer.shape[-1]):
        offset = (np.pi * int(bool(line % 4 & 0b10)))
        YUV_buffer[line, sample, 1] = (chroma_buffer[line, sample]) * np.sin(timepoint(sample - sample_pad, offset)) * 2
        YUV_buffer[line, sample, 2] = (chroma_buffer[line, sample]) * np.cos(timepoint(sample - sample_pad, offset)) * 2

print("filtering chroma...")
# chroma_N, chroma_Wn = signal.buttord(raster_subcarrier_freq-2000000, raster_subcarrier_freq, 3, 90, fs=raster_samplerate, analog=False)
# b_chroma, a_chroma = signal.butter(chroma_N, chroma_Wn, 'low', fs=raster_samplerate, analog=False)
# YUV_buffer[:, :, 1] = signal.filtfilt(b_chroma, a_chroma, YUV_buffer[:, :, 1] * 2)
# YUV_buffer[:, :, 2] = signal.filtfilt(b_chroma, a_chroma, YUV_buffer[:, :, 2] * 2)
YUV_buffer[..., 1] = signal.filtfilt(b_chroma, a_chroma, YUV_buffer[..., 1])
YUV_buffer[..., 2] = signal.filtfilt(b_chroma, a_chroma, YUV_buffer[..., 2])

print("converting YUV to RGB...")
YUV_buffer = YUV_buffer[:, (sample_pad):(YUV_buffer.shape[1] - sample_pad), :]
image_out = np.einsum('ij,klj->kli', np.linalg.inv(RGB_to_YUV), YUV_buffer, dtype=np.float64)

print("clipping values out of range...")
np.clip(image_out, 0, 1, out=image_out)
imageout = Image.fromarray(np.ubyte(np.around(image_out * 255)))

if args.wide_aspect_ratio:
    print("set to 16:9 aspect ratio...")
    imageout = imageout.resize((854, imageout.size[1]), resample=Image.Resampling.LANCZOS)
else:
    print("set to 4:3 aspect ratio...")
    imageout = imageout.resize((640, imageout.size[1]), resample=Image.Resampling.LANCZOS)

if args.debug:
    gs = gridspec.GridSpec(2, 2)
    ax = fig.add_subplot(gs[0, 0])
    ax_composite = fig.add_subplot(gs[1, 0])
    ax_luma = fig.add_subplot(gs[0, 1])
    ax_chroma = fig.add_subplot(gs[1, 1])
    
    ax.set_title("Filtered image")
    ax.imshow(imageout)
    
    active_scanline_buffer = active_scanline_buffer[:, (sample_pad):(active_scanline_buffer.shape[1] - sample_pad)]
    active_scanline_buffer -= luma_pedestal
    active_scanline_buffer /= 0.925
    ax_composite.set_title("Encoded composite image")
    ax_composite.imshow(active_scanline_buffer, cmap='gray')
    
    ax_luma.set_title("Decoded luma image")
    ax_luma.imshow(YUV_buffer[:, :, 0], cmap='gray')
    
    chroma_plot_buffer = np.dstack((np.full((YUV_buffer.shape[0], YUV_buffer.shape[1]), 0.5), YUV_buffer[...,1], YUV_buffer[...,2]))
    chroma_plot_buffer = np.einsum('ij,klj->kli', np.linalg.inv(RGB_to_YUV), chroma_plot_buffer, dtype=np.float64)
    ax_chroma.set_title("Decoded chroma image")
    ax_chroma.imshow(chroma_plot_buffer, cmap='gray')
    plt.savefig("docs/example.png", dpi=96)
else:
    ax = fig.add_subplot()
    ax.set_title("Filtered image")
    ax.imshow(imageout)
        
if not args.skip_plot:
    plt.show()

print("png export format: {0}".format(imageout.mode))
if args.interlace_separate_frames and not args.progressive:
    # plot two fields via bob deinterlacing
    print("saving fields...")
    imageout_2 = imageout
    
    imageout = imageout.resize((imageout.size[0], int(imageout.size[1]/2)), resample=Image.Resampling.NEAREST)
    imageout = imageout.resize((imageout.size[0], int(imageout.size[1]*2)), resample=Image.Resampling.NEAREST)
    imageout.save("{0}_filt_field_1.png".format(image_filename))
    imageout.close()

    imageout_2 = imageout_2.transform(imageout_2.size, Image.AFFINE, (1, 0, 0, 0, 1, -1))
    imageout_2 = imageout_2.resize((imageout_2.size[0], int(imageout_2.size[1]/2)), Image.Resampling.NEAREST)
    imageout_2 = imageout_2.resize((imageout_2.size[0], int(imageout_2.size[1]*2)), Image.Resampling.NEAREST)
    imageout_2.save("{0}_filt_field_2.png".format(image_filename))
    imageout_2.close()
else:
    print("saving frame...")
    imageout.save("{0}_filt.png".format(image_filename))
    imageout.close()
image.close()
plt.close()
