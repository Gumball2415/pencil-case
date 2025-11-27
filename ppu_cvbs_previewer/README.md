# PPU Composite Previewer

Previews an NES screenshot.

## About

it was difficult screenshotting Mesen all the time so i made this instead.

## Usage

```sh
usage: ppu_cvbs_preview.py [-h] [-d] [--plot_filters] [-cxp COLOR_CLOCK_PHASE]
                           [-raw] [-pal PALETTE] [-ppu {2C02,2C07}]
                           [-phd PHASE_DISTORTION] [-x {chroma,luma}]
                           [-filt {fir,notch,2-line,3-line}]
                           [-ftype {lanczos,lanczos_notch,gauss,box,kaiser,firls,nofilt} [{lanczos,lanczos_notch,gauss,box,kaiser,firls,nofilt} ...]]
                           [-full] [-frames FRAMES] [-avg] [-diff]
                           [-noskipdot]
                           input

NES PPU composite video shader previewer

positional arguments:
  input                 Input 256x240 .png screenshot, index matched by input
                        palette. Output will be saved as
                        "input_ppucvbs_ph_x.png".

options:
  -h, --help            show this help message and exit
  -d, --debug           debug messages
  --plot_filters        Plot chroma / luma filters. Not including the filter
                        response of comb filters.
  -cxp COLOR_CLOCK_PHASE, --color_clock_phase COLOR_CLOCK_PHASE
                        Starting clock phase of first frame. This will affect
                        all proceeding phases. default = 0, range: 0-11
  -raw, --raw_ppu_px    Indicates input is a 256x240 uint16_t array of raw
                        9-bit PPU pixels. Bit format: xxxxxxxEEELLCCCC
  -pal PALETTE, --palette PALETTE
                        Input 192-byte .pal file to index the input .png.
                        Default = "2C02.pal"
  -ppu {2C02,2C07}, --ppu {2C02,2C07}
                        Composite PPU chip used for generating colors. Default
                        = "2C02"
  -phd PHASE_DISTORTION, --phase_distortion PHASE_DISTORTION
                        The amount of voltage-dependent impedance for RC
                        lowpass, where RC = "phd * (level/composite_white) *
                        1e-8". See https://www.nesdev.org/wiki/NTSC_video#Simu
                        lating_differential_phase_distortion for more details.
                        Default = 3
  -x {chroma,luma}, --disable {chroma,luma}
                        Disables chroma by setting UV to 0. Disables luma by
                        setting Y to 0.5.
  -filt {fir,notch,2-line,3-line}, --decoding_filter {fir,notch,2-line,3-line}
                        Method for complementary luma and chroma separation.
                        Default = "notch".
  -ftype {lanczos,lanczos_notch,gauss,box,kaiser,firls,nofilt} [{lanczos,lanczos_notch,gauss,box,kaiser,firls,nofilt} ...], --fir_filter_type {lanczos,lanczos_notch,gauss,box,kaiser,firls,nofilt} [{lanczos,lanczos_notch,gauss,box,kaiser,firls,nofilt} ...]
                        FIR filter for complementary luma-chroma filtering and
                        another FIR filter for lowpassing quadrature
                        demodulated chroma. `decoding_filter` used will be
                        deduced as FIR if this is specified. If one kernel is
                        specified, it will be used for both filters.
  -full, --full_resolution
                        Saves full scanlines instead of a cropped output.
                        Scanlines are scaled to 8x to preserve 1:1 pixel
                        aspect ratio.
  -frames FRAMES        Render x consecutive frames. Usable range: 1-3.
                        default = 1
  -avg, --average       To be used with "-frames" argument. Averages all
                        rendered frames into one. Output will be saved as
                        input_ppucvbs_ph_avg_x.png
  -diff, --difference   To be used with "-frames" argument. Calculates the
                        absolute difference of the first two frames. Output
                        will be saved as input_ppucvbs_ph_avg_x.png
  -noskipdot            Turns off skipped dot rendering, generating two chroma
                        dot phases. Equivalent to rendering on 2C02s

version 0.8.1
```

- the script outputs the next `color_clock_phase` for the succeeding frame. this
  is meant to facilitate external conversion scripts such as encoding a video
  frame-by-frame, where the phase output will feed into the next frame's phase
  input.

## Requirements

- See `requirements.txt` for details.
- i'm a bit too lazy to search up the minimum version requirements so this is what i had

### Libraries

For python:

```python
numpy==2.3.0
pillow==11.2.1
scipy==1.15.3
matplotlib==3.10.3
```

additionally, FFmpeg is required for animated .mp4 previews

## Examples

- animated examples assembled from individual frames using FFmpeg.
- an averaged .png frame is also shown due to GitHub not supporting .mp4 embeds.

### 3-phase dot crawl

![3-phase dot crawl](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/addie.png)

- [3-phase dot crawl animated](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/addie.mp4)

### PLUGE test

![PLUGE](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee.png)

- [PLUGE animated](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee.mp4)

### Comb filter tests for PAL and NTSC

![Comb filter test for PAL](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_2_PAL.png)

- no animation because PAL has static dot crawl

![Comb filter test for NTSC](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_2_NTSC.png)

- [Comb filter test for NTSC animated](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_2_NTSC.mp4)

### Sharpness tests for comb and box filter

![Sharpness test: Comb filter](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_3_comb.png)

- [Sharpness test: Comb filter animated](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_3_comb.mp4)

![Sharpness test: Box filter](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_3_box.png)

- [Sharpness test: Box filter animated](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_3_box.mp4)

### Crosstalk tests

![Crosstalk test: FIR filter (Lanczos + Gaussian)](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/dither_fir_lanczos_gauss.png)

- [Crosstalk test: FIR filter (Lanczos + Gaussian) animated](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/dither.mp4)

![Crosstalk test: 2-line comb filter](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/dither_comb_2line.png)

- [Crosstalk test: 2-line comb filter animated](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/dither_2line.mp4)

![Crosstalk test: 3-line comb filter](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/dither_comb_3line.png)

- [Crosstalk test: 3-line comb filter animated](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/dither_3line.mp4)

## [License](../LICENSE_MIT-0.txt)

This implementation is licensed under MIT-0.\
Copyright 2025 Persune.

Images under fair use and are for filter demonstration purposes only. All rights reserved to their respective copyright owners.
