# PPU Composite Previewer

Previews an NES screenshot.

## About

it was difficult screenshotting Mesen all the time so i made this instead.

## Usage

```cmd
usage: ppu_cvbs_preview.py [-h] [-d] [-cxp COLOR_CLOCK_PHASE] [-raw]
                           [-pal PALETTE] [-ppu {2C02,2C07}] [-box]
                           [-phd PHASE_DISTORTION] [-comb] [-full]
                           [-frames FRAMES] [-avg] [-noskipdot]
                           input

NES PPU composite video shader previewer

positional arguments:
  input                 input 256x240 indexed .png screenshot. will save
                        output as input_ppucvbs_ph_x.png

options:
  -h, --help            show this help message and exit
  -d, --debug           debug messages
  -cxp COLOR_CLOCK_PHASE, --color_clock_phase COLOR_CLOCK_PHASE
                        starting clock phase of first frame. will affect all
                        other phases. range: 0-11
  -raw, --raw_ppu_px    input image is instead a 256x240 uint16_t array of raw
                        9-bit PPU pixels. format: EEELLCCCC
  -pal PALETTE, --palette PALETTE
                        input 192-byte .pal file
  -ppu {2C02,2C07}, --ppu {2C02,2C07}
                        PPU chip used for generating colors. default = 2C02
  -box, --box_filter    use box filter to decode luma and chroma.
  -phd PHASE_DISTORTION, --phase_distortion PHASE_DISTORTION
                        amount of voltage-dependent impedance for RC lowpass,
                        where RC = "amount * (level/composite_white) * 1e-8".
                        this will also desaturate and hue shift the resulting
                        colors nonlinearly. a value of 4 very roughly
                        corresponds to a -5 degree delta per luma row. default
                        = 0.0
  -comb, --delay_line_filter
                        use 1D delay line comb filter decoding instead of
                        single-line decoding
  -full, --full_resolution
                        store the full framebuffer
  -frames FRAMES        render x consecutive frames. range: 1-3. default = 1
  -avg, --average       use with -frames argument. averages all rendered
                        frames into one. will save output as
                        input_ppucbvs_ph_avg_x.png
  -noskipdot            turns off skipped dot rendering. equivalent to
                        rendering on 2C02s

version 0.1.2
```

## Requirements

- See `requirements.txt` for details.
- i'm a bit too lazy to search up the minimum version requirements so this is what i had

### This script requires:

```python
numpy==2.3.0
pillow==11.2.1
scipy==1.15.3
```

## Examples

- animated examples assembled from individual frames using FFmpeg.

### 3-phase dot crawl (averaged and animated)

![3-phase dot crawl averaged](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/addie.png)

<video aria-label="3-phase dot crawl animated" src="https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/addie.mp4" controls="controls" style="max-width: 480px;"></video>

### Solstice

<video aria-label="Solstice" src="https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/solstice.mp4" controls="controls" style="max-width: 480px;"></video>


### SMB

<video aria-label="SMB" src="https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/smb.mp4" controls="controls" style="max-width: 480px;"></video>

### Comb + Box filter

<video aria-label="Comb + box filter" src="https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/rockman2.mp4" controls="controls" style="max-width: 480px;"></video>

### PLUGE test

<video aria-label="PLUGE" src="https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee.mp4" controls="controls" style="max-width: 480px;"></video>

### Comb filter tests for PAL and NTSC

![Comb filter test for PAL](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_2_PAL.png)

<video aria-label="Comb filter test for NTSC" src="https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_2_NTSC.mp4" controls="controls" style="max-width: 480px;"></video>

### Sharpness tests for comb and box filter

<video aria-label="Sharpness test: Comb filter" src="https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_3_comb.mp4" controls="controls" style="max-width: 480px;"></video>

<video aria-label="Sharpness test: Box filter" src="https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_3_box.mp4" controls="controls" style="max-width: 480px;"></video>

## [License](../LICENSE_MIT-0.txt)

This implementation is licensed under MIT-0.
Copyright 2025 Persune.
