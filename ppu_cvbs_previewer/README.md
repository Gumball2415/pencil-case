# PPU Composite Previewer

Previews an NES screenshot.

## About

it was difficult screenshotting Mesen all the time so i made this instead.

## Usage

```cmd
usage: ppu_cvbs_preview.py [-h] [-d] [-cxp COLOR_CLOCK_PHASE] [-raw]
                           [-pal PALETTE] [-ppu {2C02,2C07}] [-box]
                           [-phd PHASE_DISTORTION] [-comb] [-full]
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

version 0.1.0
```

## Examples

### 3-phase dot crawl

[3-phase dot crawl](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/addie.mp4)

### Solstice

[Solstice](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/solstice.mp4)

### Comb + Box filter

[Comb + box filter](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/rockman2.mp4)

### PLUGE test

[PLUGE](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee.mp4)

### Comb filter tests for PAL and NTSC

[Comb filter test for PAL](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_2_PAL.png)

[Comb filter test for NTSC](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_2_NTSC.mp4)

### Sharpness tests for comb and box filter

[Sharpness test: Comb filter](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_3_comb.mp4)

[Sharpness test: Box filter](https://raw.githubusercontent.com/Gumball2415/pencil-case/refs/heads/main/ppu_cvbs_previewer/docs/240pee_3_box.mp4)

## [License](../LICENSE_MIT-0.txt)

This implementation is licensed under MIT-0.
Copyright 2024 Persune.
