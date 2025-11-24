#!/usr/bin/make -f
# it's recommended to set up a python .venv that installs requirements.txt
# and then run this makefile in that venv

examples_dir := docs

# taken from NROM template
ifeq (${OS}, Windows_NT)
	EXE_SUFFIX := .exe
	PY := py -3
else
	EXE_SUFFIX :=
	PY := python3
endif

animlist := addie 240pee 240pee_2_NTSC 240pee_3_comb 240pee_3_box dither dither_2line dither_3line

animlist := $(foreach o, ${animlist}, ${examples_dir}/${o}.mp4)

.PHONY: usage.txt dither dither_2line dither_3line

all: examples

clean:
	${RM} -r ${examples_dir}
	${RM} -r *_ppucvbs_ph_*.png

# addie.png:			squeak! averaging test
# dither.mp4			crosstalk tests
# dither_2line.mp4:		2-line comb filter
# dither_3line.mp4:		2-line comb filter
# 240pee.mp4:			PLUGE
# 240pee_2_PAL.png:		comb filter test for PAL and NTSC
# 240pee_2_NTSC.mp4:	comb filter test for PAL and NTSC
# 240pee_3_comb.mp4:	sharpness tests
# 240pee_3_comb.mp4:	sharpness tests

${examples_dir}:
	mkdir $@ -p

${examples_dir}/%.mp4: %_ppucvbs_ph_0.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '${notdir ${basename $@}}_ppucvbs_ph_*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=smpte170m" \
	-color_range 1 -colorspace smpte170m  -color_trc smpte170m  -color_primaries smpte170m  -movflags faststart \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_2_NTSC.mp4: 240pee_2_ppucvbs_ph_0.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_2_ppucvbs_ph_*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=smpte170m" \
	-color_range 1 -colorspace smpte170m  -color_trc smpte170m  -color_primaries smpte170m  -movflags faststart \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_3_comb.mp4: 240pee_3_ppucvbs_ph_0_comb.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_3_ppucvbs_ph_*_comb.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=smpte170m" \
	-color_range 1 -colorspace smpte170m  -color_trc smpte170m  -color_primaries smpte170m  -movflags faststart \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_3_box.mp4: 240pee_3_ppucvbs_ph_0_box.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_3_ppucvbs_ph_*_box.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=smpte170m" \
	-color_range 1 -colorspace smpte170m  -color_trc smpte170m  -color_primaries smpte170m  -movflags faststart \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/dither_2line.mp4: dither_ppucvbs_ph_0_2line.png
	ffmpeg -framerate 60 -y -pattern_type glob -i 'dither_ppucvbs_ph_*_2line*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=smpte170m" \
	-color_range 1 -colorspace smpte170m  -color_trc smpte170m  -color_primaries smpte170m  -movflags faststart \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/dither_3line.mp4: dither_ppucvbs_ph_0_3line.png
	ffmpeg -framerate 60 -y -pattern_type glob -i 'dither_ppucvbs_ph_*_3line*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=smpte170m" \
	-color_range 1 -colorspace smpte170m  -color_trc smpte170m  -color_primaries smpte170m  -movflags faststart \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/addie.png: addie.bin
	${PY} ppu_cvbs_preview.py -raw -filt 2-line -frames 3 -avg -noskipdot $<
	mv "addie_ppucvbs_ph_(0, 4, 8).png" $@

addie_ppucvbs_ph_0.png: addie.bin
	${PY} ppu_cvbs_preview.py -raw -filt 2-line -frames 3 -noskipdot $<

240pee_ppucvbs_ph_0.png: 240pee.png
	${PY} ppu_cvbs_preview.py -frames 2 $<

${examples_dir}/240pee_2_PAL.png: 240pee_2.png
	${PY} ppu_cvbs_preview.py -ppu 2C07 -filt 2-line $<
	mv 240pee_2_ppucvbs_ph_0.png $@

240pee_2_ppucvbs_ph_0.png: 240pee_2.png
	${PY} ppu_cvbs_preview.py -filt 2-line -frames 2 $<

240pee_3_ppucvbs_ph_0_comb.png: 240pee_3.png
	${PY} ppu_cvbs_preview.py -filt 2-line -frames 2 $<
	mv $(basename $<)_ppucvbs_ph_0.png 240pee_3_ppucvbs_ph_0_comb.png
	mv $(basename $<)_ppucvbs_ph_4.png 240pee_3_ppucvbs_ph_4_comb.png

240pee_3_ppucvbs_ph_0_box.png: 240pee_3.png
	${PY} ppu_cvbs_preview.py -filt fir -ftype box box -frames 2 $<
	mv $(basename $<)_ppucvbs_ph_0.png 240pee_3_ppucvbs_ph_0_box.png
	mv $(basename $<)_ppucvbs_ph_4.png 240pee_3_ppucvbs_ph_4_box.png

dither_ppucvbs_ph_0_3line.png: dither.png
	${PY} ppu_cvbs_preview.py -filt 3-line -frames 2 $<
	mv $(basename $<)_ppucvbs_ph_0.png dither_ppucvbs_ph_0_3line.png
	mv $(basename $<)_ppucvbs_ph_4.png dither_ppucvbs_ph_4_3line.png

dither_ppucvbs_ph_0_2line.png: dither.png
	${PY} ppu_cvbs_preview.py -filt 2-line -frames 2 $<
	mv $(basename $<)_ppucvbs_ph_0.png dither_ppucvbs_ph_0_2line.png
	mv $(basename $<)_ppucvbs_ph_4.png dither_ppucvbs_ph_4_2line.png

dither_ppucvbs_ph_0.png: dither.png
	${PY} ppu_cvbs_preview.py -filt fir -ftype lanczos gauss -frames 2 $<

usage.txt:
	${RM} -r $@
	${PY} ppu_cvbs_preview.py -h >> $@

examples: ${examples_dir} ${examples_dir}/addie.png ${examples_dir}/240pee_2_PAL.png ${animlist} usage.txt
