#!/usr/bin/make -f
# it's recommended to set up a python .venv that installs requirements.txt
# and then run this makefile in that venv

examples_dir := docs
input_dir := input

# taken from NROM template
ifeq (${OS}, Windows_NT)
	EXE_SUFFIX := .exe
	PY := py -3
else
	EXE_SUFFIX :=
	PY := python3
endif

.PHONY: usage.txt examples

all: examples usage.txt

clean:
	${RM} -r ${examples_dir}/*
	${RM} -r ${input_dir}/*_ppucvbs_ph_*.png

# addie.png:			squeak! averaging test
# dither.mp4			FIR crosstalk tests
# dither_2line.png/mp4:	2-line comb filter
# dither_3line.png/mp4:	3-line comb filter
# 240pee.png/mp4:		PLUGE
# 240pee_2_PAL.png:		comb filter test for PAL and NTSC
# 240pee_2_NTSC.png/mp4:	comb filter test for PAL and NTSC
# 240pee_3_comb.png/mp4:	sharpness tests
# 240pee_3_comb.png/mp4:	sharpness tests
# testcard.bin:			?

${examples_dir}:
	mkdir $@ -p
usage.txt:
	${RM} -r $@
	${PY} ppu_cvbs_preview.py -h >> $@

# TODO: deduplicate in a smarter way
examples: ${examples_dir}\
	${examples_dir}/addie.mp4\
	${examples_dir}/addie.png\
	${examples_dir}/240pee.png\
	${examples_dir}/240pee.mp4\
	${examples_dir}/240pee_2_NTSC.mp4\
	${examples_dir}/240pee_2_NTSC.png\
	${examples_dir}/240pee_2_PAL.png\
	${examples_dir}/240pee_3_comb.png\
	${examples_dir}/240pee_3_comb.mp4\
	${examples_dir}/240pee_3_box.png\
	${examples_dir}/240pee_3_box.mp4\
	${examples_dir}/dither.mp4\
	${examples_dir}/dither_fir_lanczos_gauss.png\
	${examples_dir}/dither_2line.mp4\
	${examples_dir}/dither_comb_2line.png\
	${examples_dir}/dither_3line.mp4\
	${examples_dir}/dither_comb_3line.png\

# animated examples
${examples_dir}/%.mp4: ${input_dir}/%_ppucvbs_ph_0.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '${input_dir}/${notdir ${basename $@}}_ppucvbs_ph_*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out]" \
	-color_range 1 -colorspace bt709 -color_trc iec61966-2-1 \
	-movflags faststart \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@


# special cases due to duplicate names
${examples_dir}/240pee_2_NTSC.mp4: ${input_dir}/240pee_2_ppucvbs_ph_0.png
	ffmpeg -framerate 60 -y -pattern_type glob -i \
	'${input_dir}/240pee_2_ppucvbs_ph_*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out]" \
	-color_range 1 -colorspace bt709 -color_trc iec61966-2-1 \
	-movflags faststart \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_3_comb.mp4: ${input_dir}/240pee_3_ppucvbs_ph_0_comb.png
	ffmpeg -framerate 60 -y -pattern_type glob -i \
	'${input_dir}/240pee_3_ppucvbs_ph_*_comb.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out]" \
	-color_range 1 -colorspace bt709 -color_trc iec61966-2-1 \
	-movflags faststart \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_3_box.mp4: ${input_dir}/240pee_3_ppucvbs_ph_0_box.png
	ffmpeg -framerate 60 -y -pattern_type glob -i \
	'${input_dir}/240pee_3_ppucvbs_ph_*_box.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out]" \
	-color_range 1 -colorspace bt709 -color_trc iec61966-2-1 \
	-movflags faststart \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/dither_2line.mp4: ${input_dir}/dither_ppucvbs_ph_0_2line.png
	ffmpeg -framerate 60 -y -pattern_type glob -i \
	'${input_dir}/dither_ppucvbs_ph_*_2line*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out]" \
	-color_range 1 -colorspace bt709 -color_trc iec61966-2-1 \
	-movflags faststart \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/dither_3line.mp4: ${input_dir}/dither_ppucvbs_ph_0_3line.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '${input_dir}/dither_ppucvbs_ph_*_3line*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out]" \
	-color_range 1 -colorspace bt709 -color_trc iec61966-2-1 \
	-movflags faststart \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

# specific rules for specific examples
${examples_dir}/addie.png: ${input_dir}/addie.bin
	${PY} ppu_cvbs_preview.py -raw -filt 2-line -frames 3 -avg -noskipdot $<
	mv "$(basename $<)_ppucvbs_ph_(0, 4, 8).png" $@

${input_dir}/addie_ppucvbs_ph_0.png: ${input_dir}/addie.bin
	${PY} ppu_cvbs_preview.py -raw -filt 2-line -frames 3 -noskipdot $<

${input_dir}/240pee_ppucvbs_ph_0.png: ${input_dir}/240pee.png
	${PY} ppu_cvbs_preview.py -frames 2 $<

${examples_dir}/240pee.png: ${input_dir}/240pee.png
	${PY} ppu_cvbs_preview.py -frames 2 -avg $<
	mv "$(basename $<)_ppucvbs_ph_(0, 4).png" $@

${examples_dir}/240pee_2_PAL.png: ${input_dir}/240pee_2.png
	${PY} ppu_cvbs_preview.py -ppu 2C07 -filt 2-line -b 00 $<
	mv $(basename $<)_ppucvbs_ph_0.png $@

${input_dir}/240pee_2_ppucvbs_ph_0.png: ${input_dir}/240pee_2.png
	${PY} ppu_cvbs_preview.py -filt 2-line -frames 2 -b 00 $<

${examples_dir}/240pee_2_NTSC.png: ${input_dir}/240pee_2.png
	${PY} ppu_cvbs_preview.py -filt 2-line -frames 2 -b 00 -avg $<
	mv "$(basename $<)_ppucvbs_ph_(0, 4).png" $@

${input_dir}/240pee_3_ppucvbs_ph_0_comb.png: ${input_dir}/240pee_3.png
	${PY} ppu_cvbs_preview.py -filt 2-line -frames 2 -b 00 $<
	mv $(basename $<)_ppucvbs_ph_0.png $(basename $<)_ppucvbs_ph_0_comb.png
	mv $(basename $<)_ppucvbs_ph_4.png $(basename $<)_ppucvbs_ph_4_comb.png

${examples_dir}/240pee_3_comb.png: ${input_dir}/240pee_3.png
	${PY} ppu_cvbs_preview.py -filt 2-line -frames 2 -b 00 -avg $<
	mv "$(basename $<)_ppucvbs_ph_(0, 4).png" $@

${input_dir}/240pee_3_ppucvbs_ph_0_box.png: ${input_dir}/240pee_3.png
	${PY} ppu_cvbs_preview.py -filt fir -ftype box box -frames 2 $<
	mv $(basename $<)_ppucvbs_ph_0.png $(basename $<)_ppucvbs_ph_0_box.png
	mv $(basename $<)_ppucvbs_ph_4.png $(basename $<)_ppucvbs_ph_4_box.png

${examples_dir}/240pee_3_box.png: ${input_dir}/240pee_3.png
	${PY} ppu_cvbs_preview.py -filt fir -ftype box box -frames 2 -avg $<
	mv "$(basename $<)_ppucvbs_ph_(0, 4).png" $@

${input_dir}/dither_ppucvbs_ph_0_3line.png: ${input_dir}/dither.png
	${PY} ppu_cvbs_preview.py -filt 3-line -frames 2 $<
	mv $(basename $<)_ppucvbs_ph_0.png $(basename $<)_ppucvbs_ph_0_3line.png
	mv $(basename $<)_ppucvbs_ph_4.png $(basename $<)_ppucvbs_ph_4_3line.png

${examples_dir}/dither_comb_3line.png: ${input_dir}/dither.png
	${PY} ppu_cvbs_preview.py -filt 3-line -frames 2 -avg $<
	mv "$(basename $<)_ppucvbs_ph_(0, 4).png" $@

${input_dir}/dither_ppucvbs_ph_0_2line.png: ${input_dir}/dither.png
	${PY} ppu_cvbs_preview.py -filt 2-line -frames 2 $<
	mv $(basename $<)_ppucvbs_ph_0.png $(basename $<)_ppucvbs_ph_0_2line.png
	mv $(basename $<)_ppucvbs_ph_4.png $(basename $<)_ppucvbs_ph_4_2line.png

${examples_dir}/dither_comb_2line.png: ${input_dir}/dither.png
	${PY} ppu_cvbs_preview.py -filt 2-line -frames 2 -avg $<
	mv "$(basename $<)_ppucvbs_ph_(0, 4).png" $@

${input_dir}/dither_ppucvbs_ph_0.png: ${input_dir}/dither.png
	${PY} ppu_cvbs_preview.py -filt fir -ftype lanczos gauss -frames 2 $<

${examples_dir}/dither_fir_lanczos_gauss.png: ${input_dir}/dither.png
	${PY} ppu_cvbs_preview.py -filt fir -ftype lanczos gauss -frames 2 -avg $<
	mv "$(basename $<)_ppucvbs_ph_(0, 4).png" $@

