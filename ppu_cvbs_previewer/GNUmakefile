#!/usr/bin/make -f
# it's recommended to set up a python .venv that installs requirements.txt
# and then run this makefile in that venv

exmdir := docs
objdir := obj
source := input
vnvdir := .env

# taken from NROM template
ifeq (${OS}, Windows_NT)
	EXE_SUFFIX := .exe
	PY := py -3
else
	EXE_SUFFIX :=
	PY := python3
endif

program := ./ppu_cvbs_preview.py

.PHONY: clean info examples all

.INTERMEDIATE:  ${objdir}

all: info examples

clean:
	${RM} -r ${exmdir}/*
	${RM} -r ${objdir}/*

${exmdir}:
	mkdir $@ -p

${objdir}:
	mkdir $@ -p

info: usage.txt requirements.txt

usage.txt:
	${RM} -r $@
	${program} -h >> $@

requirements.txt:
	${RM} -r $@
	pipreqs ./ --force --ignore ${vnvdir}

# TODO: deduplicate in a smarter way

examples: ${exmdir} ${objdir}\
	${exmdir}/addie.mp4\
	${exmdir}/addie_avg.png\
	${exmdir}/240pee.mp4\
	${exmdir}/240pee_avg.png\
	${exmdir}/240pee_2_NTSC.mp4\
	${exmdir}/240pee_2_NTSC_avg.png\
	${exmdir}/240pee_2_PAL.png\
	${exmdir}/240pee_3_comb.mp4\
	${exmdir}/240pee_3_comb_avg.png\
	${exmdir}/240pee_3_box.mp4\
	${exmdir}/240pee_3_box_avg.png\
	${exmdir}/dither_fir.mp4\
	${exmdir}/dither_fir_avg.png\
	${exmdir}/dither_1line.mp4\
	${exmdir}/dither_1line_avg.png\
	${exmdir}/dither_2line.mp4\
	${exmdir}/dither_2line_avg.png\
	${exmdir}/bisqwit_testcard_ntsc.mp4\
	${exmdir}/bisqwit_testcard_ntsc_avg.png\
	${exmdir}/bisqwit_testcard_pal.png\

# animated examples
${exmdir}/%.mp4: ${objdir}/%_ppucvbs_*.png
	ffmpeg -framerate 60 -y -hwaccel auto -pattern_type glob \
	-i '$<' \
	-t 2 \
	-threads 10 \
	-vf "[in]loop=loop=-1:size=3:start=0[out]" \
	-c:v libx264 \
		-crf 0 \
		-preset veryslow \
		-movflags faststart \
		-pix_fmt yuv420p \
		-color_range 1 \
		-colorspace bt709 \
		-color_primaries bt709 \
		-color_trc iec61966-2-1 \
	$@

# averaged examples
${exmdir}/%_avg.png: ${objdir}/%_avg_ppucvbs_*.png
	mv $< $@

${objdir}/%.png: ${source}/%.png ${objdir}
	cp $< $@
${objdir}/%_avg.png: ${source}/%.png ${objdir}
	cp $< $@

${objdir}/%.bin: ${source}/%.bin ${objdir}
	cp $< $@
${objdir}/%_avg.bin: ${source}/%.bin ${objdir}
	cp $< $@

# specific rules for specific examples



# bisqwit_testcard_pal	full raster renders for PAL and NTSC
${exmdir}/bisqwit_testcard_pal.png: ${objdir}/bisqwit_testcard_pal_ppucvbs_*.png
	mv $< $@
${objdir}/bisqwit_testcard_pal_ppucvbs_*.png: ${objdir}/bisqwit_testcard_pal.bin
	${program} -raw -full -filt 1-line -ppu 2C07 $<
${objdir}/bisqwit_testcard_pal.bin: ${source}/bisqwit_testcard.bin ${objdir}
	cp $< $@


# bisqwit_testcard_ntsc	full raster renders for PAL and NTSC
${objdir}/bisqwit_testcard_ntsc_ppucvbs_*.png &:: \
	${objdir}/bisqwit_testcard_ntsc.bin
	${program} -raw -full -filt 1-line -frames 2 $<
${objdir}/bisqwit_testcard_ntsc.bin: ${source}/bisqwit_testcard.bin ${objdir}
	cp $< $@
#avg
${objdir}/bisqwit_testcard_ntsc_avg_ppucvbs_*.png: \
	${objdir}/bisqwit_testcard_ntsc_avg.bin
	${program} -raw -full -filt 1-line -frames 2 $< -avg
${objdir}/bisqwit_testcard_ntsc_avg.bin: ${source}/bisqwit_testcard.bin ${objdir}
	cp $< $@


# addie			squeak! no skipped dot test
${objdir}/addie_ppucvbs_*.png: ${objdir}/addie.bin
	${program} -raw -filt 1-line -frames 3 -noskipdot $<
#avg
${objdir}/addie_avg_ppucvbs_*.png: ${objdir}/addie_avg.bin
	${program} -raw -filt 1-line -frames 3 -noskipdot $< -avg

# 240pee		PLUGE
${objdir}/240pee_ppucvbs_*.png: ${objdir}/240pee.png
	${program} -frames 2 $<
#avg
${objdir}/240pee_avg_ppucvbs_*.png: ${objdir}/240pee_avg.png
	${program} -frames 2 $< -avg


# 240pee_2_PAL	comb filter test for PAL and NTSC
${exmdir}/240pee_2_PAL.png: ${objdir}/240pee_2_PAL_ppucvbs_*.png
	mv $< $@
${objdir}/240pee_2_PAL_ppucvbs_*.png: ${objdir}/240pee_2_PAL.png
	${program} -ppu 2C07 -filt 1-line -b 00 $<
${objdir}/240pee_2_PAL.png: ${source}/240pee_2.png ${objdir}
	cp $< $@



# 240pee_2_NTSC	comb filter test for PAL and NTSC
${objdir}/240pee_2_NTSC_ppucvbs_*.png: ${objdir}/240pee_2_NTSC.png
	${program} -filt 1-line -frames 2 -b 00 $<
${objdir}/240pee_2_NTSC.png: ${source}/240pee_2.png ${objdir}
	cp $< $@
#avg
${objdir}/240pee_2_NTSC_avg_ppucvbs_*.png: ${objdir}/240pee_2_NTSC_avg.png
	${program} -filt 1-line -frames 2 -b 00 $< -avg
${objdir}/240pee_2_NTSC_avg.png: ${source}/240pee_2.png ${objdir}
	cp $< $@



# 240pee_3_comb	sharpness tests
${objdir}/240pee_3_comb_ppucvbs_*.png: ${objdir}/240pee_3_comb.png
	${program} -filt 1-line -frames 2 -b 00 $<
${objdir}/240pee_3_comb.png: ${source}/240pee_3.png ${objdir}
	cp $< $@
#avg
${objdir}/240pee_3_comb_avg_ppucvbs_*.png: ${objdir}/240pee_3_comb_avg.png
	${program} -filt 1-line -frames 2 -b 00 $< -avg
${objdir}/240pee_3_comb_avg.png: ${source}/240pee_3.png ${objdir}
	cp $< $@



# 240pee_3_box	sharpness tests
${objdir}/240pee_3_box_ppucvbs_*.png: ${objdir}/240pee_3_box.png
	${program} -ftype box -frames 2 -b 00 $<
${objdir}/240pee_3_box.png: ${source}/240pee_3.png
	cp $< $@
#avg
${objdir}/240pee_3_box_avg_ppucvbs_*.png: ${objdir}/240pee_3_box_avg.png
	${program} -ftype box -frames 2 -b 00 $< -avg
${objdir}/240pee_3_box_avg.png: ${source}/240pee_3.png ${objdir}
	cp $< $@



# dither_2line	2-line comb filter
${objdir}/dither_2line_ppucvbs_*.png: ${objdir}/dither_2line.png
	${program} -filt 2-line -frames 2 $<
${objdir}/dither_2line.png: ${source}/dither.png ${objdir}
	cp $< $@
#avg
${objdir}/dither_2line_avg_ppucvbs_*.png: ${objdir}/dither_2line_avg.png
	${program} -filt 2-line -frames 2 $< -avg
${objdir}/dither_2line_avg.png: ${source}/dither.png ${objdir}
	cp $< $@



# dither_1line	1-line comb filter
${objdir}/dither_1line_ppucvbs_*.png: ${objdir}/dither_1line.png
	${program} -filt 1-line -frames 2 $<
${objdir}/dither_1line.png: ${source}/dither.png ${objdir}
	cp $< $@
#avg
${objdir}/dither_1line_avg_ppucvbs_*.png: ${objdir}/dither_1line_avg.png
	${program} -filt 1-line -frames 2 $< -avg
${objdir}/dither_1line_avg.png: ${source}/dither.png ${objdir}
	cp $< $@



# dither_fir	FIR crosstalk tests
${objdir}/dither_fir_ppucvbs_*.png: ${objdir}/dither_fir.png
	${program} -filt compl -ftype lanczos gauss -frames 2 $<
${objdir}/dither_fir.png: ${source}/dither.png ${objdir}
	cp $< $@
#avg
${objdir}/dither_fir_avg_ppucvbs_*.png: ${objdir}/dither_fir_avg.png
	${program} -filt compl -ftype lanczos gauss -frames 2 $< -avg
${objdir}/dither_fir_avg.png: ${source}/dither.png ${objdir}
	cp $< $@
