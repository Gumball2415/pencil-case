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

all: examples

clean:
	${RM} -r ${examples_dir}
	${RM} -r *_ppucbvs_ph_*.png

.PHONY: usage.txt

# addie.mp4:			squeak! comb filter test
# solstice.mp4:			lidnariq says it looks cool
# rockman2.mp4:			another example; comb + box filter test
# 240pee.mp4:			PLUGE
# 240pee_2_PAL.png:		comb filter test for PAL and NTSC
# 240pee_2_NTSC.mp4:	comb filter test for PAL and NTSC
# 240pee_3_comb.mp4:	sharpness tests
# 240pee_3_comb.mp4:	sharpness tests
examples: ${examples_dir} \
	${examples_dir}/addie.mp4 \
	${examples_dir}/solstice.mp4 \
	${examples_dir}/rockman2.mp4 \
	${examples_dir}/240pee.mp4 \
	${examples_dir}/240pee_2_PAL.png \
	${examples_dir}/240pee_2_NTSC.mp4 \
	${examples_dir}/240pee_3_comb.mp4 \
	${examples_dir}/240pee_3_box.mp4 \
	usage.txt

${examples_dir}:
	mkdir $@ -p

${examples_dir}/addie.mp4:	addie_ppucbvs_ph_0.png \
	addie_ppucbvs_ph_4.png \
	addie_ppucbvs_ph_8.png
	ffmpeg -framerate 60 -y -pattern_type glob -i 'addie_ppucbvs_ph_*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/rockman2.mp4: rockman2_ppucbvs_ph_0.png \
	rockman2_ppucbvs_ph_4.png
	ffmpeg -framerate 60 -y -pattern_type glob -i 'rockman2_ppucbvs_ph_*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/solstice.mp4:	solstice_ppucbvs_ph_0.png \
	solstice_ppucbvs_ph_4.png
	ffmpeg -framerate 60 -y -pattern_type glob -i 'solstice_ppucbvs_ph_*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee.mp4:	240pee_ppucbvs_ph_0.png \
	240pee_ppucbvs_ph_4.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_ppucbvs_ph_*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_2_NTSC.mp4:	240pee_2_ppucbvs_ph_0.png \
	240pee_2_ppucbvs_ph_4.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_2_ppucbvs_ph_*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_3_comb.mp4:	240pee_3_ppucbvs_ph_0_comb.png \
	240pee_3_ppucbvs_ph_4_comb.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_3_ppucbvs_ph_*_comb.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_3_box.mp4:	240pee_3_ppucbvs_ph_0_box.png \
	240pee_3_ppucbvs_ph_4_box.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_3_ppucbvs_ph_*_box.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

addie_ppucbvs_ph_0.png: addie.bin
	${PY} ppu_cvbs_preview.py -raw -cxp 0 -phd 4 -comb $<
addie_ppucbvs_ph_4.png: addie.bin
	${PY} ppu_cvbs_preview.py -raw -cxp 4 -phd 4 -comb $<
addie_ppucbvs_ph_8.png: addie.bin
	${PY} ppu_cvbs_preview.py -raw -cxp 8 -phd 4 -comb $<

solstice_ppucbvs_ph_0.png: solstice.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 0 -phd 4 $<
solstice_ppucbvs_ph_4.png: solstice.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 4 -phd 4 $<

rockman2_ppucbvs_ph_0.png: rockman2.bin
	${PY} ppu_cvbs_preview.py -raw -cxp 0 -phd 4 -comb -box $<
rockman2_ppucbvs_ph_4.png: rockman2.bin
	${PY} ppu_cvbs_preview.py -raw -cxp 4 -phd 4 -comb -box $<

240pee_ppucbvs_ph_0.png: 240pee.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 0 -phd 4 $<
240pee_ppucbvs_ph_4.png: 240pee.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 4 -phd 4 $<

${examples_dir}/240pee_2_PAL.png: 240pee_2.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -ppu 2C07 -cxp 0 -phd 4 -comb $<
	mv 240pee_2_ppucbvs_ph_0.png $@

240pee_2_ppucbvs_ph_0.png: 240pee_2.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 0 -phd 4 -comb $<
240pee_2_ppucbvs_ph_4.png: 240pee_2.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 4 -phd 4 -comb $<

240pee_3_ppucbvs_ph_0_comb.png: 240pee_3.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 0 -phd 4 -comb $<
	mv $(basename $<)_ppucbvs_ph_0.png $@
240pee_3_ppucbvs_ph_4_comb.png: 240pee_3.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 4 -phd 4 -comb $<
	mv $(basename $<)_ppucbvs_ph_4.png $@

240pee_3_ppucbvs_ph_0_box.png: 240pee_3.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 0 -phd 4 -box $<
	mv $(basename $<)_ppucbvs_ph_0.png $@
240pee_3_ppucbvs_ph_4_box.png: 240pee_3.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 4 -phd 4 -box $<
	mv $(basename $<)_ppucbvs_ph_4.png $@

usage.txt:
	${RM} -r $@
	${PY} ppu_cvbs_preview.py -h >> $@