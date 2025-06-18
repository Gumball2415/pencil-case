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

animlist := addie solstice rockman2 240pee 240pee_2_NTSC 240pee_3_comb 240pee_3_box smb

animlist := $(foreach o, ${animlist}, ${examples_dir}/${o}.mp4)

.PHONY: usage.txt

all: examples

clean:
	${RM} -r ${examples_dir}
	${RM} -r *_ppucbvs_ph_*.png

# addie.mp4:			squeak! comb filter test
# solstice.mp4:			lidnariq says it looks cool
# rockman2.mp4:			another example; comb + box filter test
# 240pee.mp4:			PLUGE
# 240pee_2_PAL.png:		comb filter test for PAL and NTSC
# 240pee_2_NTSC.mp4:	comb filter test for PAL and NTSC
# 240pee_3_comb.mp4:	sharpness tests
# 240pee_3_comb.mp4:	sharpness tests
# smb.mp4:				defaults

${examples_dir}:
	mkdir $@ -p

${examples_dir}/%.mp4: %_ppucbvs_ph_0.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '${notdir ${basename $@}}_ppucbvs_ph_*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

# ${examples_dir}/addie.mp4:	addie_ppucbvs_ph_0.png \
# 	addie_ppucbvs_ph_4.png \
# 	addie_ppucbvs_ph_8.png
# 	ffmpeg -framerate 60 -y -pattern_type glob -i 'addie_ppucbvs_ph_*.png' \
# 	-t 2 \
# 	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
# 	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

# ${examples_dir}/rockman2.mp4: rockman2_ppucbvs_ph_0.png \
# 	rockman2_ppucbvs_ph_4.png
# 	ffmpeg -framerate 60 -y -pattern_type glob -i 'rockman2_ppucbvs_ph_*.png' \
# 	-t 2 \
# 	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
# 	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

# ${examples_dir}/solstice.mp4:	solstice_ppucbvs_ph_0.png \
# 	solstice_ppucbvs_ph_4.png
# 	ffmpeg -framerate 60 -y -pattern_type glob -i 'solstice_ppucbvs_ph_*.png' \
# 	-t 2 \
# 	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
# 	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

# ${examples_dir}/240pee.mp4:	240pee_ppucbvs_ph_0.png \
# 	240pee_ppucbvs_ph_4.png
# 	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_ppucbvs_ph_*.png' \
# 	-t 2 \
# 	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
# 	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_2_NTSC.mp4: 240pee_2_ppucbvs_ph_0.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_2_ppucbvs_ph_*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_3_comb.mp4: 240pee_3_ppucbvs_ph_0_comb.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_3_ppucbvs_ph_*_comb.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_3_box.mp4: 240pee_3_ppucbvs_ph_0_box.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_3_ppucbvs_ph_*_box.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

addie_ppucbvs_ph_0.png: addie.bin
	${PY} ppu_cvbs_preview.py -raw -cxp 0 -phd 4 -comb -frames 3 -noskipdot $<

solstice_ppucbvs_ph_0.png: solstice.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 0 -phd 4 -frames 2 $<

rockman2_ppucbvs_ph_0.png: rockman2.bin
	${PY} ppu_cvbs_preview.py -raw -cxp 0 -phd 4 -comb -box -frames 2 $<

240pee_ppucbvs_ph_0.png: 240pee.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 0 -phd 4 -frames 2 $<

${examples_dir}/240pee_2_PAL.png: 240pee_2.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -ppu 2C07 -cxp 0 -phd 4 -comb $<
	mv 240pee_2_ppucbvs_ph_0.png $@

240pee_2_ppucbvs_ph_0.png: 240pee_2.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 0 -phd 4 -comb -frames 2 $<

240pee_3_ppucbvs_ph_0_comb.png: 240pee_3.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 0 -phd 4 -comb -frames 2 $<
	mv $(basename $<)_ppucbvs_ph_0.png $@
	mv $(basename $<)_ppucbvs_ph_4.png $@

240pee_3_ppucbvs_ph_0_box.png: 240pee_3.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 0 -phd 4 -box -frames 2 $<
	mv $(basename $<)_ppucbvs_ph_0.png $@
	mv $(basename $<)_ppucbvs_ph_4.png $@

smb_ppucbvs_ph_0.png: smb.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 3 -phd 4 -frames 2 $<

usage.txt:
	${RM} -r $@
	${PY} ppu_cvbs_preview.py -h >> $@

examples: ${examples_dir} ${animlist} ${examples_dir}/240pee_2_PAL.png usage.txt
