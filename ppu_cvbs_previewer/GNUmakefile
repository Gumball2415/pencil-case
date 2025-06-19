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

animlist := addie solstice rockman2 240pee 240pee_2_NTSC 240pee_3_comb 240pee_3_box smb smb_box

animlist := $(foreach o, ${animlist}, ${examples_dir}/${o}.mp4)

.PHONY: usage.txt

all: examples

clean:
	${RM} -r ${examples_dir}
	${RM} -r *_ppucbvs_ph_*.png

# addie.png:			squeak! averaging test
# solstice.mp4:			lidnariq says it looks cool
# rockman2.mp4:			another example; luma comb filter
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
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt601 -color_range 1 -colorspace bt601 -color_trc bt601 -color_primaries bt601 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_2_NTSC.mp4: 240pee_2_ppucbvs_ph_0.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_2_ppucbvs_ph_*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt601 -color_range 1 -colorspace bt601 -color_trc bt601 -color_primaries bt601 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_3_comb.mp4: 240pee_3_ppucbvs_ph_0_comb.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_3_ppucbvs_ph_*_comb.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt601 -color_range 1 -colorspace bt601 -color_trc bt601 -color_primaries bt601 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/240pee_3_box.mp4: 240pee_3_ppucbvs_ph_0_box.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '240pee_3_ppucbvs_ph_*_box.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt601 -color_range 1 -colorspace bt601 -color_trc bt601 -color_primaries bt601 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/smb_box.mp4: smb_box_ppucbvs_ph_0.png
	ffmpeg -framerate 60 -y -pattern_type glob -i '${notdir ${basename $@}}_ppucbvs_ph_*.png' \
	-t 2 \
	-vf "[in]loop=loop=-1:size=3:start=0[out];[out]scale=out_color_matrix=bt601 -color_range 1 -colorspace bt601 -color_trc bt601 -color_primaries bt601 -movflags faststart" \
	-c:v libx264 -crf 1 -preset superfast -pix_fmt yuv420p $@

${examples_dir}/addie.png: addie.bin
	${PY} ppu_cvbs_preview.py -raw -comb -frames 3 -avg -noskipdot $<
	mv "addie_ppucbvs_ph_(0, 4, 8).png" $@

addie_ppucbvs_ph_0.png: addie.bin
	${PY} ppu_cvbs_preview.py -raw -comb -frames 3 -noskipdot $<

solstice_ppucbvs_ph_0.png: solstice.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -frames 2 $<

rockman2_ppucbvs_ph_0.png: rockman2.bin
	${PY} ppu_cvbs_preview.py -raw -comb -lcomb -frames 2 $<

240pee_ppucbvs_ph_0.png: 240pee.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -frames 2 $<

${examples_dir}/240pee_2_PAL.png: 240pee_2.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -ppu 2C07 -comb $<
	mv 240pee_2_ppucbvs_ph_0.png $@

240pee_2_ppucbvs_ph_0.png: 240pee_2.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -comb -frames 2 $<

240pee_3_ppucbvs_ph_0_comb.png: 240pee_3.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -comb -frames 2 $<
	mv $(basename $<)_ppucbvs_ph_0.png 240pee_3_ppucbvs_ph_0_comb.png
	mv $(basename $<)_ppucbvs_ph_4.png 240pee_3_ppucbvs_ph_4_comb.png

240pee_3_ppucbvs_ph_0_box.png: 240pee_3.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -box -frames 2 $<
	mv $(basename $<)_ppucbvs_ph_0.png 240pee_3_ppucbvs_ph_0_box.png
	mv $(basename $<)_ppucbvs_ph_4.png 240pee_3_ppucbvs_ph_4_box.png

smb_ppucbvs_ph_0.png: smb.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 3 -frames 2 $<

smb_box_ppucbvs_ph_0.png: smb.png
	${PY} ppu_cvbs_preview.py -pal 2C02G_wiki.pal -cxp 3 -frames 2 -box $<
	mv $(basename $<)_ppucbvs_ph_3.png smb_box_ppucbvs_ph_3.png
	mv $(basename $<)_ppucbvs_ph_7.png smb_box_ppucbvs_ph_7.png

usage.txt:
	${RM} -r $@
	${PY} ppu_cvbs_preview.py -h >> $@

examples: ${examples_dir} ${examples_dir}/addie.png ${examples_dir}/240pee_2_PAL.png ${animlist} usage.txt
