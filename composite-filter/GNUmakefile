#!/usr/bin/make -f
# requires a python environment with requirements.txt installed

# taken from NROM template
ifeq (${OS}, Windows_NT)
	EXE_SUFFIX := .exe
	PY := py -3
else
	EXE_SUFFIX :=
	PY := python3
endif

in:=input
out:=output
docs:=docs

program := composite_filter.py

.PHONY: all clean examples

all: examples usage.txt

clean:
	${RM} -r $(docs)/*
	${RM} -r $(out)/*
	${RM} -r usage.txt

usage.txt:
	${RM} -r $@
	${PY} ${program} -h >> $@

# uses ld-tools suite and vhs-decode from https://github.com/oyvindln/vhs-decode
# they must be in path
imgs := PhilipsCircle SMPTEColorBars TheWindAndRainCBTL

outputs := $(foreach img,$(imgs),$(out)/$(img)*.png)

# simulate heavy noise
$(out)/TheWindAndRainCBTL%.png: $(in)/TheWindAndRainCBTL%.png
	${PY} ${program} "$<" -d -q -n 12
	cvbs-decode --overwrite --frequency 13.5MHz -n -t 10 \
		"$(basename  $<)_13.5MHz_16_bit.cvbs" \
		"${out}/$(basename $(notdir $<))_13.5MHz_16_bit_decode"

	ld-dropout-correct -o -t 10\
		--input-json "${out}/$(basename $(notdir $<))_13.5MHz_16_bit_decode.tbc.json" \
		--output-json "${out}/$(basename $(notdir $<))_13.5MHz_16_bit_decode.tbc.json" \
		"${out}/$(basename $(notdir $<))_13.5MHz_16_bit_decode.tbc" - |\
	ld-chroma-decoder -t 10 \
		- \
		--input-json "${out}/$(basename $(notdir $<))_13.5MHz_16_bit_decode.tbc.json" \
		--decoder ntsc2d --ntsc-phase-comp -p y4m |\
	ffmpeg -i - "${out}/$(basename $(notdir $<))_transcode_%02d.png"

$(out)/%.png: $(in)/%.png
	echo $<
	${PY} ${program} "$<" -d -q
	cvbs-decode --overwrite --frequency 13.5MHz -n -t 10 \
		"$(basename  $<)_13.5MHz_16_bit.cvbs" \
		"${out}/$(basename $(notdir $<))_13.5MHz_16_bit_decode"
	ld-chroma-decoder \
		"${out}/$(basename $(notdir $<))_13.5MHz_16_bit_decode.tbc" \
		--decoder ntsc2d --ntsc-phase-comp -p y4m |\
		ffmpeg -i - "${out}/$(basename $(notdir $<))_transcode_%02d.png"


examples: $(outputs)
