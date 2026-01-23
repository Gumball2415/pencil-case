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

program := composite_filter.py

.PHONY: all clean examples

all: examples usage.txt

clean:
	${RM} -r docs/*
	${RM} -r output/*

usage.txt:
	${RM} -r $@
	${PY} ${program} -h >> $@

# uses ld-tools suite and vhs-decode from https://github.com/oyvindln/vhs-decode
# they must be in path
imgs := PhilipsCircle SMPTEColorBars

outputs := $(foreach img,$(imgs),$(out)/$(img)*.png)

examples: $(outputs) usage.txt

$(out)/%.png: $(in)/%.png
	echo $<
	${PY} ${program} "$<" -d -q
	cvbs-decode --overwrite --frequency 13.5MHz -n -t 10 \
		"$(basename  $<)_13.5MHz_32_bit.cvbs" \
		"${out}/$(basename $(notdir $<))_13.5MHz_32_bit_decode"
	ld-chroma-decoder \
		"${out}/$(basename $(notdir $<))_13.5MHz_32_bit_decode.tbc" \
		--decoder ntsc2d --ntsc-phase-comp -p y4m |\
		ffmpeg -i - "${out}/$(basename $(notdir $<))_transcode_%02d.png"
