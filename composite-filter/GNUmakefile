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

all: examples usage.txt

clean:
	${RM} -r $(docs)/*
	${RM} -r $(out)/*
	${RM} -r $(in)/*.cvbs
	${RM} -r usage.txt

usage.txt:
	${RM} -r $@
	${PY} ${program} -h >> $@

# uses ld-tools suite and vhs-decode from https://github.com/oyvindln/vhs-decode
# they must be in path
imgs := PhilipsCircle SMPTEColorBars TheWindAndRainCBTL
inputs :=	$(foreach img,$(imgs),$(in)/$(img).png)
encode :=	$(foreach img,$(imgs),$(in)/$(img)_13.5MHz_16_bit.cvbs)
decode :=	$(foreach img,$(imgs),$(out)/$(img)_13.5MHz_16_bit_decode.tbc)
outputs :=	$(foreach img,$(imgs),$(out)/$(img)_13.5MHz_16_bit_decode_01.png)

examples: $(outputs)

# generic rulesets
$(in)/%.png: $(inputs)

$(in)/%_13.5MHz_16_bit.cvbs: $(in)/%.png
	echo $<
	echo $@
	${PY} ${program} "$<" -d -q

$(out)/%_decode.tbc $(out)/%_decode.tbc.json $(out)/%_decode.log &: $(in)/%.cvbs
	echo $<
	echo $@
	cvbs-decode --overwrite --frequency 13.5MHz -n -t 10 "$<" \
	"$(out)/$(basename $(notdir $<))_decode"

$(out)/%_01.png: $(out)/%.tbc $(out)/%.tbc.json $(out)/%.log
	echo $^
	echo $@
	ld-dropout-correct -o -t 10\
		--input-json "$(word 2,$^)" \
		--output-json "$(word 2,$^)" \
		"$<" - |\
	ld-chroma-decoder -t 10 \
		- \
		--input-json "$<.json" \
		--decoder ntsc2d --ntsc-phase-comp -p y4m |\
	ffmpeg -i - "${basename $<}_%02d.png"

# specific rules

# simulate heavy noise
$(in)/TheWindAndRainCBTL_13.5MHz_16_bit.cvbs: $(in)/TheWindAndRainCBTL.png
	${PY} ${program} "$<" -d -q -n 12

# luma notch filter
$(in)/PhilipsCircle_13.5MHz_16_bit.cvbs: $(in)/PhilipsCircle.png
	${PY} ${program} "$<" -d -q -nl --plot_scanline 167

.PHONY: all clean examples
