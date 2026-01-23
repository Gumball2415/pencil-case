# uses ld-tools suite and vhs-decode from https://github.com/oyvindln/vhs-decode
python3 composite_filter.py "$@"
cvbs-decode -A --overwrite --frequency 13.5MHz -n -t 10\
    "${1%.*}_13.5MHz_32_bit.cvbs" \
    "${1%.*}_13.5MHz_32_bit_decode"
ld-process-vbi --input-json "${1%.*}_13.5MHz_32_bit_decode.tbc.json" -n -t 10\
    "${1%.*}_13.5MHz_32_bit_decode.tbc"

dir=$(dirname $1)
base=$(basename $1)
framedir=${base%.*}"_frames"

mkdir "$dir/$framedir"

ld-dropout-correct -o -t 10\
    --input-json "${1%.*}_13.5MHz_32_bit_decode.tbc.json" \
    --output-json "${1%.*}_13.5MHz_32_bit_decode.tbc.json" \
    "${1%.*}_13.5MHz_32_bit_decode.tbc" - |\
ld-chroma-decoder -t 10 - \
    --input-json "${1%.*}_13.5MHz_32_bit_decode.tbc.json" \
    --decoder ntsc2d --ntsc-phase-comp \
    -p y4m |\
ffmpeg -i - \
    "$dir/$framedir/${base%.*}_%07d.png"

ld-analyse "${1%.*}_13.5MHz_32_bit_decode.tbc"
