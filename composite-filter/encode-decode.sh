# uses ld-tools suite and vhs-decode from https://github.com/oyvindln/vhs-decode
python3 composite_filter.py "$@"
cvbs-decode --overwrite --frequency 13.5MHz -n -t 10 ${1%.*}_13.5MHz_32_bit.cvbs ${1%.*}_13.5MHz_32_bit_decode
ld-analyse ${1%.*}_13.5MHz_32_bit_decode.tbc
