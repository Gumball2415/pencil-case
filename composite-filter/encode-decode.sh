python3 composite_filter.py $1
vhs-decode cvbs --overwrite --frequency 13.5MHz -n -t 10 ${1%.*}_13.5MHz_32_bit.flac ${1%.*}_13.5MHz_32_bit_decode
tbc-tools ${1%.*}_13.5MHz_32_bit_decode.tbc
