#!/usr/bin/env bash
real="$(realpath "$1")"
realraw=${real%.*}
ffmpeg -y -i "${real}" -c:v libx264 -crf 30 -preset superfast -pix_fmt yuv420p -vf scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart -c:a aac -b:a 384k "${realraw}_reencode.mp4"
echo 
