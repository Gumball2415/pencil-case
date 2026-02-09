real="$(realpath "$1")"
realraw=${real%.*}

echo $real
echo $realraw

ffmpeg -y -i "${real}" -vn -c:a libopus -b:a 192k -vbr on -compression_level 10 -frame_duration 120 "${realraw}_reencode.opus"
echo 
