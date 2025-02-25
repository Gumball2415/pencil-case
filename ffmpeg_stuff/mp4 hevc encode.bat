ffmpeg -i %1 -c:v libx265 -vtag hvc1 -preset ultrafast -crf 18 -pix_fmt yuv420p -vf scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart -c:a aac -b:a 384k "%~dp1%~n1_reencode.mp4" -y
echo 
pause