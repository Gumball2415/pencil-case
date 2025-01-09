ffmpeg -i %1 -i %2 -map 0:v:0 -map 1:a:0 -c:v libx264 -crf 18 -preset superfast -pix_fmt yuv420p -vf scale=out_color_matrix=bt709 -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart -c:a aac -b:a 384k "%~dpn1_transcode.mp4"
pause
