ffmpeg -y -i %1 -vn -acodec libspeex -ar 8000 -b:a 1 -compression_level 12 "%~dp1%~n1_speex_bitcrush.ogg"
echo 
pause