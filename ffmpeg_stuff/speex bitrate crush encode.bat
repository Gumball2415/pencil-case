ffmpeg -i %1 -vn -acodec libspeex -ar 8000 -b:a 1 "%~dp1%~n1_speex_bitcrush.ogg" -y
echo 
pause