ffmpeg -y -i %1 -vn -ar 8000 -c:a libmp3lame -b:a 12k -abr 1 "%~dp1%~n1_mp3_bitcrush.mp3"
echo 
pause