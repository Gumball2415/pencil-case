ffmpeg -y -i %1 -vn -ar 32000 -c:a libmp3lame -b:a 8k -abr 1 "%~dp1%~n1_mp3_bitcrush.mp3"
echo 
pause