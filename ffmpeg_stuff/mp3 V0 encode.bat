ffmpeg -y -i %1 -vn -c:a libmp3lame -q:a 0 "%~dp1%~n1_reencode.mp3" -y
echo 
pause