ffmpeg -i -vn %1 -c:a libmp3lame -q:a 0 "%~dp1%~n1.mp3" -y
echo 
pause