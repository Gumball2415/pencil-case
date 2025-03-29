ffmpeg -y -i %1 -vn -c:a libopus -b:a 96k -vbr on -compression_level 10 -frame_duration 120 "%~dp1%~n1_reencode.opus"
echo 
pause