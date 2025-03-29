mkdir "%~dp1%~n1_frames"
ffmpeg -y -i %1 "%~dp1%~n1_frames\%~n1_%%04d.png"
echo 
pause