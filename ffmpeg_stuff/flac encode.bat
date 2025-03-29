ffmpeg -y -i %1 -vn "%~dp1%~n1_reencode.flac" -compression_level 12
echo 
pause