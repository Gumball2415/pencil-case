ffmpeg -y -i %1 -vn -c:a libopus -b:a 8k -vbr on -compression_level 10 -frame_duration 120 -application voip -packet_loss 100 -fec 1 "%~dp1%~n1_bitcrush.opus"
echo 
pause