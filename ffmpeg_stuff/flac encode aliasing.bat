ffmpeg -y -i %1 -vn -af aresample=resampler=swr:filter_size=1:phase_shift=0:linear_interp=0 -ar 48000 -compression_level 12 "%~dp1%~n1_reencode_nn_resample.flac"
echo 
pause