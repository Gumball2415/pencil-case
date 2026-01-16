@rem decodes flac compressed cvbs signals using vhs-decode suite
@setlocal
set decode_path=C:\Github\oyvindln\vhs-decode\decode

%decode_path%\decode.exe cvbs --frequency 13.5MHz -n -A -t 10 --overwrite "%1" "%~dp1%~n1_decode"
%decode_path%\ld-analyse "%~dp1%~n1_decode.tbc"
pause
%decode_path%\ld-chroma-decoder.exe --ntsc-phase-comp -f ntsc3d -t 10 -p y4m "%~dp1%~n1_decode.tbc" "%~dp1%~n1_decode.bin"
pause
ffmpeg -y -i "%~dp1%~n1_decode.bin" -c:v ffv1 -level 3 -coder 1 -context 1 -g 1 -slicecrc 1 -vf setfield=tff -flags +ilme+ildct -color_primaries smpte170m -color_trc bt709 -colorspace smpte170m -color_range tv -pix_fmt yuv422p10le -vf setdar=4/3,setfield=tff "%~dp1%~n1_decode.mkv"
ffmpeg -y -i "%~dp1%~n1_decode.bin" -vf setfield=tff -flags +ilme+ildct -color_primaries smpte170m -color_trc bt709 -colorspace smpte170m -color_range tv -pix_fmt yuv422p10le -vf setdar=4/3,setfield=tff "%~dp1%~n1_decode_%%07d.png"
del "%~dp1%~n1_decode.bin"
pause