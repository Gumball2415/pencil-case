@setlocal
@REM @echo off

@rem decodes flac compressed cvbs signals using vhs-decode suite
@rem https://github.com/oyvindln/vhs-decode

@rem Path to vhs-decode binaries.

set decode_path=C:\Github\oyvindln\decode_suite\decode

%decode_path%\decode cvbs --frequency 13.5MHz -n -t 10 --overwrite "%1" "%~dp1%~n1_decode"
%decode_path%\ld-process-vbi "%~dp1%~n1_decode.tbc" --input-json "%~dp1%~n1_decode.tbc.json" -n -t 10
%decode_path%\ld-analyse "%~dp1%~n1_decode.tbc"
pause
%decode_path%\tbc-video-export "%~dp1%~n1_decode.tbc" "%~dp1%~n1_decode.mkv" --chroma-decoder ntsc2d --overwrite -t 10 --profile ffv1
@REM ffmpeg -y -i "%~dp1%~n1_decode.bin" -c:v ffv1 -level 3 -coder 1 -context 1 -g 1 -slicecrc 1 -vf setfield=tff -flags +ilme+ildct -color_primaries smpte170m -color_trc bt709 -colorspace smpte170m -color_range tv -pix_fmt yuv422p10le -vf setdar=4/3,setfield=tff "%~dp1%~n1_decode.mkv"
ffmpeg -hwaccel auto -i "%~dp1%~n1_decode.mkv" -vf format=yuv422p10le -vf scale=2880:2176 -c:v ffv1 -aspect 4:3 -vf bwdif=1:-1:0 -c:a copy -color_range tv -color_primaries smpte170m -colorspace smpte170m -color_trc bt709 -pix_fmt yuv422p10le -vf setdar=4/3 "%~dp1%~n1_decode_reencode.mkv" -y
mkdir "%~dp1%~n1_frames"
ffmpeg -y -i "%~dp1%~n1_decode.mkv" "%~dp1%~n1_frames\%~n1_decode_%%07d.png"
pause