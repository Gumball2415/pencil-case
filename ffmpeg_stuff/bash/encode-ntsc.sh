# vhs-decode and ld-decode suite must be in path
echo "$1"

real="$(realpath "$1")"
echo $real

fullfilepath="${real%.*}"
echo $fullfilepath

ffmpeg -hwaccel auto -i "$real" -an -vf \
    "scale=758:480:flags=lanczos+accurate_rnd+full_chroma_int+bitexact,
    pad=758:486:0:3,
    fps=fps=60000/1001,
    tinterlace=interleave_bottom,
    fieldorder=bff" \
    -vcodec rawvideo \
    -pix_fmt yuv444p16 \
    -f rawvideo - |\
    ld-chroma-encoder -p yuv -f NTSC - "${fullfilepath}_encode.tbc"

# ld-process-vbi "${fullfilepath}_encode.tbc" \
#     --input-json "${fullfilepath}_encode.tbc.json" -n -t 10
# # ld-analyse "${fullfilepath}_encode.tbc"
# ld-dropout-correct -t 10 \
#     "${fullfilepath}_encode.tbc" "${fullfilepath}_encode_t.tbc"


ld-chroma-decoder --input-json "${fullfilepath}_encode.tbc.json"\
    --decoder ntsc3d -t 10 -p y4m "${fullfilepath}_encode.tbc" - \
    | ffmpeg -y -hwaccel auto -i - -i "$real" \
        -c:a copy \
        -c:v ffv1 \
            -level 3 \
            -coder 1 \
            -context 1 \
            -g 1 \
            -slicecrc 1 \
            -flags +ilme+ildct \
            -pix_fmt yuv422p10le \
        -vf "setdar=4/3,setfield=tff" \
        -map 0:v:0 -map 1:a:0? \
    "${fullfilepath}_encode.mkv"

# todo: use separatefields approach
ffmpeg -y -hwaccel auto -i "${fullfilepath}_encode.mkv" \
    -filter_complex \
        "[0:v]il=l=d:c=d:a=d,
            split=2[top][bottom];
        [top]crop=iw:ih/2:0:0:keep_aspect=1,
            scale=0:ih*2:flags=neighbor+accurate_rnd+full_chroma_int+bitexact
            [field1];
        [bottom]crop=iw:ih/2:0:(ih/2)+1:keep_aspect=1,
            scale=0:ih*2:flags=neighbor+accurate_rnd+full_chroma_int+bitexact,
            rgbashift=rv=1:gv=1:bv=1:av=1:edge=1[field2];
        [field1][field2]framepack=frameseq,
            scale=1440:1080:flags=gauss+accurate_rnd+full_chroma_int+bitexact,
            setdar=4/3"\
    -c:v libx264 -crf 18 -preset ultrafast \
        -pix_fmt yuv420p \
        -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 \
        -movflags faststart \
    -c:a aac -b:a 384k \
    "${fullfilepath}_transcode.mp4"

# ffmpeg -y -hwaccel auto -i "${fullfilepath}_transcode.mp4" -i "$1" -map 0:v:0 -map 1:a:0? -c:v copy -c:a copy "${fullfilepath}_transcode_aud.mkv"
# tbc-video-export "${fullfilepath}_encode.tbc" "${fullfilepath}_encode.mkv" --chroma-decoder ntsc2d --overwrite -t 10 --profile ffv1


# lossless 4k upscale
# ffmpeg -hwaccel auto -i "${fullfilepath}_encode.mkv" -vf "format=yuv422p10le,bwdif=mode=send_field:parity=auto:deint=all,scale=2880:2176:flags=gauss,setdar=4/3" -c:v ffv1 -aspect 4:3 -c:a copy -color_range tv -color_primaries smpte170m -colorspace smpte170m -color_trc bt709 -pix_fmt yuv422p10le "${fullfilepath}_upscale.mkv" -y

# compressed 1080p with true bobbing deinterlace
# ffmpeg -hwaccel auto -y -i "${fullfilepath}_encode_aud.mkv" -filter_complex "[0:v]scale=out_color_matrix=bt709,il=l=d:c=d:a=d,split=2[top][bottom]; [top]crop=iw:ih/2:0:0:keep_aspect=1,scale=0:ih:flags=gauss[field2]; [bottom]crop=iw:ih/2:0:(ih/2)+1:keep_aspect=1,scale=0:ih:flags=gauss[field1]; [field2][field1]framepack=frameseq,scale=1440:1088:flags=gauss,setdar=4/3[out]" -map "[out]":v:0 -map 0:a:0 -c:v libx264 -crf 18 -preset superfast -pix_fmt yuv420p -color_range 1 -colorspace bt709 -color_trc bt709 -color_primaries bt709 -movflags faststart -c:a aac -b:a 384k "${fullfilepath}_transcode.mp4"

# alternately, use this filter for bwdif
# -vf "scale=out_color_matrix=bt709,bwdif=mode=send_field:parity=auto:deint=all,scale=1440:1088:flags=gauss,setdar=4/3"
