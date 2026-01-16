setlocal
composite_filter.py "%1"
call vhs-decode-cvbs.bat "%~dp1%~n1_13.5MHz_24_bit.flac"
