setlocal
composite_filter.py %*
call vhs-decode-cvbs.bat "%~dp1%~n1_13.5MHz_32_bit.flac"
