@echo off
REM Build + run the output_serializer self-check TB (iverilog, timing TB).
call C:\Users\User\oss-cad-suite\environment.bat >nul 2>&1
cd /d C:\Users\User\Desktop\RTL_LLM_CLAUDE\nn2rtl-repo
iverilog -g2012 -o ser_tb.vvp tb\output_serializer_tb.v output\mobilenet-v2\rtl\output_serializer.v > scripts\ser_result.txt 2>&1
echo IVERILOG_EXIT=%ERRORLEVEL% >> scripts\ser_result.txt
vvp ser_tb.vvp >> scripts\ser_result.txt 2>&1
del ser_tb.vvp 2>nul
