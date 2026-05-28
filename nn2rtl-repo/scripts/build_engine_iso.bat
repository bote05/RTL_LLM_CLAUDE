@echo off
REM Build the Verilator shared_engine isolation harness natively on Windows.
REM oss-cad environment.bat sets YOSYSHQ_ROOT + bin/lib (DLLs + python3); then
REM prepend w64devkit GCC 15 so its g++ (supports -fcf-protection=none) wins.
call C:\Users\User\oss-cad-suite\environment.bat >nul 2>&1
set PATH=C:\Users\User\w64devkit\bin;%PATH%
cd /d C:\Users\User\Desktop\RTL_LLM_CLAUDE\nn2rtl-repo
verilator_bin.exe --cc --exe --build -j 0 ^
  -Wno-fatal -Wno-UNOPTFLAT -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-UNUSED -Wno-BLKANDNBLK ^
  -CFLAGS "-O1" --top-module shared_engine -DNN2RTL_ENGINE_SUBBLOCKS_PROVIDED ^
  --Mdir obj_dir_engine_iso -o engine_iso ^
  output/rtl/shared_engine_skeleton.v ^
  output/rtl/engine/address_generator.v output/rtl/engine/config_register_block.v ^
  output/rtl/engine/mac_array.v output/rtl/engine/requant_pipeline.v ^
  output/rtl/engine/bram_to_stream_bridge.v ^
  tb/engine_verilator_iso_tb.cpp
echo BUILD_EXIT=%ERRORLEVEL%
