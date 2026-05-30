@echo off
cd /d C:\Users\User\Desktop\RTL_LLM_CLAUDE\nn2rtl-repo
call C:\Users\User\oss-cad-suite\environment.bat
iverilog -g2012 -o build_inchain.vvp tb\conv200_inchain_probe_tb.v output\rtl\node_conv_200.v rtl_library\coord_scheduler.v rtl_library\line_buf_window.v rtl_library\conv_datapath_mp_k.v
echo COMPILE_RC=%errorlevel%
vvp build_inchain.vvp
echo RUN_RC=%errorlevel%
