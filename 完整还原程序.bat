@echo off
set username=SH
set password=sh
set listener=orcl
set filename=D:\oracledb1\dbback\bspacedb.dmp
set fromusername=SH
imp %username%/%password%@%listener% fromuser=%fromusername% touser=%username% file=%filename% log='D:\oracledb1\dbremark\import.log'
pause>nul
echo 导入完成，按任意键退出