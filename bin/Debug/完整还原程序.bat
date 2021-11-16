@echo off
set username=bspace
set password=y13zowee
set listener=orcl
set filename=D:\oracledb\dbback\bspacedb.dmp
set fromusername=bspace
imp %username%/%password%@%listener% fromuser=%fromusername% touser=%username% file=%filename% log='D:\oracledb\dbremark\import.log'
pause>nul
echo 导入完成，按任意键退出