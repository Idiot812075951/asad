@echo off
set year=%date:~0,4%
set month=%date:~5,2%
set day=%date:~8,2%
set filename=bspacedb
set username=SH
set password=sh
set listener=orcl
set directory=D:\oracledb1\dbback
exp %username%/%password%@%listener% grants=N file=%directory%/%filename%.dmp INDEXES=n STATISTICS=none  log='D:\oracledb1\dbremark\export.log'