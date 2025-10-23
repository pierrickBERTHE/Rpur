@echo off
REM === Detect Windows user ===
set USERNAME=%USERNAME%

REM === Choose the path according to the user ===
if /I "%USERNAME%"=="pierr" (
    cd /d C:\Users\pierr\VSC_Projects\Rpur\src
) else if /I "%USERNAME%"=="Proprietaire" (
    cd /d C:\Users\Proprietaire\Rpur\src
) else (
    echo Utilisateur non reconnu : %USERNAME%
    pause
    exit /b
)

REM === Clean up old log file if it exists ===
if exist data\output\log\process_log.txt del data\output\log\process_log.txt

REM === Launch the Python script with the Poetry environment ===
poetry run python main.py

REM === Wait for 2 seconds to ensure file operations are complete ===
timeout /t 2 >nul

REM === Define paths ===
set LOGFILE=..\data\output\log\process_log.txt
set LOGBACKUPDIR=..\..\backup\log
if not exist %LOGBACKUPDIR% mkdir %LOGBACKUPDIR%

REM === Get client acronym from the log folder ===
set CLIENT_ACRONYM=
if exist ..\data\output\log\client_acronym.txt (
    set /p CLIENT_ACRONYM=<..\data\output\log\client_acronym.txt
) else (
    echo "Acronyme du client non trouvé dans le dossier log, veuillez vérifier."
    pause
    exit /b
)

REM === Create ISO-style timestamp ===
for /f "tokens=2 delims==." %%I in ('"wmic os get localdatetime /value"') do set ldt=%%I
set DATETIME=%ldt:~0,4%-%ldt:~4,2%-%ldt:~6,2%_%ldt:~8,2%-%ldt:~10,2%

REM === Create the backup file with the timestamp ===
set BACKUPFILE=%LOGBACKUPDIR%\process_log_%CLIENT_ACRONYM%_%DATETIME%.txt

REM === Copy the log file ===
if exist %LOGFILE% (
    copy /Y "%LOGFILE%" "%BACKUPFILE%" >nul
    echo.
    echo Fichier log sauvegarde avec succes !
    echo Nom du fichier : process_log_%CLIENT_ACRONYM%_%DATETIME%.txt
    echo Emplacement   : %LOGBACKUPDIR%
) else (
    echo Aucun fichier log trouvé à copier.
)

echo.
pause
