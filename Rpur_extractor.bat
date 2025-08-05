@echo off
REM Activate the virtual environment
call C:\Users\pierr\VSC_Projects\Rpur\.env\Scripts\activate.bat

REM Change to the script directory
cd /d C:\Users\pierr\VSC_Projects\Rpur\src

REM Run the Python script
python main.py

pause
