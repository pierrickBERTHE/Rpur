@echo off
REM Active l'environnement virtuel
call C:\Users\pierr\VSC_Projects\Rpur\.env\Scripts\activate.bat

REM Se place dans le dossier du script
cd /d C:\Users\pierr\VSC_Projects\Rpur\src

REM Lance le script Python
python main.py

pause
