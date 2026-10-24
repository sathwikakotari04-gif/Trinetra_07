
@echo off
cd /d D:\sih
start "Trinetra Server" cmd /k "python app.py"
timeout /t 5 >nul
start http://127.0.0.1:5000
exit
