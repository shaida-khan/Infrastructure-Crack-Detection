@echo off
cd /d C:\AI\CrackDetection
start http://127.0.0.1:7861
powershell -ExecutionPolicy Bypass -NoExit -Command ".\.venv\Scripts\Activate.ps1; python app.py"