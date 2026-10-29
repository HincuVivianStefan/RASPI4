@echo off
rem Porneste pagina web DOFBOT si o deschide in browser.
rem Prima data instaleaza singur bibliotecile (opencv, numpy) daca lipsesc.
rem Opresti serverul inchizand aceasta fereastra (sau Ctrl+C).
cd /d "%~dp0"
chcp 65001 >nul

rem Python: launcher-ul "py" (instalarea de pe python.org) sau "python" (din PATH)
set PY=
py -3 --version >nul 2>&1 && set PY=py -3
if not defined PY python --version >nul 2>&1 && set PY=python
if not defined PY (
    echo Nu gasesc Python. Instaleaza-l de pe https://www.python.org/downloads/
    echo si bifeaza "Add Python to PATH" la instalare. Apoi porneste din nou start.bat.
    pause
    exit /b 1
)

%PY% -c "import cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo Instalez bibliotecile ^(o singura data, dureaza un minut^)...
    %PY% -m pip install --user -r requirements.txt
    %PY% -c "import cv2, numpy" >nul 2>&1
    if errorlevel 1 (
        echo Instalarea nu a mers. Ruleaza manual:  %PY% -m pip install -r requirements.txt
        pause
        exit /b 1
    )
)

%PY% -m dofbot.web --browser
pause
