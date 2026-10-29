@echo off
rem Verifica proiectul pe acest laptop: Python, biblioteci, testele si camerele.
rem Rezultatul ajunge in raport_windows.txt (trimite-l daca ceva nu merge).
rem Inchide intai start.bat, altfel serverul tine camera ocupata.
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

set PY=
py -3 --version >nul 2>&1 && set PY=py -3
if not defined PY python --version >nul 2>&1 && set PY=python
if not defined PY (
    echo Nu gasesc Python. Instaleaza-l de pe https://www.python.org/downloads/ ^(bifeaza "Add Python to PATH"^).
    pause
    exit /b 1
)

%PY% -c "import cv2, numpy" >nul 2>&1
if errorlevel 1 %PY% -m pip install --user -r requirements.txt

set OUT=raport_windows.txt
echo Raport DOFBOT > %OUT%
ver >> %OUT%
%PY% --version >> %OUT% 2>&1
%PY% -c "import cv2, numpy; print('OpenCV', cv2.__version__, '| numpy', numpy.__version__)" >> %OUT% 2>&1
echo. >> %OUT%
echo ===== Teste ===== >> %OUT%
echo Rulez testele...
%PY% -m unittest discover -s tests >> %OUT% 2>&1
echo. >> %OUT%
echo ===== Camere ===== >> %OUT%
echo Verific camerele...
%PY% tools\check_camera.py >> %OUT% 2>&1

echo.
type %OUT%
echo.
echo Gata. Rezultatul e in %OUT%
pause
