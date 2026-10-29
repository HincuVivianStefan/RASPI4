@echo off
rem Leaga portul 8000 al laptopului de pagina de pe Raspberry Pi si deschide pagina.
rem Lasa fereastra asta deschisa cat folosesti pagina. Cere parola lui raspi pe Pi.
echo Conectare la Raspberry Pi 192.168.137.121. Lasa fereastra deschisa.
start "" cmd /c "timeout /t 5 /nobreak >/dev/null & start http://localhost:8000/"
ssh -L 8000:localhost:8000 raspi@192.168.137.121
pause
