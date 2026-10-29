# deploy/ – Raspberry Pi

## Instalare

```bash
sudo apt install -y python3-opencv python3-numpy python3-smbus i2c-tools
sudo raspi-config        # Interface Options -> I2C -> Enable (brațul e pe I2C)
```

`Arm_Lib` vine cu imaginea Yahboom pentru DOFBOT. Fără ea, brațul rulează simulat: în pagină apare „braț: simulat”.

Copiați tot folderul proiectului pe Pi, de exemplu în `/home/pi/concurs`.

## Pornire manuală

```bash
cd ~/concurs
./start.sh
```

Pagina se deschide de pe laptop sau telefon, la `http://<ip-ul-pi>:8000`. IP-ul îl aflați cu `hostname -I`. Oprirea: Ctrl+C.

## Pornire automată la boot

Verificați în `dofbot-web.service` utilizatorul (`User=pi`) și calea (`/home/pi/concurs`), apoi:

```bash
sudo cp deploy/dofbot-web.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dofbot-web
```

| Ce vrei | Comanda |
|---|---|
| jurnalul în timp real | `journalctl -u dofbot-web -f` |
| oprire | `sudo systemctl stop dofbot-web` |
| repornire după o schimbare de cod | `sudo systemctl restart dofbot-web` |
| fără pornire la boot | `sudo systemctl disable dofbot-web` |

Cât timp serviciul rulează, `./start.sh` nu poate porni a doua oară: portul 8000 e ocupat. Folosiți pagina deja pornită.

## Dacă merge greu

În `dofbot/config.py`, scădeți `DETECT_FPS` (de exemplu la 5) și `STREAM_FPS` (de exemplu la 10).
