"""Placa de răcire Yahboom de pe Pi: pornește coolerul și afișează pe ecranul ei IP, temperatura, procesor, memorie.

Pe I2C: ventilatorul la 0x0d, ecranul OLED SSD1306 128x32 la 0x3c.
Pornește singur cu pagina web (config.HAT). Separat (din rădăcina proiectului):
    python3 -m dofbot.hat          cooler + ecran, până la Ctrl+C (coolerul rămâne pornit)
    python3 -m dofbot.hat --off    stinge ecranul și coolerul
Pe laptop sau fără placă nu face nimic.
"""
import socket
import sys
import threading
import time

BUS = 1
FAN_ADDR, FAN_REG = 0x0d, 0x08
OLED_ADDR = 0x3c
W, H = 128, 32
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
REFRESH_S = 2

MESSAGE = {"lines": None, "until": 0.0}   # notificarea afișată în locul stării (vezi message())


def message(*lines, seconds=20):
    """Afișează pe ecran până la 2 rânduri scurte (~20 de caractere) timp de `seconds` secunde."""
    MESSAGE["lines"], MESSAGE["until"] = [str(x) for x in lines[:2]], time.time() + seconds


INIT = [0xAE, 0xD5, 0x80, 0xA8, H - 1, 0xD3, 0x00, 0x40, 0x8D, 0x14, 0x20, 0x00,
        0xA1, 0xC8, 0xDA, 0x02, 0x81, 0x8F, 0xD9, 0xF1, 0xDB, 0x40, 0xA4, 0xA6, 0xAF]


def cmd(bus, *codes):
    for c in codes:
        bus.write_byte_data(OLED_ADDR, 0x00, c)


def show(bus, img):
    """Trimite imaginea 1-bit pe ecran, pagină cu pagină (8 rânduri de pixeli pe pagină)."""
    px = img.load()
    data = []
    for page in range(H // 8):
        for x in range(W):
            b = 0
            for bit in range(8):
                if px[x, page * 8 + bit]:
                    b |= 1 << bit
            data.append(b)
    cmd(bus, 0x21, 0, W - 1, 0x22, 0, H // 8 - 1)
    for i in range(0, len(data), 16):
        bus.write_i2c_block_data(OLED_ADDR, 0x40, data[i:i + 16])


def ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))   # nu trimite nimic, doar alege interfața
        return s.getsockname()[0]
    except OSError:
        return "fără rețea"


def temp():
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        return int(f.read()) / 1000


def cpu_pct(prev):
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle, total = v[3] + v[4], sum(v)
    pct = 0 if prev is None else 100 * (1 - (idle - prev[0]) / max(1, total - prev[1]))
    return pct, (idle, total)


def ram_pct():
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            m[k] = int(v.split()[0])
    return 100 * (1 - m["MemAvailable"] / m["MemTotal"])


def fan(bus, on):
    bus.write_byte_data(FAN_ADDR, FAN_REG, 0x01 if on else 0x00)


def screen_loop(bus, stop=None):
    """Actualizează ecranul la REFRESH_S secunde, până la stop (threading.Event) sau Ctrl+C."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(FONT, 11)
    cmd(bus, *INIT)
    prev = None
    while not (stop and stop.is_set()):
        try:
            cpu, prev = cpu_pct(prev)
            img = Image.new("1", (W, H))
            d = ImageDraw.Draw(img)
            if MESSAGE["lines"] and time.time() < MESSAGE["until"]:
                for k, line in enumerate(MESSAGE["lines"]):
                    d.text((0, 2 + 16 * k), line, font=font, fill=1)
            else:
                d.text((0, 2), f"IP {ip()}", font=font, fill=1)
                d.text((0, 18), f"{temp():.0f}°C  CPU {cpu:.0f}%  RAM {ram_pct():.0f}%", font=font, fill=1)
            show(bus, img)
        except OSError:   # I2C ocupat o clipă: încercăm la următorul pas
            pass
        time.sleep(REFRESH_S)


def start():
    """Pornește coolerul și ecranul într-un fir separat. Întoarce un mesaj pentru terminal."""
    try:
        import smbus
        bus = smbus.SMBus(BUS)
        fan(bus, True)
    except (ImportError, OSError):
        return "Placa de răcire: nu e găsită (normal pe laptop)."
    try:
        import PIL  # noqa: F401
    except ImportError:
        return "Placa de răcire: cooler pornit; ecranul are nevoie de Pillow (sudo apt install python3-pil)."
    threading.Thread(target=screen_loop, args=(bus,), daemon=True).start()
    return "Placa de răcire: cooler și ecran pornite."


def main():
    import smbus
    bus = smbus.SMBus(BUS)
    if "--off" in sys.argv:
        cmd(bus, 0xAE)
        fan(bus, False)
        print("Ecran și cooler oprite.")
        return
    fan(bus, True)
    print("Cooler pornit, ecranul se actualizează. Ctrl+C ca să-l stingi.")
    try:
        screen_loop(bus)
    except KeyboardInterrupt:
        cmd(bus, 0xAE)


if __name__ == "__main__":
    main()
