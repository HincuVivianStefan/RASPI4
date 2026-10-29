# CRI 2026 — Ghid de lucru DOFBOT (Raspberry Pi 4)

> **Notă (actualizare):** acesta e ghidul de la începutul proiectului. Codul din el e scheletul inițial și e depășit; codul actual e în `dofbot/`, iar pornirea e descrisă în `README.md` din rădăcină.

Tot ce trebuie ca să pornești: tema, setup-ul plăcii, API-ul brațului, structura proiectului și codul de start pentru ambele sarcini.

> ⚠️ Codul de mai jos e un **schelet de pornire**, netestat pe brațul vostru. Valorile marcate cu `# REGLEAZĂ` trebuie ajustate la voi (gripper, culori, timpi).

---

## 0. Tema pe scurt

**Echipament:** braț Dofbot + Raspberry Pi 4, cameră, caroiaj 3×3, cuburi colorate.

**Sarcina 1 — Reproducerea aranjării**
1. Camera recunoaște poziția (linie, coloană) și orientarea (culoarea feței de sus) fiecărui cub de pe caroiajul sursă.
2. Robotul reproduce aranjarea pe un al doilea caroiaj identic: aceeași poziție, aceeași față sus.

**Sarcina 2 — Turn sortat**
1. Identifică culorile cuburilor de pe caroiaj.
2. Construiește un turn de jos în sus, în ordinea primită la start (ex. roșu → galben → albastru → verde). În imaginea din temă turnul are 5 niveluri cu roșu de două ori, deci **o culoare se poate repeta**.
3. Pune turnul în celula aleasă de juriu.

**Punctaj (100):**

| Criteriu | Puncte |
|---|---|
| Recunoașterea pozițiilor (S1) | 20 |
| Reproducerea corectă (S1) | 20 |
| Identificarea și sortarea culorilor (S2) | 20 |
| Turnul în ordinea corectă (S2) | 20 |
| Viteză și precizie | 10 |
| Bonus creativitate | 10 |

**Constrângeri:** rulează autonom pe Pi 4, maximum **5 minute pe sarcină**. Se poate folosi orice bibliotecă open-source (OpenCV etc.).

### Întrebări pentru juriu (de lămurit cât mai repede!)
1. **„Orientare”**: cubul are fețe de culori diferite? Dofbot **nu poate răsturna un cub** (are doar rotație de încheietură, servo 5). În imaginea temei apare „jos = sus”, adică fețele opuse par să aibă aceeași culoare, deci poate nu e nevoie de răsturnare.
2. Cum primește robotul **ordinea turnului**: fișier, tastatură, cartonaș citit cu camera?
3. Cum sunt așezate **cele două caroiaje** (distanța față de braț)? Sunt fixe?
4. Cât de mari sunt **cuburile și celulele**?
5. La Sarcina 2: turnul se face pe caroiajul sursă? Ce se întâmplă cu cuburile nefolosite? Celula aleasă de juriu poate fi ocupată?

---

## 1. Setup Raspberry Pi

### 1.1 Ora (fă asta înainte de update!)
```bash
sudo timedatectl set-timezone Europe/Bucharest
sudo timedatectl set-ntp true
timedatectl
```
Dacă ora e tot greșită (rețeaua blochează NTP), o setezi manual:
```bash
sudo date -s "2026-10-29 14:30:00"
```
sau o iei de la Google:
```bash
sudo date -s "$(curl -sI https://www.google.com | grep -i '^date:' | cut -d' ' -f2-)"
```

### 1.2 Update
```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```
Pe **imaginea Yahboom** rulezi doar `sudo apt update`, fără `full-upgrade` (poate strica ROS sau driverele).

### 1.3 SSH + VNC + I2C
- Din meniu: **Preferences → Raspberry Pi Configuration → Interfaces**, apoi pornești **SSH**, **VNC** și **I2C**.
- Sau din terminal: `sudo raspi-config` → **Interface Options**.
- Verificare: `sudo systemctl status ssh` trebuie să arate `active (running)`.

IP-ul plăcii:
```bash
hostname -I
```

Conectarea de pe laptop:
```bash
ssh utilizator@raspberrypi.local
```

### 1.4 VS Code
1. Instalezi extensia **Remote - SSH** (Microsoft).
2. `F1` → **Remote-SSH: Connect to Host** → `utilizator@IP` → alegi **Linux** → scrii parola.
3. **File → Open Folder** → `~/concurs`.
4. Instalezi extensia **Python**. Se instalează pe Pi.

Probleme frecvente de rețea:
- **timeout / unreachable:** laptopul nu e în aceeași rețea, IP-ul s-a schimbat sau hotspotul izolează dispozitivele. Soluția sigură e un **cablu Ethernet direct** între laptop și Pi, apoi `ssh utilizator@raspberrypi.local`.
- **`REMOTE HOST IDENTIFICATION HAS CHANGED`:** `ssh-keygen -R IP`.

### 1.5 Pachete Python
```bash
sudo apt install -y python3-opencv python3-numpy python3-smbus i2c-tools
```
Verifici că placa brațului e vizibilă pe I2C:
```bash
i2cdetect -y 1
```
În tabel trebuie să apară **15**. Dacă nu apare: I2C nu e activat, cablul nu e bine pus sau placa nu are alimentare.

### 1.6 Arm_Lib (doar dacă NU ai imaginea Yahboom)
- Descarci `Dofbot.zip` din materialele Yahboom (linkul e în `Annex_Download_Link.txt` din https://github.com/yahboomtechnology/dofbot-pi).
- Dezarhivezi, intri în folderul `Arm_Lib` (cel cu `setup.py`) și rulezi:
```bash
sudo python3 setup.py install
```
Pe Bookworm, dacă refuză („externally managed”):
```bash
sudo pip3 install . --break-system-packages
```

Test rapid:
```bash
python3 -c "from Arm_Lib import Arm_Device; a=Arm_Device(); a.Arm_serial_servo_write6(90,90,90,90,90,90,1000)"
```
Brațul trebuie să se ridice drept în sus.

### 1.7 Test cameră
```bash
ls /dev/video*
```
```bash
python3 -c "import cv2; c=cv2.VideoCapture(0); r,f=c.read(); print(r, f.shape if r else None); cv2.imwrite('test.jpg', f)"
```

---

## 2. Brațul — ce trebuie știut

### 2.1 Servouri
| ID | Rol | Interval | Note |
|---|---|---|---|
| 1 | bază (rotație) | 0–180° | 90 = înainte |
| 2 | umăr | 0–180° | 90 = vertical |
| 3 | cot | 0–180° | |
| 4 | încheietură sus/jos | 0–180° | |
| 5 | rotație încheietură | **0–270°** | |
| 6 | gripper | 0–180° | valoare mică = deschis, mare = închis (verifică!) |

Gripperul se deschide cam **60 mm**. Nu poate răsturna cuburi.

### 2.2 API Arm_Lib
```python
from Arm_Lib import Arm_Device
Arm = Arm_Device()

Arm.Arm_serial_servo_write6(s1, s2, s3, s4, s5, s6, timp)   # toate servourile
Arm.Arm_serial_servo_write(id, unghi, timp)                 # un servo
unghi = Arm.Arm_serial_servo_read(id)                       # citire (poate întoarce None)
```
- `timp` = durata mișcării. Exemplele folosesc 400–1000. **Unitatea nu e clar documentată**: măsoar-o, nu presupune că sunt milisecunde.
- Funcția nu așteaptă terminarea mișcării, deci după ea pui `time.sleep(...)`.

### 2.3 Protocol I2C (ce face Arm_Lib dedesubt)
- Adresa plăcii: **0x15**, bus 1.
- Un servo: `[0x10+id, posH, posL, timpH, timpL]`.
- Toate: registrul `0x1D` primește 12 octeți (câte 2 pe servo), iar `0x1E` primește timpul.
- Citire: scrii `[0x30+id, 0]`, aștepți 3 ms, citești 2 octeți.
- Pozițiile brute sunt 900–3100 pentru 0–180°; servo 5 are 380–3700 pentru 0–270°. Servourile 2, 3, 4 sunt inversate în firmware.

### 2.4 Resurse
- Cursurile Yahboom: https://github.com/yahboomtechnology/dofbot-pi
  - `10.Basic control course`: control servouri;
  - `11.OpenCV course`, `12.AI vision basic course`: culori;
  - `14.AI vision catch course`: prindere și **stivuire cuburi**;
  - `18.Kinesiology advanced course`: cinematică inversă.
- Driver open-source cu protocolul: https://github.com/hipsterbrown/viam-yahboom-dofbot

---

## 3. Strategia

**Poziții învățate, nu cinematică inversă calculată.** Muți brațul o dată în fiecare poziție importantă și o salvezi în `positions.json`. La concurs doar le reiei. E mai precis și mult mai rapid de pus la punct.

Pozițiile de învățat (pentru servourile 1–5; gripperul se comandă separat):

| Nume | Ce e |
|---|---|
| `HOME` | poziție de repaus, sus |
| `LOOK_S` | camera privește caroiajul sursă de sus, tot caroiajul în cadru |
| `LOOK_D` | la fel, pentru caroiajul destinație |
| `S11_jos` … `S33_jos` | gripperul în jurul unui cub pus pe masă, în celula Sxy (x = linie, y = coloană) |
| `S11_sus` … `S33_sus` | deasupra aceleiași celule, la înălțimea unui turn de 6 cuburi |
| `D11_jos` … `D33_sus` | la fel, pentru caroiajul destinație |
| `S22_n3` etc. (opțional) | poziție exactă pentru nivelul 3 al turnului în celula S22; altfel se interpolează între `jos` și `sus` |

Ordinea unei mutări: **sus → coboară încet → închide → sus → mergi la destinație sus → coboară încet → deschide → sus.**

Pe scurt: liniile sunt R1 (sus în imagine) → R3, coloanele Col.1 (stânga) → Col.3, la fel ca în imaginea din temă.

---

## 4. Structura proiectului

```
~/concurs/
├── config.py            # constante (gripper, timpi, culori)
├── robot.py             # mișcări: pick / place / goto
├── vision.py            # cameră + detecție grilă
├── calibrate_arm.py     # înveți pozițiile din tastatură (prin SSH)
├── calibrate_vision.py  # colțurile caroiajului + test culori (prin VNC)
├── task1.py             # Sarcina 1
├── task2.py             # Sarcina 2
├── positions.json       # generat de calibrate_arm.py
└── vision.json          # generat de calibrate_vision.py
```

```bash
mkdir -p ~/concurs && cd ~/concurs
```

---

## 5. Cod

### 5.1 `config.py`
```python
# Toate valorile de reglat sunt aici.

POS_FILE = "positions.json"
VISION_FILE = "vision.json"

# Gripper (servo 6)                                   # REGLEAZĂ
GRIP_OPEN = 60
GRIP_CLOSE = 135

# Timpi de mișcare (unitatea Arm_Lib)                 # REGLEAZĂ
T_TRAVEL = 900     # între poziții "sus"
T_DESCEND = 1200   # coborâre/urcare lentă la cub
T_GRIP = 400

# Câte secunde înseamnă o unitate de timp Arm_Lib.
# Dacă mișcările se suprapun, crește; dacă stă degeaba, scade.   # REGLEAZĂ
TIME_UNIT_S = 0.001
SETTLE_S = 0.15

# Pozițiile "_sus" sunt învățate la înălțimea acestui nivel al turnului
SUS_LEVEL = 6

# Camera
CAM_INDEX = 0
CAM_W, CAM_H = 640, 480

# Culori: intervale HSV (OpenCV: H 0-179, S 0-255, V 0-255)  # REGLEAZĂ la lumina din sală
COLORS = {
    "rosu":     [((0, 90, 70), (8, 255, 255)), ((165, 90, 70), (179, 255, 255))],
    "galben":   [((18, 90, 90), (35, 255, 255))],
    "verde":    [((40, 70, 50), (85, 255, 255))],
    "albastru": [((90, 90, 50), (130, 255, 255))],
}
MIN_FRACTION = 0.35   # din petic, câți pixeli trebuie să fie de o culoare ca să conteze
PATCH = 0.45          # ce fracțiune din celulă analizăm (centrul ei)
```

### 5.2 `robot.py`
```python
import json
import time

from Arm_Lib import Arm_Device

import config as C


def load_positions():
    try:
        with open(C.POS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


class Robot:
    def __init__(self):
        self.arm = Arm_Device()
        time.sleep(0.1)
        self.pos = load_positions()
        self.grip = C.GRIP_OPEN

    def _wait(self, t):
        time.sleep(t * C.TIME_UNIT_S + C.SETTLE_S)

    def move(self, angles, t=C.T_TRAVEL):
        """angles = [s1..s5]; gripperul rămâne cum e."""
        a = [int(round(x)) for x in angles]
        self.arm.Arm_serial_servo_write6(a[0], a[1], a[2], a[3], a[4], self.grip, t)
        self._wait(t)

    def goto(self, name, t=C.T_TRAVEL):
        if name not in self.pos:
            raise KeyError(f"Poziția '{name}' nu e calibrată (rulează calibrate_arm.py)")
        self.move(self.pos[name], t)

    def gripper(self, value):
        self.grip = value
        self.arm.Arm_serial_servo_write(6, value, C.T_GRIP)
        self._wait(C.T_GRIP)

    def level_pose(self, cell, level):
        """Poziția gripperului pentru un cub aflat la nivelul `level` (1 = pe masă) în `cell`."""
        exact = f"{cell}_n{level}"
        if exact in self.pos:
            return self.pos[exact]
        jos = self.pos[f"{cell}_jos"]
        sus = self.pos[f"{cell}_sus"]
        # Interpolare liniară în unghiuri: aproximativă! Verifică și, la nevoie,
        # învață poziții exacte <cell>_n<level>.
        k = (level - 1) / (C.SUS_LEVEL - 1)
        return [j + (s - j) * k for j, s in zip(jos, sus)]

    def pick(self, cell, level=1):
        self.gripper(C.GRIP_OPEN)
        self.goto(f"{cell}_sus")
        self.move(self.level_pose(cell, level), C.T_DESCEND)
        self.gripper(C.GRIP_CLOSE)
        self.goto(f"{cell}_sus", C.T_DESCEND)

    def place(self, cell, level=1):
        self.goto(f"{cell}_sus")
        self.move(self.level_pose(cell, level), C.T_DESCEND)
        self.gripper(C.GRIP_OPEN)
        self.goto(f"{cell}_sus", C.T_DESCEND)

    def home(self):
        self.goto("HOME")
```

### 5.3 `calibrate_arm.py` (rulezi prin SSH)
```python
"""
Muți brațul din tastatură și salvezi pozițiile în positions.json.

  q/a  servo1 (bază)        w/s  servo2 (umăr)      e/d  servo3 (cot)
  r/f  servo4 (încheietură) t/g  servo5 (rotație)
  o / c  deschide / închide gripperul
  1..9   pasul (grade)
  p      salvează poziția curentă (îți cere numele)
  l      mergi la o poziție salvată
  n      ce poziții mai lipsesc
  x      ieșire
"""
import json
import sys
import termios
import tty

from Arm_Lib import Arm_Device

import config as C
from robot import load_positions

KEYS = {"q": (0, 1), "a": (0, -1), "w": (1, 1), "s": (1, -1), "e": (2, 1),
        "d": (2, -1), "r": (3, 1), "f": (3, -1), "t": (4, 1), "g": (4, -1)}
LIMITS = [(0, 180), (0, 180), (0, 180), (0, 180), (0, 270)]

REQUIRED = ["HOME", "LOOK_S", "LOOK_D"] + [
    f"{g}{r}{c}_{h}" for g in "SD" for r in (1, 2, 3) for c in (1, 2, 3) for h in ("jos", "sus")
]


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    arm = Arm_Device()
    pos = load_positions()
    angles = list(pos.get("HOME", [90, 90, 90, 90, 90]))
    grip = C.GRIP_OPEN
    step = 2
    arm.Arm_serial_servo_write6(*angles, grip, 1000)
    print(__doc__)

    while True:
        print(f"\r unghiuri={angles} gripper={grip} pas={step}      ", end="", flush=True)
        k = getch()
        if k in KEYS:
            i, sign = KEYS[k]
            lo, hi = LIMITS[i]
            angles[i] = max(lo, min(hi, angles[i] + sign * step))
            arm.Arm_serial_servo_write(i + 1, angles[i], 150)
        elif k in "123456789":
            step = int(k)
        elif k == "o":
            grip = C.GRIP_OPEN
            arm.Arm_serial_servo_write(6, grip, 300)
        elif k == "c":
            grip = C.GRIP_CLOSE
            arm.Arm_serial_servo_write(6, grip, 300)
        elif k == "p":
            name = input("\nNume poziție (ex. S11_jos): ").strip()
            if name:
                pos[name] = list(angles)
                with open(C.POS_FILE, "w") as f:
                    json.dump(pos, f, indent=2)
                print(f"Salvat {name} = {angles}")
        elif k == "l":
            name = input("\nMergi la: ").strip()
            if name in pos:
                angles = list(pos[name])
                arm.Arm_serial_servo_write6(*angles, grip, 1000)
            else:
                print("Nu există.")
        elif k == "n":
            missing = [n for n in REQUIRED if n not in pos]
            print(f"\nLipsesc ({len(missing)}): {' '.join(missing)}")
        elif k in ("x", "\x03"):
            print("\nGata.")
            break


if __name__ == "__main__":
    main()
```
Sfaturi pentru calibrare:
- Pentru `Sxy_jos` pui un cub în celulă, cobori cu gripperul deschis până îl cuprinde la jumătate, apoi verifici cu `c` că îl prinde bine. Abia apoi deschizi (`o`) și salvezi.
- `Sxy_sus` = cu cât e mai sus, cu atât mai sigur, dar mai lent. Ia înălțimea a 6 cuburi.
- Ține tensiunea constantă: aceeași sursă și aceeași masă la calibrare și la concurs.

### 5.4 `vision.py`
```python
import json
import time

import cv2
import numpy as np

import config as C

WARP = 300  # imaginea îndreptată are 300x300 px, 100 px pe celulă


def grab():
    cap = cv2.VideoCapture(C.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_H)
    time.sleep(0.3)
    frame = None
    for _ in range(8):          # aruncă primele cadre (expunere automată)
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise RuntimeError("Camera nu întoarce imagini")
    return frame


def load_corners(grid):
    with open(C.VISION_FILE) as f:
        return json.load(f)[grid]   # [[x,y] x4]: stânga-sus, dreapta-sus, dreapta-jos, stânga-jos


def warp(frame, corners):
    src = np.float32(corners)
    dst = np.float32([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (WARP, WARP))


def classify(patch_bgr):
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    best, best_frac = None, 0.0
    for name, ranges in C.COLORS.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        frac = cv2.countNonZero(mask) / total
        if frac > best_frac:
            best, best_frac = name, frac
    return (best, best_frac) if best_frac >= C.MIN_FRACTION else (None, best_frac)


def detect(grid, frame=None, debug=None):
    """Întoarce o matrice 3x3 cu numele culorii sau None. grid = 'S' sau 'D'."""
    if frame is None:
        frame = grab()
    top = warp(frame, load_corners(grid))
    cell = WARP // 3
    m = int(cell * (1 - C.PATCH) / 2)
    result = [[None] * 3 for _ in range(3)]
    for r in range(3):
        for c in range(3):
            y0, x0 = r * cell + m, c * cell + m
            patch = top[y0:y0 + cell - 2 * m, x0:x0 + cell - 2 * m]
            color, frac = classify(patch)
            result[r][c] = color
            cv2.rectangle(top, (x0, y0), (x0 + cell - 2 * m, y0 + cell - 2 * m), (255, 255, 255), 1)
            cv2.putText(top, f"{color or '-'} {frac:.2f}", (c * cell + 3, r * cell + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    if debug:
        cv2.imwrite(debug, top)
    return result


def print_grid(g):
    for r, row in enumerate(g):
        print(f"R{r + 1}: " + " | ".join(f"{(x or '-'):>8}" for x in row))
```

### 5.5 `calibrate_vision.py` (rulezi prin VNC, ai nevoie de ecran)
```python
"""
1. Brațul merge la LOOK_S (sau LOOK_D) și face o poză.
2. Dai click pe cele 4 colțuri ale caroiajului: stânga-sus, dreapta-sus, dreapta-jos, stânga-jos.
3. Se afișează live detecția. Apeși q ca să ieși.
Rulare: python3 calibrate_vision.py S   (sau D)
"""
import json
import sys

import cv2

import config as C
import vision
from robot import Robot

grid = sys.argv[1] if len(sys.argv) > 1 else "S"
Robot().goto(f"LOOK_{grid}")

frame = vision.grab()
pts = []


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


cv2.namedWindow("colturi")
cv2.setMouseCallback("colturi", on_click)
while len(pts) < 4:
    cv2.imshow("colturi", frame)
    if cv2.waitKey(20) == 27:
        sys.exit()
cv2.destroyAllWindows()

try:
    with open(C.VISION_FILE) as f:
        data = json.load(f)
except FileNotFoundError:
    data = {}
data[grid] = pts
with open(C.VISION_FILE, "w") as f:
    json.dump(data, f, indent=2)
print("Colțuri salvate:", pts)

# Test live
cap = cv2.VideoCapture(C.CAM_INDEX)
while True:
    ok, f = cap.read()
    if not ok:
        continue
    g = vision.detect(grid, f, debug="/tmp/grid.jpg")
    cv2.imshow("detectie", cv2.imread("/tmp/grid.jpg"))
    if cv2.waitKey(200) & 0xFF == ord("q"):
        break
cap.release()
vision.print_grid(g)
```
Dacă o celulă e detectată greșit, uită-te la fracția afișată și ajustează intervalele HSV din `config.py`.

Ca să vezi ce valori HSV are o culoare, faci clic pe imagine:
```python
# hsv_probe.py
import cv2, vision
f = vision.grab(); h = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
cv2.namedWindow("p"); cv2.setMouseCallback("p", lambda e, x, y, *_: e == 1 and print(h[y, x]))
cv2.imshow("p", f); cv2.waitKey(0)
```

### 5.6 `task1.py` — Reproducerea aranjării
```python
import time

import vision
from robot import Robot


def main():
    t0 = time.time()
    rob = Robot()

    rob.goto("LOOK_S")
    src = vision.detect("S", debug="s1_sursa.jpg")
    print("Sursa:")
    vision.print_grid(src)

    # De la rândul cel mai îndepărtat spre cel mai apropiat de braț, ca să nu
    # lovești cuburile deja puse. Dacă R1 e cel mai aproape de braț, inversează.
    for r in range(3):
        for c in range(3):
            if src[r][c]:
                print(f"Mut {src[r][c]} din S{r + 1}{c + 1} în D{r + 1}{c + 1}")
                rob.pick(f"S{r + 1}{c + 1}")
                rob.place(f"D{r + 1}{c + 1}")

    # Verificare
    rob.goto("LOOK_D")
    dst = vision.detect("D", debug="s1_dest.jpg")
    print("Destinație:")
    vision.print_grid(dst)
    errors = [(r, c) for r in range(3) for c in range(3) if src[r][c] != dst[r][c]]
    print("OK!" if not errors else f"Diferențe la: {errors}")

    rob.home()
    print(f"Timp: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
```
Dacă nu ai voie să iei cuburile de pe sursă (dacă sursa trebuie să rămână neatinsă), va trebui să iei cuburile dintr-o rezervă separată. Asta trebuie lămurit cu juriul.

### 5.7 `task2.py` — Turnul sortat
```python
"""
Rulare:  python3 task2.py 22 rosu galben verde albastru rosu
         (celula juriului = linie+coloană, apoi culorile de JOS în SUS)
"""
import sys
import time

import config as C
import vision
from robot import Robot


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    target = "S" + sys.argv[1]
    order = [c.lower() for c in sys.argv[2:]]
    unknown = [c for c in order if c not in C.COLORS]
    if unknown:
        sys.exit(f"Culori necunoscute: {unknown}. Valide: {list(C.COLORS)}")

    t0 = time.time()
    rob = Robot()
    rob.goto("LOOK_S")
    grid = vision.detect("S", debug="s2_sursa.jpg")
    vision.print_grid(grid)

    cells = {f"S{r + 1}{c + 1}": grid[r][c] for r in range(3) for c in range(3)}

    # Verifică dacă există destule cuburi din fiecare culoare
    for col in set(order):
        need = order.count(col)
        have = sum(1 for v in cells.values() if v == col)
        if have < need:
            sys.exit(f"Lipsesc cuburi {col}: am {have}, trebuie {need}")

    level = 1
    # Celula țintă e ocupată?
    if cells[target]:
        if cells[target] == order[0]:
            print(f"Baza ({order[0]}) e deja în {target}")
            level = 2
        else:
            free = next((k for k, v in cells.items() if v is None and k != target), None)
            if free is None:
                sys.exit("Celula țintă e ocupată și nu am unde muta cubul")
            print(f"Mut {cells[target]} din {target} în {free}")
            rob.pick(target)
            rob.place(free)
            cells[free], cells[target] = cells[target], None

    for i in range(level - 1, len(order)):
        col = order[i]
        src = next(k for k, v in cells.items() if v == col and k != target)
        print(f"Nivel {i + 1}: {col} din {src}")
        rob.pick(src)
        rob.place(target, level=i + 1)
        cells[src] = None

    rob.home()
    print(f"Turn gata în {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
```

---

## 6. Ordinea de lucru

1. [ ] Pi pregătit: ora, update, SSH, VS Code, I2C, `i2cdetect` arată `15`.
2. [ ] Test Arm_Lib: brațul se ridică la 90°.
3. [ ] Măsoară unitatea de timp: pornești o mișcare cu `timp=1000`, cronometrezi și ajustezi `TIME_UNIT_S`.
4. [ ] Reglează `GRIP_OPEN` / `GRIP_CLOSE` pe un cub real.
5. [ ] Fixează caroiajele pe masă cu bandă, ca să nu se miște niciodată. Marchează și poziția bazei brațului.
6. [ ] `calibrate_arm.py`: HOME, LOOK_S, LOOK_D, apoi 36 de poziții `_jos` / `_sus`. Cu `n` vezi ce mai lipsește.
7. [ ] `calibrate_vision.py S` și `calibrate_vision.py D` + reglaj HSV la lumina din sală.
8. [ ] Test `pick` / `place` pe o singură celulă, apoi pe toate.
9. [ ] `task1.py`.
10. [ ] Test turn: verifică nivelurile 2–5; unde interpolarea nu e bună, învață `Sxy_nK`.
11. [ ] `task2.py`.
12. [ ] Cronometrează, apoi accelerează: `T_TRAVEL` mai mic, păstrând `T_DESCEND` lent.
13. [ ] Bonus (vezi mai jos).
14. [ ] **Backup al cardului SD!**

---

## 7. Idei pentru bonusul de creativitate
- **Verificare după fiecare cub** cu camera și corectare automată.
- **Ordinea turnului citită de pe un cartonaș** cu pătrate colorate, pus în fața camerei.
- Ecran live (VNC sau un mic server web cu Flask) care arată ce detectează robotul.
- LED-ul RGB sau buzzerul plăcii semnalizează fazele: detecție, lucru, gata.
- Comenzi vocale („construiește turnul roșu galben verde”).

---

## 8. Pornire automată (fără intervenție după start)
`/etc/systemd/system/concurs.service`:
```ini
[Unit]
Description=Concurs DOFBOT
After=multi-user.target

[Service]
WorkingDirectory=/home/UTILIZATOR/concurs
ExecStart=/usr/bin/python3 /home/UTILIZATOR/concurs/task1.py
User=UTILIZATOR

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable concurs.service
```
Mai practic e un buton pe GPIO care pornește sarcina 1 sau 2. Așa nu pornește brațul singur la fiecare boot.

---

## 9. Probleme frecvente

| Simptom | Cauză probabilă |
|---|---|
| `i2cdetect` nu arată `15` | I2C dezactivat, cablul nu e bine pus, placa nu are alimentare |
| Pi-ul se resetează când se mișcă brațul | sursă slabă: folosește 5V/3A pentru Pi, separat de servouri |
| Brațul „tremură” sau ratează celula | poziții calibrate cu alt cub sau altă tensiune; coborâre prea rapidă |
| Culorile se încurcă (roșu/galben) | lumina e altfel, reglează HSV; evită soarele direct |
| `Arm_serial_servo_read` întoarce `None` | citire eșuată, încearcă din nou |
| Mișcările se suprapun | `TIME_UNIT_S` prea mic |
| `apt update`: Failed to fetch | data greșită, nu ai internet sau DNS-ul nu merge |

---

## 10. Surse
- Yahboom DOFBOT-Pi: https://github.com/yahboomtechnology/dofbot-pi
- Yahboom, driver library: https://www.yahboom.net/public/upload/upload-html/1699255422/8.Driver%20library%20and%20communication.html
- Protocol I2C (driver Viam): https://github.com/hipsterbrown/viam-yahboom-dofbot
- Viam DOFBOT quickstart: https://codelabs.viam.com/guide/dofbot-arm-quickstart
- Tema: https://www.dyd.ro/cri/tema/
