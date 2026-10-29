# Toate valorile de reglat sunt aici.
# Cele din TUNABLE (jos) se pot schimba și din pagina web; se salvează în settings.json.
import copy
import json
import os
import sys

# Pe Windows consola poate fi cp1252 și crapă la diacritice (ț, ș).
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))     # dofbot/
ROOT = os.path.dirname(HERE)                          # rădăcina proiectului
DATA_DIR = os.path.join(ROOT, "data")                 # tot ce se calibrează / salvează
CAPTURES_DIR = os.path.join(ROOT, "captures")         # poze salvate (pagină + depanare sarcini)
POS_FILE = os.path.join(DATA_DIR, "positions.json")
VISION_FILE = os.path.join(DATA_DIR, "vision.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

# Simulare (laptop): DOFBOT_SIM=1 forțează brațul simulat, DOFBOT_SIM=0 forțează brațul real.
# Nesetat: se folosește brațul real dacă Arm_Lib e instalat, altfel simularea.
SIM_ENV = os.environ.get("DOFBOT_SIM")
SIM_TIME_SCALE = 0.1   # în simulare, pauzele sunt de 10x mai scurte
SIM_VERBOSE = True     # afișează fiecare comandă trimisă brațului simulat

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

# Camera: indexul camerei (0 = camera laptopului / prima cameră USB)
CAM_INDEX = int(os.environ.get("DOFBOT_CAM", "0"))
CAM_W, CAM_H = 640, 480
CAM_FPS = 30
# DOFBOT_IMAGE=poza.jpg -> vision folosește o poză în loc de cameră (test fără cameră)
# DOFBOT_IMAGE=synth    -> caroiaj desenat de program; brațul simulat mută cuburile pe el
TEST_IMAGE = os.environ.get("DOFBOT_IMAGE")

# Pagina web (python3 web.py)
WEB_PORT = int(os.environ.get("DOFBOT_PORT", "8000"))
STREAM_FPS = 10       # cadre pe secundă trimise paginii (fluxul video, cu detecția desenată peste)
DETECT_FPS = 5        # de câte ori pe secundă rulează detecția (separat de flux; Pi 4: 4-5)
GRID_EVERY = 1        # caroiajul găsit se recaută la fiecare al atâtelea cadru (1 = mereu: urmărește camera când se mișcă)
GRID_NEED_CUBES = 1   # caroiajul S găsit de la zero e acceptat doar cu atâtea cuburi în celule (nu grile de pe jos); 0 = oricum
MOTION_DIFF = 3.0     # cât diferă două cadre (0-255, pe imaginea micșorată) ca să zicem că s-a mișcat camera; zgomotul ~0.8
CUBE_EVERY = 3        # cuburile se detectează la fiecare al atâtelea cadru (votul le ține precise); în sarcini: la fiecare
CV_THREADS = 2        # câte nuclee folosește OpenCV (Pi 4 are 4; restul rămân liberi, altfel procesorul ajunge la 100%)
HAT = True           # pe Pi: pornește coolerul și ecranul plăcii de răcire Yahboom odată cu pagina (dofbot/hat.py)
PAD = True           # pagina citește controllerul USB: control manual + oprire de urgență (dofbot/pad.py)
AUTO_AIM = True      # sarcinile își centrează singure caroiajul după LOOK_S / LOOK_D (dofbot/aim.py)
JPEG_QUALITY = 70

# Culori: intervale HSV (OpenCV: H 0-179, S 0-255, V 0-255)  # REGLEAZĂ la lumina din sală
COLORS = {
    "rosu":     [((0, 90, 70), (8, 255, 255)), ((165, 90, 70), (179, 255, 255))],
    "galben":   [((18, 90, 90), (35, 255, 255))],
    "verde":    [((40, 70, 50), (85, 255, 255))],
    "albastru": [((90, 90, 50), (130, 255, 255))],
}
# Caroiaj negru pe foaie albă, găsit automat în fiecare poză.
# Dacă nu e găsit, se folosesc colțurile din vision.json.
AUTO_GRID = True
GRID_C = 20           # cât de întunecată față de hârtia din jur trebuie să fie o linie (mai mic = mai sensibil)
GRID_MAX_S = 110      # liniile negre sunt nesaturate; peste asta e un cub colorat, nu linie
GRID_MIN_AREA = 0.04  # caroiajul ocupă cel puțin 4% din imagine
GRID_MIN_SCORE = 0.45 # cât de sigur trebuie să fie că e caroiaj (0..1)
GRID_MIN_LINE = 0.5   # fiecare din cele 4 linii interioare trebuie să fie vizibilă pe măcar atâta din lungime
GRID_MAX_RATIO = 2.2  # lățime/înălțime maximă a caroiajului în imagine (perspectiva îl turtește)

MIN_FRACTION = 0.35   # din petic, câți pixeli trebuie să fie de o culoare ca să conteze
PATCH = 0.45          # ce fracțiune din celulă analizăm (centrul ei)
STABLE_FRAMES = 5     # detecția folosită de sarcini = votul majoritar pe atâtea cadre

# Cuburi ca obiecte: o pată de culoare contează drept cub doar dacă arată ca un cub.
# Cu CUBE_MODE = False se revine la metoda veche (procentul de pixeli din centrul celulei).
CUBE_MODE = True
CUBE_MIN_AREA = 0.12      # aria petei, ca fracțiune din aria unei celule (fără caroiaj: din imagine / 50)
CUBE_MAX_AREA = 1.3
CUBE_MAX_ASPECT = 1.7     # latura lungă / latura scurtă (1 = pătrat perfect)
CUBE_MIN_FILL = 0.6       # aria / aria dreptunghiului care o încadrează (mică = formă zdrențuită)
MAX_CUBES = 9             # cel mult atâtea cuburi (cele mai mari); restul apar la respinse, cu motivul „prea multe”
# Cubul întreg (3D): doar fața de sus e colorată și stă la înălțimea cubului, deci văzută oblic cade
# deasupra celulei. Cu CUBE_3D, baza cubului se calculează din poza camerei, iar celula e cea a bazei.
CUBE_3D = True
CUBE_TOP = 0.9            # latura părții colorate / latura cubului (marginile cubului sunt albe)
CUBE_SIDE = 0.0           # latura cubului, în celule (ex. 0.8), ca să recunoască turnurile; 0 = orice pată e un cub
CAM_FOCAL = 0.0           # focala camerei, în pixeli la CAM_W; 0 = învățată singură din caroiaj
CAM_FOCAL_GUESS = 1.0     # până e învățată: focala / lățimea imaginii

# Sarcini
PAUSE_BETWEEN_PHASES = False   # True: după memorare, robotul așteaptă "Continuă" (Enter / buton web)
S1_ROW_ORDER = [1, 2, 3]       # în ce ordine se umplu rândurile destinației (întâi cel mai departe de braț)
S2_TARGET = "22"               # celula turnului (linie, coloană)
S2_ORDER = ["rosu", "galben", "verde", "albastru", "rosu"]   # de JOS în SUS

# Scena sintetică (DOFBOT_IMAGE=synth): culorile de pe caroiajul S la pornire / reset
SYNTH_LAYOUT = [["galben", "verde", "albastru"],
                ["rosu", "galben", "verde"],
                ["albastru", "rosu", "galben"]]

# ---------- setări modificabile din pagina web ----------
TUNABLE = ["COLORS", "AUTO_GRID", "GRID_C", "GRID_MAX_S", "GRID_MIN_AREA", "GRID_MIN_SCORE", "GRID_MIN_LINE",
           "GRID_MAX_RATIO",
           "MIN_FRACTION", "PATCH", "STABLE_FRAMES", "CUBE_MODE", "CUBE_MIN_AREA", "CUBE_MAX_AREA",
           "CUBE_MAX_ASPECT", "CUBE_MIN_FILL", "CUBE_3D", "CUBE_TOP", "CUBE_SIDE", "CAM_FOCAL", "PAUSE_BETWEEN_PHASES",
           "S1_ROW_ORDER", "S2_TARGET", "S2_ORDER", "SYNTH_LAYOUT"]
_DEFAULTS = {k: copy.deepcopy(globals()[k]) for k in TUNABLE}


def settings():
    return {k: copy.deepcopy(globals()[k]) for k in TUNABLE}


def _check(key, value):
    """Verifică și convertește o valoare primită (din JSON). Ridică ValueError dacă e greșită."""
    default = _DEFAULTS[key]
    if key == "COLORS":
        if not isinstance(value, dict) or not value:
            raise ValueError("COLORS trebuie să fie {nume: [[lo, hi], ...]}")
        out = {}
        for name, ranges in value.items():
            out[str(name)] = []
            for lo, hi in ranges:
                lo, hi = [int(v) for v in lo], [int(v) for v in hi]
                if len(lo) != 3 or len(hi) != 3:
                    raise ValueError(f"{name}: fiecare limită are 3 valori H, S, V")
                out[str(name)].append((tuple(lo), tuple(hi)))
        return out
    if key == "S1_ROW_ORDER":
        value = [int(v) for v in value]
        if sorted(value) != [1, 2, 3]:
            raise ValueError("S1_ROW_ORDER trebuie să fie o permutare a lui 1, 2, 3")
        return value
    if key == "S2_TARGET":
        value = str(value).strip()
        if len(value) != 2 or value[0] not in "123" or value[1] not in "123":
            raise ValueError("S2_TARGET = linie+coloană, ex. 22")
        return value
    if key == "S2_ORDER":
        value = [str(v).lower() for v in value]
        if not value:
            raise ValueError("S2_ORDER nu poate fi gol")
        return value
    if key == "SYNTH_LAYOUT":
        rows = [[(str(v) if v else None) for v in row] for row in value]
        if len(rows) != 3 or any(len(r) != 3 for r in rows):
            raise ValueError("SYNTH_LAYOUT trebuie să fie 3x3")
        return rows
    if isinstance(default, bool):
        return bool(value)
    if key == "STABLE_FRAMES":
        return max(1, min(30, int(value)))
    return type(default)(value)


def apply_settings(new):
    """Aplică doar cheile cunoscute; întâi le verifică pe toate, apoi le scrie."""
    checked = {}
    for k, v in new.items():
        if k not in TUNABLE:
            raise ValueError(f"Setare necunoscută: {k}")
        checked[k] = _check(k, v)
    unknown = [c for c in checked.get("S2_ORDER", S2_ORDER) if c not in checked.get("COLORS", COLORS)]
    if unknown:
        raise ValueError(f"Culori necunoscute în S2_ORDER: {unknown}")
    globals().update(checked)


def reset_settings():
    globals().update(copy.deepcopy(_DEFAULTS))


def save_settings():
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings(), f, indent=2, ensure_ascii=False)


def load_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            apply_settings(json.load(f))
    except FileNotFoundError:
        pass
    except (ValueError, TypeError) as e:
        print(f"[config] settings.json ignorat: {e}")


load_settings()
