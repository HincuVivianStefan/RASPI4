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
MEMORY_FILE = os.path.join(DATA_DIR, "memory.json")   # ce a memorat Sarcina 1 (rămâne și după repornire)
VISION_FILE = os.path.join(DATA_DIR, "vision.json")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

# Simulare (laptop): DOFBOT_SIM=1 forțează brațul simulat, DOFBOT_SIM=0 forțează brațul real.
# Nesetat: se folosește brațul real dacă Arm_Lib e instalat, altfel simularea.
SIM_ENV = os.environ.get("DOFBOT_SIM")
SIM_TIME_SCALE = 0.1   # în simulare, pauzele sunt de 10x mai scurte
SIM_VERBOSE = True     # afișează fiecare comandă trimisă brațului simulat
SIM_START = [90, 112, 15, 1, 90]   # unde stă brațul simulat la pornire (camera privește caroiajul S)

# Gripper (servo 6)                                   # REGLEAZĂ
GRIP_OPEN = 60
GRIP_CLOSE = 150   # gripperul gol se închide până la 180; pe cubul din set: 135 nu-l strânge, 165 prea tare (2026-10-30)

# Limitele servourilor 1..5 (bază, umăr, cot, încheietură, rotație), în grade. Arm_Lib ignoră în tăcere
# o comandă peste 180 (270 la servo 5) și trimite un impuls greșit sub 0, deci le verificăm noi.
LIMITS = [(0, 180), (0, 180), (0, 180), (0, 180), (0, 270)]

# Mișcarea lină: Robot.move trimite puncte intermediare (profil minimum-jerk: pornește și oprește fără
# smucitură, toate articulațiile ajung odată), ca pad.py. Durata = cea cerută, dar nu mai repede decât V_MAX/A_MAX.
MOTION_HZ = 40      # puncte pe secundă trimise brațului
V_MAX = 90          # grade/s, viteza maximă a unei articulații                    # REGLEAZĂ
A_MAX = 300         # grade/s², accelerația maximă (mai mic = mai lin, mai lent)   # REGLEAZĂ
SETTLE_TOL = 2.0    # grade: brațul a ajuns când toate servourile citite sunt atât de aproape de țintă
SETTLE_MAX_S = 0.6  # cât așteptăm, cel mult, să ajungă (după care mergem mai departe oricum)

# Poziția sigură (butonul din pagină și pornirea paginii) = de unde începe căutarea caroiajului: SEARCH dacă e
# învățată, altfel înălțimea din LOOK_S cu baza drept în față, altfel SAFE_POSE.
SAFE_POSE = [90, 112, 15, 1, 90]   # servo1..5: brațul ca cifra 2, camera în jos spre masă, în față  # REGLEAZĂ
SAFE_ON_START = True               # pe brațul real, pagina îl duce în poziția sigură când pornește
SAFE_SLOW_MS = 3000                # un servo necitit (de ex. trecut de capătul cursei) e adus încet, într-o comandă

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
# Controalele camerei (Linux): fixe, culorile nu se mai schimbă când brațul mută camera.
# Butonul „Auto-expune pe hârtie” din pagină reglează singur CAM_EXPOSURE.
CAM_EXPOSURE = 0      # 0 = automat; altfel expunere manuală (unitățile camerei, de obicei 100 µs)
CAM_GAIN = -1         # -1 = cum e; altfel amplificarea senzorului
CAM_WB = 0            # 0 = balans de alb automat; altfel temperatura, în K (ex. 4600)
# DOFBOT_IMAGE=poza.jpg -> vision folosește o poză în loc de cameră (test fără cameră)
# DOFBOT_IMAGE=synth    -> caroiaj desenat de program; brațul simulat mută cuburile pe el
TEST_IMAGE = os.environ.get("DOFBOT_IMAGE")

# Pagina web (python3 web.py)
WEB_PORT = int(os.environ.get("DOFBOT_PORT", "8000"))
# Aceeași pagină pe https (certificat făcut de program, în data/): browserul dă camera laptopului
# doar unei pagini https sau de pe localhost. 0 = fără https.
WEB_HTTPS_PORT = int(os.environ.get("DOFBOT_HTTPS_PORT", "8443"))
CERT_FILE = os.path.join(DATA_DIR, "https_cert.pem")
KEY_FILE = os.path.join(DATA_DIR, "https_key.pem")
STREAM_FPS = 10       # cadre pe secundă trimise paginii (fluxul video, cu detecția desenată peste)
DETECT_FPS = 5        # de câte ori pe secundă rulează detecția (separat de flux; Pi 4: 4-5)
GRID_EVERY = 5        # caroiajul e verificat la fiecare cadru; colțurile lui se reajustează la fiecare al atâtelea
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
# Lumina: foaia albă din cadru e adusă mereu la alb (balans de alb + luminozitate), înainte de culori.
# Așa pragurile S/V de mai sus sunt „față de hârtie” și culorile rămân recunoscute în lumină slabă.
LIGHT_NORM = True
LIGHT_MAX_GAIN = 4.0  # cel mult de atâtea ori mai luminos (peste, zgomotul senzorului acoperă culorile)

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
CAM_ROLLS = False         # camera se rotește cu servo 5? (pe brațul nostru nu: e prinsă înainte de rotația gripperului)

# Sarcini
PAUSE_BETWEEN_PHASES = False   # True: după memorare, robotul așteaptă "Continuă" (Enter / buton web)
S1_ROW_ORDER = [1, 2, 3]       # în ce ordine se umplu rândurile destinației (întâi cel mai departe de braț)
S2_TARGET = "22"               # celula turnului (linie, coloană)
S2_ORDER = ["rosu", "galben", "verde", "albastru", "rosu"]   # de JOS în SUS

# Sarcina 1, partea a doua (dofbot/fetch.py): cuburile mutate într-un loc pe care camera nu-l vedea („unghiul
# mort”) sunt căutate, luate unul câte unul și puse înapoi în celulele memorate.          # REGLEAZĂ pe brațul real
FETCH_CUBE_M = 0.033      # latura cubului, în metri (cubul din set: 3,3 cm)
GRIP_LEN = 0.146          # încheietura → punctul dintre degetele gripperului unde ajunge centrul cubului, m (Yahboom FK:
                          # brațul întins, punctul de prindere la 0,312 m = 2 × 0,08285 + 0,146; măsurat: calibrarea din pagină)
GRIP_UP = 0.0             # m: punctul de prindere, în afara axei gripperului (spre „sus” al camerei); calibrare
GRIP_SIDE = 0.0           # m: punctul de prindere, la dreapta camerei (camera montată puțin într-o parte); calibrare
GRIP_DZ = 0.0             # m: brațul real ajunge cu atât mai sus decât modelul (umărul / servourile); calibrare
GRIP_DZ_R = 0.0           # m pe m: și cu atât mai mult la fiecare metru mai departe de GRIP_DZ_R0 (întins, brațul se
GRIP_DZ_R0 = 0.17         # lasă sub greutatea lui); din calibrarea gripperului, cu cuburi aproape și departe
GRIP_DZ_RS = (0.135, 0.22)  # m: GRIP_DZ_R doar între aceste distanțe (cât acoperă măsurătorile din grip_samples.json:
                            # rândul 3, la ~11 cm, extrapolat, era lăsat de prea sus și cădea)
GRIP_PITCH = []           # [a, b]: înclinarea gripperului preferată la distanța r (m) de bază = a + b·r (grade; 180 =
                          # vertical): cum l-ai ținut la calibrare, unde valorile GRIP_* sunt bune. [] = cât mai vertical
# GRIP_*: măsurate din pagină (Braț → Calibrare gripper, gripcal.py) și scrise în data/settings.json
FETCH_LIFT = 0.05         # poziția de deasupra cubului (de acolo coboară să-l ia / să-l pună): atâția metri mai sus
LIFT_VERTICAL = True      # deasupra = pe verticală (coboară drept, nu trage foaia); False: înapoi de-a lungul gripperului
FINAL_DZ = 0.01           # m: ultimii atâția până jos, coborâți încet (T_FINAL)                       # REGLEAZĂ
T_FINAL = 700             # timpul ultimului centimetru (unitatea Arm_Lib)                           # REGLEAZĂ
GRIP_RELEASE = 115        # la punere gripperul se deschide doar până aici (larg, cu GRIP_OPEN, împingea cubul;
                          # pe cub: 135 nu-l mai strânge)                                              # REGLEAZĂ
FETCH_TOL = 0.004         # m: poziția cubului e bună când două priviri de aproape diferă cu mai puțin de atât
FETCH_OFFSET = [0.0, 0.0] # m, adunat la punctul de prindere (dinspre bază spre cub / la stânga lui), dacă brațul
                          # greșește mereu la fel (întâi: calibrarea gripperului din pagină, gripcal.py)
FETCH_ROLL_SIGN = -1      # servo 5 rotește gripperul invers decât crede modelul (pe brațul nostru: comandat +12°, cubul
                          # ieșea rotit -34°, 2026-10-30)
FETCH_MIN_PITCH = 110     # cât de înclinat poate fi gripperul pentru un cub departe (180 = vertical, 90 = orizontal)
FETCH_EMPTY_TOL = 6       # grade: gripperul închis la mai puțin de atât de GRIP_CLOSE = nu a prins nimic

# Scena sintetică (DOFBOT_IMAGE=synth): culorile de pe caroiajul S la pornire / reset
SYNTH_LAYOUT = [["galben", "verde", "albastru"],
                ["rosu", "galben", "verde"],
                ["albastru", "rosu", "galben"]]

# ---------- setări modificabile din pagina web ----------
TUNABLE = ["COLORS", "LIGHT_NORM", "LIGHT_MAX_GAIN", "CAM_EXPOSURE", "CAM_GAIN", "CAM_WB",
           "AUTO_GRID", "GRID_C", "GRID_MAX_S", "GRID_MIN_AREA", "GRID_MIN_SCORE", "GRID_MIN_LINE",
           "GRID_MAX_RATIO",
           "MIN_FRACTION", "PATCH", "STABLE_FRAMES", "CUBE_MODE", "CUBE_MIN_AREA", "CUBE_MAX_AREA",
           "CUBE_MAX_ASPECT", "CUBE_MIN_FILL", "CUBE_3D", "CUBE_TOP", "CUBE_SIDE", "CAM_FOCAL", "PAUSE_BETWEEN_PHASES",
           "GRIP_LEN", "GRIP_UP", "GRIP_SIDE", "GRIP_DZ", "GRIP_DZ_R", "GRIP_PITCH", "GRIP_RELEASE", "T_FINAL",
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
    if key == "GRIP_PITCH":
        value = [float(v) for v in value]
        if len(value) not in (0, 2):
            raise ValueError("GRIP_PITCH = [] sau [a, b] (înclinarea = a + b·distanța)")
        return value
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
