"""Calibrarea gripperului: unde e de fapt punctul de prindere, măsurat pe brațul real (pagina: Braț → Calibrare gripper).

Pentru fiecare măsurătoare: brațul găsește un cub pe masă cu camera (find_cube), apoi îl prinzi din controller (degetele
pe cub, centrat, cubul pe masă) și se salvează unghiurile. Din mai multe cuburi, puse în locuri diferite, se află
GRIP_LEN (încheietura → punctul de prindere, de-a lungul gripperului), GRIP_UP (cât e punctul în afara axei lui) și
GRIP_SIDE (cât e punctul de prindere la dreapta camerei: camera e montată puțin într-o parte) și GRIP_DZ (cât ajunge
brațul real mai sus decât modelul), astfel încât punctul de prindere calculat să cadă în centrul cuburilor.
Pentru GRIP_LEN și GRIP_DZ trebuie cuburi la distanțe diferite de bază (aproape și departe): altfel se compensează. Măsurătorile se păstrează în data/grip_samples.json.
"""
import json
import os

import numpy as np

from . import config as C
from . import fetch
from . import kinematics

SAMPLES_FILE = os.path.join(C.DATA_DIR, "grip_samples.json")
LOOK = [90, 105, 0, 0, 90]          # de unde caută cubul (privește masa din fața brațului)
BASES = (90, 70, 110, 50, 130)      # baza, pe rând, până vede un cub
SHOULDERS = (100, 112, 124, 134)    # și umărul (cotul, încheietura la 0): privește la ~13, ~17, ~21, ~26 cm de bază
FIT_LEN = (0.08, 0.20)              # m: între ce valori se caută GRIP_LEN
FIT_UP = 0.05                       # m: GRIP_UP între -atât și +atât
FIT_SIDE = 0.03                     # m: GRIP_SIDE între -atât și +atât
KEYS = ("GRIP_LEN", "GRIP_UP", "GRIP_SIDE", "GRIP_DZ", "GRIP_DZ_R")
DZ_R_MIN = 0.05                     # m: GRIP_DZ_R se află doar din cuburi întinse pe cel puțin atâta (aproape–departe)
DZ_R_MAX = 0.2                      # m pe m: și cel mult atât (1 cm la 5 cm): mai mult e zgomotul măsurătorilor


def pitch_line(samples):
    """[a, b]: înclinarea gripperului la calibrare ≈ a + b·distanța (m) de bază, sau [] dacă nu se poate afla
    (prea puține măsurători / toate la aceeași distanță)."""
    r = np.array([np.hypot(*s["xy"]) for s in samples])
    p = np.array([np.degrees(kinematics.pitches(s["angles"])[2]) for s in samples])
    if len(samples) < 2 or np.ptp(r) < 0.03:
        return []
    b, a = np.polyfit(r, p, 1)
    return [round(float(a), 1), round(float(b), 1)]


def load():
    try:
        with open(SAMPLES_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return []


def save(samples):
    os.makedirs(C.DATA_DIR, exist_ok=True)
    with open(SAMPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)


def find_cube(rob, get_frame):
    """Cubul cel mai apropiat de mijlocul imaginii: baza pe rând BASES, la fiecare umărul pe rând SHOULDERS (aproape →
    departe, apoi înapoi: fără drum în gol). {color, xy (m)} sau None."""
    for k, base in enumerate(BASES):
        for s2 in (SHOULDERS if k % 2 == 0 else SHOULDERS[::-1]):
            pose = list(LOOK)
            pose[0], pose[1] = base, s2
            rob.move(pose, 800)
            rob.sleep(0.5)
            cubes = fetch.table_cubes(get_frame(), pose)
            if cubes:
                c = min(cubes, key=lambda o: o["off"])
                return {"color": c["color"], "xy": [round(v, 4) for v in c["xy"]]}
    return None


def target(sample):
    """Centrul cubului (m): unde trebuia să fie punctul de prindere."""
    return np.array([sample["xy"][0], sample["xy"][1], C.FETCH_CUBE_M / 2])


def model_point(angles, glen, gup, side, dz, dz_r=0.0):
    w, d, up = kinematics.wrist(angles)
    p = w + glen * d + gup * up + side * kinematics.side(angles)
    return p + [0, 0, dz + dz_r * (np.hypot(*p[:2]) - C.GRIP_DZ_R0)]


def error(sample, values=None):
    """Cât greșește modelul (m, x / y / z) la o măsurătoare: punctul de prindere calculat minus centrul cubului.
    values = {GRIP_LEN, ...}; implicit cele de acum."""
    v = values or {k: getattr(C, k) for k in KEYS}
    return model_point(sample["angles"], *(v[k] for k in KEYS)) - target(sample)


def fit(samples):
    """GRIP_LEN, GRIP_UP, GRIP_SIDE, GRIP_DZ, GRIP_DZ_R (și eroarea rămasă, m) care duc punctul de prindere cel mai
    aproape de centrele cuburilor (căutare pe grilă, toate măsurătorile odată). Cu o singură măsurătoare GRIP_DZ
    rămâne 0; GRIP_DZ_R doar cu cel puțin 3, întinse pe DZ_R_MIN, și cel mult DZ_R_MAX."""
    if not samples:
        raise ValueError("Nicio măsurătoare: găsește un cub, prinde-l din controller, apoi „Salvează prinderea”")
    L = np.arange(FIT_LEN[0], FIT_LEN[1] + 1e-9, 0.002)[:, None, None, None]
    U = np.arange(-FIT_UP, FIT_UP + 1e-9, 0.002)[None, :, None, None]
    S = np.arange(-FIT_SIDE, FIT_SIDE + 1e-9, 0.002)[None, None, :, None]
    res = []
    for s in samples:
        w, d, up = kinematics.wrist(s["angles"])
        res.append(w - target(s) + L * d + U * up + S * kinematics.side(s["angles"]))
    res = np.stack(res)                                   # măsurătoare × lungime × sus × lateral × xyz
    # distanța de bază a fiecărui punct calculat (minus GRIP_DZ_R0): înălțimea = GRIP_DZ + GRIP_DZ_R × aceasta
    xy = np.float64([target(s)[:2] for s in samples])[:, None, None, None]
    r = np.hypot(res[..., 0] + xy[..., 0], res[..., 1] + xy[..., 1]) - C.GRIP_DZ_R0
    z = -res[..., 2]
    if len(samples) >= 3 and np.ptp([np.hypot(*s["xy"]) for s in samples]) >= DZ_R_MIN:
        rc = r - r.mean(0)
        dz_r = np.clip((rc * (z - z.mean(0))).sum(0) / (rc ** 2).sum(0), -DZ_R_MAX, DZ_R_MAX)
    else:
        dz_r = np.zeros(res.shape[1:4])
    dz = (z - dz_r * r).mean(0) if len(samples) > 1 else np.zeros(res.shape[1:4])
    res[..., 2] += dz + dz_r * r
    err = np.sqrt((res ** 2).sum(-1).mean(0))
    i, j, k = np.unravel_index(np.argmin(err), err.shape)
    return {"GRIP_LEN": round(float(L[i, 0, 0, 0]), 4), "GRIP_UP": round(float(U[0, j, 0, 0]), 4),
            "GRIP_SIDE": round(float(S[0, 0, k, 0]), 4), "GRIP_DZ": round(float(dz[i, j, k]), 4),
            "GRIP_DZ_R": round(float(dz_r[i, j, k]), 4), "GRIP_PITCH": pitch_line(samples),
            "err": float(err[i, j, k])}


def apply(values):
    """Valorile găsite: în program (imediat) și în data/settings.json (păstrate)."""
    C.apply_settings(values)
    try:
        with open(C.SETTINGS_FILE, encoding="utf-8") as f:
            s = json.load(f)
    except (FileNotFoundError, ValueError):
        s = {}
    s.update(values)
    os.makedirs(C.DATA_DIR, exist_ok=True)
    with open(C.SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
