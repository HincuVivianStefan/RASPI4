"""Sarcina 1, după memorare: cuburile luate de pe caroiaj sunt căutate stânga / dreapta și puse înapoi în celulele lor.

Brațul e simetric în jurul bazei: ce prinde drept în față la o distanță, prinde la fel la orice unghi al bazei.
Deci se învață doar două prinderi în față (cu tools/calibrate_arm.py), cu baza la 90°:
  PICK_NEAR_sus / PICK_NEAR_jos   gripperul deasupra / în jurul unui cub aproape de braț
  PICK_FAR_sus  / PICK_FAR_jos    la fel, pentru un cub departe
Între ele se interpolează (t = 0 aproape, 1 departe). Apoi, din pagină, „Învață prinderea” (aproape / departe),
cu cubul tot acolo: ține minte unde se vede cubul din poziția de scanare și de deasupra lui (data/pick.json).

La rulare:
1. scan(): rotește baza stânga / dreapta din poziția de privire și găsește cuburile; din cât se mută un cub
   în imagine între două opriri ale bazei află unghiul bazei la care e fiecare cub.
2. approach(): baza spre cub, gripperul deasupra (t ghicit din poziția cubului în scanare), apoi corecții mici
   (baza și t) până când cubul se vede exact unde se vedea la învățare, adică e sub gripper.
3. Rotește încheietura după cub, coboară, prinde, îl pune în celula memorată (pozițiile Sxy_sus / Sxy_jos).
"""
import json
import os

import numpy as np

from . import config as C

SCAN_SPAN = 75          # grade de o parte și de alta a caroiajului
SCAN_STEP = 15          # grade între opririle bazei
SCAN_FOV = 60           # grade văzute pe lățimea imaginii, dacă nu reiese din scanare
SETTLE_S = 0.8          # așteptarea după o mișcare, să se oprească brațul și camera
SAME_AZ = 10            # două vederi ale aceleiași culori la mai puțin de atâtea grade = același cub
GRID_AZ = 12            # la o a doua scanare: cuburile din dreptul caroiajului sunt deja puse
SERVO_STEPS = 8         # corecții deasupra cubului, cel mult
SERVO_TOL = 0.03        # cât de aproape de punctul învățat e destul (fracțiune din imagine)
PROBE_BASE = 4          # grade de bază pentru proba de deasupra cubului
PROBE_T = 0.15          # cât din drumul aproape -> departe pentru probă
MAX_BASE = 8            # grade, cel mult, la o corecție
MAX_T = 0.25
T_RANGE = (-0.2, 1.2)   # puțin mai aproape / departe decât prinderile învățate
HOVER_REF = 1 / 6       # deasupra cubului, cubul e mare: „celula” de referință = 1/6 din imagine
WRIST_SIGN = 1          # +1 / -1: în ce sens se rotește încheietura (servo 5) când cubul e rotit în imagine  # REGLEAZĂ
LIMITS = [(0, 180), (0, 180), (0, 180), (0, 180), (0, 270)]
REQUIRED = [f"PICK_{k}_{h}" for k in ("NEAR", "FAR") for h in ("sus", "jos")]


def load():
    try:
        with open(C.PICK_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save(data):
    os.makedirs(os.path.dirname(C.PICK_FILE), exist_ok=True)
    with open(C.PICK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def missing(pos, learned=None):
    """Ce lipsește ca să meargă prinderea: poziții (calibrate_arm.py) și „Învață prinderea” din pagină."""
    learned = load() if learned is None else learned
    return [n for n in REQUIRED if n not in pos] + [f"învață {k}" for k in ("NEAR", "FAR") if k not in learned]


def lerp(a, b, t):
    return [x + (y - x) * t for x, y in zip(a, b)]


def wrist_delta(angle, ref):
    """Cu cât e rotit cubul față de cum era la învățare, în (-45, 45]: un pătrat arată la fel la fiecare 90°."""
    return (angle - ref + 45) % 90 - 45


class Picker:
    """rob: Robot. see(ref): cuburile dintr-un cadru nou, [{color, x, y, angle}] cu x, y fracțiuni din imagine;
    ref = mărimea unui cub (fracțiune din imagine) sau None. look: unghiurile din care se vede caroiajul."""

    def __init__(self, rob, see, log=print, look=None, learned=None, scan_ref=None):
        self.rob, self.see, self.log = rob, see, log
        self.look = list(look or rob.pos["LOOK_S"])
        self.learned = load() if learned is None else learned
        self.scan_ref = scan_ref
        self.J = None           # cum mută baza și t cubul în imagine, deasupra lui (măsurat o dată)
        self.deg_per_x = None   # grade de bază pe lățimea imaginii, la scanare (cu semn)

    # ------------------------------------------------------------ poziții
    def hover(self, t, base, wrist=0.0):
        p = self.rob.pos
        pose = lerp(p["PICK_NEAR_sus"], p["PICK_FAR_sus"], t)
        pose[0], pose[4] = base, pose[4] + wrist
        return self.clip(pose)

    def down(self, t, base, wrist=0.0):
        p = self.rob.pos
        pose = lerp(p["PICK_NEAR_jos"], p["PICK_FAR_jos"], t)
        pose[0], pose[4] = base, pose[4] + wrist
        return self.clip(pose)

    def target(self, t):
        """Unde se vede cubul (x, y) când e sub gripper, la distanța t."""
        n, f = self.learned["NEAR"], self.learned["FAR"]
        return np.array(lerp(n["hover"], f["hover"], t))

    def ref_angle(self, t):
        n, f = self.learned["NEAR"], self.learned["FAR"]
        return n.get("angle", 0) + (f.get("angle", 0) - n.get("angle", 0)) * t

    @staticmethod
    def clip(pose):
        return [max(lo, min(hi, a)) for a, (lo, hi) in zip(pose, LIMITS)]

    def go(self, pose, t=600):
        self.rob.move(pose, t)
        self.rob.sleep(SETTLE_S)

    # ------------------------------------------------------------ 1. scanare
    def scan(self, skip_grid=False):
        """Cuburile din jur: [{color, az (unghiul bazei spre cub), y (în imagine, la scanare)}]."""
        views = []
        b0 = round(self.look[0])
        bases = [b for b in range(b0 - SCAN_SPAN, b0 + SCAN_SPAN + 1, SCAN_STEP) if 0 <= b <= 180]
        for k, b in enumerate(bases):
            pose = list(self.look)
            pose[0] = b
            self.go(pose, 1500 if k == 0 else 600)
            seen = max((self.see(self.scan_ref) for _ in range(2)), key=len)
            views.append((b, seen))
            if seen:
                self.log(f"  baza {b}°: {', '.join(sorted(o['color'] for o in seen))}")
        self.deg_per_x = self.measure_fov(views)
        found = []
        for b, seen in views:
            for o in seen:
                az = b + (0.5 - o["x"]) * self.deg_per_x
                same = next((f for f in found if f["color"] == o["color"] and abs(f["az"] - az) < SAME_AZ
                             and abs(f["y"] - o["y"]) < 0.15), None)
                if same:        # același cub, văzut din altă oprire: media, cântărită spre vederea mai centrală
                    w = 1 - abs(o["x"] - 0.5)
                    same["az"] = (same["az"] * same["w"] + az * w) / (same["w"] + w)
                    same["y"] = (same["y"] * same["w"] + o["y"] * w) / (same["w"] + w)
                    same["w"] += w
                else:
                    found.append({"color": o["color"], "az": az, "y": o["y"], "w": 1 - abs(o["x"] - 0.5)})
        if skip_grid:
            found = [f for f in found if abs(f["az"] - self.look[0]) > GRID_AZ]
        for f in found:
            f["az"] = round(f["az"], 1)
            f.pop("w")
        return found

    def measure_fov(self, views):
        """Grade de bază pe lățimea imaginii: cât se mută același cub între două opriri vecine ale bazei."""
        ratios = []
        for (b1, s1), (b2, s2) in zip(views, views[1:]):
            for o in s1:
                near = [p for p in s2 if p["color"] == o["color"] and abs(p["y"] - o["y"]) < 0.1]
                if len(near) == 1 and abs(near[0]["x"] - o["x"]) > 0.05:
                    ratios.append((b2 - b1) / (near[0]["x"] - o["x"]))
        if ratios:
            k = float(np.median(ratios))
            if 15 < abs(k) < 150:
                self.log(f"  la scanare imaginea cuprinde ~{abs(k):.0f}° de bază")
                return k
        return self.deg_per_x or SCAN_FOV

    # ------------------------------------------------------------ 2. deasupra cubului
    def guess_t(self, y):
        n, f = self.learned["NEAR"].get("scan_y"), self.learned["FAR"].get("scan_y")
        if n is None or f is None or abs(f - n) < 0.02:
            return 0.5
        return float(np.clip((y - n) / (f - n), *T_RANGE))

    def find(self, color, t):
        """Cubul de culoarea cerută cel mai apropiat de punctul de sub gripper, sau None."""
        seen = [o for o in self.see(HOVER_REF) if o["color"] == color]
        if not seen:
            return None
        tgt = self.target(t)
        return min(seen, key=lambda o: np.hypot(o["x"] - tgt[0], o["y"] - tgt[1]))

    def error(self, color, t, base):
        self.go(self.hover(t, base))
        o = self.find(color, t)
        return (None, None) if o is None else (np.array([o["x"], o["y"]]) - self.target(t), o)

    def probe(self, color, t, base, e):
        """Cum mută baza și t cubul în imagine (coloanele lui J)."""
        J = np.zeros((2, 2))
        for k, (db, dt) in enumerate(((PROBE_BASE, 0), (0, PROBE_T))):
            e2, _ = self.error(color, t + dt, base + db)
            if e2 is None:
                return None
            J[:, k] = (e2 - e) / (db or dt)
        if abs(np.linalg.det(J)) < 1e-6:
            return None
        return J

    def approach(self, cube):
        """Gripperul exact deasupra cubului: (t, baza, cubul văzut) sau None dacă nu-l găsește."""
        base, color = cube["az"], cube["color"]
        e = o = None
        for t in dict.fromkeys([self.guess_t(cube["y"]), 0.5, 0.0, 1.0]):
            e, o = self.error(color, t, base)
            if e is not None:
                break
        if e is None:
            self.log(f"  nu văd cubul {color} de deasupra (baza {base:.0f}°)")
            return None
        for it in range(SERVO_STEPS):
            if np.abs(e).max() < SERVO_TOL:
                return t, base, o
            if self.J is None:
                self.J = self.probe(color, t, base, e)
                if self.J is None:
                    self.log("  proba de deasupra cubului nu a mers (cubul a ieșit din imagine)")
                    return None
                e, o = self.error(color, t, base)
                if e is None:
                    return None
            d = np.linalg.solve(self.J, -e)
            db, dt = float(np.clip(d[0], -MAX_BASE, MAX_BASE)), float(np.clip(d[1], -MAX_T, MAX_T))
            nb, nt = base + db, float(np.clip(t + dt, *T_RANGE))
            e2, o2 = self.error(color, nt, nb)
            if e2 is None or np.abs(e2).max() > np.abs(e).max() * 1.2:
                self.J = None   # corecția a stricat: măsurăm din nou de aici
                if e2 is None:
                    self.go(self.hover(t, base))
                    continue
            base, t, e, o = nb, nt, e2, o2
        if np.abs(e).max() < 2 * SERVO_TOL:
            return t, base, o
        self.log(f"  nu am ajuns deasupra cubului {color} (eroare {np.abs(e).max():.2f})")
        return None

    # ------------------------------------------------------------ 3. prinde și pune
    def take(self, cube):
        got = self.approach(cube)
        if got is None:
            return False
        t, base, o = got
        wrist = WRIST_SIGN * wrist_delta(o.get("angle", 0), self.ref_angle(t))
        self.log(f"  prind {cube['color']}: baza {base:.0f}°, distanța {t:.2f}, încheietura {wrist:+.0f}°")
        self.rob.pick_pose(self.hover(t, base, wrist), self.down(t, base, wrist), what=cube["color"])
        return True

    def rebuild(self, memorized, grid="S", row_order=(1, 2, 3), present=None):
        """Pune cuburile din jur în celulele memorate. Întoarce {"placed": [...], "missing": [...]}.
        present = ce e acum pe caroiaj (3x3): celulele ocupate rămân cum sunt, iar la scanare cuburile
        din dreptul caroiajului nu se iau (sunt cele de pe el)."""
        present = present or [[None] * 3 for _ in range(3)]
        on_grid = any(any(row) for row in present)
        if not any(memorized[r][c] and not present[r][c] for r in range(3) for c in range(3)):
            self.log("Nu lipsește niciun cub de pe caroiaj.")
            return {"placed": [], "missing": []}
        cubes = self.scan(skip_grid=on_grid)
        self.log(f"Am găsit {len(cubes)} cuburi: " + ", ".join(f"{c['color']} la {c['az']:.0f}°" for c in cubes))
        placed, miss, rescanned = [], [], False
        for r in [x - 1 for x in row_order]:
            for c in range(3):
                color = memorized[r][c]
                if not color:
                    continue
                cell = f"{grid}{r + 1}{c + 1}"
                if present[r][c]:
                    if present[r][c] != color:
                        self.log(f"În {cell} e {present[r][c]}, dar am memorat {color}: îl las acolo.")
                    continue
                while True:
                    cands = [k for k in cubes if k["color"] == color]
                    if not cands and not rescanned:
                        self.log(f"Nu mai am {color} în listă: mai caut o dată în jur.")
                        cubes, rescanned = self.scan(skip_grid=True), True
                        continue
                    if not cands:
                        break
                    here = self.rob.current()[0]
                    cube = min(cands, key=lambda k: abs(k["az"] - here))
                    cubes.remove(cube)
                    if self.take(cube):
                        self.rob.place(cell)
                        placed.append(cell)
                        self.log(f"[{len(placed)}] {color} pus în {cell}")
                        break
                if cell not in placed:
                    miss.append(cell)
                    self.log(f"Nu am găsit un cub {color} pentru {cell}.")
        return {"placed": placed, "missing": miss}
