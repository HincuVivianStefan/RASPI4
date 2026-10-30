"""Brațul își caută singur caroiajul și se așază ca să-l vadă cât mai bine (cu tot cu cuburi).

1. Dacă nu vede caroiajul, rotește baza pe toată cursa (la înălțimea actuală, apoi puțin mai aproape și
   mai departe) până îl găsește.
2. Mișcă de probă baza, umărul, cotul și încheietura câte câteva grade și vede cum se mută caroiajul în imagine
   (o singură dată: efectul lor, Jacobianul, se ține minte în data/aim.json și se corectează după fiecare mișcare).
3. Corectează în pași mici până când caroiajul e la mijloc pe orizontală, cu linia de jos (cea mai apropiată de braț,
   mereu bine văzută) aproape de marginea de jos a imaginii (BOTTOM_Y), mare, dar cu margine în jur: fețele de sus
   ale cuburilor cad deasupra celulelor în imagine și trebuie să încapă și ele, mai ales la rândul de sus.
Dacă se vede doar o parte din caroiaj (un colț), restul e extrapolat (vision.find_partial_grid), deci știe încotro.

Fiecare mișcare trece prin Robot, deci Stop din pagină sau orice buton de pe controller o oprește imediat.
Folosit din pagină (butonul „Caută caroiajul”, sarcinile cu AUTO_AIM) și din tools/center_grid.py.
"""
import itertools
import json
import os

import cv2
import numpy as np

from . import config as C
from . import fetch
from . import kinematics
from . import vision

JOINTS = [0, 1, 2, 3]       # baza, umărul, cotul, încheietura (indici 0..4 în unghiuri)
NAMES = ["baza", "umărul", "cotul", "încheietura", "rotația"]
PROBE = 3                   # grade pentru mișcarea de probă
MAX_STEP = 4                # grade, cel mult, la o corecție
MIN_STEP = 1.0              # grade, cel puțin (servourile au ~1° joc; un pas mai mic nu mută imaginea)
SIZE_RANGE = (0.40, 0.62)   # cât din imagine să ocupe caroiajul (radical din arie): cu margine, dar mare
TOLERANCE = 0.05            # cât de aproape de mijloc e destul (din lățimea / înălțimea imaginii)
DAMPING = 3e-5              # frânează corecțiile pe care articulațiile le pot face doar cu mișcări mari
SETTLE_S = 0.3              # cât mai așteptăm după o mișcare (lină, verificată la sosire), să stea camera
T_MOVE = 600                # durata unei corecții (unitatea Arm_Lib)
SEARCH_STEP = 10            # grade între opririle bazei când caută caroiajul
SEARCH_CENTER = 90          # baza „drept în față”
# Căutarea în față: cu cât se mișcă umărul (1), cotul (2), încheietura (3) față de poziția de start, în grade.
# Cotul ridicat mult duce camera la orizont (vede sala), deci el se mișcă puțin.
SWEEP = {1: (0, -8, 8, -16, 16, -24, 24), 2: (0, -15, 15), 3: (0, -10, 10, -20, 20, -30, 30)}
SWEEP_UNIT = {1: 8, 2: 15, 3: 10}    # pentru ordine: întâi pozițiile apropiate de start
SWEEP_MAX = 45              # câte poziții încearcă, cel mult
CLIMB_STEPS = [(0, 5), (0, -5), (1, 6), (1, -6), (2, 10), (2, -10), (3, 8), (3, -8)]   # (articulație, grade)
CLIMB_ITERS = 5             # de câte ori, cel mult, mută poziția spre una mai bună
CLIMB_GOOD = 0.08           # scor suficient de bun: ne oprim
GRID_TRIES = 10             # unde vede cuburi: pe câte cadre încearcă să găsească liniile caroiajului
SEARCH_TILT = 12            # grade de umăr pentru a doua / a treia tură de căutare
EDGE_MARGIN = 0.07          # un colț mai aproape de marginea imaginii = nu încape cu tot cu cuburi: îl micșorăm
# Linia de jos a caroiajului (cea mai apropiată de braț) e adusă aici, din înălțimea imaginii: jos, ca deasupra
# caroiajului să rămână loc pentru fețele de sus ale cuburilor. Sub ea nu cade nimic, deci marginea e mai mică.
# Întâi caroiajul e centrat (acolo detecția e sigură și efectul articulațiilor se măsoară bine), apoi e coborât.
BOTTOM_Y = 0.87                                                                              # REGLEAZĂ
BOTTOM_MARGIN = 0.03
LOWER_STEPS = 8             # pași de corecție, cel mult, la coborâre
# Căutarea liniară (caroiajul e fix în fața brațului): baza drept în față, cotul și încheietura la capăt (camera
# cât mai sus), doar umărul se lasă pe spate în pași egali de la LINE_NEAR la LINE_FAR. Camera se trage înapoi și urcă:
# privește tot mai departe (~15 → ~30 cm) și de la distanță tot mai mare, deci caroiajul intră în imagine pe sus, coboară
# și se micșorează. Rămâne unde linia de jos e cel mai aproape de BOTTOM_Y, cu tot caroiajul în imagine.
# Servo1..4; servo 5 rămâne cum e. Pozițiile învățate „LINE_NEAR” / „LINE_FAR” (positions.json) au întâietate.
LINE_NEAR = [90, 105, 0, 0]                                                                  # REGLEAZĂ
LINE_FAR = [90, 140, 0, 0]                                                                   # REGLEAZĂ
LINE_STEPS = 12
# Încadrarea (zoom in / zoom out) după ce traseul liniar a găsit caroiajul: caroiajul e dus pe masă (m), iar poziția
# în care se vede cel mai bine e calculată din cinematică: linia de jos la BOTTOM_Y, la mijloc, cât mai mare (până la
# SIZE_RANGE), cu tot cu fețele de sus ale cuburilor în imagine. Ce greșește modelul se măsoară și se corectează.
FRAME_ITERS = 3
FRAME_GRID = [range(60, 151, 6), range(0, 91, 10), range(0, 61, 10)]  # umărul, cotul, încheietura (căutarea grosieră)
SIZE_WEIGHT = 1.0           # cât contează mărimea față de centrare
STEPS = 15                  # pași de corecție, cel mult
MIN_CUBES = 1               # fără linii vizibile, grupul de cuburi ține loc de caroiaj (apoi se verifică)
LIMITS = C.LIMITS
JACOBIAN_FILE = os.path.join(C.DATA_DIR, "aim.json")   # cum mută fiecare articulație caroiajul în imagine


def quad_metrics(c, w, h):
    """(centru x, centru y, mărime, 1 dacă un colț e prea aproape de margine, linia de jos y) ale unui patrulater
    (colțuri în px), ca fracțiuni din imagine."""
    c = np.float32(c)
    rel = c / [w, h]
    bottom = np.sort(rel[:, 1])[-2:].mean()     # cele două colțuri de jos
    out = float((rel < EDGE_MARGIN).any() or (rel[:, 0] > 1 - EDGE_MARGIN).any()
                or (rel[:, 1] > 1 - BOTTOM_MARGIN).any())
    cx, cy = rel.mean(0)
    return np.array([cx, cy, np.sqrt(cv2.contourArea(c) / (w * h)), out, bottom])


def grid_metrics(frame, prev=None, need_cubes=0):
    """(centru x, centru y, mărime, 1 dacă un colț e prea aproape de margine, linia de jos y), ca fracțiuni din imagine,
    plus colțurile caroiajului (sau None dacă a folosit cuburile). (None, None) dacă nu vede nimic.
    prev = colțurile de la măsurătoarea anterioară: cu ele e găsit și caroiajul acoperit de cuburi.
    need_cubes: găsit de la zero (fără prev), caroiajul trebuie să aibă atâtea cuburi în celule."""
    corners = vision.find_grid(frame, prev)
    if corners is not None and prev is None and need_cubes and vision.cubes_inside(frame, corners) < need_cubes:
        corners = None      # o grilă fără cuburi: probabil altceva de pe masă / de pe jos
    if corners is not None:
        c = np.float32(corners)
    elif (part := vision.find_partial_grid(frame)) is not None:
        # Doar o parte e în imagine: restul e extrapolat, deci centrul și mărimea spun încotro și cât să mergem.
        return partial_metrics(part, frame.shape[1], frame.shape[0]), None
    else:
        # Caroiaj plin: liniile se văd prea puțin. Dreptunghiul care cuprinde toate cuburile e aproape caroiajul.
        cubes, _ = vision.find_cubes(frame)
        if len(cubes) < MIN_CUBES:
            hint = vision.lines_hint(frame)     # măcar bucăți de linie: spre ele
            return (None, None) if hint is None else (np.array([*hint, sum(SIZE_RANGE) / 2, 1.0, hint[1]]), None)
        x, y, bw, bh = cv2.boundingRect(np.float32(np.vstack([o["box"] for o in cubes])))
        h, w = frame.shape[:2]
        # Doar cuburile: marginea lor de jos ține loc de linia de jos, dar mărimea lor nu spune cât e caroiajul.
        return np.array([(x + bw / 2) / w, (y + bh / 2) / h, sum(SIZE_RANGE) / 2, 0.0, (y + bh) / h]), None
    h, w = frame.shape[:2]
    return quad_metrics(c, w, h), corners


def partial_metrics(corners, w, h):
    """Metricile (ca grid_metrics) unui caroiaj extrapolat: „iese din imagine” = 1."""
    m = quad_metrics(corners, w, h)
    m[3] = 1.0
    return m


def project(angles, pts, size):
    """Unde se văd în imagine punctele pts (m, sistemul brațului) din unghiurile `angles`, ca fracțiuni din imagine.
    None dacă vreunul e în spatele camerei."""
    pos, R = kinematics.camera_pose(angles)
    w, h = size
    f = vision.focal(w)
    pc = (np.float64(pts) - pos) @ R.T
    if (pc[:, 2] < 0.02).any():
        return None
    return np.column_stack([f * pc[:, 0] / pc[:, 2] + w / 2, f * pc[:, 1] / pc[:, 2] + h / 2]) / [w, h]


def predict(angles, table, size, bias):
    """(centru x, linia de jos, mărime, cât iese din imagine) prezise pentru caroiajul de pe masă `table` (4 colțuri, m)
    și fețele de sus ale cuburilor (aceleași colțuri, ridicate cu latura cubului). bias = ce greșește modelul."""
    ground = np.column_stack([table, np.zeros(4)])
    tops = ground + [0, 0, C.FETCH_CUBE_M]
    p = project(angles, np.vstack([ground, tops]), size)
    if p is None:
        return None
    p = p + [bias[0], bias[1]]
    g = p[:4]
    bottom = np.sort(g[:, 1])[-2:].mean()
    area = 0.5 * abs(np.dot(g[:, 0], np.roll(g[:, 1], 1)) - np.dot(g[:, 1], np.roll(g[:, 0], 1)))
    size_ = np.sqrt(area) * bias[2]
    over = (np.maximum(0, EDGE_MARGIN - p).sum() + np.maximum(0, p[:, 0] - (1 - EDGE_MARGIN)).sum()
            + np.maximum(0, g[:, 1] - (1 - BOTTOM_MARGIN)).sum())
    return np.array([g[:, 0].mean(), bottom, size_, over])


def frame_cost(m):
    return (abs(m[1] - BOTTOM_Y) + abs(m[0] - 0.5) + 2 * max(0, SIZE_RANGE[0] - m[2]) + 2 * max(0, m[2] - SIZE_RANGE[1])
            + 0.5 * (SIZE_RANGE[1] - min(m[2], SIZE_RANGE[1])) + 10 * m[3])


def best_frame(table, size, bias, s5):
    """Poziția (servo1..5) din care caroiajul de pe masă se vede cel mai bine, și ce s-ar vedea de acolo."""
    c = np.float64(table).mean(0)
    base = float(np.clip(90 + np.degrees(np.arctan2(c[1], c[0])), *LIMITS[0]))
    best = (np.inf, None, None)

    def tryp(s2, s3, s4):
        nonlocal best
        if not all(LIMITS[j][0] <= v <= LIMITS[j][1] for j, v in ((1, s2), (2, s3), (3, s4))):
            return
        pose = [base, s2, s3, s4, s5]
        m = predict(pose, table, size, bias)
        if m is not None and (cost := frame_cost(m)) < best[0]:
            best = (cost, pose, m)

    for s2 in FRAME_GRID[0]:
        for s3 in FRAME_GRID[1]:
            for s4 in FRAME_GRID[2]:
                tryp(s2, s3, s4)
    for step in (3, 1):     # apoi tot mai fin, în jurul celei mai bune
        if best[1] is None:
            break
        _, (_, b2, b3, b4, _), _ = best
        for d2, d3, d4 in itertools.product((-2 * step, -step, 0, step, 2 * step), repeat=3):
            tryp(b2 + d2, b3 + d3, b4 + d4)
    return best[1], best[2]


def load_jacobian():
    try:
        with open(JACOBIAN_FILE, encoding="utf-8") as f:
            J = np.array(json.load(f)["J"], float)
        return J if J.shape == (3, len(JOINTS)) and np.all(np.isfinite(J)) else None
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return None


def save_jacobian(J):
    os.makedirs(C.DATA_DIR, exist_ok=True)
    with open(JACOBIAN_FILE, "w", encoding="utf-8") as f:
        json.dump({"J": np.round(J, 6).tolist(), "joints": [NAMES[j] for j in JOINTS],
                   "rows": ["centru x", "centru y", "mărime"]}, f, indent=2, ensure_ascii=False)


class Aimer:
    """rob: Robot (move, current, sleep). get_frame(): un cadru nou de la cameră. log(text): jurnalul."""

    def __init__(self, rob, get_frame, log=print, prev=None, on_corners=None, need_cubes=0, look=None):
        self.rob, self.get_frame, self.log = rob, get_frame, log
        # look(): ce vede detecția paginii pe un cadru nou, {"corners", "cubes", "cells", "size"}. Cu ea folosim
        # aceeași urmărire a caroiajului ca în pagină (merge și când cuburile acoperă liniile); fără ea, cadre brute.
        self.look = look
        self.need_cubes = need_cubes   # caroiajul găsit de la zero trebuie să aibă cuburi (S; D poate fi gol)
        self.prev = prev        # ultimele colțuri găsite (după pași mici, caroiajul e tot pe acolo)
        self.on_corners = on_corners or (lambda corners: None)   # pagina desenează ce a găsit

    def look_metrics(self):
        """(m, e caroiaj?) din detecția paginii; m = None dacă nu vede nici caroiaj, nici cuburi."""
        o = self.look()
        self.last_look = o
        w, h = o["size"]
        grid = o["corners"] is not None and o["cells"] >= self.need_cubes
        if grid:
            c = np.float32(o["corners"])
        elif o.get("partial") is not None:
            return partial_metrics(o["partial"], w, h), False
        elif o["cubes"]:
            x, y, bw, bh = cv2.boundingRect(np.float32(np.vstack([b for b in o["cubes"]])))
            return np.array([(x + bw / 2) / w, (y + bh / 2) / h, sum(SIZE_RANGE) / 2, 0.0, (y + bh) / h]), False
        elif o.get("lines") is not None:     # doar bucăți de linie: spre ele
            return np.array([*o["lines"], sum(SIZE_RANGE) / 2, 1.0, o["lines"][1]]), False
        else:
            return None, False
        return quad_metrics(c, w, h), True

    def measure(self, tries=2):
        for _ in range(tries):
            self.rob.check()
            if self.look:
                m, _ = self.look_metrics()
                if m is not None:
                    return m
                continue
            try:
                m, corners = grid_metrics(self.get_frame(), self.prev, self.need_cubes)
            except RuntimeError:   # camera nu trimite cadre o clipă
                m = corners = None
            if m is not None:
                if corners is not None:
                    self.prev = corners
                self.on_corners(corners)
                return m
        return None

    def measure_stable(self, n=3):
        """Mediana a n măsurători: un cadru detectat greșit nu mai trage brațul după el."""
        ms = [m for m in (self.measure(tries=1) for _ in range(n)) if m is not None]
        return np.median(ms, axis=0) if len(ms) >= (n + 1) // 2 else None

    def move(self, angles, t=T_MOVE):
        for i, (lo, hi) in enumerate(LIMITS):
            angles[i] = max(lo, min(hi, angles[i]))
        self.rob.move(angles, t)
        self.rob.sleep(SETTLE_S)

    def poses(self, angles):
        """Unde caută: baza rămâne (caroiajul e în față); umărul, cotul și încheietura se mișcă, ca să privească
        mai aproape / mai departe. Întâi pozițiile apropiate de cea de start, apoi tot mai departe de ea."""
        cands = []
        for d in itertools.product(*(SWEEP[j] for j in (1, 2, 3))):
            pose = list(angles)
            for j, dj in zip((1, 2, 3), d):
                pose[j] += dj
            if any(dj for dj in d) and all(LIMITS[j][0] <= pose[j] <= LIMITS[j][1] for j in (1, 2, 3)):
                cands.append((sum(abs(dj) / SWEEP_UNIT[j] for j, dj in zip((1, 2, 3), d)), pose))
        cands.sort(key=lambda c: c[0])
        self.log(f"Caut caroiajul cu cuburi în față: mișc umărul, cotul și încheietura ({min(len(cands), SWEEP_MAX)} poziții)")
        return [pose for _, pose in cands[:SWEEP_MAX]]

    def confirmed(self, tries=3):
        """Chiar e caroiajul (cu liniile găsite și cuburi în el), nu doar niște cuburi?
        Găsit de la zero, caroiajul acoperit de cuburi iese doar pe unele cadre: încercăm pe mai multe."""
        for _ in range(tries):
            self.rob.check()
            if self.look:
                if self.look_metrics()[1]:
                    return True
                continue
            try:
                m, corners = grid_metrics(self.get_frame(), self.prev, self.need_cubes)
            except RuntimeError:
                continue
            if corners is not None:
                self.prev = corners
                self.on_corners(corners)
                return True
        return False

    def cubes_here(self):
        """Câte cuburi vede acum (cel mai mult din 2 cadre). Ieftin: ~10 ms, față de până la 2 s pentru caroiaj."""
        best = 0
        for _ in range(2):
            try:
                n = len(self.look()["cubes"]) if self.look else len(vision.find_cubes(self.get_frame())[0])
            except RuntimeError:
                continue
            best = max(best, n)
        return best

    def score(self):
        """Cât de bine se vede caroiajul (0 = perfect: la mijloc, linia de jos la BOTTOM_Y, mărime bună, întreg în
        imagine); inf = nu-l vede."""
        best = np.inf
        for _ in range(2):
            m, grid = self.look_metrics()
            if not grid:
                continue
            size = m[2]
            best = min(best, abs(m[0] - 0.5) + abs(m[4] - BOTTOM_Y) + 2 * max(0, size - SIZE_RANGE[1])
                       + 2 * max(0, SIZE_RANGE[0] - size) + 0.5 * m[3])
        return best

    def climb(self, angles):
        """Se așază să vadă caroiajul cât mai bine: încearcă pași mici în jur (baza, umărul, cotul, încheietura)
        și păstrează poziția cu cea mai bună imagine; se oprește când niciun pas nu mai ajută."""
        here = self.score()
        self.log(f"Mă așez să văd tot caroiajul (scor {here:.2f}, 0 = perfect)")
        for it in range(CLIMB_ITERS):
            if here < CLIMB_GOOD:
                break
            best, best_pose = here, None
            for j, d in CLIMB_STEPS:
                pose = list(angles)
                pose[j] += d
                if not LIMITS[j][0] <= pose[j] <= LIMITS[j][1]:
                    continue
                self.move(pose, 500)
                sc = self.score()
                if sc < best - 0.02:
                    best, best_pose = sc, pose
            if best_pose is None:
                self.log("  Niciun pas nu mai ajută.")
                break
            angles, here = best_pose, best
            self.log(f"  Pas {it + 1}: {[round(a) for a in angles]} scor {here:.2f}")
        self.move(list(angles))
        return [round(a) for a in angles] if np.isfinite(here) else None

    def solve(self, J, err, angles):
        """Cele mai mici mișcări (amortizate, cel mult MAX_STEP) care repară eroarea.
        O articulație care ar trece de limită rămâne la limită, iar celelalte preiau restul."""
        d = np.zeros(len(JOINTS))
        free = list(range(len(JOINTS)))
        for _ in range(len(JOINTS)):
            Jf = J[:, free]
            rest = err - J @ d
            df = np.linalg.solve(Jf.T @ Jf + DAMPING * np.eye(len(free)), Jf.T @ rest)
            df = np.clip(df, -MAX_STEP, MAX_STEP)
            stuck = []
            for k, v in zip(free, df):
                lo, hi = LIMITS[JOINTS[k]]
                new = angles[JOINTS[k]] + v
                if new < lo or new > hi:
                    d[k] = np.clip(new, lo, hi) - angles[JOINTS[k]]
                    stuck.append(k)
                else:
                    d[k] = v
            if not stuck:
                break
            free = [k for k in free if k not in stuck]
            if not free:
                break
            for k in free:
                d[k] = 0.0
        return d

    def grid_now(self, tries=2):
        """(m, e caroiajul întreg?) pe cadrul curent: m ca la grid_metrics, sau None dacă nu vede nimic."""
        m = None
        for _ in range(tries):
            self.rob.check()
            if self.look:
                m, grid = self.look_metrics()
            else:
                try:
                    m, corners = grid_metrics(self.get_frame(), self.prev, self.need_cubes)
                except RuntimeError:
                    continue
                grid = corners is not None
                if grid:
                    self.prev = corners
                    self.on_corners(corners)
            if grid:
                return m, True
        return m, False

    def line_score(self, m):
        return abs(m[4] - BOTTOM_Y) + abs(m[0] - 0.5) + 0.5 * m[3]

    def frame(self, angles, size=(640, 480)):
        """Zoom in / zoom out: din unde e acum (caroiajul întreg în imagine) calculează poziția cea mai bună (vezi
        FRAME_ITERS), se mută acolo și verifică; ce a greșit modelul corectează la pasul următor. Întoarce unghiurile."""
        angles = list(angles)
        for it in range(FRAME_ITERS):
            o = self.look()
            if o["corners"] is None:
                self.log("  Încadrare: nu mai văd caroiajul întreg.")
                break
            size = o["size"]
            m = quad_metrics(o["corners"], *size)
            table = fetch.grid_area(angles, o["corners"], size)
            if table is None:
                break
            pred = predict(angles, table, size, (0, 0, 1))
            if pred is None:
                break
            bias = (m[0] - pred[0], m[4] - pred[1], m[2] / max(pred[2], 1e-6))
            self.log(f"  Încadrare {it + 1}: linia de jos {m[4]:.2f}, mărime {m[2]:.2f}"
                     + ("  (iese din imagine)" if m[3] > 0.5 else ""))
            if (abs(m[4] - BOTTOM_Y) < TOLERANCE and abs(m[0] - 0.5) < TOLERANCE and m[3] < 0.5
                    and m[2] >= SIZE_RANGE[0]):
                self.log("  Caroiajul e încadrat bine.")
                break
            pose, want = best_frame(table, size, bias, angles[4])
            if pose is None:
                break
            if max(abs(a - b) for a, b in zip(pose, angles)) < MIN_STEP:
                break
            if want[3] > 0.02:
                self.log("  Caroiajul e prea aproape de braț: nu încape tot nici de la cea mai mare distanță. "
                         "Mutați foaia puțin mai departe.")
            zoom = "zoom in" if want[2] > m[2] + 0.02 else "zoom out" if want[2] < m[2] - 0.02 else "mut"
            self.log(f"  {zoom}: {[round(a) for a in pose]} (aștept linia de jos la {want[1]:.2f}, mărime {want[2]:.2f})")
            self.prev = None
            self.move(list(pose), 800)
            angles = pose
        return [round(a) for a in angles]

    @staticmethod
    def facing(center, near, far, size=(640, 480)):
        """Poziția de pe traseul liniar (umărul între near și far, baza rotită spre caroiaj) din care mijlocul imaginii
        cade cel mai aproape de punctul `center` de pe masă (m)."""
        from .panorama import look_point
        base = float(np.clip(90 + np.degrees(np.arctan2(center[1], center[0])), *LIMITS[0]))
        best, pose = np.inf, None
        lo, hi = sorted((near[1], far[1]))
        for s2 in np.arange(lo, hi + 0.01, 1.0):
            p = [base, float(s2)] + list(near[2:])
            lp = look_point(p, size)
            if lp is not None and (d := np.hypot(*(lp - center))) < best:
                best, pose = d, p
        return pose if pose is not None else list(near)

    def linear(self, near, far, steps=LINE_STEPS):
        """Căutarea liniară de la near la far (vezi LINE_NEAR). Se oprește când linia de jos a trecut de BOTTOM_Y
        sau caroiajul a ieșit din imagine. Întoarce unghiurile cele mai bune (rotunjite) sau None."""
        self.log(f"Caut caroiajul în față, liniar: {[round(a) for a in near]} → {[round(a) for a in far]} ({steps + 1} poziții)")
        best, best_pose, seen = np.inf, None, False
        part, part_pose = np.inf, None      # unde s-a văzut măcar o parte (pătratul din mijloc, câteva cuburi)
        centers = []                        # unde e caroiajul pe masă (m), după fiecare parte văzută
        for k in range(steps + 1):
            pose = [a + (b - a) * k / steps for a, b in zip(near, far)]
            self.prev = None
            self.move(pose, 1200 if k == 0 else 400)
            m, grid = self.grid_now()
            where = [round(a) for a in pose]
            if not grid:
                self.log(f"  {where}: " + ("se vede doar o parte" if m is not None else "nimic"))
                if m is not None and (sc := abs(m[4] - 0.5) + abs(m[0] - 0.5)) < part:
                    part, part_pose = sc, pose
                o = getattr(self, "last_look", None) or {}
                if o.get("partial") is not None:
                    est = fetch.grid_area(pose, o["partial"], o["size"])
                    if est is not None:
                        centers.append(np.mean(est, 0))
                if seen:
                    break       # a ieșit din imagine pe jos
                continue
            seen = True
            sc = self.line_score(m)
            self.log(f"  {where}: caroiaj, linia de jos la {m[4]:.2f} (țintă {BOTTOM_Y:.2f})"
                     + ("  (iese din imagine)" if m[3] > 0.5 else ""))
            if sc < best:
                best, best_pose = sc, pose
            if m[4] >= BOTTOM_Y:
                break           # mai departe coboară și mai mult
        if best_pose is None:
            if part_pose is None:
                return None
            if centers:
                c = np.median(centers, 0)
                best_pose = self.facing(c, near, far)
                self.log(f"Nu l-am văzut întreg; după părțile văzute e la ({c[0] * 100:.0f}, {c[1] * 100:+.0f}) cm: "
                         f"mă uit spre mijlocul lui {[round(a) for a in best_pose]} (completez din mai multe priviri).")
            else:
                self.log("Nu l-am văzut întreg; rămân unde se vede cel mai bine o parte din el (completez din mai multe priviri).")
                best_pose = part_pose
        self.move(list(best_pose))
        return [round(a) for a in best_pose]

    def run(self, search=True, center=True):
        """Întoarce unghiurile finale (servo1..5) sau None dacă nu a găsit caroiajul.
        Pornește de unde e brațul; dacă nu vede caroiajul (cu cuburi), îl caută.
        center=False: se oprește unde l-a găsit, fără să se mai așeze în mijloc."""
        here = self.rob.current()
        self.log(f"Pornesc de la {[round(a) for a in here]}")
        first = True
        for pose in [here] + (self.poses(here) if search else []):
            if not first:
                self.prev = None    # după un salt mare, colțurile vechi nu mai sunt bune
                self.move(pose, T_MOVE)
            first = False
            if not center:
                # Întâi cuburile (ieftin); caroiajul îl căutăm doar unde se văd cuburi.
                n = self.cubes_here()
                self.log(f"  {[round(a) for a in pose]}: {n} cuburi")
                if n < max(1, self.need_cubes):
                    continue
                if self.confirmed(GRID_TRIES):
                    final = [round(a) for a in pose]
                    self.log(f"Caroiajul cu cuburi e în imagine la {final}")
                    return final
                self.log("    văd cuburi, dar nu și liniile caroiajului")
                continue
            m = self.measure(tries=3)
            if m is None:
                continue
            self.log(f"Văd ceva la {[round(a) for a in pose]}: mă așez să-l văd bine")
            final = self.center(list(pose), m)
            if final is None and self.look:
                final = self.climb(list(pose))      # rezervă: probe în jur, fără Jacobian
            if final is not None and self.confirmed(GRID_TRIES):
                return final
            self.log("  Nu e caroiajul cu cuburi; caut mai departe.")
        self.log("Nu am găsit caroiajul cu cuburi.")
        return None

    def probe(self, angles, m):
        """Jacobianul măsurat: mișcă de probă fiecare articulație și vede cum se mută caroiajul (o coloană pe
        articulație). Întoarce (J, m măsurat din nou) sau (None, None) dacă a pierdut caroiajul."""
        J = np.zeros((3, len(JOINTS)))
        for k, j in enumerate(JOINTS):
            probe = list(angles)
            probe[j] += PROBE if angles[j] + PROBE <= LIMITS[j][1] else -PROBE
            self.move(probe)
            m2 = self.measure_stable()
            self.move(list(angles))
            if m2 is None:
                self.log(f"  {NAMES[j]}: caroiajul a ieșit din imagine la probă, nu folosesc articulația")
                continue
            J[:, k] = (m2 - m)[:3] / (probe[j] - angles[j])
        m = self.measure_stable()
        if m is None:
            self.log("Am pierdut caroiajul după probe.")
            return None, None
        return J, m

    def center(self, angles, m):
        """Corectează în pași mici până când caroiajul e la mijloc, la mărimea bună, apoi îl coboară în imagine până
        când linia de jos e la BOTTOM_Y (rămâne loc sus pentru cuburi). Întoarce unghiurile.
        Jacobianul vine din data/aim.json (fără probe) sau din probe; după fiecare pas e corectat cu ce s-a văzut
        (actualizare Broyden) și la final e salvat pentru data viitoare."""
        J = load_jacobian()
        if J is None:
            J, m = self.probe(angles, m)
            if J is None:
                return None
        else:
            self.log("Folosesc efectul articulațiilor ținut minte (data/aim.json): fără mișcări de probă.")
        best_err, best_angles = np.inf, list(angles)
        centered = lowered = False
        for it in range(STEPS + LOWER_STEPS):
            size_err = np.clip(m[2], *SIZE_RANGE) - m[2]     # 0 cât timp mărimea e în plajă
            if m[3] > 0.5:      # nu încape cu margine: întâi îl micșorăm
                size_err = min(size_err, -0.05)
            # Centrul și linia de jos se mută la fel cu articulațiile: Jacobianul pe centru merge și la coborâre.
            err = np.array([0.5 - m[0], (BOTTOM_Y - m[4]) if centered else (0.5 - m[1]), SIZE_WEIGHT * size_err])
            self.log(f"Pas {it + 1}: centru=({m[0]:.2f}, {m[1]:.2f}) linia de jos={m[4]:.2f} mărime={m[2]:.2f}"
                     + ("  (iese din imagine)" if m[3] > 0.5 else ""))
            if np.linalg.norm(err) < best_err:
                best_err, best_angles = np.linalg.norm(err), list(angles)
            if (np.abs(err[:2]) < TOLERANCE).all() and size_err == 0:
                if centered:
                    self.log("Caroiajul e la mijloc, cu linia de jos aproape de marginea de jos.")
                    lowered = True
                    break
                self.log(f"Caroiajul e centrat. Îl cobor în imagine: linia de jos la {BOTTOM_Y:.2f}.")
                centered = True
                save_jacobian(J)
                best_err, best_angles = np.inf, list(angles)
                continue
            if not centered and it >= STEPS:
                break
            d = self.solve(J, 0.5 * err, angles)
            if np.abs(d).max() < 1e-3:
                self.log("Nu mai pot muta caroiajul în imagine (articulații la limită sau mișcările nu-l mută).")
                break
            if np.abs(d).max() < MIN_STEP:      # aproape gata: un pas prea mic nu s-ar vedea în imagine
                d *= MIN_STEP / np.abs(d).max()
            for k, j in enumerate(JOINTS):
                angles[j] += d[k]
            self.move(list(angles))
            new = self.measure_stable()
            if new is None:
                self.log("Am pierdut caroiajul: mă întorc un pas.")
                for k, j in enumerate(JOINTS):
                    angles[j] -= d[k]
                self.move(list(angles))
                break
            # Broyden: J corectat cu efectul văzut al acestui pas (fără probe noi).
            dm = (new - m)[:3]
            dm[1] = (new - m)[4] if centered else dm[1]
            J = J + np.outer(dm - J @ d, d) / max(float(d @ d), 1e-6)
            m = new
        if not lowered:
            self.log("Nu am ajuns exact; rămân la poziția cea mai bună.")
            angles = best_angles
            self.move(list(angles))
        if centered:
            save_jacobian(J)
        final = [round(a) for a in angles]
        self.log(f"Unghiuri finale: {final}")
        return final
