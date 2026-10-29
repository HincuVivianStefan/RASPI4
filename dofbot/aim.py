"""Brațul își caută singur caroiajul și se așază ca să-l vadă cât mai bine (cu tot cu cuburi).

1. Dacă nu vede caroiajul, rotește baza pe toată cursa (la înălțimea actuală, apoi puțin mai aproape și
   mai departe) până îl găsește.
2. Mișcă de probă baza, umărul, cotul și încheietura câte câteva grade și vede cum se mută caroiajul în imagine.
3. Corectează în pași mici până când caroiajul e în mijloc, mare, dar cu margine în jur: fețele de sus ale
   cuburilor cad deasupra celulelor în imagine și trebuie să încapă și ele.

Fiecare mișcare trece prin Robot, deci Stop din pagină sau orice buton de pe controller o oprește imediat.
Folosit din pagină (butonul „Caută caroiajul”, sarcinile cu AUTO_AIM) și din tools/center_grid.py.
"""
import itertools

import cv2
import numpy as np

from . import vision

JOINTS = [0, 1, 2, 3]       # baza, umărul, cotul, încheietura (indici 0..4 în unghiuri)
NAMES = ["baza", "umărul", "cotul", "încheietura", "rotația"]
PROBE = 3                   # grade pentru mișcarea de probă
MAX_STEP = 4                # grade, cel mult, la o corecție
SIZE_RANGE = (0.40, 0.62)   # cât din imagine să ocupe caroiajul (radical din arie): cu margine, dar mare
TOLERANCE = 0.05            # cât de aproape de mijloc e destul (din lățimea / înălțimea imaginii)
DAMPING = 3e-5              # frânează corecțiile pe care articulațiile le pot face doar cu mișcări mari
SETTLE_S = 0.8              # cât mai așteptăm după o mișcare, să se oprească brațul și camera
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
SIZE_WEIGHT = 1.0           # cât contează mărimea față de centrare
STEPS = 15                  # pași de corecție, cel mult
MIN_CUBES = 1               # fără linii vizibile, grupul de cuburi ține loc de caroiaj (apoi se verifică)
LIMITS = [(0, 180), (0, 180), (0, 180), (0, 180), (0, 270)]


def grid_metrics(frame, prev=None, need_cubes=0):
    """(centru x, centru y, mărime, 1 dacă un colț e prea aproape de margine), ca fracțiuni din imagine,
    plus colțurile caroiajului (sau None dacă a folosit cuburile). (None, None) dacă nu vede nimic.
    prev = colțurile de la măsurătoarea anterioară: cu ele e găsit și caroiajul acoperit de cuburi.
    need_cubes: găsit de la zero (fără prev), caroiajul trebuie să aibă atâtea cuburi în celule."""
    corners = vision.find_grid(frame, prev)
    if corners is not None and prev is None and need_cubes and vision.cubes_inside(frame, corners) < need_cubes:
        corners = None      # o grilă fără cuburi: probabil altceva de pe masă / de pe jos
    if corners is not None:
        c = np.float32(corners)
    else:
        # Caroiaj plin: liniile se văd prea puțin. Dreptunghiul care cuprinde toate cuburile e aproape caroiajul.
        cubes, _ = vision.find_cubes(frame)
        if len(cubes) < MIN_CUBES:
            return None, None
        x, y, bw, bh = cv2.boundingRect(np.float32(np.vstack([o["box"] for o in cubes])))
        c = np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]])
    h, w = frame.shape[:2]
    cx, cy = c.mean(0)
    if corners is None:     # doar cuburile: le aducem în mijloc, dar mărimea lor nu spune cât e caroiajul
        return np.array([cx / w, cy / h, sum(SIZE_RANGE) / 2, 0.0]), None
    size = np.sqrt(cv2.contourArea(c) / (w * h))
    rel = c / [w, h]
    out = float((rel < EDGE_MARGIN).any() or (rel > 1 - EDGE_MARGIN).any())
    return np.array([cx / w, cy / h, size, out]), corners


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
        w, h = o["size"]
        grid = o["corners"] is not None and o["cells"] >= self.need_cubes
        if grid:
            c = np.float32(o["corners"])
        elif o["cubes"]:
            x, y, bw, bh = cv2.boundingRect(np.float32(np.vstack([b for b in o["cubes"]])))
            return np.array([(x + bw / 2) / w, (y + bh / 2) / h, sum(SIZE_RANGE) / 2, 0.0]), False
        else:
            return None, False
        cx, cy = c.mean(0)
        rel = c / [w, h]
        out = float((rel < EDGE_MARGIN).any() or (rel > 1 - EDGE_MARGIN).any())
        return np.array([cx / w, cy / h, np.sqrt(cv2.contourArea(c) / (w * h)), out]), True

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
        """Cât de bine se vede caroiajul (0 = perfect: în mijloc, mărime bună, întreg în imagine); inf = nu-l vede."""
        best = np.inf
        for _ in range(2):
            m, grid = self.look_metrics()
            if not grid:
                continue
            size = m[2]
            best = min(best, abs(m[0] - 0.5) + abs(m[1] - 0.5) + 2 * max(0, size - SIZE_RANGE[1])
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
            final = self.climb(list(pose)) if self.look else self.center(list(pose), m)
            if final is not None and self.confirmed(GRID_TRIES):
                return final
            self.log("  Nu e caroiajul cu cuburi; caut mai departe.")
        self.log("Nu am găsit caroiajul cu cuburi.")
        return None

    def center(self, angles, m):
        """Corectează în pași mici până când caroiajul e în mijloc, la mărimea bună. Întoarce unghiurile."""
        # Cum mută fiecare articulație caroiajul în imagine (o coloană pe articulație)
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
            return None
        best_err, best_angles = np.inf, list(angles)

        for it in range(STEPS):
            size_err = np.clip(m[2], *SIZE_RANGE) - m[2]     # 0 cât timp mărimea e în plajă
            if m[3] > 0.5:      # nu încape cu margine: întâi îl micșorăm
                size_err = min(size_err, -0.05)
            err = np.array([0.5 - m[0], 0.5 - m[1], SIZE_WEIGHT * size_err])
            self.log(f"Pas {it + 1}: centru=({m[0]:.2f}, {m[1]:.2f}) mărime={m[2]:.2f}"
                     + ("  (prea aproape de margine)" if m[3] > 0.5 else ""))
            if np.linalg.norm(err) < best_err:
                best_err, best_angles = np.linalg.norm(err), list(angles)
            if (np.abs(err[:2]) < TOLERANCE).all() and size_err == 0:
                self.log("Caroiajul e centrat.")
                break
            d = self.solve(J, 0.5 * err, angles)
            if np.abs(d).max() < 0.5:
                self.log("Nu mai pot apropia caroiajul de mijloc (articulații la limită sau mișcările nu-l mută în imagine).")
                break
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
            m = new
        else:
            self.log(f"Nu am ajuns exact în {STEPS} pași; rămân la poziția cea mai bună.")
            angles = best_angles
            self.move(list(angles))
        final = [round(a) for a in angles]
        self.log(f"Unghiuri finale: {final}")
        return final
