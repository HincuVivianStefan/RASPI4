import json
import math
import time

from . import aim
from . import config as C
from . import kinematics
from .arm_driver import get_arm
from .sim_world import WORLD


# Pozițiile pe care trebuie să le înveți cu tools/calibrate_arm.py.
REQUIRED = ["HOME", "LOOK_S", "LOOK_D"] + [
    f"{g}{r}{c}_{h}" for g in "SD" for r in (1, 2, 3) for c in (1, 2, 3) for h in ("jos", "sus")
]


class Stopped(Exception):
    """Sarcina a fost oprită (butonul Stop / Reset)."""


LIMIT_SLACK = 5   # grade: o poziție cerută cu atât peste limite e o greșeală (calibrare / calcul), nu o rotunjire


def limit(angles):
    """Unghiurile servo1..5 aduse în LIMITS. Ridică ValueError dacă unul e mult în afară."""
    out = []
    for i, (a, (lo, hi)) in enumerate(zip(angles, C.LIMITS)):
        a = float(a)
        if not lo - LIMIT_SLACK <= a <= hi + LIMIT_SLACK:
            raise ValueError(f"servo {i + 1}: {a:.1f}° e în afara limitelor {lo}–{hi}")
        out.append(min(hi, max(lo, a)))
    return out


def min_jerk(s):
    """Cât din drum (0..1) la momentul s (0..1): pornește și se oprește cu viteză și accelerație zero."""
    return s * s * s * (10 - 15 * s + 6 * s * s)


def duration(dist, t):
    """Secunde pentru o mișcare de `dist` grade (cea mai lungă articulație): cel puțin t (unitatea Arm_Lib), dar
    nu mai repede decât V_MAX și A_MAX (la minimum-jerk, viteza maximă = 1,875 × dist / T, accelerația ≈ 5,77 ×
    dist / T²)."""
    return max(t * C.TIME_UNIT_S, 1.875 * dist / C.V_MAX, math.sqrt(5.77 * dist / C.A_MAX))


def load_positions():
    try:
        with open(C.POS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


class Robot:
    def __init__(self, should_stop=None, on_look=None):
        self.arm, self.sim = get_arm()
        time.sleep(0.1)
        self.pos = load_positions()
        self.grip = C.GRIP_OPEN
        self.last = None    # ultimele unghiuri trimise (servo1..5): de aici pornește următoarea mișcare
        self.should_stop = should_stop or (lambda: False)
        self.on_look = on_look or (lambda grid: None)

    def check(self):
        if self.should_stop():
            raise Stopped()

    def sleep(self, s):
        """Așteaptă s secunde, dar iese imediat (Stopped) dacă sarcina e oprită."""
        end = time.time() + (s * C.SIM_TIME_SCALE if self.sim else s)
        while time.time() < end:
            self.check()
            time.sleep(min(0.05, max(0.0, end - time.time())))

    def _wait(self, t):
        self.sleep(t * C.TIME_UNIT_S + C.SETTLE_S)

    def read(self, ids=range(1, 6)):
        """Unghiurile citite de pe braț (o citire ratată se reîncearcă); None unde nu s-a putut citi."""
        angles = []
        for sid in ids:
            a = None
            for _ in range(3):
                a = self.arm.Arm_serial_servo_read(sid)
                if a is not None:
                    break
                time.sleep(0.03)
            angles.append(None if a is None else float(a))
        return angles

    def current(self):
        """Unghiurile servo1..5 citite de pe braț (o citire ratată: HOME sau 90)."""
        home = self.pos.get("HOME", [90, 90, 90, 90, 90])
        return [float(h if a is None else a) for a, h in zip(self.read(), home)]

    def search_pose(self, grid="S"):
        """De unde începe căutarea caroiajului: SEARCH (dacă e învățată), altfel înălțimea din LOOK_<caroiaj>
        (sau SAFE_POSE), cu baza drept în față."""
        if "SEARCH" in self.pos:
            return list(self.pos["SEARCH"])
        pose = list(self.pos.get(f"LOOK_{grid}") or C.SAFE_POSE)
        pose[0] = aim.SEARCH_CENTER
        return pose

    def safe(self):
        """Poziția sigură = search_pose(): brațul ca cifra 2, gata de căutat caroiajul. Întâi ridică umărul / cotul / încheietura cu
        baza pe loc (să nu măture cuburile), apoi rotește baza. Gripperul rămâne cum e (poate ține un cub).
        Un servo care nu poate fi citit (de ex. trecut de capătul cursei) e adus încet, dintr-o comandă, nu printr-un salt."""
        self.check()
        target = limit(self.search_pose())
        grip = self.read([6])[0]
        if grip is not None:
            self.grip = round(grip)
        got = self.read()
        if None in got:
            self._send(target, C.SAFE_SLOW_MS)
            self.last = target
            self.sleep(C.SAFE_SLOW_MS * C.TIME_UNIT_S)
            self.settle(target)
            return
        self.last = limit(got)
        self.move([got[0]] + target[1:4] + [got[4]])
        self.move(target)

    def freeze(self):
        """Oprește brațul unde a ajuns: citește unghiurile din mers și i le trimite înapoi."""
        a = self.current()
        self._send(a, 300)
        self.last = a

    def _get(self, name):
        if name in self.pos:
            return self.pos[name]
        if self.sim:
            # Pe laptop nu ai poziții calibrate: folosim una fictivă ca să meargă logica.
            if C.SIM_VERBOSE:
                print(f"  [SIM] poziția '{name}' nu e calibrată, folosesc [90]*5")
            return [90, 90, 90, 90, 90]
        raise KeyError(f"Poziția '{name}' nu e calibrată (rulează calibrate_arm.py)")

    def move(self, angles, t=C.T_TRAVEL, via=()):
        """angles = [s1..s5]; gripperul rămâne cum e. Lin: puncte intermediare trimise la MOTION_HZ, cu profil
        minimum-jerk (fără smucitură la pornire și oprire), toate articulațiile ajung odată. Durata = t (unitatea
        Arm_Lib), dar nu mai repede decât permit V_MAX / A_MAX. via = unghiuri prin care trece pe drum (de ex.
        kinematics.line: gripperul în linie dreaptă), tot cu un singur profil, fără opriri. Stop îl oprește la
        următorul punct (≤ 25 ms)."""
        self.check()
        target = limit(angles)
        pts = [self.last or self.current()] + [limit(v) for v in via or ()] + [target]
        seg = [max(abs(b - a) for a, b in zip(p, q)) for p, q in zip(pts, pts[1:])]
        dist = sum(seg)
        n = max(1, math.ceil(duration(dist, t) * C.MOTION_HZ))
        step_ms = int(2000 / C.MOTION_HZ)      # fiecare punct: atins în două perioade, ca mișcarea să nu stea
        t0 = time.time()
        for k in range(1, n + 1):
            self.check()
            d, i = min_jerk(k / n) * dist, 0       # cât din drum (grade) și pe a câta bucată a lui
            while i < len(seg) - 1 and d > seg[i]:
                d -= seg[i]
                i += 1
            s = min(1.0, d / seg[i]) if seg[i] else 1.0
            self._send([a + (b - a) * s for a, b in zip(pts[i], pts[i + 1])], step_ms)
            if not self.sim or C.SIM_TIME_SCALE:
                wait = t0 + k / C.MOTION_HZ * (C.SIM_TIME_SCALE if self.sim else 1) - time.time()
                if wait > 0:
                    time.sleep(wait)
        self.last = target
        if self.sim and C.SIM_VERBOSE:
            print(f"  [SIM] servo1-5 -> {[round(a, 1) for a in target]}  ({n} puncte)")
        self.settle(target)

    def _send(self, angles, ms):
        a = [round(float(x), 1) for x in angles]   # Arm_Lib primește și zecimale (~0,08° pe pas)
        self.arm.Arm_serial_servo_write6(a[0], a[1], a[2], a[3], a[4], self.grip, ms)

    def settle(self, target):
        """Așteaptă să ajungă brațul (servourile citite la SETTLE_TOL de țintă, cel mult SETTLE_MAX_S), apoi SETTLE_S.
        Pe brațul simulat, doar SETTLE_S."""
        if not self.sim:
            end = time.time() + C.SETTLE_MAX_S
            while time.time() < end:
                self.check()
                got = [self.arm.Arm_serial_servo_read(i + 1) for i in range(5)]
                if all(g is None or abs(g - a) <= C.SETTLE_TOL for g, a in zip(got, target)):
                    break
                time.sleep(0.02)
        self.sleep(C.SETTLE_S)

    def goto(self, name, t=C.T_TRAVEL):
        if self.sim and C.SIM_VERBOSE:
            print(f"  -> {name}")
        self.move(self._get(name), t)
        if name.startswith("LOOK_"):
            grid = name[5:]
            if self.sim:
                WORLD.look(grid)
            self.on_look(grid)

    def gripper(self, value):
        self.check()
        self.grip = value
        self.arm.Arm_serial_servo_write(6, value, C.T_GRIP)
        self._wait(C.T_GRIP)

    def level_pose(self, cell, level):
        """Poziția gripperului pentru un cub aflat la nivelul `level` (1 = pe masă) în `cell`."""
        exact = f"{cell}_n{level}"
        if exact in self.pos:
            return self.pos[exact]
        jos = self._get(f"{cell}_jos")
        sus = self._get(f"{cell}_sus")
        # Interpolare liniară în unghiuri: aproximativă! Verifică și, la nevoie,
        # învață poziții exacte <cell>_n<level>.
        k = (level - 1) / (C.SUS_LEVEL - 1)
        return [j + (s - j) * k for j, s in zip(jos, sus)]

    def pick(self, cell, level=1):
        self.gripper(C.GRIP_OPEN)
        self.goto(f"{cell}_sus")
        self.move(self.level_pose(cell, level), C.T_DESCEND)
        self.gripper(C.GRIP_CLOSE)
        if self.sim and WORLD.pick(cell) is None:
            print(f"  [SIM] atenție: nu era niciun cub de luat în {cell}")
        self.goto(f"{cell}_sus", C.T_DESCEND)

    def descend(self, above, down):
        """De deasupra, jos: gripperul în linie dreaptă (nu pe curba dintre unghiuri, care poate atinge foaia),
        ultimii FINAL_DZ m încet (T_FINAL)."""
        g = kinematics.grip(down)[0]
        near = kinematics.ik(g + [0, 0, C.FINAL_DZ], math.degrees(kinematics.pitches(down)[2]), down[4])
        if near is None:
            self.move(down, C.T_DESCEND, kinematics.line(above, down))
            return
        self.move(near, C.T_DESCEND, kinematics.line(above, near))
        self.move(down, C.T_FINAL)

    def grab_at(self, above, down):
        """Ia un cub de oriunde de pe masă (nu dintr-o celulă): above / down = unghiurile calculate de fetch.py.
        Întoarce True dacă cubul e în gripper."""
        self.gripper(C.GRIP_OPEN)
        self.move(above)
        self.descend(above, down)
        self.gripper(C.GRIP_CLOSE)
        if self.sim:
            WORLD.pick_at(*kinematics.grip(down))
        held = self.holding()
        self.move(above, C.T_DESCEND, kinematics.line(down, above))
        return held

    def place_at(self, above, down, cell=None):
        """Pune cubul din gripper oriunde pe masă: above / down = unghiurile calculate de fetch.py; cell = celula
        caroiajului, dacă acolo îl pune (None = pe masă, de ex. pe locul de îndreptare). Drumul până deasupra: la
        înălțimea cea mai mare dintre plecare și sosire (kinematics.transit), deci cubul nu trece jos peste cuburile
        puse; coborârea și urcarea în linie dreaptă; gripperul se deschide doar până la GRIP_RELEASE (larg împinge
        cubul)."""
        start = self.last or self.current()
        self.move(above, C.T_TRAVEL, kinematics.transit(start, limit(above)))
        self.descend(above, down)
        self.gripper(C.GRIP_RELEASE)
        if self.sim and cell:
            WORLD.place(cell)
        elif self.sim:
            WORLD.drop_at(*kinematics.grip(down))
        self.move(above, C.T_DESCEND, kinematics.line(down, above))

    def holding(self):
        """E un cub în gripper? Pe brațul real: servo 6 s-a oprit înainte de GRIP_CLOSE (cubul îl ține)."""
        if self.sim:
            return WORLD.held is not None
        a = self.arm.Arm_serial_servo_read(6)
        return a is None or a < C.GRIP_CLOSE - C.FETCH_EMPTY_TOL

    def place(self, cell, level=1):
        self.goto(f"{cell}_sus")
        self.move(self.level_pose(cell, level), C.T_DESCEND)
        self.gripper(C.GRIP_OPEN)
        if self.sim:
            WORLD.place(cell)
        self.goto(f"{cell}_sus", C.T_DESCEND)

    def home(self):
        self.goto("HOME")
