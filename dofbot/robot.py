import json
import math
import time

from . import config as C
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

    def current(self):
        """Unghiurile servo1..5 citite de pe braț (o citire ratată se reîncearcă; altfel HOME sau 90)."""
        home = self.pos.get("HOME", [90, 90, 90, 90, 90])
        angles = []
        for i in range(5):
            a = None
            for _ in range(3):
                a = self.arm.Arm_serial_servo_read(i + 1)
                if a is not None:
                    break
                time.sleep(0.03)
            angles.append(float(a if a is not None else home[i]))
        return angles

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

    def move(self, angles, t=C.T_TRAVEL):
        """angles = [s1..s5]; gripperul rămâne cum e. Lin: puncte intermediare trimise la MOTION_HZ, cu profil
        minimum-jerk (fără smucitură la pornire și oprire), toate articulațiile ajung odată. Durata = t (unitatea
        Arm_Lib), dar nu mai repede decât permit V_MAX / A_MAX. Stop îl oprește la următorul punct (≤ 25 ms)."""
        self.check()
        target = limit(angles)
        start = self.last or self.current()
        dist = max(abs(b - a) for a, b in zip(start, target))
        n = max(1, math.ceil(duration(dist, t) * C.MOTION_HZ))
        step_ms = int(2000 / C.MOTION_HZ)      # fiecare punct: atins în două perioade, ca mișcarea să nu stea
        t0 = time.time()
        for k in range(1, n + 1):
            self.check()
            s = min_jerk(k / n)
            self._send([a + (b - a) * s for a, b in zip(start, target)], step_ms)
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

    def place(self, cell, level=1):
        self.goto(f"{cell}_sus")
        self.move(self.level_pose(cell, level), C.T_DESCEND)
        self.gripper(C.GRIP_OPEN)
        if self.sim:
            WORLD.place(cell)
        self.goto(f"{cell}_sus", C.T_DESCEND)

    def home(self):
        self.goto("HOME")
