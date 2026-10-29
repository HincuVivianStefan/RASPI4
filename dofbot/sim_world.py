"""Lumea simulată: ce cuburi sunt pe caroiajele S și D, ca brațul simulat să le poată muta.

Folosită când brațul e simulat; camera sintetică (DOFBOT_IMAGE=synth) o desenează,
deci sarcinile pot rula complet pe laptop, fără braț și fără cameră.

arm_view: ce vede camera de pe brațul simulat, din unghiurile lui (sursa „synth-arm” din pagină). Caroiajele
stau pe masă la locurile din PLACES; așa căutarea și centrarea caroiajului (aim.py) se pot încerca fără braț.
Cuburile pot sta și oriunde pe masă (`table`): scatter() le mută de pe S într-un „unghi mort”, iar pick_at() le
ia de acolo dacă gripperul ajunge pe ele (fetch.py).
"""
import random
import threading

import numpy as np

from . import config as C
from . import kinematics
from . import synth

# Unde stau caroiajele pe masă: (unghiul față de „înainte”, în grade, pozitiv la stânga; distanța de la bază până
# la centrul caroiajului, m). Rândul 3 e cel mai aproape de braț, coloana 1 la stânga, văzute din braț.
PLACES = {"S": (0.0, 0.24), "D": (-75.0, 0.24)}
CELL_M = 0.045      # latura unei celule (cubul are synth.SIDE din ea, ~3 cm)
FOCAL = 520         # focala camerei simulate, în pixeli (la 640 lățime, ~63° câmp vizual)
# Unde poate fi „unghiul mort”: (grade față de înainte, pozitiv la stânga), în afara caroiajelor S (0°) și D (-75°);
# distanța de la bază, m.
DEAD_ZONES = ((30.0, 70.0), (-50.0, -28.0))
DEAD_DIST = (0.15, 0.21)
NEAREST, FARTHEST = 0.125, 0.25   # m: brațul ajunge la cuburile dintre atâția metri de bază
PICK_TOL = 0.012    # m: cât de departe de centrul cubului poate ajunge gripperul ca să-l prindă
PICK_ANGLE = 25     # grade: cât de strâmb față de fețele cubului se pot închide degetele


class SimWorld:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self, layout=None):
        layout = layout or C.SYNTH_LAYOUT
        with self.lock:
            # Fiecare celulă e o stivă de culori, de jos în sus.
            self.grids = {
                "S": [[[layout[r][c]] if layout[r][c] else [] for c in range(3)] for r in range(3)],
                "D": [[[] for _ in range(3)] for _ in range(3)],
            }
            self.view = "S"      # ce caroiaj "vede" camera (ultimul LOOK_x)
            self.held = None     # cubul din gripper
            self.table = []      # cuburile de pe masă, în afara caroiajelor: {x, y (m), yaw (grade), color}

    def look(self, grid):
        with self.lock:
            self.view = grid

    def _stack(self, cell):
        g, r, c = cell[0], int(cell[1]) - 1, int(cell[2]) - 1
        return self.grids[g][r][c]

    def pick(self, cell):
        """Întoarce culoarea luată sau None dacă celula e goală."""
        with self.lock:
            stack = self._stack(cell)
            if self.held is not None or not stack:
                return None
            self.held = stack.pop()
            return self.held

    def place(self, cell):
        with self.lock:
            if self.held is None:
                return None
            color, self.held = self.held, None
            self._stack(cell).append(color)
            return color

    def scatter(self, seed=None, packed=None):
        """Mută cuburile de pe S într-un loc la întâmplare de pe masă (unghiul mort), rotite oricum.
        packed = lipite unele de altele (altfel împrăștiate cu loc între ele); None = la întâmplare."""
        rng = random.Random(seed)
        packed = rng.random() < 0.5 if packed is None else packed
        side = synth.SIDE * CELL_M
        with self.lock:
            colors = [c for row in self.grids["S"] for stack in row for c in stack]
            self.grids["S"] = [[[] for _ in range(3)] for _ in range(3)]
            spots = []
            while len(spots) < len(colors):     # alt loc, dacă în cel ales nu încap toate
                a = np.radians(rng.uniform(*rng.choice(DEAD_ZONES)))
                center = rng.uniform(*DEAD_DIST) * np.array([np.cos(a), np.sin(a)])
                spots = []
                if packed:   # un rând de cuburi lipite (cu 1-3 mm între ele), apoi încă unul lângă
                    turn = rng.uniform(-45, 45)
                    u = np.array([np.cos(np.radians(turn)), np.sin(np.radians(turn))])
                    v = np.array([-u[1], u[0]])
                    for k in range(len(colors)):
                        gap = side + rng.uniform(0.001, 0.003)
                        spots.append((center + ((k % 3) - 1) * gap * u + (k // 3) * gap * v, turn + rng.uniform(-4, 4)))
                    if not all(_free(q, side) for q, _ in spots):
                        spots = []
                    continue
                for _ in range(400):    # cu loc între ele: ~2,5 cm
                    p = center + rng.uniform(0, 0.06) * np.array([np.cos(t := rng.uniform(0, 6.3)), np.sin(t)])
                    if _free(p, side) and all(np.linalg.norm(p - q) > 1.8 * side for q, _ in spots):
                        spots.append((p, rng.uniform(-45, 45)))
                        if len(spots) == len(colors):
                            break
            self.table = [{"x": float(p[0]), "y": float(p[1]), "yaw": float(t), "color": c}
                          for c, (p, t) in zip(colors, spots)]
            return packed

    def pick_at(self, point, closing):
        """Gripperul se închide în `point` (m), degetele pe direcția `closing`: ia cubul de pe masă dacă e prins
        bine (centrul aproape și degetele pe fețe). Întoarce culoarea sau None."""
        side = synth.SIDE * CELL_M
        with self.lock:
            if self.held is not None or not 0 < point[2] < side:
                return None
            best = min(self.table, key=lambda o: np.hypot(o["x"] - point[0], o["y"] - point[1]), default=None)
            if best is None or np.hypot(best["x"] - point[0], best["y"] - point[1]) > PICK_TOL:
                return None
            ang = np.degrees(np.arctan2(closing[1], closing[0])) - best["yaw"]
            if abs((ang + 45) % 90 - 45) > PICK_ANGLE:
                return None
            self.table.remove(best)
            self.held = best["color"]
            return self.held

    def snapshot(self, grid=None):
        """(culoarea de sus, înălțimea) pentru fiecare celulă a caroiajului."""
        with self.lock:
            g = self.grids[grid or self.view]
            return [[(s[-1] if s else None, len(s)) for s in row] for row in g]


WORLD = SimWorld()


def _free(p, side):
    """Un cub pus cu centrul în p (m) e la îndemâna brațului și nu atinge niciun caroiaj."""
    if not NEAREST < np.linalg.norm(p) < FARTHEST:
        return False
    m = side / CELL_M     # marginea, în celule
    for g in PLACES:
        x, y = (np.linalg.inv(cells_to_arm(g)) @ [p[0], p[1], 0, 1])[:2]
        if -m < x < 3 + m and -m < y < 3 + m:
            return False
    return True


def cells_to_arm(grid):
    """Matricea 4x4: coordonate de caroiaj (x = coloane, y = rânduri, z în sus, în celule) → sistemul brațului (m)."""
    yaw, dist = PLACES[grid]
    c, a = CELL_M, np.radians(yaw)
    # Rândurile spre braț (x scade), coloanele spre dreapta brațului (y scade); apoi totul rotit cu yaw.
    local = np.array([[0, -c, 0, dist + 1.5 * c], [-c, 0, 0, 1.5 * c], [0, 0, c, 0], [0, 0, 0, 1.0]])
    rot = np.array([[np.cos(a), -np.sin(a), 0, 0], [np.sin(a), np.cos(a), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    return rot @ local


def camera_matrix(grid, angles, size=(640, 480)):
    """(P 3x4: coordonate de caroiaj → imagine, poziția camerei în coordonate de caroiaj) pentru brațul la `angles`."""
    pos, R = kinematics.camera_pose(angles)
    K = np.array([[FOCAL * size[0] / 640, 0, size[0] / 2], [0, FOCAL * size[0] / 640, size[1] / 2], [0, 0, 1.0]])
    M = cells_to_arm(grid)
    P = K @ np.column_stack([R, -R @ pos]) @ M
    eye = (np.linalg.inv(M) @ np.append(pos, 1.0))[:3]
    return P, eye


def true_corners(grid, angles, size=(640, 480)):
    """Colțurile caroiajului în imaginea camerei de pe braț (pot fi și în afara imaginii). Pentru teste."""
    P, _ = camera_matrix(grid, angles, size)
    p = np.array([[0, 0, 0, 1], [3, 0, 0, 1], [3, 3, 0, 1], [0, 3, 0, 1.0]]) @ P.T
    return (p[:, :2] / p[:, 2:]).tolist()


def arm_view(angles, size=(640, 480), seed=0, **kw):
    """Imaginea camerei de pe brațul simulat, cu caroiajele S și D și cuburile din WORLD (și cele de pe masă)."""
    scenes = [(WORLD.snapshot(g), *camera_matrix(g, angles, size)) for g in PLACES]
    # Cuburile de pe masă, în coordonatele caroiajului S (celule).
    inv = np.linalg.inv(cells_to_arm("S"))
    with WORLD.lock:
        table = list(WORLD.table)
    loose = []
    for o in table:
        x, y = (inv @ [o["x"], o["y"], 0, 1])[:2]
        t = np.radians(o["yaw"])
        dx, dy = inv[:2, :2] @ [np.cos(t), np.sin(t)]
        loose.append((x, y, o["color"], float(np.degrees(np.arctan2(dy, dx)))))
    P, eye = scenes[list(PLACES).index("S")][1:]
    return synth.render_scenes(scenes, size, seed=seed, loose=(P, eye, loose), **kw)
