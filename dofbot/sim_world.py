"""Lumea simulată: ce cuburi sunt pe caroiajele S și D, ca brațul simulat să le poată muta.

Folosită când brațul e simulat; camera sintetică (DOFBOT_IMAGE=synth) o desenează,
deci sarcinile pot rula complet pe laptop, fără braț și fără cameră.

arm_view: ce vede camera de pe brațul simulat, din unghiurile lui (sursa „synth-arm” din pagină). Caroiajele
stau pe masă la locurile din PLACES; așa căutarea și centrarea caroiajului (aim.py) se pot încerca fără braț.
"""
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

    def snapshot(self, grid=None):
        """(culoarea de sus, înălțimea) pentru fiecare celulă a caroiajului."""
        with self.lock:
            g = self.grids[grid or self.view]
            return [[(s[-1] if s else None, len(s)) for s in row] for row in g]


WORLD = SimWorld()


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
    """Imaginea camerei de pe brațul simulat, cu caroiajele S și D și cuburile din WORLD."""
    scenes = [(WORLD.snapshot(g), *camera_matrix(g, angles, size)) for g in PLACES]
    return synth.render_scenes(scenes, size, seed=seed, **kw)
