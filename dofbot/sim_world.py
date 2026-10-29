"""Lumea simulată: ce cuburi sunt pe caroiajele S și D, ca brațul simulat să le poată muta.

Folosită când brațul e simulat; camera sintetică (DOFBOT_IMAGE=synth) o desenează,
deci sarcinile pot rula complet pe laptop, fără braț și fără cameră.
"""
import threading

from . import config as C


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

    def hold(self, color):
        """Brațul a luat un cub de pe masă, din afara caroiajelor."""
        with self.lock:
            if self.held is None:
                self.held = color

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
