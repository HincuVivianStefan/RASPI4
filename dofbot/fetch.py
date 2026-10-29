"""Sarcina 1, partea a doua: cuburile luate de pe caroiaj și puse într-un loc pe care camera nu-l vedea („unghiul
mort”) sunt căutate, luate unul câte unul și puse înapoi în celulele memorate.

1. Căutarea: baza se rotește stânga / dreapta, pe două rânduri (de unde se vede caroiajul și puțin mai aproape).
   Petele de culoare sunt duse pe masă cu poza camerei de pe gripper, calculată din unghiurile brațului
   (kinematics.py), și măsurate în metri: un cub e recunoscut oricât de aproape ar fi camera.
2. Alegerea: un cub de o culoare care mai trebuie pusă, cel cu cel mai mult loc liber în jur (cuburile pot fi
   lipite); degetele se închid pe direcția pe care au mai mult loc.
3. Apropierea: camera ajunge deasupra cubului (cinematica inversă), îl vede de aproape și poziția se
   corectează până nu se mai schimbă; servo 5 rotește gripperul după cub; coboară, prinde, urcă.
4. Plasarea: în celula memorată, din pozițiile învățate (Robot.place).

Pe brațul real, greșelile constante (lungimea gripperului, camera montată puțin altfel) se corectează din config:
GRIP_LEN, FETCH_OFFSET, FETCH_ROLL_SIGN.
"""
import cv2
import numpy as np

from . import config as C
from . import kinematics
from . import vision

SCAN_STEP = 15          # grade între opririle bazei la căutare
SCAN_SPAN = 75          # cât de departe stânga / dreapta de poziția de privit caroiajul
SCAN_NEAR = -28         # al doilea rând: umărul cu atâtea grade mai în față (vede masa mai aproape de braț)
VIEW_DIST = (0.12, 0.16)  # m: de cât de aproape privește camera cubul, drept de sus (dacă nu ajunge, de mai departe)
SETTLE_S = 0.3          # cât stă camera după o mișcare, înainte de cadru
ITER = 5                # de câte ori, cel mult, se corectează poziția cubului privit de aproape
LOST = 0.04             # m: privit de aproape, cubul e căutat în atâta jurul poziției știute
SAME = 0.6              # două detecții de aceeași culoare la mai puțin de atâtea laturi = același cub
GRID_MARGIN = 0.005     # m: cuburile cu centrul pe caroiaj (sau la atât de el) sunt deja puse, nu se iau
BORDER = 3              # px: o pată care atinge marginea imaginii e tăiată, deci nu e măsurată


def to_table(angles, pts, size, z):
    """Puncte din imagine → puncte pe planul orizontal de la înălțimea z (m, sistemul brațului).
    Întoarce (xy, ok): ok = punctul e chiar în fața camerei, sub ea."""
    pos, R = kinematics.camera_pose(angles)
    w, h = size
    f = vision.focal(w)
    pts = np.float64(pts).reshape(-1, 2)
    rays = np.column_stack([(pts[:, 0] - w / 2) / f, (pts[:, 1] - h / 2) / f, np.ones(len(pts))]) @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (z - pos[2]) / rays[:, 2]
    return (pos + rays * k[:, None])[:, :2], k > 0


def table_cubes(frame, angles):
    """Cuburile văzute în cadru, puse pe masă: dict cu color, xy (m), yaw (grade, direcția unei laturi),
    px (centrul în imagine), off (cât de departe de mijlocul imaginii, 0..1)."""
    small, s = vision.shrink(frame)
    small = vision.normalize_light(small)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    a = C.FETCH_CUBE_M
    kernel = np.ones((3, 3), np.uint8)
    out = []
    for name in C.COLORS:
        m = cv2.morphologyEx(vision.color_mask(hsv, name), cv2.MORPH_OPEN, kernel)
        for cnt in vision.split_touching(m):
            if cv2.contourArea(cnt) < 30:
                continue
            img = cnt.reshape(-1, 2).astype(np.float64) / s
            if (img.min(0) < BORDER).any() or (img.max(0) > [w - BORDER, h - BORDER]).any():
                continue
            pts, ok = to_table(angles, img, (w, h), a)
            if not ok.all():
                continue
            mm = np.float32(pts * 1000)          # minAreaRect în milimetri (numere mai mari decât în metri)
            rect = cv2.minAreaRect(mm)
            (_, _), (bw, bh), _ = rect
            area = cv2.contourArea(mm)
            side = np.sqrt(area) / C.CUBE_TOP / 1000
            if not (0.6 * a <= side <= 1.5 * a and max(bw, bh) / max(min(bw, bh), 1e-6) <= C.CUBE_MAX_ASPECT
                    and area / max(bw * bh, 1e-6) >= C.CUBE_MIN_FILL):
                continue
            box = cv2.boxPoints(rect)
            edge = box[1] - box[0]
            center = np.float64(rect[0]) / 1000
            px = img.mean(0)
            out.append({"color": name, "xy": center.tolist(), "yaw": float(np.degrees(np.arctan2(edge[1], edge[0]))),
                        "px": px.tolist(), "off": float(np.abs(px / [w, h] - 0.5).max() * 2)})
    return out


def grid_area(angles, corners, size):
    """Colțurile caroiajului pe masă (m), din colțurile lui în imagine și unghiurile brațului. None dacă nu se poate."""
    if corners is None:
        return None
    pts, ok = to_table(angles, corners, size, 0.0)
    return pts.tolist() if ok.all() else None


def wrap(a, period=90):
    """Unghiul adus în [-period/2, period/2)."""
    return (a + period / 2) % period - period / 2


class Fetcher:
    def __init__(self, rob, get_frame, log=print, grid=None):
        self.rob, self.get_frame, self.log = rob, get_frame, log
        self.grid = None if grid is None else np.float32(grid)   # caroiajul pe masă (m): cuburile de acolo stau
        self.far = []      # unde sunt cuburile la care brațul nu ajunge (m): nu se mai încearcă

    def on_grid(self, xy):
        if self.grid is None:
            return False
        return cv2.pointPolygonTest(self.grid * 1000, (float(xy[0]) * 1000, float(xy[1]) * 1000), True) \
            > -GRID_MARGIN * 1000

    def view(self, pose, t=600):
        self.rob.move(pose, t)
        self.rob.sleep(SETTLE_S)
        return [c for c in table_cubes(self.get_frame(), pose) if not self.on_grid(c["xy"])]

    def scan(self, look):
        """Caută cuburile în jur, pe două rânduri. Același cub văzut de mai multe ori se păstrează o dată, de unde
        a fost văzut cel mai aproape de mijlocul imaginii."""
        near = list(look)
        near[1] = min(max(near[1] + SCAN_NEAR, C.LIMITS[1][0]), C.LIMITS[1][1])
        bases = [b for b in range(round(look[0]) - SCAN_SPAN, round(look[0]) + SCAN_SPAN + 1, SCAN_STEP)
                 if C.LIMITS[0][0] <= b <= C.LIMITS[0][1]]
        found, first = [], True
        for row, pose0 in enumerate((look, near)):
            for b in (bases if row == 0 else bases[::-1]):      # șerpuit: fără drum înapoi între rânduri
                pose = list(pose0)
                pose[0] = b
                for c in self.view(pose, 1500 if first else 600):
                    same = [o for o in found if o["color"] == c["color"]
                            and np.hypot(*np.subtract(o["xy"], c["xy"])) < SAME * C.FETCH_CUBE_M]
                    if not same:
                        found.append(c)
                    elif c["off"] < same[0]["off"]:
                        same[0].update(c)
                first = False
        return found

    @staticmethod
    def free(cube, cubes):
        """Locul liber din jurul cubului: distanța până la cel mai apropiat alt cub (m)."""
        return min((np.hypot(*np.subtract(o["xy"], cube["xy"])) for o in cubes if o is not cube), default=1.0)

    @staticmethod
    def roll(cube, cubes, pitch):
        """Rotația gripperului (grade față de servo 5 = 90): degetele pe fețele cubului, pe direcția cu mai mult
        loc. Cu gripperul înclinat, degetele rămân aproape orizontale (rotația cea mai mică)."""
        theta = np.degrees(np.arctan2(cube["xy"][1], cube["xy"][0]))
        options = []
        for axis in (cube["yaw"], cube["yaw"] + 90):
            u = np.array([np.cos(np.radians(axis)), np.sin(np.radians(axis))])
            clear = 1.0
            for o in cubes:
                if o is cube:
                    continue
                v = np.subtract(o["xy"], cube["xy"])
                if abs(v @ [-u[1], u[0]]) < 1.1 * C.FETCH_CUBE_M:    # vecinul e în calea degetelor
                    clear = min(clear, abs(v @ u))
            options.append((clear, wrap(axis - theta, 180)))
        if pitch < 165:
            return min((r for _, r in options), key=abs)
        return max(options, key=lambda o: (round(o[0], 3), -abs(o[1])))[1]

    def poses(self, xy, roll=0.0):
        """(prindere, deasupra, înclinarea) pentru un cub cu centrul în xy, sau None dacă nu ajunge."""
        xy = np.float64(xy)
        u = xy / max(np.linalg.norm(xy), 1e-9)
        xy = xy + C.FETCH_OFFSET[0] * u + C.FETCH_OFFSET[1] * np.array([-u[1], u[0]])
        return kinematics.reach([xy[0], xy[1], C.FETCH_CUBE_M / 2], C.FETCH_LIFT,
                                90 + C.FETCH_ROLL_SIGN * roll)

    def approach(self, cube):
        """Privește cubul de deasupra și îi corectează poziția până nu se mai schimbă. Întoarce cubul văzut de
        aproape (sau None: nu ajunge / nu-l mai vede)."""
        cur = dict(cube)
        for _ in range(ITER):
            sol = self.poses(cur["xy"])
            if sol is None:
                self.log(f"  {cube['color']} la {np.round(cur['xy'], 3).tolist()} m: nu ajung la el")
                self.far.append(cur["xy"])
                return None
            top = [cur["xy"][0], cur["xy"][1], C.FETCH_CUBE_M]
            # camera drept deasupra cubului; dacă nu ajunge acolo, de unde l-ar lua (cubul e mai spre margine)
            pose = next((a for a in (kinematics.look_at(top, d) for d in VIEW_DIST) if a), sol[1])
            seen = [c for c in self.view(pose) if c["color"] == cube["color"]]
            dist = [np.hypot(*np.subtract(c["xy"], cur["xy"])) for c in seen]
            if not seen or min(dist) > LOST:
                self.log(f"  {cube['color']}: nu-l mai văd de aproape")
                return None
            best = seen[int(np.argmin(dist))]
            moved = min(dist)
            cur.update(best)
            if moved < C.FETCH_TOL:
                break
        return cur

    def fetch(self, cube, cubes):
        """Ia cubul: apropiere, rotația degetelor, coborâre, prindere. True dacă e în gripper."""
        close = self.approach(cube)
        if close is None:
            return False
        sol = self.poses(close["xy"])
        roll = self.roll(close, [close] + [o for o in cubes if o is not cube], sol[2])
        sol = self.poses(close["xy"], roll)
        if sol is None:
            self.log(f"  {cube['color']}: nu ajung la el cu gripperul rotit")
            return False
        down, above, pitch = sol
        self.log(f"  iau {cube['color']} de la {np.round(close['xy'], 3).tolist()} m "
                 f"(gripper la {pitch}°, rotit {roll:+.0f}°)")
        if self.rob.grab_at(above, down):
            return True
        self.log(f"  {cube['color']}: nu l-am prins")
        return False


def restore(rob, ctx, memorized, look, fetcher, grid="S"):
    """Pune cuburile din jur înapoi pe caroiaj, ca în `memorized`. Întoarce raportul (puse, lipsă)."""
    rows = [r - 1 for r in C.S1_ROW_ORDER]   # întâi rândul cel mai depărtat: brațul nu trece peste cuburile puse
    todo = [(f"{grid}{r + 1}{c + 1}", memorized[r][c]) for r in rows for c in range(3) if memorized[r][c]]
    if not rob.sim:
        missing = sorted({f"{cell}_{h}" for cell, _ in todo for h in ("jos", "sus")} - set(rob.pos))
        if missing:
            raise RuntimeError(f"Învață întâi pozițiile {', '.join(missing)} (tools/calibrate_arm.py)")
    cubes = fetcher.scan(look)
    ctx.publish("found", [{"color": c["color"], "xy": [round(v, 3) for v in c["xy"]]} for c in cubes])
    ctx.log(f"Găsite {len(cubes)} cuburi: " + (", ".join(
        f"{c['color']} la {np.round(c['xy'], 3).tolist()}" for c in cubes) or "niciunul"))
    placed, tries, rescanned = [], 3 * len(todo) + 3, False
    while todo and tries > 0:
        need = {color for _, color in todo}
        cand = [c for c in cubes if c["color"] in need and not any(
            np.hypot(*np.subtract(c["xy"], xy)) < LOST for xy in fetcher.far)]
        if not cand:
            if rescanned:
                break
            ctx.log("Nu mai văd cuburile care lipsesc; caut din nou.")
            cubes, rescanned = fetcher.scan(look), True
            continue
        tries -= 1
        cube = max(cand, key=lambda c: fetcher.free(c, cubes))
        ok = fetcher.fetch(cube, cubes)
        cubes.remove(cube)
        if not ok:
            rescanned = False   # poate s-a mișcat: îl regăsește căutarea următoare
            continue
        cell = next(cell for cell, color in todo if color == cube["color"])
        ctx.log(f"Pun {cube['color']} în {cell}")
        rob.place(cell)
        todo.remove((cell, cube["color"]))
        placed.append(cell)
        rescanned = False
    if fetcher.far:
        ctx.log("Nu ajung la cuburile de la " + ", ".join(
            f"{np.hypot(*xy) * 100:.0f} cm de bază" for xy in fetcher.far))
    return {"placed": placed, "missing": todo, "far": len(fetcher.far)}
