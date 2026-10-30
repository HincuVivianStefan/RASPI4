"""Sarcina 1, partea a doua: cuburile luate de pe caroiaj și puse într-un loc pe care camera nu-l vedea („unghiul
mort”) sunt căutate, luate unul câte unul și puse înapoi în celulele memorate.

1. Căutarea: baza se rotește stânga / dreapta, pe două rânduri (de unde se vede caroiajul și puțin mai aproape).
   Petele de culoare sunt duse pe masă cu poza camerei de pe gripper, calculată din unghiurile brațului
   (kinematics.py), și măsurate în metri: un cub e recunoscut oricât de aproape ar fi camera.
2. Alegerea: un cub de o culoare care mai trebuie pusă, cel cu cel mai mult loc liber în jur (cuburile pot fi
   lipite); degetele se închid pe direcția pe care au mai mult loc.
3. Prinderea: direct din poziția văzută la căutare (de acolo cubul se vede întreg; de aproape iese din imagine):
   servo 5 rotește gripperul după cub; deasupra lui, coboară, prinde, urcă. Cu LOOK_CLOSE, camera se duce întâi
   deasupra cubului și îi corectează poziția (approach).
5. Verificarea: după punere, camera se uită la celulă; pe același cadru găsește și liniile caroiajului (grid_here),
   deci măsoară cubul față de celula adevărată (greșeala modelului, diferită de la o poziție la alta, se anulează).
   Dacă e mai departe de PLACE_TOL de centru sau rotit mai mult de PLACE_YAW, e luat și pus din nou, corectat
   (corecția rămâne și pentru cuburile următoare: greșelile sunt aceleași).
4. Plasarea: în celula memorată. Din pozițiile învățate (Robot.place) dacă există; altfel calculată: centrul celulei
   din colțurile caroiajului pe masă (văzute la memorare), cinematica inversă ca la prindere, gripperul rotit după
   laturile caroiajului (Fetcher.place).

Pe brațul real, greșelile constante (lungimea gripperului, camera montată puțin altfel) se corectează din config:
GRIP_LEN, FETCH_OFFSET, FETCH_ROLL_SIGN.
"""
import os
import time

import cv2
import numpy as np

from . import config as C
from . import kinematics
from . import vision

SCAN_STEP = 15          # grade între opririle bazei la căutare
SCAN_SPAN = 75          # cât de departe stânga / dreapta de poziția de privit caroiajul
SCAN_NEAR = -12         # rândul de aproape: umărul cu atâtea grade mai în față (vede masa mai aproape de braț)
SCAN_FAR = 12           # rândul de departe: umărul cu atâtea grade mai pe spate (vede masa mai departe)
VIEW_DIST = (0.16, 0.20)  # m: de cât de aproape privește camera cubul, drept de sus (dacă nu-l vede / nu ajunge, de mai departe)
SETTLE_S = 0.3          # cât stă camera după o mișcare, înainte de cadru
ITER = 5                # de câte ori, cel mult, se corectează poziția cubului privit de aproape
LOST = 0.04             # m: privit de aproape, cubul e căutat în atâta jurul poziției știute
SAME = 0.6              # două detecții de aceeași culoare la mai puțin de atâtea laturi = același cub
GRID_MARGIN = 0.005     # m: cuburile cu centrul pe caroiaj (sau la atât de el) sunt deja puse, nu se iau
BORDER = 3              # px: o pată care atinge marginea imaginii e tăiată, deci nu e măsurată
PLACE_DZ = 0.008        # m: cubul e lăsat să cadă de atât deasupra foii (nu apasă în foaie; de la 1,5 cm se rotea la cădere;
                        # calibrarea gripperului greșește pe înălțime cu până la ~1 cm)
PLACE_TOL = 0.006       # m: cubul pus mai departe de atât de centrul celulei e luat și pus din nou (măsurarea: ~0,3–0,5 cm)
PLACE_TRIES = 2         # de câte ori, cel mult, se reface o punere
PLACE_YAW = 10          # grade: cubul rotit mai mult de atât față de caroiaj e pus din nou (o dată)
GRID_SHIFT = 0.03       # m: caroiajul văzut la verificare poate fi atât de departe de cel memorat (altfel e altceva)
GRAB_DZ = 0.005         # m: prinde cu atât deasupra mijlocului cubului (calibrarea gripperului e bună la ~1 cm: să nu
                        # intre în masă)
LOOK_CLOSE = False      # înainte de prindere, camera se duce deasupra cubului să-i corecteze poziția (pe brațul real
                        # cubul iese din imagine de aproape: îl luăm direct de unde l-a văzut la căutare)


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
        self.sheet = []    # cadrele căutării, micșorate, cu cuburile marcate (planșa din captures/)
        self.look = None   # unghiurile din care se vede caroiajul (restore): de aici se verifică punerea
        self.fix = np.zeros(2)   # m: cât greșește punerea (măsurat după puneri), scăzut din țintele următoare

    def on_grid(self, xy):
        if self.grid is None:
            return False
        return cv2.pointPolygonTest(self.grid * 1000, (float(xy[0]) * 1000, float(xy[1]) * 1000), True) \
            > -GRID_MARGIN * 1000

    def view(self, pose, t=600, note=False, grid_too=False):
        """Cuburile din afara caroiajului (grid_too: și de pe el) văzute din `pose`. note: cadrul intră în planșa
        căutării."""
        self.rob.move(pose, t)
        self.rob.sleep(SETTLE_S)
        frame = self.get_frame()
        seen = table_cubes(frame, pose)
        for c in seen:
            c["pose"] = list(pose)      # de unde a fost văzut (verificarea după prindere se uită tot de acolo)
        out = [c for c in seen if grid_too or not self.on_grid(c["xy"])]
        if note:
            small = cv2.resize(frame, (256, 192))
            k = 256 / frame.shape[1]
            for c in seen:
                color = (0, 255, 0) if c in out else (160, 160, 160)    # verde = de luat, gri = pe caroiaj
                cv2.circle(small, (int(c["px"][0] * k), int(c["px"][1] * k)), 12, color, 2)
            cv2.putText(small, f"{[round(a) for a in pose[:2]]} {len(out)}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, cv2.LINE_AA)
            self.sheet.append(small)
        return out

    def save_sheet(self):
        """Planșa căutării în captures/ (un cadru pe oprire). Întoarce calea."""
        if not self.sheet:
            return None
        tiles = list(self.sheet)
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        os.makedirs(C.CAPTURES_DIR, exist_ok=True)
        path = os.path.join(C.CAPTURES_DIR, f"scan_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))
        return path

    def scan(self, look, need=None):
        """Caută cuburile în jur: baza pleacă de lângă caroiaj spre exterior, întâi pe o parte, apoi pe cealaltă; pe
        rândul privirii de memorare, apoi mai departe (SCAN_FAR), apoi mai aproape (SCAN_NEAR), cât timp nu a găsit tot. need = {culoare: câte trebuie}: se oprește
        când le-a găsit. Același cub văzut de mai multe ori se păstrează o dată, de unde a fost văzut cel mai aproape
        de mijlocul imaginii."""
        rows = []
        for d in (0, SCAN_FAR, SCAN_NEAR):          # privirea de memorare, apoi mai departe, apoi mai aproape
            p = list(look)
            p[1] = min(max(p[1] + d, C.LIMITS[1][0]), C.LIMITS[1][1])
            rows.append(p)
        b0 = round(look[0])
        sides = [[b for b in range(b0 + d * SCAN_STEP, b0 + d * (SCAN_SPAN + 1), d * SCAN_STEP)
                  if C.LIMITS[0][0] <= b <= C.LIMITS[0][1]] for d in (1, -1)]    # stânga, apoi dreapta
        found, first = [], True

        def enough():
            return need is not None and all(sum(c["color"] == k for c in found) >= n for k, n in need.items())

        try:
            for row, pose0 in enumerate(rows):
                for b in sides[0] + sides[1]:
                    pose = list(pose0)
                    pose[0] = b
                    cubes = self.view(pose, 1500 if first else 600, note=True)
                    first = False
                    if cubes:
                        self.log(f"  baza {b}°, rândul {row + 1}: " + ", ".join(c["color"] for c in cubes))
                    for c in cubes:
                        same = [o for o in found if o["color"] == c["color"]
                                and np.hypot(*np.subtract(o["xy"], c["xy"])) < SAME * C.FETCH_CUBE_M]
                        if not same:
                            found.append(c)
                        elif c["off"] < same[0]["off"]:
                            same[0].update(c)
                    if enough():
                        self.log("  Am găsit toate cuburile care lipsesc.")
                        return found
        finally:
            path = self.save_sheet()
            if path:
                self.log(f"Planșa căutării: {path}")
            self.sheet = []
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

    def poses(self, xy, roll=0.0, dz=0.0):
        """(prindere, deasupra, înclinarea) pentru un cub cu centrul în xy (ridicat cu dz), sau None dacă nu ajunge."""
        xy = np.float64(xy)
        u = xy / max(np.linalg.norm(xy), 1e-9)
        xy = xy + C.FETCH_OFFSET[0] * u + C.FETCH_OFFSET[1] * np.array([-u[1], u[0]])
        return kinematics.reach([xy[0], xy[1], C.FETCH_CUBE_M / 2 + dz], C.FETCH_LIFT,
                                90 + C.FETCH_ROLL_SIGN * roll)

    def approach(self, cube):
        """Privește cubul de deasupra și îi corectează poziția până nu se mai schimbă. Întoarce cubul văzut de
        aproape (sau None: nu ajunge / nu-l mai vede). Folosită doar cu LOOK_CLOSE."""
        cur = dict(cube)
        for _ in range(ITER):
            sol = self.poses(cur["xy"])
            if sol is None:
                self.log(f"  {cube['color']} la {np.round(cur['xy'], 3).tolist()} m: nu ajung la el")
                self.far.append(cur["xy"])
                return None
            top = [cur["xy"][0], cur["xy"][1], C.FETCH_CUBE_M]
            # camera drept deasupra cubului (de la VIEW_DIST, întâi de aproape); dacă de acolo nu-l vede (modelul
            # greșește mai mult cu camera aproape: cubul iese din imagine), de mai sus; dacă nu ajunge deloc
            # acolo, de unde l-ar lua (cubul e mai spre margine)
            views = [a for a in (kinematics.look_at(top, d) for d in VIEW_DIST) if a] or [sol[1]]
            seen = dist = None
            for pose in views:
                seen = [c for c in self.view(pose) if c["color"] == cube["color"]]
                dist = [np.hypot(*np.subtract(c["xy"], cur["xy"])) for c in seen]
                if seen and min(dist) <= LOST:
                    break
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
        """Ia cubul: (apropiere, dacă LOOK_CLOSE), rotația degetelor, coborâre, prindere. True dacă e în gripper."""
        close = self.approach(cube) if LOOK_CLOSE else cube
        if close is None:
            return False
        sol = self.poses(close["xy"], dz=GRAB_DZ)
        if sol is None:
            self.log(f"  {cube['color']} la {np.round(close['xy'], 3).tolist()} m: nu ajung la el")
            self.far.append(close["xy"])
            return False
        roll = self.roll(close, [close] + [o for o in cubes if o is not cube], sol[2])
        sol = self.poses(close["xy"], roll, GRAB_DZ)
        if sol is None:
            self.log(f"  {cube['color']}: nu ajung la el cu gripperul rotit")
            return False
        down, above, pitch = sol
        self.log(f"  iau {cube['color']} de la {np.round(close['xy'], 3).tolist()} m "
                 f"(gripper la {pitch}°, rotit {roll:+.0f}°)")
        if self.rob.grab_at(above, down):
            return True
        # Servo 6 zice „gol”, dar pe cubul din set gripperul se închide aproape cât gol: ne uităm cu camera.
        if not self.rob.sim and close.get("pose") and not self.still_there(close):
            self.log(f"  {cube['color']}: nu mai e pe masă, deci e în gripper")
            return True
        self.log(f"  {cube['color']}: nu l-am prins")
        return False

    def still_there(self, cube):
        """Cubul e tot pe masă, unde era sau împins puțin (privit de unde fusese văzut la căutare)?"""
        seen = [c for c in self.view(cube["pose"], grid_too=True) if c["color"] == cube["color"]]
        return any(np.hypot(*np.subtract(c["xy"], cube["xy"])) < 2 * LOST for c in seen)

    def cell_xy(self, r, c, grid=None):
        """Centrul celulei (r, c), 0..2, pe masă (m): din colțurile caroiajului (implicit cel memorat; stânga-sus,
        dreapta-sus, dreapta-jos, stânga-jos în imagine; rândul 0 = sus în imagine = cel mai departe de braț)."""
        tl, tr, br, bl = np.float64(self.grid if grid is None else grid)
        u, v = (c + 0.5) / 3, (r + 0.5) / 3
        return (1 - v) * ((1 - u) * tl + u * tr) + v * ((1 - u) * bl + u * br)

    def place(self, cell):
        """Pune cubul din gripper în celula `cell` (de ex. „S22”), calculat din caroiajul văzut (minus greșeala
        măsurată la punerile de dinainte, self.fix). True dacă a ajuns."""
        xy = self.cell_xy(int(cell[1]) - 1, int(cell[2]) - 1) - self.fix
        tl, tr = np.float64(self.grid[:2])
        edge = np.degrees(np.arctan2(*(tr - tl)[::-1]))        # direcția rândurilor pe masă
        theta = np.degrees(np.arctan2(xy[1], xy[0]))
        roll = wrap(edge - theta)                              # degetele pe laturile caroiajului, rotația cea mai mică
        sol = self.poses(xy, roll, PLACE_DZ)
        if sol is None:
            self.log(f"  {cell}: nu ajung la celulă ({np.round(xy, 3).tolist()} m)")
            return False
        down, above, pitch = sol
        self.log(f"  pun în {cell} la {np.round(xy, 3).tolist()} m (gripper la {pitch}°, rotit {roll:+.0f}°)")
        self.rob.place_at(above, down, cell)
        return True

    def grid_here(self, frame, pose):
        """Caroiajul (4 colțuri pe masă, m) cum se vede pe acest cadru: cel memorat, mutat (cel mult GRID_SHIFT) și
        rotit puțin până când liniile lui cad cel mai bine pe liniile negre din imaginea de sus a cadrului. Merge și
        când se vede doar o parte din caroiaj. None dacă liniile văzute nu se potrivesc destul."""
        from .panorama import MM, Panorama
        mem = np.float64(self.grid)
        pano = Panorama(mem.mean(0))
        pano.add(frame, pose)
        pano.compose(lambda *a: None)
        small, s = vision.shrink(pano.img[0.0], vision.GRID_W)
        mask = cv2.dilate(vision.grid_mask(small), np.ones((3, 3), np.uint8))
        h, w = mask.shape
        R = int(GRID_SHIFT / MM * s)
        px = pano.to_pixels(mem) * s
        H0 = cv2.getPerspectiveTransform(np.float32([[0, 0], [3, 0], [3, 3], [0, 3]]), np.float32(px))
        best = None
        for ang in range(-4, 5):
            M = cv2.getRotationMatrix2D(tuple(px.mean(0)), ang, 1.0)
            tpl = np.zeros((h, w), np.uint8)
            for k in range(4):
                for a, b in (([k, 0], [k, 3]), ([0, k], [3, k])):
                    p = cv2.transform(cv2.perspectiveTransform(np.float32([[a, b]]), H0), M)[0]
                    cv2.line(tpl, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), 255, 1)
            ys, xs = np.nonzero(tpl)
            if len(xs) == 0:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            if x0 - R < 0 or y0 - R < 0 or x1 + R > w or y1 + R > h:
                continue
            res = cv2.matchTemplate(mask[y0 - R:y1 + R, x0 - R:x1 + R], tpl[y0:y1, x0:x1], cv2.TM_CCORR)
            _, score, _, loc = cv2.minMaxLoc(res)
            score /= 255.0 * 255.0 * len(xs)            # cât din liniile caroiajului cad pe negru
            if best is None or score > best[0]:
                best = (score, M, loc[0] - R, loc[1] - R)
        if best is None or best[0] < 0.2:
            return None
        _, M, dx, dy = best
        moved = cv2.transform(np.float32([px]), M)[0] + [dx, dy]
        return pano.to_table(moved / s)

    def placed_at(self, cell, color):
        """(cubul pus în `cell`, caroiajul văzut pe același cadru sau None) — sau (None, None) dacă nu vede cubul.
        Privește din poziția de lângă `look` (panorama.VIEWS) în care centrul celulei cade cel mai aproape de
        mijlocul imaginii."""
        from .aim import project
        from .panorama import VIEWS
        if self.look is None:
            return None, None
        xy = self.cell_xy(int(cell[1]) - 1, int(cell[2]) - 1)
        best, pose = None, None
        for db, ds in VIEWS:
            p = list(self.look)
            p[0] += db
            p[1] += ds
            q = project(p, np.float64([[xy[0], xy[1], C.FETCH_CUBE_M]]), (C.CAM_W, C.CAM_H))
            if q is not None and (best is None or np.abs(q[0] - 0.5).max() < best):
                best, pose = float(np.abs(q[0] - 0.5).max()), p
        if pose is None:
            return None, None
        self.rob.move(pose, 800)
        self.rob.sleep(SETTLE_S)
        frame = self.get_frame()
        grid = self.grid_here(frame, pose)
        center = self.cell_xy(int(cell[1]) - 1, int(cell[2]) - 1, grid)
        side = np.linalg.norm(np.float64(self.grid[1]) - np.float64(self.grid[0])) / 3
        seen = [c for c in table_cubes(frame, pose)
                if c["color"] == color and np.hypot(*np.subtract(c["xy"], center)) < 0.7 * side]
        if not seen:
            return None, grid
        c = min(seen, key=lambda o: np.hypot(*np.subtract(o["xy"], center)))
        c["pose"] = pose
        return c, grid

    def place_checked(self, cell, color):
        """Pune cubul din gripper în `cell`, apoi verifică (față de liniile caroiajului văzute atunci): dacă e mai
        departe de PLACE_TOL de centru sau rotit mai mult de PLACE_YAW, îl ia și îl pune din nou, corectat (de cel mult
        PLACE_TRIES ori). False dacă nu ajunge la celulă."""
        if not self.place(cell):
            return False
        r, c = int(cell[1]) - 1, int(cell[2]) - 1
        last_yaw = None
        for k in range(PLACE_TRIES + 1):
            cube, grid = self.placed_at(cell, color)
            if cube is None:
                self.log(f"  {cell}: nu văd cubul pus, nu pot verifica")
                return True
            g = np.float64(self.grid if grid is None else grid)
            off = np.subtract(cube["xy"], self.cell_xy(r, c, g))
            yaw = wrap(cube["yaw"] - np.degrees(np.arctan2(*(g[1] - g[0])[::-1])))
            self.log(f"  {cell}: cubul e la {np.hypot(*off) * 100:.1f} cm de centru ({off[0] * 100:+.1f}, "
                     f"{off[1] * 100:+.1f} cm), rotit {yaw:+.0f}° față de caroiaj"
                     + ("" if grid is not None else " (liniile nu le văd: față de caroiajul memorat)"))
            if np.hypot(*off) <= PLACE_TOL and abs(yaw) <= PLACE_YAW:
                return True
            if np.hypot(*off) <= PLACE_TOL and last_yaw is not None:
                return True             # rotit, dar deja repus o dată pentru rotație: mai mult n-ar ajuta
            last_yaw = yaw
            self.fix = self.fix + off                     # data viitoare țintim cu atât mai încolo
            if k == PLACE_TRIES:
                self.log(f"  {cell}: îl las așa")
                return True
            self.log(f"  {cell}: îl iau și îl pun din nou, corectat")
            if not self.fetch(cube, [cube]):
                self.log(f"  {cell}: nu l-am putut lua din nou")
                return True
            if not self.place(cell):
                return False
        return True

def restore(rob, ctx, memorized, look, fetcher, grid="S", present=None):
    """Pune cuburile din jur înapoi pe caroiaj, ca în `memorized`. Întoarce raportul (puse, lipsă).
    present = ce e acum pe caroiaj (3x3): celulele ocupate rămân cum sunt (cuburile luate unul câte unul)."""
    present = present or [[None] * 3 for _ in range(3)]
    fetcher.look = look
    rows = [r - 1 for r in C.S1_ROW_ORDER]   # întâi rândul cel mai depărtat: brațul nu trece peste cuburile puse
    for r in rows:
        for c in range(3):
            if present[r][c] and present[r][c] != memorized[r][c]:
                ctx.log(f"În {grid}{r + 1}{c + 1} e {present[r][c]}, dar am memorat {memorized[r][c] or 'nimic'}: îl las acolo.")
    todo = [(f"{grid}{r + 1}{c + 1}", memorized[r][c]) for r in rows for c in range(3)
            if memorized[r][c] and not present[r][c]]
    if not todo:
        ctx.log("Nu lipsește niciun cub de pe caroiaj.")
        return {"placed": [], "missing": [], "far": 0}
    def learned(cell):
        return rob.sim or {f"{cell}_jos", f"{cell}_sus"} <= set(rob.pos)
    if fetcher.grid is None and not all(learned(cell) for cell, _ in todo):
        raise RuntimeError("Nu știu unde e caroiajul pe masă (nu l-am văzut întreg la memorare): memorează din nou")
    wanted = lambda: {k: sum(c == k for _, c in todo) for _, k in todo}   # noqa: E731  câte din fiecare culoare
    cubes = fetcher.scan(look, wanted())
    ctx.publish("found", [{"color": c["color"], "xy": [round(v, 3) for v in c["xy"]]} for c in cubes])
    ctx.log(f"Găsite {len(cubes)} cuburi: " + (", ".join(
        f"{c['color']} la {np.round(c['xy'], 3).tolist()}" for c in cubes) or "niciunul"))
    placed, failed, tries, rescanned = [], [], 3 * len(todo) + 3, False
    while todo and tries > 0:
        need = {color for _, color in todo}
        cand = [c for c in cubes if c["color"] in need and not any(
            np.hypot(*np.subtract(c["xy"], xy)) < LOST for xy in fetcher.far)]
        if not cand:
            if rescanned:
                break
            ctx.log("Nu mai văd cuburile care lipsesc; caut din nou.")
            cubes, rescanned = fetcher.scan(look, wanted()), True
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
        if learned(cell):
            rob.place(cell)
        elif not fetcher.place_checked(cell, cube["color"]):
            rob.gripper(C.GRIP_OPEN)    # nu ajunge la celulă: îl lasă aici, să nu rămână în gripper
            todo.remove((cell, cube["color"]))
            failed.append((cell, cube["color"]))
            continue
        todo.remove((cell, cube["color"]))
        placed.append(cell)
        rescanned = False
    if fetcher.far:
        ctx.log("Nu ajung la cuburile de la " + ", ".join(
            f"{np.hypot(*xy) * 100:.0f} cm de bază" for xy in fetcher.far))
    return {"placed": placed, "missing": todo + failed, "far": len(fetcher.far)}
