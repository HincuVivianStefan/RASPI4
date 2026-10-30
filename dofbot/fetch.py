"""Sarcina 1, partea a doua: cuburile luate de pe caroiaj și puse într-un loc pe care camera nu-l vedea („unghiul
mort”) sunt căutate, luate unul câte unul și puse înapoi în celulele memorate.

1. Căutarea: baza se rotește stânga / dreapta, pe două rânduri (de unde se vede caroiajul și puțin mai aproape).
   Petele de culoare sunt duse pe masă cu poza camerei de pe gripper, calculată din unghiurile brațului
   (kinematics.py), și măsurate în metri: un cub e recunoscut oricât de aproape ar fi camera.
2. Alegerea: un cub de o culoare care mai trebuie pusă, cel cu cel mai mult loc liber în jur (cuburile pot fi
   lipite); degetele se închid pe direcția pe care au mai mult loc.
3. Prinderea: direct din poziția văzută la căutare (de acolo cubul se vede întreg; de aproape iese din imagine):
   deasupra lui, coboară, prinde, urcă. Cu LOOK_CLOSE, camera se duce întâi deasupra cubului și îi corectează
   poziția (approach).
   Servo 5: cu gripperul înclinat, răsucirea nu doar rotește degetele, le și apleacă, iar cubul prins / lăsat cu
   degetele aplecate iese cu baza strâmbă. De aceea servo 5 se răsucește doar cât axa degetelor rămâne aproape
   orizontală (GRAB_TILT, calculat din cinematică: plan()). Dacă așa degetele nu ajung pe fețele cubului, cubul
   e luat cu servo 5 drept (fălcile îl îndreaptă între ele) și dus pe locul de îndreptare, lângă bază, unde
   gripperul e aproape vertical: lăsat jos, privit de sus și luat din nou cu degetele pe fețele lui (straighten).
5. Verificarea: după punere, camera se uită la celulă; pe același cadru găsește și liniile caroiajului (grid_here),
   deci măsoară cubul față de celula adevărată (greșeala modelului, diferită de la o poziție la alta, se anulează).
   Dacă e mai departe de PLACE_TOL de centru sau rotit mai mult de PLACE_YAW, e luat și pus din nou, corectat
   (corecția rămâne și pentru cuburile următoare: greșelile sunt aceleași).
4. Plasarea: în celula memorată. Din pozițiile învățate (Robot.place) dacă există; altfel calculată: centrul celulei
   din colțurile caroiajului pe masă (văzute la memorare), cinematica inversă ca la prindere, degetele cât mai pe
   laturile caroiajului fără să se aplece peste GRAB_TILT (Fetcher.place; pe rândurile departe de bază cubul
   poate rămâne rotit câteva grade, dar cu baza dreaptă).

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
GRAB_DZ = 0.008         # m: prinde cu atât deasupra mijlocului cubului (cu 0,5 cm apăsa în masă, fără GRIP_DZ_R:
                        # aproape de bază greșea ~1 cm; cu 1,5 cm prindea prea sus, de colțul cubului)  # REGLEAZĂ
LOOK_CLOSE = False      # înainte de prindere, camera se duce deasupra cubului să-i corecteze poziția (pe brațul real
                        # cubul iese din imagine de aproape: îl luăm direct de unde l-a văzut la căutare)
GRAB_TILT = 4           # grade: cât de aplecată poate fi axa degetelor la prindere / punere (peste: baza cubului strâmbă)
TILT_COST = 2           # plan(): un grad de aplecare a degetelor cântărește cât atâtea grade de cub rotit
PITCH_COST = 0.5        # plan(): și un grad de gripper mai înclinat / mai drept decât la calibrare (în afara ei
                        # modelul greșește înălțimea: gripperul apăsa în masă la 158–166°, calibrat la 109–137°)
GRAB_YAW_TOL = 10       # grade: degetele rotite mai mult de atât față de fețele cubului = prins strâmb → îndreptare
FINGER_ROOM = 1.6       # laturi de cub: atâta loc trebuie pe direcția degetelor, ca să încapă lângă un cub vecin
STAGE = True            # cubul care nu poate fi prins drept e îndreptat pe locul de lângă bază (straighten)
GRAB_REACH = -0.015     # m: prinde cu atât mai departe de bază decât centrul văzut al cubului (negativ = mai spre
                        # braț: cu 0 și -0,008 prindea de colțul exterior, cu +0,008 pe lângă cub)    # REGLEAZĂ
PLACE_REACH = -0.008    # m: la fel, la punere (cu -0,015 cubul ajungea prea spre braț pe rândul 2)  # REGLEAZĂ
STAGE_R = 0.16          # m de bază: locul de îndreptare (gripperul până la ~160°; la 12,5 cm cubul ajungea prea
                        # aproape de bază ca să-l mai poată lua)
STAGE_BASES = (140, 40, 150, 30, 130, 50)   # unde poate fi, în ordine (baza; departe de caroiaj, în stânga / dreapta)
STAGE_TILT = 10         # grade: pe locul de îndreptare gripperul e aproape vertical, servo 5 se poate răsuci mai mult
STAGE_CLEAR = 2.0       # laturi de cub: locul de îndreptare e liber dacă niciun alt cub nu e mai aproape de atât
LOOK_FRAMES = 3         # la verificarea punerii și pe locul de îndreptare: atâtea cadre, cubul = mediana lor
STEADY_FRAMES = 3       # cadre pentru măsurarea de aproape (verificarea punerii, locul de îndreptare): mediana lor
TOWER_DZ = 0.005        # m: cubul pus pe turn e lăsat de atât deasupra cubului de dedesubt               # REGLEAZĂ
TOWER_TOL = 0.005       # m: cubul de sus mai departe de atât de axa turnului e luat și pus din nou      # REGLEAZĂ
TOWER_MIN_PITCH = 95    # grade: pentru vârful turnului pe rândul departe, gripperul poate fi și mai înclinat de
                        # FETCH_MIN_PITCH (pe rândul 1, nivelul 5 cere ~96°)


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


def table_cubes(frame, angles, level=1):
    """Cuburile văzute în cadru, puse pe masă: dict cu color, xy (m), yaw (grade, direcția unei laturi),
    px (centrul în imagine), off (cât de departe de mijlocul imaginii, 0..1). level = fețele de sus căutate la
    înălțimea cubului de pe acest nivel (2 = pus peste altul, în turn)."""
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
            pts, ok = to_table(angles, img, (w, h), level * a)
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


def merge(lists):
    """Cuburile văzute pe mai multe cadre (o listă table_cubes pe cadru), unite: același cub = aceeași culoare la mai
    puțin de SAME laturi. Rămân doar cele văzute pe cel puțin jumătate din cadre (o pată de o clipă nu e cub), cu
    poziția = mediana (o măsurare greșită nu o mută) și rotația = media pe cerc modulo 90° (cubul arată la fel
    rotit cu 90°: 44° și -44° sunt aproape)."""
    groups = []
    for seen in lists:
        for c in seen:
            g = next((g for g in groups if g[0]["color"] == c["color"]
                      and np.hypot(*np.subtract(g[0]["xy"], c["xy"])) < SAME * C.FETCH_CUBE_M), None)
            if g is None:
                groups.append([c])
            else:
                g.append(c)
    out = []
    for g in groups:
        if 2 * len(g) < len(lists):
            continue
        k = np.radians(4 * np.float64([c["yaw"] for c in g]))
        yaw = np.degrees(np.arctan2(np.sin(k).mean(), np.cos(k).mean())) / 4
        best = min(g, key=lambda c: c["off"])
        out.append(dict(best, xy=np.median([c["xy"] for c in g], axis=0).tolist(), yaw=float(yaw)))
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


def merge(lists):
    """Cuburile văzute pe mai multe cadre din aceeași poziție (table_cubes pe fiecare): doar cele găsite pe mai mult
    de jumătate din cadre, cu poziția = mediana (o pată greșită pe un cadru nu mută cubul) și rotația = media pe
    cercul de 90° (latura la 44° și cea la -44° sunt aproape aceeași)."""
    groups = []
    for seen in lists:
        for c in seen:
            g = next((g for g in groups if g[0]["color"] == c["color"]
                      and np.hypot(*np.subtract(g[0]["xy"], c["xy"])) < SAME * C.FETCH_CUBE_M), None)
            if g is None:
                groups.append([c])
            else:
                g.append(c)
    out = []
    for g in groups:
        if len(g) <= len(lists) / 2:
            continue
        yaw = np.angle(np.mean([np.exp(4j * np.radians(o["yaw"])) for o in g]))
        out.append(dict(g[0], xy=np.median([o["xy"] for o in g], axis=0).tolist(),
                        px=np.median([o["px"] for o in g], axis=0).tolist(),
                        yaw=float(np.degrees(yaw) / 4), off=min(o["off"] for o in g)))
    return out


class Fetcher:
    def __init__(self, rob, get_frame, log=print, grid=None):
        self.rob, self.get_frame, self.log = rob, get_frame, log
        self.grid = None if grid is None else np.float32(grid)   # caroiajul pe masă (m): cuburile de acolo stau
        self.far = []      # unde sunt cuburile la care brațul nu ajunge (m): nu se mai încearcă
        self.sheet = []    # cadrele căutării, micșorate, cu cuburile marcate (planșa din captures/)
        self.look = None   # unghiurile din care se vede caroiajul (restore): de aici se verifică punerea
        self.fix = np.zeros(2)   # m: cât greșește punerea (măsurat după puneri), scăzut din țintele următoare
        self.expect = 0.0        # grade: cât de rotit față de caroiaj iese cubul la ultima punere (plan)
        self.axis = {}           # Sarcina 2: {celulă: (cubul de jos, m; poziția din care a fost măsurat)}
        self.tfix = np.zeros(2)  # m: cât greșește punerea pe turn (măsurat), scăzut din țintele următoare

    def on_grid(self, xy):
        if self.grid is None:
            return False
        return cv2.pointPolygonTest(self.grid * 1000, (float(xy[0]) * 1000, float(xy[1]) * 1000), True) \
            > -GRID_MARGIN * 1000

    def view(self, pose, t=600, note=False, grid_too=False, frames=1):
        """Cuburile din afara caroiajului (grid_too: și de pe el) văzute din `pose` (frames > 1: unite din atâtea
        cadre, merge). note: cadrul intră în planșa căutării."""
        self.rob.move(pose, t)
        self.rob.sleep(SETTLE_S)
        frame = self.get_frame()
        seen = table_cubes(frame, pose)
        if frames > 1:
            seen = merge([seen] + [table_cubes(self.get_frame(), pose) for _ in range(frames - 1)])
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
    def axes(cube, cubes):
        """Cele două direcții (grade pe masă) pe care se pot închide degetele pe fețele cubului, fiecare cu locul
        liber de pe ea până la cel mai apropiat vecin din cale (m)."""
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
            options.append((clear, axis))
        return options

    def poses(self, xy, roll=0.0, dz=0.0, pitch=None):
        """(prindere, deasupra, înclinarea) pentru un cub cu centrul în xy (ridicat cu dz), sau None dacă nu ajunge.
        pitch = doar cu această înclinare a gripperului (altfel: cea aleasă de kinematics.reach)."""
        xy = np.float64(xy)
        u = xy / max(np.linalg.norm(xy), 1e-9)
        xy = xy + C.FETCH_OFFSET[0] * u + C.FETCH_OFFSET[1] * np.array([-u[1], u[0]])
        return kinematics.reach([xy[0], xy[1], C.FETCH_CUBE_M / 2 + dz], C.FETCH_LIFT,
                                90 + C.FETCH_ROLL_SIGN * roll, None if pitch is None else [pitch])

    def plan(self, xy, want=None, dz=0.0, max_tilt=GRAB_TILT, period=90):
        """Prinderea / punerea unui cub cu centrul în xy: (jos, deasupra, înclinarea gripperului, rotația, abaterea).
        Degetele cât mai pe direcția `want` (grade pe masă, modulo `period`; None = servo 5 drept), cu axa lor
        aplecată cel mult max_tilt grade (altfel cubul e ținut aplecat și iese cu baza strâmbă): cu gripperul
        vertical servo 5 le poate roti oricât, cu gripperul înclinat doar puțin. Se alege cea mai mică abatere +
        TILT_COST × aplecarea (aplecarea strâmbă baza, abaterea doar rotește cubul) + PITCH_COST × depărtarea de
        înclinarea aleasă de reach (calibrarea e bună acolo). abaterea = cu câte grade rămân degetele rotite față
        de `want`. None dacă nu ajunge."""
        first = self.poses(xy, 0.0, dz)
        if first is None:
            return None
        if want is None:
            return (*first, 0.0, 0.0)
        best = None
        for pitch in [first[2]] + [p for p in range(180, C.FETCH_MIN_PITCH - 11, -2) if p != first[2]]:
            sol = first if pitch == first[2] else self.poses(xy, 0.0, dz, pitch)
            if sol is None:
                continue
            a = list(sol[0])
            for sign in (1, -1):
                for k in range(0 if sign > 0 else 1, 46):     # aplecarea crește cu rotația: ne oprim la max_tilt
                    a[4] = 90 + C.FETCH_ROLL_SIGN * sign * k
                    yaw, tilt = kinematics.fingers(a)
                    if tilt > max_tilt:
                        break
                    err = abs(wrap(yaw - want, period))
                    cost = err + TILT_COST * tilt + PITCH_COST * abs(pitch - first[2])
                    if best is None or cost < best[0]:
                        best = (cost, pitch, sign * k, err)
        _, pitch, roll, err = best          # servo 5 drept are axa degetelor orizontală: există mereu o soluție
        down, above, pitch = self.poses(xy, roll, dz, pitch)
        return down, above, pitch, float(roll), float(err)

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

    def grab_plan(self, cube, cubes, max_tilt=GRAB_TILT, dz=0.0):
        """plan() pentru prinderea cubului (ridicat cu dz: pus peste altul): degetele pe una dintre cele două perechi
        de fețe; întâi una pe care au loc lângă vecini (FINGER_ROOM), apoi cea la care ajung mai drept."""
        best = None
        for clear, axis in self.axes(cube, cubes):
            p = self.plan(self.grab_xy(cube["xy"]), axis, GRAB_DZ + dz, max_tilt, period=180)
            if p is None:
                return None
            key = (clear < FINGER_ROOM * C.FETCH_CUBE_M, round(p[4] / 2), -clear)
            if best is None or key < best[0]:
                best = (key, p)
        return best[1]

    @staticmethod
    def grab_xy(xy, reach=GRAB_REACH):
        """Unde se închid / se deschid degetele pentru cubul cu centrul (văzut, sau dorit la punere) în xy: cu
        `reach` mai departe de bază (GRAB_REACH la prindere, PLACE_REACH la punere)."""
        xy = np.float64(xy)
        return xy + reach * xy / max(np.linalg.norm(xy), 1e-9)

    def grab(self, cube, sol, what="iau"):
        """Ia cubul cu unghiurile sol = plan(). True dacă e în gripper."""
        down, above, pitch, roll, err = sol
        self.log(f"  {what} {cube['color']} de la {np.round(cube['xy'], 3).tolist()} m "
                 f"(gripper la {pitch}°, servo 5 rotit {roll:+.0f}°, degetele la {err:.0f}° de fețele lui)")
        if self.rob.grab_at(above, down):
            return True
        # Servo 6 zice „gol”, dar pe cubul din set gripperul se închide aproape cât gol: ne uităm cu camera.
        if not self.rob.sim and cube.get("pose") and not self.still_there(cube):
            self.log(f"  {cube['color']}: nu mai e pe masă, deci e în gripper")
            return True
        # Nici servo 6, nici camera (poate vede cubul chiar din gripper) nu sunt sigure: dacă totuși e în gripper,
        # îl lasă jos exact de unde l-a luat, ca să nu-l plimbe.
        self.log(f"  {cube['color']}: nu l-am prins; îl las jos la loc, în caz că totuși e în gripper")
        self.rob.place_at(above, down)
        return False

    def fetch(self, cube, cubes, dz=0.0):
        """Ia cubul (ridicat cu dz: pus peste altul): (apropiere, dacă LOOK_CLOSE), degetele pe fețele lui fără să le
        aplece; dacă nu se poate, cu servo 5 drept, apoi îndreptat pe locul de lângă bază (straighten). True dacă e
        în gripper."""
        close = self.approach(cube) if LOOK_CLOSE and not dz else cube
        if close is None:
            return False
        others = [o for o in cubes if o is not cube]
        sol = self.grab_plan(close, [close] + others, dz=dz)
        if sol is None:
            self.log(f"  {cube['color']} la {np.round(close['xy'], 3).tolist()} m: nu ajung la el")
            self.far.append(close["xy"])
            return False
        crooked = STAGE and sol[4] > GRAB_YAW_TOL
        if crooked:
            self.log(f"  {cube['color']}: degetele ar rămâne la {sol[4]:.0f}° de fețele lui; îl iau cu servo 5 "
                     "drept și îl îndrept lângă bază")
            sol = self.plan(self.grab_xy(close["xy"]), None, GRAB_DZ + dz)
        if sol is None or not self.grab(close, sol):
            return False
        return self.straighten(cube["color"], others) if crooked else True

    def stage_spot(self, cubes):
        """Locul de îndreptare (m): la STAGE_R de bază, departe de caroiaj și de celelalte cuburi, unde ajunge
        gripperul. None dacă nu e niciunul liber."""
        a = C.FETCH_CUBE_M
        for b in STAGE_BASES:
            t = np.radians(b - 90)
            xy = STAGE_R * np.array([np.cos(t), np.sin(t)])
            if self.grid is not None and cv2.pointPolygonTest(
                    self.grid * 1000, (float(xy[0]) * 1000, float(xy[1]) * 1000), True) > -1.5 * a * 1000:
                continue
            if any(np.hypot(*np.subtract(o["xy"], xy)) < STAGE_CLEAR * a for o in cubes):
                continue
            if any(np.hypot(*np.subtract(f, xy)) < STAGE_CLEAR * a for f in self.far):
                continue
            if self.plan(self.grab_xy(xy, PLACE_REACH), None, PLACE_DZ) is not None:    # acolo coboară straighten
                return xy
        return None

    def look_down(self, xy, color):
        """Cubul de culoarea `color` din jurul lui xy (m), privit cât mai de sus (de unde îi vede bine rotația), sau
        None dacă nu-l vede."""
        from .aim import project
        top = [xy[0], xy[1], C.FETCH_CUBE_M]
        views = [a for a in (kinematics.look_at(top, d) for d in VIEW_DIST[::-1]) if a]
        if self.look is not None:       # și ca la căutare, cu baza spre el: întâi de unde cade mai aproape de mijloc
            rows = []
            for d in (0, SCAN_NEAR, 2 * SCAN_NEAR):
                p = list(self.look)
                p[0] = 90 + np.degrees(np.arctan2(xy[1], xy[0]))
                p[1] = min(max(p[1] + d, C.LIMITS[1][0]), C.LIMITS[1][1])
                q = project(p, np.float64([top]), (C.CAM_W, C.CAM_H))
                if q is not None and np.abs(q[0] - 0.5).max() < 0.4:
                    rows.append((float(np.abs(q[0] - 0.5).max()), p))
            views += [p for _, p in sorted(rows)]
        for pose in views:
            seen = [c for c in self.view(pose, grid_too=True, frames=STEADY_FRAMES) if c["color"] == color
                    and np.hypot(*np.subtract(c["xy"], xy)) < 2 * LOST]
            if seen:
                return min(seen, key=lambda c: np.hypot(*np.subtract(c["xy"], xy)))
        return None

    def straighten(self, color, cubes):
        """Cubul din gripper, prins cu servo 5 drept: îl lasă pe locul de îndreptare (lângă bază, gripperul aproape
        vertical, deci servo 5 îl poate răsuci fără să-l aplece), se uită la el de sus și îl ia din nou cu degetele pe
        fețele lui. True dacă e iar în gripper."""
        spot = self.stage_spot(cubes)
        if spot is None:
            self.log("  nu am loc liber pentru îndreptare: îl duc cum l-am prins")
            return True
        down, above, pitch, _, _ = self.plan(self.grab_xy(spot, PLACE_REACH), None, PLACE_DZ)
        self.log(f"  îl las pe locul de îndreptare, la {np.round(spot, 3).tolist()} m (gripper la {pitch}°)")
        self.rob.place_at(above, down)
        cube = self.look_down(spot, color)
        if cube is None:
            self.log(f"  {color}: nu-l văd pe locul de îndreptare; îl iau cum l-am lăsat")
            cube = {"color": color, "xy": list(spot), "yaw": kinematics.fingers(down)[0]}
        sol = self.grab_plan(cube, [cube] + cubes, STAGE_TILT)
        if sol is None:         # văzut puțin mai încolo, unde nu ajunge: îl ia de unde l-a lăsat (acolo ajunge sigur)
            self.log(f"  {color}: îl văd la {np.round(cube['xy'], 3).tolist()} m, unde nu ajung; îl iau de unde l-am lăsat")
            cube = dict(cube, xy=list(spot))
            sol = self.grab_plan(cube, [cube] + cubes, STAGE_TILT)
        if sol is None:
            self.log(f"  {color}: nu mai ajung la el pe locul de îndreptare")
            self.far.append(cube["xy"])
            return False
        return self.grab(cube, sol, "îl iau din nou:")

    def still_there(self, cube):
        """Cubul e tot pe masă, unde era sau împins puțin (privit de unde fusese văzut la căutare)? Doar în jurul
        lui (SAME laturi): un alt cub de aceeași culoare din celula vecină (~5 cm) nu e el."""
        seen = [c for c in self.view(cube["pose"], grid_too=True) if c["color"] == cube["color"]]
        return any(np.hypot(*np.subtract(c["xy"], cube["xy"])) < SAME * C.FETCH_CUBE_M for c in seen)

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
        sol = self.plan(self.grab_xy(xy, PLACE_REACH), edge, PLACE_DZ)  # degetele pe laturile caroiajului
        if sol is None:
            self.log(f"  {cell}: nu ajung la celulă ({np.round(xy, 3).tolist()} m)")
            return False
        down, above, pitch, roll, self.expect = sol
        self.log(f"  pun în {cell} la {np.round(xy, 3).tolist()} m (gripper la {pitch}°, servo 5 rotit {roll:+.0f}°"
                 f", cubul iese rotit ~{self.expect:.0f}° față de caroiaj)")
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

    def placed_at(self, cell, color, level=1, pose=None):
        """(cubul pus în `cell` pe nivelul `level`, caroiajul văzut pe același cadru sau None) — sau (None, None)
        dacă nu vede cubul. Privește din `pose` sau din poziția de lângă `look` (panorama.VIEWS) în care centrul
        celulei cade cel mai aproape de mijlocul imaginii."""
        from .aim import project
        from .panorama import VIEWS
        if self.look is None:
            return None, None
        xy = self.cell_xy(int(cell[1]) - 1, int(cell[2]) - 1)
        best = None
        for db, ds in VIEWS if pose is None else ():
            p = list(self.look)
            p[0] += db
            p[1] += ds
            q = project(p, np.float64([[xy[0], xy[1], level * C.FETCH_CUBE_M]]), (C.CAM_W, C.CAM_H))
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
        cubes = merge([table_cubes(frame, pose, level)] + [table_cubes(self.get_frame(), pose, level)
                                                           for _ in range(STEADY_FRAMES - 1)])
        seen = [c for c in cubes if c["color"] == color and np.hypot(*np.subtract(c["xy"], center)) < 0.7 * side]
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
            if np.hypot(*off) <= PLACE_TOL and abs(yaw) <= PLACE_YAW + self.expect:
                return True             # rotit cât era de așteptat (mai drept nu-l poate pune fără să-l aplece)
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

    # ---------- Sarcina 2: turnul ----------
    def tower_plan(self, xy, want, dz):
        """plan() pentru punerea pe turn: dacă nu ajunge (vârful turnului pe rândul departe), și cu gripperul mai
        înclinat, până la TOWER_MIN_PITCH (în afara calibrării: verificarea cu camera corectează)."""
        sol = self.plan(xy, want, dz)
        if sol is None and TOWER_MIN_PITCH < C.FETCH_MIN_PITCH:
            old, C.FETCH_MIN_PITCH = C.FETCH_MIN_PITCH, TOWER_MIN_PITCH
            try:
                sol = self.plan(xy, want, dz)
            finally:
                C.FETCH_MIN_PITCH = old
        return sol

    def base_of(self, cell, color):
        """(unde e cubul de jos din `cell`, m; poziția din care a fost măsurat sau None): axa turnului. Toate
        cuburile de deasupra se țintesc pe el, nu pe centrul celulei, și se măsoară din aceeași poziție (greșeala
        camerei se anulează). Se măsoară cu gripperul gol (altfel cubul din gripper poate acoperi imaginea)."""
        if cell not in self.axis:
            cube, _ = self.placed_at(cell, color)
            if cube is None:
                self.log(f"  {cell}: nu văd cubul de jos; țintesc centrul celulei")
                self.axis[cell] = (self.cell_xy(int(cell[1]) - 1, int(cell[2]) - 1) - self.fix, None)
            else:
                self.axis[cell] = (np.float64(cube["xy"]), cube["pose"])
        return self.axis[cell]

    def take(self, cell, level, color, stacks):
        """Ia cubul de sus din `cell` (pe nivelul `level`), privit întâi de aproape. stacks = {celulă: [culori de
        jos în sus]}: celelalte cuburi de pe caroiaj, ca degetele să se închidă pe direcția cu loc. True dacă e în
        gripper."""
        r, c = int(cell[1]) - 1, int(cell[2]) - 1
        cube, grid = self.placed_at(cell, color, level)
        if cube is None:
            tl, tr = np.float64(self.grid[:2])
            self.log(f"  {cell}: nu văd {color}; îl iau din centrul celulei")
            cube = {"color": color, "xy": self.cell_xy(r, c).tolist(),
                    "yaw": float(np.degrees(np.arctan2(*(tr - tl)[::-1])))}
        others = [{"color": s[-1], "xy": self.cell_xy(int(k[1]) - 1, int(k[2]) - 1).tolist()}
                  for k, s in stacks.items() if s and k != cell]
        return self.fetch(cube, [cube] + others, dz=(level - 1) * C.FETCH_CUBE_M)

    def stack(self, cell, level, color, base_color):
        """Pune cubul din gripper pe nivelul `level` (≥ 2) în `cell`: deasupra cubului de jos (base_of), lăsat de
        TOWER_DZ peste cel de dedesubt, cu degetele pe laturile caroiajului. Apoi îl privește (din poziția din care
        a fost măsurat cubul de jos): mai departe de TOWER_TOL de axă → îl ia și îl pune din nou, corectat
        (self.tfix, păstrat pentru nivelurile următoare). False dacă nu ajunge."""
        a = C.FETCH_CUBE_M
        base, pose = self.base_of(cell, base_color)
        tl, tr = np.float64(self.grid[:2])
        edge = np.degrees(np.arctan2(*(tr - tl)[::-1]))
        for k in range(PLACE_TRIES + 1):
            xy = base - self.tfix
            sol = self.tower_plan(self.grab_xy(xy, PLACE_REACH), edge, TOWER_DZ + (level - 1) * a)
            if sol is None:
                self.log(f"  {cell}: nu ajung la nivelul {level}")
                return False
            down, above, pitch, roll, err = sol
            self.log(f"  pun {color} pe nivelul {level} în {cell} (gripper la {pitch}°, servo 5 rotit {roll:+.0f}°)")
            self.rob.place_at(above, down, cell)
            cube, _ = self.placed_at(cell, color, level, pose)
            if cube is None:
                self.log(f"  {cell}: nu văd cubul de pe nivelul {level}, nu pot verifica")
                return True
            off = np.subtract(cube["xy"], base)
            self.log(f"  {cell} nivelul {level}: la {np.hypot(*off) * 100:.1f} cm de axa turnului "
                     f"({off[0] * 100:+.1f}, {off[1] * 100:+.1f} cm)")
            if np.hypot(*off) <= TOWER_TOL:
                return True
            self.tfix = self.tfix + off
            if k == PLACE_TRIES:
                self.log(f"  {cell}: îl las așa")
                return True
            self.log(f"  {cell}: îl iau și îl pun din nou, corectat")
            if not self.fetch(cube, [cube], dz=(level - 1) * a):
                self.log(f"  {cell}: nu l-am putut lua din nou")
                return True
        return True


def build_tower(fetcher, ctx, cells, moves, target):
    """Sarcina 2 cu cinematica și camera (fără poziții învățate): execută mutările din planner.plan_tower.
    cells = {celulă: culoare/None} văzute pe caroiaj. Întoarce raportul (mutări făcute, turnul)."""
    a = C.FETCH_CUBE_M
    height = max(level for _, dst, level, _ in moves if dst == target) if moves else 0
    xy = fetcher.grab_xy(fetcher.cell_xy(int(target[1]) - 1, int(target[2]) - 1), PLACE_REACH)
    for level in range(1, height + 1):
        if fetcher.tower_plan(xy, 0.0, TOWER_DZ + (level - 1) * a) is None:
            raise RuntimeError(f"Nu ajung la nivelul {level} al turnului în {target}: apropiați foaia de braț")
    stacks = {k: [v] if v else [] for k, v in cells.items()}
    done = []
    for i, (src, dst, level, src_level) in enumerate(moves, 1):
        color = stacks[src][-1]
        base_color = stacks[dst][0] if level > 1 else None
        if level > 1:
            fetcher.base_of(dst, base_color)        # cu gripperul încă gol
        ctx.log(f"[{i}/{len(moves)}] {color}: {src}" + (f" (nivel {src_level})" if src_level > 1 else "")
                + f" → {dst} nivel {level}" + ("" if dst == target else " (eliberez ținta)"))
        if not fetcher.take(src, src_level, color, stacks):
            raise RuntimeError(f"Nu am putut lua {color} din {src}")
        stacks[src].pop()
        if level == 1:
            if not fetcher.place_checked(dst, color):
                fetcher.rob.gripper(C.GRIP_OPEN)
                raise RuntimeError(f"Nu ajung la {dst}")
            fetcher.axis.pop(dst, None)             # pus acum: axa se măsoară din nou, de unde a ajuns
        elif not fetcher.stack(dst, level, color, base_color):
            fetcher.rob.gripper(C.GRIP_OPEN)
            raise RuntimeError(f"Nu ajung la nivelul {level} în {dst}")
        stacks[dst].append(color)
        done.append((src, dst, level))
    ctx.log(f"Turnul din {target}, de jos în sus: {' → '.join(stacks[target])}")
    return {"moves": done, "tower": stacks[target]}


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
        # celula cea mai depărtată care mai așteaptă și pentru care am un cub (brațul nu trece peste cuburile puse);
        # dintre cuburile de culoarea ei, cel cu cel mai mult loc liber în jur
        color = next(color for _, color in todo if any(c["color"] == color for c in cand))
        cube = max((c for c in cand if c["color"] == color), key=lambda c: fetcher.free(c, cubes))
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
