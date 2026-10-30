"""Imaginea mare a mesei, văzută de sus, din mai multe priviri (centru, stânga, dreapta, sus, jos, diagonale).

Când caroiajul nu încape întreg într-o imagine (de obicei se vede măcar pătratul din mijloc), brațul se uită în jur:
1. Fiecare cadru e proiectat pe masă din poza camerei (kinematics, din unghiurile brațului).
2. Modelul greșește cu câțiva mm / grade, deci privirile sunt apoi potrivite după imagine: aceleași detalii (colțuri
   de linii, desenele de pe cuburi, marginile foii) găsite în privirea nouă și în ce e deja lipit dau mutarea și
   rotirea care o aliniază. Fiecare punct e luat din privirea în care e cel mai aproape de mijlocul imaginii.
3. Se fac două imagini mari: la înălțimea mesei (aici se văd liniile caroiajului) și la înălțimea cubului (aici fața
   de sus a fiecărui cub, singura colorată, cade fix peste celula lui).
4. Caroiajul e căutat pe imaginea mesei ca rețea de 4 + 4 linii drepte, egal depărtate (find_lattice): liniile
   acoperite parțial de cuburi contează și ele, contururile cuburilor nu fac 4 linii egal depărtate.
5. Culorile: din centrul fiecărei celule, pe imaginea de la înălțimea cubului.
Colțurile caroiajului ies direct în metri, pe masă (ca fetch.grid_area).
"""
import os
import time

import cv2
import numpy as np

from . import config as C
from . import fetch
from . import vision

MM = 0.0005         # m pe pixel în imaginea de sus
HALF = 0.17         # m: cât acoperă imaginea de sus în jurul punctului privit din centru, în fiecare direcție
# Privirile, ca (baza, umărul) în grade față de poziția din centru. Baza + = stânga, umărul + = mai departe (sus),
# umărul - = mai aproape de braț (rândul -20: marginea caroiajului dinspre braț, cu linia de jos și ultimul rând).
# Întâi cele de pe coloana din mijloc (se potrivesc cu centrul), apoi lateralele (cu ce s-a lipit deja).
VIEWS = [(0, 0), (0, 8), (0, -8), (0, -20), (12, 0), (-12, 0), (12, 8), (-12, 8), (12, -8), (-12, -8),
         (12, -20), (-12, -20)]                                                                     # REGLEAZĂ
SETTLE_S = 0.4      # cât stă camera după o mișcare, înainte de cadru
HORIZON_K = 1.0     # m: rândurile de imagine care privesc masa mai departe de atât nu se folosesc (prea oblic)
REG_MAX_PX = 60     # potrivirea după imagine: modelul greșește cu cel mult atâția pixeli ai imaginii de sus
REG_MIN = 8         # potriviri bune necesare; altfel privirea rămâne unde a pus-o modelul
CELL_M = (0.025, 0.06)   # latura unei celule a caroiajului, cât poate fi (m)
LATTICE_ANGLE = 15  # grade: cât de rotit poate fi caroiajul în imaginea de sus


def plane_map(angles, size, center, z):
    """(H, rândul de sus folosit): H duce imaginea camerei pe imaginea de sus (planul de la înălțimea z), rândurile
    de deasupra rândului întors privesc prea departe / peste orizont. None dacă nu privește deloc masa."""
    w, h = size
    for a in np.arange(0.0, 0.95, 0.05):
        src = np.float64([[0, a * h], [w - 1, a * h], [w - 1, h - 1], [0, h - 1]])
        xy, ok = fetch.to_table(angles, src, size, z)
        if ok.all() and np.hypot(*(xy - center).T).max() < HORIZON_K:
            dst = np.column_stack([(center[1] + HALF - xy[:, 1]) / MM, (center[0] + HALF - xy[:, 0]) / MM])
            return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst)), int(a * h)
    return None, None


def weights(size, top):
    """Cât de bun e fiecare pixel al camerei: 1 în mijloc, spre 0 la margini; 0 deasupra rândului `top`."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    wt = 1.0 - np.maximum(np.abs(xx / w - 0.5), np.abs(yy / h - 0.5)) * 2
    wt[:top] = 0
    wt[:3], wt[-3:], wt[:, :3], wt[:, -3:] = 0, 0, 0, 0
    return np.clip(wt, 0, 1)


def register(img, wt, ref, ref_wt):
    """Mutarea + rotirea (3x3) care aliniază imaginea de sus `img` peste `ref`, după detaliile comune; None dacă nu
    găsește destule."""
    det = cv2.SIFT_create(3000)
    g1, g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    k1, d1 = det.detectAndCompute(g1, (wt > 0.05).astype(np.uint8))
    k0, d0 = det.detectAndCompute(g0, (ref_wt > 0.05).astype(np.uint8))
    if d0 is None or d1 is None or len(k0) < 2 or len(k1) < 2:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d0, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    good = [m for m in good if np.hypot(*np.subtract(k1[m.queryIdx].pt, k0[m.trainIdx].pt)) < REG_MAX_PX]
    if len(good) < REG_MIN:
        return None
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p0 = np.float32([k0[m.trainIdx].pt for m in good])
    A, inl = cv2.estimateAffinePartial2D(p1, p0, method=cv2.RANSAC, ransacReprojThreshold=3)
    if A is None or inl.sum() < REG_MIN:
        return None
    return np.vstack([A, [0, 0, 1]])


def _best4(prof, smin, smax):
    """(scor, start, pas): 4 vârfuri egal depărtate în profil (suma lor, plus de două ori cel mai slab: toate 4
    liniile trebuie să existe)."""
    p = np.convolve(prof, np.ones(3), "same")
    n, best = len(p), (-1.0, 0, 0)
    for s in range(smin, smax + 1):
        if 3 * s >= n:
            break
        four = [p[k * s:n - (3 - k) * s] for k in range(4)]
        sc = sum(four) + 2 * np.minimum.reduce(four)
        i = int(np.argmax(sc))
        if sc[i] > best[0]:
            best = (float(sc[i]), i, s)
    return best


def find_lattice(mask, smin, smax):
    """Caroiajul 3x3 pe o imagine văzută de sus: colțurile (stânga-sus, dreapta-sus, dreapta-jos, stânga-jos), sau
    None. Pentru fiecare rotire, liniile dau vârfuri în suma măștii pe rânduri / coloane."""
    h, w = mask.shape
    best = None
    for ang in np.arange(-LATTICE_ANGLE, LATTICE_ANGLE + 0.01, 0.5):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        r = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        sy = _best4(r.sum(1).astype(float) / 255, smin, smax)
        sx = _best4(r.sum(0).astype(float) / 255, smin, smax)
        if abs(sx[2] - sy[2]) > 0.2 * max(sx[2], sy[2], 1):     # văzut de sus, celulele sunt (aproape) pătrate
            continue
        if best is None or sx[0] + sy[0] > best[0]:
            best = (sx[0] + sy[0], M, sx, sy)
    if best is None:
        return None
    _, M, sx, sy = best
    x0, y0, x3, y3 = sx[1], sy[1], sx[1] + 3 * sx[2], sy[1] + 3 * sy[2]
    Mi = cv2.invertAffineTransform(M)
    return np.float32([[x0, y0], [x3, y0], [x3, y3], [x0, y3]]) @ Mi[:, :2].T + Mi[:, 2]


class Panorama:
    def __init__(self, center):
        """center = punctul de pe masă (m) în jurul căruia se face imaginea de sus."""
        self.center = np.float64(center)
        n = int(round(2 * HALF / MM))
        self.size = (n, n)
        self.views = []     # (cadru, unghiuri)
        self.img = {}

    def add(self, frame, angles):
        self.views.append((frame, list(angles)))

    def compose(self, log=print):
        """Lipește privirile: proiecția din model, corectată după imagine (vezi register)."""
        n = self.size[0]
        hs = C.FETCH_CUBE_M
        self.img = {z: np.zeros((n, n, 3), np.uint8) for z in (0.0, hs)}
        best = {z: np.zeros((n, n), np.float32) for z in self.img}
        fixed = 0
        for k, (frame, angles) in enumerate(self.views):
            size = (frame.shape[1], frame.shape[0])
            H0, top = plane_map(angles, size, self.center, 0.0)
            if H0 is None:
                continue
            wt0 = weights(size, top)
            A = np.eye(3)
            if k > 0:
                img = cv2.warpPerspective(frame, H0, self.size)
                A = register(img, cv2.warpPerspective(wt0, H0, self.size), self.img[0.0], best[0.0])
                if A is None:
                    A = np.eye(3)
                else:
                    fixed += 1
            for z in self.img:
                H, top = (H0, top) if z == 0.0 else plane_map(angles, size, self.center, z)
                if H is None:
                    continue
                H = A @ H
                warped = cv2.warpPerspective(frame, H, self.size)
                wt = cv2.warpPerspective(weights(size, top), H, self.size)
                better = wt > best[z]
                self.img[z][better] = warped[better]
                best[z][better] = wt[better]
        log(f"  Am lipit {len(self.views)} priviri ({fixed} potrivite după imagine, restul după model).")

    def to_table(self, pts):
        """Pixeli din imaginea de sus → puncte pe masă (m)."""
        p = np.float64(pts)
        return np.column_stack([self.center[0] + HALF - p[:, 1] * MM, self.center[1] + HALF - p[:, 0] * MM])

    def to_pixels(self, table):
        """Puncte pe masă (m) → pixeli din imaginea de sus."""
        t = np.float64(table)
        return np.column_stack([(self.center[1] + HALF - t[:, 1]) / MM, (self.center[0] + HALF - t[:, 0]) / MM])

    def colors(self, corners):
        """Culorile 3x3 din centrul fiecărei celule, pe imaginea de la înălțimea cubului (colțuri în pixeli)."""
        top = vision.warp(vision.normalize_light(self.img[C.FETCH_CUBE_M]), corners)
        cell = vision.WARP // 3
        m = int(cell * (1 - C.PATCH) / 2)
        fracs = [[vision.fractions(top[r * cell + m:(r + 1) * cell - m, c * cell + m:(c + 1) * cell - m])
                  for c in range(3)] for r in range(3)]
        return [[max(f, key=f.get) if max(f.values()) >= C.MIN_FRACTION else None for f in row] for row in fracs]

    def read(self, table):
        """(colțurile în pixeli, culorile 3x3) ale caroiajului cunoscut (m, pe masă): fără să-l mai caute."""
        if not self.img:
            self.compose()
        corners = self.to_pixels(table).tolist()
        return corners, self.colors(corners)

    def detect(self):
        """(colțurile caroiajului în pixeli ai imaginii de sus, pe masă în m, culorile 3x3) sau (None, None, None)."""
        if not self.img:
            self.compose()
        small, s = vision.shrink(self.img[0.0], vision.GRID_W)
        found = find_lattice(vision.grid_mask(small), int(CELL_M[0] / MM * s), int(CELL_M[1] / MM * s))
        if found is None:
            return None, None, None
        corners = (found / s).tolist()
        return corners, self.to_table(corners).tolist(), self.colors(corners)

    def save(self, corners=None):
        """Imaginile de sus în captures/ (cu caroiajul desenat, dacă e găsit). Întoarce calea primei."""
        os.makedirs(C.CAPTURES_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        paths = []
        for z, name in ((0.0, "masa"), (C.FETCH_CUBE_M, "cuburi")):
            img = self.img[z].copy()
            if corners is not None:
                H = cv2.getPerspectiveTransform(np.float32([[0, 0], [3, 0], [3, 3], [0, 3]]), np.float32(corners))
                for k in range(4):
                    for a, b in (([k, 0], [k, 3]), ([0, k], [3, k])):
                        p = cv2.perspectiveTransform(np.float32([[a, b]]), H)[0]
                        cv2.line(img, tuple(int(x) for x in p[0]), tuple(int(x) for x in p[1]), (0, 0, 255), 2)
            path = os.path.join(C.CAPTURES_DIR, f"panorama_{stamp}_{name}.png")
            cv2.imwrite(path, img)
            paths.append(path)
        return paths[0]


def look_point(angles, size):
    """Unde privește mijlocul imaginii, pe masă (m), sau None."""
    xy, ok = fetch.to_table(angles, [[size[0] / 2, size[1] / 2]], size, 0.0)
    return xy[0] if ok.all() else None


def cover_views(angles, table, size):
    """Caroiajul cunoscut (m, pe masă): doar privirile din VIEWS de care e nevoie ca fiecare celulă (centrul și fața
    cubului de deasupra) să se vadă bine în măcar una, alese din cinematică (cea care aduce cele mai multe celule)."""
    from .aim import project
    t = np.float64(table)
    tl, tr, br, bl = t
    pts = []
    for r in range(3):
        for c in range(3):
            u, v = (c + 0.5) / 3, (r + 0.5) / 3
            xy = (1 - v) * ((1 - u) * tl + u * tr) + v * ((1 - u) * bl + u * br)
            pts += [[xy[0], xy[1], 0.0], [xy[0], xy[1], C.FETCH_CUBE_M]]
    pts = np.float64(pts)

    def sees(view):
        pose = list(angles)
        pose[0] += view[0]
        pose[1] += view[1]
        p = project(pose, pts, size)
        if p is None:
            return np.zeros(9, bool)
        return ((p > 0.1) & (p < 0.9)).all(1).reshape(9, 2).all(1)

    seen, chosen = np.zeros(9, bool), []
    while not seen.all():
        best = max(VIEWS, key=lambda v: (int((sees(v) & ~seen).sum()), -abs(v[0]) - abs(v[1])))
        new = sees(best) & ~seen
        if not new.any():
            break
        chosen.append(best)
        seen |= new
    return chosen or [(0, 0)]


def build(rob, get_frame, log, angles, size=(640, 480), table=None):
    """Brațul se uită în jur din `angles` și face imaginea mare: toate privirile din VIEWS, sau, cu caroiajul cunoscut
    (table, m pe masă), doar cele care îl acoperă (cover_views). Întoarce Panorama sau None."""
    center = look_point(angles, size) if table is None else np.float64(table).mean(0)
    if center is None:
        log("Camera nu privește masa: nu pot face imaginea mare.")
        return None
    pano = Panorama(center)
    views = VIEWS if table is None else cover_views(angles, table, size)
    log(f"Imaginea mare: mă uit în jur din {[round(a) for a in angles]} ({len(views)} priviri)")
    for db, ds in views:
        pose = list(angles)
        pose[0] += db
        pose[1] += ds
        if not all(C.LIMITS[j][0] <= pose[j] <= C.LIMITS[j][1] for j in (0, 1)):
            continue
        rob.move(pose, 600)
        rob.sleep(SETTLE_S)
        pano.add(get_frame(), pose)
    rob.move(list(angles), 600)
    pano.compose(log)
    return pano
