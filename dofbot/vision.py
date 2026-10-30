import json
import os
import time
from collections import Counter, deque

import cv2
import numpy as np

from . import camera
from . import config as C

WARP = 300     # imaginea îndreptată are 300x300 px, 100 px pe celulă
PROC_W = 320   # cuburile se caută pe imaginea micșorată la lățimea asta (de ~4 ori mai rapid)
GRID_W = 480   # caroiajul: la 320 px liniile văzute oblic au sub 1 px și se pierd
GRID_MIN_BORDER = 0.25   # cât din fiecare margine a caroiajului trebuie să se vadă (restul, sub cuburi)


def camera_backends():
    """Pe Windows, DirectShow pornește camera mult mai repede, dar unele laptopuri (camere Intel MIPI,
    Surface...) dau imagine doar prin Media Foundation. Le încercăm pe rând."""
    if os.name == "nt":
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
    return [("ANY", cv2.CAP_ANY)]


_WORKED = {}   # camera -> backendul care a mers (la reconectare îl încercăm primul)


def is_infrared(frame):
    """Camerele IR (Windows Hello) dau imagine alb-negru: cele 3 canale sunt aproape egale peste tot."""
    small = cv2.resize(frame, (64, 48)).astype(np.int16)
    return float(np.abs(small[..., 0] - small[..., 2]).mean()) < 2 and \
        float(np.abs(small[..., 1] - small[..., 2]).mean()) < 2


def imread(path):
    """cv2.imread care merge și cu diacritice în cale (pe Windows cv2.imread/imwrite nu le suportă)."""
    try:
        data = np.fromfile(path, np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img):
    """cv2.imwrite care merge și cu diacritice în cale. Întoarce True dacă a scris."""
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img)
    if ok:
        buf.tofile(path)
    return bool(ok)


def open_camera(index=None):
    """Deschide camera cu primul backend care chiar trimite cadre color (nu doar „se deschide”)."""
    index = C.CAM_INDEX if index is None else index
    backends = sorted(camera_backends(), key=lambda b: b[0] != _WORKED.get(index))
    tried = []
    # Pe Windows, camerele USB dau 30 cadre/s la 640x480 doar comprimat (MJPG); altfel unele scad la 5-15.
    # Dacă o cameră nu dă imagine în MJPG, o încercăm și în formatul ei implicit.
    attempts = [(name, backend, mjpg) for name, backend in backends
                for mjpg in ((True, False) if os.name == "nt" else (False,))]
    for name, backend, mjpg in attempts:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            if not mjpg or os.name != "nt":
                tried.append(f"{name}: nu se deschide")
            continue
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_H)
        cap.set(cv2.CAP_PROP_FPS, C.CAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # imaginea cea mai nouă, nu una de acum câteva cadre
        frame = None
        for _ in range(15):                    # primele citiri pot eșua cât pornește camera
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
                break
        if frame is None:
            cap.release()
            if not mjpg:
                tried.append(f"{name}: fără cadre")
            continue
        if frame.ndim == 3 and is_infrared(frame):
            cap.release()
            tried.append(f"{name}: imagine alb-negru (cameră infraroșu?)")
            continue
        _WORKED[index] = name
        failed = camera_controls(cap)
        if failed:
            print(f"[camera] {index}: nu merg controalele: {', '.join(failed)}")
        return cap
    raise RuntimeError(f"Nu pot deschide camera {index} ({'; '.join(tried)}). Folosită de altă aplicație? "
                       f"Permisiune cameră? Alege alta din pagină")


def grab():
    if camera.shared is not None:     # pornită de web.py / task1 / task2
        frame = camera.shared.fresh()[0]
        if camera.shared.error:
            raise RuntimeError(camera.shared.error)
        return frame
    if C.TEST_IMAGE == "synth":
        from . import synth
        from .sim_world import WORLD
        return synth.render(WORLD.snapshot(), seed=int(time.time() * 1000) % 10000)
    if C.TEST_IMAGE:
        frame = imread(C.TEST_IMAGE)
        if frame is None:
            raise RuntimeError(f"Nu pot citi imaginea {C.TEST_IMAGE}")
        return frame
    cap = open_camera()
    time.sleep(0.3)
    frame = None
    for _ in range(8):          # aruncă primele cadre (expunere automată)
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise RuntimeError("Camera nu întoarce imagini")
    return frame


def load_corners(grid):
    try:
        with open(C.VISION_FILE, encoding="utf-8") as f:
            return json.load(f)[grid]   # [[x,y] x4]: stânga-sus, dreapta-sus, dreapta-jos, stânga-jos
    except (FileNotFoundError, KeyError):
        raise RuntimeError(f"Caroiajul {grid} nu e găsit automat și nici calibrat "
                           f"(deschide pagina web.py și pune colțurile din Setări)")


def warp(frame, corners):
    src = np.float32(corners)
    dst = np.float32([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (WARP, WARP))


def order_corners(pts):
    """4 puncte -> [stânga-sus, dreapta-sus, dreapta-jos, stânga-jos]."""
    pts = np.float32(pts).reshape(4, 2)
    s = pts.sum(1)
    d = pts[:, 1] - pts[:, 0]
    return [pts[i].tolist() for i in (np.argmin(s), np.argmin(d), np.argmax(s), np.argmax(d))]


def shrink(frame, width=PROC_W):
    """(imaginea micșorată la `width`, factorul de scară mic/original)."""
    h, w = frame.shape[:2]
    if w <= width:
        return frame, 1.0
    s = width / w
    return cv2.resize(frame, (width, round(h * s)), interpolation=cv2.INTER_AREA), s


# ---------- lumina ----------
# Foaia albă e în fiecare cadru: o aducem mereu la aceeași luminozitate și la gri (balans de alb). Așa
# intervalele HSV din COLORS (mai ales pragurile V) înseamnă „față de hârtie”, nu valori absolute, și un verde
# închis în lumină slabă rămâne verde. S (saturația) nu se schimbă cu lumina, doar cu nuanța ei.
PAPER_V = 220         # la cât aducem foaia albă
PAPER_MAX_S = 130     # „aproape gri”: foaia, și sub un bec cald; fețele colorate ale cuburilor au S peste
PAPER_PCT = 80        # foaia = cele mai luminoase (peste această percentilă) dintre pixelii aproape gri
DENOISE_GAIN = 1.8    # de la acest câștig zgomotul senzorului se vede: filtru median


def paper_color(img):
    """Culoarea foii albe (B, G, R) în imagine: mediana celor mai luminoși pixeli aproape gri. None dacă nu e."""
    h, w = img.shape[:2]
    small = img if w <= 160 else cv2.resize(img, (160, max(1, round(h * 160 / w))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    gray = (hsv[..., 1] < PAPER_MAX_S) & (v > 8)     # negrul pur = fără date (marginile imaginii mari), nu hârtie
    if np.count_nonzero(gray) < 50:
        return None
    paper = gray & (v >= np.percentile(v[gray], PAPER_PCT))
    return np.median(small[paper].astype(np.float32), axis=0)


def light_gains(img):
    """Câștigurile pe canal (B, G, R) care aduc foaia albă la PAPER_V, sau None dacă nu se vede nimic gri.
    Doar luminăm (câștig ≥ 1), cel mult LIGHT_MAX_GAIN ori."""
    ref = paper_color(img)
    return None if ref is None else np.clip(PAPER_V / np.maximum(ref, 1.0), 1.0, C.LIGHT_MAX_GAIN)


# ---------- controalele camerei (Linux / V4L2) ----------
EXPOSE_TARGET = 200     # auto_expose: la ce luminozitate aducem foaia (sub 255, ca să nu se ardă)
EXPOSE_RANGE = (1, 5000)


def camera_controls(cap):
    """Aplică CAM_EXPOSURE, CAM_GAIN, CAM_WB din setări (0 / -1 = lăsat pe automat). Pe brațul care se mișcă,
    expunerea automată schimbă culorile de la o poză la alta; fixă, ele rămân la fel. Întoarce ce nu a mers
    (nu toate camerele au toate controalele). Doar pe Linux: pe Windows valorile au alt sens."""
    if os.name == "nt" or (C.CAM_EXPOSURE <= 0 and C.CAM_GAIN < 0 and C.CAM_WB <= 0):
        return []
    failed = []

    def put(prop, value, name):
        if not cap.set(prop, value):
            failed.append(name)

    if C.CAM_EXPOSURE > 0:
        put(cv2.CAP_PROP_AUTO_EXPOSURE, 1, "expunere manuală")     # V4L2: 1 = manual, 3 = automat
        put(cv2.CAP_PROP_EXPOSURE, C.CAM_EXPOSURE, "expunere")
    else:
        put(cv2.CAP_PROP_AUTO_EXPOSURE, 3, "expunere automată")
    if C.CAM_GAIN >= 0:
        put(cv2.CAP_PROP_GAIN, C.CAM_GAIN, "amplificare")
    if C.CAM_WB > 0:
        put(cv2.CAP_PROP_AUTO_WB, 0, "balans de alb manual")
        put(cv2.CAP_PROP_WB_TEMPERATURE, C.CAM_WB, "balans de alb")
    else:
        put(cv2.CAP_PROP_AUTO_WB, 1, "balans de alb automat")
    return failed


def auto_expose(cap, get_frame, log=print, tries=8):
    """Expunere manuală reglată până când foaia albă are V ≈ EXPOSE_TARGET, apoi lăsată fixă (în CAM_EXPOSURE,
    se păstrează cu „Salvează”). get_frame() = un cadru nou. Întoarce expunerea sau None dacă nu a mers."""
    exp = C.CAM_EXPOSURE if C.CAM_EXPOSURE > 0 else (cap.get(cv2.CAP_PROP_EXPOSURE) or 0)
    exp = exp if exp > 0 else 150
    if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1):
        log("Camera nu acceptă expunere manuală.")
        return None
    for i in range(tries):
        cap.set(cv2.CAP_PROP_EXPOSURE, exp)
        for _ in range(3):              # cadrele deja pornite au expunerea veche
            frame = get_frame()
        ref = paper_color(frame)
        if ref is None:
            log("Nu văd foaia albă: pune caroiajul în fața camerei.")
            return None
        v = float(ref.max())
        log(f"  expunere {exp:.0f}: foaia are {v:.0f}")
        if abs(v - EXPOSE_TARGET) < 0.08 * EXPOSE_TARGET:
            break
        # Ars (255): nu știm cât de mult, deci doar înjumătățim; altfel luminozitatea e ~ proporțională cu expunerea.
        exp = exp / 2 if v >= 250 else exp * EXPOSE_TARGET / max(v, 5.0)
        exp = float(np.clip(exp, *EXPOSE_RANGE))
    C.apply_settings({"CAM_EXPOSURE": int(round(exp))})
    return C.CAM_EXPOSURE


def normalize_light(img):
    """Imaginea cu foaia adusă la alb (vezi light_gains). Fără efect cu LIGHT_NORM oprit sau în lumină bună.
    Când câștigul e mare, și un filtru median: zgomotul amplificat ar strica și culorile, și liniile negre
    (pe un pixel aproape negru, saturația e doar zgomot)."""
    if not C.LIGHT_NORM:
        return img
    g = light_gains(img)
    if g is None or g.max() < 1.03:
        return img
    lut = np.clip(np.arange(256, dtype=np.float32)[:, None] * g[None, :], 0, 255).astype(np.uint8)[None]
    out = cv2.LUT(img, lut)
    return cv2.medianBlur(out, 3) if g.max() >= DENOISE_GAIN else out


def grid_mask(frame):
    """Pixelii negri ai caroiajului: linii subțiri mai întunecate decât hârtia din jur și nesaturate.
    Black-hat-ul păstrează doar ce e mai îngust decât fereastra lui, deci cuburile (mari) nu intră în mască."""
    hsv = cv2.cvtColor(normalize_light(frame), cv2.COLOR_BGR2HSV)
    k = max(5, (frame.shape[1] // 50) | 1)
    bh = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    dark = np.where(bh > C.GRID_C, 255, 0).astype(np.uint8)
    dark[hsv[..., 1] > C.GRID_MAX_S] = 0
    # liniile subțiri văzute oblic ies punctate: le unim
    return cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def grid_score(m, cov=None):
    """m = masca îndreptată. Mare dacă liniile interioare sunt negre și celulele albe.
    Dacă una din cele 4 linii interioare lipsește (ex. o fereastră cu 3 ochiuri), scorul e 0.
    cov = zona acoperită de cuburi (îndreptată): acolo nu știm nimic, deci nu penalizăm."""
    cell, band = WARP // 3, WARP // 40
    cells = m
    if cov is not None:
        m, cells = m | cov, m & ~cov
    lines = []
    for k in (1, 2):
        p = k * cell
        lines.append(m[:, p - band:p + band].any(axis=1).mean())
        lines.append(m[p - band:p + band, :].any(axis=0).mean())
    if min(lines) < C.GRID_MIN_LINE:
        return 0.0
    # Și marginile trebuie să fie pe linii (măcar parțial: cuburile le pot acoperi). Altfel un caroiaj
    # mutat cu un rând trece și el: liniile lui interioare cad tot pe linii adevărate.
    borders = [m[:, :2 * band].any(axis=1).mean(), m[:, -2 * band:].any(axis=1).mean(),
               m[:2 * band, :].any(axis=0).mean(), m[-2 * band:, :].any(axis=0).mean()]
    if min(borders) < GRID_MIN_BORDER:
        return 0.0
    b = 2 * band
    inside = np.mean([cells[r * cell + b:(r + 1) * cell - b, c * cell + b:(c + 1) * cell - b].mean() / 255
                      for r in range(3) for c in range(3)])
    return float(np.mean(lines) - inside)


def squareish(corners):
    """Caroiajul e pătrat: chiar văzut oblic, lățimea și înălțimea nu diferă de mai mult de GRID_MAX_RATIO ori."""
    tl, tr, br, bl = np.float32(corners)
    wd = np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)
    ht = np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)
    return 1 / C.GRID_MAX_RATIO < wd / max(ht, 1) < C.GRID_MAX_RATIO


# Puncte în coordonatele caroiajului (0..3): de-a lungul celor 8 linii, și în interiorul celulelor.
_T = np.linspace(0.02, 2.98, 40)
_LINE_PTS = np.vstack([np.column_stack([np.full_like(_T, k), _T]) for k in range(4)] +
                      [np.column_stack([_T, np.full_like(_T, k)]) for k in range(4)])
_G = np.linspace(0.25, 0.75, 4)
_CELL_PTS = np.array([[c + a, r + b] for r in range(3) for c in range(3) for a in _G for b in _G])
_FIT_PTS = np.float32(np.vstack([_LINE_PTS, _CELL_PTS]))[None]
_SQUARE = np.float32([[0, 0], [3, 0], [3, 3], [0, 3]])
FIT_GOOD = 0.7      # peste acest scor colțurile sunt bune; sub el le ajustăm pe liniile văzute
MAX_COVERED = 0.8   # un caroiaj acoperit de cuburi mai mult de atât nu poate fi verificat
FIT_COVERED = 0.85  # cu cuburi peste caroiaj, liniile rămase vizibile trebuie să se potrivească foarte bine
MAX_NUDGE = 6       # caroiaj aproape acoperit: cât (px, la GRID_W) are voie să se mute față de cadrul anterior


def grid_points(pts, corners):
    """Puncte din coordonatele caroiajului (0..3) în imagine, ca pixeli întregi. Cele care cad la infinit
    (caroiaj extrapolat mult în afara imaginii) ies în afara oricărei imagini, nu dau erori."""
    H = cv2.getPerspectiveTransform(_SQUARE, np.float32(corners))
    q = np.nan_to_num(cv2.perspectiveTransform(pts, H)[0], nan=-1e6, posinf=-1e6, neginf=-1e6)
    return np.rint(np.clip(q, -1e6, 1e6)).astype(np.int32)


def fit_score(near, mask, corners, cov=None):
    """Cât de bine se suprapune un caroiaj 3x3 cu aceste colțuri peste liniile negre:
    partea din cele 8 linii care cade pe negru, minus partea din celule care e neagră.
    Punctele din `cov` (acoperite de cuburi) nu contează deloc."""
    p = grid_points(_FIT_PTS, corners)
    h, w = mask.shape
    ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    x, y = np.where(ok, p[:, 0], 0), np.where(ok, p[:, 1], 0)
    n = len(_LINE_PTS)
    if cov is None:
        lines = np.count_nonzero(near[y[:n], x[:n]] & ok[:n])
        cells = np.count_nonzero(mask[y[n:], x[n:]] & ok[n:])
        return lines / n - cells / (len(ok) - n)
    seen = ok & (cov[y, x] == 0)
    lines = np.count_nonzero(near[y[:n], x[:n]] & seen[:n]) / max(np.count_nonzero(seen[:n]), 1)
    cells = np.count_nonzero(mask[y[n:], x[n:]] & seen[n:]) / max(np.count_nonzero(seen[n:]), 1)
    return lines - cells


def fit_grid(mask, corners, steps=(8, 4, 2, 1), cov=None):
    """Mută colțurile câte puțin cât timp caroiajul 3x3 se potrivește mai bine pe liniile văzute.
    Așa găsim și un colț acoperit de un cub: îl dau liniile rămase vizibile."""
    near = cv2.dilate(mask, np.ones((3, 3), np.uint8))   # o linie contează și dacă trece la 1 px
    c = np.float32(corners)
    best = fit_score(near, mask, c, cov)
    for st in steps:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for j in range(2):
                    for sgn in (1, -1):
                        t = c.copy()
                        t[i, j] += sgn * st
                        sc = fit_score(near, mask, t, cov)
                        if sc > best + 1e-4:
                            best, c, improved = sc, t, True
    return c.tolist()


def grid_candidates(mask):
    """Patrulaterele conturului exterior al liniilor, cele mai mari primele."""
    h, w = mask.shape
    # liniile subțiri ies uneori punctate: le îngroșăm doar pentru căutarea conturului
    search = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(search, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        if not C.GRID_MIN_AREA * h * w < cv2.contourArea(cnt) < 0.95 * h * w:
            continue
        hull = cv2.convexHull(cnt)
        for eps in (0.02, 0.03, 0.045, 0.06, 0.08):   # un cub lipit de margine adaugă un colț în plus
            quad = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
            if len(quad) <= 4:
                break
        if len(quad) == 4 and squareish(order_corners(quad)):
            out.append(order_corners(quad))
    return out


def cell_candidates(mask, near):
    """Caroiaje deduse din celulele albe închise de linii (cele neacoperite de cuburi).
    Nu știm care celulă e, așa că încercăm toate cele 9 poziții; cele mai potrivite primele."""
    h, w = mask.shape
    contours, _ = cv2.findContours(255 - near, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    out = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not C.GRID_MIN_AREA / 9 * h * w < area < 0.2 * h * w:
            continue
        hull = cv2.convexHull(cnt)
        if area < 0.85 * cv2.contourArea(hull):
            continue
        quad = cv2.approxPolyDP(hull, 0.06 * cv2.arcLength(hull, True), True)
        if len(quad) != 4 or not squareish(order_corners(quad)):
            continue
        H = cv2.getPerspectiveTransform(unit, np.float32(order_corners(quad)))
        for r in range(3):
            for c in range(3):
                g = np.float32([[-c, -r], [3 - c, -r], [3 - c, 3 - r], [-c, 3 - r]])
                corners = cv2.perspectiveTransform(g[None], H)[0]
                out.append((fit_score(near, mask, corners), corners.tolist()))
    return [c for _, c in sorted(out, key=lambda t: -t[0])]


def find_grid(frame, prev=None):
    """Caută caroiajul negru 3x3. Întoarce cele 4 colțuri (în imaginea originală) sau None.
    prev = colțurile găsite la cadrul anterior: de acolo se pornește când cuburile acoperă liniile."""
    full = frame
    frame, s = shrink(frame, GRID_W)
    mask = grid_mask(frame)
    best, best_score = None, C.GRID_MIN_SCORE
    sure = False    # cel mai bun caroiaj de până acum e sigur: nu mai căutăm altul
    hidden = False  # caroiajul anterior e aproape tot acoperit: nici căutarea completă nu-l poate găsi

    near = cv2.dilate(mask, np.ones((3, 3), np.uint8))

    def consider(corners):
        nonlocal best, best_score, sure, hidden
        if not squareish(corners):
            return
        covered = cube_cover(full, [[x / s, y / s] for x, y in corners], s, mask.shape)
        if covered is None:
            score = grid_score(warp(mask, corners))
            if score < FIT_GOOD:
                corners = fit_grid(mask, corners)
                score = grid_score(warp(mask, corners)) if squareish(corners) else 0.0
        else:
            # Cuburile acoperă liniile, iar laturile lor (umbrite, cu desene) ar trage potrivirea:
            # potrivim și judecăm caroiajul doar pe ce nu e acoperit de ele.
            corners = fit_grid(mask, corners, cov=covered)
            wc = warp(covered, corners)
            if not squareish(corners) or wc.mean() / 255 > MAX_COVERED:
                hidden = hidden or wc.mean() / 255 > MAX_COVERED
                return
            fit = fit_score(near, mask, corners, covered)
            score = min(grid_score(warp(mask, corners), wc), fit) if fit >= FIT_COVERED else 0.0
        if score > best_score:
            # Acoperit de cuburi, scorul nu poate fi mare, dar liniile rămase s-au potrivit foarte bine
            # (FIT_COVERED): o căutare completă n-ar găsi altceva mai bun, doar ar costa.
            best, best_score = corners, score
            sure = score >= FIT_GOOD or covered is not None

    if prev is not None:     # caroiajul e fix pe masă: de obicei ajunge o mică ajustare
        consider([[x * s, y * s] for x, y in prev])
    if not sure and not hidden:   # căutare completă (și când camera s-a mutat): contur exterior + celule
        outer = grid_candidates(mask)
        if outer:
            consider(max(outer, key=lambda c: fit_score(near, mask, c)))
        if not sure:
            for corners in cell_candidates(mask, near)[:3]:
                consider(corners)
    if best is None and prev is not None:
        # Caroiajul aproape plin: liniile vizibile sunt prea puține ca să găsim altul, dar caroiajul stă fix
        # pe masă. Păstrăm colțurile anterioare dacă ce se mai vede din linii se potrivește încă cu ele.
        covered = cube_cover(full, prev, s, mask.shape)
        if covered is not None:
            start = np.float32([[x * s, y * s] for x, y in prev])
            fitted = np.float32(fit_grid(mask, start.tolist(), steps=(2, 1), cov=covered))
            if np.abs(fitted - start).max() > MAX_NUDGE:   # cu atât de puține linii, nu-l lăsăm să fugă
                fitted = start
            if fit_score(near, mask, fitted.tolist(), covered) >= FIT_COVERED:
                return (fitted / s).tolist()
    if best is None:
        return grid_from_cubes(full)    # caroiaj plin: din rețeaua fețelor de sus ale cuburilor
    return None if best is None else [[x / s, y / s] for x, y in best]


VERIFY_OK = 0.6       # verify_grid: de la acest scor, caroiajul ținut minte chiar e încă acolo
VERIFY_MIN_SEEN = 24  # sub atâtea puncte de linie neacoperite de cuburi, nu putem spune nimic


def verify_grid(frame, corners):
    """Caroiajul cu aceste colțuri chiar se vede în cadru? Scor 0..1 (cât din liniile neacoperite de cuburi
    cad pe negru, minus celulele negre) sau None dacă aproape totul e acoperit și nu se poate spune.
    Ieftin (câteva ms): se rulează la fiecare cadru, ca un caroiaj dispărut să nu mai rămână desenat."""
    small, s = shrink(frame, GRID_W)
    mask = grid_mask(small)
    # ±2 px: caroiajul ținut minte e aproape, dar nu la pixel (netezire, cadre noi). Un caroiaj real dă ~0,95,
    # unul mutat cu 12 px ~0,15, unul dispărut 0.
    near = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    c = np.float32(corners) * s
    cov = cube_cover(frame, corners, s, mask.shape)
    if cov is not None:
        H = cv2.getPerspectiveTransform(_SQUARE, c)
        p = np.rint(cv2.perspectiveTransform(_FIT_PTS, H)[0]).astype(np.int32)
        h, w = mask.shape
        ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        x, y = np.where(ok, p[:, 0], 0), np.where(ok, p[:, 1], 0)
        if np.count_nonzero((ok & (cov[y, x] == 0))[:len(_LINE_PTS)]) < VERIFY_MIN_SEEN:
            return None
    return fit_score(near, mask, c.tolist(), cov)


# ---------- caroiajul văzut doar parțial ----------
# Brațul vede doar o parte: caroiajul iese din imagine. Îl extrapolăm (colțurile pot fi în afara imaginii) din
# liniile lui (line_candidates) sau din celulele văzute, ca să știm încotro să mutăm camera. Dacă se văd toate cele
# 9 celule (partial_usable), e „aproximat”: culorile și cuburile se citesc și din el.
_E = np.concatenate([np.linspace(-0.6, -0.2, 5), np.linspace(3.2, 3.6, 5)])
_EXT_PTS = np.float32(np.vstack([np.column_stack([np.full_like(_E, k), _E]) for k in range(4)] +
                                [np.column_stack([_E, np.full_like(_E, k)]) for k in range(4)]))[None]
PARTIAL_OK = 0.75       # potrivirea minimă pe partea vizibilă
PARTIAL_MIN_SEEN = 0.2  # cât din linii trebuie să cadă în imagine, neacoperite de cuburi
PARTIAL_TRIES = 6       # câte ipoteze ajustăm (cele mai promițătoare)
PARTIAL_USE_MARGIN = 0.02   # caroiajul extrapolat e folosit și pentru culori dacă centrele celor 9 celule sunt în imagine


def partial_score(near, mask, corners, cov):
    """(scor, cât din linii se vede): ca fit_score, doar pe ce e în imagine și neacoperit, minus partea din
    prelungirile liniilor dincolo de margini care e neagră (după marginea caroiajului e hârtie albă; o ipoteză
    care pune marginea în mijlocul caroiajului adevărat pierde aici)."""
    h, w = mask.shape
    fit = fit_score(near, mask, corners, cov)
    p = grid_points(_FIT_PTS, corners)[:len(_LINE_PTS)]
    ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    seen = np.count_nonzero(ok & (cov[np.where(ok, p[:, 1], 0), np.where(ok, p[:, 0], 0)] == 0)) / len(p)
    e = grid_points(_EXT_PTS, corners)
    ok = (e[:, 0] >= 0) & (e[:, 0] < w) & (e[:, 1] >= 0) & (e[:, 1] < h)
    ok &= cov[np.where(ok, e[:, 1], 0), np.where(ok, e[:, 0], 0)] == 0
    ext = np.count_nonzero(near[e[ok, 1], e[ok, 0]]) / max(np.count_nonzero(ok), 1) if ok.sum() >= 5 else 0.0
    return fit - ext, seen


LINES_HINT_MIN = 0.004   # lines_hint: din pixelii imaginii, sub atât liniile negre văzute sunt zgomot


def lines_hint(frame):
    """Unde sunt liniile negre subțiri din cadru: (centrul lor x, y, ca fracțiuni din imagine) sau None.
    Ultima rezervă a centrării: nu se vede caroiajul, nici o parte recunoscută din el, nici cuburi, dar o bucată
    de linie la marginea imaginii spune încotro e."""
    small, _ = shrink(frame, PROC_W)
    m = grid_mask(small)
    if np.count_nonzero(m) < LINES_HINT_MIN * m.size:
        return None
    ys, xs = np.nonzero(m)
    return float(xs.mean()) / m.shape[1], float(ys.mean()) / m.shape[0]


def body_cover(cubes, s, shape):
    """Ce acoperă cuburile găsite fără caroiaj (doar fața de sus în imagine): fața și corpul de sub ea, la scara s."""
    cov = np.zeros(shape, np.uint8)
    for o in cubes:
        box = np.float32(o["box"]) * s
        (_, _), (bw, bh), _ = cv2.minAreaRect(box)
        body = np.float32(np.vstack([box, box + [0, max(bw, bh)]]))   # văzut de sus-oblic, corpul e sub față
        cv2.fillConvexPoly(cov, np.int32(cv2.convexHull(body)), 255)
    return cv2.dilate(cov, np.ones((9, 9), np.uint8))


def find_partial_grid(frame):
    """Caroiajul ieșit parțial din imagine: colțurile lui (extrapolate, și în afara imaginii) sau None."""
    small, s = shrink(frame, GRID_W)
    mask = grid_mask(small)
    near = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    cubes, _ = find_cubes(frame)
    cov = body_cover(cubes, s, mask.shape)
    cands = line_candidates(mask) + cell_candidates(mask, near)
    if len(cubes) >= LATTICE_MIN:
        g = grid_from_cubes(frame)      # cuburile, chiar și doar o parte din ele, dau rețeaua
        if g is not None:
            cands.insert(0, [[x * s, y * s] for x, y in g])
    # Ipotezele dintr-o singură celulă sunt aproximative: le ordonăm cu toleranță mare (±4 px), apoi le ajustăm
    # pe cele mai promițătoare și le judecăm strict.
    wide = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    ranked = []
    for c in cands:
        if squareish(c):
            sc, seen = partial_score(wide, mask, c, cov)
            if seen >= PARTIAL_MIN_SEEN:
                ranked.append((sc, c))
    best, best_score = None, PARTIAL_OK
    for _, c in sorted(ranked, key=lambda t: -t[0])[:PARTIAL_TRIES]:
        c = fit_grid(mask, c, steps=(4, 2, 1), cov=cov)
        sc, seen = partial_score(near, mask, c, cov)
        if seen >= PARTIAL_MIN_SEEN and sc > best_score and squareish(c):
            best, best_score = c, sc
    return None if best is None else [[x / s, y / s] for x, y in best]


LINE_MIN = 0.12    # line_candidates: un segment de linie mai scurt decât atât din lățime nu contează
LINE_SAME = 5      # px (la GRID_W): segmente pe aceeași dreaptă, la atât una de alta = aceeași linie
LINE_KEEP = 0.3    # o linie a caroiajului are cel puțin atât din lungimea celei mai lungi din familia ei (restul: muchii
                   # de cuburi, umbre)


def _lines(mask):
    """Dreptele lungi din masca liniilor negre, unite din segmente: listă de (punct, direcție unitară, lungime)."""
    w = mask.shape[1]
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, 40, minLineLength=LINE_MIN * w, maxLineGap=12)
    groups = []
    for x1, y1, x2, y2 in ([] if segs is None else segs[:, 0]):
        a, b = np.float64([x1, y1]), np.float64([x2, y2])
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        for g in groups:
            p, gd = g["p"], g["d"]
            n = np.array([-gd[1], gd[0]])
            if abs(d @ gd) > 0.99 and abs((a - p) @ n) < LINE_SAME and abs((b - p) @ n) < LINE_SAME:
                g["pts"] += [a, b]
                g["len"] += np.linalg.norm(b - a)
                break
        else:
            groups.append({"p": a, "d": d, "pts": [a, b], "len": np.linalg.norm(b - a)})
    out = []
    for g in groups:
        vx, vy, x0, y0 = cv2.fitLine(np.float32(g["pts"]), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        out.append((np.float64([x0, y0]), np.float64([vx, vy]), g["len"]))
    return out


def _cross(l1, l2):
    """Intersecția a două drepte (punct, direcție) sau None dacă sunt aproape paralele."""
    (p, d), (q, e) = l1[:2], l2[:2]
    m = np.array([d, -e]).T
    if abs(np.linalg.det(m)) < 1e-6:
        return None
    t = np.linalg.solve(m, q - p)
    return p + t[0] * d


def _fourth(ts):
    """Pozițiile (1D, de-a lungul unei drepte) a 3 linii egal depărtate pe masă, văzute în perspectivă → poziția celei
    de-a patra, după ultima: omografia 1D t ↦ (a·k + b)/(c·k + 1) prin k = 0, 1, 2, evaluată la k = 3."""
    A = np.array([[k, 1, -k * t] for k, t in enumerate(ts)], float)
    try:
        a, b, c = np.linalg.solve(A, np.float64(ts))
    except np.linalg.LinAlgError:
        return None
    den = 3 * c + 1
    return None if abs(den) < 1e-6 else (3 * a + b) / den


def line_candidates(mask):
    """Ipoteze de caroiaj (colțuri, la scara măștii) din liniile lui: două familii (cam orizontale / cam verticale),
    fiecare cu 4 linii văzute sau 3 (a patra calculată din perspectivă, de o parte sau de alta). Merge și când o
    margine a caroiajului e în afara imaginii, dacă se văd destule linii din el."""
    lines = _lines(mask)
    fams = [[l for l in lines if abs(l[1][0]) >= abs(l[1][1])], [l for l in lines if abs(l[1][0]) < abs(l[1][1])]]
    c = np.float64(mask.shape[::-1]) / 2
    sets = []
    for k, fam in enumerate(fams):
        # ordonate pe direcția perpendiculară (orizontalele de sus în jos, verticalele de la stânga la dreapta)
        key = (lambda l: _cross(l, (c, np.float64([0, 1])))) if k == 0 else (lambda l: _cross(l, (c, np.float64([1, 0]))))
        pos = [(key(l), l) for l in fam]
        pos = sorted([(p[1 - k], l) for p, l in pos if p is not None], key=lambda t: t[0])
        longest = max((l[2] for _, l in pos), default=0)
        kept = []
        for t, l in pos:
            if l[2] < LINE_KEEP * longest:
                continue
            if kept and t - kept[-1][0] < 2 * LINE_SAME:    # aceeași linie, găsită de două ori: rămâne cea mai lungă
                if l[2] > kept[-1][1][2]:
                    kept[-1] = (t, l)
                continue
            kept.append((t, l))
        pos = kept
        opts = []
        for i in range(len(pos) - 3):
            opts.append([l for _, l in pos[i:i + 4]])
        for i in range(len(pos) - 2):
            three = [l for _, l in pos[i:i + 3]]
            axis = (c, np.float64([0, 1]) if k == 0 else np.float64([1, 0]))
            ts = [_cross(l, axis)[1 - k] for l in three]
            p0, d0 = three[1][:2]
            for order in (ts, ts[::-1]):        # a patra linie după ultima sau înaintea primei
                t4 = _fourth(order)
                if t4 is None:
                    continue
                q = c.copy()
                q[1 - k] = t4
                new = (q, (three[0][1] + three[2][1]) / 2, 0.0)   # direcția: media vecinilor (aproximativ)
                opts.append(three + [new] if order is ts else [new] + three)
        sets.append(opts)
    cands = []
    for hs in sets[0]:
        for vs in sets[1]:
            pts = [_cross(hs[0], vs[0]), _cross(hs[0], vs[3]), _cross(hs[3], vs[3]), _cross(hs[3], vs[0])]
            if all(p is not None for p in pts):
                cands.append([[float(x), float(y)] for x, y in pts])
    return cands


def cells_visible(corners, size):
    """Ce celule (3x3, bool) se văd destul ca să citim cubul din ele: centrul celulei și o bandă deasupra lui (unde
    cade fața de sus a cubului văzut oblic) sunt în imagine, la PARTIAL_USE_MARGIN de margine."""
    w, h = size
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [3, 0], [3, 3], [0, 3]]), np.float32(corners))
    cells = [[c + 0.5, r + dr] for r in range(3) for c in range(3) for dr in (0.5, 0.1)]
    pts = cv2.perspectiveTransform(np.float32([[p] for p in cells]), M).reshape(-1, 2) / [w, h]
    m = PARTIAL_USE_MARGIN
    return ((pts >= m) & (pts <= 1 - m)).all(1).reshape(3, 3, 2).all(2)


def partial_usable(corners, size):
    """Caroiajul extrapolat (ieșit parțial din imagine) e destul ca să citim culorile și cuburile: se văd toate cele
    9 celule (cells_visible)."""
    return bool(cells_visible(corners, size).all())


def cube_cover(frame, corners, s, shape):
    """Masca (la scara s) a ce acoperă cuburile găsite pe acest caroiaj: tot corpul lor, nu doar fața colorată."""
    cubes, _ = find_cubes(frame, corners, fast=True)
    if not cubes:
        return None
    cov = np.zeros(shape, np.uint8)
    for cube in cubes:
        pts = np.vstack([cube["cube3d"]["top"], cube["cube3d"]["base"]]) if cube.get("cube3d") else cube["box"]
        cv2.fillConvexPoly(cov, np.int32(cv2.convexHull(np.float32(pts) * s)), 255)
    return cv2.dilate(cov, np.ones((9, 9), np.uint8))   # și umbra / muchiile din jurul cubului


_LAST = {}   # ultimele colțuri găsite pentru fiecare caroiaj: de acolo pornim când cuburile acoperă liniile


def forget():
    """Uită ce s-a învățat între cadre (focala, ultimele colțuri). Pentru teste / camera schimbată."""
    _LAST.clear()
    _FOCALS.clear()


def get_corners(grid, frame):
    """Colțurile găsite automat; dacă nu merge, cele calibrate pentru `grid`."""
    if C.AUTO_GRID:
        prev = _LAST.get(grid)
        if prev is None:
            try:
                prev = load_corners(grid)   # colțurile salvate: punct de plecare când cuburile acoperă liniile
            except RuntimeError:
                pass
        found = find_grid(frame, prev)
        if found is None and prev is not None:
            found = find_grid(frame)     # poate s-a mutat camera
        if found is not None:
            _LAST[grid] = found
            return found
        print(f"[vision] Caroiajul nu a fost găsit automat, folosesc vision.json ({grid})")
    return load_corners(grid)


def color_mask(hsv, name):
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in C.COLORS[name]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


def fractions(patch_bgr):
    """{culoare: fracțiunea de pixeli din petic care cad în intervalul ei}."""
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    return {name: cv2.countNonZero(color_mask(hsv, name)) / total for name in C.COLORS}


def classify(patch_bgr):
    fr = fractions(patch_bgr)
    best = max(fr, key=fr.get)
    return (best, fr[best]) if fr[best] >= C.MIN_FRACTION else (None, fr[best])


# ---------- cubul întreg, în 3D ----------
# Cuburile au colorată doar fața de sus, care stă la înălțimea cubului. Văzută oblic, ea cade deasupra
# celulei în care stă cubul. Din caroiaj aflăm poza camerei; din mărimea feței de sus, înălțimea cubului;
# coborând fața pe masă (de-a lungul razelor camerei) aflăm baza, adică celula adevărată.
# Coordonatele lumii: x, y în celule (caroiajul = 0..3), z în celule, în sus spre cameră.
_FOCALS = deque(maxlen=60)   # focalele (/ lățimea imaginii) învățate din caroiaje văzute curat
FOCAL_MAX_ERR = 0.01         # cât de „rotație adevărată” trebuie să iasă poza ca focala să fie învățată
FOCAL_MIN_GAIN = 0.01        # și cât de mult mai rău trebuie să iasă la focala de 1,5 ori mai mică / mai mare
                             # (văzut aproape de sus, caroiajul nu spune mare lucru despre focală)


def _pose_err(H, K):
    a = np.linalg.inv(K) @ H
    a1, a2 = a[:, 0], a[:, 1]
    n1, n2 = np.linalg.norm(a1), np.linalg.norm(a2)
    return (a1 @ a2 / (n1 * n2)) ** 2 + ((n1 - n2) / (n1 + n2)) ** 2


def _K(f, w, h):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


_FS = np.exp(np.linspace(np.log(0.4), np.log(5), 120))   # focalele încercate, × lățimea imaginii


def _pose_errs(H, fs, w, h):
    """_pose_err pentru toate focalele `fs` deodată (inversa lui K e cunoscută: fără 120 de inversări)."""
    H = np.asarray(H, float)
    r0 = (H[0] - w / 2 * H[2])[None, :] / fs[:, None]
    r1 = (H[1] - h / 2 * H[2])[None, :] / fs[:, None]
    r2 = np.broadcast_to(H[2], r0.shape)
    a1 = np.stack([r0[:, 0], r1[:, 0], r2[:, 0]], 1)
    a2 = np.stack([r0[:, 1], r1[:, 1], r2[:, 1]], 1)
    n1, n2 = np.linalg.norm(a1, axis=1), np.linalg.norm(a2, axis=1)
    return (np.sum(a1 * a2, 1) / (n1 * n2)) ** 2 + ((n1 - n2) / (n1 + n2)) ** 2


def learn_focal(H, w, h):
    """Focala pentru care caroiajul văzut e un pătrat adevărat în 3D. O păstrăm doar dacă potrivirea
    e foarte bună (caroiaj neacoperit, colțuri exacte); cu cuburi peste colțuri estimarea e prea zgomotoasă."""
    fs = w * _FS
    errs = _pose_errs(H, fs, w, h)
    i = int(np.argmin(errs))
    near = errs[np.abs(np.log(fs / fs[i])) < np.log(1.5)]
    if errs[i] < FOCAL_MAX_ERR and 0 < i < len(fs) - 1 and min(errs[0], errs[-1], near.max()) > errs[i] + FOCAL_MIN_GAIN:
        _FOCALS.append(fs[i] / w)


def focal(w):
    """Focala în pixeli: cea din setări, altfel mediana celor învățate, altfel o valoare tipică."""
    if C.CAM_FOCAL > 0:
        return C.CAM_FOCAL * w / C.CAM_W
    return (float(np.median(_FOCALS)) if _FOCALS else C.CAM_FOCAL_GUESS) * w


def camera_pose(corners, w, h, learn=True):
    """Poza camerei față de caroiaj: (P = matricea de proiecție 3x4, poziția camerei în lume).
    learn = colțurile sunt ale caroiajului găsit (nu un candidat încercat): din ele se poate învăța focala."""
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [3, 0], [3, 3], [0, 3]]),
                                    np.float32(corners)).astype(float)
    if C.CAM_FOCAL <= 0 and learn:
        learn_focal(H, w, h)
    return pose_from_homography(H, focal(w), w, h)


def pose_from_homography(H, f, w, h):
    """Ca camera_pose, din omografia plan → imagine și focala f (pixeli)."""
    K = _K(f, w, h)
    a = np.linalg.inv(K) @ H
    lam = 2 / (np.linalg.norm(a[:, 0]) + np.linalg.norm(a[:, 1]))
    if a[2, 2] < 0:   # caroiajul e în fața camerei
        lam = -lam
    r1, r2, t = lam * a[:, 0], lam * a[:, 1], lam * a[:, 2]
    u, _, vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = u @ vt        # cea mai apropiată rotație adevărată
    cam = -R.T @ t
    if cam[2] < 0:    # z = în sus, spre cameră
        R[:, 2] *= -1
        cam = -R.T @ t
    return K @ np.column_stack([R, t]), cam


def project(P, pts):
    p = np.column_stack([np.float64(pts), np.ones(len(pts))]) @ P.T
    return p[:, :2] / p[:, 2:]


def lift_cube(table_pts, cam):
    """table_pts = conturul feței colorate, dus pe planul mesei (în celule). Întoarce
    (centrul bazei, latura, câte cuburi are stiva, colțurile feței de sus și ale bazei, în 3D)."""
    rect = cv2.minAreaRect(np.float32(table_pts))
    s0 = np.sqrt(max(cv2.contourArea(cv2.convexHull(np.float32(table_pts))), 1e-6))
    nadir, cz = cam[:2], cam[2]
    # Pe masă, fața de sus (la înălțimea h) pare de cz / (cz - h) ori mai mare: s0 = CUBE_TOP × a × cz / (cz - h).
    # Un singur cub: h = a. Cu latura a cunoscută, h iese din s0, deci recunoaștem și un turn de n cuburi.
    single = s0 * cz / (C.CUBE_TOP * cz + s0)
    a = C.CUBE_SIDE
    levels = 1
    if a > 0:
        levels = max(1, int(round(cz * (1 - C.CUBE_TOP * a / s0) / a)))
    h = single if levels == 1 else levels * a
    side = single if levels == 1 else a
    k = (cz - h) / cz
    center = nadir + (np.float32(rect[0]) - nadir) * k
    box = nadir + (cv2.boxPoints(rect) - nadir) * k
    box = center + (box - center) / C.CUBE_TOP     # toată fața de sus, cu marginile necolorate
    top = np.column_stack([box, np.full(4, h)])
    base = np.column_stack([box, np.zeros(4)])
    return center, side, levels, top, base


# Pe ce parte e fața colorată (normala ei, în coordonatele caroiajului). „sus” = cubul stă normal.
FACES = {"sus": "sus", "fata": "spre rândul 3", "spate": "spre rândul 1",
         "stanga": "spre coloana 1", "dreapta": "spre coloana 3"}
SIDE_MARGIN = 0.03  # cât de mult mai pătrată trebuie să iasă fața ca latură (vezi square_err) ca să o credem


def face_quad(pts):
    """Cele 4 colțuri ale unei fețe (pată aproape patrulateră), în ordine pe contur."""
    hull = cv2.convexHull(np.float32(pts))
    perimeter = cv2.arcLength(hull, True)
    for eps in (0.02, 0.04, 0.06, 0.09, 0.12):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
        if len(approx) < 4:
            break
    return cv2.boxPoints(cv2.minAreaRect(np.float32(pts)))


def face_edges(quad, P, to_table):
    """(marginea de jos, marginea de sus) a unei fețe verticale, în imagine. Muchiile verticale (paralele cu
    direcția „în sus” a lumii, cum apare ea acolo în imagine) nu pot fi margini de jos sau de sus."""
    edges = [(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    gx, gy = cv2.perspectiveTransform(np.float32([[quad.mean(axis=0)]]), to_table)[0, 0]
    up = np.diff(project(P, [[gx, gy, 0], [gx, gy, 0.5]]), axis=0)[0]
    up /= max(np.linalg.norm(up), 1e-6)

    def vertical(e):
        d = e[1] - e[0]
        return abs(d @ up) > 0.85 * max(np.linalg.norm(d), 1e-6)

    flat = [e for e in edges if not vertical(e)] or edges
    low = lambda e: e[0][1] + e[1][1]
    return np.array(max(flat, key=low)), np.array(min(flat, key=low))


def side_cube(img_pts, P, cam, to_table):
    """Ipoteza „fața colorată e o latură”: e verticală, cu marginea de jos pe masă. Întoarce dict cu
    forma feței în planul ei (rect, area) și cubul reconstruit, sau None."""
    # Marginea de jos a feței, în imagine. Un cub rotit pe masă își arată fața ca paralelogram înclinat,
    # deci o luăm de pe conturul feței (aproximat cu 4 colțuri), nu din dreptunghiul care o încadrează.
    bottom, _ = face_edges(face_quad(img_pts), P, to_table)
    bottom = bottom[np.argsort(bottom[:, 0])]
    b1, b2 = cv2.perspectiveTransform(np.float32(bottom)[None], to_table)[0].astype(np.float64)
    length = np.linalg.norm(b2 - b1)
    if length < 1e-3:
        return None
    d = (b2 - b1) / length
    # Planul vertical prin marginea de jos: (u de-a lungul marginii, z în sus) -> imagine.
    Hv = P @ np.array([[d[0], 0, b1[0]], [d[1], 0, b1[1]], [0, 1, 0], [0, 0, 1.0]])
    try:
        uz = cv2.perspectiveTransform(img_pts[None], np.linalg.inv(Hv))[0]
    except np.linalg.LinAlgError:
        return None
    rect = cv2.minAreaRect(uz)
    area = cv2.contourArea(uz)
    face = np.sqrt(max(cv2.contourArea(cv2.convexHull(uz)), 1e-6))
    a = face / C.CUBE_TOP
    n = np.array([-d[1], d[0]])
    mid = (b1 + b2) / 2
    if n @ (cam[:2] - mid) < 0:          # normala feței văzute arată spre cameră
        n = -n
    center = b1 + d * rect[0][0] - n * a / 2   # cubul e în spatele feței
    sq = np.array([center + (-d - n) * a / 2, center + (d - n) * a / 2,
                   center + (d + n) * a / 2, center + (-d + n) * a / 2])
    if abs(n[1]) >= abs(n[0]):
        name = "fata" if n[1] > 0 else "spate"
    else:
        name = "dreapta" if n[0] > 0 else "stanga"
    return {"rect": rect, "area": area, "center": center, "side": a, "face": name,
            "top": np.column_stack([sq, np.full(4, a)]), "base": np.column_stack([sq, np.zeros(4)])}


def square_err(rect, area):
    """Cât de departe e o formă de un pătrat plin: raportul laturilor (log) + cât nu umple dreptunghiul.
    În planul greșit, fața (pătrată) iese alungită sau trapez."""
    (_, _), (bw, bh), _ = rect
    return abs(np.log(max(bw, 1e-6) / max(bh, 1e-6))) + 1 - area / max(bw * bh, 1e-6)


def hidden_face_cube(img_pts, P, cam, a, occluders, to_table):
    """O latură colorată tăiată jos de un cub din fața ei (se vede doar o fâșie). Marginea ei de sus e întreagă
    și stă la înălțime cunoscută (latura a), deci de acolo aflăm unde e cubul. Doar dacă fâșia se sprijină
    pe un cub găsit (altfel ar putea fi orice dungă colorată). Întoarce dict ca side_cube, sau None."""
    bottom, top = face_edges(face_quad(img_pts), P, to_table)
    mid = tuple(float(v) for v in (bottom[0] + bottom[1]) / 2)
    if not any(cv2.pointPolygonTest(hull, mid, True) > -OCCLUDER_GAP for hull in occluders):
        return None
    top = top[np.argsort(top[:, 0])]
    zt = a * (1 + C.CUBE_TOP) / 2                       # marginea de sus a părții colorate
    Hz = P[:, [0, 1, 3]].copy()
    Hz[:, 2] += zt * P[:, 2]
    t1, t2 = cv2.perspectiveTransform(np.float32(top)[None], np.linalg.inv(Hz))[0].astype(np.float64)
    length = np.linalg.norm(t2 - t1)
    if not 0.6 * C.CUBE_TOP * a < length < 1.5 * C.CUBE_TOP * a:   # marginea de sus are lățimea unei fețe?
        return None
    d = (t2 - t1) / length
    n = np.array([-d[1], d[0]])
    m = (t1 + t2) / 2
    if n @ (cam[:2] - m) < 0:
        n = -n
    center = m - n * a / 2
    sq = np.array([center + (-d - n) * a / 2, center + (d - n) * a / 2,
                   center + (d + n) * a / 2, center + (-d + n) * a / 2])
    name = ("fata" if n[1] > 0 else "spate") if abs(n[1]) >= abs(n[0]) else ("dreapta" if n[0] > 0 else "stanga")
    return {"center": center, "side": a, "face": name,
            "top": np.column_stack([sq, np.full(4, a)]), "base": np.column_stack([sq, np.zeros(4)])}


# Puncte de-a lungul celor 8 linii ale caroiajului (în celule), pentru „care linii se văd”.
_T_DENSE = np.linspace(0.0, 3.0, 181)
_DENSE_LINES = np.vstack([np.column_stack([np.full_like(_T_DENSE, k), _T_DENSE]) for k in range(4)] +
                         [np.column_stack([_T_DENSE, np.full_like(_T_DENSE, k)]) for k in range(4)])
LINES_W = 0.25         # cât cântărește partea de linii vizibile sub corpul cubului, față de eroarea de formă
LINES_MIN = 8         # sub atâtea puncte de linie sub corp, liniile nu spun nimic


def visible_lines(small, s, P):
    """Punctele liniilor caroiajului (în imagine, mărime originală) și care din ele se văd negre în cadru."""
    mask = cv2.dilate(grid_mask(small), np.ones((3, 3), np.uint8))
    pts = project(P, np.column_stack([_DENSE_LINES, np.zeros(len(_DENSE_LINES))]))
    xi, yi = np.rint(pts[:, 0] * s).astype(int), np.rint(pts[:, 1] * s).astype(int)
    h, w = mask.shape
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    seen = np.zeros(len(pts), bool)
    seen[ok] = mask[yi[ok], xi[ok]] > 0
    return pts[ok], seen[ok]


def in_hull(pts, poly, margin=0):
    """Care puncte cad în învelitoarea convexă a lui `poly`: margin > 0 = la cel puțin atâția px înăuntru,
    margin < 0 = și până la -margin px în afară. Desenăm poligonul într-o mască mică (toate punctele dintr-o
    dată), nu pointPolygonTest pe fiecare punct (de sute de ori mai lent)."""
    hull = cv2.convexHull(np.float32(poly)).reshape(-1, 2)
    pad = abs(margin) + 2
    x0, y0 = np.floor(hull.min(axis=0)) - pad
    w, h = (np.ceil(hull.max(axis=0)) + pad - [x0, y0] + 1).astype(int)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.int32(np.rint(hull - [x0, y0])), 255)
    if margin:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(margin) + 1,) * 2)
        mask = cv2.erode(mask, k) if margin > 0 else cv2.dilate(mask, k)
    xi, yi = np.rint(pts[:, 0] - x0).astype(int), np.rint(pts[:, 1] - y0).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros(len(pts), bool)
    out[ok] = mask[yi[ok], xi[ok]] > 0
    return out


def lines_under(body_top, body_base, blob, lines):
    """Ce parte din liniile caroiajului de sub corpul cubului (în afara feței colorate) se vede. Corpul
    le ascunde, deci la varianta corectă ies puține. None dacă sub corp nu e nicio linie."""
    pts, seen = lines
    inside = in_hull(pts, np.vstack([body_top, body_base]), 2) & ~in_hull(pts, blob, -3)
    n = int(inside.sum())
    return seen[inside].mean() if n >= LINES_MIN else None


OCCLUDER_GAP = 4      # px: cât de aproape de un cub găsit trebuie să fie marginea de jos a unei fețe tăiate


def split_touching(mask):
    """Contururile petelor dintr-o mască; două fețe de aceeași culoare care se ating în imagine (legate printr-o
    „gâtuire”) sunt despărțite după harta distanțelor până la margine."""
    out = []
    for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(cnt)
        blob = np.zeros((h + 2, w + 2), np.uint8)
        cv2.drawContours(blob, [cnt - [x - 1, y - 1]], -1, 255, -1)
        dist = cv2.distanceTransform(blob, cv2.DIST_L2, 3)
        n, peaks = cv2.connectedComponents(np.uint8(dist > SPLIT_PEAK * dist.max()))
        if n <= 2:
            out.append(cnt)
            continue
        # fiecare pixel merge la cel mai apropiat vârf
        _, nearest = cv2.distanceTransformWithLabels(np.uint8(peaks == 0), cv2.DIST_L2, 3,
                                                     labelType=cv2.DIST_LABEL_CCOMP)
        for k in np.unique(nearest[blob > 0]):
            part = np.uint8((nearest == k) & (blob > 0)) * 255
            for c in cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                out.append(c + [x - 1, y - 1])
    return out


def halves(img_pts):
    """O pată alungită tăiată în două, de-a curmezișul laturii lungi (două fețe lipite una de alta)."""
    (cx, cy), (w, h), ang = cv2.minAreaRect(img_pts)
    t = np.radians(ang)
    axis = np.array([np.cos(t), np.sin(t)]) if w >= h else np.array([-np.sin(t), np.cos(t)])
    along = (img_pts - [cx, cy]) @ axis
    return [img_pts[along < 0], img_pts[along >= 0]]


SPLIT_PEAK = 0.6    # vârfurile hărții de distanțe (× maximul) care contează ca centre de fețe separate
HALF_ASPECT = (1.6, 2.6)   # o pată respinsă cu raportul ăsta poate fi două fețe lipite: încercăm jumătățile


def find_cubes(frame, corners=None, fast=False):
    """Caută cuburile ca obiecte: pete de culoare de mărimea unui cub, aproape pătrate și pline.
    Petele mai mici decât un cub sunt zgomot și se ignoră. Întoarce (cuburi, respinse);
    fiecare e un dict cu color, center, box, angle, area (× celulă), aspect, fill, reason.
    Cu caroiajul găsit, forma se judecă pe planul mesei (văzută de sus), iar `center` e centrul bazei
    cubului; `cube3d` = colțurile feței de sus și ale bazei, în imagine, pentru desen; `face` = pe ce parte
    e fața colorată (sus sau o latură, vezi FACES): cubul poate fi întors, cu sus alb.
    fast = doar cât acoperă cuburile (pentru căutarea caroiajului): fără ipoteza „cub întors”."""
    small, s = shrink(frame)
    small = normalize_light(small)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, w = small.shape[:2]
    # Mărimea de referință: o celulă a caroiajului; fără caroiaj, ~1/50 din imagine.
    ref = cv2.contourArea(np.float32(corners)) * s * s / 9 if corners else h * w / 50
    P = cam = to_table = None
    if corners:
        to_table = cv2.getPerspectiveTransform(np.float32(corners), np.float32([[0, 0], [3, 0], [3, 3], [0, 3]]))
        if C.CUBE_3D:
            # În căutarea caroiajului (fast) colțurile sunt doar candidați: nu învățăm focala din ei.
            P, cam = camera_pose(corners, frame.shape[1], frame.shape[0], learn=not fast)
            if cam[2] <= 0.5:
                P = cam = None
    kernel = np.ones((3, 3), np.uint8)
    cubes, rejected, partial = [], [], []
    cache = {}   # liniile vizibile ale caroiajului, calculate doar dacă e nevoie

    def lines():
        if "lines" not in cache:
            cache["lines"] = visible_lines(small, s, P)
        return cache["lines"]

    def measure(name, img_pts):
        """O pată -> dict de cub (cu reason = de ce e respinsă, sau None)."""
        img_rect = cv2.minAreaRect(img_pts)
        cube3d, levels, face = None, 1, "sus"
        if corners:
            # Fața de sus e paralelă cu masa, deci pe planul mesei e tot un pătrat (doar mărit și mutat).
            pts = cv2.perspectiveTransform(img_pts[None], to_table)[0]
            rect = cv2.minAreaRect(pts)
            area = cv2.contourArea(pts)
            if cam is not None:
                base_c, side, levels, top, base = lift_cube(pts, cam)
                # Sau fața colorată e o latură (cub întors): o credem dacă așa iese mai pătrată.
                sc = None if fast else side_cube(img_pts, P, cam, to_table)
                use_side = False
                if sc:
                    err_top, err_side = square_err(rect, area), square_err(sc["rect"], sc["area"])
                    # Și corpul cubului ascunde liniile din spatele lui: varianta care pune corpul peste
                    # linii care se văd e cea greșită (cântărește mult când forma nu decide sigur).
                    f_top = lines_under(project(P, top), project(P, base), img_pts, lines())
                    f_side = lines_under(project(P, sc["top"]), project(P, sc["base"]), img_pts, lines())
                    if f_top is not None and f_side is not None:
                        err_top += LINES_W * f_top
                        err_side += LINES_W * f_side
                    use_side = err_side + SIDE_MARGIN < err_top
                if use_side:
                    rect, area, base_c, side, levels = sc["rect"], sc["area"], sc["center"], sc["side"], 1
                    top, base, face = sc["top"], sc["base"], sc["face"]
                center = project(P, [[base_c[0], base_c[1], 0]])[0].tolist()
                cube3d = {"top": project(P, top).tolist(), "base": project(P, base).tolist()}
                size = side * side
            else:
                center, size = list(img_rect[0]), area
        else:
            rect, area = img_rect, cv2.contourArea(img_pts)
            center, size = list(img_rect[0]), area * s * s / ref
        (_, _), (bw, bh), _ = rect
        aspect = max(bw, bh) / max(min(bw, bh), 1e-6)
        fill = area / max(bw * bh, 1e-6)
        reason = ("mic" if size < C.CUBE_MIN_AREA else "mare" if size > C.CUBE_MAX_AREA
                  else "alungit" if aspect > C.CUBE_MAX_ASPECT
                  else "formă" if fill < C.CUBE_MIN_FILL else None)
        return {"color": name, "center": center, "box": cv2.boxPoints(img_rect).tolist(), "cube3d": cube3d,
                "angle": round(float(rect[2]), 1), "area": round(float(size), 2), "aspect": round(aspect, 2),
                "fill": round(fill, 2), "reason": reason, "cell": None, "levels": levels, "face": face}

    for name in C.COLORS:
        m = cv2.morphologyEx(color_mask(hsv, name), cv2.MORPH_OPEN, kernel)
        for cnt in split_touching(m):
            if cv2.contourArea(cnt) < C.CUBE_MIN_AREA * ref * 0.3:   # zgomot evident, nici nu-l măsurăm
                continue
            img_pts = cnt.reshape(-1, 2).astype(np.float32) / s
            obj = measure(name, img_pts)
            if obj["reason"] in ("alungit", "formă", "mare"):
                (_, _), (iw, ih), _ = cv2.minAreaRect(img_pts)
                if HALF_ASPECT[0] < max(iw, ih) / max(min(iw, ih), 1e-6) < HALF_ASPECT[1]:
                    parts = [measure(name, cv2.convexHull(p).reshape(-1, 2).astype(np.float32))
                             for p in halves(img_pts) if len(p) >= 3]
                    if len(parts) == 2 and all(o["reason"] is None for o in parts):
                        cubes.extend(parts)
                        continue
            if obj["reason"] == "mic":
                continue
            (rejected if obj["reason"] else cubes).append(obj)
            if obj["reason"] in ("alungit", "formă") and obj["cube3d"] is not None:
                partial.append((obj, img_pts))
    if partial and cubes and not fast:
        # Fețe tăiate de cuburile din fața lor: le refacem din marginea de sus.
        a = C.CUBE_SIDE or float(np.median([np.sqrt(o["area"]) for o in cubes]))
        occluders = [cv2.convexHull(np.float32(o["cube3d"]["top"] + o["cube3d"]["base"]))
                     for o in cubes if o["cube3d"]]
        for obj, img_pts in partial:
            hc = hidden_face_cube(img_pts, P, cam, a, occluders, to_table)
            if hc is None:
                continue
            rejected.remove(obj)
            obj.update(center=project(P, [[hc["center"][0], hc["center"][1], 0]])[0].tolist(),
                       cube3d={"top": project(P, hc["top"]).tolist(), "base": project(P, hc["base"]).tolist()},
                       area=round(a * a, 2), face=hc["face"], reason=None, partial=True)
            cubes.append(obj)
    if len(cubes) > C.MAX_CUBES:
        cubes.sort(key=lambda o: o["area"], reverse=True)
        for obj in cubes[C.MAX_CUBES:]:
            obj["reason"] = "prea multe"
        rejected.extend(cubes[C.MAX_CUBES:])
        cubes = cubes[:C.MAX_CUBES]
    return cubes, rejected


# ---------- caroiajul plin, găsit din cuburi ----------
# Cu 9 cuburi pe caroiaj, liniile se văd prea puțin ca să-l găsim de la zero. Fețele de sus ale cuburilor sunt însă
# clare și stau pe o rețea 3x3, pe planul aflat la înălțimea cubului. Pornind de la fiecare față, încercăm toate
# cele 9 celule în care poate fi; ipoteza bună pune și celelalte fețe pe centrele celulelor. Cu rețeaua găsită și
# focala camerei aflăm poza camerei; coborând planul cu latura cubului ajungem la caroiajul de pe masă, pe care îl
# ajustăm apoi pe liniile care se mai văd printre cuburi.
_LATTICE = np.float32([[c + 0.5, r + 0.5] for r in range(3) for c in range(3)])
LATTICE_MIN = 4          # cel puțin atâtea fețe pe rețea
LATTICE_TOL = 0.35       # cât de departe (în celule) de centrul celulei poate fi o față (cuburi puse neglijent)
LATTICE_FACE = (0.5, 0.65, 0.8)   # latura feței colorate, în celule (nu o știm: încercăm câteva)


def face_lattice(cubes):
    """(omografia rețea → imagine pe planul fețelor de sus, indicii fețelor folosite) sau None."""
    centers = np.float32([np.mean(o["box"], axis=0) for o in cubes])
    best = None
    for o in cubes:
        quad = np.float32(order_corners(o["box"]))
        side = np.mean([np.linalg.norm(quad[i] - quad[(i + 1) % 4]) for i in range(4)])
        for q in LATTICE_FACE:
            square = np.float32([[-q / 2, -q / 2], [q / 2, -q / 2], [q / 2, q / 2], [-q / 2, q / 2]])
            for cell in _LATTICE:
                H = cv2.getPerspectiveTransform(square + cell, quad)
                pred = cv2.perspectiveTransform(_LATTICE[None], H)[0]
                d = np.linalg.norm(centers[:, None] - pred[None], axis=2)   # fețe × celule
                near, idx = d.min(1), d.argmin(1)
                tol = LATTICE_TOL * side / q
                inl = near < tol
                if len(set(idx[inl])) < inl.sum():      # două fețe în aceeași celulă: ipoteză greșită
                    continue
                score = inl.sum() - 0.1 * near[inl].sum() / tol
                if best is None or score > best[0]:
                    best = (score, idx[inl], np.flatnonzero(inl))
    if best is None or len(best[1]) < LATTICE_MIN:
        return None
    _, cells, used = best
    H, _ = cv2.findHomography(_LATTICE[cells], centers[used])
    return None if H is None else (H, used)


def grid_from_cubes(frame):
    """Caroiajul (aproape) plin, găsit din rețeaua fețelor de sus ale cuburilor, sau None."""
    cubes, _ = find_cubes(frame)
    if len(cubes) < LATTICE_MIN:
        return None
    found = face_lattice(cubes)
    if found is None:
        return None
    Ht, used = found
    h, w = frame.shape[:2]
    top = cv2.perspectiveTransform(np.float32([[[0, 0], [3, 0], [3, 3], [0, 3]]]), Ht)[0]
    if (not np.all(np.isfinite(top)) or not squareish(top.tolist())
            or not cv2.isContourConvex(np.float32(top).reshape(-1, 1, 2))):
        return None
    Hs = cv2.getPerspectiveTransform(np.float32([[0, 0], [3, 0], [3, 3], [0, 3]]), top).astype(float)
    # Latura cubului (în celule), din mărimea fețelor duse pe planul lor.
    inv = np.linalg.inv(Ht)
    a = float(np.median([np.sqrt(cv2.contourArea(cv2.perspectiveTransform(np.float32([cubes[i]["box"]]), inv)[0]))
                         for i in used])) / C.CUBE_TOP
    # Focala: cea știută și cea pentru care rețeaua fețelor e un pătrat adevărat; păstrăm ce se potrivește mai bine.
    fs = w * _FS
    focals = {focal(w), float(fs[int(np.argmin(_pose_errs(Hs, fs, w, h)))])}
    small, s = shrink(frame, GRID_W)
    mask = grid_mask(small)
    near = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    best, best_score = None, FIT_COVERED
    for f in focals:
        try:
            P, _ = pose_from_homography(Hs, f, w, h)
        except np.linalg.LinAlgError:
            continue
        table = project(P, [[0, 0, -a], [3, 0, -a], [3, 3, -a], [0, 3, -a]])
        if not np.all(np.isfinite(table)):
            continue
        c = [[x * s, y * s] for x, y in table]
        cov = cube_cover(frame, table.tolist(), s, mask.shape)
        c = fit_grid(mask, c, steps=(4, 2, 1), cov=cov)
        corners = [[x / s, y / s] for x, y in c]
        if not squareish(corners):
            continue
        score = fit_score(near, mask, c, cube_cover(frame, corners, s, mask.shape))
        if score > best_score:
            best, best_score = corners, score
    return best


def cubes_inside(frame, corners):
    """Câte celule ale caroiajului au un cub: un caroiaj găsit de la zero fără niciun cub poate fi altceva
    (o grilă de pe jos, plăci de gresie)."""
    cubes, _ = find_cubes(frame, corners)
    return sum(c is not None for row in cubes_to_grid(cubes, corners) for c in row)


def cubes_to_grid(cubes, corners):
    """Pune fiecare cub în celula în care îi cade centrul; la două cuburi în aceeași celulă câștigă cel mare."""
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32([[0, 0], [3, 0], [3, 3], [0, 3]]))
    grid = [[None] * 3 for _ in range(3)]
    best = [[0.0] * 3 for _ in range(3)]
    for cube in cubes:
        gx, gy = cv2.perspectiveTransform(np.float32([[cube["center"]]]), M)[0, 0]
        if 0 <= gx < 3 and 0 <= gy < 3:
            r, c = int(gy), int(gx)
            cube["cell"] = f"{r + 1}{c + 1}"
            if cube["area"] > best[r][c]:
                grid[r][c], best[r][c] = cube["color"], cube["area"]
    return grid


def faces_grid(cubes):
    """Pe ce parte e fața colorată a cubului din fiecare celulă (după cubes_to_grid): "sus", "fata"... sau None."""
    grid = [[None] * 3 for _ in range(3)]
    best = [[0.0] * 3 for _ in range(3)]
    for cube in cubes:
        if cube.get("cell"):
            r, c = int(cube["cell"][0]) - 1, int(cube["cell"][1]) - 1
            if cube["area"] > best[r][c]:
                grid[r][c], best[r][c] = cube.get("face", "sus"), cube["area"]
    return grid


def detect_full(frame, corners):
    """Toată detecția pe un caroiaj: culori 3x3, fracțiuni de pixeli, cuburi găsite / respinse, vederea de sus,
    și pe ce parte e fața colorată a fiecărui cub (faces)."""
    top = warp(normalize_light(frame), corners)
    cubes, rejected = find_cubes(frame, corners)
    fracs = None
    if C.CUBE_MODE:
        colors = cubes_to_grid(cubes, corners)
        faces = faces_grid(cubes)
    else:   # metoda veche: procentul de pixeli din centrul fiecărei celule
        cell = WARP // 3
        m = int(cell * (1 - C.PATCH) / 2)
        fracs = [[fractions(top[r * cell + m:(r + 1) * cell - m, c * cell + m:(c + 1) * cell - m])
                  for c in range(3)] for r in range(3)]
        colors = [[max(f, key=f.get) if max(f.values()) >= C.MIN_FRACTION else None for f in row] for row in fracs]
        faces = [["sus" if x else None for x in row] for row in colors]
    return {"colors": colors, "faces": faces, "fracs": fracs, "cubes": cubes, "rejected": rejected, "top": top}


def detect_cells(frame, corners):
    """Întoarce (culori 3x3, fracțiuni 3x3 {culoare: frac}, imaginea îndreptată)."""
    d = detect_full(frame, corners)
    return d["colors"], d["fracs"], d["top"]


def annotate_top(top, colors, fracs, cubes=None, corners=None):
    cell = WARP // 3
    m = int(cell * (1 - C.PATCH) / 2)
    if cubes and corners:
        # Cuburile găsite, desenate în vederea de sus.
        M = cv2.getPerspectiveTransform(np.float32(corners),
                                        np.float32([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]]))
        for cube in cubes:
            box = cube["cube3d"]["base"] if cube.get("cube3d") else cube["box"]   # baza = unde stă cubul
            box = cv2.perspectiveTransform(np.float32([box]), M)[0]
            cv2.polylines(top, [np.int32(box)], True, (255, 255, 255), 2)
    for r in range(3):
        for c in range(3):
            y0, x0 = r * cell + m, c * cell + m
            label = colors[r][c] or "-"
            if fracs:
                cv2.rectangle(top, (x0, y0), (x0 + cell - 2 * m, y0 + cell - 2 * m), (255, 255, 255), 1)
                label += f" {max(fracs[r][c].values()):.2f}"
            cv2.putText(top, label, (c * cell + 3, r * cell + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1)
    return top


def detect_annotated(frame, corners):
    """Întoarce (matrice 3x3 cu culori sau None, imaginea îndreptată adnotată)."""
    d = detect_full(frame, corners)
    return d["colors"], annotate_top(d["top"], d["colors"], d["fracs"], d["cubes"], corners)


def vote(results):
    """Votul majoritar, celulă cu celulă, pe mai multe matrice 3x3."""
    return [[Counter(g[r][c] for g in results).most_common(1)[0][0] for c in range(3)] for r in range(3)]


def detect(grid, frame=None, debug=None):
    """Întoarce o matrice 3x3 cu numele culorii sau None. grid = 'S' sau 'D'."""
    if frame is None:
        frame = grab()
    result, top = detect_annotated(frame, get_corners(grid, frame))
    if debug:
        imwrite(debug, top)
    return result


def detect_stable(grid, n=None, debug=None):
    """Detecția pe `n` cadre noi (implicit STABLE_FRAMES), cu vot majoritar pe fiecare celulă."""
    n = max(1, n or C.STABLE_FRAMES)
    results = [detect(grid, debug=debug if i == n - 1 else None) for i in range(n)]
    return vote(results)


def print_grid(g):
    for r, row in enumerate(g):
        print(f"R{r + 1}: " + " | ".join(f"{(x or '-'):>8}" for x in row))
