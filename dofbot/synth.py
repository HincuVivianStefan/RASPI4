"""Desenează o poză falsă cu caroiajul 3x3 (negru pe foaie albă) și cuburi văzute de o cameră înclinată.

Cuburile sunt ca cele reale: albe, cu doar fața de sus colorată; un turn e o cutie mai înaltă.
Servește la teste și la rularea fără cameră (DOFBOT_IMAGE=synth).
"""
import cv2
import numpy as np

# Culori tipice ale fețelor cuburilor (BGR), alese în intervalele HSV din config.
CUBE_BGR = {"rosu": (45, 40, 200), "galben": (30, 195, 230), "verde": (60, 160, 40), "albastru": (200, 110, 30)}

PAPER = 600          # foaia, în pixeli, înainte de perspectivă
GRID = 450           # caroiajul
LINE = 6             # grosimea liniilor
CELL = GRID / 3      # pixeli de foaie pe celulă
SIDE = 0.7           # latura cubului, în celule
TOP = 0.9            # latura părții colorate / latura cubului

# Camera virtuală. Lumea: x, y în celule (caroiajul = 0..3, y spre cameră), z în sus.
EYE = np.array([1.6, 5.2, 8.0])
TARGET = np.array([1.5, 1.6, 0.0])
FOCAL = 560


def _camera(size=(640, 480)):
    fwd = (TARGET - EYE) / np.linalg.norm(TARGET - EYE)
    right = np.array([1.0, 0, 0]) - fwd[0] * fwd
    right /= np.linalg.norm(right)
    down = np.cross(right, fwd)
    R = np.vstack([right, down, fwd])
    K = np.array([[FOCAL, 0, size[0] / 2], [0, FOCAL, size[1] / 2], [0, 0, 1.0]])
    return K @ np.column_stack([R, -R @ EYE])


def _project(P, pts):
    p = np.column_stack([np.float64(pts), np.ones(len(pts))]) @ P.T
    return p[:, :2] / p[:, 2:]


def _paper_to_image(P):
    """Omografia foaie (pixeli) -> imagine."""
    o = (PAPER - GRID) / 2
    S = np.array([[1 / CELL, 0, -o / CELL], [0, 1 / CELL, -o / CELL], [0, 0, 1.0]])
    return P[:, [0, 1, 3]] @ S


def grid_corners(size=(640, 480)):
    """Colțurile caroiajului (mijlocul liniilor exterioare) în imaginea finală."""
    return _project(_camera(size), [[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0]]).tolist()


def _draw_cube(img, P, x, y, color, height, face="sus", yaw=0.0):
    """face = pe ce parte e fața colorată: "sus", "fata" (spre rândul 3), "spate", "stanga", "dreapta"
    (cea de pe o latură nevăzută nu apare). yaw = rotația cubului pe masă, în grade."""
    s, z = SIDE / 2, SIDE * height
    t = np.radians(yaw)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    base = np.array([x, y]) + np.array([[-s, -s], [s, -s], [s, s], [-s, s]]) @ rot.T
    lo = np.column_stack([base, np.zeros(4)])
    hi = np.column_stack([base, np.full(4, z)])
    ih = _project(P, hi)
    bgr = CUBE_BGR.get(color, (128, 128, 128))
    # Laturile întoarse spre cameră, întâi cea mai depărtată.
    sides = []
    for (a, b), name, shade in (((0, 1), "spate", 205), ((1, 2), "dreapta", 200),
                                ((2, 3), "fata", 218), ((3, 0), "stanga", 200)):
        mid = (base[a] + base[b]) / 2
        normal = mid - [x, y]
        if normal @ (EYE[:2] - mid) > 0:
            sides.append((-np.hypot(*(EYE[:2] - mid)), a, b, name, shade))
    for _, a, b, name, shade in sorted(sides):
        quad = np.array([lo[a], lo[b], hi[b], hi[a]])
        poly = np.int32(np.rint(_project(P, quad)))
        cv2.fillConvexPoly(img, poly, (shade,) * 3, cv2.LINE_AA)
        cv2.polylines(img, [poly], True, (175, 175, 175), 1, cv2.LINE_AA)
        if name == face and height == 1:
            mid = quad.mean(axis=0)
            inset = mid + (quad - mid) * TOP
            cv2.fillConvexPoly(img, np.int32(np.rint(_project(P, inset))), bgr, cv2.LINE_AA)
    cv2.fillConvexPoly(img, np.int32(np.rint(ih)), (232, 232, 232), cv2.LINE_AA)
    cv2.polylines(img, [np.int32(np.rint(ih))], True, (175, 175, 175), 1, cv2.LINE_AA)
    if face == "sus":
        inset = np.column_stack([np.array([x, y]) + (base - [x, y]) * TOP, np.full(4, z)])
        cv2.fillConvexPoly(img, np.int32(np.rint(_project(P, inset))), bgr, cv2.LINE_AA)


def render(cells, size=(640, 480), seed=0, noise=6.0, jitter=6, yaw=0.0):
    """cells[r][c] = culoare, None, (culoare, înălțime) sau (culoare, înălțime, fața colorată).
    jitter = cât de departe de centrul celulei e pus cubul (pixeli de foaie, max); yaw = cât de rotit (grade, max).
    Întoarce imaginea BGR."""
    rng = np.random.default_rng(seed)
    paper = np.full((PAPER, PAPER, 3), 235, np.uint8)
    o = (PAPER - GRID) // 2
    step = GRID // 3
    for i in range(4):
        p = o + i * step
        cv2.line(paper, (p, o), (p, o + GRID), (25, 25, 25), LINE)
        cv2.line(paper, (o, p), (o + GRID, p), (25, 25, 25), LINE)

    w, h = size
    P = _camera(size)
    img = np.full((h, w, 3), (95, 105, 110), np.uint8)   # masa
    cv2.warpPerspective(paper, _paper_to_image(P), (w, h), dst=img, borderMode=cv2.BORDER_TRANSPARENT)

    cubes = []
    for r in range(3):
        for c in range(3):
            v = cells[r][c]
            color, height, face = (tuple(v) + ("sus",))[:3] if isinstance(v, (tuple, list)) else (v, 1, "sus")
            if not color or height < 1:
                continue
            x = c + 0.5 + rng.integers(-jitter, jitter + 1) / CELL
            y = r + 0.5 + rng.integers(-jitter, jitter + 1) / CELL
            cubes.append((x, y, color, height, face, rng.uniform(-yaw, yaw) if yaw else 0.0))
    # Întâi cele mai depărtate de cameră, ca cele din față să le acopere.
    for x, y, color, height, face, turn in sorted(cubes, key=lambda q: -np.hypot(q[0] - EYE[0], q[1] - EYE[1])):
        _draw_cube(img, P, x, y, color, height, face, turn)

    # Lumină neuniformă + zgomot de senzor (zgomotul e pregătit o dată; generarea lui la fiecare cadru e lentă).
    xs = np.linspace(0.85, 1.08, w, dtype=np.float32)
    img = img.astype(np.float32) * xs[None, :, None]
    img += _noise(img.shape, noise, seed % 4)
    return np.clip(img, 0, 255).astype(np.uint8)


_NOISE = {}


def _noise(shape, sigma, k):
    key = (shape, sigma, k)
    if key not in _NOISE:
        _NOISE[key] = np.random.default_rng(k).normal(0, sigma, shape).astype(np.float32)
    return _NOISE[key]
