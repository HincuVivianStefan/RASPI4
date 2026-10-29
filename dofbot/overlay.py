"""Desenele peste imaginea camerei (folosite de web.py)."""
import cv2
import numpy as np

HEX = {"rosu": "#d62828", "galben": "#f1c40f", "verde": "#2a9d3a", "albastru": "#1f5fd1", None: "#d0d0d0"}
BGR = {"rosu": (40, 40, 214), "galben": (15, 196, 241), "verde": (58, 157, 42), "albastru": (209, 95, 31)}
OTHER = (200, 0, 200)  # culori adăugate de voi în COLORS


def draw_cubes(view, cubes, rejected=()):
    """Cuburile găsite: chenar gros în culoarea lor. Petele respinse: chenar gri subțire + motivul."""
    for obj in rejected:
        box = np.int32(obj["box"])
        cv2.polylines(view, [box], True, (150, 150, 150), 1, cv2.LINE_AA)
        x, y = box.min(axis=0)
        cv2.putText(view, obj["reason"], (int(x), int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (200, 200, 200), 1, cv2.LINE_AA)
    for obj in cubes:
        col = BGR.get(obj["color"], OTHER)
        if obj.get("cube3d"):
            # Tot cubul: fața de sus, baza (unde stă pe caroiaj) și muchiile verticale.
            top, base = np.int32(obj["cube3d"]["top"]), np.int32(obj["cube3d"]["base"])
            for width, c in ((4, (255, 255, 255)), (2, col)):
                cv2.polylines(view, [top, base], True, c, width, cv2.LINE_AA)
                for p, q in zip(top, base):
                    cv2.line(view, tuple(map(int, p)), tuple(map(int, q)), c, width, cv2.LINE_AA)
            cx, cy = map(int, obj["center"])
            cv2.drawMarker(view, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 3)
            cv2.drawMarker(view, (cx, cy), col, cv2.MARKER_CROSS, 12, 1)
            box = np.vstack([top, base])
        else:
            box = np.int32(obj["box"])
            cv2.polylines(view, [box], True, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.polylines(view, [box], True, col, 2, cv2.LINE_AA)
        x, y = box.min(axis=0)
        label = f"cub {obj['color']}" + (f" x{obj['levels']}" if obj.get("levels", 1) > 1 else "")
        if obj.get("face", "sus") != "sus":
            label += f" (culoare pe {obj['face']})"   # cub întors: sus e alb
        cv2.putText(view, label, (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(view, label, (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    col, 1, cv2.LINE_AA)


def draw_grid(view, corners, result, scale=1.0):
    """Proiectează liniile caroiajului 3x3 înapoi pe imaginea camerei."""
    src = np.float32(corners) * scale
    dst = np.float32([[0, 0], [3, 0], [3, 3], [0, 3]])
    inv = cv2.getPerspectiveTransform(dst, src)

    def P(pts):
        return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), inv).reshape(-1, 2)

    for i in range(4):
        for a, b in (((i, 0), (i, 3)), ((0, i), (3, i))):
            p, q = P([a, b])
            cv2.line(view, tuple(map(int, p)), tuple(map(int, q)), (255, 255, 255), 1, cv2.LINE_AA)
    for r in range(3):
        for c in range(3):
            x, y = map(int, P([(c + 0.5, r + 0.5)])[0])
            color = result[r][c] if result else None
            if color:
                cv2.circle(view, (x, y), 9, BGR.get(color, OTHER), -1)
                cv2.circle(view, (x, y), 9, (255, 255, 255), 2)
            cv2.putText(view, f"{r + 1}{c + 1}", (x - 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 255, 255), 1, cv2.LINE_AA)


def draw_partial(view, corners):
    """Caroiajul văzut doar parțial (extrapolat și în afara imaginii): linii punctate portocalii și o săgeată
    din mijlocul imaginii spre centrul lui, adică încotro trebuie să se uite camera."""
    inv = cv2.getPerspectiveTransform(np.float32([[0, 0], [3, 0], [3, 3], [0, 3]]), np.float32(corners))
    orange = (0, 140, 255)
    t = np.linspace(0, 3, 61)
    for i in range(4):
        for line in (np.column_stack([np.full_like(t, i), t]), np.column_stack([t, np.full_like(t, i)])):
            pts = np.clip(cv2.perspectiveTransform(np.float32(line)[None], inv)[0], -30000, 30000)
            for p, q in zip(pts[0::2], pts[1::2]):      # punctat: o bucată da, una nu
                cv2.line(view, tuple(map(int, p)), tuple(map(int, q)), orange, 2, cv2.LINE_AA)
    h, w = view.shape[:2]
    c = np.clip(cv2.perspectiveTransform(np.float32([[[1.5, 1.5]]]), inv)[0, 0], [10, 10], [w - 10, h - 10])
    cv2.arrowedLine(view, (w // 2, h // 2), tuple(map(int, c)), orange, 3, cv2.LINE_AA, tipLength=0.08)
    cv2.putText(view, "caroiaj partial", (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, orange, 2, cv2.LINE_AA)


def draw_corners(view, corners, scale=1.0):
    pts = [(int(x * scale), int(y * scale)) for x, y in corners]
    for i, p in enumerate(pts):
        cv2.circle(view, p, 6, (0, 0, 255), -1)
        cv2.putText(view, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if len(pts) > 1:
        cv2.polylines(view, [np.int32(pts)], len(pts) == 4, (0, 0, 255), 1)


def dim_outside(view, mask):
    """Păstrează doar pixelii din mască; restul se întunecă."""
    return np.where(mask[..., None] > 0, view, (view * 0.25).astype(np.uint8))
