"""Cinematica directă a brațului DOFBOT: din unghiurile servourilor, unde e încheietura și camera de pe gripper.

Lungimile sunt din URDF-ul Yahboom (dofbot_info/urdf/dofbot.urdf). Convențiile de semn sunt cele ale pozițiilor
noastre: servo 1 = 90 înainte, servourile 2–4 = 90 drept în sus; umărul sub 90 se apleacă în față, cotul și
încheietura sub 90 se îndoaie spre masă. Folosită de lumea simulată (camera pe brațul simulat); pe brațul real,
camera și semnele pot diferi puțin, de aceea centrarea (aim.py) își măsoară singură efectul fiecărei articulații.

Sistemul brațului: x înainte (baza la 90°), y la stânga, z în sus, în metri, cu originea pe masă sub bază.
Cinematica inversă (ik / reach) duce punctul dintre degetele gripperului la un punct de pe masă (fetch.py).
"""
import numpy as np

from . import config as C

BASE_H = 0.06605 + 0.04145   # masa → axa umărului
L1 = 0.08285                 # umăr → cot
L2 = 0.08285                 # cot → încheietură
L3 = 0.07385                 # încheietură → axa rotației (servo 5)
CAM_ALONG = 0.05             # camera: atâția metri după încheietură, de-a lungul gripperului
CAM_UP = 0.03                # și atât deasupra lui (spre partea de sus a imaginii)


def pitches(angles):
    """Unghiurile umărului, cotului și încheieturii față de verticală (radiani, pozitiv = înainte)."""
    s2, s3, s4 = angles[1:4]
    t2 = np.radians(90 - s2)
    t3 = t2 + np.radians(90 - s3)
    return t2, t3, t3 + np.radians(90 - s4)


def wrist(angles):
    """(încheietura, direcția gripperului, „sus” în imaginea camerei), vectori 3D în sistemul brațului."""
    t2, t3, t4 = pitches(angles)
    b = np.radians(angles[0] - 90)
    axis = np.array([np.cos(b), np.sin(b), 0.0])          # direcția „înainte” a planului brațului
    r = L1 * np.sin(t2) + L2 * np.sin(t3)
    z = BASE_H + L1 * np.cos(t2) + L2 * np.cos(t3)
    d = np.sin(t4) * axis + np.array([0, 0, np.cos(t4)])
    up = -np.cos(t4) * axis + np.array([0, 0, np.sin(t4)])  # spre gripperul ridicat: partea îndepărtată a mesei
    return r * axis + np.array([0, 0, z]), d, up


def _rolled(angles, roll):
    """(încheietura, direcția gripperului, dreapta, sus) cu imaginea rotită cu `roll` grade."""
    w, d, up = wrist(angles)
    r = np.radians(roll)
    right = np.cross(d, up)
    right, up = np.cos(r) * right + np.sin(r) * up, np.cos(r) * up - np.sin(r) * right
    return w, d, right, up


def camera_pose(angles):
    """(poziția camerei, R) cu R = rândurile [dreapta, jos, înainte] ale camerei, în sistemul brațului
    (deci un punct X se vede în coordonatele camerei ca R @ (X - poziția)). Cu CAM_ROLLS, servo 5 rotește imaginea
    (camera pe partea care se rotește); pe brațul nostru camera e înainte de rotație (măsurat: servo 5 nu o mișcă)."""
    roll = ((angles[4] if len(angles) > 4 else 90) - 90) if C.CAM_ROLLS else 0.0
    w, d, right, up = _rolled(angles, roll)
    pos = w + CAM_ALONG * d + CAM_UP * up
    return pos, np.vstack([right, -up, d])


def dz(r):
    """Cât ajunge brațul real mai sus decât modelul, la r m de bază (GRIP_DZ, GRIP_DZ_R; în afara GRIP_DZ_RS, ca la
    capătul lui: nu se extrapolează)."""
    return C.GRIP_DZ + C.GRIP_DZ_R * (min(max(r, C.GRIP_DZ_RS[0]), C.GRIP_DZ_RS[1]) - C.GRIP_DZ_R0)


def grip(angles):
    """(punctul dintre degetele gripperului, direcția în care se închid degetele; servo 5 le rotește)."""
    w, d, up = wrist(angles)
    _, _, right, _ = _rolled(angles, C.FETCH_ROLL_SIGN * ((angles[4] if len(angles) > 4 else 90) - 90))
    p = w + C.GRIP_LEN * d + C.GRIP_UP * up + C.GRIP_SIDE * side(angles)
    return p + [0, 0, dz(np.hypot(*p[:2]))], right


def fingers(angles):
    """(direcția în care se închid degetele, văzută pe masă, grade; cât de înclinată e față de orizontală, grade).
    Cu gripperul vertical, servo 5 rotește doar direcția; cu gripperul înclinat, o și apleacă: cubul prins / lăsat
    cu axa degetelor aplecată iese cu baza strâmbă."""
    _, right = grip(angles)
    return float(np.degrees(np.arctan2(right[1], right[0]))), float(np.degrees(np.arcsin(min(1.0, abs(right[2])))))


def direction(base, pitch):
    """Direcția gripperului pentru baza `base` și înclinarea `pitch` (grade; 180 = vertical în jos)."""
    b, t = np.radians(base - 90), np.radians(pitch)
    return np.array([np.sin(t) * np.cos(b), np.sin(t) * np.sin(b), np.cos(t)])


def side(angles):
    """Direcția orizontală „dreapta” a brațului (și a camerei: e perpendiculară pe planul brațului)."""
    b = np.radians(angles[0] - 90)
    return np.array([np.sin(b), -np.cos(b), 0.0])


def ik(point, pitch, roll=90.0):
    """Unghiurile servo1..5 care duc punctul gripperului în `point` (m), cu gripperul înclinat `pitch` grade față de
    vertical în sus (180 = drept în jos). None dacă nu ajunge sau iese din limite."""
    x, y, z = point
    base = 90 + np.degrees(np.arctan2(y, x))
    for _ in range(4):      # punctul de prindere e la GRIP_SIDE în afara planului brațului: planul trece pe lângă punct
        x, y = np.subtract(point[:2], C.GRIP_SIDE * side([base])[:2])
        base = 90 + np.degrees(np.arctan2(y, x))
    t4 = np.radians(pitch)
    r = np.hypot(x, y)
    # încheietura, față de umăr: punctul de prindere e la GRIP_LEN pe gripper și GRIP_UP în afara axei lui
    px = r - C.GRIP_LEN * np.sin(t4) + C.GRIP_UP * np.cos(t4)
    pz = z - dz(np.hypot(*point[:2])) - C.GRIP_LEN * np.cos(t4) - C.GRIP_UP * np.sin(t4) - BASE_H
    D = (px * px + pz * pz - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if abs(D) > 1:
        return None
    e = np.arccos(D)                    # cotul îndoit în față (spre masă)
    t2 = np.arctan2(px, pz) - np.arctan2(L2 * np.sin(e), L1 + L2 * np.cos(e))
    t3 = t2 + e
    angles = [base, 90 - np.degrees(t2), 90 - np.degrees(e), 90 - np.degrees(t4 - t3), roll]
    if not all(lo <= a <= hi for a, (lo, hi) in zip(angles, C.LIMITS)):
        return None
    return [float(a) for a in angles]


def reach(point, lift=0.0, roll=90.0, pitches=None):
    """(unghiurile de prindere, cele de deasupra, înclinarea): gripperul cât mai aproape de înclinarea de la
    calibrare (GRIP_PITCH, acolo valorile GRIP_* sunt bune), altfel cât mai vertical, de la 180° în jos până la
    FETCH_MIN_PITCH, la prima la care ajunge și cu `lift` m înapoi de-a lungul gripperului. Deasupra = cu `lift` m
    mai sus pe verticală (LIFT_VERTICAL: coboară drept, nu trage foaia; cu brațul întins nu încape: mai puțin, până
    la 40%), altfel cea de-a lungul gripperului. pitches = doar aceste înclinări, în ordine. None dacă nu ajunge."""
    point = np.float64(point)
    base = 90 + np.degrees(np.arctan2(point[1], point[0]))
    if pitches is None and C.GRIP_PITCH:
        want = C.GRIP_PITCH[0] + C.GRIP_PITCH[1] * np.hypot(*point[:2])
        pitches = sorted(range(180, C.FETCH_MIN_PITCH - 11, -1), key=lambda p: abs(p - want))
    elif pitches is None:
        pitches = range(180, C.FETCH_MIN_PITCH - 1, -5)
    for pitch in pitches:
        down = ik(point, pitch, roll)
        axis = ik(point - lift * direction(base, pitch), pitch, roll) if down else None
        if not axis:            # înclinarea se alege tot ca înainte (înălțimile sunt reglate pe ea)
            continue
        for k in (1, 0.8, 0.6, 0.4) if C.LIFT_VERTICAL else ():
            up = ik(point + [0, 0, k * lift], pitch, roll)
            if up:
                return down, up, pitch
        return down, axis, pitch
    return None


def line(a, b, step=0.01):
    """Unghiurile intermediare (fără capete) cu care punctul gripperului merge în linie dreaptă de la poziția `a` la
    `b` (câte unul la `step` m), cu înclinarea și servo 5 trecând treptat de la cele din a la cele din b. Mișcarea
    din unghi în unghi a servourilor duce gripperul pe o curbă (poate atinge foaia); pe acest drum, nu.
    None dacă vreun punct nu e accesibil."""
    pa, pb = grip(a)[0], grip(b)[0]
    ta, tb = np.degrees(pitches(a)[2]), np.degrees(pitches(b)[2])
    n = int(np.ceil(np.linalg.norm(pb - pa) / step))
    out = []
    for k in range(1, n):
        s = k / n
        q = ik(pa + (pb - pa) * s, ta + (tb - ta) * s, a[4] + (b[4] - a[4]) * s)
        if q is None:
            return None
        out.append(q)
    return out


def transit(a, b, step=0.02):
    """Ca line, dar la înălțimea celui mai sus dintre a și b: baza se rotește și brațul se întinde / strânge fără ca
    gripperul (cu cubul) să coboare pe drum peste cuburile puse; la capăt coboară vertical. None dacă nu ajunge."""
    pa, pb = grip(a)[0], grip(b)[0]
    ta, tb = np.degrees(pitches(a)[2]), np.degrees(pitches(b)[2])
    ra, rb = np.hypot(*pa[:2]), np.hypot(*pb[:2])
    fa, fb = np.arctan2(pa[1], pa[0]), np.arctan2(pb[1], pb[0])
    z = max(pa[2], pb[2])
    n = int(np.ceil(max(abs(rb - ra), max(ra, rb) * abs(fb - fa), abs(pb[2] - pa[2])) / step))
    out = []
    for k in range(1, n):
        s = k / n
        r, f = ra + (rb - ra) * s, fa + (fb - fa) * s
        q = ik([r * np.cos(f), r * np.sin(f), z], ta + (tb - ta) * s, a[4] + (b[4] - a[4]) * s)
        if q is None:
            return None
        out.append(q)
    return out


def look_at(point, dist):
    """Unghiurile din care camera privește punctul `point` (m) drept pe mijlocul imaginii, de la `dist` m, cât mai
    de sus (de la vertical până la FETCH_MIN_PITCH). None dacă nu ajunge."""
    point = np.float64(point)
    base = 90 + np.degrees(np.arctan2(point[1], point[0]))
    b = np.radians(base - 90)
    axis = np.array([np.cos(b), np.sin(b), 0.0])
    for pitch in range(180, C.FETCH_MIN_PITCH - 1, -5):
        t4 = np.radians(pitch)
        d = direction(base, pitch)
        up = -np.cos(t4) * axis + np.array([0, 0, np.sin(t4)])
        # camera = gripper + (CAM_ALONG - GRIP_LEN) d + CAM_UP up, iar camera trebuie să fie la point - dist d
        angles = ik(point - (dist + CAM_ALONG - C.GRIP_LEN) * d - CAM_UP * up, pitch)
        if angles:
            return angles
    return None
