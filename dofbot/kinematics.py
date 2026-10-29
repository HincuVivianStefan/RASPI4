"""Cinematica directă a brațului DOFBOT: din unghiurile servourilor, unde e încheietura și camera de pe gripper.

Lungimile sunt din URDF-ul Yahboom (dofbot_info/urdf/dofbot.urdf). Convențiile de semn sunt cele ale pozițiilor
noastre: servo 1 = 90 înainte, servourile 2–4 = 90 drept în sus; umărul sub 90 se apleacă în față, cotul și
încheietura sub 90 se îndoaie spre masă. Folosită de lumea simulată (camera pe brațul simulat); pe brațul real,
camera și semnele pot diferi puțin, de aceea centrarea (aim.py) își măsoară singură efectul fiecărei articulații.

Sistemul brațului: x înainte (baza la 90°), y la stânga, z în sus, în metri, cu originea pe masă sub bază.
"""
import numpy as np

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


def camera_pose(angles):
    """(poziția camerei, R) cu R = rândurile [dreapta, jos, înainte] ale camerei, în sistemul brațului
    (deci un punct X se vede în coordonatele camerei ca R @ (X - poziția)). Servo 5 rotește imaginea."""
    w, d, up = wrist(angles)
    roll = np.radians((angles[4] if len(angles) > 4 else 90) - 90)
    right = np.cross(d, up)
    right, up = np.cos(roll) * right + np.sin(roll) * up, np.cos(roll) * up - np.sin(roll) * right
    pos = w + CAM_ALONG * d + CAM_UP * up
    return pos, np.vstack([right, -up, d])
