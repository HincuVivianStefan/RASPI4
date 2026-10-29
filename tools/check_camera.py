"""Verifică ce camere vede OpenCV pe acest calculator și prin ce backend dau imagine.

Rulare (din folderul proiectului, cu pagina web OPRITĂ, altfel ea ține camera ocupată):
    py tools/check_camera.py        # Windows
    python3 tools/check_camera.py   # Pi / Mac
Trimite tot ce afișează dacă nu merge camera.
"""
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from dofbot import config as C  # noqa: E402
from dofbot.vision import camera_backends, is_infrared  # noqa: E402


def probe(index, name, backend):
    t0 = time.time()
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return "nu se deschide"
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAM_H)
    frame = None
    for _ in range(15):
        ok, f = cap.read()
        if ok:
            frame = f
            break
    cap.release()
    if frame is None:
        return "se deschide, dar NU trimite cadre"
    h, w = frame.shape[:2]
    bright = frame.mean()
    note = ("  <- imagine NEAGRĂ: capac / buton de confidențialitate?" if bright < 8
            else "  <- alb-negru: cameră infraroșu (Windows Hello), alege alta" if is_infrared(frame) else "")
    return f"OK {w}x{h}, luminozitate {bright:.0f}, {time.time() - t0:.1f} s{note}"


def main():
    print(f"Python {sys.version.split()[0]}, OpenCV {cv2.__version__}, {platform.platform()}")
    found = False   # o cameră care dă imagine color
    for index in range(4):
        for name, backend in camera_backends():
            result = probe(index, name, backend)
            print(f"camera {index} / {name:5}: {result}")
            found |= result.startswith("OK") and "<-" not in result
    if found:
        print("\nAlege în pagină camera cu OK (Sursa imaginii: Camera N).")
    else:
        print("\nNicio cameră nu dă imagine. Verifică:\n"
              " - Windows: Setări > Confidențialitate și securitate > Cameră: pornit, inclusiv\n"
              "   „Permite aplicațiilor desktop să acceseze camera”;\n"
              " - camera nu e deschisă în altă aplicație (Teams, Zoom, Camera, browser);\n"
              " - capacul / tasta de confidențialitate a camerei (unele laptopuri au tastă F8/F10);\n"
              " - camera apare în Device Manager > Cameras, fără semn de exclamare.")


if __name__ == "__main__":
    main()
