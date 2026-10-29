"""
Mută brațul până când caroiajul ajunge în mijlocul imaginii, la o mărime bună.
Dacă nu-l vede deloc, rotește baza pe toată cursa (la înălțimea actuală) până îl găsește.
Logica e în dofbot/aim.py (aceeași ca butonul „Caută caroiajul” din pagină, care e mai comod:
se vede pe flux și se oprește din controller).

Are nevoie de pagina web pornită (camera e a ei). Nu porniți în paralel o sarcină sau tools/gamepad.py.

Rulare (din rădăcina proiectului):
  python3 tools/center_grid.py               centrează
  python3 tools/center_grid.py --save LOOK_S centrează și salvează poziția în positions.json
"""
import json
import os
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # rădăcina proiectului

from dofbot import config as C  # noqa: E402
from dofbot.aim import Aimer  # noqa: E402
from dofbot.robot import Robot  # noqa: E402

URL = f"http://localhost:{C.WEB_PORT}"


def keep_stream_open():
    """Pagina ține camera pornită doar cât se uită cineva: ne uităm și noi."""
    def run():
        while True:
            try:
                with urllib.request.urlopen(URL + "/stream/cam", timeout=10) as r:
                    while r.read(65536):
                        pass
            except OSError:
                time.sleep(1)
    threading.Thread(target=run, daemon=True).start()


def get_frame():
    try:
        data = urllib.request.urlopen(URL + "/frame.jpg", timeout=5).read()
    except OSError as e:
        raise RuntimeError(f"Pagina nu răspunde: {e}") from e
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Niciun cadru")
    return frame


def main():
    save = sys.argv[sys.argv.index("--save") + 1] if "--save" in sys.argv else None
    C.SIM_VERBOSE = False
    rob = Robot()
    keep_stream_open()
    time.sleep(1.5)
    final = Aimer(rob, get_frame).run()
    if final is None:
        return 1
    if save:
        rob.pos[save] = final
        os.makedirs(C.DATA_DIR, exist_ok=True)
        with open(C.POS_FILE, "w", encoding="utf-8") as f:
            json.dump(rob.pos, f, indent=2)
        print(f"Salvat {save} = {final} în {C.POS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
