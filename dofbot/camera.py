"""Camera citită continuu într-un fir separat, ca pagina web și sarcinile să o folosească în același timp.

Surse: "camera:0" (sau alt index), "synth" (scena desenată + lumea simulată), "image:poza.jpg",
"synth-arm" (ce vede camera de pe brațul simulat, din ultimele unghiuri trimise: căutare / centrare fără braț).
Dacă sursa nu merge, se trimit cadre cu mesajul de eroare și se reîncearcă singur.
"""
import threading
import time

import cv2
import numpy as np

from . import config as C

shared = None   # camera pornită de web.py; vision.grab() o folosește dacă există
RETRY_S = 3     # după cât timp se reîncearcă o cameră care nu merge


def default_source():
    if C.TEST_IMAGE == "synth":
        return "synth"
    if C.TEST_IMAGE:
        return "image:" + C.TEST_IMAGE
    return f"camera:{C.CAM_INDEX}"


def error_frame(msg):
    img = np.full((C.CAM_H, C.CAM_W, 3), 40, np.uint8)
    y = C.CAM_H // 2 - 20
    for line in ["FARA IMAGINE", msg[:60], msg[60:120]]:
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2, cv2.LINE_AA)
        y += 32
    return img


class Camera:
    def __init__(self, source=None):
        self.source = source or default_source()
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.error = None
        self.cap = None       # cv2.VideoCapture deschis (doar pentru o cameră adevărată)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, args=(self.source, self._stop), daemon=True,
                                        name="camera")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(3)

    def switch(self, source):
        self.stop()
        self.source, self.error = source, None
        self.start()

    def _open(self, source):
        """Întoarce o funcție care citește un cadru (sau None)."""
        if source == "synth":
            from . import synth
            from .sim_world import WORLD
            return lambda: synth.render(WORLD.snapshot(), seed=self.seq), None
        if source == "synth-arm":
            from . import sim_world
            from .arm_driver import LAST
            return lambda: sim_world.arm_view(LAST["angles"] or C.SIM_START, seed=self.seq), None
        if source.startswith("image:"):
            from . import vision
            img = vision.imread(source[6:])
            if img is None:
                raise RuntimeError(f"Nu pot citi imaginea {source[6:]}")
            return lambda: img.copy(), None
        if source.startswith("camera:"):
            from . import vision
            cap = vision.open_camera(int(source[7:]))
            return (lambda: cap.read()[1]), cap
        raise RuntimeError(f"Sursă necunoscută: {source}")

    def _publish(self, frame):
        with self.cond:
            self.frame, self.seq = frame, self.seq + 1
            self.cond.notify_all()

    def _loop(self, source, stop):
        while not stop.is_set():
            try:
                read, cap = self._open(source)
            except Exception as e:   # noqa: BLE001 - firul nu are voie să moară
                self.error = str(e)
                self._publish(error_frame(self.error))
                stop.wait(RETRY_S)
                continue
            self.error, fails = None, 0
            live = cap is not None
            self.cap = cap
            while not stop.is_set():
                t0 = time.time()
                f = read()
                if f is None:
                    fails += 1
                    if fails > 30:
                        self.error = "Camera nu mai întoarce imagini (deconectată?)"
                        self._publish(error_frame(self.error))
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                self._publish(f)
                if not live:    # poză / scenă simulată: ritmul fluxului
                    stop.wait(max(0.0, 1 / C.STREAM_FPS - (time.time() - t0)))
            self.cap = None
            if cap is not None:
                cap.release()
            stop.wait(RETRY_S if self.error else 0)

    def controls(self):
        """Reaplică setările CAM_* pe camera deschisă. Întoarce ce nu a mers, sau None dacă nu e o cameră."""
        cap = self.cap
        if cap is None:
            return None
        from . import vision
        return vision.camera_controls(cap)

    def latest(self):
        with self.cond:
            return (None if self.frame is None else self.frame.copy()), self.seq

    def fresh(self, after=None, timeout=3.0):
        """Așteaptă un cadru mai nou decât `after` (implicit: cel curent) și îl întoarce."""
        with self.cond:
            after = self.seq if after is None else after
            if not self.cond.wait_for(lambda: self.seq > after, timeout):
                raise RuntimeError(self.error or "Camera nu trimite cadre noi")
            return self.frame.copy(), self.seq
