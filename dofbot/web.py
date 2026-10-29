"""
Pagina web de pe Raspberry Pi: tot ce vede camera, ce detectează și cum, plus rularea sarcinilor.

Rulare (din rădăcina proiectului):
         python3 -m dofbot.web            apoi în browser: http://<ip-ul-pi>:8000
         py -m dofbot.web --browser       (laptop) pornește și deschide pagina; sau dublu-click pe start.bat
Sursa imaginii (camera 0..3 sau scena simulată) se alege din pagină.

Doar biblioteca standard + OpenCV/numpy. Nu are parolă: rulați-l doar în rețeaua voastră.
"""
import json
import os
import socket
import sys
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from . import camera
from . import aim
from . import config as C
from . import hat
from . import pad
from . import overlay
from . import planner
from . import tasks
from . import vision
from .arm_driver import LAST
from .robot import REQUIRED, Robot, Stopped, load_positions
from .sim_world import WORLD

STATIC = os.path.join(C.HERE, "static")
SMOOTH = 0.5          # netezirea colțurilor găsite automat (0 = fără)
LOST_AFTER = 2        # după câte cadre la rând fără caroiaj spunem că e pierdut (nu mai e desenat)
LOST_S = 0.6          # sau după atâtea secunde fără confirmare (când procesarea e lentă)
PARTIAL_EVERY = 3     # caroiaj pierdut, fără sarcină: căutăm unul parțial (un colț) la fiecare al atâtelea cadru
IDLE_AFTER = 3.0      # fără pagină deschisă și fără sarcină, după atâtea secunde nu mai procesăm nimic
STREAMS = ("cam", "top", "mask")


def arm_is_sim():
    if C.SIM_ENV in ("0", "1"):
        return C.SIM_ENV == "1"
    try:
        import Arm_Lib  # noqa: F401
        return False
    except ImportError:
        return True


def load_vision():
    try:
        with open(C.VISION_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def mean_hsv(frame, x, y, r=3):
    h, w = frame.shape[:2]
    x, y = int(min(max(x, 0), w - 1)), int(min(max(y, 0), h - 1))
    patch = frame[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(axis=0)
    return [int(v) for v in hsv]


def combine(colors, faces):
    """Culoarea și orientarea votate împreună: un cub întors e alt rezultat decât unul normal."""
    return [[(c, f or "sus") if c else None for c, f in zip(cr, fr or [None] * 3)]
            for cr, fr in zip(colors, faces or [None] * 3)]


def split(grid):
    return ([[x[0] if x else None for x in row] for row in grid],
            [[x[1] if x else None for x in row] for row in grid])


# ---------------------------------------------------------------- procesarea imaginilor
class Pipeline:
    """Ia cadrele camerei, găsește caroiajul, detectează culorile și pregătește imaginile pentru pagină."""

    def __init__(self, cam):
        self.cam = cam
        self.lock = threading.Lock()
        self.cond = threading.Condition()
        self.grid = "S"
        self.view = {"mask": "toate", "rejected": False}
        self.wanted = time.time()   # ultima dată când cineva a cerut detecția (pagina sau o sarcină)
        self.count = 0
        self.last_small = None   # cadrul anterior, micșorat: din el vedem dacă s-a mișcat camera
        self.saved_ok = True     # colțurile salvate sunt bune doar până se mișcă camera (ea e pe braț)
        self.last_good = None    # ultimele colțuri găsite automat: de aici căutăm după ce camera stă iar pe loc
        self.paused = False   # detecția stă (nefolosit acum: căutarea brațului folosește chiar detecția paginii)
        self.done_t = 0.0     # când s-a terminat ultima procesare a unui cadru (căutarea brațului o așteaptă)
        self.urgent = False   # o sarcină așteaptă detecția: cuburile la fiecare cadru, nu doar la CUBE_EVERY
        self.viewers = {k: 0 for k in STREAMS}
        self.jpeg = {k: None for k in STREAMS}
        self.jseq = {k: 0 for k in STREAMS}
        self.frame = None
        self.result = {"corners": None, "colors": None, "cubes": [], "rejected": []}
        self.stream_fps = 0.0
        self.frozen = None    # cadrul înghețat (reglaje pe o imagine fixă) sau None
        self.saved = load_vision()
        self.reset()

    def reset(self):
        with self.lock:
            self.corners = None
            self.corner_src = None
            self.missed = LOST_AFTER
            self.last_ok = 0.0
            self.grid_status, self.grid_conf = "pierdut", None
            self.partial = None   # caroiajul văzut doar parțial (colțuri extrapolate), când cel întreg lipsește
            self.lines = None     # fără caroiaj, nici parțial: centrul liniilor negre văzute (x, y în 0..1)
            self.history = deque(maxlen=30)   # 3x3 cu (culoare, fața colorată) sau None
            self.stable_faces = None
            self.result = {"corners": None, "colors": None, "cubes": [], "rejected": []}
            self.state = {"colors": None, "fracs": None, "stable": None, "cubes": [], "rejected": []}

    def set_grid(self, grid):
        """Apelată și când brațul ajunge la LOOK_S / LOOK_D: acolo colțurile salvate sunt din nou bune."""
        with self.lock:
            self.saved_ok, self.last_small = True, None
            if grid != self.grid:
                self.grid = grid
                self.corners = self.last_good = self.partial = None
                self.history.clear()

    def save_corners(self, grid, corners):
        data = load_vision()
        if corners:
            data[grid] = [[int(x), int(y)] for x, y in corners]
        else:
            data.pop(grid, None)
        os.makedirs(C.DATA_DIR, exist_ok=True)
        with open(C.VISION_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.saved = data
        with self.lock:
            self.corners = self.corner_src = None
            self.saved_ok, self.last_small = True, None   # tocmai trase pe imaginea de acum

    def idle(self):
        return time.time() - self.wanted > IDLE_AFTER and not any(self.viewers.values())

    def next_frame(self, seq, fps, last):
        """Următorul cadru (sau cel înghețat), cel mult `fps` pe secundă. Întoarce (cadru, seq, momentul)."""
        wait = 1 / fps - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        if self.frozen is not None:
            return self.frozen, seq, time.time()
        frame, seq = self.cam.fresh(after=seq)
        return frame, seq, time.time()

    def run_stream(self):
        """Fluxul pentru pagină: fiecare cadru al camerei, cu ultima detecție desenată peste el."""
        seq, last, fps = 0, 0.0, 0.0
        q = [cv2.IMWRITE_JPEG_QUALITY, C.JPEG_QUALITY]
        while True:
            if self.idle():
                time.sleep(0.2)     # nimeni nu se uită: lăsăm procesorul liber
                continue
            try:
                frame, seq, now = self.next_frame(seq, C.STREAM_FPS, last)
            except RuntimeError:
                continue
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3) if last else 0.0
            last = now
            self.frame = frame
            if not self.viewers["cam"] and self.jpeg["cam"] is not None:
                continue
            try:
                view = frame.copy()
                r = self.result
                overlay.draw_cubes(view, r["cubes"], r["rejected"] if self.view["rejected"] else ())
                if r["corners"]:
                    overlay.draw_grid(view, r["corners"], r["colors"])
                    overlay.draw_corners(view, r["corners"])
                elif r.get("partial"):
                    overlay.draw_partial(view, r["partial"])
                self.publish("cam", cv2.imencode(".jpg", view, q)[1].tobytes())
                self.stream_fps = fps
            except Exception:   # noqa: BLE001 - fluxul nu are voie să se oprească
                traceback.print_exc()
                time.sleep(0.2)

    def publish(self, kind, data):
        with self.cond:
            self.jpeg[kind] = data
            self.jseq[kind] += 1
            self.cond.notify_all()

    def run_detect(self):
        """Detecția, în ritmul ei (DETECT_FPS); fluxul nu o așteaptă."""
        seq, last = 0, 0.0
        while True:
            if self.idle() or self.paused:
                time.sleep(0.2)
                continue
            try:
                frame, seq, last = self.next_frame(seq, C.DETECT_FPS, last)
                self.process(frame)
            except RuntimeError:
                continue
            except Exception:   # noqa: BLE001 - o setare greșită nu trebuie să oprească pagina
                traceback.print_exc()
                time.sleep(0.5)

    def moved(self, frame):
        """Cu cât s-a deplasat imaginea față de cadrul anterior: (dx, dy) în pixeli, sau None dacă stă pe loc.
        Diferența medie spune dacă s-a mișcat; corelația de fază, cu cât (pe imaginea micșorată la 160x120)."""
        small = np.float32(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA))
        last, self.last_small = self.last_small, small
        if last is None or cv2.absdiff(small, last).mean() <= C.MOTION_DIFF:
            return None
        (dx, dy), _ = cv2.phaseCorrelate(last, small)
        return dx * frame.shape[1] / 160, dy * frame.shape[0] / 120

    def track(self, frame):
        """Colțurile caroiajului. Automat: cele ținute minte sunt verificate pe fiecare cadru (ieftin) și căutate
        din nou doar când nu se mai potrivesc; după LOST_AFTER cadre / LOST_S fără confirmare caroiajul e pierdut
        și nu mai e desenat. vision.json e doar un punct de plecare pentru căutare, nu se afișează neverificat.
        Manual (AUTO_GRID oprit): colțurile salvate, cât timp camera nu s-a mișcat."""
        self.count += 1
        shift = self.moved(frame)
        moving = shift is not None
        if moving:
            self.saved_ok = False
            # Colțurile merg odată cu imaginea: de acolo ajustarea găsește caroiajul și când cuburile acoperă liniile.
            move = lambda c: [[x + shift[0], y + shift[1]] for x, y in c] if c else None  # noqa: E731
            self.corners, self.last_good = move(self.corners), move(self.last_good)
        saved = self.saved.get(self.grid) if self.saved_ok else None
        if not C.AUTO_GRID:
            self.corners, self.corner_src = (saved, "manual") if saved else (None, None)
            self.grid_status, self.grid_conf = ("manual" if saved else "pierdut"), None
            return
        now = time.time()
        if self.corners is not None and not moving:
            conf = vision.verify_grid(frame, self.corners)
            # None = liniile sunt aproape toate sub cuburi; cuburile stau însă tot pe el, deci e încă acolo.
            if conf is None or conf >= vision.VERIFY_OK:
                self.missed, self.last_ok, self.grid_conf = 0, now, conf
                self.grid_status = "acoperit" if conf is None else "urmărit"
                if self.count % C.GRID_EVERY == 0:      # din când în când, ajustăm colțurile pe imagine
                    self.accept(vision.find_grid(frame, self.corners))
                return
        # Pierdut sau camera în mișcare: căutăm pornind de la ce știm (colțurile vechi, apoi cele salvate),
        # apoi de la zero. Orice rezultat al lui find_grid e verificat pe imagine.
        prev = self.corners or self.last_good or saved
        found = vision.find_grid(frame, prev) if prev is not None else None
        if found is None:
            found = vision.find_grid(frame)
            if (found is not None and self.grid == "S" and C.GRID_NEED_CUBES
                    and vision.cubes_inside(frame, found) < C.GRID_NEED_CUBES):
                found = None    # găsit de la zero, dar fără cuburi: probabil o grilă de pe jos, nu caroiajul
        if found is not None:
            self.accept(found)
            self.missed, self.last_ok, self.grid_status, self.grid_conf = 0, now, "găsit", None
            self.partial = self.lines = None
            return
        self.missed += 1
        if self.missed >= LOST_AFTER or now - self.last_ok > LOST_S:
            self.corners, self.corner_src = None, None
            self.grid_status, self.grid_conf = "pierdut", None
            # Poate se vede doar o parte (un colț): îl extrapolăm, ca centrarea să știe încotro să meargă. E mai
            # scump, deci în afara centrării / sarcinilor doar la fiecare al PARTIAL_EVERY-lea cadru.
            if self.urgent or self.count % PARTIAL_EVERY == 0:
                self.partial = vision.find_partial_grid(frame)
            self.lines = vision.lines_hint(frame) if self.partial is None else None
            if self.partial is not None:
                self.grid_status = "parțial"

    def accept(self, found):
        """Colțuri noi găsite automat, netezite cu cele vechi dacă sunt aproape (fără tremur pe imagine)."""
        if found is None:
            return
        old = self.corners if self.corner_src == "auto" else None
        if old and all(abs(a - b) < 25 for p, q in zip(found, old) for a, b in zip(p, q)):
            found = [[SMOOTH * a + (1 - SMOOTH) * b for a, b in zip(p, q)] for p, q in zip(old, found)]
        self.corners, self.corner_src, self.last_good = found, "auto", found

    def process(self, frame):
        t0 = time.time()
        with self.lock:
            self.track(frame)
            corners, src, partial = self.corners, self.corner_src, self.partial
        if not (self.urgent or self.count % C.CUBE_EVERY == 0):
            # Doar caroiajul (urmărește camera); cuburile rămân cele de la ultima detecție.
            self.result = dict(self.result, corners=corners, partial=partial)
            self.state = dict(self.state, corners=corners, corner_src=src, grid=self.grid, partial=partial,
                              grid_found=corners is not None, grid_status=self.grid_status,
                              grid_conf=self.grid_conf, frozen=self.frozen is not None,
                              stream_fps=round(self.stream_fps, 1))
            self.done_t = time.time()
            return
        colors = fracs = faces = None
        top = np.zeros((vision.WARP, vision.WARP, 3), np.uint8)
        if corners:
            d = vision.detect_full(frame, corners)
            colors, faces, fracs, cubes, rejected = d["colors"], d["faces"], d["fracs"], d["cubes"], d["rejected"]
            if self.viewers["top"]:
                top = vision.annotate_top(d["top"], colors, fracs, cubes, corners)
        else:
            cubes, rejected = vision.find_cubes(frame)   # și fără caroiaj arătăm ce obiecte vedem
        self.result = {"corners": corners, "colors": colors, "cubes": cubes, "rejected": rejected, "partial": partial}
        with self.lock:
            if colors:
                self.history.append(combine(colors, faces))
            recent = list(self.history)[-C.STABLE_FRAMES:]
            stable, stable_faces = split(vision.vote(recent)) if recent else (None, None)

        q = [cv2.IMWRITE_JPEG_QUALITY, C.JPEG_QUALITY]
        if self.viewers["top"]:
            self.publish("top", cv2.imencode(".jpg", top, q)[1].tobytes())
        if self.viewers["mask"]:
            self.publish("mask", cv2.imencode(".jpg", self.mask_image(vision.shrink(frame)[0]), q)[1].tobytes())
        self.state = {
            "colors": colors, "faces": faces, "fracs": fracs, "stable": stable, "stable_faces": stable_faces,
            "corners": corners,
            "cubes": [{k: o.get(k) for k in ("color", "cell", "face", "angle", "area", "aspect")} for o in cubes],
            "rejected": [{k: o[k] for k in ("color", "reason", "area", "aspect", "fill")} for o in rejected],
            "corner_src": src, "grid": self.grid, "grid_found": corners is not None, "partial": partial,
            "grid_status": self.grid_status, "grid_conf": self.grid_conf,
            "frozen": self.frozen is not None, "stream_fps": round(self.stream_fps, 1),
            "ms": round((time.time() - t0) * 1000), "frame_size": [frame.shape[1], frame.shape[0]],
        }
        self.done_t = time.time()

    def mask_image(self, frame):
        frame = vision.normalize_light(frame)   # ce vede detecția: foaia adusă la alb
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m = self.view["mask"]
        if m == "grila":
            return overlay.dim_outside(frame, vision.grid_mask(frame))
        if m in C.COLORS:
            return overlay.dim_outside(frame, vision.color_mask(hsv, m))
        out = np.zeros_like(frame)      # "toate": fiecare culoare desenată cu culoarea ei
        for name in C.COLORS:
            out[vision.color_mask(hsv, name) > 0] = overlay.BGR.get(name, overlay.OTHER)
        return out

    def stable_now(self, timeout=12.0):
        """Votul pe ultimele STABLE_FRAMES cadre noi (așteaptă să se adune)."""
        with self.lock:
            self.history.clear()
        end = time.time() + timeout + C.STABLE_FRAMES / C.DETECT_FPS
        self.urgent = True
        try:
            while time.time() < end:
                self.wanted = time.time()
                with self.lock:
                    if len(self.history) >= C.STABLE_FRAMES:
                        colors, self.stable_faces = split(vision.vote(list(self.history)[-C.STABLE_FRAMES:]))
                        return colors
                time.sleep(0.05)
        finally:
            self.urgent = False
        raise RuntimeError("Caroiajul nu e găsit (nici automat, nici salvat)")


# ---------------------------------------------------------------- rularea sarcinilor
class Runner:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.thread = None
        self.stop_ev = threading.Event()
        self.cont_ev = threading.Event()
        self.on_memorize = lambda colors: None   # App: afișează aranjarea memorată de Pasul 1
        self.clear()

    def clear(self):
        self.status = "liber"
        self.task = None
        self.logs = deque(maxlen=400)
        self.result = None
        self.published = {}
        self.pause_msg = None
        self.notice = None

    # --- ctx pentru tasks.py
    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        print(line)
        self.logs.append(line)

    def detect(self, grid):
        self.pipeline.set_grid(grid)
        return self.pipeline.stable_now()

    def faces(self):
        """Pe ce parte e culoarea fiecărui cub, la ultima detecție a sarcinii (vezi vision.FACES)."""
        return self.pipeline.stable_faces

    def frame(self):
        """Un cadru nou de la cameră (pentru căutarea caroiajului); ține și detecția pornită."""
        self.pipeline.wanted = time.time()
        return self.pipeline.cam.fresh()[0]

    def notify(self, text, short=None):
        """Notificare: jurnal, mesaj în pagină și pe ecranul plăcii de răcire (short = 2 rânduri scurte)."""
        self.log(f">>> {text}")
        self.notice = {"id": time.time(), "text": text}
        hat.message(*(short or [text[:20]]))

    def search_pose(self, rob, grid):
        """De unde începe căutarea: SEARCH (dacă e învățată), altfel înălțimea din LOOK_<caroiaj> sau
        cea de acum, cu baza drept în față."""
        if "SEARCH" in rob.pos:
            return list(rob.pos["SEARCH"])
        pose = list(rob.pos.get(f"LOOK_{grid}") or rob.current())
        pose[0] = aim.SEARCH_CENTER
        return pose

    def step1(self, rob):
        """Pasul 1: privește drept în față, caută caroiajul cu cuburi, ține minte unghiurile de unde îl vede
        (LOOK_<caroiaj>) și memorează cuburile. Apoi Pasul 2."""
        grid = self.pipeline.grid
        start = self.search_pose(rob, grid)
        self.log(f"Mă uit drept în față: {[round(a) for a in start]}")
        rob.move(start, 1500)
        rob.sleep(0.5)
        angles = self.aim(rob, center=False)
        if angles is None:
            self.notify("Nu am găsit caroiajul cu cuburi.", ["Caroiaj negasit"])
            raise RuntimeError("Nu am găsit caroiajul")
        # Găsit, dar poate doar pe jumătate în imagine: ne așezăm să-l vedem tot, apoi memorăm.
        self.log("L-am găsit. Mă așez să văd tot caroiajul, apoi memorez cuburile.")
        hat.message("Caroiaj gasit", "Ma asez...")
        better = self.aim(rob, search=False, center=True)
        if better is not None:
            angles = better
        else:
            self.log("Nu am reușit să-l văd mai bine; memorez de unde sunt.")
        rob.sleep(0.5)
        rob.pos[f"LOOK_{grid}"] = angles
        os.makedirs(C.DATA_DIR, exist_ok=True)
        with open(C.POS_FILE, "w", encoding="utf-8") as f:
            json.dump(rob.pos, f, indent=2)
        self.log(f"Caroiaj {grid} găsit. Unghiurile salvate ca LOOK_{grid} = {angles}")
        colors = self.detect(grid)
        self.on_memorize(colors)
        self.publish("source", colors)
        n = sum(c is not None for row in colors for c in row)
        self.log(f"Cuburi memorate:\n{tasks.grid_text(colors)}")
        self.notify(f"Am găsit caroiajul cu cuburi și l-am memorat: {n} cuburi.", ["Caroiaj gasit!", "Memorat: %d cuburi" % n])
        return {"grid": grid, "angles": angles, "colors": colors}   # Pasul 2 (self.step2) vine după

    def step2(self, rob, grid, look, memorized):
        """Pasul 2: așteaptă să dispară toate cuburile de pe caroiaj, apoi se uită stânga / dreapta după ele."""
        left, empty_votes = None, 0
        while empty_votes < 2:      # de două ori la rând gol (fiecare = vot pe STABLE_FRAMES cadre)
            rob.check()
            try:
                colors = self.detect(grid)
            except RuntimeError:    # o mână acoperă caroiajul: încercăm din nou
                rob.sleep(0.5)
                continue
            n = sum(c is not None for row in colors for c in row)
            empty_votes = empty_votes + 1 if n == 0 else 0
            if n != left and n:
                self.log(f"Mai sunt {n} cuburi pe caroiaj; aștept să fie luate.")
            left = n
        self.notify("Caroiajul e gol. Caut cuburile stânga / dreapta.", ["Caroiaj gol", "Caut cuburile"])

        found = {}
        bases = [b for b in range(round(look[0]) - 75, round(look[0]) + 76, 15) if 0 <= b <= 180]
        for k, b in enumerate(bases):
            pose = list(look)
            pose[0] = b
            rob.move(pose, 1500 if k == 0 else 600)
            rob.sleep(aim.SETTLE_S)
            best = []
            for _ in range(2):      # două cadre; îl păstrăm pe cel cu mai multe cuburi
                try:
                    cubes, _ = vision.find_cubes(self.frame())
                except RuntimeError:
                    continue
                if len(cubes) > len(best):
                    best = cubes
            if best:
                found[b] = sorted(o["color"] for o in best)
                self.log(f"  baza {b}°: {', '.join(found[b])}")
        rob.move(look, 1500)
        self.publish("found", found)
        # Același cub se vede din mai multe unghiuri vecine: pentru fiecare culoare, cele mai multe dintr-o privire.
        counts = {}
        for colors in found.values():
            for c in set(colors):
                counts[c] = max(counts.get(c, 0), colors.count(c))
        total = sum(counts.values())
        want = [c for row in memorized for c in row if c]
        missing = {c: want.count(c) - counts.get(c, 0) for c in set(want) if want.count(c) > counts.get(c, 0)}
        self.publish("found_counts", counts)
        self.notify(f"Am găsit cam {total} cuburi ({', '.join(f'{c} {n}' for c, n in sorted(counts.items())) or 'niciunul'}); "
                    f"memorate {len(want)}" + (f"; lipsesc: {missing}" if missing else "."),
                    [f"Gasite: ~{total} cuburi", f"Memorate: {len(want)}"])
        return {"grid": grid, "angles": look, "colors": memorized, "found": found}

    def look(self, timeout=3.0):
        """Ce vede detecția paginii pe un cadru procesat după acest moment (aceeași urmărire ca în pagină)."""
        p, t0 = self.pipeline, time.time()
        while p.done_t <= t0 and time.time() - t0 < timeout:
            p.wanted = time.time()
            time.sleep(0.03)
        with p.lock:
            corners = p.corners if p.corner_src == "auto" and p.missed == 0 else None
            partial = p.partial if corners is None else None
            lines = p.lines if corners is None and partial is None else None
        cubes = p.result["cubes"]
        h, w = (p.frame.shape[:2] if p.frame is not None else (C.CAM_H, C.CAM_W))
        return {"corners": corners, "partial": partial, "lines": lines, "cubes": [o["box"] for o in cubes],
                "cells": sum(1 for o in cubes if o.get("cell")), "size": (w, h)}

    def aim(self, rob, search=True, center=True):
        p = self.pipeline
        p.urgent = True     # cuburile la fiecare cadru, nu doar la CUBE_EVERY
        try:
            need = C.GRID_NEED_CUBES if p.grid == "S" else 0     # D (destinația) poate fi gol
            return aim.Aimer(rob, self.frame, self.log, need_cubes=need, look=self.look).run(search, center)
        finally:
            p.urgent = False

    def publish(self, key, value):
        self.published[key] = value

    def pause(self, msg):
        self.log(f"PAUZĂ: {msg}")
        self.status, self.pause_msg = "pauză", msg
        self.cont_ev.clear()
        while not self.cont_ev.wait(0.1):
            if self.stop_ev.is_set():
                raise Stopped()
        self.status, self.pause_msg = "rulează", None

    # --- comenzi din pagină
    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, task):
        if task not in ("task1", "task2", "aim"):
            raise ValueError("Sarcină necunoscută")
        if self.busy():
            raise ValueError("O sarcină rulează deja")
        self.clear()
        self.stop_ev.clear()
        self.task, self.status = task, "rulează"
        self.thread = threading.Thread(target=self._run, args=(task,), daemon=True, name=task)
        self.thread.start()

    def _run(self, task):
        t0 = time.time()
        try:
            rob = Robot(should_stop=self.stop_ev.is_set, on_look=self.pipeline.set_grid)
            self.log(f"Pornesc {task} (braț {'simulat' if rob.sim else 'REAL'})")
            fn = {"task1": tasks.run_task1, "task2": tasks.run_task2, "aim": lambda r, ctx: ctx.step1(r)}[task]
            self.result = fn(rob, self)
            self.status = "gata"
        except Stopped:
            self.status = "oprit"
            self.log("Oprit.")
            try:
                rob.freeze()
            except Exception:   # noqa: BLE001
                traceback.print_exc()
        except Exception as e:   # noqa: BLE001
            self.status = "eroare"
            self.log(f"EROARE: {e}")
            traceback.print_exc()
        self.log(f"Timp total: {time.time() - t0:.1f} s")

    def stop(self):
        self.stop_ev.set()
        self.cont_ev.set()

    def wait_stopped(self, timeout=10):
        if self.thread is not None:
            self.thread.join(timeout)
        return not self.busy()


# ---------------------------------------------------------------- aplicația
class App:
    def __init__(self):
        self.cam = camera.Camera().start()
        camera.shared = self.cam
        self.pipeline = Pipeline(self.cam)
        self.runner = Runner(self.pipeline)
        self.runner.on_memorize = self.memorize_step1
        self.mode = "task1"
        self.memorized = None
        self.verify = None
        self.sim = arm_is_sim()
        threading.Thread(target=self.pipeline.run_stream, daemon=True, name="stream").start()
        threading.Thread(target=self.pipeline.run_detect, daemon=True, name="detect").start()
        threading.Thread(target=self.read_arm, daemon=True, name="read-arm").start()
        self.pad = pad.Pad(self.runner)
        if C.PAD:
            self.pad.start()
        else:
            self.pad.status = "dezactivat (config.PAD)"

    def memorize_step1(self, colors):
        self.memorized, self.verify = colors, None

    @staticmethod
    def read_arm():
        """O singură citire a servourilor la pornire, ca simularea din pagină să arate brațul de la început."""
        try:
            Robot().current()
        except Exception:   # noqa: BLE001
            traceback.print_exc()

    @staticmethod
    def say(msg):
        return {"ok": True, "message": msg}

    def state(self):
        self.pipeline.wanted = time.time()
        p = self.pipeline.state
        s = {
            "mode": self.mode, "sim": self.sim, "source": self.cam.source, "camera_error": self.cam.error,
            "vision": p, "view": self.pipeline.view, "settings": C.settings(),
            "colors_hex": {k: overlay.HEX.get(k, "#c060c0") for k in C.COLORS},
            "memorized": self.memorized, "verify": self.verify,
            "task": {"status": self.runner.status, "name": self.runner.task, "pause": self.runner.pause_msg,
                     "logs": list(self.runner.logs)[-150:], "result": self.runner.result,
                     "notice": self.runner.notice,
                     "published": self.runner.published},
            "positions_missing": self.positions_missing(),
            "saved_corners": sorted(self.pipeline.saved),
            "arm": {"angles": LAST["angles"], "pad": self.pad.state()},
        }
        if self.mode == "task2":
            s["plan2"] = self.preview_tower(p.get("stable"))
        return s

    @staticmethod
    def positions_missing():
        pos = load_positions()
        return [n for n in REQUIRED if n not in pos]

    @staticmethod
    def preview_tower(stable):
        if not stable:
            return {"error": "Caroiajul nu e detectat"}
        try:
            moves = planner.plan_tower(planner.grid_to_cells(stable), "S" + C.S2_TARGET, C.S2_ORDER)
            return {"moves": moves}
        except ValueError as e:
            return {"error": str(e)}

    # --- acțiuni (body = JSON din pagină)
    def a_mode(self, b):
        if b.get("mode") not in ("task1", "task2"):
            raise ValueError("mode = task1 sau task2")
        self.mode = b["mode"]
        self.pipeline.set_grid("S")
        return self.say(f"Mod: {'Sarcina 1' if self.mode == 'task1' else 'Sarcina 2'}")

    def a_grid(self, b):
        if b.get("grid") not in ("S", "D"):
            raise ValueError("grid = S sau D")
        self.pipeline.set_grid(b["grid"])
        if self.sim:
            WORLD.look(b["grid"])
        return self.say(f"Caroiaj {b['grid']}")

    def a_view(self, b):
        if "mask" in b:
            self.pipeline.view["mask"] = str(b["mask"])
        if "rejected" in b:
            self.pipeline.view["rejected"] = bool(b["rejected"])
        return {"ok": True}

    def a_settings(self, b):
        values = b.get("values", {})
        C.apply_settings(values)
        if any(k.startswith("CAM_") for k in values):
            failed = self.cam.controls()
            if failed:
                return self.say(f"Setări aplicate, dar camera nu acceptă: {', '.join(failed)}")
        if b.get("save"):
            C.save_settings()
            return self.say("Setări aplicate și salvate în settings.json")
        return self.say("Setări aplicate (nesalvate)")

    def a_settings_save(self, b):
        C.save_settings()
        return self.say("Setări salvate în settings.json")

    def a_settings_defaults(self, b):
        C.reset_settings()
        return self.say("Setări implicite (din config.py) aplicate; apasă Salvează ca să rămână")

    def a_corners(self, b):
        grid, corners = b.get("grid", self.pipeline.grid), b.get("corners")
        if corners is not None and len(corners) != 4:
            raise ValueError("Trebuie exact 4 colțuri")
        self.pipeline.save_corners(grid, vision.order_corners(corners) if corners else None)
        if not corners:
            return self.say(f"Colțuri {grid} șterse din vision.json")
        # Altfel detecția automată ar muta colțurile înapoi.
        C.apply_settings({"AUTO_GRID": False})
        return self.say(f"Colțuri {grid} salvate în vision.json; caroiaj automat oprit")

    def a_expose(self, b):
        """Expunere manuală reglată pe foaia albă (vezi vision.auto_expose), apoi lăsată fixă."""
        if self.cam.cap is None:
            raise ValueError("Merge doar cu o cameră adevărată")
        exp = vision.auto_expose(self.cam.cap, lambda: self.cam.fresh()[0], self.runner.log)
        if exp is None:
            raise ValueError("Nu am putut regla expunerea (vezi jurnalul)")
        return self.say(f"Expunere fixă: {exp} (apasă Salvează ca să rămână)")

    def a_freeze(self, b):
        p = self.pipeline
        p.frozen = None if p.frozen is not None or p.frame is None else p.frame.copy()
        return self.say("Imagine înghețată: reglajele se văd pe cadrul fix" if p.frozen is not None
                        else "Imagine live")

    def a_probe(self, b):
        frame = self.pipeline.frame
        if frame is None:
            raise ValueError("Niciun cadru încă")
        # Pe imaginea normalizată (LIGHT_NORM), ca valorile învățate să fie cele pe care le folosește detecția.
        hsv = mean_hsv(vision.normalize_light(frame), b["x"], b["y"])
        px = np.uint8([[hsv]])
        match = [n for n in C.COLORS if vision.color_mask(px, n)[0, 0]]
        return {"ok": True, "x": int(b["x"]), "y": int(b["y"]), "hsv": hsv, "match": match}

    def a_learn(self, b):
        """Lărgește intervalul HSV al unei culori ca să includă pixelul de la (x, y)."""
        name = b.get("color")
        if name not in C.COLORS:
            raise ValueError(f"Culoare necunoscută: {name}")
        h, s, v = self.a_probe(b)["hsv"]
        ranges = [[list(lo), list(hi)] for lo, hi in C.COLORS[name]]

        def hue_dist(rg):
            lo, hi = rg[0][0], rg[1][0]
            return 0 if lo <= h <= hi else min(abs(h - lo), abs(h - hi))

        lo, hi = min(ranges, key=hue_dist)
        mh, ms = 4, 40
        lo[:] = [max(0, min(lo[0], h - mh)), max(0, min(lo[1], s - ms)), max(0, min(lo[2], v - ms))]
        hi[:] = [min(179, max(hi[0], h + mh)), min(255, max(hi[1], s + ms)), min(255, max(hi[2], v + ms))]
        colors = C.settings()["COLORS"]
        colors[name] = ranges
        C.apply_settings({"COLORS": colors})
        return self.say(f"{name}: interval lărgit la H {lo[0]}-{hi[0]}, S {lo[1]}-{hi[1]}, V {lo[2]}-{hi[2]} "
                        "(nesalvat)")

    def a_memorize(self, b):
        self.memorized = self.pipeline.stable_now()
        self.verify = None
        return self.say("Aranjarea curentă a fost memorată")

    def a_verify(self, b):
        if not self.memorized:
            raise ValueError("Nu ai memorat încă o aranjare")
        now = self.pipeline.stable_now()
        self.verify = {"now": now, "errors": planner.compare(self.memorized, now)}
        n = len(self.verify["errors"])
        return self.say("Identic cu aranjarea memorată" if not n else f"{n} celule diferă")

    def a_run(self, b):
        self.runner.start(b.get("task", self.mode))
        return self.say(f"Pornit {b.get('task', self.mode)}")

    def a_aim(self, b):
        self.runner.start("aim")
        return self.say("Caut caroiajul cu cuburi în față (orice buton de pe controller oprește)")

    def a_stop(self, b):
        self.runner.stop()
        return self.say("Oprire cerută (se oprește la următoarea mișcare)")

    def a_continue(self, b):
        self.runner.cont_ev.set()
        return self.say("Continuă")

    def a_reset(self, b):
        self.runner.stop()
        if not self.runner.wait_stopped():
            raise ValueError("Sarcina nu s-a oprit încă; mai încearcă")
        self.runner.clear()
        WORLD.reset()
        self.pipeline.reset()
        self.pipeline.set_grid("S")
        self.memorized = self.verify = None
        if not self.sim:
            try:
                Robot().home()
            except Exception as e:   # noqa: BLE001
                return self.say(f"Resetat, dar brațul nu a mers la HOME: {e}")
        return self.say("Resetat: sarcină oprită, memorie ștearsă, scena simulată refăcută")

    def a_camera(self, b):
        src = str(b.get("source", ""))
        if not (src in ("synth", "synth-arm") or (src.startswith("camera:") and src[7:].isdigit())):
            raise ValueError("Sursă invalidă")
        self.cam.switch(src)
        self.pipeline.reset()
        vision.forget()     # altă cameră: altă focală
        return self.say(f"Sursa imaginii: {src}")

    def a_synth(self, b):
        C.apply_settings({"SYNTH_LAYOUT": b["layout"]})
        WORLD.reset()
        return self.say("Scena sintetică refăcută")

    def a_snapshot(self, b):
        frame = self.pipeline.frame
        if frame is None:
            raise ValueError("Niciun cadru încă")
        os.makedirs(C.CAPTURES_DIR, exist_ok=True)
        name = os.path.join(C.CAPTURES_DIR, time.strftime("captura_%Y%m%d_%H%M%S"))
        vision.imwrite(name + ".jpg", frame)
        for k in ("cam", "top", "mask"):
            if self.pipeline.jpeg[k]:
                with open(f"{name}_{k}.jpg", "wb") as f:
                    f.write(self.pipeline.jpeg[k])
        return self.say(f"Salvat {os.path.relpath(name, C.ROOT)}*.jpg")

    def a_record(self, b):
        """Înregistrează N cadre brute (PNG, fără pierderi) pentru reglajul detecției fără braț: în
        captures/set_<dată>/, cu meta.json (setări, colțuri) și labels.json = aranjarea detectată acum.
        Verifici labels.json de mână, pui "verificat": true, apoi tests/test_captures.py le folosește."""
        n = max(1, min(60, int(b.get("n", 10))))
        folder = os.path.join(C.CAPTURES_DIR, time.strftime("set_%Y%m%d_%H%M%S"))
        os.makedirs(folder)
        seq = None
        for i in range(n):
            frame, seq = self.cam.fresh(after=seq)
            vision.imwrite(os.path.join(folder, f"{i:02d}.png"), frame)
            time.sleep(0.25)
        st = self.pipeline.state
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"grid": self.pipeline.grid, "corners": st.get("corners"), "source": self.cam.source,
                       "settings": C.settings(), "light_gains": (lambda g: None if g is None else g.tolist())(
                           vision.light_gains(frame))}, f, indent=2, ensure_ascii=False)
        with open(os.path.join(folder, "labels.json"), "w", encoding="utf-8") as f:
            json.dump({"verificat": False, "colors": st.get("stable") or st.get("colors")}, f, indent=2,
                      ensure_ascii=False)
        return self.say(f"{n} cadre în {os.path.relpath(folder, C.ROOT)}: verifică labels.json și pune "
                        f"\"verificat\": true")


APP = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):   # fără un rând în terminal pentru fiecare cerere
        pass

    def send(self, code, body, ctype="application/json; charset=utf-8", cache=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(STATIC, "index.html"), "rb") as f:
                return self.send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/state":
            return self.send(200, APP.state())
        if path == "/frame.jpg":
            frame = APP.pipeline.frame
            if frame is None:
                return self.send(503, {"error": "Niciun cadru"})
            return self.send(200, cv2.imencode(".jpg", frame)[1].tobytes(), "image/jpeg")
        if path.startswith("/stream/") and path[8:] in STREAMS:
            return self.stream(path[8:])
        self.send(404, {"error": "Nu există"})

    def stream(self, kind):
        p = APP.pipeline
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        with p.cond:
            p.viewers[kind] += 1
        seq = -1
        try:
            self.wfile.write(b"--frame\r\n")
            while True:
                with p.cond:
                    p.cond.wait_for(lambda: p.jseq[kind] != seq, timeout=2)
                    seq, data = p.jseq[kind], p.jpeg[kind]
                if data is None:
                    continue
                # Granița următorului cadru se trimite imediat: Chrome afișează un cadru abia când o vede.
                self.wfile.write(b"Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n--frame\r\n"
                                 % (len(data), data))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            with p.cond:
                p.viewers[kind] -= 1

    def do_POST(self):
        name = self.path.split("?")[0].removeprefix("/api/").replace("/", "_")
        fn = getattr(APP, "a_" + name, None)
        if fn is None:
            return self.send(404, {"error": "Acțiune necunoscută"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            self.send(200, fn(body))
        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            self.send(400, {"ok": False, "error": str(e)})


def local_ip():
    """IP-ul laptopului în rețeaua locală (fără să trimită nimic: doar alege interfața spre o adresă)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
    except OSError:
        return None


def main():
    global APP
    cv2.setNumThreads(C.CV_THREADS)
    url = f"http://localhost:{C.WEB_PORT}"
    open_browser = "--browser" in sys.argv
    try:
        server = ThreadingHTTPServer(("0.0.0.0", C.WEB_PORT), Handler)
    except OSError:
        print(f"Portul {C.WEB_PORT} e ocupat: serverul rulează deja? Deschide {url}")
        if open_browser:
            import webbrowser
            webbrowser.open(url)
        sys.exit(1)
    server.daemon_threads = True
    APP = App()
    ip = local_ip()
    print(f"Pagina: {url}" + (f"  (de pe telefon / alt laptop din aceeași rețea: http://{ip}:{C.WEB_PORT})" if ip else ""))
    if os.name == "nt":
        print("Dacă Windows întreabă de firewall: „Allow” / „Permite”, ca pagina să se vadă și din rețea.")
    print(f"Camera: {APP.cam.source} | braț: {'simulat' if APP.sim else 'real'}")
    if C.HAT:
        print(hat.start())
    print("Oprire: Ctrl+C")
    if open_browser:
        import webbrowser
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        APP.runner.stop()
        APP.cam.stop()


if __name__ == "__main__":
    main()
