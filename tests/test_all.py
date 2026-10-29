"""Teste fără braț și fără cameră.  Rulare (din D:\\concurs):  py -m unittest discover -s tests -v"""
import json
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from dofbot import config as C  # noqa: E402
from dofbot import planner, synth, tasks, vision  # noqa: E402
from dofbot.robot import Robot, Stopped  # noqa: E402
from dofbot.sim_world import WORLD  # noqa: E402

COLORS = ["rosu", "galben", "verde", "albastru"]


def random_layout(rng, fill=0.7):
    return [[rng.choice(COLORS) if rng.random() < fill else None for _ in range(3)] for _ in range(3)]


class Quiet:
    def __init__(self):
        self.logs, self.published, self.pauses = [], {}, 0

    def log(self, msg):
        self.logs.append(msg)

    def pause(self, msg):
        self.pauses += 1

    def publish(self, k, v):
        self.published[k] = v


def setUpModule():
    C.reset_settings()
    C.CAPTURES_DIR = tempfile.mkdtemp()   # pozele de depanare ale sarcinilor nu ajung în proiect
    C.SIM_ENV, C.TEST_IMAGE = "1", "synth"
    C.SIM_TIME_SCALE, C.SIM_VERBOSE = 0, False
    C.STABLE_FRAMES = 3
    # Ca o instalare calibrată: colțurile caroiajelor salvate (scena simulată are S și D în același loc).
    corners = [[round(x), round(y)] for x, y in synth.grid_corners()]
    C.VISION_FILE = os.path.join(C.CAPTURES_DIR, "vision.json")
    with open(C.VISION_FILE, "w", encoding="utf-8") as f:
        json.dump({"S": corners, "D": corners}, f)


class TestPlanner(unittest.TestCase):
    def test_copy_follows_row_order(self):
        src = [["rosu", None, None], [None, "verde", None], [None, None, "galben"]]
        self.assertEqual(planner.plan_copy(src, [3, 2, 1]),
                         [("S33", "D33", 1, 1), ("S22", "D22", 1, 1), ("S11", "D11", 1, 1)])

    def test_tower_simple(self):
        cells = planner.grid_to_cells([["rosu", "galben", None], [None, None, None], ["verde", None, None]])
        moves = planner.plan_tower(cells, "S22", ["rosu", "galben", "verde"])
        self.assertEqual(moves, [("S11", "S22", 1, 1), ("S12", "S22", 2, 1), ("S31", "S22", 3, 1)])

    def test_tower_target_has_first_color(self):
        cells = planner.grid_to_cells([["galben", None, None], [None, "rosu", None], [None, None, None]])
        self.assertEqual(planner.plan_tower(cells, "S22", ["rosu", "galben"]), [("S11", "S22", 2, 1)])

    def test_tower_target_blocked_is_cleared(self):
        cells = planner.grid_to_cells([["rosu", None, None], [None, "verde", None], [None, None, None]])
        moves = planner.plan_tower(cells, "S22", ["rosu", "verde"])
        self.assertEqual(moves[0][0], "S22")              # întâi eliberează celula
        self.assertEqual([m[2] for m in moves[1:]], [1, 2])

    def test_tower_missing_color(self):
        cells = planner.grid_to_cells([["rosu", None, None], [None] * 3, [None] * 3])
        with self.assertRaises(ValueError):
            planner.plan_tower(cells, "S22", ["rosu", "rosu"])

    def test_tower_full_grid_parks_blocker_on_spare_cube(self):
        # Ca în imaginea din temă: caroiaj plin, ținta S22 are galben, turnul începe cu roșu.
        g = [["galben", "verde", "albastru"], ["rosu", "galben", "verde"], ["albastru", "rosu", "galben"]]
        moves = planner.plan_tower(planner.grid_to_cells(g), "S22", ["rosu", "galben", "verde", "albastru", "rosu"])
        park = moves[0]
        self.assertEqual((park[0], park[2]), ("S22", 2))            # galbenul de pe țintă, pus peste alt cub
        self.assertEqual([m[2] for m in moves[1:]], [1, 2, 3, 4, 5])
        self.assertNotIn(park[1], [m[0] for m in moves[1:] if m[3] == 1])   # cubul de dedesubt nu e folosit

    def test_tower_full_grid_no_spare(self):
        g = [["rosu", "galben", "verde"], ["albastru", "galben", "verde"], ["rosu", "albastru", "galben"]]
        order = ["rosu", "rosu", "galben", "galben", "galben", "verde", "verde", "albastru", "albastru"]
        with self.assertRaises(ValueError):
            planner.plan_tower(planner.grid_to_cells(g), "S22", order)


class TestVision(unittest.TestCase):
    def setUp(self):
        vision.forget()

    def test_finds_grid_and_colors(self):
        """De la zero: caroiaj cu câteva cuburi (cuburile 3D îi acoperă mult din linii)."""
        rng = random.Random(1)
        for seed in range(15):
            layout = random_layout(rng, fill=0.3)
            frame = synth.render(layout, seed=seed)
            corners = vision.find_grid(frame)
            self.assertIsNotNone(corners, f"caroiaj negăsit (seed {seed})")
            for p, q in zip(corners, synth.grid_corners()):
                self.assertLess(max(abs(p[0] - q[0]), abs(p[1] - q[1])), 15, f"colț greșit (seed {seed})")
            colors, _, _ = vision.detect_cells(frame, corners)
            self.assertEqual(colors, layout, f"seed {seed}")

    def test_tracks_grid_under_many_cubes(self):
        """Ca în pagină: caroiajul e găsit o dată, apoi urmărit de la cadru la cadru, chiar dacă e aproape plin.
        Cuburile sunt puse în celula în care le stă baza, nu în cea peste care cade fața lor de sus."""
        rng = random.Random(2)
        prev = vision.find_grid(synth.render([[None] * 3 for _ in range(3)]))
        for seed in range(15):
            layout = random_layout(rng)
            frame = synth.render(layout, seed=seed)
            corners = vision.find_grid(frame, prev)
            self.assertIsNotNone(corners, f"caroiaj negăsit (seed {seed})")
            # Colțurile acoperite de cuburi se deduc din liniile rămase, deci pot fi mai puțin exacte.
            for p, q in zip(corners, synth.grid_corners()):
                self.assertLess(max(abs(p[0] - q[0]), abs(p[1] - q[1])), 25, f"colț greșit (seed {seed})")
            colors, _, _ = vision.detect_cells(frame, corners)
            self.assertEqual(colors, layout, f"seed {seed}")

    def test_low_camera_uses_cube_base(self):
        """Cameră joasă (ca un laptop): fața de sus a cubului cade peste celula din spate sau în afara
        caroiajului. Cubul trebuie pus în celula în care îi stă baza."""
        eye = synth.EYE
        synth.EYE = np.array([1.6, 7.5, 6.5])
        try:
            layout = [["rosu", None, "verde"], [None, "galben", None], ["albastru", None, "rosu"]]
            corners = synth.grid_corners()
            cubes, rejected = vision.find_cubes(synth.render(layout, seed=1), corners)
            self.assertEqual(vision.cubes_to_grid(cubes, corners), layout)
            self.assertEqual(rejected, [])
        finally:
            synth.EYE = eye

    def test_cube_turned_color_on_side(self):
        """Cub întors: culoarea e pe fața dinspre cameră, sus e alb. Tot trebuie găsit, în celula lui,
        cu fața colorată recunoscută; cuburile normale rămân „sus”."""
        eye = synth.EYE
        try:
            for synth.EYE in (eye, np.array([1.6, 7.5, 6.5])):   # camera de pe braț și una joasă
                layout = [[("rosu", 1, "fata"), None, "verde"], [None, ("galben", 1, "fata"), "albastru"],
                          [("albastru", 1, "fata"), None, ("rosu", 1, "fata")]]
                corners = synth.grid_corners()
                cubes, _ = vision.find_cubes(synth.render(layout, seed=5), corners)
                vision.cubes_to_grid(cubes, corners)
                want = {f"{r + 1}{c + 1}": (v, "sus") if isinstance(v, str) else (v[0], v[2])
                        for r, row in enumerate(layout) for c, v in enumerate(row) if v}
                self.assertEqual({o["cell"]: (o["color"], o["face"]) for o in cubes}, want)
                faces = vision.detect_full(synth.render(layout, seed=5), corners)["faces"]
                self.assertEqual(faces, [[want[f"{r + 1}{c + 1}"][1] if f"{r + 1}{c + 1}" in want else None
                                          for c in range(3)] for r in range(3)])
        finally:
            synth.EYE = eye

    def test_cubes_placed_sloppily(self):
        """Cuburi puse neglijent: mutate până la un sfert de celulă, rotite până la 40°, unele întoarse
        (culoarea pe față), câte 4 pe caroiaj. Aproape toate trebuie găsite în celula corectă."""
        rng = random.Random(7)
        corners = synth.grid_corners()
        good = 0
        for seed in range(20):
            layout = [[None] * 3 for _ in range(3)]
            for cell in rng.sample(range(9), 4):
                layout[cell // 3][cell % 3] = (rng.choice(COLORS), 1, rng.choice(["sus", "sus", "fata"]))
            frame = synth.render(layout, seed=seed, jitter=30, yaw=40)
            found = vision.cubes_to_grid(vision.find_cubes(frame, corners)[0], corners)
            good += found == [[v[0] if v else None for v in row] for row in layout]
        self.assertGreaterEqual(good, 18)

    def test_empty_grid(self):
        frame = synth.render([[None] * 3 for _ in range(3)])
        corners = vision.find_grid(frame)
        self.assertIsNotNone(corners)
        self.assertEqual(vision.detect_cells(frame, corners)[0], [[None] * 3 for _ in range(3)])

    def _cell_center(self, r, c):
        g = np.float32(synth.grid_corners())
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [3, 0], [3, 3], [0, 3]]), g)
        return cv2.perspectiveTransform(np.float32([[[c + 0.5, r + 0.5]]]), M)[0, 0]

    def test_cubes_are_objects(self):
        layout = [["rosu", None, None], [None, "verde", None], [None, None, None]]
        frame = synth.render(layout, seed=3)
        corners = vision.find_grid(frame)
        x, y = map(int, self._cell_center(0, 2))
        cv2.rectangle(frame, (x - 50, y - 10), (x + 50, y + 10), synth.CUBE_BGR["albastru"], -1)  # bandă
        x, y = map(int, self._cell_center(2, 0))
        cv2.circle(frame, (x, y), 3, synth.CUBE_BGR["galben"], -1)                              # punct
        cubes, rejected = vision.find_cubes(frame, corners)
        self.assertEqual(vision.cubes_to_grid(cubes, corners), layout)
        self.assertEqual(sorted(o["reason"] for o in rejected if o["color"] == "albastru"), ["alungit"])

    def test_big_colored_thing_is_not_a_cube(self):
        frame = synth.render([[None] * 3 for _ in range(3)], seed=4)
        corners = vision.find_grid(frame)
        cv2.rectangle(frame, (0, 0), (200, 140), synth.CUBE_BGR["rosu"], -1)   # ex. un tricou roșu
        self.assertEqual(vision.cubes_to_grid(vision.find_cubes(frame, corners)[0], corners),
                         [[None] * 3 for _ in range(3)])
        self.assertIn("mare", [o["reason"] for o in vision.find_cubes(frame, corners)[1]])

    def test_images_in_path_with_diacritics(self):
        """Pe Windows, cv2.imwrite/imread nu merg cu diacritice în cale (ex. utilizatorul „Ștefan”)."""
        folder = os.path.join(C.CAPTURES_DIR, "Descărcări Ștefan")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "poză.jpg")
        frame = synth.render([[None] * 3 for _ in range(3)])
        self.assertTrue(vision.imwrite(path, frame))
        self.assertEqual(vision.imread(path).shape, frame.shape)

    def test_vote(self):
        a = [["rosu"] * 3] * 3
        b = [[None] * 3] * 3
        self.assertEqual(vision.vote([a, b, a]), a)


class TestTasksSimulated(unittest.TestCase):
    """Brațul simulat mută cuburile în lumea simulată; camera sintetică o desenează."""

    def setUp(self):
        vision.forget()

    def test_task1_end_to_end(self):
        layout = [["galben", None, "albastru"], ["rosu", "galben", None], [None, "rosu", "verde"]]
        WORLD.reset(layout)
        ctx = Quiet()
        res = tasks.run_task1(Robot(), ctx)
        self.assertEqual(res["source"], layout)
        self.assertEqual(res["errors"], [], "\n".join(ctx.logs))
        self.assertEqual([[t for t, _ in row] for row in WORLD.snapshot("D")], layout)

    def test_task2_end_to_end(self):
        WORLD.reset(C.SYNTH_LAYOUT)
        order = ["rosu", "galben", "verde", "albastru", "rosu"]
        tasks.run_task2(Robot(), Quiet(), "22", order)
        self.assertEqual(WORLD.grids["S"][1][1], order)

    def test_stop(self):
        WORLD.reset(C.SYNTH_LAYOUT)
        with self.assertRaises(Stopped):
            tasks.run_task1(Robot(should_stop=lambda: True), Quiet())


class TestSettings(unittest.TestCase):
    def tearDown(self):
        C.reset_settings()
        C.STABLE_FRAMES = 3

    def test_apply_and_validate(self):
        C.apply_settings({"MIN_FRACTION": "0.5", "S2_TARGET": "13"})
        self.assertEqual((C.MIN_FRACTION, C.S2_TARGET), (0.5, "13"))
        for bad in ({"S2_TARGET": "44"}, {"S1_ROW_ORDER": [1, 1, 2]}, {"S2_ORDER": ["mov"]}, {"NIMIC": 1}):
            with self.assertRaises(ValueError):
                C.apply_settings(bad)
        self.assertEqual(C.S2_TARGET, "13")   # setările greșite nu schimbă nimic


if __name__ == "__main__":
    unittest.main()
