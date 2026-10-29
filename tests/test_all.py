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
from dofbot import planner, sim_world, synth, tasks, vision  # noqa: E402
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

    def test_low_and_colored_light(self):
        """Sală întunecată, bec cald / rece: foaia e adusă la alb, deci caroiajul și culorile (și verdele
        închis) rămân recunoscute."""
        rng = random.Random(4)
        for light, cast in ((0.35, (1, 1, 1)), (0.2, (1, 1, 1)), (0.35, (1.0, 0.85, 0.6)), (0.35, (0.8, 0.9, 1.0))):
            prev = vision.find_grid(synth.render([[None] * 3 for _ in range(3)], light=light, cast=cast))
            self.assertIsNotNone(prev, f"caroiaj gol negăsit la lumina {light} {cast}")
            for seed in range(5):
                layout = random_layout(rng)
                frame = synth.render(layout, seed=seed, light=light, cast=cast)
                corners = vision.find_grid(frame, prev)
                self.assertIsNotNone(corners, f"lumina {light} {cast}, seed {seed}")
                self.assertEqual(vision.detect_cells(frame, corners)[0], layout, f"lumina {light} {cast}, seed {seed}")

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


class FakeCap:
    """Cameră falsă: poza e cu atât mai luminoasă cu cât expunerea e mai mare (300 = lumină normală)."""

    def __init__(self, exposure):
        self.props = {cv2.CAP_PROP_EXPOSURE: exposure}

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def frame(self):
        return synth.render([["verde", None, None], [None] * 3, [None] * 3],
                            light=self.props[cv2.CAP_PROP_EXPOSURE] / 300)


class TestCamera(unittest.TestCase):
    def tearDown(self):
        C.reset_settings()
        C.STABLE_FRAMES = 3

    def test_auto_expose_brings_paper_to_white(self):
        """Din întuneric și din imagine arsă: expunerea ajunge să facă foaia ~EXPOSE_TARGET, apoi rămâne fixă."""
        for start in (30, 2000):
            cap = FakeCap(start)
            exp = vision.auto_expose(cap, cap.frame, log=lambda m: None)
            self.assertEqual(exp, C.CAM_EXPOSURE)
            self.assertEqual(cap.props[cv2.CAP_PROP_AUTO_EXPOSURE], 1)    # manual
            v = vision.paper_color(cap.frame()).max()
            self.assertLess(abs(v - vision.EXPOSE_TARGET), 0.1 * vision.EXPOSE_TARGET, f"de la {start}")


class TestTracking(unittest.TestCase):
    """Urmărirea caroiajului din pagină (web.Pipeline.track), cadru cu cadru."""

    def setUp(self):
        from dofbot import web
        vision.forget()
        self.p = web.Pipeline(cam=None)
        self.p.set_grid("S")

    def feed(self, frame, n=1):
        for _ in range(n):
            self.p.track(frame)
            self.p.last_small = None    # cadre sintetice identice: fără „cameră în mișcare”
        return self.p.corners

    def test_grid_removed_is_not_drawn(self):
        """Caroiajul luat de pe masă (cu sau fără cuburi) dispare în cel mult LOST_AFTER cadre."""
        from dofbot import web
        half = [["rosu", None, "verde"], [None, "galben", None], ["albastru", None, None]]
        for gone in (synth.render(half, lines=False), synth.render(half, paper=False),
                     synth.render([[None] * 3 for _ in range(3)], paper=False)):
            self.setUp()
            self.assertIsNotNone(self.feed(synth.render(half, seed=2), 2))
            self.assertIsNone(self.feed(gone, web.LOST_AFTER))
            self.assertEqual(self.p.grid_status, "pierdut")

    def test_wrong_saved_corners_are_not_shown(self):
        """Colțuri salvate pe alt calculator (vision.json greșit): nu se desenează un caroiaj inexistent."""
        bad = [[x + 120, y - 60] for x, y in synth.grid_corners()]
        self.p.saved = {"S": bad}
        self.assertIsNone(self.feed(synth.render([[None] * 3 for _ in range(3)], paper=False), 3))
        found = self.feed(synth.render([["rosu", None, None], [None] * 3, [None] * 3]))
        for p, q in zip(found, synth.grid_corners()):     # cu caroiajul în cadru, e găsit la locul lui
            self.assertLess(max(abs(p[0] - q[0]), abs(p[1] - q[1])), 15)

    def test_full_grid_stays_tracked(self):
        """Caroiajul plin (liniile aproape toate sub cuburi) rămâne urmărit."""
        self.feed(synth.render([[None] * 3 for _ in range(3)]))
        self.assertIsNotNone(self.feed(synth.render(C.SYNTH_LAYOUT, seed=3), 6))
        self.assertIn(self.p.grid_status, ("urmărit", "acoperit", "găsit"))


class TestArmCamera(unittest.TestCase):
    """Camera de pe brațul simulat (sim_world.arm_view): caroiajul plin, cel văzut doar parțial și centrarea."""
    HALF = [["rosu", None, "verde"], [None, "galben", None], ["albastru", None, "rosu"]]

    def setUp(self):
        vision.forget()

    def tearDown(self):
        WORLD.reset()

    @staticmethod
    def err(a, b):
        return float(np.abs(np.float32(a) - np.float32(b)).max())

    def test_full_grid_found_from_cubes(self):
        """9 cuburi, liniile aproape toate ascunse: caroiajul e găsit de la zero, din rețeaua fețelor de sus."""
        rng, good = random.Random(11), 0
        for k in range(6):
            layout = [[rng.choice(COLORS) for _ in range(3)] for _ in range(3)]
            WORLD.reset(layout)
            a = [90 + rng.uniform(-6, 6), 112 + rng.uniform(-5, 5), 15 + rng.uniform(-5, 5), 1 + rng.uniform(0, 6), 90]
            frame = sim_world.arm_view(a, seed=k)
            vision.forget()
            corners = vision.find_grid(frame)
            good += (corners is not None and self.err(corners, sim_world.true_corners("S", a)) < 30
                     and vision.detect_full(frame, corners)["colors"] == layout)
        self.assertGreaterEqual(good, 5)

    def test_partial_grid_points_the_way(self):
        """Se vede doar o parte (baza rotită): caroiajul e extrapolat în afara imaginii, cu centrul pe partea bună."""
        for layout in ([[None] * 3 for _ in range(3)], self.HALF):
            WORLD.reset(layout)
            for base in (66, 118):
                a = [base, 108, 18, 4, 90]
                frame = sim_world.arm_view(a, seed=1)
                truth = np.float32(sim_world.true_corners("S", a))
                self.assertTrue(((truth < 0) | (truth > [640, 480])).any(), "caroiajul trebuie să iasă din imagine")
                vision.forget()
                part = vision.find_partial_grid(frame)
                if part is None:
                    hint = vision.lines_hint(frame)
                    self.assertIsNotNone(hint, f"baza {base}: nici parțial, nici linii")
                    got = np.float32(hint) * [640, 480]
                else:
                    got = np.float32(part).mean(0)
                want = truth.mean(0)
                # spre stânga / dreapta ca în realitate (semnul față de mijlocul imaginii)
                self.assertEqual(np.sign(got[0] - 320), np.sign(want[0] - 320), f"baza {base}, {layout == self.HALF}")

    def test_no_lines_on_empty_table(self):
        self.assertIsNone(vision.lines_hint(synth.render([[None] * 3 for _ in range(3)], paper=False)))
        self.assertIsNone(vision.find_partial_grid(synth.render([[None] * 3 for _ in range(3)], paper=False)))

    def test_aim_from_corner(self):
        """Brațul vede doar un colț al caroiajului: se mută până îl vede întreg, centrat. A doua oară pornește
        cu efectul articulațiilor ținut minte (data/aim.json), fără mișcări de probă."""
        from dofbot import aim
        old = aim.JACOBIAN_FILE
        aim.JACOBIAN_FILE = os.path.join(C.CAPTURES_DIR, "aim.json")
        try:
            for start in ([68, 108, 18, 4, 90], [114, 112, 15, 1, 90]):
                WORLD.reset(self.HALF)
                vision.forget()
                rob = Robot()
                rob.move(start, 200)
                final = aim.Aimer(rob, lambda: sim_world.arm_view(rob.last, seed=len(rob.arm.history)),
                                  lambda m: None, need_cubes=1).run(search=False, center=True)
                self.assertIsNotNone(final, f"de la {start}")
                t = np.float32(sim_world.true_corners("S", final)) / [640, 480]
                self.assertTrue(((t > 0.02) & (t < 0.98)).all(), f"de la {start}: caroiajul nu e întreg în imagine")
                self.assertLess(np.abs(t.mean(0) - 0.5).max(), 0.12)
            self.assertTrue(os.path.exists(aim.JACOBIAN_FILE))
        finally:
            aim.JACOBIAN_FILE = old


class TestFetch(unittest.TestCase):
    """Sarcina 1 cu unghiul mort: cuburile mutate în jurul brațului sunt găsite, luate și puse la loc."""
    LAYOUT = [["rosu", None, "verde"], [None, "galben", None], ["albastru", None, "rosu"]]
    LOOK = [90, 112, 15, 1, 90]

    def setUp(self):
        self.focal, C.CAM_FOCAL = C.CAM_FOCAL, sim_world.FOCAL

    def tearDown(self):
        C.CAM_FOCAL = self.focal
        WORLD.reset()

    def test_ik_round_trip(self):
        from dofbot import kinematics
        for p in ([0.15, 0, 0.015], [0.2, 0.08, 0.015], [0.12, -0.05, 0.015]):
            down, up, _ = kinematics.reach(p, 0.05)
            np.testing.assert_allclose(kinematics.grip(down)[0], p, atol=1e-6)

    def test_restore_scattered_and_packed(self):
        from dofbot import fetch
        for seed, packed in ((3, False), (11, True), (14, False), (15, True)):
            WORLD.reset(self.LAYOUT)
            WORLD.scatter(seed=seed, packed=packed)
            rob = Robot()
            rob.move(self.LOOK, 200)
            grid = fetch.grid_area(self.LOOK, sim_world.true_corners("S", self.LOOK), (640, 480))
            f = fetch.Fetcher(rob, lambda: sim_world.arm_view(rob.last, seed=len(rob.arm.history)), lambda m: None, grid)
            res = fetch.restore(rob, Quiet(), self.LAYOUT, self.LOOK, f)
            self.assertEqual(res["missing"], [], f"seed {seed}, lipite {packed}")
            self.assertEqual([[t for t, _ in row] for row in WORLD.snapshot("S")], self.LAYOUT)

    def test_restore_some_left_on_grid(self):
        """Cuburile luate unul câte unul, nu toate: cele rămase pe caroiaj nu se mută."""
        from dofbot import fetch
        WORLD.reset(self.LAYOUT)
        WORLD.scatter(seed=3, packed=False)
        present = [[None] * 3 for _ in range(3)]
        for r, c in ((0, 2), (1, 1)):    # verde din S13 și galben din S22 rămân pe caroiaj
            cube = next(o for o in WORLD.table if o["color"] == self.LAYOUT[r][c])
            WORLD.table.remove(cube)
            WORLD.grids["S"][r][c] = [cube["color"]]
            present[r][c] = cube["color"]
        rob = Robot()
        rob.move(self.LOOK, 200)
        grid = fetch.grid_area(self.LOOK, sim_world.true_corners("S", self.LOOK), (640, 480))
        f = fetch.Fetcher(rob, lambda: sim_world.arm_view(rob.last, seed=len(rob.arm.history)), lambda m: None, grid)
        res = fetch.restore(rob, Quiet(), self.LAYOUT, self.LOOK, f, present=present)
        self.assertEqual(res["missing"], [])
        self.assertEqual(sorted(res["placed"]), ["S11", "S31", "S33"])
        self.assertEqual([[t for t, _ in row] for row in WORLD.snapshot("S")], self.LAYOUT)
        self.assertEqual([h for row in WORLD.snapshot("S") for _, h in row], [1, 0, 1, 0, 1, 0, 1, 0, 1])


class TestMotion(unittest.TestCase):
    """Mișcarea lină a lui Robot.move, văzută prin comenzile primite de brațul simulat."""

    def sent(self, rob):
        return [np.float64(a[:5]) for _, a in rob.arm.history]

    def test_move_is_smooth_and_exact(self):
        rob = Robot()
        rob.last = [90.0] * 5
        rob.arm.history.clear()
        target = [130, 60, 100, 75, 90]
        rob.move(target, 600)
        pts = [np.float64([90] * 5)] + self.sent(rob)
        steps = np.array([np.abs(b - a).max() for a, b in zip(pts, pts[1:])])
        np.testing.assert_allclose(pts[-1], target, atol=0.05)
        self.assertLessEqual(steps.max(), C.V_MAX / C.MOTION_HZ * 1.01)          # viteza maximă respectată
        self.assertLess(steps[0], 0.1 * steps.max())                             # pornește lin
        self.assertLess(steps[-1], 0.1 * steps.max())                            # oprește lin
        self.assertGreaterEqual(len(steps), 0.6 * C.MOTION_HZ)                   # 40° la 90°/s: ≥ 0,83 s
        # Toate articulațiile ajung odată: la jumătatea drumului, fiecare e la jumătatea propriei distanțe.
        mid, moving = pts[len(pts) // 2], np.float64(target) != 90
        np.testing.assert_allclose((mid - 90)[moving] / (np.float64(target) - 90)[moving], 0.5, atol=0.06)

    def test_stop_mid_move(self):
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] > 10
        rob = Robot(should_stop=stop)
        rob.last = [90.0] * 5
        rob.arm.history.clear()
        with self.assertRaises(Stopped):
            rob.move([150, 90, 90, 90, 90], 1000)
        self.assertLessEqual(len(rob.arm.history), 10)    # nu mai trimite niciun punct după Stop

    def test_limits(self):
        rob = Robot()
        rob.last = [90.0] * 5
        with self.assertRaises(ValueError):
            rob.move([200, 90, 90, 90, 90])               # mult în afară: greșeală, nu se mișcă
        rob.move([182, 90, 90, 90, -1], 200)              # puțin în afară: adus la limită
        self.assertEqual(rob.last[0], 180)
        self.assertEqual(rob.last[4], 0)


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
