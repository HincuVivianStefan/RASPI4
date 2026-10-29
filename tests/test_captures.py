"""Detecția pe poze adevărate, înregistrate din pagină („Înregistrează 10 cadre”), fără braț.

Fiecare captures/set_*/ are cadrele (00.png ...) și labels.json cu aranjarea corectă. Se folosesc doar seturile
verificate de mână ("verificat": true). Rulare (din rădăcina proiectului):  python3 tests/test_captures.py
Setările sunt cele din data/settings.json, deci așa verifici un reglaj nou pe toate pozele vechi.
"""
import glob
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dofbot import config as C  # noqa: E402
from dofbot import vision  # noqa: E402

MIN_OK = 0.9    # partea din cadrele unui set care trebuie să iasă exact ca în labels.json


def labeled_sets():
    for folder in sorted(glob.glob(os.path.join(C.ROOT, "captures", "set_*"))):
        try:
            with open(os.path.join(folder, "labels.json"), encoding="utf-8") as f:
                labels = json.load(f)
        except (FileNotFoundError, ValueError):
            continue
        if labels.get("verificat"):
            yield folder, labels["colors"]


class TestCaptures(unittest.TestCase):
    def test_labeled_sets(self):
        sets = list(labeled_sets())
        if not sets:
            self.skipTest("niciun set verificat în captures/ (înregistrează din pagină, apoi verifică labels.json)")
        for folder, want in sets:
            with self.subTest(set=os.path.basename(folder)):
                vision.forget()
                prev, good, frames = None, 0, sorted(glob.glob(os.path.join(folder, "*.png")))
                for path in frames:
                    frame = vision.imread(path)
                    corners = (vision.find_grid(frame, prev) if prev else None) or vision.find_grid(frame)
                    if corners is None:
                        print(f"  {os.path.basename(folder)}/{os.path.basename(path)}: caroiaj negăsit")
                        continue
                    prev = corners
                    got = vision.detect_full(frame, corners)["colors"]
                    good += got == want
                    if got != want:
                        print(f"  {os.path.basename(folder)}/{os.path.basename(path)}: {got}")
                self.assertGreaterEqual(good, MIN_OK * len(frames), f"{good}/{len(frames)} cadre corecte")


if __name__ == "__main__":
    unittest.main()
