"""Sarcina 1 din linia de comandă.  Rulare (din rădăcina proiectului): python3 tools/task1.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # rădăcina proiectului

from dofbot import camera, tasks  # noqa: E402
from dofbot.robot import Robot  # noqa: E402


def main():
    camera.shared = camera.Camera().start()
    try:
        tasks.run_task1(Robot(), tasks.ConsoleContext())
    finally:
        camera.shared.stop()


if __name__ == "__main__":
    main()
