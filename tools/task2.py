"""
Rulare (din rădăcina proiectului):
         python3 tools/task2.py 22 rosu galben verde albastru rosu
         (celula juriului = linie+coloană, apoi culorile de JOS în SUS)
         python3 tools/task2.py   -> folosește S2_TARGET și S2_ORDER din config / settings.json
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # rădăcina proiectului

from dofbot import camera, tasks  # noqa: E402
from dofbot.robot import Robot  # noqa: E402


def main():
    args = sys.argv[1:]
    if len(args) == 1:
        print(__doc__)
        sys.exit(1)
    target = args[0] if args else None
    order = [c.lower() for c in args[1:]] or None

    camera.shared = camera.Camera().start()
    try:
        tasks.run_task2(Robot(), tasks.ConsoleContext(), target, order)
    except ValueError as e:
        sys.exit(str(e))
    finally:
        camera.shared.stop()


if __name__ == "__main__":
    main()
