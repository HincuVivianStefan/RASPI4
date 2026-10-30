"""
Muți brațul din tastatură și salvezi pozițiile în positions.json.

  q/a  servo1 (bază)        w/s  servo2 (umăr)      e/d  servo3 (cot)
  r/f  servo4 (încheietură) t/g  servo5 (rotație)
  o / c  deschide / închide gripperul
  1..9   pasul (grade)
  p      salvează poziția curentă (îți cere numele)
  l      mergi la o poziție salvată
  n      ce poziții mai lipsesc
  x      ieșire
Rulare (din rădăcina proiectului): python3 tools/calibrate_arm.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # rădăcina proiectului

from dofbot import config as C  # noqa: E402
from dofbot.robot import REQUIRED, Robot  # noqa: E402

KEYS = {"q": (0, 1), "a": (0, -1), "w": (1, 1), "s": (1, -1), "e": (2, 1),
        "d": (2, -1), "r": (3, 1), "f": (3, -1), "t": (4, 1), "g": (4, -1)}
LIMITS = C.LIMITS

try:
    import msvcrt   # Windows

    def getch():
        return msvcrt.getwch()
except ImportError:  # Linux (Pi)
    import termios
    import tty

    def getch():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    C.SIM_VERBOSE = False   # aici linia de stare arată deja unghiurile
    rob = Robot()
    print("Duc brațul în poziția sigură (HOME sau SAFE_POSE)...")
    rob.safe()
    arm, pos = rob.arm, rob.pos
    angles = [round(a) for a in rob.last]
    grip = rob.grip
    step = 2
    print(__doc__)

    while True:
        print(f"\r unghiuri={angles} gripper={grip} pas={step}      ", end="", flush=True)
        k = getch()
        if k in KEYS:
            i, sign = KEYS[k]
            lo, hi = LIMITS[i]
            angles[i] = max(lo, min(hi, angles[i] + sign * step))
            arm.Arm_serial_servo_write(i + 1, angles[i], 150)
        elif k in "123456789":
            step = int(k)
        elif k == "o":
            grip = C.GRIP_OPEN
            arm.Arm_serial_servo_write(6, grip, 300)
        elif k == "c":
            grip = C.GRIP_CLOSE
            arm.Arm_serial_servo_write(6, grip, 300)
        elif k == "p":
            name = input("\nNume poziție (ex. S11_jos): ").strip()
            if name:
                pos[name] = list(angles)
                os.makedirs(C.DATA_DIR, exist_ok=True)
                with open(C.POS_FILE, "w", encoding="utf-8") as f:
                    json.dump(pos, f, indent=2)
                print(f"Salvat {name} = {angles}")
        elif k == "l":
            name = input("\nMergi la: ").strip()
            if name in pos:
                angles = list(pos[name])
                arm.Arm_serial_servo_write6(*angles, grip, 1000)
            else:
                print("Nu există.")
        elif k == "n":
            missing = [n for n in REQUIRED if n not in pos]
            print(f"\nLipsesc ({len(missing)}): {' '.join(missing)}")
        elif k in ("x", "\x03"):
            print("\nGata.")
            break


if __name__ == "__main__":
    main()
