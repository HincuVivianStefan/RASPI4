"""
Muți brațul dintr-un controller USB. Butoanele sunt pentru gamepadul DragonRise (cifrele de pe el):

  stick stâng  stânga/dreapta  servo1 (bază)      sus/jos  servo2 (umăr)
  L1 / R1  (5 / 6)   servo4 (încheietură) jos / sus
  L2 / R2  (7 / 8)   servo3 (cot) jos / sus
  3 / 4              servo5 (rotația gripperului) stânga / dreapta
  1 / 2              deschide / închide gripperul
  Select (9)         salvează poziția curentă (îți cere numele în terminal)
  apăsat stick stâng (11)  mergi la HOME
  D-pad sus/jos      viteza mai mare / mai mică
  Start (10)         ieșire
  stick drept (doar cu Analog aprins)  servo5 / servo3, ca butoanele de mai sus
Cu pagina web pornită nu e nevoie de el: pagina citește singură controllerul (dofbot/pad.py).
Nu le porniți pe amândouă: ar comanda brațul în același timp.
Rulare (din rădăcina proiectului):
  python3 tools/gamepad.py          controlul brațului
  python3 tools/gamepad.py --test   afișează numerele axelor și butoanelor (pentru MAP)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # rădăcina proiectului

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")              # fără fereastră, merge și prin SSH
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402

from dofbot import config as C  # noqa: E402
from dofbot.arm_driver import get_arm  # noqa: E402
from dofbot.robot import load_positions  # noqa: E402

# Butoanele și vitezele sunt aceleași ca în pagină: dofbot/pad.py (acolo se reglează MAP).
from dofbot.pad import DEADZONE, GRIP_SPEED, HZ, LIMITS, MAP, RAMP_S, SPEEDS, START_MS, read_servo  # noqa: E402


def wait_for_joystick():
    print("Aștept controllerul USB... (Ctrl+C ca să renunți)")
    while pygame.joystick.get_count() == 0:
        if pygame.event.get(pygame.QUIT):   # SDL transformă Ctrl+C în QUIT
            raise KeyboardInterrupt
        time.sleep(0.5)
    js = pygame.joystick.Joystick(0)
    print(f"Controller: {js.get_name()}  ({js.get_numaxes()} axe, {js.get_numbuttons()} butoane, "
          f"{js.get_numhats()} D-pad)")
    return js


def test_mode(js):
    print("Mișcă stickurile și apasă butoanele. Ctrl+C pentru ieșire.")
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise KeyboardInterrupt
            if ev.type == pygame.JOYAXISMOTION and abs(ev.value) > 0.5:
                print(f"axa {ev.axis}: {ev.value:+.2f}")
            elif ev.type == pygame.JOYBUTTONDOWN:
                print(f"buton {ev.button}")
            elif ev.type == pygame.JOYHATMOTION:
                print(f"D-pad {ev.hat}: {ev.value}")
        time.sleep(0.02)


def start_angles(arm, pos):
    """Unghiurile actuale ale brațului, ca să nu sară la pornire."""
    home = list(pos.get("HOME", [90, 90, 90, 90, 90]))
    angles = []
    for i in range(5):
        a = read_servo(arm, i + 1)
        if a is None:
            print(f"servo{i + 1} nu răspunde: merge încet la {home[i]}°")
        angles.append(a if a is not None else home[i])
        time.sleep(0.02)
    grip = read_servo(arm, 6)
    return angles, grip if grip is not None else C.GRIP_OPEN


def save_position(pos, angles):
    try:
        name = input("\nNume poziție (ex. S11_jos, gol = renunță): ").strip()
    except EOFError:   # pornit fără terminal
        return
    if name:
        pos[name] = [round(a) for a in angles]
        os.makedirs(C.DATA_DIR, exist_ok=True)
        with open(C.POS_FILE, "w", encoding="utf-8") as f:
            json.dump(pos, f, indent=2)
        print(f"Salvat {name} = {pos[name]}")


def main():
    pygame.display.init()
    pygame.joystick.init()
    js = wait_for_joystick()
    if "--test" in sys.argv:
        return test_mode(js)

    C.SIM_VERBOSE = False   # linia de stare arată deja unghiurile
    arm, sim = get_arm()
    pos = load_positions()
    angles, grip = start_angles(arm, pos)
    # Prima comandă, lentă: dacă vreun servo n-a putut fi citit, nu sare brusc (vârf de curent = Pi resetat).
    arm.Arm_serial_servo_write6(*angles, round(grip), START_MS)
    time.sleep(START_MS * C.TIME_UNIT_S)
    grip_target = grip
    vel = [0.0] * 5   # viteza fiecărui servo, -1..1 din viteza maximă
    speed = 1
    sent = None
    print(__doc__)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise KeyboardInterrupt
            if ev.type == pygame.JOYBUTTONDOWN:
                if ev.button == MAP["open"]:
                    grip_target = C.GRIP_OPEN
                elif ev.button == MAP["close"]:
                    grip_target = C.GRIP_CLOSE
                elif ev.button == MAP["home"] and "HOME" in pos:
                    angles = [float(a) for a in pos["HOME"]]
                    arm.Arm_serial_servo_write6(*angles, round(grip), 1000)
                    time.sleep(1000 * C.TIME_UNIT_S)
                    vel = [0.0] * 5
                    sent = None
                elif ev.button == MAP["save"]:
                    save_position(pos, angles)
                elif ev.button == MAP["quit"]:
                    print("\nGata.")
                    return
            elif ev.type == pygame.JOYHATMOTION and ev.value[1]:
                speed = max(0, min(len(SPEEDS) - 1, speed + ev.value[1]))
            elif ev.type == pygame.JOYDEVICEREMOVED:
                print("\nController deconectat: brațul rămâne pe loc.")
                js = wait_for_joystick()

        # Viteza dorită (-1..1) din stickuri și butoane; viteza reală o urmează treptat,
        # ca brațul să accelereze și să frâneze lin chiar dacă stickul e digital (doar 0 sau ±1).
        want = [0.0] * 5
        for axis, (i, sign) in MAP["axes"].items():
            v = js.get_axis(axis) if axis < js.get_numaxes() else 0.0
            if abs(v) > DEADZONE:
                want[i] += sign * v
        for i, (minus, plus) in MAP["buttons"].items():
            want[i] += js.get_button(plus) - js.get_button(minus)
        ramp = 1 / (RAMP_S * HZ)
        for i, (lo, hi) in enumerate(LIMITS):
            w = max(-1.0, min(1.0, want[i]))
            vel[i] += max(-ramp, min(ramp, w - vel[i]))
            angles[i] += vel[i] * SPEEDS[speed] / HZ
            if not lo <= angles[i] <= hi:
                angles[i] = max(lo, min(hi, angles[i]))
                vel[i] = 0.0
        grip += max(-GRIP_SPEED / HZ, min(GRIP_SPEED / HZ, grip_target - grip))

        # Arm_Lib primește și zecimale: pași de 0.1° în loc de 1-2° la fiecare comandă
        cmd = [round(a, 1) for a in angles] + [round(grip)]
        if cmd != sent:
            arm.Arm_serial_servo_write6(*cmd, int(1000 / HZ * 2))
            sent = cmd
            axes = " ".join(f"{js.get_axis(a):+.1f}" for a in range(js.get_numaxes()))
            print(f"\r unghiuri={[round(a) for a in angles]} gripper={round(grip)} "
                  f"viteza={SPEEDS[speed]}°/s axe=[{axes}]  ", end="", flush=True)
        time.sleep(1 / HZ)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGata.")
