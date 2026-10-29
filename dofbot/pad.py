"""Controllerul USB, citit direct de pagina web.

- Cât timp nu rulează nimic automat, muți brațul din controller (ca tools/gamepad.py).
- OPRIRE DE URGENȚĂ: în timpul unei sarcini sau al căutării caroiajului, orice buton, stick sau D-pad
  o oprește pe loc și brațul îngheață unde a ajuns. Controlul manual revine după ce lași totul liber.
- Select (9) pornește Pasul 1 (caută caroiajul, salvează unghiurile, memorează cuburile); Start (10) doar oprește (nu iese, ca în gamepad.py).
Fără pygame sau fără controller, pagina merge normal (doar fără toate acestea).
"""
import os
import threading
import time
import traceback

from . import config as C
from .arm_driver import get_arm
from .robot import load_positions

# Numerele axelor/butoanelor diferă între controllere: verifică cu tools/gamepad.py --test.   # REGLEAZĂ
# Valorile de mai jos sunt pentru gamepadul DragonRise (0079:0006); numărul butonului = cifra de pe el - 1.
MAP = {
    "axes": {0: (0, -1), 1: (1, -1), 4: (2, -1), 3: (4, 1)},   # axă -> (servo 0..4, sens)
    "buttons": {3: (4, 5), 2: (6, 7), 4: (2, 3)},               # servo 0..4 -> (buton minus, buton plus)
    "open": 0, "close": 1, "save": 8, "home": 10,
    "quit": 9,                                                  # Start
}
LIMITS = C.LIMITS
DEADZONE = 0.15
SPEEDS = [20, 45, 90]   # grade pe secundă la stick împins la maxim
RAMP_S = 0.4            # în cât timp ajunge la viteza maximă / se oprește (mai mare = mai lin)
GRIP_SPEED = 150        # grade pe secundă pentru deschiderea / închiderea gripperului
HZ = 40                 # de câte ori pe secundă se citește controllerul și se trimit unghiurile
START_MS = 1500         # mișcare lentă când un servo n-a putut fi citit (fără smucituri = fără vârf de curent)


def read_servo(arm, sid, tries=5):
    """Citirea unui servo dă uneori None (I2C ocupat): mai încercăm de câteva ori."""
    for _ in range(tries):
        a = arm.Arm_serial_servo_read(sid)
        if a is not None:
            return a
        time.sleep(0.05)
    return None


def sdl_env():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")              # fără fereastră, merge și prin SSH
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    # Altfel SDL preia Ctrl+C / SIGTERM (le face evenimente pe care nu le citim) și pagina nu se mai oprește.
    os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")


class Pad:
    def __init__(self, runner):
        self.runner = runner        # web.Runner: busy(), stop(), wait_stopped(), start("aim"), log()
        self.status = "pornește"
        self.name = None
        self.stops = 0
        self.arm = None

    def state(self):
        return {"status": self.status, "name": self.name, "stops": self.stops}

    def start(self):
        try:
            sdl_env()
            import pygame  # noqa: F401
        except ImportError:
            self.status = "lipsește pygame"
            return self
        threading.Thread(target=self._run, daemon=True, name="pad").start()
        return self

    def _run(self):
        import pygame
        pygame.display.init()
        pygame.joystick.init()
        while True:
            try:
                self.status, self.name = "fără controller", None
                while pygame.joystick.get_count() == 0:
                    pygame.event.get()
                    time.sleep(1.0)
                js = pygame.joystick.Joystick(0)
                self.name = js.get_name()
                self.runner.log(f"Controller conectat: {self.name}")
                if self.arm is None:
                    self.arm, _ = get_arm()
                self._loop(pygame, js)
                self.runner.log("Controller deconectat.")
            except Exception:   # noqa: BLE001 - controllerul nu are voie să oprească pagina
                traceback.print_exc()
                time.sleep(1.0)

    @staticmethod
    def _want(js):
        """Viteza dorită (-1..1) pentru servo1..5, din stickuri și butoane."""
        want = [0.0] * 5
        for axis, (i, sign) in MAP["axes"].items():
            v = js.get_axis(axis) if axis < js.get_numaxes() else 0.0
            if abs(v) > DEADZONE:
                want[i] += sign * v
        for i, (minus, plus) in MAP["buttons"].items():
            want[i] += js.get_button(plus) - js.get_button(minus)
        return want

    @staticmethod
    def _held(js):
        """Ține cineva apăsat ceva (buton, stick, D-pad)?"""
        return (any(js.get_button(b) for b in range(js.get_numbuttons()))
                or any(abs(js.get_axis(a)) > DEADZONE for a in range(js.get_numaxes()))
                or any(js.get_hat(h) != (0, 0) for h in range(js.get_numhats())))

    def _sync(self):
        """Unghiurile de unde preluăm brațul. Un servo necitit merge încet la HOME, nu sare."""
        home = list(load_positions().get("HOME", [90, 90, 90, 90, 90]))
        angles, missing = [], False
        for i in range(5):
            a = read_servo(self.arm, i + 1)
            missing |= a is None
            angles.append(float(a if a is not None else home[i]))
        grip = read_servo(self.arm, 6)
        grip = grip if grip is not None else C.GRIP_OPEN
        if missing:
            self.arm.Arm_serial_servo_write6(*[round(a) for a in angles], round(grip), START_MS)
            time.sleep(START_MS * C.TIME_UNIT_S)
        return angles, float(grip)

    def _loop(self, pygame, js):
        angles = None       # None = nu controlăm noi brațul acum
        hold = False        # după o oprire de urgență: așteptăm să lase controllerul liber
        vel, speed, sent, grip, grip_target = [0.0] * 5, 1, None, 0.0, 0.0
        while True:
            time.sleep(1 / HZ)
            pressed, hat = [], 0
            for ev in pygame.event.get():
                if ev.type == pygame.JOYDEVICEREMOVED:
                    return
                if ev.type == pygame.JOYBUTTONDOWN:
                    pressed.append(ev.button)
                elif ev.type == pygame.JOYHATMOTION:
                    hat = ev.value[1]
            held = self._held(js)

            if self.runner.busy():
                angles = None
                self.status = "automat (orice buton = STOP)"
                if hold:        # încă ține apăsat Select, cu care tocmai a pornit căutarea
                    hold = held
                    continue
                if held:
                    self.stops += 1
                    self.status = "OPRIT din controller"
                    self.runner.log("OPRIRE DE URGENȚĂ din controller.")
                    self.runner.stop()
                    self.runner.wait_stopped(5)
                    hold = True
                continue
            if hold:
                if not held:
                    hold = False
                    self.status = "manual"
                continue
            if angles is None:
                self.status = "manual"
                if not held:
                    continue    # nu trimitem nimic brațului până nu atinge cineva controllerul
                angles, grip = self._sync()
                grip_target, vel = grip, [0.0] * 5
                sent = [round(a, 1) for a in angles] + [round(grip)]

            for b in pressed:
                if b == MAP["open"]:
                    grip_target = C.GRIP_OPEN
                elif b == MAP["close"]:
                    grip_target = C.GRIP_CLOSE
                elif b == MAP["save"]:
                    try:
                        self.runner.start("aim")
                    except ValueError:
                        pass
                    hold = True     # butonul încă e apăsat: să nu oprească imediat căutarea pornită
                elif b == MAP["home"]:
                    home = load_positions().get("HOME")
                    if home:
                        angles = [float(a) for a in home]
                        self.arm.Arm_serial_servo_write6(*angles, round(grip), 1000)
                        time.sleep(1000 * C.TIME_UNIT_S)
                        vel, sent = [0.0] * 5, None
            if hold:
                continue
            if hat:
                speed = max(0, min(len(SPEEDS) - 1, speed + hat))

            # Viteza reală urmează treptat viteza dorită, ca brațul să accelereze și să frâneze lin.
            want = self._want(js)
            ramp = 1 / (RAMP_S * HZ)
            for i, (lo, hi) in enumerate(LIMITS):
                w = max(-1.0, min(1.0, want[i]))
                vel[i] += max(-ramp, min(ramp, w - vel[i]))
                angles[i] += vel[i] * SPEEDS[speed] / HZ
                if not lo <= angles[i] <= hi:
                    angles[i] = max(lo, min(hi, angles[i]))
                    vel[i] = 0.0
            grip += max(-GRIP_SPEED / HZ, min(GRIP_SPEED / HZ, grip_target - grip))
            cmd = [round(a, 1) for a in angles] + [round(grip)]
            if cmd != sent:
                self.arm.Arm_serial_servo_write6(*cmd, int(1000 / HZ * 2))
                sent = cmd
