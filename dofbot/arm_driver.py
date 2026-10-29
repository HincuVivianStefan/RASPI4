"""Alege brațul real (Arm_Lib) sau un braț simulat, ca să poți testa pe laptop."""
import time

from . import config as C


class FakeArm:
    """Imită Arm_Device: ține unghiurile în memorie și afișează comenzile."""

    def __init__(self):
        self.angles = [90, 90, 90, 90, 90, C.GRIP_OPEN]

    def Arm_serial_servo_write6(self, s1, s2, s3, s4, s5, s6, t):
        self.angles = [s1, s2, s3, s4, s5, s6]
        if C.SIM_VERBOSE:
            print(f"  [SIM] servo1-6 -> {self.angles}  t={t}")

    def Arm_serial_servo_write(self, id, angle, t):
        self.angles[id - 1] = angle
        if C.SIM_VERBOSE:
            print(f"  [SIM] servo{id} -> {angle}  t={t}")

    def Arm_serial_servo_read(self, id):
        return self.angles[id - 1]


# Ultimele unghiuri trimise brațului (servo1..6), de oricine din acest proces: pagina le desenează
# fără să mai citească servourile pe I2C. None până la prima comandă sau citire.
LAST = {"angles": None, "t": 0.0}


class Tracked:
    """Învelește brațul (real sau simulat) și ține minte în LAST ce i s-a trimis."""

    def __init__(self, arm):
        self.arm = arm

    def _note(self, i, angle):
        if LAST["angles"] is None:
            LAST["angles"] = [90, 90, 90, 90, 90, C.GRIP_OPEN]
        LAST["angles"][i] = round(float(angle), 1)
        LAST["t"] = time.time()

    def Arm_serial_servo_write6(self, s1, s2, s3, s4, s5, s6, t):
        self.arm.Arm_serial_servo_write6(s1, s2, s3, s4, s5, s6, t)
        for i, a in enumerate((s1, s2, s3, s4, s5, s6)):
            self._note(i, a)

    def Arm_serial_servo_write(self, id, angle, t):
        self.arm.Arm_serial_servo_write(id, angle, t)
        self._note(id - 1, angle)

    def Arm_serial_servo_read(self, id):
        a = self.arm.Arm_serial_servo_read(id)
        if a is not None:
            self._note(id - 1, a)
        return a

    def __getattr__(self, name):
        return getattr(self.arm, name)


def get_arm():
    """Întoarce (arm, is_sim)."""
    if C.SIM_ENV == "1":
        print("[SIM] Braț simulat (DOFBOT_SIM=1)")
        return Tracked(FakeArm()), True
    try:
        from Arm_Lib import Arm_Device
    except ImportError:
        if C.SIM_ENV == "0":
            raise
        print("[SIM] Arm_Lib nu e instalat -> braț simulat")
        return Tracked(FakeArm()), True
    return Tracked(Arm_Device()), False
