"""Alege brațul real (Arm_Lib) sau un braț simulat, ca să poți testa pe laptop."""
import threading
import time

from . import config as C


STREAM_MS = 100   # comenzile mai scurte de atât sunt puncte dintr-o mișcare trimisă în flux (nu le afișăm)


class FakeArm:
    """Imită Arm_Device: ține unghiurile în memorie și afișează comenzile (fără punctele din flux).
    history = toate comenzile write6, (momentul, unghiurile), pentru teste."""

    def __init__(self):
        self.angles = list(C.SIM_START) + [C.GRIP_OPEN]
        self.history = []

    def Arm_serial_servo_write6(self, s1, s2, s3, s4, s5, s6, t):
        self.angles = [s1, s2, s3, s4, s5, s6]
        self.history.append((time.time(), list(self.angles)))
        del self.history[:-2000]
        if C.SIM_VERBOSE and t > STREAM_MS:
            print(f"  [SIM] servo1-6 -> {self.angles}  t={t}")

    def Arm_serial_servo_write(self, id, angle, t):
        self.angles[id - 1] = angle
        if C.SIM_VERBOSE and t > STREAM_MS:
            print(f"  [SIM] servo{id} -> {angle}  t={t}")

    def Arm_serial_servo_read(self, id):
        return self.angles[id - 1]


# Ultimele unghiuri trimise brațului (servo1..6), de oricine din acest proces: pagina le desenează
# fără să mai citească servourile pe I2C. None până la prima comandă sau citire.
LAST = {"angles": None, "t": 0.0}

# Un singur acces la I2C odată, din tot procesul (sarcina, controllerul, citirile): write6 sunt două
# tranzacții (timpul, apoi unghiurile) și nu au voie să se amestece cu ale altcuiva.
I2C_LOCK = threading.RLock()


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
        with I2C_LOCK:
            self.arm.Arm_serial_servo_write6(s1, s2, s3, s4, s5, s6, int(t))
        for i, a in enumerate((s1, s2, s3, s4, s5, s6)):
            self._note(i, a)

    def Arm_serial_servo_write(self, id, angle, t):
        with I2C_LOCK:
            self.arm.Arm_serial_servo_write(id, angle, int(t))
        self._note(id - 1, angle)

    def Arm_serial_servo_read(self, id):
        with I2C_LOCK:
            a = self.arm.Arm_serial_servo_read(id)
        if a is not None:
            self._note(id - 1, a)
        return a

    def __getattr__(self, name):
        attr = getattr(self.arm, name)
        if not callable(attr):
            return attr

        def locked(*args, **kwargs):     # restul funcțiilor Arm_Lib (torque, buzzer...) tot prin lacăt
            with I2C_LOCK:
                return attr(*args, **kwargs)
        return locked


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
