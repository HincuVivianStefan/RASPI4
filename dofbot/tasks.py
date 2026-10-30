"""Logica celor două sarcini, folosită și din linia de comandă (task1.py, task2.py) și din web.py.

`ctx` face legătura cu cine rulează sarcina:
  ctx.log(text)        mesaj în terminal / jurnalul paginii
  ctx.pause(text)      așteaptă Enter / butonul "Continuă"
  ctx.publish(k, v)    rezultate intermediare (afișate în pagină)
"""
import os
import time

from . import camera
from . import config as C
from . import planner
from . import vision


class ConsoleContext:
    def log(self, msg):
        print(msg)

    def pause(self, msg):
        input(f"{msg}  [Enter] ")

    def publish(self, key, value):
        pass


def grid_text(g):
    return "\n".join(f"R{r + 1}: " + " | ".join(f"{(x or '-'):>8}" for x in row) for r, row in enumerate(g))


def look_and_detect(rob, ctx, grid, debug):
    aim = C.AUTO_AIM and hasattr(ctx, "aim")
    if f"LOOK_{grid}" in rob.pos or not aim:
        rob.goto(f"LOOK_{grid}")
    else:
        ctx.log(f"LOOK_{grid} nu e calibrată: caut caroiajul de unde e brațul (verifică să fie {grid}!)")
        rob.on_look(grid)
    if aim and ctx.aim(rob) is None:
        ctx.log("Caroiajul nu a fost centrat automat; încerc detecția de aici.")
    rob.check()
    # Din pagina web: aceeași detecție pe care o vezi (nu încă una în paralel, care ar încărca Pi-ul).
    if hasattr(ctx, "detect"):
        g = ctx.detect(grid)
    else:
        os.makedirs(C.CAPTURES_DIR, exist_ok=True)
        g = vision.detect_stable(grid, debug=os.path.join(C.CAPTURES_DIR, debug))
    ctx.log(f"Caroiaj {grid} ({C.STABLE_FRAMES} cadre, vot majoritar):\n{grid_text(g)}")
    faces = ctx.faces() if hasattr(ctx, "faces") else None
    turned = [f"{grid}{r + 1}{c + 1} {g[r][c]}: culoarea {vision.FACES.get(f, f)}, sus e alb"
              for r, row in enumerate(faces or []) for c, f in enumerate(row) if f and f != "sus" and g[r][c]]
    if turned:
        ctx.log("Cuburi întoarse (orientarea se păstrează la mutare):\n" + "\n".join(turned))
    return g


def run_task1(rob, ctx):
    t0 = time.time()
    src = look_and_detect(rob, ctx, "S", "s1_sursa.jpg")
    ctx.publish("source", src)
    if not any(any(row) for row in src):
        ctx.log("Niciun cub detectat pe sursă: verifică lumina, colțurile și intervalele HSV.")
    if C.PAUSE_BETWEEN_PHASES:
        ctx.pause("Aranjarea e memorată. Continuă pentru mutare.")

    moves = planner.plan_copy(src, C.S1_ROW_ORDER)
    ctx.publish("plan", moves)
    ctx.log(f"Plan: {len(moves)} mutări")
    for i, (a, b, _, _) in enumerate(moves, 1):
        ctx.log(f"[{i}/{len(moves)}] {src[int(a[1]) - 1][int(a[2]) - 1]}: {a} -> {b}")
        rob.pick(a)
        rob.place(b)

    synth_cam = camera.shared.source.startswith("synth") if camera.shared else C.TEST_IMAGE == "synth"
    if rob.sim and not synth_cam:
        # Brațul simulat nu a mutat nimic în realitate: aranjează tu destinația ca să vezi verificarea.
        ctx.pause("[SIM] Aranjează caroiajul destinație în fața camerei, apoi continuă.")
    dst = look_and_detect(rob, ctx, "D", "s1_dest.jpg")
    ctx.publish("dest", dst)
    errors = planner.compare(src, dst)
    ctx.log("Verificare OK: destinația e identică cu sursa." if not errors
            else f"Diferențe la (rând, coloană): {errors}")
    rob.home()
    dt = time.time() - t0
    ctx.log(f"Sarcina 1 gata în {dt:.1f} s")
    return {"source": src, "dest": dst, "errors": errors, "seconds": round(dt, 1)}


def run_task2(rob, ctx, target=None, order=None):
    target = "S" + (target or C.S2_TARGET)
    order = list(order or C.S2_ORDER)
    unknown = [c for c in order if c not in C.COLORS]
    if unknown:
        raise ValueError(f"Culori necunoscute: {unknown}. Valide: {list(C.COLORS)}")

    t0 = time.time()
    ctx.log(f"Turn în {target}, de jos în sus: {' -> '.join(order)}")
    learned = rob.sim or all({f"{k}_jos", f"{k}_sus"} <= set(rob.pos) for k in planner.grid_to_cells([[None] * 3] * 3))
    if not learned and hasattr(ctx, "step1"):
        return run_task2_camera(rob, ctx, target, order, t0)
    grid = look_and_detect(rob, ctx, "S", "s2_sursa.jpg")
    ctx.publish("source", grid)
    moves = planner.plan_tower(planner.grid_to_cells(grid), target, order)
    ctx.publish("plan", moves)
    if C.PAUSE_BETWEEN_PHASES:
        ctx.pause(f"Plan gata ({len(moves)} mutări). Continuă pentru construcție.")

    for src, dst, level, src_level in moves:
        if dst == target:
            ctx.log(f"Nivel {level}: din {src}" + (f" (nivel {src_level})" if src_level > 1 else "") + f" în {dst}")
        else:
            ctx.log(f"Eliberez {src} -> {dst}" + (f" (peste cubul de acolo, nivel {level})" if level > 1 else ""))
        rob.pick(src, level=src_level)
        rob.place(dst, level=level)

    rob.home()
    dt = time.time() - t0
    ctx.log(f"Turn gata în {dt:.1f} s")
    return {"moves": moves, "seconds": round(dt, 1)}


def run_task2_camera(rob, ctx, target, order, t0):
    """Sarcina 2 fără poziții învățate (din pagină): găsește caroiajul și culorile (ca la Sarcina 1, ctx.step1),
    planifică turnul (planner.plan_tower) și îl construiește cu cinematica și camera (fetch.build_tower): fiecare cub
    privit de aproape înainte de prindere, fiecare nivel țintit pe cubul de jos măsurat și verificat după punere."""
    from . import fetch
    first = ctx.step1(rob)
    if ctx.grid_area is None:
        raise RuntimeError("Nu știu unde e caroiajul pe masă: nu pot construi turnul")
    cells = planner.grid_to_cells(first["colors"])
    moves = planner.plan_tower(dict(cells), target, order)
    ctx.publish("plan", moves)
    ctx.log(f"Plan: {len(moves)} mutări")
    if C.PAUSE_BETWEEN_PHASES:
        ctx.pause(f"Plan gata ({len(moves)} mutări). Continuă pentru construcție.")
    fetcher = fetch.Fetcher(rob, ctx.frame, ctx.log, ctx.grid_area)
    fetcher.look = first["angles"]
    pipeline = getattr(ctx, "pipeline", None)
    if pipeline is not None:
        pipeline.grid_off = True        # cât ia / pune cuburi, caroiajul nu se mai caută pe imagine
    try:
        report = fetch.build_tower(fetcher, ctx, cells, moves, target)
    finally:
        if pipeline is not None:
            pipeline.grid_off = False
    rob.move(first["angles"], 1500)
    dt = time.time() - t0
    ctx.log(f"Turn gata în {dt:.1f} s")
    return dict(report, seconds=round(dt, 1))
