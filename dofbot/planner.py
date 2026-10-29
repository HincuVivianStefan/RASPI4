"""Planificarea mutărilor, fără braț și fără cameră (se poate testa oriunde).

O mutare = (din_celulă, în_celulă, nivel_destinație, nivel_sursă), ex. ("S12", "D12", 1, 1).
Nivelul 1 = pe masă.
"""


def cell_name(grid, r, c):
    return f"{grid}{r + 1}{c + 1}"


def plan_copy(src, row_order=(1, 2, 3)):
    """Sarcina 1: fiecare cub din S ajunge în aceeași celulă din D.
    row_order = ordinea rândurilor (1..3); întâi cel mai îndepărtat de braț, ca să nu lovești cuburile puse."""
    moves = []
    for r in [x - 1 for x in row_order]:
        for c in range(3):
            if src[r][c]:
                moves.append((cell_name("S", r, c), cell_name("D", r, c), 1, 1))
    return moves


def plan_tower(cells, target, order):
    """Sarcina 2: turn în `target` cu culorile `order` (de jos în sus).
    cells = {"S11": culoare/None, ...}; se modifică pe loc. Ridică ValueError dacă nu se poate."""
    if target not in cells:
        raise ValueError(f"Celula {target} nu există")
    for col in set(order):
        need = order.count(col)
        have = sum(1 for v in cells.values() if v == col)
        if have < need:
            raise ValueError(f"Lipsesc cuburi {col}: am {have}, trebuie {need}")

    moves = []
    level = 1
    parked, parked_color = None, None   # cubul de pe țintă, pus temporar peste alt cub (nivel 2)
    # Celula țintă e ocupată?
    if cells[target]:
        blocker = cells[target]
        if blocker == order[0]:
            level = 2
        else:
            free = next((k for k, v in cells.items() if v is None and k != target), None)
            if free is not None:
                moves.append((target, free, 1, 1))
                cells[free] = blocker
            else:
                # Caroiaj plin: îl punem peste un cub de care turnul nu are nevoie.
                def spare(col):
                    return sum(1 for v in cells.values() if v == col) - order.count(col)
                parked = next((k for k, v in cells.items() if k != target and spare(v) > 0), None)
                if parked is None:
                    raise ValueError("Celula țintă e ocupată, caroiajul e plin și toate cuburile sunt "
                                     "necesare turnului: nu am unde muta cubul")
                moves.append((target, parked, 2, 1))
                parked_color = blocker
            cells[target] = None

    for i in range(level - 1, len(order)):
        if parked_color == order[i]:
            moves.append((parked, target, i + 1, 2))
            parked_color = None
            continue
        src = next(k for k, v in cells.items() if v == order[i] and k not in (target, parked))
        moves.append((src, target, i + 1, 1))
        cells[src] = None
    return moves


def grid_to_cells(grid, name="S"):
    return {cell_name(name, r, c): grid[r][c] for r in range(3) for c in range(3)}


def compare(a, b):
    """Celulele (rând, coloană), numerotate de la 1, unde două matrice 3x3 diferă."""
    return [(r + 1, c + 1) for r in range(3) for c in range(3) if a[r][c] != b[r][c]]
