# tools/ – unelte din linia de comandă

Se rulează din rădăcina proiectului.

## check_camera.py – nu merge camera?

```bash
py tools/check_camera.py        # Windows (cu pagina web oprită)
python3 tools/check_camera.py   # Pi / Mac
```

Încearcă camerele 0–3 prin fiecare backend (pe Windows: DirectShow, Media Foundation, implicit) și spune
care dă imagine. Dacă niciuna nu merge, afișează ce trebuie verificat (permisiuni, altă aplicație, capac).

## calibrate_arm.py – pozițiile brațului (obligatoriu pe Pi)

```bash
python3 tools/calibrate_arm.py
```

La pornire duce brațul în poziția sigură (ca butonul din pagină). Apoi muți brațul din tastatură și salvezi fiecare poziție în `data/positions.json`:

| Taste | Ce mișcă |
|---|---|
| q / a | servo 1 (bază) |
| w / s | servo 2 (umăr) |
| e / d | servo 3 (cot) |
| r / f | servo 4 (încheietură) |
| t / g | servo 5 (rotația gripperului) |
| o / c | deschide / închide gripperul |
| 1…9 | pasul, în grade |
| p | salvează poziția curentă (îți cere numele) |
| l | mergi la o poziție salvată |
| n | ce poziții mai lipsesc |
| x | ieșire |

Pozițiile necesare (39 în total; pagina arată câte mai lipsesc):

| Nume | Ce e |
|---|---|
| `HOME` | poziția de repaus |
| `LOOK_S`, `LOOK_D` | de unde camera vede caroiajul S, respectiv D |
| `S11_jos` … `S33_jos`, `D11_jos` … `D33_jos` | gripperul jos, la cubul de pe masă, în fiecare celulă |
| `S11_sus` … `D33_sus` | deasupra celulei, la înălțimea a 6 cuburi |

Nivelurile turnului se calculează între `_jos` și `_sus`. Dacă la un nivel brațul nu nimerește, învață poziția exactă, de exemplu `S22_n3`.

## center_grid.py – brațul își centrează caroiajul

```bash
python3 tools/center_grid.py               # centrează (cu pagina pornită)
python3 tools/center_grid.py --save LOOK_S # și salvează poziția
```

Aceeași logică ca butonul „Caută caroiajul” din pagină (`dofbot/aim.py`). Util ca să înveți `LOOK_S` / `LOOK_D`.

## gamepad.py – brațul din controller, fără pagină

Cu pagina pornită nu e nevoie: pagina citește singură controllerul. Nu le porniți pe amândouă.
Butoanele se reglează în `dofbot/pad.py` (`MAP`); `python3 tools/gamepad.py --test` arată numerele lor.

## task1.py, task2.py – sarcinile fără pagină

```bash
python3 tools/task1.py
python3 tools/task2.py 22 rosu galben verde albastru rosu   # celula, apoi culorile de JOS în SUS
python3 tools/task2.py                                     # celula și ordinea din setări
```

Folosesc aceleași setări (`data/settings.json`) și aceeași logică ca pagina. Pozele de depanare ajung în `captures/`.
