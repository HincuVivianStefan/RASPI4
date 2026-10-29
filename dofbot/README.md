# dofbot/ – codul

Pachet Python. Se pornește din rădăcina proiectului: `python3 -m dofbot.web` (sau `start.bat` / `start.sh`).

## Cum circulă datele

```
camera.py ──cadre──► web.py: Pipeline ──► fluxul /stream/cam (15 cadre/s, cu ultima detecție desenată peste)
                          │
                          └─► detecția (8/s, separat): vision.find_grid → vision.detect_full → vot pe cadre
                                                                              │
pagina (static/index.html) ◄── /api/state ◄───────────────────────────────────┘
        │
        └── butoane ──► /api/<acțiune> ──► App.a_<acțiune>  (setări, colțuri, rulare, reset...)
                                              │
                                              └─► Runner ──► tasks.run_task1 / run_task2 ──► robot.py ──► brațul
                                                               (planner.py calculează mutările)
```

Fluxul video nu așteaptă detecția: fiecare cadru al camerei e trimis cu desenul **ultimei** detecții.
Sarcinile pornite din pagină folosesc aceeași detecție (nu rulează una a doua în paralel).

## Fișiere

| Fișier | Rol |
|---|---|
| `config.py` | toate valorile de reglat. Cele din `TUNABLE` se schimbă din pagină și se salvează în `data/settings.json` |
| `web.py` | serverul web (doar biblioteca standard): fluxuri MJPEG, `/api/state`, acțiuni, rularea sarcinilor |
| `static/index.html` | pagina (HTML + JS simplu, fără biblioteci) |
| `camera.py` | citește continuu camera, poza sau scena simulată; la eroare reîncearcă singur |
| `vision.py` | caroiajul (linii negre pe alb), cuburile ca obiecte, celulele, votul |
| `overlay.py` | desenele peste imagine |
| `planner.py` | mutările pentru S1 și S2 (fără braț, testat) |
| `tasks.py` | pașii sarcinilor (privește → detectează → planifică → mută → verifică) |
| `robot.py` | pick / place / goto peste pozițiile învățate; oprire la cerere; lista `REQUIRED` |
| `arm_driver.py` | brațul real (`Arm_Lib`) sau unul simulat; ține minte ultimele unghiuri trimise (pentru desenul din pagină) |
| `aim.py` | brațul își caută singur caroiajul și se așază să-l vadă bine (butonul „Caută caroiajul”, sarcinile cu `AUTO_AIM`) |
| `pad.py` | controllerul USB în pagină: control manual; în timpul automatizării, orice buton = oprire de urgență |
| `hat.py` | placa de răcire: coolerul și ecranul (IP, temperatură, procesor), pornite odată cu pagina |
| `sim_world.py`, `synth.py` | lumea simulată (ce cub e unde) și desenarea ei ca imagine de cameră |

## Detecția, pe scurt

1. **Caroiajul**: pixelii întunecați și nesaturați, apoi cel mai bun patrulater care arată ca o grilă 3×3. Se caută pe imaginea micșorată (320 px) și doar la fiecare `GRID_EVERY` cadre, pentru că stă fix pe masă. Dacă nu e găsit, se folosesc colțurile salvate.
2. **Cuburile**: pentru fiecare culoare (intervale HSV), petele de mărimea unui cub (`CUBE_MIN_AREA`…`CUBE_MAX_AREA` × celula), aproape pătrate (`CUBE_MAX_ASPECT`) și pline (`CUBE_MIN_FILL`). Petele mai mici sunt zgomot și se ignoră.
3. **Cubul întreg (3D)**: cuburile au colorată doar fața de sus, care stă la înălțimea cubului, deci văzută oblic cade *deasupra* celulei reale (uneori chiar în afara caroiajului). Din colțurile caroiajului se calculează poza camerei, din mărimea feței de sus înălțimea cubului, iar fața coborâtă pe masă dă **baza** cubului. Forma (`CUBE_MAX_ASPECT`, `CUBE_MIN_FILL`) se judecă pe planul mesei, unde fața de sus e pătrată. În imagine se desenează tot cubul: fața de sus, baza și muchiile. Setări: `CUBE_3D` (pornit), `CUBE_TOP` (cât din latură e colorat), `CAM_FOCAL` (0 = învățată singură din caroiajul văzut curat), `CUBE_SIDE` (setată, recunoaște și turnurile).
   Caroiajul acoperit de cuburi se potrivește doar pe liniile rămase vizibile; când e aproape plin, se păstrează colțurile anterioare (sau cele salvate) dacă liniile vizibile se potrivesc încă.
   Găsit **de la zero** (fără colțuri anterioare), caroiajul S e acceptat doar dacă are cuburi în celule (`GRID_NEED_CUBES`): altfel poate fi o grilă de pe jos. D poate fi gol, deci nu se cere.
   Când se mișcă camera (e pe braț), colțurile se mută odată cu imaginea (deplasarea măsurată între cadre, `MOTION_DIFF`), apoi se ajustează pe liniile vizibile.
4. **Celulele**: fiecare cub intră în celula în care îi stă baza.
5. **Stabil**: votul majoritar pe ultimele `STABLE_FRAMES` detecții.

Cu `CUBE_MODE = False` (bifa „cuburi ca obiecte” din Setări) se revine la metoda veche: procentul de pixeli colorați din centrul celulei.

## Viteză (Raspberry Pi 4)

Se reglează în `config.py`:
- `STREAM_FPS`: fluxul video, implicit 10;
- `DETECT_FPS`: detecția, implicit 5;
- `GRID_EVERY`: din câte în câte cadre se recaută caroiajul (1 = la fiecare, ca să urmărească camera de pe braț);
- `CUBE_EVERY`: din câte în câte cadre se detectează cuburile (în timpul sarcinilor: la fiecare);
- `CV_THREADS`: câte nuclee folosește OpenCV (2 din 4: restul rămân liberi, chiar și când caroiajul e căutat la fiecare cadru);
- `JPEG_QUALITY`: calitatea imaginilor trimise paginii.

Fără pagină deschisă și fără sarcină, serverul nu procesează nimic. Vederile de depanare (caroiajul văzut de sus, măștile) se calculează doar cât sunt deschise în pagină.
