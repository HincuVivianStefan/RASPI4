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
| `camera.py` | citește continuu camera, poza sau scena simulată; la eroare reîncearcă singur; aplică expunerea / balansul de alb din setări |
| `vision.py` | lumina (foaia adusă la alb), caroiajul (întreg, plin de cuburi sau doar parțial), cuburile ca obiecte, celulele, votul |
| `overlay.py` | desenele peste imagine |
| `planner.py` | mutările pentru S1 și S2 (fără braț, testat) |
| `tasks.py` | pașii sarcinilor (privește → detectează → planifică → mută → verifică) |
| `robot.py` | mișcarea lină (puncte trimise la `MOTION_HZ`, profil minimum-jerk, limite, oprire în ≤ 25 ms), pick / place / goto peste pozițiile învățate; lista `REQUIRED` |
| `arm_driver.py` | brațul real (`Arm_Lib`) sau unul simulat; un singur acces la I2C odată (`I2C_LOCK`); ține minte ultimele unghiuri trimise (pentru desenul din pagină) |
| `aim.py` | brațul își caută singur caroiajul și se așază să-l vadă bine, și pornind de la un singur colț văzut; efectul articulațiilor se ține minte în `data/aim.json` |
| `kinematics.py` | unde ajung încheietura și camera din unghiurile servourilor (lungimile din URDF-ul Yahboom); folosit de brațul simulat |
| `pad.py` | controllerul USB în pagină: control manual; în timpul automatizării, orice buton = oprire de urgență |
| `hat.py` | placa de răcire: coolerul și ecranul (IP, temperatură, procesor), pornite odată cu pagina |
| `sim_world.py`, `synth.py` | lumea simulată (ce cub e unde) și desenarea ei ca imagine de cameră: o cameră fixă („Scenă simulată”) sau camera de pe brațul simulat („Braț simulat”), cu lumină slabă / colorată la cerere |

## Detecția, pe scurt

0. **Lumina** (`LIGHT_NORM`): foaia albă din cadru (cei mai luminoși pixeli aproape gri) e adusă la alb, pe fiecare canal, deci și balansul de alb. Intervalele HSV înseamnă astfel „față de hârtie”: un verde închis în lumină slabă rămâne verde. Câștigul e cel mult `LIGHT_MAX_GAIN`; peste ×1,8 se adaugă un filtru median, altfel zgomotul amplificat strică și culorile, și liniile negre. Sonda HSV și „Învață” lucrează pe aceeași imagine normalizată.
1. **Caroiajul**: pixelii întunecați și nesaturați, apoi cel mai bun patrulater care arată ca o grilă 3×3, pe imaginea micșorată (480 px).
   - **Urmărit**: colțurile ținute minte sunt verificate la fiecare cadru (`verify_grid`, ~15 ms) și reajustate la fiecare `GRID_EVERY` cadre. Dacă verificarea pică, sunt căutate din nou; după `LOST_AFTER` cadre (sau `LOST_S` secunde) fără caroiaj, e **pierdut** și nu mai e desenat.
   - Colțurile salvate (`vision.json`) sunt doar un punct de plecare pentru căutare: se folosesc numai dacă se potrivesc cu imaginea. Excepție: cu „caroiaj automat” oprit, se folosesc ele, cât timp camera nu s-a mișcat.
   - **Plin de cuburi** (liniile aproape toate ascunse): din rețeaua 3×3 a fețelor de sus ale cuburilor și focala camerei se află poza camerei; coborând planul cu latura cubului se ajunge la caroiajul de pe masă, ajustat apoi pe liniile rămase vizibile (`grid_from_cubes`).
   - **Parțial** (doar un colț în imagine): caroiajul e extrapolat și în afara imaginii (`find_partial_grid`), desenat punctat, cu o săgeată spre centrul lui. Culorile nu se citesc din el; servește centrării (`aim.py`). Ca ultimă rezervă, centrul liniilor negre văzute (`lines_hint`).
2. **Cuburile**: pentru fiecare culoare (intervale HSV), petele de mărimea unui cub (`CUBE_MIN_AREA`…`CUBE_MAX_AREA` × celula), aproape pătrate (`CUBE_MAX_ASPECT`) și pline (`CUBE_MIN_FILL`). Petele mai mici sunt zgomot și se ignoră.
3. **Cubul întreg (3D)**: cuburile au colorată doar fața de sus, care stă la înălțimea cubului, deci văzută oblic cade *deasupra* celulei reale (uneori chiar în afara caroiajului). Din colțurile caroiajului se calculează poza camerei, din mărimea feței de sus înălțimea cubului, iar fața coborâtă pe masă dă **baza** cubului. Forma (`CUBE_MAX_ASPECT`, `CUBE_MIN_FILL`) se judecă pe planul mesei, unde fața de sus e pătrată. În imagine se desenează tot cubul: fața de sus, baza și muchiile. Setări: `CUBE_3D` (pornit), `CUBE_TOP` (cât din latură e colorat), `CAM_FOCAL` (0 = învățată singură din caroiajul văzut curat), `CUBE_SIDE` (setată, recunoaște și turnurile).
   Caroiajul acoperit de cuburi se potrivește doar pe liniile rămase vizibile; când e aproape plin, se păstrează colțurile anterioare (sau cele salvate) dacă liniile vizibile se potrivesc încă.
   Găsit **de la zero** (fără colțuri anterioare), caroiajul S e acceptat doar dacă are cuburi în celule (`GRID_NEED_CUBES`): altfel poate fi o grilă de pe jos. D poate fi gol, deci nu se cere.
   Focala se învață doar din caroiajele acceptate, nu din candidații încercați în timpul căutării.
   Când se mișcă camera (e pe braț), colțurile se mută odată cu imaginea (deplasarea măsurată între cadre, `MOTION_DIFF`), apoi se ajustează pe liniile vizibile.
4. **Celulele**: fiecare cub intră în celula în care îi stă baza.
5. **Stabil**: votul majoritar pe ultimele `STABLE_FRAMES` detecții.

Cu `CUBE_MODE = False` (bifa „cuburi ca obiecte” din Setări) se revine la metoda veche: procentul de pixeli colorați din centrul celulei.

## Viteză (Raspberry Pi 4)

Se reglează în `config.py`:
- `STREAM_FPS`: fluxul video, implicit 10;
- `DETECT_FPS`: detecția, implicit 5;
- `GRID_EVERY`: din câte în câte cadre se reajustează colțurile caroiajului (verificat e la fiecare cadru);
- `CUBE_EVERY`: din câte în câte cadre se detectează cuburile (în timpul sarcinilor: la fiecare);
- `CV_THREADS`: câte nuclee folosește OpenCV (2 din 4: restul rămân liberi, chiar și când caroiajul e căutat la fiecare cadru);
- `JPEG_QUALITY`: calitatea imaginilor trimise paginii.

Fără pagină deschisă și fără sarcină, serverul nu procesează nimic. Vederile de depanare (caroiajul văzut de sus, măștile) se calculează doar cât sunt deschise în pagină.

Durate măsurate pe Pi 4 (scenă simulată, 640×480): verificarea caroiajului ~15 ms, regăsirea lui din colțurile anterioare ~130 ms, căutarea de la zero 50–300 ms, cuburile 3D ~50 ms, caroiajul parțial 0,3–0,6 s (doar cât e pierdut: la fiecare cadru în timpul centrării, altfel la fiecare `PARTIAL_EVERY` cadre).

## Mișcarea brațului

`Robot.move` nu trimite o singură comandă: împarte mișcarea în puncte trimise la `MOTION_HZ` (ca `pad.py`), cu profil minimum-jerk (pornește și oprește fără smucitură; toate articulațiile ajung odată). Durata e cea cerută, dar nu sub ce permit `V_MAX` / `A_MAX`. Unghiurile se trimit cu zecimale și sunt verificate față de `LIMITS` (Arm_Lib ignoră în tăcere peste 180° și nu verifică sub 0°). La final, brațul e citit până ajunge la `SETTLE_TOL` de țintă (cel mult `SETTLE_MAX_S`). Stop / controllerul îl opresc la următorul punct.
