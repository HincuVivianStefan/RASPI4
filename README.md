# DOFBOT – CRI 2026

Braț robotic DOFBOT cu Raspberry Pi 4 care rezolvă cele două sarcini ale concursului:

1. **Reproducerea unei aranjări**: recunoaște cuburile de pe caroiajul 3×3 sursă (S) și le pune la fel pe al doilea caroiaj (D).
2. **Turnul sortat**: construiește un turn în ordinea de culori dată, în celula aleasă de juriu.

Tema completă, rezumată: [docs/tema.md](docs/tema.md).

Totul se controlează dintr-o **pagină web**, care arată fluxul camerei cu detecția desenată peste el, rezultatul pe caroiaj, setările, rularea sarcinilor și jurnalul.

## Pornire rapidă

| Unde | Cum | Pagina |
|---|---|---|
| Laptop (Windows) | dublu-click pe `start.bat` (prima dată instalează singur opencv și numpy) | se deschide singură: http://localhost:8000 |
| Raspberry Pi | `./start.sh` | de pe laptop/telefon: `http://<ip-ul-pi>:8000` (ip: `hostname -I`) |
| Pi, automat la boot | vezi [deploy/README.md](deploy/README.md) | la fel |

Pe Windows, dacă ceva nu merge (de exemplu camera): închide `start.bat` și dă dublu-click pe `verificare.bat`. Rulează testele și verificarea camerelor și scrie totul în `raport_windows.txt`.

Pe laptop, brațul e **simulat** automat, pentru că acolo nu există `Arm_Lib`. Pe Pi se folosește brațul real.
Sursa imaginii se alege din pagină: **Camera 0–3** sau **Scenă simulată**, adică un caroiaj desenat pe care brațul simulat chiar mută cuburile. Așa poți testa ambele sarcini cap-coadă fără braț și fără cameră.

Prima instalare:

```bash
# laptop
py -m pip install -r requirements.txt
# Raspberry Pi
sudo apt install -y python3-opencv python3-numpy python3-smbus i2c-tools
```

## Ce faci în pagină

- **Fluxul camerei** (15 cadre/s), cu desenul ultimei detecții peste el:
  - caroiajul găsit și colțurile lui (roșii, le poți trage cu mouse-ul);
  - caroiajul văzut doar parțial (un colț în imagine): punctat portocaliu, cu o săgeată spre centrul lui;
  - cuburile găsite, cu chenar colorat.
- Eticheta caroiajului arată starea lui: **găsit**, **urmărit** (cu încrederea), **acoperit** (liniile sub cuburi), **parțial**, **NEGĂSIT**. Un caroiaj care dispare din imagine nu mai rămâne desenat.
- **Detecție**: caroiajul 3×3 pe cadrul curent și varianta **stabilă** (votul pe ultimele cadre). Sarcinile folosesc varianta stabilă.
- **Sarcina 1 / Sarcina 2**: setările fiecăreia, butoanele Rulează / Stop / Continuă și jurnalul.
- **Setări**: glisoare care se văd imediat pe imagine. „Salvează” le scrie în `data/settings.json`.
- Click pe imagine afișează valorile HSV ale punctului. Butonul „Învață” lărgește intervalul unei culori ca să cuprindă punctul ales.
- **Lumina**: bifa „foaia adusă la alb” (implicit pornită) face culorile, și verdele închis, recunoscute în lumină slabă sau colorată. „Auto-expune pe hârtie” fixează expunerea camerei, ca brațul să nu schimbe culorile când mută camera.
- **Îngheață** oprește imaginea pe un cadru, ca să reglezi pe aceeași imagine.
- **Reset** oprește sarcina, șterge memoria și jurnalul.
- **Braț**: desenul brațului (lateral și de sus) din ultimele unghiuri trimise și butonul **Pasul 1 + 2** (sau Select pe controller):
  1. se uită drept în față (poziția `SEARCH` dacă e învățată, altfel înălțimea din `LOOK_S` cu baza la 90°) și caută caroiajul **cu cuburi** alternând stânga / dreapta; îl centrează, salvează unghiurile ca `LOOK_S` / `LOOK_D` și memorează cuburile → notificare (în pagină și pe ecranul plăcii de răcire);
  2. așteaptă să fie luate toate cuburile de pe caroiaj, apoi se uită stânga / dreapta după ele și spune ce culori a găsit și ce lipsește.
- **Controllerul USB** e citit de pagină: cât timp nu rulează nimic, muți brațul din el (butoanele din [tools/gamepad.py](tools/gamepad.py)); Select pornește căutarea caroiajului. **În timpul unei sarcini sau căutări, orice buton sau stick o oprește pe loc** și brațul îngheață unde a ajuns. Nu porniți în paralel `tools/gamepad.py`.
- Pe Pi, coolerul și ecranul plăcii de răcire pornesc odată cu pagina (IP, temperatură, procesor, memorie).
- **Salvează poze** scrie în `captures/`. **Înregistrează 10 cadre** salvează poze brute pentru reglajul detecției fără braț (vezi [tests/README.md](tests/README.md)).
- Sursa **Braț simulat (cameră pe gripper)**: ce ar vedea camera de pe braț, din unghiurile brațului simulat. Mută brațul din controller sau pornește „Pasul 1 + 2” și vezi cum își caută și centrează caroiajul, fără braț.
- Brațul se mișcă lin: pornește și oprește fără smucituri, iar Stop îl oprește pe loc (în ≤ 25 ms).

## Pași până la concurs

1. Pe Pi: `python3 tools/calibrate_arm.py` – înveți pozițiile brațului (vezi [tools/README.md](tools/README.md)).
2. Pornești pagina și verifici că ambele caroiaje sunt găsite (S și D).
3. Reglezi culorile la lumina din sală (click pe cub, apoi „Învață”) și apeși „Salvează”.
4. Rulezi fiecare sarcină din pagină, întâi cu „Pauză între faze” bifată.

## Structura proiectului

| Folder | Conține |
|---|---|
| [`dofbot/`](dofbot/README.md) | codul: camera, detecția, brațul, planificarea, sarcinile, serverul web și pagina |
| [`tools/`](tools/README.md) | unelte din linia de comandă: calibrarea brațului, sarcinile fără pagină |
| [`tests/`](tests/README.md) | testele (fără braț și fără cameră) |
| [`data/`](data/README.md) | ce se calibrează și se salvează: poziții, colțuri, setări |
| [`deploy/`](deploy/README.md) | pornirea automată pe Raspberry Pi |
| [`docs/`](docs/) | tema rezumată și ghidul inițial |
| `captures/` | pozele salvate (se creează singur) |
| `start.bat`, `start.sh` | pornirea pe Windows / Pi |

## Note

- Pagina nu are parolă: rulați-o doar în rețeaua voastră.
- `data/vision.json` făcut pe laptop e pentru camera laptopului; pe Pi refaceți colțurile din pagină.
