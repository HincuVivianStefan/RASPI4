# data/ – ce se calibrează și se salvează

| Fișier | Cine îl scrie | Ce conține |
|---|---|---|
| `positions.json` | `tools/calibrate_arm.py` | unghiurile servomotoarelor pentru fiecare poziție (HOME, LOOK_S, S11_jos…) |
| `vision.json` | pagina: colțuri trase / 4 click-uri | colțurile caroiajelor S și D în imagine: punct de plecare pentru căutarea automată (folosite doar dacă se potrivesc cu imaginea), sau colțurile folosite direct când „caroiaj automat” e oprit |
| `settings.json` | pagina: butonul „Salvează” | setările modificate față de `dofbot/config.py`: culori HSV, lumina, expunerea camerei, praguri, ordinea turnului… |
| `aim.json` | centrarea caroiajului (`dofbot/aim.py`), după o centrare reușită | cum mută fiecare articulație caroiajul în imagine; cu el, centrarea nu mai face mișcări de probă. Șterge-l dacă muți camera pe braț |

- Lipsa unui fișier nu e o eroare: se folosesc valorile implicite.
- Ca să revii la setările din `config.py`, șterge `settings.json` sau apasă în pagină „Valori implicite”, apoi „Salvează”.
- Valorile depind de cameră și de lumină. Cele făcute pe laptop nu sunt bune pe Pi: refaceți-le acolo.
- Faceți o copie a folderului după calibrarea de dinaintea concursului.
- Pozele înregistrate din pagină („Înregistrează 10 cadre”) ajung în `captures/set_*/`, nu aici (vezi `tests/README.md`).
