# data/ – ce se calibrează și se salvează

| Fișier | Cine îl scrie | Ce conține |
|---|---|---|
| `positions.json` | `tools/calibrate_arm.py` | unghiurile servomotoarelor pentru fiecare poziție (HOME, LOOK_S, S11_jos…) |
| `vision.json` | pagina: colțuri trase / 4 click-uri | colțurile caroiajelor S și D în imagine, folosite când caroiajul nu e găsit automat |
| `pick.json` | pagina: „Învață prinderea” | unde se vede cubul (la scanare și de deasupra) pentru `PICK_NEAR` / `PICK_FAR` |
| `settings.json` | pagina: butonul „Salvează” | setările modificate față de `dofbot/config.py`: culori HSV, praguri, ordinea turnului… |

- Lipsa unui fișier nu e o eroare: se folosesc valorile implicite.
- Ca să revii la setările din `config.py`, șterge `settings.json` sau apasă în pagină „Valori implicite”, apoi „Salvează”.
- Valorile depind de cameră și de lumină. Cele făcute pe laptop nu sunt bune pe Pi: refaceți-le acolo.
- Faceți o copie a folderului după calibrarea de dinaintea concursului.
