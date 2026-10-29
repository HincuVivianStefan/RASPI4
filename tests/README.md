# tests/ – teste fără braț și fără cameră

```bash
py -m unittest discover -s tests -v        # Windows
python3 -m unittest discover -s tests -v   # Pi
```

Testele folosesc brațul simulat și scena desenată (`dofbot/synth.py`). Verifică:

- **planificarea**: ordinea rândurilor la S1; turnul la S2, inclusiv când ținta e ocupată sau caroiajul e plin (ca în imaginea din temă) și când lipsesc cuburi;
- **detecția**: caroiajul și culorile pe 15 aranjări aleatoare, caroiajul gol, votul pe mai multe cadre;
- **cuburile ca obiecte**: o bandă colorată sau o pată mare (de exemplu un tricou) nu sunt luate drept cuburi;
- **lumina**: sală întunecată (până la 20% lumină) și bec cald / rece: caroiajul și toate culorile, inclusiv verdele, rămân recunoscute;
- **urmărirea**: caroiajul luat de pe masă dispare din pagină în 2 cadre; colțuri salvate greșit nu desenează un caroiaj inexistent; caroiajul plin rămâne urmărit;
- **camera de pe brațul simulat**: caroiajul cu 9 cuburi găsit de la zero, caroiajul văzut doar parțial (direcția spre el), brațul care pornește de la un colț și ajunge să vadă tot caroiajul, centrat;
- **mișcarea**: lină (fără salt la pornire / oprire, viteza maximă respectată), exactă, oprită imediat la Stop, limitele servourilor;
- **camera**: auto-expunerea pe o cameră falsă aduce foaia la alb;
- **sarcinile cap-coadă**: S1 și S2 rulate complet în lumea simulată, plus oprirea la cerere;
- **setările**: valorile greșite sunt refuzate și nu schimbă nimic.

Rulați-le după orice schimbare în `dofbot/`. Durează cam un minut pe Pi (centrarea simulată singură ~40 s).

## Poze adevărate: `test_captures.py`

Detecția rulată pe poze reale înregistrate din pagină, fără braț:

1. cu camera conectată, în lumina în care vreți să meargă: „Înregistrează 10 cadre” → `captures/set_<dată>/` (PNG, `meta.json`, `labels.json`);
2. deschideți `labels.json`: `colors` e aranjarea detectată în momentul înregistrării. Corectați-o dacă e greșită și puneți `"verificat": true`;
3. `python3 tests/test_captures.py`: pe fiecare set verificat, cel puțin 90% din cadre trebuie să dea exact aranjarea din `labels.json`.

Setările folosite sunt cele din `data/settings.json`, deci așa verificați un reglaj nou (culori, lumină) pe toate pozele vechi. Fără seturi verificate, testul e sărit.
