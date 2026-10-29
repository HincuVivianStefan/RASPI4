# tests/ – teste fără braț și fără cameră

```bash
py -m unittest discover -s tests -v        # Windows
python3 -m unittest discover -s tests -v   # Pi
```

Testele folosesc brațul simulat și scena desenată (`dofbot/synth.py`). Verifică:

- **planificarea**: ordinea rândurilor la S1; turnul la S2, inclusiv când ținta e ocupată sau caroiajul e plin (ca în imaginea din temă) și când lipsesc cuburi;
- **detecția**: caroiajul și culorile pe 15 aranjări aleatoare, caroiajul gol, votul pe mai multe cadre;
- **cuburile ca obiecte**: o bandă colorată sau o pată mare (de exemplu un tricou) nu sunt luate drept cuburi;
- **sarcinile cap-coadă**: S1 și S2 rulate complet în lumea simulată, plus oprirea la cerere;
- **setările**: valorile greșite sunt refuzate și nu schimbă nimic.

Rulați-le după orice schimbare în `dofbot/`. Durează sub 2 secunde.
