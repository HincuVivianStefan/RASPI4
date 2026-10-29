# Tema CRI 2026 – rezumat

Sursa: https://www.dyd.ro/cri/tema/ (cu login). Pagina conține un Google Sheet cu text și două imagini.
Citatele sunt exacte; ce nu apare în temă e marcat „nu se precizează”.

## Echipament

„Braț robotic Dofbot cu Raspberry Pi 4”, „Cameră compatibilă montată pe sistem”, „Caroiaj de 3 × 3 celule”, „Set de cuburi colorate”.

## Sarcina 1 – reproducerea unei aranjări

„Se da un caroiaj de 3 x 3, care contine cuburi colorate, situate in interiorul caroiajului pozitionate aleatoriu.” Sistemul trebuie să:

1. recunoască prin cameră poziția (linie, coloană) și orientarea fiecărui cub;
2. reproducă aranjarea „pe un al doilea caroiaj identic (aranjarea destinație)”, respectând poziția exactă și „Orientarea fiecărui cub (culoarea feței superioare)”.

## Sarcina 2 – turnul sortat

Sistemul trebuie să:

1. identifice culorile cuburilor de pe caroiaj;
2. construiască un turn în „o ordine predefinită a culorilor dată la start (ex: roșu → galben → albastru → verde, de jos în sus)”;
3. îl plaseze „într-o poziție indicată pe caroiaj, de catre juriu”.

## Punctaj (100)

| Criteriu | Puncte |
|---|---|
| Recunoașterea corectă a pozițiilor cuburilor (S1) | 20 |
| Reproducerea completă și corectă a pozițiilor și culorilor pe un caroiaj gol (S1) | 20 |
| Identificarea și sortarea culorilor (S2) | 20 |
| Construirea turnului în ordinea corectă (S2) | 20 |
| Viteză de execuție și precizie mecanică, în timpul prezentării | 10 |
| Bonus pentru creativitate | 10 |

Criteriile de departajare nu se precizează.

## Constrângeri

- „complet” autonom pe Raspberry Pi 4, „de preferat fără intervenție umană după start”;
- orice bibliotecă open-source (OpenCV, TensorFlow Lite etc.);
- maximum 5 minute pe sarcină.

## Imaginile din temă

- **Sarcina 1:** o grilă 3×3 plină, cu 9 cuburi albe cu fața de sus colorată și eticheta „jos: <aceeași culoare>”. Asta sugerează că fața de sus și cea de jos au aceeași culoare.

  |  | Col. 1 | Col. 2 | Col. 3 |
  |---|---|---|---|
  | R1 | galben | verde | albastru |
  | R2 | roșu | galben | verde |
  | R3 | albastru | roșu | galben |

- **Sarcina 2:** un turn de 5 cuburi, de jos în sus: roșu, galben, verde, albastru, roșu. Deci o culoare se poate repeta, iar turnul poate fi mai lung decât exemplul din text.
- Nicio imagine nu arată brațul sau poziția caroiajelor față de el.

## Ce nu se precizează (de întrebat organizatorii)

- De unde ia brațul cuburile pentru caroiajul destinație? Cum sunt așezate cele două caroiaje față de braț?
- Ce dimensiuni au cuburile și celulele, câte cuburi sunt și ce culori au celelalte fețe?
- Cum se dă ordinea turnului și cum se indică celula juriului? Celula poate fi ocupată?
- E permisă o confirmare (tastă/buton) între faze?

## Cum tratează codul aceste lucruri

- **Două caroiaje**, S și D, cu pozițiile brațului învățate pentru fiecare celulă.
- **„Orientarea”** = culoarea feței de sus.
- **Celula turnului, dacă e ocupată:** cubul e mutat într-o celulă liberă. Dacă nu există una, e pus peste un cub care nu e folosit în turn.
- **Ordinea turnului și celula** se dau din pagină sau din linia de comandă.
- **Pauza între faze** e opțională și dezactivată implicit.
