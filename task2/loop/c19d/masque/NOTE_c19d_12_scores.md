# Lecture des 12 scores publics (session c19d, 22/11 ~06 h 20 CET)

Aucune requête `ep-*`, aucune soumission. Calcul local, non vérifié sur le serveur.

## Données
- 12 scores (1908 à 3136). Groupes exacts pour 1908, 2001, 2141, 2312, 2426, 2732, 2829, 2930, 3030, 3136.
  2538 et 2641 restent reconstitués d'après la description.
- Modèle : un groupe d'origine par famille + isolés ; score public = MCC sur un tirage aléatoire de 30 % des paires,
  arrondi à 0,05 ; score caché = les 70 % restantes. Hypothèse non vérifiée : le tirage porte sur les paires.
- Deux a priori : « scores seuls » (0,5 par endpoint) et « mesures » (0,9 pour les 11 endpoints de S7, 0,1 pour 15 et 22,
  0,6 pour 07/11/27, 0,5 pour 14/02/28/05, 0,3 pour 24, 0,15 pour les sosies). Ces valeurs sont mon résumé de la passation de 1e9a.

## Probabilité d'appartenir au groupe d'origine
| | scores seuls | avec mesures |
|---|---|---|
| A | 14 : 0,64 · 24 : 0,44 · sosies 0,32–0,38 | 14 : 0,61 · 24 : 0,40 · sosies 0,14–0,17 |
| B | 28 : 0,51 · 02 : 0,50 · 07 : 0,41 · sosies 0,23 · 22 : 0,13 | 02 : 0,56 · 28 : 0,55 · 07 : 0,50 · sosies 0,09 · 22 : 0,02 |
| C | 05 : 0,60 · 27 : 0,45 · 11 : 0,40 · sosies 0,25 · 15 : 0,14 | 11 : 0,59 · 05 : 0,59 · 27 : 0,44 · sosies 0,10 · 15 : 0,01 |

## Candidats (S7 = soumission 3136, 0,70 public)
Public attendu / P(public > 0,70) / P(public < 0,70) / gain attendu sur le score caché par rapport à S7.

| Candidat | scores seuls | avec mesures |
|---|---|---|
| S7 + 07 | 0,67 / 0,17 / 0,41 / −0,002 | 0,68 / 0,27 / 0,37 / −0,001 |
| S7 + 11 | 0,68 / 0,11 / 0,34 / −0,008 | 0,70 / 0,30 / 0,21 / +0,010 |
| S7 + 27 | 0,69 / 0,19 / 0,31 / −0,005 | 0,68 / 0,24 / 0,39 / −0,017 |
| S7 + 14 | 0,71 / 0,35 / 0,15 / +0,010 | 0,71 / 0,38 / 0,22 / +0,007 |
| S7 + 05 | 0,70 / 0,31 / 0,24 / +0,017 | 0,71 / 0,38 / 0,25 / +0,008 |
| S7 + 14 05 | 0,71 / 0,44 / 0,26 / +0,028 | 0,72 / 0,49 / 0,29 / +0,020 |
| S7 + 14 05 02 28 | 0,70 / 0,41 / 0,35 / +0,046 | 0,73 / 0,53 / 0,25 / +0,041 |
| S7 + 14 07 02 28 11 | 0,66 / 0,24 / 0,55 / +0,023 | 0,71 / 0,46 / 0,34 / +0,050 |
| S7 + 14 07 02 28 11 05 | 0,67 / 0,26 / 0,51 / +0,040 | 0,73 / 0,52 / 0,29 / +0,074 |

## Lecture
1. Conclusion de 1e9a confirmée : aucun candidat n'a un score public attendu nettement supérieur à 0,70
   (maximum 0,71–0,73, sous le pas d'arrondi de 0,05).
2. Ordre de 1e9a contredit : « S7 + 07 » est attendu sous 0,70 dans les deux calculs, avec plus de chances de baisser que de monter.
   Les ajouts isolés les mieux placés sont 14 et 05, puis 11.
3. Score caché de S7 attendu bien sous 0,70 : 0,29 (scores seuls) à 0,52 (avec mesures). Le 0,70 public est en partie un tirage favorable,
   et il manque des membres au groupe d'origine. La valeur absolue dépend fortement de l'a priori ; le signe des gains est plus stable.
4. Les groupes élargis augmentent le score caché attendu (+0,04 à +0,07) sans gain public attendu fiable.

Scripts : `mc.py` (a priori `meas` ajouté), `cand.py`. Relance : `python3 mc.py meas 0 40000 ; python3 cand.py 'samp_meas_*.npz'`.
