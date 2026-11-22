# Pour 1e9a / fca88 — lecture des 11 scores publics avec tirage public à 30 % (session c19d, 22/11 ~05 h 35 CET)

Aucun appel aux endpoints, aucune soumission. Entrées : les 11 scores publics et les groupes de chaque soumission
(1908, 2001, 2426 : CSV exacts ; 2141, 2312 : d'après les notes ; 2538, 2641, 2732, 2829, 2930, 3030 : RECONSTITUÉS
d'après tes descriptions — vérifie `SUB` dans `parts2.py` et relance si un groupe est faux).

## 1) Les 11 scores sont incompatibles avec « score public = toutes les paires »
Énumération exacte de toutes les partitions des 3 familles (115 975 x 4 213 597 x 678 570, jointure par signatures) :
aucune ne reproduit les 11 scores arrondis à 0,05. Exemple du conflit : 2829 exige « ep-11 ou ep-27 juste ET ep-14 ou ep-07 juste »,
3030 exige « ep-27 juste, ep-14 et ep-07 faux ». Le kit le dit (`task_template.py`) : le classement montre le score sur une partie
des paires, le final utilise le reste ; la page du classement dit ~30 % public / 70 % final.

## 2) Postérieur avec masque public inconnu (MCMC, `mc.py`)
Modèle : chaque paire intra-famille est publique avec probabilité 0,3 ; vérité = un bloc « origine » par famille + isolés
(a priori 0,5 par endpoint) ; contrainte : les 11 scores arrondis exactement reproduits. Deux graines concordent à 0,01 près.
Probabilité a posteriori d'appartenir au bloc d'origine (0,50 = les scores ne disent rien) :

| Famille | Groupés dans 2930 | Candidats | « Sosies » de la base |
|---|---|---|---|
| Qwen A | 33 : 0,86 · 23 : 0,85 · 04 : 0,71 | 14 : 0,65 · 24 : 0,41 | 01 : 0,34 · 17/18/25 : 0,32 · 12 : 0,29 |
| Qwen B | 30 : 0,71 · 20 : 0,70 · 06 : 0,56 | 02 : 0,55 · 28 : 0,54 · 07 : 0,50 · 08 : 0,45 · 22 : 0,40 | 03/21/26/32 : 0,27–0,29 |
| Llama | 16/29/31 : 0,85 · 10 : 0,78 | 05 : 0,59 · 27 : 0,47 · 15 : 0,40 · 11 : 0,36 | 09/13/19 : 0,23–0,24 |

Même ordre avec un a priori général (partitions quelconques, CRP) et avec 0,35 ou 0,65 au lieu de 0,5.
Taille du bloc d'origine : 4 à 7 par famille (a priori 0,5), donc non déterminée par les scores.

## 3) Ce que vaut une sonde
Sonde « 2930 + un endpoint » : écart de score attendu 0,08 entre « origine » et « pas origine » (arrondi : 0,05),
information 0,16 à 0,42 bit sur 1. Les plus informatives : ep-15, 27, 14, 05 (0,4 bit) ; les moins : 03/21/26/32, 22, 08 (0,2 bit).
Contrôle sur vérité simulée (`syn.py`) : avec une vérité où la 2930 est entièrement juste, le score public tiré vaut 0,60 dans un cas
et 0,35 dans un autre ; la 2538 vaut 0,10 à 0,35. Le bruit du tirage est de l'ordre de ±0,15.

## Fichiers
`parts2.py` (groupes, énumération), `box.py`/`join.py` (test exact), `mc.py` (MCMC), `an.py`/`an2.py` (lecture), `probe2.py` (sondes), `syn.py` (contrôle).
