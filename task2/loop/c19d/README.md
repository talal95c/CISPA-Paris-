# Tâche 2 — portefeuille de soumissions (session c19d)

Scripts d'analyse locaux. Aucune clé dans ce dossier (`SPRINTML_API_KEY` est lue dans l'environnement).

## Contenu

- `masque/` : modèle du score public et simulation du score caché.
  - `parts2.py` : partitions soumises et scores publics observés (n° 1908 à 3782).
  - `mc.py` : MCMC sur (partition d'origine, masque public) compatible avec les scores arrondis à 0,05 ; score = 0,1 + 0,9·max(0, MCC) sur les paires intra-famille.
  - `umask.py`, `uan.py` : comparaison des masques publics « par paire », « par endpoint, les deux dans U », « au moins un dans U ».
  - `emax5.py`, `emax6.py` : choix de l'ordre des CSV qui maximise l'espérance du meilleur score caché.
  - `NOTE_*.md` : notes intermédiaires.
- `poids/` : mesures sur les poids de base (CPU).
  - `qllr.py`, `an.py` : vraisemblance des réponses sous les poids de base contre une quantification simulée (fp8, int8, NF4).
  - `top1.py`, `top1an.py` : température ajustée et excès de top-1.
  - `fp.py`, `fan3.py` : empreinte par gradient ; non concluante sur les références, non utilisée pour décider.
- `csv/` : `s11.csv`, `s15.csv`, `s15_17.csv`, `s15_12.csv`, `s15_01.csv`, et `sched.py` (soumission de secours, un essai par créneau).

## Scores publics (serveur) à partir du n° 3235

| n° | CSV | public |
|---|---|---|
| 3235 | S8 = A{04 23 33} B{06 08 20 30} C{10 13 16 29 31} | 0,85 |
| 3319 | S8 + 05 | 0,85 |
| 3413 | S8 + 26 | 0,65 |
| 3503 | S11 = S8 + 32 | 1,00 |
| 3605 | S11 + 17 | 1,00 |
| 3694 | S11 + 12 | 1,00 |
| 3782 | S11 + 25 | 0,75 |
| 3872 | S15 = S11 + 27 07 14 | 1,00 |
| 3956 | S15 + 17 | 1,00 |
| 4035 | S15 + 12 | 1,00 |
| 4118 | S15 + 01 | 1,00 |

Le score public ne porte que sur une petite partie des paires (environ 13 % d'après l'ajustement local) : une égalité à 1,00 ne valide pas l'endpoint ajouté, seule une baisse est informative. Les scores cachés ne sont pas connus ; les valeurs « attendues » des scripts sont des simulations locales.
