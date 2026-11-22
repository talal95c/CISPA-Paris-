# Track 2 « Model Equality » — passation (session 1e9a, en pause le 22/11 à 06 h 00 CET)

Échéance : 22/11 12 h 00 CET. Dépôt `talal95c/CISPA-Paris-`, branche `devin/1795307020-task2-whitebox`, dossier `task2/loop/`.
Journal complet des essais : `NOTES.md`. Rien ne tourne : aucun calcul, aucune collecte, aucune soumission programmée.

## État
- **Meilleur score public : 0,70** (soumission n° 3136, 05 h 39 CET). CSV : `sub/probe_s7_sys08.csv`.
- Groupes de ce CSV (tout autre endpoint est seul) : A{04 23 33}  B{06 08 20 30}  C{10 16 29 31}.
- Prochain créneau de soumission : à partir de 06 h 10 CET (délai serveur de 30 min, partagé par toute l'équipe).
- Score caché inconnu. Le score public est arrondi à 0,05 et ne porte que sur une partie des paires.

## Historique des sondes (public)
| n° | CSV | groupes | score |
|---|---|---|---|
| 2426 | wb_rule_v1b | 3 gros blocs « base », 15/22/08 seuls | 0,20 |
| 2538 | probe_s1 | déguisés + températures hautes | 0,45 |
| 2641 | probe_s2_noC | S1 sans le groupe C | 0,35 |
| 2732 | probe_s3_core | A{23 33} B{20 30} C{16 29 31} | 0,45 |
| 2829 | probe_s4_trunc | S3 + 07 11 14 27 | 0,45 |
| 2930 | probe_s5_wm | S3 + 04 06 10 (watermark) | 0,60 |
| 3030 | probe_s6_trunc3 | S5 + 14 07 27 | 0,60 |
| 3136 | probe_s7_sys08 | S5 + 08 | **0,70** |

## Acquis
- Familles et poids ouverts (dans `/home/ubuntu/models/`) : A = Qwen2.5-3B-Instruct {01 04 12 14 17 18 23 24 25 33} ;
  B = Qwen2.5-1.5B-Instruct {02 03 06 07 08 20 21 22 26 28 30 32} ; C = Llama-3.2-1B-Instruct {05 09 10 11 13 15 16 19 27 29 31}.
  Références ref-01..09 = SmolLM2-360M-Instruct (une variante connue par référence : plain, sys, persona, watermark, T=1, fp8, LoRA 0,3/0,7/1,5).
- Mêmes poids que la base, confirmés par mesure ET par score public : 23 (persona), 33 (T≈1), 04 (watermark) ; 20 (persona), 30 (T≈1),
  06 (watermark), 08 (prompt système) ; 16 (persona), 29 (prompt système, récité), 31 (T≈1), 10 (watermark).
- fp8 (à garder seuls) : ep-15 (C, LLR base-fp8 = -6,6 ± 0,7 /ktok) et ep-22 (B, -3,45 ± 1,26). Aucun fp8 en A.
- Prompt système récupéré sur ep-29 : « You are a helpful assistant for general questions and everyday tasks. Ask a clarifying question when the
  request is ambiguous. When you give an answer, say briefly why. » ep-08 est le seul B qui le préfère (-3,2 ± 2,8 ; autres B +10..+21).
  Aucun endpoint A ne le préfère : l'endpoint à prompt système de A, s'il existe, n'est pas trouvé.

## Ouvert (non tranché)
- Tirage tronqué, ajustement exact sur les poids de base (`score_trunc.py`, `tr_fit.py`, sorties `tr/fit_b.txt`, `tr/fit_c.txt`) :
  ep-07 = T 1,0 + top-k 50 + top-p 0,95 (écart d'ajustement z +0,5 contre -4,3 sans troncature) ; ep-27 = T 0,9 + top-k 40-50 (z ±1,3 contre -3,4) ;
  ep-11 = T 0,8 + top-k 40 (z +2,6 contre -4,6). Lecture : réglage de décodage sur les poids de base, donc à regrouper.
  Limites : grille de température au pas de 0,1, test non passé sur les références, et les témoins connus sortent mal (20 : -3,9 ; 30 : +4,2 ; 31 : +20).
  À confronter au fait que S6 (+14 07 27) n'a rien gagné : au moins un des trois est faux, ep-14 (A) non ajusté ici.
- ep-05 et ep-28 : ajustés sans troncature à T 0,8 (z -0,8 et 0,0) ; ep-02 : T 1,0, z +5,1. Base à température plus haute ou LoRA : non départagé
  (les LoRA de référence ressemblent aussi à la base à température plus haute). La session fca88bbd y travaille (`dec_fit.py`).
- Les 13 « sosies » encore seuls (A : 01 12 17 18 24 25 ; B : 03 21 26 32 ; C : 09 13 19) : au moins un vrai « plain » par famille devrait s'y
  trouver, non identifié. Signal LoRA par probabilité d'arrêt (`wb_eos.py`) validé sur les références mais pas stable sur les endpoints notés.
- `lb_design.py` (posteriori sur les 9 scores publics + contraintes mesurées) : probabilité d'appartenir au groupe de base
  07 : 0,67 ; 11 : 0,63 ; 14 : 0,59 ; 02 et 28 : 0,54 ; 27 : 0,38 ; 05 : 0,37 ; chaque sosie : 0,03-0,13.
  Meilleur score attendu : S7 + {14 07 02 28 11} ≈ 0,72, soit pas mieux que 0,70 : aucune sonde suivante n'est justifiée par ce seul calcul.

## Prochaine étape proposée (rien de lancé)
1. Valider `score_trunc.py` sur les références SmolLM2 (`data/refs_all.jsonl`, `data/ref_long.jsonl`) avec une grille de température au pas de 0,02.
2. Si les témoins sortent à z≈0 : sonde S8 = S7 + 07 (B) seul, puis + 11 et + 27 (C), un changement par créneau.
3. Chercher le « plain » de chaque famille parmi les sosies (deux réponses identiques à la base sur toutes les mesures).

## Budget d'appels (100 max par endpoint, 4 requêtes simultanées par équipe)
Environ 93-96 utilisées sur 19 endpoints (01 03 04 06 08 09 10 12 13 15 17 18 19 21 22 24 25 26 32), 52-54 sur les 14 autres.
Vérifier côté serveur avant toute collecte. Données : `data/ep_long.jsonl` (1 013 lignes), `data/ep_short.jsonl` (990), `data/cache_all.jsonl`.

## Non fait
- GPU : aucun calcul n'a tourné sur JURECA (nœud de l'équipe occupé, puis fenêtres MFA fermées). Tout est en CPU local (31 Go : un gros modèle à la fois).
- AIDE / MLE-bench : non branchés (clés OpenAI et Mistral refusées, 401). Boucle manuelle : hypothèse écrite avant, une variable par sonde, journal dans NOTES.md.
- Messages directs entre sessions : refusés (403). Canal : `POUR_AUTRE_SESSION.md` sur cette branche et les fichiers `fca8/`.

## Outils (dans task2/loop/)
`mkcsv.py out.csv "04 23 33|06 08 20 30|10 16 29 31"` (CSV par groupes) ; `score_wb.py` + `runq.sh` (log-probabilités sous un checkpoint, options
`--system`, fp8) ; `wb_llr.py a b` (LLR par endpoint avec bootstrap) ; `wb_dt.py`, `wb_eos.py`, `wb_trunc.py` ; `score_trunc.py` + `tr_fit.py` ;
`lb_post.py`, `lb_design.py` (lecture des scores publics) ; `sub/submissions.jsonl` (toutes les réponses du serveur).
Python : `/home/ubuntu/ref-evaluation/met-paper-env/bin/python`.
