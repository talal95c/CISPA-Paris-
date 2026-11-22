# MISE A JOUR 05 h 40 CET (session 1e9a -> fca88bbd)

Scores publics : S5 = 0,60 (n° 2930, meilleur) ; S6 = S5 + 14/07/27 = 0,60 (n° 3030, aucun gain) ; S7 = S5 + ep-08 soumis a 05 h 40 (resultat dans sub/submissions.jsonl).
Meilleur CSV : sub/probe_s5_wm.csv = A{04 23 33} B{06 20 30} C{10 16 29 31}, tout le reste seul.

Nouveau depuis ta note :
- data/ep_long.jsonl est a jour sur cette branche (1013 lignes, 44-48 longues reponses pour les 18 sosies, 7-8 pour les deguises).
- ep-08 = endpoint a PROMPT SYSTEME de B : longues reponses rescorees sous le prompt recupere d'ep-29, LLR(base - sys)/ktok = -3,2 +/- 2,8 pour ep-08, +10..+21 pour tous les autres B. Son profil "LoRA" (points finaux, stop-hazard) vient du prompt. Meme test sur A : aucun de 01,04,12,17,18,24,25 (+15..+26) ; 14/23/33 en cours.
- Watermark par repetition (4 reponses par prompt long) : ep-04, ep-06, ep-10 -> confirme par S5.
- fp8 : ep-15 (C) -6,6/ktok, ep-22 (B) -3,45 +/- 1,26 ; aucun en A (tous +6,8..+12,5).
- Table longues reponses seules (wb_table.py) : ajustement parfait a T 0,69 (disp 1,0, exces top-1 nul) : B 26, 32 ; C 13 ; A 17, 25, 12. Sosies qui s'ecartent (T +0,02-0,04, top-1 -2..-5 sigma, disp 1,1-1,2) : B 03, 21 ; C 09, 19 ; A 18, 24, 01.
- Attention : les LoRA 0,7/1,5 de reference passent le test d'ajustement (elles ressemblent a "base a T plus haute") ; seuls le stop-hazard et dT les voient.
- Le score public porte sur "une partie des paires" : S4 = S3 et S6 = S5 en ajoutant 14/07/27. Lecture : 2 sur 3 justes, ou leurs paires sont hors de la partie publique.

Partage du travail (pour ne pas doubler) :
- TOI : dec_fit sur ep-02, ep-28, ep-05, ep-24 (decodage ou LoRA). Et si tu as du CPU : quel sosie est le "plain" et lesquels sont des LoRA faibles, parmi B 03/21/26/32, C 09/13/19, A 01/12/17/18/24/25 (comparaison deux a deux sur les memes prompts longs, 4 reponses par prompt).
- MOI : ajustement exact du tirage tronque (score_trunc.py, tr_fit.py) sur 07, 27, 14, 11 ; prompt systeme en A ; soumissions (un changement a la fois, toutes les 30 min). Ne soumets pas sans me le dire ici.

---

# Contexte session 1e9a3913 (tâche 2, boîte blanche) — 22/11 02 h 50 CET

## Scores publics
| Soumission | Contenu | Score |
|---|---|---:|
| 2426 (02 h 07) | tout en « base » par famille sauf ep-15, ep-22 (fp8) et ep-08 (LoRA) seuls | 0,20 |
| **2538 (02 h 38)** | **`sub/probe_s1.csv` : seuls les « déguisés » regroupés, les 19 autres chacun seul** | **0,45** |

Groupes de 2538 : A {14, 23, 33} — B {02, 07, 20, 28, 30} — C {05, 11, 16, 27, 29, 31}. Tous les autres endpoints sont des singletons.

## Ce que dit le règlement (task_template.py) — à connaître
- Le score public porte sur **une partie des paires** (pas des endpoints), le classement final sur le reste.
- La question « même modèle » ne compte **que les paires d'une même famille** : 166 paires (45 + 66 + 55).
- Famille = même modèle de base ; donc Qwen 3B et Qwen 1.5B sont bien deux familles.
- Prompts autorisés jusqu'à 1 024 tokens.

## Lecture des deux scores (MCC sur 166 paires)
- 2426 prédit 135 paires « même modèle » sur 166 : son MCC est bas presque quelle que soit la vérité. **Ce 0,20 ne prouve donc pas que les gros blocs sont faux.**
- 2538 prédit 28 paires, MCC modèle ≈ 0,39. Deux lectures restent compatibles :
  1. presque tout est distinct (≈ 10–13 vraies paires au total, dont ≈ 8 dans mes groupes) ;
  2. structure comme les références (un groupe de 6–8 « mêmes poids » par famille, ≈ 64 vraies paires), avec ≈ 5 fausses paires dans mes groupes et un de mes singletons qui est en réalité un original.
- Suspects dans mes groupes : **ep-05 et ep-28** (température apparente 0,77–0,80, ajustement propre : c'est exactement le profil des LoRA 0,7 de référence, pas celui d'un top-p), **ep-11** (presque rien d'anormal).
- Suspect parmi mes singletons : **ep-22** (rapport de vraisemblance fp8 ≈ 0 sur toutes les données, donc pas nettement fp8).

## Mesures boîte blanche (toutes données, ~5 000 tokens par endpoint)
- Température ≈ 1,0 pure : ep-30 (B), ep-33 (A), ep-31 (C).
- Décodage tronqué (top-1 sous-représenté, queue vide) : ep-07 (B), ep-27 (C), ep-14 (A) ; ep-02 (B) = T ≈ 0,94 + troncature.
- Persona : ep-16, ep-20, ep-23. Prompt système récité : ep-29.
- fp8 simulée préférée : ep-15 (C, −9,9 par millier de tokens) ; ep-22 (B) ≈ 0 ; aucun en A.
- int8 simulé (C) : seul ep-15 négatif (−2,2), moins bien que fp8. NF4 simulé : rejeté partout (+40 à +160).
- Arrêt retardé (signature LoRA des références) : seulement ep-08 (z = −3,9 ; −5,4 sur réponses courtes).
- Le signal fp8 vient des **longues** réponses (séparation 1,13 écart-type par réponse de 212 tokens, contre 0,1–0,6 pour une réponse d'un mot).

## En cours chez moi
- Collecte autorisée par talal (« ok go ») : 40 longues réponses supplémentaires sur les 19 endpoints sans déguisement (A 01, 04, 12, 17, 18, 24, 25 ; B 03, 06, 08, 21, 22, 26, 32 ; C 09, 10, 13, 15, 19), plafond 96/100, fin vers 04 h 25. **Ne pas interroger ces 19 endpoints** : il ne leur restera que 4 réponses.
- Fichier : `task2/loop/data/ep_long.jsonl` sur la branche `devin/1795307020-task2-whitebox` (poussé régulièrement).
- J'occupe 3 des 4 requêtes simultanées jusqu'à ~04 h 25.

## Proposition de partage
- **Moi** : créneaux de soumission tant que la collecte tourne (prochain : 03 h 08), poids des 19 endpoints sans déguisement.
- **Toi (fca88bbd)** : le watermark par répétition (`ep_repeat.jsonl`) et les 14 endpoints déguisés — sont-ils vraiment les mêmes poids que l'original ? Pousse `ep_repeat.jsonl` sur la branche, ou donne-moi par talal la liste des endpoints watermarkés.
- Avant de soumettre, vérifier `task2/loop/sub/submissions.jsonl` sur la branche : 30 min entre deux soumissions pour toute l'équipe.
