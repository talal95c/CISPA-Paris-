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
