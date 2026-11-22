# Tâche 2 « Model Equality Testing » (CISPA / EuroAI) : passation complète

Rédigé par la session fca88bbd le 22/11/2026 à 06 h 05 CET, pour l'agent qui reprend.
Toutes les heures sont en CET. Les horodatages des fichiers et de `submissions.jsonl` sont en UTC (CET − 1).

Chaque affirmation porte son origine : **[mesuré fca8]** (calculé dans cette session), **[1e9a]** (lu dans les fichiers de
la session partenaire, non recalculé ici), **[serveur]** (réponse du serveur de la compétition), **[hypothèse]**.

---

## 1. Le problème

33 endpoints d'API (`ep-01` à `ep-33`) servent des modèles de langage. Il faut rendre un CSV
`endpoint,family,model` : même `family` = même modèle de base, même `model` = exactement les mêmes poids.

- **Score** = 0,1 × MCC(familles) + 0,9 × MCC(modèles). Le MCC modèles ne compte que les 166 paires intra-famille
  (45 en A, 66 en B, 55 en C). Un MCC négatif est ramené à 0. (`official/task_template.py`)
- **Score public** : calculé sur une partie des paires seulement, arrondi à 0,05. Le classement final utilise un jeu caché.
  Le classement garde le meilleur score ; par prudence, resoumettre le meilleur CSV en dernier. [1e9a]
- **Échéance des soumissions : 22/11 à 12 h 00 CET** (lu par 1e9a sur la page du classement). [1e9a]
- **Délai entre deux soumissions : 30 min, partagé par toute l'équipe** (HTTP 429 sinon). [serveur]
- **Budget : 100 réponses par endpoint pour toute la compétition**, 4 requêtes simultanées par équipe.
- 9 endpoints de référence (`ref-01` à `ref-09`, SmolLM2-360M-Instruct) ont un budget à part et une vérité connue :

| ref | variante | modèle |
|---|---|---|
| ref-01 | plain | S (base) |
| ref-02 | quantifiée FP8 | T1 |
| ref-03 | prompt système | S |
| ref-04 | LoRA force 0,7 | T3 |
| ref-05 | persona | S |
| ref-06 | watermark | S |
| ref-07 | LoRA force 1,5 | T4 |
| ref-08 | température 1,0 | S |
| ref-09 | LoRA force 0,3 | T2 |

Leçon des références : **tout ce qui change seulement le décodage ou le prompt (température, persona, prompt système,
watermark) garde les poids de base ; FP8 et LoRA sont des modèles différents et ont l'air « normaux ».**

---

## 2. État actuel

- **Meilleur score public : 0,70** (soumission n° 3136, 05 h 39, par la session 1e9a). [serveur, lu dans `sub/submissions.jsonl`]
- CSV correspondant : `task2/loop/sub/probe_s7_sys08.csv` sur la branche `devin/1795307020-task2-whitebox`.
- Groupes de ce CSV, tout autre endpoint étant seul dans son modèle :
  - A : `ep-04`, `ep-23`, `ep-33`
  - B : `ep-06`, `ep-08`, `ep-20`, `ep-30`
  - C : `ep-10`, `ep-16`, `ep-29`, `ep-31`
- Score caché : inconnu.
- La session 1e9a est en pause depuis 06 h 00 ; rien ne tourne chez elle. Prochain créneau de soumission : à partir de 06 h 10.
- Chez fca88bbd : deux calculs CPU tournent encore (section 7). Aucune collecte, aucune soumission programmée.

### Historique des soumissions (score public)

| n° | par | contenu | score |
|---|---|---|---|
| 1908, 2001 | fca8 | premiers regroupements larges | 0,10 |
| 2141 | fca8 | un gros bloc « base » par famille, `ep-02`, `28`, `05` isolés | 0,10 |
| 2312 | fca8 | sonde : `01/04/12`, `06/21/22`, `10/13/19` groupés, reste seul | 0,10 |
| 2426 | 1e9a | 3 gros blocs « base », `15`, `22`, `08` seuls | 0,20 |
| 2538 | 1e9a | S1 : déguisés + températures hautes groupés | 0,45 |
| 2641 | 1e9a | S2 : S1 sans le groupe C | 0,35 |
| 2732 | 1e9a | S3 : A{23 33} B{20 30} C{16 29 31} | 0,45 |
| 2829 | 1e9a | S4 : S3 + 07 11 14 27 | 0,45 |
| 2930 | 1e9a | S5 : S3 + 04 06 10 (watermark) | 0,60 |
| 3030 | 1e9a | S6 : S5 + 14 07 27 | 0,60 |
| 3136 | 1e9a | S7 : S5 + 08 | **0,70** |

Pourquoi les quatre premiers 0,10 : on regroupait les endpoints qui ressemblent à la base (en fait surtout des pièges) et on
isolait les déguisés (en fait le vrai bloc de base). C'était l'inverse de la bonne lecture.

---

## 3. Familles et modèles ouverts

| Famille | Modèle ouvert | Endpoints |
|---|---|---|
| A | `Qwen/Qwen2.5-3B-Instruct` | 01 04 12 14 17 18 23 24 25 33 |
| B | `Qwen/Qwen2.5-1.5B-Instruct` | 02 03 06 07 08 20 21 22 26 28 30 32 |
| C | `unsloth/Llama-3.2-1B-Instruct` | 05 09 10 11 13 15 16 19 27 29 31 |

Les familles sont considérées justes : le score de 0,10 des premières soumissions correspond à 0,1 × 1 + 0,9 × 0. [hypothèse cohérente avec le serveur]
A et B sont deux familles distinctes malgré le nom Qwen.

---

## 4. Ce qui est établi, endpoint par endpoint

| Endpoint | Fam. | Lecture | Preuve |
|---|---|---|---|
| 23, 20, 16 | A, B, C | persona, poids de base | mesure + S3 [1e9a, serveur] |
| 33, 30, 31 | A, B, C | température ≈ 1, poids de base | mesure + S3 [1e9a, serveur] ; `ep-30` T = 1,008 sans écart [mesuré fca8] |
| 04, 06, 10 | A, B, C | watermark, poids de base | répétition de textes longs + S5 (0,45 → 0,60) [1e9a, serveur] |
| 29 | C | prompt système, poids de base | prompt récité par l'endpoint [1e9a] |
| 08 | B | prompt système, poids de base | préfère le prompt de `ep-29` + S7 (0,60 → 0,70) [1e9a, serveur] |
| 15 | C | **FP8, à garder seul** | LLR base − FP8 = −6,6 ± 0,7 par millier de tokens [1e9a] ; −4,1 [mesuré fca8] |
| 22 | B | **FP8, à garder seul** | −3,45 ± 1,26 [1e9a] ; 2,9 écarts-types sous la base sur réponses longues [mesuré fca8] |
| 02, 07, 20, 28, 30 | B | pas des FP8 | à plus de 6 écarts-types d'une FP8, entre −1,0 et +1,0 de la base [mesuré fca8, `exp_br.txt`] |
| famille A | A | aucune FP8 | tous entre +6,8 et +12,5 [1e9a] |

Prompt système récupéré sur `ep-29` [1e9a] : « You are a helpful assistant for general questions and everyday tasks.
Ask a clarifying question when the request is ambiguous. When you give an answer, say briefly why. »
Aucun endpoint de A ne préfère ce prompt : l'endpoint à prompt système de A, s'il existe, n'est pas trouvé.

Point d'attention : `ep-08` sortait comme « LoRA » au test de probabilité d'arrêt (−4,6 à −5,1). C'était l'effet de son prompt
système, pas de ses poids. **Le test d'arrêt confond donc prompt système et LoRA sur les endpoints notés.**

---

## 5. Ce qui reste ouvert

D'après le score public, il manque environ 2 membres par groupe de base [estimation 1e9a, `lb_post.py`].

**a) Tirage tronqué : `ep-07` (B), `ep-14` (A), `ep-27` et `ep-11` (C)**
- Ajustement exact sur les poids de base [1e9a, `tr/fit_b.txt`, `tr/fit_c.txt`] : `ep-07` = T 1,0 + top-k 50 + top-p 0,95 ;
  `ep-27` = T 0,9 + top-k 40-50 ; `ep-11` = T 0,8 + top-k 40.
- Limites de ce test : grille de température au pas de 0,1, non validé sur les références, et les témoins connus sortent mal
  (20 : −3,9 ; 30 : +4,2 ; 31 : +20).
- Les ajouter au groupe n'a rien changé au score (S4 = S3, S6 = S5). Soit un des trois est faux, soit leurs paires sont hors
  de la partie publique.

**b) Température élevée : `ep-02`, `ep-28` (B), `ep-05` (C), `ep-24` (A)**
- Température ajustée sous les poids de base [mesuré fca8, `dec_an.py`] : `ep-02` 0,916 ; `ep-28` 0,783 ; `ep-26` 0,711.
- Pas de signature de top-p ou top-k sur `ep-02` et `ep-28` : le nombre de tokens observés hors du noyau top-p 0,90 ou
  hors du top-10 est égal à l'attendu (132 contre 130 et 231 contre 217 pour `ep-02` ; 98 contre 99 et 105 contre 107
  pour `ep-28`). [mesuré fca8, environ 2 000 tokens chacun]
- Ce n'est **pas** une preuve de poids de base : les LoRA 0,7 et 1,5 des références ressemblent aussi à « la base à
  température plus haute ». Le test de la section 7 est fait pour trancher.
- `ep-05` et `ep-24` : pas encore passés dans mes calculs.

**c) Les 13 sosies encore seuls** : A `01 12 17 18 24 25` ; B `03 21 26 32` ; C `09 13 19`
- Au moins un vrai « plain » par famille devrait s'y trouver. Non identifié.
- Table de 1e9a sur réponses longues : ajustement parfait à T 0,69 pour B `26`, `32` ; C `13` ; A `17`, `25`, `12`.
  Sosies qui s'écartent : B `03`, `21` ; C `09`, `19` ; A `18`, `24`, `01`. [1e9a]
- La sonde 2312 (`01/04/12`, `06/21/22`, `10/13/19` groupés) n'a donné que 0,10 : regrouper des sosies entre eux est risqué.

**d) Probabilités a posteriori de 1e9a** (`lb_design.py`, à partir des scores publics) d'appartenir au groupe de base :
07 : 0,67 ; 11 : 0,63 ; 14 : 0,59 ; 02 et 28 : 0,54 ; 27 : 0,38 ; 05 : 0,37 ; chaque sosie : 0,03 à 0,13.
Meilleur score attendu avec S7 + {14 07 02 28 11} ≈ 0,72, donc pas de gain garanti par ce seul calcul.

---

## 6. Les tests disponibles et ce qu'ils valent

| Test | Ce qu'il détecte | Validation sur les références | Limite connue |
|---|---|---|---|
| LLR base contre FP8 simulée (`exp_llr.py`, `wb_llr.py`) | FP8 | ref-02 à −6,1 sur réponses longues, les autres entre −2,3 et +0,7 | n'utiliser que les réponses longues ; ne voit pas les LoRA |
| Probabilité d'arrêt (`wb_eos.py`) | LoRA, sur questions à réponse en un mot, `max_tokens=10` | LoRA à −6 à −21, le reste entre −1,6 et +2,8 | confondu par un prompt système (`ep-08`) ; aucun signal net en A ni en C |
| Répétition de la même question longue (`rep_prefix.py`) | watermark | ref-06 : 28,9 mots identiques en tête contre 0,2 à 2,4 | il faut plusieurs réponses longues par question ; 2 questions reposées une fois ne suffisent pas |
| Température ajustée (`dec_fit.py`, `dec_an.py`) | décodage à température modifiée | ref-08 à 1,01 à 1,03 | les LoRA fortes montent aussi la température ajustée |
| Ajustement tronqué exact (`score_trunc.py`, `tr_fit.py`) | top-k, top-p | non validé | témoins mal ajustés |
| Queue de distribution (`tail_fit.py`, `tail_an.py`) | LoRA contre « base à température haute » | voir section 7 | LoRA 0,3 faiblement séparée |

Simulation FP8 : arrondi E4M3 avec une échelle par tenseur (`max|w| / 448`) sur les couches `Linear`, hors `lm_head`.

Score local : `ref_score.py` applique une règle aux 9 références et calcule le MCC officiel. La règle « arrêt z < −4 → LoRA ;
sinon LLR FP8 z < 0 → FP8 ; sinon base » donne 1,000 ou 0,734 selon la série de réponses [1e9a]. Les seuils ont été choisis
sur ces mêmes références : c'est optimiste, et ce score local ne s'est pas retrouvé tel quel sur le public.

---

## 7. Calcul en cours chez fca88bbd : test de queue de distribution

**Idée.** Pour chaque endpoint, on ajuste la température sous les poids de base, puis on compte les tokens émis de rang ≥ K
(K = 5, 10, 20, 50, 100) et on compare au nombre attendu sous la base à cette température. Un endpoint qui est vraiment
« base à température haute » doit tomber sur l'attendu ; un LoRA émet moins de tokens rares que prévu.

**Résultat sur les références** (999 réponses sur 1 149 traitées, toutes longueurs, écarts en écarts-types) [mesuré fca8] :

| ref | variante | T ajustée | rang ≥ 10 | rang ≥ 20 | rang ≥ 50 | observé / attendu (rang ≥ 20) |
|---|---|---|---|---|---|---|
| ref-01 | plain | 0,701 | −0,3 | −0,3 | +0,5 | 20 / 21 |
| ref-02 | FP8 | 0,738 | −1,1 | −0,2 | 0,0 | 28 / 29 |
| ref-03 | prompt système | 0,775 | −0,2 | +0,1 | +1,1 | 33 / 32 |
| ref-05 | persona | 0,771 | −1,0 | +0,6 | +1,4 | 51 / 47 |
| ref-06 | watermark | 0,715 | +1,7 | +1,1 | +2,4 | 31 / 26 |
| ref-08 | température 1,0 | 1,030 | +0,3 | −0,5 | −1,7 | 250 / 256 |
| ref-09 | LoRA 0,3 | 0,746 | −1,4 | −2,2 | −0,1 | 22 / 34 |
| ref-04 | LoRA 0,7 | 0,824 | −2,9 | −2,4 | −1,8 | 46 / 64 |
| ref-07 | LoRA 1,5 | 0,956 | −4,8 | −5,0 | −5,4 | 100 / 156 |

Lecture : la température 1,0 tombe sur l'attendu, la LoRA 1,5 est nette (−5), les LoRA 0,7 et 0,3 sont faibles (−2 à −3)
avec environ 2 500 tokens par endpoint. Les endpoints notés ont 12 000 tokens de réponses longues pour les sosies, donc
plus de puissance ; les déguisés n'en ont que 7 ou 8 réponses longues.

**Ce qui tourne** (VM de fca88bbd, dossier `/home/ubuntu/t2quick/other/task2/loop/`) :

| Sortie | Modèle | Avancement à 06 h 03 | Fin estimée |
|---|---|---|---|
| `tail/r.npz` | SmolLM2 (références) | 999 / 1 149 | quelques minutes |
| `tail/b.npz` | Qwen2.5-1.5B (12 endpoints B) | 49 / 535 | environ 1 h |
| `tail/c.npz` | Llama-3.2-1B (11 endpoints C) | 49 / 433 | environ 1 h |

Les sorties sont écrites toutes les 50 réponses et le script reprend où il s'est arrêté. Analyse :
`../../../env/bin/python tail_an.py tail/b.npz tail/c.npz` (ajouter `--min 50` pour ne garder que les réponses longues).
La famille A (Qwen 3B) n'est pas lancée : à faire seule, après B et C, faute de mémoire.

---

## 8. Prochains objectifs

Objectif : dépasser 0,70 public avant 12 h 00, sans dégrader le score caché, donc avec des changements justifiés par une
mesure et pas par un ajustement au score public.

1. **Finir le test de queue sur B et C, puis A.** Critère fixé à l'avance : un endpoint à température élevée dont les
   écarts de rang restent dans ±2 (comme ref-08) est rattaché au groupe de base ; un endpoint à −3 ou moins sur deux seuils
   (comme ref-04, ref-07) reste seul. Concerne `ep-02`, `28`, `05`, `24`, `14`.
2. **Valider l'ajustement tronqué sur les références** avec une grille de température au pas de 0,02 (`score_trunc.py`).
   Si les témoins sortent à z ≈ 0, les lectures de `ep-07`, `ep-27`, `ep-11` deviennent utilisables.
3. **Sondes, un seul changement par créneau de 30 min**, avec prédiction écrite avant :
   - S8 = S7 + `ep-07` (candidat le plus probable d'après 1e9a) ;
   - puis le meilleur candidat de l'étape 1 dans chaque famille ;
   - puis `ep-11` et `ep-27` en C.
   Un changement qui ne fait pas monter le score est retiré.
4. **Chercher le « plain » de chaque famille** parmi les sosies : candidats `26`, `32` (B), `13` (C), `17`, `25`, `12` (A).
   Le test de queue sur 12 000 tokens doit dire lesquels sont exactement sur la base.
5. **Avant 11 h 30** : resoumettre le meilleur CSV en dernier.

Il reste au plus 12 créneaux entre 06 h 10 et 12 h 00 (un toutes les 30 min, le dernier à 11 h 40).

---

## 9. Règles de travail et pièges

- **Budget des 33 endpoints.** L'utilisateur a fixé dans cette session une limite de 60 réponses utilisées par endpoint.
  La session 1e9a indique environ 93 à 96 utilisées sur 19 endpoints (01 03 04 06 08 09 10 12 13 15 17 18 19 21 22 24 25 26 32)
  et 52 à 54 sur les 14 autres. Je n'ai pas revérifié ces chiffres auprès du serveur, et je ne sais pas si ce dépassement
  de la limite de 60 a été validé par l'utilisateur. **Vérifier le compteur du serveur et demander à l'utilisateur avant
  toute nouvelle collecte.** Lecture du compteur sans rien dépenser : `GET http://34.147.229.55/33-model-equality/v1/models`,
  champ `budget.used`, en-tête `X-API-Key`.
- **Clé d'API** : lue dans la variable d'environnement `SPRINTML_API_KEY` (ou `T2KEY` dans mes anciens scripts). Elle n'est pas
  dans les secrets enregistrés et n'est écrite dans aucun fichier ; la redemander à l'utilisateur si besoin. Ne jamais l'écrire.
- **Soumission** : l'utilisateur veut un score local calculé avant toute soumission, et pas de soumission si ce n'est pas
  mieux. Une seule session soumet à la fois (délai de 30 min partagé).
- **Mémoire** : VM de 31 Go, 8 cœurs. Un seul calcul Qwen 3B à la fois. Deux en parallèle ont saturé la mémoire à 00 h 30 :
  le noyau a tué un calcul et le service `containerd`, qui a redémarré seul.
- **Ne pas utiliser `pkill -f` ou `pgrep -f`** avec un motif présent dans la commande appelante : le shell se tue lui-même.
  Lancer par un petit script (`tail_run.sh`, `dec_run.sh`) et tuer par PID.
- **Vérifier les calculs oubliés** avec `ps` : un ancien `exp_llr.py` a tourné près de 3 h et ralenti les autres avant d'être arrêté à 05 h 59.
- **Messages directs entre sessions : refusés (HTTP 403).** Le canal qui marche est Git : chaque session écrit sur sa branche
  et lit celle de l'autre.
- Ne pas présenter un score local comme un score public, ni le score public comme le score final.
- Le test FP8 n'est fiable que sur les réponses longues ; le test d'arrêt que sur les réponses courtes.

---

## 10. Où sont les choses

Dépôt `talal95c/CISPA-Paris-`. Aucune branche n'est à fusionner.

| Quoi | Où |
|---|---|
| Branche de 1e9a (données, sondes, journal) | `devin/1795307020-task2-whitebox`, dossier `task2/loop/` |
| Sa passation et son journal | `RECAP_HANDOFF.md`, `NOTES.md`, `POUR_AUTRE_SESSION.md` |
| Réponses des endpoints | `data/ep_long.jsonl` (1 013 lignes), `data/ep_short.jsonl` (990), `data/cache_all.jsonl` |
| Réponses des références | `data/refs_all.jsonl` (1 149), `data/ref_long.jsonl` |
| Prompts | `probes_short.json`, `probes_long.json`, `probes_id.json` |
| CSV soumis et réponses du serveur | `sub/*.csv`, `sub/submissions.jsonl` |
| Fabriquer un CSV | `mkcsv.py out.csv "04 23 33|06 08 20 30|10 16 29 31"` |
| Branche de fca8 (cette passation, mes scripts) | `devin/1795313260-task2-fca8-share`, dossier `task2/loop/fca8/` |
| Mes notes | `PASSATION_FCA8.md` (ce fichier), `NOTE_FCA8.md`, `POUR_1E9A.md` |
| Test FP8 calibré | `exp_llr.py`, `exp_an2.py`, résultat `exp_br.txt` |
| Ajustement de décodage | `dec_fit.py`, `dec_an.py`, `dec_tail.py`, `dec_run.sh` |
| Test de queue | `tail_fit.py`, `tail_an.py`, `tail_run.sh` |
| Reposes pour le watermark | `ep_repeat.jsonl` (2 questions reposées aux 33 endpoints, non concluant) |

Sur la VM de fca88bbd (non versionné) :
- `/home/ubuntu/t2quick/other/task2/loop/` : copie de travail des fichiers de 1e9a et mes sorties (`tail/`, `dec/`, `wb/`).
- `/home/ubuntu/t2quick/share_wt/` : worktree Git de ma branche de partage.
- `/home/ubuntu/t2quick/env/bin/python` : Python avec torch et transformers.
- `/home/ubuntu/t2quick/official/` : scoreur et gabarit officiels ; `/home/ubuntu/t2quick/task2_pdf.txt` : énoncé.
- `/home/ubuntu/met/practice_answers.csv` : vérité des 9 références.
- Poids des modèles : cache Hugging Face (`~/.cache/huggingface/hub`).

Lancer un test de queue : `./tail_run.sh NOM MODELE THREADS data/ep_long.jsonl,data/cache_all.jsonl ep-xx,ep-yy` depuis
`task2/loop/` ; sortie dans `tail/NOM.npz`, journal dans `tail/NOM.log`.
