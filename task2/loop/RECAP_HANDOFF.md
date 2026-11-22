# Tâche 2 « Model Equality » — document de passation complet

Rédigé par la session Devin `1e9a3913b14a4521ae9db6d08b5b27a7`, mise en pause le 22/11/2026 à 06 h 00 CET.
Rien ne tourne : aucun calcul, aucune collecte, aucune soumission programmée.

| | |
|---|---|
| Échéance des soumissions | 22/11/2026 12 h 00 CET |
| Meilleur score public | **0,70** (soumission n° 3136, 05 h 39 CET) |
| CSV correspondant | `task2/loop/sub/probe_s7_sys08.csv` |
| Dépôt, branche | `talal95c/CISPA-Paris-`, `devin/1795307020-task2-whitebox` |
| Dossier de travail | `task2/loop/` (journal détaillé : `NOTES.md`) |
| Prochain créneau de soumission | à partir de 06 h 10 CET, puis toutes les 30 min |

---

## 1. La tâche et ses règles

33 endpoints anonymes `ep-01` à `ep-33` servent des modèles de langue. Il faut rendre un CSV `endpoint,family,model` qui dit, pour chaque
endpoint, sa famille (modèle de base) et son modèle exact. 9 endpoints de référence `ref-01` à `ref-09`, dont la variante est connue, servent à
valider les méthodes.

- **Même modèle** = même checkpoint + même quantification + même adaptateur LoRA.
- **Ne changent pas le modèle** : température, top-k/top-p, prompt système, persona, watermark. Ces endpoints doivent rester groupés avec l'original.
- **Score** = 10 % famille + 90 % modèle. Chaque part est un MCC sur des paires d'endpoints ; la part modèle ne compte que les 166 paires
  internes à une famille (45 en A, 66 en B, 55 en C). Avec les familles justes : score = 0,1 + 0,9 × MCC modèle.
- Le score public ne porte que sur **une partie des paires** et il est arrondi à 0,05. Le classement final utilise les autres paires. Score caché inconnu.
- 4 équipes sont à 1,00 public : une séparation nette existe.

Contraintes de l'API :
- 100 réponses maximum par endpoint noté, 4 requêtes simultanées pour toute l'équipe, 20 à 30 s par réponse longue.
- La requête n'accepte que `model`, `messages`, `max_tokens`. `temperature` et `seed` sont ignorés, les messages système sont refusés.
- 30 minutes entre deux soumissions, pour toute l'équipe.
- Le serveur expose des routes `/admin/*` : elles sont aux organisateurs, ne pas y toucher.

Adresses :
- collecte : `http://34.147.229.55/33-model-equality/v1/chat/completions`
- liste des modèles : `http://34.147.229.55/33-model-equality/v1/models`
- soumission : `POST http://34.147.229.55/submit/33-model-equality` (en-tête `X-API-Key`, champ `file`)
- classement : `http://34.147.229.55/leaderboard_page`

La clé d'API est dans `~/.config/cispa/env` (variable `SPRINTML_API_KEY`) sur la machine de la session 1e9a. Elle n'est pas dans le dépôt :
un autre agent doit la recevoir de talal par le champ sécurisé.

---

## 2. Ce qu'on a fait, dans l'ordre

1. **Méthodes « texte » (boîte noire).** Distances entre réponses, TF-IDF, MMD, k-NN, k-means. Meilleur score local 0,373 sur les références,
   0,10 public sur cinq soumissions. Abandonné : ces méthodes confondent température et poids.
2. **Identification des poids ouverts.** Les endpoints se présentent comme Qwen et Llama. En comparant les réponses en cache aux prédictions
   des checkpoints publics, on a trouvé les quatre modèles (section 3). C'est le tournant : on peut calculer la probabilité exacte de chaque mot
   de chaque réponse sous les vrais poids.
3. **Mesures « boîte blanche » validées sur les références** (section 4) : température apparente, fp8 simulée, probabilité d'arrêt (LoRA),
   watermark par répétition, prompt système par recalcul.
4. **Collectes** : 30 questions courtes sur les 33 endpoints, puis des longues réponses (jusqu'à 4 répétitions de 12 questions) sur les
   19 endpoints qui ressemblent à l'original, avec l'accord de talal pour monter à environ 93-96 réponses sur ceux-là.
5. **Sondes sur le classement** : une hypothèse écrite à l'avance par soumission, un seul changement à la fois (section 5). Score passé de
   0,10 à 0,70.
6. **Coordination** avec la session `fca88bbd4c3e469cb3a4b990d815bc13` (même tâche, même utilisateur) : les messages directs sont refusés
   (HTTP 403), on échange par `POUR_AUTRE_SESSION.md` sur la branche et par les fichiers que talal recopie. Elle ne collecte pas et ne soumet pas ;
   elle calcule sur nos données (8 cœurs). Ses résultats sont repris ici.

Demandes de talal non réalisées :
- **GPU JURECA** : aucun calcul n'y a tourné. Le nœud de l'équipe (`arifette2`, dossier Teletubbies) était occupé, Slurm ne laisse que le
  propriétaire d'un job y ajouter une commande, et les fenêtres MFA du compte `aamir1` se sont fermées. Tout a été calculé sur le processeur
  de la machine (8 cœurs, 31 Go). Ce n'est pas bloquant : les quatre modèles tiennent en mémoire, un gros à la fois.
- **AIDE / MLE-bench** : non branchés, les clés OpenAI et Mistral enregistrées sont refusées (401). À la place : la même méthode à la main
  (journal d'essais, hypothèse avant chaque essai, correction des échecs, amélioration du meilleur).

---

## 3. Les modèles

| Famille | Checkpoint ouvert | Endpoints |
|---|---|---|
| A (10) | `Qwen/Qwen2.5-3B-Instruct` | 01 04 12 14 17 18 23 24 25 33 |
| B (12) | `Qwen/Qwen2.5-1.5B-Instruct` | 02 03 06 07 08 20 21 22 26 28 30 32 |
| C (11) | `unsloth/Llama-3.2-1B-Instruct` | 05 09 10 11 13 15 16 19 27 29 31 |
| Références | `HuggingFaceTB/SmolLM2-360M-Instruct` | ref-01 à ref-09 |

Copies locales dans `/home/ubuntu/models/` (nom avec `__` à la place de `/`). Les familles sont considérées comme sûres : toutes les
soumissions ont la part famille au maximum.

Les 9 références couvrent : original, prompt système, persona, watermark (ref-06), température 1,0 (ref-08), fp8 (ref-02), LoRA de force
0,3, 0,7 et 1,5. **Hypothèse de travail** : chaque famille notée suit le même schéma, soit un groupe « poids d'origine » d'environ 5 à 7
endpoints (original, prompt système, persona, watermark, température 1,0, tirage tronqué) et des endpoints isolés (fp8, plusieurs LoRA,
peut-être autre chose). Les sondes S3, S5 et S7 la soutiennent : environ 34 vraies paires au total.

---

## 4. Les mesures, et ce que valent leurs signaux

Toutes partent de `score_wb.py` : il rejoue chaque réponse sous un checkpoint et enregistre la log-probabilité de chaque mot.

| Signal | Outil | Validé sur les références | Résultat sur les endpoints notés |
|---|---|---|---|
| Température apparente | `wb_table.py`, `wb_feat.py` | oui : original 0,69-0,72, ref-08 à 1,01 | défaut ≈ 0,70 ; ≈ 1,0 pour 33, 30, 31 |
| fp8 (base contre fp8 simulée, format E4M3 par tenseur) | `wb_llr.py X_long X_long_fp8` | oui sur les longues réponses seulement (ref-02 seule négative) | ep-15 : −6,6 ± 0,7 ; ep-22 : −3,45 ± 1,26 ; tous les autres positifs |
| LoRA par probabilité d'arrêt | `wb_eos.py` | oui : −6 à −7 (0,3), −11 (0,7), −18 à −21 (1,5), autres −1,6 à +2,8 | aucun signal stable ; le faux positif ep-08 venait de son prompt système |
| Watermark par répétition (mêmes premiers mots quand on repose la question) | `rep_prefix.py` | oui : ref-06 jusqu'à 130 mots identiques, autres ≤ 12 | ep-04, ep-06, ep-10 : 116 à 187 mots, autres ≤ 26 |
| Prompt système (recalcul sous le prompt récupéré) | `score_wb.py --system`, `wb_llr.py` | non | ep-29 le récite ; ep-08 seul B à le préférer (−3,2 ± 2,8, autres +10 à +21) ; aucun A |
| Tirage tronqué (mots de rang ≥ 50 absents) | `wb_trunc.py` | non | ep-07 : 0 sur 4 757 ; ep-27 : 2 ; ep-14 : 3 ; ep-11 : 4 (top-k seul) |
| Ajustement exact T × top-k × top-p | `score_trunc.py`, `tr_fit.py` | non, et témoins mal ajustés (pas de 0,1) | ep-07 = T 1,0, top-k 50, top-p 0,95 ; ep-27 = T 0,9, top-k 40-50 ; ep-11 = T 0,8, top-k 40 |
| Écart de température haute/basse entropie (dT) | `wb_dt.py` | faible : LoRA 0,7 = +0,08, 1,5 = +0,19, bruit 0,02-0,04 | ep-02 +0,08, ep-21 +0,06, ep-28 +0,06, ep-14 +0,05 |
| Empreinte de fréquences de mots | `score_uni.py`, `uni_*.py`, `tokfreq.py` | LoRA fortes et watermark seulement | rien d'exploitable, abandonné |
| int8 et nf4 simulées (famille C) | `score_wb.py --variant` | impossible (pas de référence) | nf4 exclue en C ; int8 interrompu avant la fin (mémoire) |

Prompt système récupéré sur ep-29 :
> You are a helpful assistant for general questions and everyday tasks. Ask a clarifying question when the request is ambiguous. When you give an answer, say briefly why.

Pièges rencontrés :
- La fp8 ne se voit que sur les longues réponses ; la LoRA par arrêt ne se voit que sur les questions à réponse d'un mot.
- Les LoRA 0,7 et 1,5 de référence ressemblent à l'original à température plus haute : un bon ajustement à T 0,8 ne prouve pas des poids d'origine.
- Ne jamais comparer deux endpoints sur des jeux de questions différents (les 19 « longs » contre les autres) : cela crée de faux blocs.
- Le score local sur les références (0,72 à 1,00) a été réglé sur ces mêmes références : optimiste, et il ne dit rien des 33 notés.

---

## 5. Soumissions (toutes dans `sub/submissions.jsonl`, heures en UTC dans ce fichier)

| n° | CSV | Groupes (le reste seul) | Public | Ce que ça a montré |
|---|---|---|---|---|
| 1908, 2001, 2141, 2312 | méthodes texte et gros blocs | un bloc « base » par famille, ou {01 04 12} {06 21 22} {10 13 19} | 0,10 | les endpoints qui ressemblent le plus à l'original ne sont pas tous le même modèle |
| 2426 | `wb_rule_v1b` | blocs de base, 15 / 22 / 08 seuls | 0,20 | isoler les fp8 aide |
| 2538 | `probe_s1` | A{14 23 33} B{02 07 20 28 30} C{05 11 16 27 29 31} | 0,45 | les déguisés partagent des poids |
| 2641 | `probe_s2_noC` | S1 sans le groupe C | 0,35 | le groupe C contient de vraies paires |
| 2732 | `probe_s3_core` | A{23 33} B{20 30} C{16 29 31} | 0,45 | noyau sûr ; environ 30 vraies paires au total |
| 2829 | `probe_s4_trunc` | S3 + 07 11 14 27 | 0,45 | une partie seulement des « tronqués » est juste |
| 2930 | `probe_s5_wm` | S3 + 04 06 10 | 0,60 | les trois watermarks ont les poids d'origine |
| 3030 | `probe_s6_trunc3` | S5 + 14 07 27 | 0,60 | même lecture que S4 |
| 3136 | `probe_s7_sys08` | S5 + 08 | **0,70** | ep-08 = poids d'origine + prompt système |

---

## 6. État des 33 endpoints

| Endpoint | Famille | Statut | Dans le CSV 0,70 | Preuve |
|---|---|---|---|---|
| 04 | A | watermark, poids d'origine | groupe A | répétition + S5 |
| 23 | A | persona, poids d'origine | groupe A | S3 |
| 33 | A | température ≈ 1,0, poids d'origine | groupe A | ajustement T + S3 |
| 14 | A | tirage tronqué probable (T 0,85) | seul | rangs ; S4 et S6 sans gain |
| 24 | A | à part (T 0,726, ajustement imparfait) | seul | non tranché |
| 17, 25, 12 | A | ajustement parfait à T 0,69 : candidats « original » | seuls | non testés |
| 18, 01 | A | ajustement imparfait (profil fp8 / LoRA 0,3 / prompt système) | seuls | non tranché ; fp8 exclue |
| 06 | B | watermark, poids d'origine | groupe B | répétition + S5 |
| 08 | B | prompt système, poids d'origine | groupe B | recalcul + S7 |
| 20 | B | persona, poids d'origine | groupe B | S3 |
| 30 | B | température ≈ 1,0, poids d'origine | groupe B | ajustement T + S3 |
| 22 | B | **fp8** | seul | LLR −3,45 ± 1,26 ; test calibré de fca88bbd |
| 07 | B | tirage tronqué (T 1,0, top-k 50, top-p 0,95) | seul | ajustement exact ; S4 et S6 sans gain |
| 02 | B | T ≈ 0,94, queue trop courte pour une température pure | seul | décodage ou LoRA, non tranché |
| 28 | B | T ≈ 0,80, ajustement propre | seul | température ou LoRA, non tranché |
| 26, 32 | B | ajustement parfait à T 0,69 : candidats « original » | seuls | non testés |
| 03, 21 | B | ajustement imparfait (03 : T 0,731, dT −0,04 ; 21 : dT +0,06) | seuls | non tranché ; fp8 exclue |
| 10 | C | watermark, poids d'origine | groupe C | répétition + S5 |
| 16 | C | persona, poids d'origine | groupe C | S2, S3 |
| 29 | C | prompt système (récité), poids d'origine | groupe C | S2, S3 |
| 31 | C | température ≈ 1,0, poids d'origine | groupe C | ajustement T + S3 |
| 15 | C | **fp8** | seul | LLR −6,6 ± 0,7 |
| 27 | C | tirage tronqué (T 0,9, top-k 40-50) | seul | ajustement exact ; S4 et S6 sans gain |
| 11 | C | top-k seul (T 0,8, top-k 40) | seul | ajustement exact moyen ; S4 sans gain |
| 05 | C | T ≈ 0,80, ajustement propre | seul | température ou LoRA, non tranché |
| 13 | C | ajustement parfait à T 0,69 : candidat « original » | seul | non testé |
| 09, 19 | C | ajustement imparfait ; 09/13/19 partagent des débuts de réponse | seuls | non tranché |

Lecture des scores publics par `lb_design.py` (probabilité d'appartenir au groupe d'origine, à ne pas sur-interpréter car le score ne
couvre qu'une partie des paires) : 07 : 0,67 ; 11 : 0,63 ; 14 : 0,59 ; 02 et 28 : 0,54 ; 27 : 0,38 ; 05 : 0,37 ; chacun des 13 autres : 0,03 à 0,13.

---

## 7. Objectifs pour la suite

1. **Trouver l'endpoint « original » de chaque famille.** C'est le gain le plus probable : un endpoint ajouté à un groupe juste de 3 ou 4
   apporte 3 ou 4 vraies paires. Candidats : A 17, 25, 12 ; B 26, 32 ; C 13.
2. **Trancher les tirages tronqués** 07, 27, 11, 14 : un réglage de décodage ne change pas le modèle, donc s'ils s'ajustent sur les poids
   d'origine ils rejoignent le groupe.
3. **Trancher « température ou LoRA »** pour 02, 28, 05, 24.
4. **Trouver l'endpoint à prompt système de A** (le prompt d'ep-29 ne convient à aucun A : il a peut-être un autre prompt).
5. **Regrouper entre eux les endpoints modifiés identiques**, s'il y en a (deux endpoints de la même LoRA, par exemple) : aujourd'hui tous
   les endpoints hors groupe sont seuls, ce qui perd ces paires.
6. Finir avec le meilleur CSV soumis en dernier avant 12 h 00.

---

## 8. Plan proposé

Il reste environ 11 créneaux (06 h 10, 06 h 40, …, 11 h 40), partagés avec toute l'équipe. Garder le dernier pour le meilleur CSV.

**Étape 1, sans appel ni soumission (30 à 45 min de calcul)**
- Passer `score_trunc.py` sur les références (`data/refs_all.jsonl`, `data/ref_long.jsonl`, modèle SmolLM2-360M) avec une grille de
  température au pas de 0,02. Critère : l'original, le prompt système, le persona, le watermark et T 1,0 doivent sortir à |z| < 3, les LoRA en dehors.
- Si le test passe, le relancer sur B, C puis A (un modèle à la fois) pour 07, 27, 11, 14, 02, 28, 05, 24.

**Étape 2, sondes à un changement (une par créneau, hypothèse et score attendu écrits avant)**

| Ordre | Changement par rapport au CSV 0,70 | Si vrai | Si faux |
|---|---|---|---|
| S8 | + 13 dans C (candidat original) | environ +0,05 à +0,10 | environ −0,05 |
| S9 | + 26 dans B (sinon 32) | idem | idem |
| S10 | + 17 dans A (sinon 25, puis 12) | idem | idem |
| S11 | + 07 dans B, si l'étape 1 le soutient | idem | idem |
| S12 | + 27 puis + 11 dans C, si l'étape 1 les soutient | idem | idem |

Règles : garder un changement seulement s'il monte le score **et** qu'une mesure sur les poids le soutient ; repartir du meilleur CSV après
chaque sonde ratée ; un écart de 0,05 seul (l'arrondi) ne décide rien.

**Étape 3, s'il reste du budget d'appels**
- Les 14 endpoints peu sollicités (02 05 07 11 14 16 20 23 27 28 29 30 31 33) ont encore environ 46 appels chacun. Reposer 2 à 3 fois les
  12 questions longues à 02, 28, 05, 14, 07, 27, 11 donnerait la précision qu'ont déjà les 19 autres (fp8, dT, répétitions).
- Pour le prompt système de A : poser à chaque A une question d'extraction différente de F2 (2 à 3 appels, seulement sur 14, 23, 33 ;
  les sept autres A n'ont plus que 4 à 7 appels).

**Étape 4** : resoumettre le meilleur CSV au dernier créneau avant 12 h 00.

---

## 9. Budget d'appels restant

| Endpoints | Réponses utilisées | Reste |
|---|---|---|
| 01 03 04 06 08 09 10 12 13 15 17 18 19 21 22 24 25 26 32 | environ 93 à 96 | 4 à 7 |
| 02 05 07 11 14 16 20 23 27 28 29 30 31 33 | environ 52 à 54 | environ 46 |

Ces chiffres comptent nos collectes et les répétitions de la session fca88bbd. Vérifier avant toute collecte ; `epcollect.py` prend un
plafond `--cap`.

---

## 10. Reprendre le travail

Données dans le dépôt (`task2/loop/data/`) : `ep_long.jsonl` (1 013 longues réponses), `ep_short.jsonl` (990 courtes), `cache_all.jsonl`
(448 réponses notées + 198 de référence), `refs_all.jsonl`, `ref_long.jsonl`, `fca8/ep_repeat.jsonl` (répétitions de l'autre session).
Questions : `probes_long.json`, `probes_short.json`, `probes_id.json`.

**Pas dans le dépôt** (664 Mo, seulement sur la machine de la session 1e9a, dossier `task2/loop/wb/`) : les log-probabilités par mot.
Sur une autre machine il faut télécharger les quatre checkpoints et relancer `score_wb.py` : environ 14 s par longue réponse pour A,
7 s pour B, 6 s pour C, le double avec la fp8.

Commandes :
```bash
cd task2/loop
PY=/home/ubuntu/ref-evaluation/met-paper-env/bin/python          # torch + transformers
# log-probabilités sous un checkpoint -> wb/NOM.jsonl (reprend où il s'est arrêté)
./runq.sh 4 b_long /home/ubuntu/models/Qwen__Qwen2.5-1.5B-Instruct data/ep_long.jsonl --endpoints ep-07,ep-02
./runq.sh 4 b_long_fp8 /home/ubuntu/models/Qwen__Qwen2.5-1.5B-Instruct data/ep_long.jsonl --variant fp8
$PY wb_llr.py b_long b_long_fp8          # LLR base - variante par 1000 mots, avec erreur bootstrap (négatif = la variante explique mieux)
$PY wb_table.py b_long b_long_fp8        # température, excès de top-1, dispersion, queue, arrêt
$PY score_trunc.py --model M --data data/ep_short.jsonl,data/ep_long.jsonl,data/cache_all.jsonl --out tr/x.pkl --endpoints ep-07
$PY tr_fit.py tr/x.pkl                   # meilleur réglage (T, top-k, top-p) et écart d'ajustement
python3 mkcsv.py sub/s8.csv "04 23 33|06 08 20 30|10 13 16 29 31"   # CSV par groupes, le reste seul
set -a; . ~/.config/cispa/env; set +a; python3 submit.py sub/s8.csv # soumission, réponse ajoutée à sub/submissions.jsonl
$PY lb_design.py 0.05 "13 26 32 17" 12   # lecture des scores publics ; ajouter chaque nouvelle sonde dans lb_post.py
```

Précautions machine : 31 Go de mémoire, donc un seul calcul Qwen 3B à la fois et deux calculs au plus ; ne jamais utiliser `pkill -f`
avec un motif qui figure dans la commande elle-même.

Avant chaque soumission : lire `sub/submissions.jsonl` (délai de 30 min) et vérifier que l'autre session ou un coéquipier ne prend pas le créneau.
