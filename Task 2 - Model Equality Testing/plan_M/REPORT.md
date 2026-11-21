# Plan M · Pipeline et validation hors ligne

**GO pour classer un duel plain/fp8 plausible sous un checkpoint identifié. NO-GO pour détecter seul la présence d’une fp8.** Aucun appel aux endpoints de référence ou notés, aucune soumission.

## Résultats

Le LLR compare, sur les mêmes prompts, la somme des log-probabilités fp8 moins bf16 des tokens visibles de chaque candidat. On compare ensuite les deux sommes. Les réponses sont échantillonnées indépendamment et les blocs utilisent des prompts disjoints. Seuil fixé avant le test : précision ≥ 0,60. Les répliques utilisent des poids arrondis au bf16, une émulation E4M3FN et du calcul fp32 sur CPU.

| Test local | Résultat | Verdict |
|---|---:|---|
| TinyStories-1M, test initial sur les IDs simulés, 6 × 64 tokens max | 10/10, marge médiane +87,71 nats | GO |
| TinyStories-1M, texte visible, 6 × 64 tokens max | 10/10, +87,78 nats | GO |
| Pythia-14M, texte visible, 6 × 64 tokens max | 10/10, +17,28 nats | GO |
| TinyStories-1M, bloc des 30 longues, 256 tokens max | 5/5, +1 187,32 nats | GO local |
| Pythia-14M, bloc des 30 longues, 256 tokens max | 5/5, +407,78 nats | GO local |
| Pythia-14M, réponses W8A8 scorées avec une réplique W8A16 | 20/20 sur 12 prompts × 64 tokens max | GO sur cet écart de schéma |
| Pythia-14M, témoin bf16/bf16, permutation bilatérale à 5 % | 0/20 alertes | Témoin négatif compatible |
| Pythia-14M, LoRA synthétique 0,3 / 0,7 / 1,5 contre bf16 | 1/10, 2/10, 0/10 LoRA prises pour une fp8 à p ≤ 0,05 | NO-GO pour le seuil de 5 % |

Le contrôle LoRA utilise la même perturbation aléatoire de rang 4 avec trois amplitudes. Les trois séries partagent les prompts. Le test par permutation suppose l’échangeabilité des deux réponses sous la nulle. Il peut rejeter cette nulle quand une LoRA est présente, d’où l’abandon de ce test comme détecteur fp8 autonome.

### Phase 0 sur réponses simulées

Les portails boîte noire utilisent la génération libre de SmolLM2-135M-Instruct fournie par R, d’après `phase_results.json`. Les portails white-box reprennent les expériences locales ci-dessus.

| Portail | Mesure et verdict |
|---|---|
| B2, nulle S–S | 100 paires textuelles à 20 items, distance q95 de 0,991. Zéro comparaison avec items récupérables. **Nulle textuelle uniquement.** |
| B2’, dispersion | À 20 items, 1/100 paires plain identiques et 10/10 greedy. **NO-GO strict**. À 12 items, 0/10 plain et 10/10 greedy. **GO pilote**. |
| B3, format et efficacité | À 20 items, 0/200 rafales plain complètes. À 12, 0/20. Hors format 100 % pour système et persona aux deux tailles. **NO-GO**, efficacité A/B non estimable. |
| B4, watermark | 0 item exploitable. **Non mesurable**. |
| Drapeaux | Système 5/10, persona 9/10, FP8 et LoRA 0/10 pour chaque variante. **NO-GO** au seuil conjoint. |
| B6, axe LoRA | Aucune rafale libre exploitable pour estimer le cosinus. **Non mesurable**. |
| B7, température | AUC 0,690, fausses alarmes 20 %. **NO-GO** pour le drapeau boîte noire. En white-box, T et top-p correctement retrouvés dans 4/4 pilotes de 12 réponses. Portail white-box sur variantes encore non validé. |
| R1, classement du plain | **Non validé** en boîte noire. En duel white-box plausible, voir les résultats contrôlés et le holdout DistilGPT2. |
| D4, ciblage FP8 | Aucun item de génération libre avec amplitude δ vérifiable. **Non mesurable**. |
| C, identification | Le bon checkpoint est premier dans 4/4 pilotes parmi deux checkpoints locaux connus. **Pilote local uniquement**. Checkpoint P et paramètres du serveur inconnus. |

## MCC sur cinq replays conditionnels

On projette les décisions plain/fp8 des cinq blocs disjoints sur deux familles de neuf endpoints. Les labels de famille, de déguisement et de LoRA sont imposés par la vérité terrain. La formule officielle (0,1 × MCC famille + 0,9 × MCC modèle) donne **médiane 1,000, q10 1,000, minimum 1,000**. Le repli qui rattache les deux candidats donne **0,760** avec les mêmes labels imposés.

## Contrôle de généralisation

**Résultat du holdout · 18/20 (90 %), seuil pré-fixé de 17/20 franchi.** Les deux erreurs et la marge médiane de +6,57 nats indiquent un signal plus faible sur ce checkpoint que dans les pilotes. Cette réplication couvre un nouveau checkpoint et un nouveau type de prompts dans un duel simulé où une FP8 est présente et la loi de décodage est connue.

Protocole fixé dans le commit `2329d33` avant scoring. Le troisième checkpoint est [DistilGPT2](https://huggingface.co/distilbert/distilgpt2/tree/2290a62682d06624634c1f46a6ad5be0f47f38aa). Les prompts explicatifs remplacent entièrement les histoires des premiers essais. Un bloc contient 12 prompts sur un seul sujet, distinct des sujets des 19 autres blocs. Les réponses visibles sont plafonnées à 64 tokens. Le score reste le signe du LLR W8A16 contre bf16, avec écrêtage à ±3, T=1 et top-p=1. On fixe le seuil de réplication à 17 blocs bien classés sur 20, avec une graine fixée à 3001. La probabilité binomiale d’atteindre 17/20 ou mieux si la vraie précision est 0,60 vaut environ 1,6 % sous l’hypothèse d’indépendance des blocs.

Les poids, la grille de décodage et la règle de classement sont gelés avant l’ouverture de ce holdout. Toute règle révisée réclame un autre checkpoint et un autre type de prompts. Le score public sert au contrôle du format.

## Rejeu intégral simulé

Cinq blocs de 18 endpoints × 100 appels simulés par split. Le développement utilise SmolLM2-135M-Instruct et TinyStories-33M. Le holdout utilise Pythia-14M et TinyStories-8M, avec des textes de prompts distincts du développement et un type de rafale ainsi que 23 clés A/B réservées. Chaque bloc utilise des tirages de réponse indépendants. Les prompts longs d’un split sont réutilisés entre ses blocs. Les rafales des rejeux utilisent le moteur A/B contraint de R. Les résultats de format en génération libre sont ceux de la phase 0.

Chaque cellule donne la médiane / q10 / minimum selon le score officiel par paires.

| Split et méthode | MCC famille | MCC modèle | Score combiné |
|---|---|---|---|
| Développement, simple / R / M avec duels renseignés | 0 / 0 / 0 | 0,19136 / 0,19136 / 0,19136 | 0,17223 / 0,17223 / 0,17223 |
| Holdout, simple | 0,01242 / 0,00497 / 0 | 0 / 0 / 0 | 0,00124 / 0,00050 / 0 |
| Holdout, R | 0,04025 / 0,00497 / 0 | 0 / 0 / 0 | 0,00402 / 0,00050 / 0 |
| Holdout, M avec duels renseignés | 0,01242 / 0,00497 / 0 | 0,19136 / 0,07655 / 0 | 0,17347 / 0,07604 / 0,01192 |

**NO-GO global** au seuil médiane ≥ 0,80 et q10 ≥ 0,60. La dernière ligne est conditionnée par les checkpoints, candidats et ancres plain fournis par le simulateur. Le LLR classe 10/10 duels au développement et 10/10 au holdout avec ces informations. La partition fusionne les deux familles au développement, déclenchant l’abstention dans 10/10 duels. Au holdout, elle sépare les membres de 6 duels sur 10 et le LLR n’ajuste que les 4 autres. Les répliques, températures et critères d’abstention sont figés avant l’évaluation assistée du holdout. Les réglages restent identiques après les mesures.

## Journal commun vers CSV

Le script accepte une ligne JSON par réponse avec `endpoint_id`, `prompt_id`, `prompt` et `response`. Une réponse vide ou `null` compte comme ligne renseignée, avec zéro contribution aux signatures. Le journal append-only de R est aussi accepté. `prompt_id` identifie le créneau du manifeste, y compris pour une réserve. Pour un export à quatre champs, une réserve qui répète un prompt est rattachée au premier créneau correspondant encore sans réponse, dans l’ordre du manifeste.

Le convertisseur réutilise les signatures, la partition et l’écriture CSV de R, avec le profil `simple` gelé par défaut. Le journal complet doit couvrir 33 endpoints pour `--target scored`, 9 pour `--target refs`. Pour les simulations utiliser `--target replay`. `--dry-run` valide le journal et son manifeste. Pour le protocole R, `--items` et `--manifest` servent à reprendre la version exacte du collecteur. `--allow-partial` accepte des créneaux manquants lorsque tous les endpoints attendus figurent déjà dans le journal R.

### Protocole de collecte actuel de l’équipe

Le format `endpoint,id,repeat,prompt,response` avec `F1`–`F4` et `L01`–`L29` est détecté automatiquement. `--allow-partial` se charge des endpoints et créneaux manquants. Le CSV comporte toujours les 33 IDs de la cible notée, avec un label individuel provisoire pour chacun des endpoints sans texte. La liste JSONL `--prompts` permet d’épingler les 33 textes collectés. F1 fournit les ancres de famille. Une réponse ambiguë peut être rattachée sur ses longues si elle est comparable à au moins deux ancres par famille candidate sur trois prompts partagés, pour au moins deux familles, et si les distances médianes sont séparées d’au moins 0,03 sur l’échelle cosinus. Sinon, le pipeline laisse la famille inconnue pour l’étape de partition. Les rafales et leurs estimateurs sont indisponibles pour ce protocole.

L’instantané reçu de l’équipe contient 32 réponses F1. Les déclarations identifiables donnent 20 Qwen, 10 Llama et 1 Anthropic. Une réponse est ambiguë et un endpoint est absent. Le CSV provisoire contient 33 labels modèle individuels. F1 renseigne l’identité déclarée, avec une granularité parfois limitée au fournisseur. Le regroupement par checkpoint exact et le repli par longues n’ont pas été validés sur un jeu étiqueté de ce protocole.

Depuis la racine du dépôt, récupérer la branche de la [PR R](https://github.com/talal95c/CISPA-Paris-/pull/4), puis pointer sur sa version testée, `09b00088616849cfa142bdce982e9360052836d1`.

```sh
git fetch origin devin/1795286204-task2-plan-r
git worktree add --detach ../cispa-plan-r 09b00088616849cfa142bdce982e9360052836d1
PLAN_R="../cispa-plan-r/Task 2 - Model Equality Testing/plan_R"
PLAN_M="Task 2 - Model Equality Testing/plan_M"
python "$PLAN_R/collect.py" --dry-run --target refs
python "$PLAN_R/collect.py" --dry-run --target scored
python "$PLAN_M/submit_offline.py" journal.jsonl --shared-dir "$PLAN_R" --dry-run
python "$PLAN_M/submit_offline.py" journal.jsonl --shared-dir "$PLAN_R" --output submission.csv
python "$PLAN_M/submit_offline.py" answers.jsonl --shared-dir "$PLAN_R" \
  --prompts wave1_prompts.jsonl --allow-partial --output submission_team_provisional.csv
```

Une configuration `--duels` est facultative. La liste JSON contient, pour chaque famille confirmée, deux `candidates` plausibles plain/FP8, un `plain_anchor` équivalent au plain déjà identifié, un `checkpoint` local, le `prompt_format` (`raw`, `chat` ou `user-assistant`), `temperature`, `top_p` et éventuellement `scheme` (`w8a16` par défaut). Le classement demande 12 longues distinctes appariées. Le pipeline s’abstient lorsque les trois endpoints sont séparés, lorsque plusieurs duels tombent dans la même famille inférée ou en l’absence de marge. L’équipe confirme le checkpoint et l’ancre avant d’activer ce canal. Les fonctions `checkpoint_evidence` et `fit_decode` sont disponibles pour effectuer cette vérification sur un jeu de calibration séparé.

## Rejouer et intégrer

Prérequis testés sous Python 3.12 : PyTorch 2.6.0 CPU, Transformers 4.51.3 et les modèles téléchargés au préalable. Révisions HF utilisées : [TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M/tree/77f1b168e219585646439073245fe87e56b3023e) et [Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m/tree/cf967c0a9a04383db6f7b1108d86b2962634b4ac).

```sh
python -m unittest -v test_whitebox test_journal
python run_fp8_gate.py --model /chemin/tinystories-1m
python run_fp8_gate.py --model /chemin/tinystories-1m --visible-text --seed 1013 \
  --blocks 5 --prompts 30 --tokens 256 --threads 2 --batch-size 24 --output tiny.json
python run_fp8_gate.py --model /chemin/pythia-14m --visible-text --seed 1009 \
  --blocks 5 --prompts 30 --tokens 256 --threads 2 --batch-size 24 --output pythia.json
python score_replays.py tiny.json pythia.json
python run_fp8_gate.py --model /chemin/distilgpt2 --prompt-family expository \
  --visible-text --seed 3001 --blocks 20 --prompts 12 --tokens 64 \
  --threads 6 --batch-size 24 --output holdout.json
python score_text.py --model /chemin/checkpoint --pairs duels.jsonl
```

Depuis la racine du dépôt, avec les archives de rejeu R et les checkpoints locaux indiqués dans `replay_offline.py`.

```sh
python "$PLAN_M/replay_offline.py" --shared-dir "$PLAN_R" --split development \
  --checkpoint-root /chemin/models replay1.jsonl replay2.jsonl
python "$PLAN_M/replay_offline.py" --shared-dir "$PLAN_R" --split holdout \
  --checkpoint-root /chemin/models holdout3.jsonl holdout4.jsonl
```

Chaque ligne de `duels.jsonl` contient les chaînes `prompt`, `left` et `right`. Un LLR positif favorise la réplique fp8 à gauche. Le CLI écarte les paires avec une réponse vide et indique leur nombre. Si toutes les paires sont écartées, le score vaut `null`. L'appelant conserve les mêmes prompts et le même décodage pour les deux candidats. En présence d'un template de chat, reproduire le prompt tokenisé, avec le séparateur avant la réponse. C conserve son repli si le checkpoint ou le duel est incertain.

Le collecteur R a passé trois tests en dry-run et sur simulateur HTTP de boucle locale, couvrant la reprise, les réserves et le délai. L’export a été contrôlé sur 1 800 réponses simulées puis sur 3 300 lignes de test avec les 33 IDs officiels.
