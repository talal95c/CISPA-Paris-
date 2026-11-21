# Agents de recherche — Task 2

Le rôle `task2-famou` prépare et évalue des hypothèses ; `task2-aide` part du meilleur programme et explore des variantes. Les deux utilisent le même arbitre. Aucune commande ci-dessous ne collecte de réponses et aucune ne soumet de CSV. L'installation seule nécessite Internet.

## Installation et pilote CPU

Depuis ce dossier, avec `git`, `uv`, Python 3.12 et un compilateur C/C++ :

```sh
bash install.sh
mkdir -p runs
.harness/famou-env/bin/python pack_famou.py runs/famou-01
.harness/famou-env/bin/famou-ctl test -c runs/famou-01/config.yaml --json
.harness/aide-env/bin/python aide_cpu.py --famou-candidate runs/famou-01/init.py \
    --gguf /path/to/coder.gguf --steps 3 --output runs/finalist.json
```

L'installation fixe les révisions des deux dépôts, installe le SDK Famou et le cœur AIDE dans des environnements distincts. `HARNESS_HOME` permet de changer leur emplacement. Le client Famou valide et note le problème localement ; l'évolution hébergée chez Baidu est **distincte** et nécessite un compte, une URL et une clé Famou. Ne pas y envoyer le dépôt, le corrigé de contrôle, les réponses réelles ou la clé du concours.

Le moteur local utilise réellement `Agent.search_policy`, `Journal`, `Node` et `MetricValue` d'AIDE. Le générateur GGUF et la grammaire remplacent son interpréteur Python libre et son évaluateur LLM. Modèle du pilote : [Qwen2.5-Coder-1.5B-Instruct Q4_K_M](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF), SHA256 `cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046`. Aucun GPU ni serveur LLM requis.

## Ce que mesure le pilote

Le fichier de développement contient uniquement les distances agrégées et les groupes anonymisés des simulations Plan R `replay1` et `replay2` (cinq blocs par famille). Le proposant GGUF ne reçoit que les noms des signaux, les formules tentées et les scores. La correction n'est pas secrète vis-à-vis d'une personne lisant le dépôt ; une expression candidate, elle, ne peut ni ouvrir un fichier, ni modifier le juge.

Pour chaque expression : regrouper par liaison moyenne, choisir un seuil sur la première famille et mesurer le MCC modèle sur la seconde, puis inverser les rôles. Maximiser le pire des deux MCC. Une fois l'expression choisie, geler un seul seuil ajusté sur les deux familles. Le `combined_score` demandé par Famou remet ce MCC entre 0 et 1 : **0,5 correspond à un MCC nul**. Le score de famille n'est pas estimé par ces simulations conditionnelles ; le score du concours est `0,1 × MCC famille + 0,9 × MCC modèle`.

## Contrôle par le coordinateur

Avant d'ouvrir le contrôle, comparer avec `python control.py N` : `N` est le même budget de propositions, **y compris la base**, les doublons et les essais invalides. Geler le gagnant sur le développement ; à égalité, préférer la méthode simple. `--interactive` permet à l'agent appelant de proposer lui-même les expressions sans GGUF.

```sh
python features.py /path/to/plan_r_journals.tgz runs/held.json holdout3 holdout4
python check_candidate.py runs/finalist.json runs/held.json \
    /path/to/refs_same_prompts.jsonl /path/to/practice_answers.csv
```

Geler le finaliste **avant** ce contrôle. Critère préalable à une soumission : dépasser à la fois zéro et le MCC du point de départ `lex` sur **chacune** des deux familles simulées et la deuxième génération des références réelles (L01–L09). Les familles simulées ont servi à des expériences précédentes ; ce n'est pas une preuve de généralisation sur des checkpoints inédits. Après un échec, ne pas optimiser sur ces scores : il faut un nouveau jeu de contrôle pour une nouvelle campagne.

Si le critère passe, `--real-answers journal.jsonl --previous-csv precedent.csv --write-csv nouveau.csv` applique le regroupement aux **réponses déjà en cache** avec les familles estimées, vérifie les 33 identifiants et refuse une partition inchangée. Soumettre seulement après le délai d'équipe de 30 minutes. Conserver le retour de cette soumission : le leaderboard n'affiche que le meilleur score public, pas nécessairement celui du dernier essai.
