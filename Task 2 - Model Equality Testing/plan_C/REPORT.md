# Task 2 — bilan local et collecte

## Conclusion

**C ne bat pas R en simulation → C = R, plan C non retenu.**
Le score simulé de la compétition mixte est d'environ **0,3**, bien en dessous
des **~0,68 estimés pour R dans le plan**. Même le « plancher R » n'est donc pas
validé. Ces simulations ne justifient pas une soumission automatique.

Aucune requête HTTP vers la compétition, aucun téléchargement pendant ces
deux dernières étapes. Le mode live décrit ci-dessous **n'a jamais été lancé**.
Ses tests utilisent un transport factice, sans socket réseau.

## Ce qui est fait

- Simulateur CPU : 3 bases × 9 variantes ; banque seedée de 300 items, caches
  teacher-forcing, FP8 poids, perturbations low-rank, sys/persona, KGW, T=1.
- Décision : estimateur U non biaisé de distance L2², standardisé par la nulle ;
  familles par complete-linkage ; pool S, règles R ; classement/cross-fit C.
- Parseur de lignes `q7k Yes`, catégorie OTHER pour réponse manquante/malformée,
  lecture `answers.jsonl`, export `endpoint,family,model` avec `infer.py`.
- Réglage **135M A seulement**, type « choix de mot » exclu. Comparaison 60/125
  items admissibles, choix de 60 ; pas d'emprunt d'items B pour comparer à 300.
  Paramètres gelés et commités dans **9b44cfa avant l'évaluation holdout**.
  Aucun re-réglage depuis. Les TV globales de l'étape 1 avaient déjà été vues :
  ce gel concerne la décision de l'étape 2, pas une session entièrement aveugle.
- Évaluation : 200 rejeux indépendants, compositions 7–9 variantes/famille,
  plain présent ; 360M/Qwen B hors réglage, puis type entier exclu séparément.
- `collect.py` : protocole 100 prompts, dry-run par défaut, mock local,
  autorisation live explicite, budget, journal complet et reprise sécurisée.

## Portails gelés

| Portail | Mesure sur 135M A | Verdict |
|---|---|---|
| B2 nulle U | moyenne 0,000384 ; SD 0,005201 ; q95 0,009306 | GO |
| B4 watermark/code | détection 37 % ; FPR variantes ≤4 % | NO-GO, coupé |
| Sys/persona | q=0,9 et FPR=5 % fournis par simulation | GO nominal à q=0,9 ; coupé à q=0,7 |
| B6 axe LoRA | 0/20 tirages passent les critères cosinus | NO-GO, coupé |
| B7 température | AUC 0,999675 ; FPR variantes ≤1,5 % | GO réglage, **échec transfert** |
| R1 classement plain | 81 % contre FP8 ; 49,5 % contre L0,3 ; 46 % en triple | NO-GO global ; C=R |
| D4 items FP8 | 23 items à écart >0,05 | GO diagnostic seulement, pas un portail white-box |

Le seuil température gelé atteint **24,5 % de faux positifs sur Qwen FP8**,
18 % sur Qwen L1,5, 13 % sur 360M L1,5, malgré une AUC de 1. Nous n'avons pas
retiré cette brique après observation pour améliorer artificiellement le
score holdout : la méthode gelée est déclarée non robuste. Toute nouvelle
méthode nécessiterait une nouvelle validation indépendante.

## Scores gelés, q=0,9

| Données | R médiane / q10 | C médiane / q10 | Écart médiane au réglage |
|---|---:|---:|---:|
| Réglage 135M A | 0,3317 / 0,0271 | 0,3317 / 0,0271 | — |
| Holdout 360M B | 0,2127 / 0,0000 | 0,2127 / 0,0000 | −0,1189 |
| Holdout Qwen B | 0,2324 / 0,0000 | 0,2324 / 0,0000 | −0,0993 |
| Holdout 360M + Qwen B | 0,3000 / 0,0902 | 0,3000 / 0,0902 | −0,0316* |
| Type exclu, 360M + Qwen | 0,2591 / 0,0733 | 0,2591 / 0,0733 | −0,0726* |
| Compétition mixte 3 bases B | 0,3067 / 0,1375 | 0,3067 / 0,1375 | diagnostic mixte |

*Sur une seule famille, MCC_family=0 par convention du scoreur ; comparer un
score multi-famille ajoute donc un crédit famille différent. Les écarts de
médiane **MCC_model** sont −0,1322 (360M), −0,1103 (Qwen), −0,1256 (holdout
joint). La compétition mixte inclut 135M et n'est pas un holdout de famille pur.

À q=0,7, les drapeaux sys/persona sont coupés par le portail de sensibilité :
score médian nul sur les expériences mono-famille ; 0,0873 sur le holdout
joint (crédit familles). Table complète des deux q : `results_phase0.md`.

### Diagnostic final : uniquement 135M A, aucun nouveau seuil

| Diagnostic | Résultat |
|---|---|
| R avec les 4 drapeaux oracle parfaits | médiane **0,5970**, q10 **0,3273** |
| Tout en singletons sauf familles vraies, 135M A | **0,0000** |
| Oracle parfait, contrôle scoreur à deux familles synthétiques | **1,0000** |

Sur une famille unique, l'oracle parfait vaut 0,9 et non 1 par convention
MCC_family. Les drapeaux parfaits font gagner 0,2653 à la médiane de R, mais
la décision/clustering garde une perte importante par rapport à ce plafond.
Le contrôle à deux familles est une fixture du scoreur, pas de nouvelles
données holdout. Diagnostic reproductible : `python diagnostic_oracle.py`.

## Pistes abandonnées et limites

- **Axe LoRA** : aucun succès joint au portail B6 ; projection/rank-sum coupés.
- **Classement** : succès contre FP8 seul, pas contre L0,3 ni en triple ; aucun
  gain C revendiqué et pas de sélection opportuniste sur le holdout.
- **Watermark par dépendance au code** : 37 % de sensibilité sur 22 probes
  réellement recalculées ; arrêt avant toute passe modèle holdout. Le plain
  varie aussi avec le code, et les chiffres forcent un espace prédécesseur.
- **Seuil T** : faux positifs excessifs au transfert, malgré l'AUC élevée.
- **Format rafale 135M** : cinq générations libres de 20 items ont donné
  zéro ligne code+option valide sur 20 chacune. Pas de recherche de prompt.

Le simulateur utilise des **teacher-forcings indépendants par item**, pas des
rafales libres corrélées ; sa masse OTHER n'est pas le taux de validité d'une
ligne complète. LoRA est une direction **aléatoire**, pas entraînée ; FP8 est
un aller-retour des poids, pas un kernel/quantification d'activations réel ;
sys/persona utilisent des **drapeaux simulés**, pas une précision textuelle
mesurée. Le bonus d'identité ne comptait qu'un exemple par variante.

## Collecteur et vérifications

`collect.py` génère `prompts.jsonl`, `manifest.json`, `plan.json` : **4 drapeaux
+ 30 longues + 62 rafales de 22 items, dont deux répétées à l'identique,
+ 4 réserve = 100 requêtes/endpoint**. Codes de trois caractères seedés et
fixés, sans crochets ni deux-points avant la réponse. Maximum observé sur les
trois tokenizers publics locaux : **756 tokens de prompt**, sous 1 024.

- Default dry-run : aucun envoi ; 3 300 réponses prévues pour 33 endpoints.
- Mock refs : **900 réponses, 100/ref**, reprise sans doublons ; génération
  des rafales via distributions locales, autres réponses = placeholders.
- Parcours mock → `infer.py` → `submission.csv` → `score_practice.py` terminé :
  fichier accepté, MCC **0,000** (26/36 paires justes). C'est **un contrôle de
  format**, pas une validation de précision, ni du réglage sur les refs.
- Tests collecte : **4 tests OK** ; budget insuffisant, garde live, 429 et
  Retry-After, JSON complet, reprise, requête incertaine non réémise. Transport
  injecté sans réseau. Tests précédents : 11 tests statistiques/décision OK,
  puis 1 test CLI d'inférence OK.

Le mode live exige **les trois options** `--live --i-understand-budget
--target refs|scored`, et `MODEL_EQUALITY_API_KEY` dans l'environnement. Quatre
requêtes au maximum ; lecture du budget avant les complétions et dans les
réponses ; journal JSON complet append-only et fsync ; reprise sans réémettre
les réponses déjà reçues. Une erreur de livraison incertaine bloque la reprise
automatique : inspecter le journal/budget, ne pas risquer une double dépense.
Les 429 avec Retry-After sont réessayés de façon bornée ; pas de retry aveugle
des POST après timeout/5xx. Les dossiers mock et live ne sont pas mélangeables.

**Attention :** le collecteur suit le protocole demandé à 100 prompts ; il ne
reproduit pas exactement l'allocation paired-code du pilote gelé. Aucun score
de ce nouveau protocole complet n'a été évalué. Les seuils précédents ne sont
pas validés pour l'utiliser tel quel en décision finale.

## Ce qui manque

Références **réelles** et calibration des drapeaux texte ; exploitation validée
des 30 longues ; robustesse du parseur/format sur les modèles distants ; accès
GPU/JURECA pour les pistes plus lourdes. La **brique white-box de la PR #3
n'est pas intégrée**, faute de temps et de portail FP8 white-box validé en
simulation. D4 seul ne valide pas le LLR apparié. Aucun score réel ni avantage
par rapport au plancher annoncé ne peut être affirmé.

## Commandes de collecte notée — une seule campagne, par l'équipe

**Instructions uniquement : aucune de ces commandes live n'a été exécutée ici.**
L'équipe doit décider explicitement de dépenser le budget. Garder le même
dossier pour une reprise ; ne jamais lancer deux campagnes/dossiers en
parallèle. Ce lancement consomme aussi les **quatre prompts réserve**.

```sh
cd "/home/ubuntu/repos/CISPA-Paris-/Task 2 - Model Equality Testing/plan_C"
PY="$HOME/me/venv/bin/python"
OUT="$HOME/me/collect_scored_once"

# Préparer et relire les 100 prompts, sans aucun envoi.
"$PY" collect.py --dry-run --target scored --out "$OUT"

# Clé injectée par le gestionnaire de secrets, jamais stockée dans le dépôt.
: "${MODEL_EQUALITY_API_KEY:?Injecter la clé de l'équipe dans l'environnement}"

# UNE SEULE CAMPAGNE PAR L'ÉQUIPE, après décision explicite de dépense.
"$PY" collect.py --live --i-understand-budget --target scored --out "$OUT"
```

Après collecte, conserver `responses.jsonl` intégral et `answers.jsonl`.
`infer.py` peut produire un CSV technique, mais **ne pas soumettre aveuglément
sa sortie** : R n'est pas validé et les drapeaux réels ne sont pas calibrés.
Ne pas mettre la clé dans les fichiers de sortie, les logs ou Git.

Pour reproduire le contrôle **sans aucun réseau** :

```sh
"$PY" collect.py --mock --target refs --out "$HOME/me/collection_mock_refs"
"$PY" infer.py --answers "$HOME/me/collection_mock_refs/answers.jsonl" \
  --manifest "$HOME/me/collection_mock_refs/manifest.json" \
  --output "$HOME/me/collection_mock_refs/submission.csv"
"$PY" "$HOME/me/score_practice.py" "$HOME/me/collection_mock_refs/submission.csv"
```

Les caches publics locaux de l'étape 1 restent nécessaires au dry-run/mock.
Le scoreur practice n'est utilisé que comme contrôle format/score, jamais pour
ajuster les décisions. Voir `results_phase0.md` et les JSON associés pour le
détail des portails, des scores et du gel.

## Ajout : `cluster.py` sur les vraies réponses partielles (vague 1)

`python3 cluster.py answers.jsonl -o submission.csv` (stdlib seule ; diagnostic sur stderr ; données partielles OK).
Familles : mots-clés d'identité en F1 ; découpe en 2 d'un groupe si les profils 4-grammes des L* sont séparés (`--split-threshold`).
F4 est prioritaire par défaut : listes des dix premiers nombres premiers sans espace après virgule → A, avec espaces → B, si chaque signature compte ≥ 3 endpoints (`--no-f4-split` pour désactiver) ; les réponses sans signature vont au groupe le plus petit (égalité → A).
Une famille scindée par F4 ne subit pas de split L* supplémentaire ; labels S et doublons sont recalculés dans chaque sous-famille.
Modèles : déguisements signalés (persona via F1, fuite de prompt système via F2) → label S de la famille ; doublons exacts de L* fusionnés ; le reste en singletons. `--prefix-link N` optionnel (désactivé : plain, fp8 et LoRA 0,3 partagent leurs préfixes en décodage quasi déterministe).
Sur l'instantané F1+F2 : Llama-S = ep-16, ep-29 ; Qwen-S = ep-20, ep-23. Le score public de 0,05 d'un seul modèle par famille vient du seul MCC_family (≈ 0,5) : il y a probablement plus de 2 familles, à trancher avec les L*.
