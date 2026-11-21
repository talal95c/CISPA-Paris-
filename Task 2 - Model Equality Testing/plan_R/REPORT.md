# Plan R — Model Equality Testing

**Aucune requête vers la compétition**, ni aux 33 endpoints notés, ni aux 9 références. Les seuls appels de l'API de collecte ont visé `127.0.0.1`. Les scripts et le corrigé officiels proviennent du dépôt public Hugging Face.

## Réalisé

- `collect.py` : dry-run sans réseau par défaut. Manifeste déterministe : 4 drapeaux, 30 longues et 62 rafales dont deux répétitions identiques, puis quatre réserves. Au plus quatre requêtes simultanées, plafond de 256 tokens, réponses JSON intégrales, gestion de `budget` et `Retry-After`. Une requête dont le résultat est incertain bloque la reprise jusqu'à réconciliation ; elle n'est jamais répétée automatiquement.
- `simulate.py` / `simulate_server.py` : neuf variantes par base HF locale : base, FP8 E4M3 poids seuls, LoRA rang 4 aux forces 0,3/0,7/1,5, sys, persona, KGW et température. Le simulateur valide les 39 préfixes aléatoires sous chaque tokenizer. La variante greedy est un contrôle séparé.
- `features.py` / `partition.py` / `submit.py` : drapeaux conjoints, dispersion et argmax si greedy, log-cotes normalisées par endpoint, LLR gaussiens écrêtés à ±3. Les règles de rattachement de R s'appliquent à chaque hypothèse de drapeaux. L'optimisation est exacte sur le produit de ces hypothèses (jusqu'à trois drapeaux faibles par famille) et utilise les comptes MCC sommés avant de calculer le score. CSV officiel ; `--provisional` pour S1.

## Phase 0 et score

### Score simulé — cinq blocs disjoints de 100 réponses/endpoint par split

Score : `0,1 × MCC_family + 0,9 × MCC_model`. Le MCC modèle ne compte que les paires d'une même famille **vraie**.

| Profil gelé | Développement médiane / q10 | Holdout médiane / q10 | Écart de médiane | Minimum holdout |
|---|---:|---:|---:|---:|
| R | 0,17223 / 0,17223 | 0,00402 / 0,00050 | +0,16820 | 0,00000 |
| Simple | 0,17223 / 0,17223 | 0,00124 / 0,00050 | +0,17098 | 0,00000 |

**Portail C global : no-go** (exige médiane ≥ 0,80 et q10 ≥ 0,60). Le MCC modèle est nul sur chacun des deux folds famille inédite pour les deux profils. Le profil simple est le repli préenregistré vu l'écart > 0,05, **pas** un gain empirique ni une recommandation de soumettre sans contrôle. Le 0,68 théorique du plan n'est pas reproduit. Pour R, le stress test des drapeaux bruités au taux mesuré sur le développement donne une médiane de 0,08740 et un q10 de 0,00191 ; ce n'est pas le score principal.

Le CSV du bloc simulé SmolLM2 (alias `ref-01…09`) est accepté par le scoreur officiel : MCC 0,273 sur 36 paires, 27 correctes. Ce chiffre vérifie le format, pas les performances sur les vraies références. Aucun changement de réglage après ce contrôle.

### Portails de Phase 0

| Portail T4 | Mesure locale | Verdict |
|---|---|---|
| B2 — nulle S–S | 20 répétitions × 10 prompts à 20 items. 100 paires de textes : distance trigramme moyenne 0,726, q95 0,991 ; 0 item apparié. | Nulle textuelle mesurée ; nulle par item indisponible. |
| B2′ — greedy | À 20 items : 10/10 paires greedy identiques, contre 1/100 plain. À 12 items : 10/10 contre 0/10. | No-go strict à 20 ; go au pilote à 12. |
| B3 — efficacité / format | À 20 items : 0/200 plain complets et 100 % hors format pour sys et persona (20 chacun). À 12 : 0/20 plain complets et 100 % hors format (20 chacun). Couverture parsée : 0 % aux deux tailles. | No-go aux deux tailles (seuils ≥ 0,30 et ≤ 20 %). Efficacité non estimable ; aucun signal pour monter vers C. |
| B4 — watermark KGW | 20 réponses KGW libres par taille ; 0 item exploitable pour la variance de δ et son q95 nul. | Non mesurable, pas de go. Ne pas supposer la neutralisation KGW. |
| Drapeaux | Sys 5/10, persona 9/10 ; fp8 et les trois LoRA : 0/10 chacun. | No-go : sys < 90 % ; drapeaux faibles seulement. |
| B7 — température | Rafales contrôlées : AUC 0,690, fausses alarmes 20 % (8/40) ; forme : AUC 0,673, 17,5 % (7/40). | No-go (≥ 0,90 et ≤ 10 %), même avec la forme. |

Pas de classement relatif du plain, ni d'axe LoRA signé, ni de LLR white-box : hors Plan R. Les tests réels `n`, préremplissage assistant et `usage` restent à vérifier si autorisés ; aucun déploiement réel ni accès JURECA/GPU vérifié. Sur CPU, le petit pilote de génération libre justifie de ne pas élargir les essais sys/persona/KGW à 200 chacun sans un format exploitable.

### Protocole anti-surapprentissage

`frozen_r.json` a été commité **avant le premier calcul du score holdout**. Aucun paramètre ajusté sur les neuf références ou le fichier officiel. Le profil R utilise des a priori fixes ; le profil simplifié préenregistré enlève température et signal de rafale. La règle de repli, fixée à l'avance, recommande le profil simple si l'écart absolu de médiane développement–holdout atteint 0,05. Aucun profil n'est réoptimisé sur le holdout.

Développement : SmolLM2-135M-Instruct, TinyStories-33M. Holdout : Pythia-14M, TinyStories-8M. Aucun prompt complet commun ; drapeaux et longues reformulés, rafales tirées séparément. L'analyse de rafale réserve un type entier et 23 des 75 clés des autres types au holdout. Les résultats par famille sont calculés en leave-one-family-out sans adaptation ; le score combiné vient des deux familles inédites prédites conjointement, car le MCC famille n'est pas informatif avec une seule famille.

Les rejeux ont 100 réponses/endpoint chacun. Les drapeaux et longues sont générés librement (28 tokens maximum sur CPU) ; les rafales sont accélérées par échantillonnage des deux logits conditionnels de chaque item. L'efficacité et le hors-format de la Phase 0 sont, eux, mesurés en génération libre. Ce score simulé n'est donc pas une prévision fiable pour les endpoints réels.

Reproduire les rejeux sur CPU (dépendances dans `requirements.txt`, ajouter `--offline` si les poids sont déjà en cache) :

```sh
python simulate.py --engine probe --skip-phase --families 0 1 --free-max-new-tokens 28 --output dev.jsonl
python simulate.py --engine probe --skip-phase --heldout-prompts --families 2 3 --free-max-new-tokens 28 --output held.jsonl
python replay.py --development dev.jsonl --holdout held.jsonl --output evaluation.json --practice-output practice-sim.csv
```

## Résultats hors ligne sur réponses réelles (vague 1)

Instantané F1/F2 : 63 réponses (F2 absent pour ep-24, ep-32 et ep-33). Instantané suivant : 176 réponses (F1–F4 et L01 complets, 11 L02). Aucun nouvel appel ni soumission. L'adaptateur valide les prompts d'origine puis projette F1–F3 et les longues sur les canaux homologues de R ; ce ne sont pas les formulations du manifeste gelé. F4 est conservé à part : une liste de nombres premiers ne teste pas `FORMAT_OK`.

- **Gelé — `submission_R_F1F2.csv` et `submission_R_wave1_frozen.csv` sont identiques.** Avec `--allow-partial --profile simple` (repli préenregistré, aucun réglage), R place les 33 endpoints dans F1, avec 33 modèles distincts. Aucun drapeau identité/persona ni fuite n'est déclenché sur les 33 endpoints ; le détecteur F2 exige deux motifs, contre 0, 1 et 0 observés pour ep-16, ep-29 et ep-31, signalés par l'équipe.
- **Exploratoire — `submission_R_wave1_exact_F3F4.csv`.** Conserve les familles gelées et relie les modèles si les deux réponses brutes F3 et F4 sont exactement identiques. Groupes : {ep-04, ep-12, ep-17, ep-18, ep-25} et {ep-19, ep-29} ; 26 singletons. Ce signal de format et de style ne prouve pas l'égalité des modèles.

Écarts par paires (famille / modèle), indépendants des noms de classes : gelé vs C F1/F2 **242 / 2**, vs M drapeaux **376 / 0**, vs M longues **283 / 0**. Pour la variante, ces écarts sont **242 / 13**, **376 / 11** et **283 / 11**. Le scoreur local de R donne 0 au gelé quelle que soit la vérité, car les deux prédictions binaires sont constantes. À titre conditionnel, la variante donnerait 0,90 si ses 11 paires modèle étaient exactement les vraies, avec des négatifs dans les vraies familles ; si les 11 étaient fausses, le score local écrêté serait 0. Ce ne sont pas des estimations du score caché. Le 0,05 public communiqué par l'équipe provient d'une autre soumission.

```sh
python import_snapshot.py journal answers_snapshot.jsonl wave1_prompts.jsonl wave1_offline.jsonl
python submit.py wave1_offline.jsonl submission_R_wave1_frozen.csv --allow-partial --profile simple
python import_snapshot.py exact-flags wave1_offline.jsonl submission_R_wave1_frozen.csv submission_R_wave1_exact_F3F4.csv
```

## Collecte notée — à lancer plus tard, une seule fois par l'équipe

Depuis `Task 2 - Model Equality Testing/plan_R/`. Avant la vague 1, convenir du manifeste des trois plans, de la taille des rafales et d'un journal commun persistant. L'exemple fige 20 items ; si l'équipe valide le repli à 12, remplacer **partout** `--items 20` par `--items 12`. Les deux vagues doivent utiliser le même manifeste.

```sh
python collect.py --dry-run --target scored --items 20 --manifest-out manifest.json
# Seulement après autorisation, avec MODEL_EQUALITY_API_KEY déjà provisionnée :
python collect.py --live --confirm-scored --target scored --wave 1 --items 20 --output answers.jsonl
python submit.py answers.jsonl s1.csv --allow-partial --provisional --profile simple --items 20
python collect.py --live --confirm-scored --target scored --wave 2 --items 20 --output answers.jsonl
python submit.py answers.jsonl submission.csv --profile simple --items 20
```

La vague 2 saute les réponses déjà journalisées. Les réserves réutilisent en priorité les prompts rejetés ou sans texte ; une requête invalide non facturée peut laisser moins de 100 réponses, sans requête hors manifeste. En cas d'écart de budget, arrêter et réconcilier avec l'équipe. Ne pas effacer le journal ni relancer avec un autre chemin.

Après autorisation distincte de tester les vraies références, utiliser `collect.py --live --target refs` et un journal séparé, puis `submit.py … --target refs` et le scoreur officiel **uniquement pour contrôle final**. Ne régler aucun paramètre sur les références. Les détails du classement final et de la clôture n'ont pas été vérifiés auprès des organisateurs.
