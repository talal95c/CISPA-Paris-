# Plan M · Validation locale du LLR fp8

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

### Portails restant ouverts

- **B2/B2’** · témoin bf16/bf16 réalisé avec 20 blocs distincts. La détection greedy reste à tester.
- **B7** · T et top-p correctement retrouvés dans 4/4 pilotes de 12 réponses, avec les réglages simulés (0,65 et 1) et (1 et 0,95). L’AUC du drapeau et les fausses alarmes sur les variantes restent à mesurer.
- **B3, B4, drapeaux, B6, R1 boîte noire et D4** · rafales, watermark, déguisements, axe signé, classement boîte noire et criblage restent à évaluer sur la chaîne C.
- **C** · le portail apparié est franchi sur deux familles simulées. La vraisemblance retrouve le bon checkpoint dans les 4 pilotes parmi ces deux candidats. L’identité du checkpoint P et la fidélité aux réglages des serveurs restent inconnues. Ces deux conditions déterminent si le canal est disponible pour une famille.

## MCC sur cinq replays conditionnels

On projette les décisions plain/fp8 des cinq blocs disjoints sur deux familles de neuf endpoints. Les labels de famille, de déguisement et de LoRA sont imposés par la vérité terrain. La formule officielle (0,1 × MCC famille + 0,9 × MCC modèle) donne **médiane 1,000, q10 1,000, minimum 1,000**. Le repli qui rattache les deux candidats donne **0,760** avec les mêmes labels imposés. Le MCC global sur des rejeux intégraux de 100 réponses et le portail global de C restent à mesurer.

## Rejouer et intégrer

Prérequis testés sous Python 3.12 : PyTorch 2.6.0 CPU, Transformers 4.51.3 et les modèles téléchargés au préalable. Révisions HF utilisées : [TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M/tree/77f1b168e219585646439073245fe87e56b3023e) et [Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m/tree/cf967c0a9a04383db6f7b1108d86b2962634b4ac).

```sh
python -m unittest -v test_whitebox
python run_fp8_gate.py --model /chemin/tinystories-1m
python run_fp8_gate.py --model /chemin/tinystories-1m --visible-text --seed 1013 \
  --blocks 5 --prompts 30 --tokens 256 --threads 2 --batch-size 24 --output tiny.json
python run_fp8_gate.py --model /chemin/pythia-14m --visible-text --seed 1009 \
  --blocks 5 --prompts 30 --tokens 256 --threads 2 --batch-size 24 --output pythia.json
python score_replays.py tiny.json pythia.json
python score_text.py --model /chemin/checkpoint --pairs duels.jsonl
```

Chaque ligne de `duels.jsonl` contient les chaînes `prompt`, `left` et `right`. Un LLR positif favorise la réplique fp8 à gauche. L'appelant conserve les mêmes prompts et le même décodage pour les deux candidats. En présence d'un template de chat, reproduire le prompt effectivement tokenisé, y compris le séparateur avant la réponse. C conserve son repli si le checkpoint ou le duel est incertain.

Pour la collecte notée ultérieure, utiliser le collecteur commun de C/R et conserver les 30 longues du protocole partagé dans le journal. Cette branche réemploie leurs réponses pour produire le score hors ligne.
