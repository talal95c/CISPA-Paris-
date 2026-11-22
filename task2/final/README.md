# Tâche 2 — Model Equality Testing : solution finale

Équipe Teletubbies. Fichier rendu : `submission_final.csv` (mêmes 33 prédictions que [le CSV du dossier officiel](../../Task%202%20-%20Model%20Equality%20Testing/final_submission.csv) ; seules les fins de ligne diffèrent) (soumissions n° 4035 et 4211, score public 1,00).

## Réponse finale

| Famille | Modèle de base | Endpoints regroupés (même modèle) | Endpoints laissés seuls |
|---|---|---|---|
| A | Qwen/Qwen2.5-3B-Instruct | 04 12 14 23 33 | 01 17 18 24 25 |
| B | Qwen/Qwen2.5-1.5B-Instruct | 06 07 08 20 30 32 | 02 03 21 22 26 28 |
| C | unsloth/Llama-3.2-1B-Instruct | 10 13 16 27 29 31 | 05 09 11 15 19 |

Un groupe par famille = le modèle de base servi tel quel ou sous un déguisement qui ne change pas les poids
(température, persona, system prompt, watermark, troncature top-k). Les endpoints seuls ont des poids modifiés
(LoRA, fp8, autre).

## Ce qui est vérifié par le serveur et ce qui ne l'est pas

- Vérifié (score public) : le noyau A 04 23 33, B 06 08 20 30 32, C 10 13 16 29 31 donne 1,00 (n° 3503).
  `ep-26` regroupé avec B fait tomber le score à 0,65 ; `ep-25` regroupé avec A le fait tomber à 0,75.
- Non vérifiable en public (égalité à 1,00 avec ou sans) : `ep-27`, `ep-07`, `ep-14` (troncature top-k 40 sur
  poids de base, mesure locale) et `ep-12` (seul sosie de A dont le profil est celui d'un modèle de base :
  température ajustée 0,704 et excès de top-1 nul, reproduit par deux implémentations).
- Le score caché n'est pas connu. Variantes soumises pour couvrir les autres hypothèses : n° 3605 et 3694
  (noyau + 17, noyau + 12), 3872 (noyau + 27 07 14), 3956 (+ 17), 4118 (+ 01).

## Méthode

1. Famille : identification du modèle de base par les réponses (`../loop/classify.py`, `../loop/NOTES.md`).
2. Collecte : `../loop/epcollect.py` (budget 100 requêtes par endpoint, lu sur `/models` avant chaque lot),
   questions dans `../loop/probes_*.json`, réponses dans `../loop/data/`.
3. Mesure en boîte blanche : on rescore chaque réponse avec les poids publics du modèle de base.
   - `../loop/dt_test.py`, `dt_an.py`, `dt_gof.py` : température ajustée, écart dT entre tokens piqués et plats
     (LoRA ⇒ dT > 0), excès de top-1.
   - `../loop/trunc_test.py`, `tt_an.py` : support tronqué (top-k / top-p).
   - `../loop/sys_ll.py` : vraisemblance avec ou sans system prompt.
   - `../loop/wb_*.py`, `score_wb.py` : fp8, watermark, arrêt.
4. Sondes publiques : un CSV par hypothèse, une soumission toutes les 30 min ; une baisse élimine, une égalité
   à 1,00 ne prouve rien. Historique complet : `submissions/submissions.log` et `NOTES.md`.
5. Choix du portefeuille sur la partie cachée : `../loop/c19d/` (simulation du masque public et espérance du
   meilleur score caché).

## Reproduire

```sh
export TK=<clé d'équipe>            # jamais dans un fichier
python mk.py out.csv "04 23 33 14 12|06 08 20 30 32 07|10 13 16 29 31 27"
./submit2.sh out.csv                # option : epoch avant lequel ne pas soumettre
```

Les scripts de `../loop/` lisent la clé dans `SPRINTML_API_KEY`. Aucune clé n'est dans ce dépôt.
