# CISPA-Paris-

[Agents Famou et AIDE pour la tâche 2](Task%202%20-%20Model%20Equality%20Testing/agents/README.md) : installation, pilote CPU et contrôle avant soumission.

## Datasets du hackathon

Depuis la racine du dépôt, lancez cette commande avec Python 3 pour télécharger
les fichiers d'origine (environ 1,44 Go). Aucun compte Hugging Face n'est requis.

```sh
python3 download_datasets.py
```

| Dossier | Fichiers téléchargés | Source officielle |
| --- | --- | --- |
| `Task 1 - Watermark Localization/` | `train.zip` et `test.zip` | [Watermark Localization](https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris) |
| `Task 2 - Model Equality Testing/` | `practice_answers.csv`, les réponses des endpoints de référence | [Model Equality Testing](https://huggingface.co/datasets/SprintML/model_equality) |
| `Task 3 - Unlearning/` | `forget_images.npz` et le checkpoint initial `model.pt` | [Unlearning](https://huggingface.co/datasets/SprintML/unlearning) |

Le script vérifie les empreintes SHA-256 et réutilise les fichiers déjà valides.
Git ignore les fichiers téléchargés pour que l'équipe les récupère directement
depuis Hugging Face. Les PDF des sujets se trouvent dans les dossiers correspondants.