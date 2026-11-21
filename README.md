# CISPA-Paris-

## Datasets du hackathon

Les fichiers officiels sont dans les dossiers des sujets. Les quatre fichiers
volumineux (1,44 Go au total) sont stockés sur GitHub via Git LFS. Installez
[Git LFS](https://git-lfs.com/) avant de cloner le dépôt. Pour un clone existant,
exécutez ces commandes après avoir mis votre branche à jour.

```sh
git lfs install
git lfs pull
```

| Dossier | Fichiers | Source officielle |
| --- | --- | --- |
| `Task 1 - Watermark Localization/` | `train.zip` et `test.zip` | [Watermark Localization](https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris) |
| `Task 2 - Model Equality Testing/` | `practice_answers.csv`, les réponses des endpoints de référence | [Model Equality Testing](https://huggingface.co/datasets/SprintML/model_equality) |
| `Task 3 - Unlearning/` | `forget_images.npz` et le checkpoint initial `model.pt` | [Unlearning](https://huggingface.co/datasets/SprintML/unlearning) |

Vous pouvez aussi récupérer les fichiers depuis Hugging Face avec
`python3 download_datasets.py`. Le script vérifie les empreintes SHA-256 et
réutilise les fichiers déjà valides.