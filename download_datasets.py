"""Download the official hackathon data into the existing task folders."""

import hashlib
import shutil
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
HUB = "https://huggingface.co/datasets/SprintML"
DATASETS = (
    (
        "Task 1 - Watermark Localization",
        "Hackathon_Watermark_Localisation_Task_Paris",
        "8d27b2c9cc37c4829c8501e43e11fcb1b5db58c0",
        {
            "train.zip": "32ea0beaa173d2a8234574b893cab053cbad960a7b22d26b85e2b16ee782f783",
            "test.zip": "3949a2e66d7b003db743d6f2913369149284053cde072a45610d8f31d06c1be3",
        },
    ),
    (
        "Task 2 - Model Equality Testing",
        "model_equality",
        "330eba8b238791727be5a198362bb58ba3b7b388",
        {"practice_answers.csv": "ff5a29aef5280bfa6f884f6ebb162ae1dddaa3c9a5182c60eb4ae24488c2efba"},
    ),
    (
        "Task 3 - Unlearning",
        "unlearning",
        "beac43807fbc29a32c1037ea1d8ee2ed6548595a",
        {
            "forget_images.npz": "5b1d9deaf475db0905973aa028511ab35aebc33196412f5ef9ebbef09f1541c7",
            "model.pt": "485881a03ad064b5e6defc245ae3f45fd96714bf065ff96543a5eeca9f642571",
        },
    ),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target, expected_hash):
    if target.is_file() and sha256(target) == expected_hash:
        print(f"Already verified: {target.relative_to(ROOT)}", flush=True)
        return

    print(f"Downloading: {target.relative_to(ROOT)}", flush=True)
    with NamedTemporaryFile(dir=ROOT, prefix=".download-", delete=False) as output:
        temporary = Path(output.name)
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if sha256(temporary) != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {target.name}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    for folder, repo, revision, files in DATASETS:
        for name, expected_hash in files.items():
            target = ROOT / folder / name
            url = f"{HUB}/{repo}/resolve/{revision}/{name}"
            try:
                download(url, target, expected_hash)
            except (OSError, ValueError) as error:
                print(f"{target.relative_to(ROOT)}: {error}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
