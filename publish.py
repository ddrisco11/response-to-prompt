"""Publish final MLX adapter and evaluation to the authenticated user's account."""
import json
from pathlib import Path

from huggingface_hub import HfApi

from report import results_table, update_public_readme
from settings import BASE, BASE_REV, DATASET, DATA_REV


def main():
    folder = Path('adapters/semantic')
    scores = json.loads((folder / 'evaluation.json').read_text())
    if set(scores) != {'base', 'supervised', 'adapter', 'paired_cosine_gain'}:
        raise ValueError('Finish both held-out evaluations before publishing.')
    train_count = sum(1 for _ in open('data/train.jsonl'))
    card = f"""---
base_model: {BASE}
datasets:
- {DATASET}
license: apache-2.0
library_name: mlx-lm
---
# TROUT

![TROUT model card](https://raw.githubusercontent.com/ddrisco11/response-to-prompt/main/trout.png)

Text Reconstruction from Output Using Transformers.

An MLX LoRA by David Driscoll for reconstructing the first user prompt from one
assistant response. The output is a guess. Multiple prompts can produce the same answer.

Trained on {train_count} filtered synthetic first-turn pairs from UltraChat
revision `{DATA_REV}`, with base revision `{BASE_REV}`. After supervised training, we sample two reconstructions per response on a subset of the training split.
A frozen MiniLM encoder scores each candidate against the original prompt.
An additional LoRA training pass learns from the higher-scoring candidates that pass the
quality and margin thresholds. Mean cosine similarity measures alignment in meaning and
can miss mistakes involving numbers or negation. Edit distance is normalized by the longer string.

{results_table(scores)}

[Training and inference code](https://github.com/ddrisco11/response-to-prompt).
Related work includes [Output2Prompt](https://aclanthology.org/2024.emnlp-main.819/).
"""
    (folder / 'README.md').write_text(card)
    update_public_readme(results_table(scores))
    api = HfApi()
    owner = api.whoami()['name']
    repo = f'{owner}/TROUT'
    api.create_repo(repo, exist_ok=True)
    print(api.upload_folder(repo_id=repo, folder_path=folder,
                            allow_patterns=['README.md', 'adapters.safetensors',
                                            'adapter_config.json', 'evaluation.json']), flush=True)


if __name__ == '__main__':
    main()
