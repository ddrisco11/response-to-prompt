"""Publish final MLX adapter and evaluation to the authenticated user's account."""
import json
from pathlib import Path

from huggingface_hub import HfApi

from settings import BASE, BASE_REV, DATASET, DATA_REV


def main():
    folder = Path('adapters')
    scores = json.loads((folder / 'evaluation.json').read_text())
    if set(scores) != {'base', 'adapter'}:
        raise ValueError('Finish both held-out evaluations before publishing.')
    train_count = sum(1 for _ in open('data/train.jsonl'))
    card = f"""---
base_model: {BASE}
datasets:
- {DATASET}
license: apache-2.0
library_name: mlx-lm
---
# Response to Prompt

An MLX LoRA by David Driscoll for reconstructing the first user prompt from one
assistant response. The output is a guess. Multiple prompts can produce the same answer.

Trained on {train_count} filtered synthetic first-turn pairs from UltraChat
revision `{DATA_REV}`, with base revision `{BASE_REV}`. The target preserves
the original wording. Greedy held-out metrics below include raw exact match and
mean normalized character edit distance (lower is better). Scores cover this
filtered corpus and do not establish verbatim recovery in arbitrary conversations.

```json
{json.dumps(scores, indent=2)}
```

[Training and inference code](https://github.com/ddrisco11/response-to-prompt).
Related work includes [Output2Prompt](https://aclanthology.org/2024.emnlp-main.819/).
"""
    (folder / 'README.md').write_text(card)
    api = HfApi()
    owner = api.whoami()['name']
    repo = f'{owner}/response-to-prompt-qwen2.5-0.5b-lora'
    api.create_repo(repo, exist_ok=True)
    print(api.upload_folder(repo_id=repo, folder_path=folder,
                            allow_patterns=['README.md', 'adapters.safetensors',
                                            'adapter_config.json', 'evaluation.json']), flush=True)


if __name__ == '__main__':
    main()
