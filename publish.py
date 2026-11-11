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
language:
- en
pipeline_tag: text-generation
tags:
- prompt-reconstruction
- prompt-inversion
- prompt-recovery
- inverse-prompting
- response-to-prompt
- semantic-similarity
- ultrachat
- qwen2.5
- lora
- mlx
- apple-silicon
- 8-bit
- quantized
- small-language-model
base_model: {BASE}
datasets:
- {DATASET}
license: apache-2.0
library_name: mlx-lm
---
# TROUT

![TROUT model card](https://raw.githubusercontent.com/ddrisco11/response-to-prompt/main/trout.png)

inpuT Reconstruction from Output Using Transformers.

David Driscoll · Reconstructs a plausible user prompt from one assistant response. Trained using {train_count} UltraChat first-turn pairs.

**Held-out UltraChat evaluation** · 8-bit Qwen2.5-0.5B-Instruct with MLX LoRA.

{results_table(scores)}

Download with `hf download ddrisco11/TROUT --local-dir TROUT`. Load with `mlx_lm.load("TROUT/base-8bit", adapter_path="TROUT")`.

[GitHub and inference instructions](https://github.com/ddrisco11/response-to-prompt)

**Sources** · [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) · [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) · [MiniLM sentence encoder used for evaluation](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [Hu et al., LoRA](https://arxiv.org/abs/2106.09685) · [Zhang, Morris & Shmatikov, Output2Prompt (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.819/).
"""
    (folder / 'README.md').write_text(card)
    update_public_readme(results_table(scores))
    api = HfApi()
    owner = api.whoami()['name']
    repo = f'{owner}/TROUT'
    api.create_repo(repo, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=".assets/base-8bit", path_in_repo="base-8bit", ignore_patterns=["README.md"])
    print(api.upload_folder(repo_id=repo, folder_path=folder,
                            allow_patterns=['README.md', 'adapters.safetensors',
                                            'adapter_config.json', 'evaluation.json']), flush=True)


if __name__ == '__main__':
    main()
