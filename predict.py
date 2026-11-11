"""Generate a single prompt from a single response on Apple silicon."""
import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_lm import generate, load

from settings import BASE, BASE_REV, SYSTEM


def load_predictor(adapter):
    base = Path('.assets/base')
    weights = str(base) if base.is_dir() else snapshot_download(BASE, revision=BASE_REV)
    model, tokenizer = load(weights, adapter_path=adapter)

    def predict(response, max_tokens=256):
        messages = [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': response},
        ]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, return_dict=False, add_generation_prompt=True
        )
        return generate(model, tokenizer, prompt=ids, max_tokens=max_tokens, verbose=False)

    return predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', default='adapters')
    args = parser.parse_args()
    print(load_predictor(args.adapter)(sys.stdin.read()))
