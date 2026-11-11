"""Learn from sampled reconstructions ranked by prompt embedding similarity."""
import json
import random
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

from predict import load_predictor
from semantic import PromptSimilarity
from settings import SEMANTIC_SYSTEM


def semantic_refinement():
    root = Path('data/refinement')
    root.mkdir(exist_ok=True)
    train_rows = Path('data/train.jsonl').read_text().splitlines()
    rows = random.Random(71).sample(train_rows, 1024)
    predict = load_predictor('adapters')
    scorer = PromptSimilarity()
    mx.random.seed(71)
    sampler = make_sampler(temp=0.75)
    retained = []
    for i, row in enumerate(rows, start=1):
        _, response, reference = json.loads(row)['messages']
        candidates = [predict(response['content'], max_tokens=160, sampler=sampler) for _ in range(2)]
        if all(candidate.strip() for candidate in candidates):
            target, a, b = scorer.encode([reference['content'], *candidates])
            scores = [float(target @ a), float(target @ b)]
            winner = scores.index(max(scores))
            if max(scores) >= 0.70 and abs(scores[0] - scores[1]) >= 0.05:
                messages = [dict(role='system', content=SEMANTIC_SYSTEM), response,
                            dict(role='assistant', content=candidates[winner])]
                retained.append(json.dumps(dict(messages=messages)))
        if i % 64 == 0:
            print(f'Scored {i} training responses, retained {len(retained)}', flush=True)
    if len(retained) < 4:
        raise RuntimeError('Too few qualifying candidates for a refinement batch.')
    (root / 'train.jsonl').write_text('\n'.join(retained) + '\n')
    (root / 'valid.jsonl').write_bytes(Path('data/valid.jsonl').read_bytes())
    subprocess.run([sys.executable, '-m', 'mlx_lm', 'lora', '--config', 'train.yaml',
                    '--model', '.assets/base', '--data', str(root), '--adapter-path', 'adapters/semantic',
                    '--resume-adapter-file', 'adapters/adapters.safetensors',
                    '--learning-rate', '0.00001', '--iters', str(len(retained) // 4)], check=True)


if __name__ == '__main__':
    semantic_refinement()
