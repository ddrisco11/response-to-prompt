"""Score matched predictions and report the paired difference against the base."""
import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from predict import load_predictor
from semantic import PromptSimilarity


def compare(rows, adapter, include_warmup=False):
    references = [row['messages'][2]['content'] for row in rows]
    scorer = PromptSimilarity()
    reference_vectors = np.vstack([scorer.encode(references[i:i + 32])
                                  for i in range(0, len(rows), 32)])
    scores = {}
    similarities = {}
    models = [('base', None)]
    if include_warmup:
        models.append(('supervised', 'adapters'))
    models.append(('adapter', adapter))
    for name, weights in models:
        predict = load_predictor(weights)
        outputs = [predict(row['messages'][1]['content']) for row in rows]
        del predict
        gc.collect()
        vectors = np.vstack([scorer.encode(outputs[i:i + 32]) for i in range(0, len(rows), 32)])
        cosines = np.einsum('ij,ij->i', reference_vectors, vectors)
        similarities[name] = cosines
        scores[name] = dict(examples=len(rows), mean_cosine=float(cosines.mean()),
                            exact_matches=sum(a == b for a, b in zip(outputs, references)),
                            mean_normalized_edit_distance=float(np.mean([
                                Levenshtein.normalized_distance(a, b) for a, b in zip(outputs, references)])))
    delta = similarities['adapter'] - similarities['base']
    samples = np.random.default_rng(41).choice(delta, (3000, len(delta)), replace=True)
    scores['paired_cosine_gain'] = dict(mean=float(delta.mean()),
                                        ci95=np.quantile(samples.mean(axis=1), [0.025, 0.975]).tolist())
    return scores


if __name__ == '__main__':
    # The standard run reaches this point after the supervised warm-up.
    subprocess.run([sys.executable, 'refine.py'], check=True)
    rows = [json.loads(line) for line in Path('data/test.jsonl').read_text().splitlines()]
    scores = compare(rows, 'adapters/semantic', include_warmup=True)
    Path('adapters/semantic/evaluation.json').write_text(json.dumps(scores, indent=2) + '\n')
    print(scores, flush=True)
