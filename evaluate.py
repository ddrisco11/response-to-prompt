"""Compare greedy adapter and base predictions on identical held-out responses."""
import gc
import json
from pathlib import Path

from rapidfuzz.distance import Levenshtein
from predict import load_predictor


def score(rows, adapter):
    predict = load_predictor(adapter)
    exact = 0
    distance = 0
    for row in rows:
        _, answer, prompt = row['messages']
        actual = predict(answer['content'])
        expected = prompt['content']
        exact += actual == expected
        distance += Levenshtein.normalized_distance(actual, expected)
    return dict(examples=len(rows), exact_matches=exact, exact_match=exact / len(rows),
                mean_normalized_edit_distance=distance / len(rows))


if __name__ == '__main__':
    rows = [json.loads(line) for line in Path('data/test.jsonl').read_text().splitlines()]
    scores = {}
    for name, adapter in [('adapter', 'adapters'), ('base', None)]:
        scores[name] = score(rows, adapter)
        gc.collect()
        Path('adapters/evaluation.json').write_text(json.dumps(scores, indent=2) + '\n')
        print(name, scores[name], flush=True)
