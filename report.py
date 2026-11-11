"""Render the same measured outcomes for the project and the model card."""
import base64
import json
from pathlib import Path
import subprocess

START = '<!-- held-out-results -->'
END = '<!-- end-held-out-results -->'


def results_table(scores):
    lines = ['| Model | Responses | Mean prompt cosine ↑ | Verbatim matches | Mean edit distance ↓ |',
             '|---|---:|---:|---:|---:|']
    for key, name in [('base', 'Untuned Qwen'), ('supervised', 'Supervised LoRA'), ('adapter', 'LoRA adapter')]:
        if key not in scores:
            continue
        row = scores[key]
        lines.append(f"| {name} | {row['examples']} | {row['mean_cosine']:.3f} | "
                     f"{row['exact_matches']} | {row['mean_normalized_edit_distance']:.3f} |")
    gain = scores['paired_cosine_gain']
    low, high = gain['ci95']
    lines.extend(['', f"Mean paired cosine gain: {gain['mean']:+.3f}. "
                  f"95% bootstrap interval across responses: [{low:+.3f}, {high:+.3f}]."])
    return '\n'.join(lines)


def update_public_readme(table):
    api = ['gh', 'api', 'repos/ddrisco11/response-to-prompt/contents/README.md']
    meta = json.loads(subprocess.check_output(api))
    content = base64.b64decode(meta['content']).decode()
    if content.count(START) != 1 or content.count(END) != 1:
        raise ValueError('Missing or ambiguous results section')
    before, rest = content.split(START)
    _, after = rest.split(END)
    new = before + START + '\n\n' + table + '\n\n' + END + after
    if new != content:
        payload = dict(message='Report held-out prompt reconstruction metrics', sha=meta['sha'],
                       content=base64.b64encode(new.encode()).decode())
        subprocess.run(api + ['--method', 'PUT', '--input', '-', '--silent'],
                       input=json.dumps(payload), text=True, check=True)
    Path('README.md').write_text(new)
