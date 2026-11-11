"""Build reverse pairs while retaining the raw target string."""
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from settings import BASE, BASE_REV, DATASET, DATA_REV, SHARD, SYSTEM


def family(text):
    normalized = re.sub(r'https?://\S+|\d+|“[^”]+”|"[^"]+"', ' SLOT ', text.lower())
    return ' '.join(re.findall(r'\w+', normalized))


def main():
    local = Path('.assets') / SHARD
    if not local.exists():
        local = hf_hub_download(DATASET, SHARD, repo_type='dataset', revision=DATA_REV)
    base = Path('.assets/base')
    tok = AutoTokenizer.from_pretrained(str(base) if base.exists() else BASE, revision=BASE_REV)
    rows = pq.read_table(local, columns=['messages','prompt']).to_pylist()
    random.Random(41).shuffle(rows)
    splits = {k: [] for k in ('train','valid','test')}
    caps = dict(train=25000, valid=500, test=500)
    seen = set()
    for row in rows:
        turns = row['messages']
        if len(turns)<2 or [t['role'] for t in turns[:2]] != ['user','assistant']:
            continue
        question, answer = (t['content'] for t in turns[:2])
        if question != row['prompt'] or not question.strip() or not answer.strip():
            continue
        key = family(question)
        bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'big') % 100
        split = 'test' if bucket<3 else 'valid' if bucket<6 else 'train'
        if len(splits[split]) == caps[split] or key in seen or ('answer',family(answer)) in seen:
            continue
        messages = [dict(role='system',content=SYSTEM),dict(role='user',content=answer),dict(role='assistant',content=question)]
        if len(tok.apply_chat_template(messages,tokenize=True,return_dict=False)) > 768:
            continue
        splits[split].append(dict(messages=messages))
        seen.update((key,('answer',family(answer))))
        if all(len(splits[k])==caps[k] for k in caps):
            break
    Path('data').mkdir(exist_ok=True)
    for split, records in splits.items():
        if not records:
            raise ValueError(f'Empty {split} split')
        with Path(f'data/{split}.jsonl').open('w') as out:
            for record in records:
                out.write(json.dumps(record,ensure_ascii=False)+'\n')
    print({k:len(v) for k,v in splits.items()},flush=True)


if __name__=='__main__':
    main()
