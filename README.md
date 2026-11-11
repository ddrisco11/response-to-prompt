# Response to Prompt

Given one assistant answer, predict the first user prompt word for word. This is a standalone Apple silicon experiment by David Driscoll. The model sees the response text alone. Several prompts can produce the same answer, so a plausible reconstruction may still miss the target.

## Model and data

The base is [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). MLX-LM trains rank-8 LoRA adapters on the query and value projections in the final eight layers, with the input masked out of the loss. Corpus and base revisions are pinned in settings.py.

Training uses 25,000 first-turn pairs from [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), with 500 validation and 500 held-out test pairs. We preserve the recorded prompt text, cap the entire example at 768 tokens, group by normalized prompt template and drop repeated normalized answers. Paraphrases can still cross splits. The test measures reconstruction on short synthetic dialogues.

## Reproduce on Apple silicon

Use Python 3.11 in a virtual environment.

```sh
pip install -r requirements.txt
python download.py
python prepare.py
python -m mlx_lm lora --config train.yaml --model .assets/base --data data --adapter-path adapters
python evaluate.py
printf '%s' 'The capital of France is Paris.' | python predict.py
```

Evaluation compares the untuned base and adapter using the same 500 responses and greedy decoding with a 256-token output limit. Primary metric is raw character-for-character exact match, including capitalization, whitespace and punctuation. We also report mean normalized character edit distance. Results are written to adapters/evaluation.json. The native MLX adapter requires MLX-LM.

## Credits

This experiment adapts Qwen on UltraChat for local prompt reconstruction. Earlier research directly investigated inferring prompts from model outputs, including [Output2Prompt](https://aclanthology.org/2024.emnlp-main.819/). Qwen weights are Apache-2.0 licensed and the UltraChat 200k dataset card lists MIT. Dataset examples and base weights are downloaded separately.
