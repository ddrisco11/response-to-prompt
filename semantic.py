"""Cosine similarity from the frozen, mean-pooled MiniLM sentence encoder."""
import torch
from transformers import AutoModel, AutoTokenizer

ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = 'c9745ed1d9f207416be6d2e6f8de32d1f16199bf'


class PromptSimilarity:
    def __init__(self):
        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(ENCODER, revision=REVISION)
        self.model = AutoModel.from_pretrained(ENCODER, revision=REVISION).cpu().eval()

    @torch.inference_mode()
    def encode(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True,
                                padding=True, max_length=256)
        hidden = self.model(**inputs).last_hidden_state
        mask = inputs.attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return torch.nn.functional.normalize(pooled).numpy()
