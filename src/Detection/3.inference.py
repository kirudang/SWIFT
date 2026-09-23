#!/usr/bin/env python3
import os
os.environ["HF_TOKEN"] = "Your Hugging Face Token"
cache_dir = 'Your Cache Directory'
import json, csv, argparse, hmac, hashlib, string
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# =============== Minimal model pieces (must mirror training) ===============

def key_hmac_expand_to_H(key: str, H: int) -> torch.Tensor:
    """HMAC-SHA256(key, counter) -> bytes -> float vector in [-1,1], length H."""
    key_b = key.encode("utf-8")
    out = bytearray(); ctr = 0
    while len(out) < H:
        msg = ctr.to_bytes(4, "big", signed=False)
        out.extend(hmac.new(key_b, msg, hashlib.sha256).digest())
        ctr += 1
    raw = bytes(out[:H])
    v = torch.tensor(list(raw), dtype=torch.float32) / 255.0
    v = (v - 0.5) * 2.0
    return v

class KeyProjectorFromFFN(nn.Module):
    """Frozen FFN (intermediate->GELU->output) copied from transformer layer."""
    def __init__(self, backbone: AutoModel, layer_idx: int = 0):
        super().__init__()
        # This expects a BERT/RoBERTa-like backbone
        enc_layer = backbone.encoder.layer[layer_idx]
        in_dense  = enc_layer.intermediate.dense
        out_dense = enc_layer.output.dense
        self.intermediate = nn.Linear(in_dense.in_features,  in_dense.out_features,  bias=True)
        self.output       = nn.Linear(out_dense.in_features, out_dense.out_features, bias=True)
        with torch.no_grad():
            self.intermediate.weight.copy_(in_dense.weight); self.intermediate.bias.copy_(in_dense.bias)
            self.output.weight.copy_(out_dense.weight);       self.output.bias.copy_(out_dense.bias)
        for p in self.parameters(): p.requires_grad = False
        self.act = nn.GELU()

    @torch.no_grad()
    def forward(self, vH: torch.Tensor) -> torch.Tensor:
        return self.output(self.act(self.intermediate(vH)))

class KeyDirectCharEmbed(nn.Module):
    """Baseline (train-time only). Kept for completeness so checkpoints load."""
    def __init__(self, H: int, vocab: str = string.printable, emb_dim: int = 64):
        super().__init__()
        self.vocab = vocab
        self.stoi = {ch:i for i,ch in enumerate(vocab)}
        self.emb = nn.Embedding(num_embeddings=len(vocab), embedding_dim=emb_dim)
        self.toH = nn.Linear(emb_dim, H)

    def forward(self, keys: List[str]) -> torch.Tensor:
        idxs = []
        for k in keys:
            ids = [self.stoi.get(ch, 0) for ch in k] or [0]
            idxs.append(torch.tensor(ids, dtype=torch.long))
        lens = [len(x) for x in idxs]
        maxlen = max(lens)
        pad = torch.stack([F.pad(x, (0, maxlen - len(x)), value=0) for x in idxs], dim=0)  # [B,L]
        mask = (torch.arange(maxlen).unsqueeze(0) < torch.tensor(lens).unsqueeze(1)).to(pad.device)  # [B,L]
        E = self.emb(pad)  # [B,L,emb]
        m = (E * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return self.toH(m)  # [B,H]

class TextEncoder(nn.Module):
    def __init__(self, model_name="roberta-large"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # [B,T,H]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled  # [B,H]

class FusionHead(nn.Module):
    """[h_t, h_k, h_t⊙h_k, |h_t-h_k|] -> MLP -> logit"""
    def __init__(self, H: int, hidden=512, p=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4*H, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, 1)
        )

    def forward(self, h_t, h_k):
        x = torch.cat([h_t, h_k, h_t * h_k, torch.abs(h_t - h_k)], dim=-1)
        return self.mlp(x).squeeze(-1)

class Detector(nn.Module):
    """
    key_mode:
      - "hash_mlp": expand-to-H via HMAC, then frozen FFN
      - "direct":   char-embed baseline
    """
    def __init__(self, model_name="roberta-large", key_mode="hash_mlp", ffn_layer_idx=0):
        super().__init__()
        assert key_mode in {"hash_mlp","direct"}
        self.key_mode = key_mode
        self.text = TextEncoder(model_name=model_name)
        H = self.text.hidden_size
        backbone = self.text.backbone
        if key_mode == "hash_mlp":
            self.key_proj = KeyProjectorFromFFN(backbone, layer_idx=ffn_layer_idx)
        else:
            self.key_proj = KeyDirectCharEmbed(H=H, emb_dim=64)
        self.head = FusionHead(H=H, hidden=512, p=0.1)

    def key_embed(self, keys: List[str]) -> torch.Tensor:
        if self.key_mode == "hash_mlp":
            H = self.text.hidden_size
            v_batch = torch.stack([key_hmac_expand_to_H(k, H) for k in keys], dim=0).to(next(self.parameters()).device)
            with torch.no_grad():
                hkey = self.key_proj(v_batch)
            return hkey
        else:
            return self.key_proj(keys)

    def forward(self, input_ids, attention_mask, keys: List[str]):
        h_t = self.text(input_ids, attention_mask)  # [B,H]
        h_k = self.key_embed(keys)                  # [B,H]
        logit = self.head(h_t, h_k)                 # [B]
        return logit

# ===================== IO helpers =====================

def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rows.append(json.loads(line))
        return rows
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # try to find a likely list
            for k,v in data.items():
                if isinstance(v, list):
                    return v
            raise ValueError("JSON root is an object; expected a list of rows.")
        else:
            raise ValueError("Unsupported JSON structure")
    else:
        raise ValueError("Input must be .json or .jsonl")

def write_csv(rows: List[Dict[str, Any]], out_path: str):
    if not rows:
        # write empty file with header
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["idx","pred","prob"])
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

# ===================== Inference =====================

@torch.no_grad()
def batched_predict(model: Detector,
                    tokenizer,
                    texts: List[str],
                    key: str,
                    batch_size: int = 32,
                    max_length: int = 384,
                    device: torch.device = None,
                    threshold: float = 0.5) -> List[Dict[str, Any]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    outputs: List[Dict[str, Any]] = []
    N = len(texts)
    for i in range(0, N, batch_size):
        chunk = texts[i:i+batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        keys = [key] * len(chunk)

        logits = model(input_ids, attention_mask, keys)  # [B]
        probs = torch.sigmoid(logits).cpu().tolist()
        preds = [1 if p >= threshold else 0 for p in probs]

        for j, (p, lbl) in enumerate(zip(probs, preds)):
            outputs.append({"idx": i + j, "pred": lbl, "prob": float(p)})
    return outputs

def main():
    ap = argparse.ArgumentParser("Watermark detection inference (feature-fusion classifier)")
    ap.add_argument("--ckpt", type=str,default="trained_models_Main_llama2_sentence_added_Mistral_DeepSeek_universal_3keys/detector.pt", help="Path to detector checkpoint.pt")
    ap.add_argument("--input", type=str,default="Test_Main_llama2_varying_0.92.jsonl", help="Input test dataset file in jsonl format")
    ap.add_argument("--text-field", type=str, default="Watermarked_output", choices=["Original_output", "Watermarked_output", "Watermarked_summary","paraphrased_response","summary"], help="Field in input containing the text")
    ap.add_argument("--key", type=str, default="Your Key", choices=["Adaptive_key_v1", "Mistral_user_002", "DeepSeek_LLM"], help="Secret key string for detection")
    ap.add_argument("--out", type=str, default="Your Output CSV path", help="Output CSV path")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for label 1")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use")
    # Optional overrides for testing different LMs; if omitted, values from checkpoint are used
    ap.add_argument("--model_name", type=str, default=None, help="Backbone model name to use (overrides checkpoint if provided)")
    ap.add_argument("--ffn_layer_idx", type=int, default=None, help="FFN layer index to reuse for hash_mlp (overrides checkpoint if provided)")
    args = ap.parse_args()

    # Load rows & extract texts
    rows = read_json_or_jsonl(args.input)
    texts = []
    for r in rows:
        if args.text_field not in r:
            raise KeyError(f"Row missing '{args.text_field}': {r}")
        texts.append(str(r[args.text_field]))

    # Load checkpoint & rebuild model/tokenizer
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # Use overrides if provided; otherwise load from checkpoint (keeps training-time architecture)
    model_name     = args.model_name or ckpt.get("model_name", "roberta-large")
    key_mode       = ckpt.get("key_mode", "hash_mlp")
    ffn_layer_idx  = args.ffn_layer_idx if args.ffn_layer_idx is not None else ckpt.get("ffn_layer_idx", 0)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = Detector(model_name=model_name, key_mode=key_mode, ffn_layer_idx=ffn_layer_idx)
    # Try strict load first; if it fails due to shape mismatch (different LM), surface a clear error
    try:
        model.load_state_dict(ckpt["model"], strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to load checkpoint weights strictly. This usually happens when the provided --model_name "
            "differs from the model used during training (hidden sizes/layers differ). "
            "Re-run without --model_name to use the training backbone, or train a checkpoint for the desired backbone.\n"
            f"Details: {e}"
        )

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Processing {len(texts)} texts with key: {args.key}")
    
    results = batched_predict(model, tokenizer, texts, args.key,
                              batch_size=args.batch_size,
                              max_length=args.max_length,
                              device=device,
                              threshold=args.threshold)

    # Save CSV
    write_csv(results, args.out)
    
    # Print summary statistics
    total_samples = len(results)
    watermarked_count = sum(1 for r in results if r["pred"] == 1)
    watermarked_rate = watermarked_count / total_samples if total_samples > 0 else 0
    avg_prob = sum(r["prob"] for r in results) / total_samples if total_samples > 0 else 0
    
    print(f"\n=== Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Watermarked (pred=1): {watermarked_count} ({watermarked_rate:.2%})")
    print(f"Not watermarked (pred=0): {total_samples - watermarked_count} ({1-watermarked_rate:.2%})")
    print(f"Average probability: {avg_prob:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Saved predictions to: {args.out}")

if __name__ == "__main__":
    main()
