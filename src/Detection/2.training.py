import os
# Ensure the HF_HOME environment variable points to your desired cache location
os.environ["HF_TOKEN"] = "Your Hugging Face Token"
cache_dir = 'Your Cache Directory'
import json, math, hmac, hashlib, random, argparse, string
from typing import List, Dict, Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------- Utils ---------------------

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def bce_from_logits(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels.float())

def accuracy(probs, labels, thr=0.5):
    return ((probs >= thr).long() == labels.long()).float().mean().item()

# --------------------- Data ---------------------

class PairDataset(Dataset):
    """
    JSONL where each line: {"text": "...", "key": "key_string", "label": 0/1}
    """
    def __init__(self, path: str):
        self.rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                assert "text" in obj and "key" in obj and "label" in obj
                self.rows.append(obj)

    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]

# --------------------- Key encoders ---------------------

def hmac_blocks(key: str, total_bytes: int) -> bytes:
    """HMAC-SHA256(key, counter[4b]) blocks until we have total_bytes."""
    key_b = key.encode("utf-8")
    out = bytearray(); ctr = 0
    while len(out) < total_bytes:
        msg = ctr.to_bytes(4, "big", signed=False)
        out.extend(hmac.new(key_b, msg, hashlib.sha256).digest())
        ctr += 1
    return bytes(out[:total_bytes])

def key_expand_to_H(key: str, H: int) -> torch.Tensor:
    raw = hmac_blocks(key, H)
    v = torch.tensor(list(raw), dtype=torch.float32) / 255.0  # [H] in [0,1]
    # Optional: shift/scale roughly to zero-mean
    v = (v - 0.5) * 2.0
    return v

class KeyProjectorFromFFN(nn.Module):
    """
    Preferred (Hash-MLP) projector:
    Expand to H via HMAC counters, then pass through a *frozen* FFN
    copied from the transformer (intermediate->GELU->output).
    Deterministic; no training on the key path.
    """
    def __init__(self, backbone: AutoModel, layer_idx: int = 0):
        super().__init__()
        enc_layer = backbone.encoder.layer[layer_idx]
        in_dense = enc_layer.intermediate.dense
        out_dense = enc_layer.output.dense
        self.intermediate = nn.Linear(in_dense.in_features, in_dense.out_features, bias=True)
        self.output = nn.Linear(out_dense.in_features, out_dense.out_features, bias=True)
        with torch.no_grad():
            self.intermediate.weight.copy_(in_dense.weight); self.intermediate.bias.copy_(in_dense.bias)
            self.output.weight.copy_(out_dense.weight);     self.output.bias.copy_(out_dense.bias)
        for p in self.parameters(): p.requires_grad = False
        self.act = nn.GELU()

    @torch.no_grad()
    def forward(self, vH: torch.Tensor) -> torch.Tensor:
        return self.output(self.act(self.intermediate(vH)))  # [B,H]

class KeyDirectCharEmbed(nn.Module):
    """
    Ablation (direct embedding): embed chars and mean-pool.
    WARNING: this is exactly the kind of thing that can memorize strings; use only as baseline.
    """
    def __init__(self, H: int, vocab: str = string.printable, emb_dim: int = 64):
        super().__init__()
        self.vocab = vocab
        self.stoi = {ch:i for i,ch in enumerate(vocab)}
        self.emb = nn.Embedding(num_embeddings=len(vocab), embedding_dim=emb_dim)
        self.toH = nn.Linear(emb_dim, H)

    def forward(self, keys: List[str]) -> torch.Tensor:
        # convert each key to indices (clip/keep all)
        idxs = []
        for k in keys:
            ids = [self.stoi.get(ch, 0) for ch in k]
            if len(ids) == 0: ids = [0]
            idxs.append(torch.tensor(ids, dtype=torch.long))
        # pad & embed
        lens = [len(x) for x in idxs]
        maxlen = max(lens)
        pad = torch.stack([F.pad(x, (0, maxlen - len(x)), value=0) for x in idxs], dim=0)  # [B,L]
        mask = (torch.arange(maxlen).unsqueeze(0) < torch.tensor(lens).unsqueeze(1)).to(pad.device)  # [B,L]
        E = self.emb(pad)  # [B,L,emb]
        m = (E * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return self.toH(m)  # [B,H]

# --------------------- Text encoder ---------------------

class TextEncoder(nn.Module):
    def __init__(self, model_name="roberta-large", train_backbone=True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size
        if not train_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # [B,T,H]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled  # [B,H]

# --------------------- Fusion head ---------------------

class FusionHead(nn.Module):
    """
    f = [h_t, h_k, h_t ⊙ h_k, |h_t - h_k|] -> MLP -> logit
    """
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

# --------------------- Full model ---------------------

class Detector(nn.Module):
    """
    key_mode:
      - "hash_mlp": expand-to-H via HMAC, then frozen FFN
      - "direct":   char-embed baseline (trainable)
    """
    def __init__(self, model_name="roberta-large", key_mode="hash_mlp",
                 train_backbone=True, ffn_layer_idx=0, char_emb_dim=64, fusion_hidden=512, p=0.1):
        super().__init__()
        assert key_mode in {"hash_mlp", "direct"}
        self.key_mode = key_mode

        self.text = TextEncoder(model_name=model_name, train_backbone=train_backbone)
        H = self.text.hidden_size

        backbone = self.text.backbone  # reuse weights
        if key_mode == "hash_mlp":
            self.key_proj = KeyProjectorFromFFN(backbone, layer_idx=ffn_layer_idx)
        else:
            self.key_proj = KeyDirectCharEmbed(H=H, emb_dim=char_emb_dim)

        self.head = FusionHead(H=H, hidden=fusion_hidden, p=p)

    def key_embed(self, keys: List[str]) -> torch.Tensor:
        if self.key_mode == "hash_mlp":
            # Deterministic: expand each key to H and run frozen FFN
            H = self.text.hidden_size
            v_batch = torch.stack([key_expand_to_H(k, H) for k in keys], dim=0).to(next(self.parameters()).device)
            with torch.no_grad():
                hkey = self.key_proj(v_batch)
            return hkey
        else:
            # Trainable char-embed baseline
            return self.key_proj(keys)

    def forward(self, input_ids, attention_mask, keys: List[str]):
        h_t = self.text(input_ids, attention_mask)       # [B,H]
        h_k = self.key_embed(keys)                       # [B,H]
        logit = self.head(h_t, h_k)                      # [B]
        return logit, (h_t, h_k)

# --------------------- Contrastive loss (sampled) ---------------------

import random
from typing import List as _List

def contrastive_infonce_sampled(
    h_t: torch.Tensor,           # [B,H] text embeddings
    h_k: torch.Tensor,           # [B,H] key embeddings
    labels: torch.Tensor,        # [B] 0/1; positives are label==1
    keys: _List[str],            # len B; key string per row
    temperature: float = 0.07,
    neg_per_pos: int = 8,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    Sampled symmetric InfoNCE:
      - anchors: only rows with label==1
      - positives: (t_i,k_i)
      - negatives: keys from different key-strings (k_j != k_i), sampled
    Complexity: O(B_+ * neg_per_pos) instead of O(B^2).
    """
    device = h_t.device

    pos_idx = torch.nonzero(labels == 1, as_tuple=False).squeeze(-1)
    if pos_idx.numel() < 1:
        return h_t.new_tensor(0.0)

    ht = F.normalize(h_t, dim=-1)
    hk = F.normalize(h_k, dim=-1)

    # group row indices by key string
    key_to_rows: Dict[str, _List[int]] = {}
    for i, k in enumerate(keys):
        key_to_rows.setdefault(k, []).append(i)

    # need at least 2 distinct keys
    if len(key_to_rows) < 2:
        return h_t.new_tensor(0.0)

    def _loss_one_direction(anchors: torch.Tensor, query_is_text: bool) -> torch.Tensor:
        total = []
        for i in anchors.tolist():
            k_i = keys[i]
            # negatives: any row with different key string
            neg_pool = []
            for kk, idxs in key_to_rows.items():
                if kk != k_i:
                    neg_pool.extend(idxs)
            if len(neg_pool) == 0:
                continue

            if len(neg_pool) >= neg_per_pos:
                neg_js = random.sample(neg_pool, neg_per_pos)
            else:
                neg_js = [random.choice(neg_pool) for _ in range(neg_per_pos)]

            if query_is_text:
                q = ht[i]
                pos = torch.matmul(q, hk[i])
                neg = torch.stack([torch.matmul(q, hk[j]) for j in neg_js])
            else:
                q = hk[i]
                pos = torch.matmul(q, ht[i])
                neg = torch.stack([torch.matmul(q, ht[j]) for j in neg_js])

            logits = torch.cat([pos.view(1), neg], dim=0) / temperature
            target = logits.new_zeros((), dtype=torch.long)  # class 0 is the positive
            total.append(F.cross_entropy(logits.view(1, -1), target.view(1)))

        if len(total) == 0:
            return h_t.new_tensor(0.0)
        return torch.stack(total).mean()

    loss_t2k = _loss_one_direction(pos_idx, query_is_text=True)

    if symmetric:
        loss_k2t = _loss_one_direction(pos_idx, query_is_text=False)
        return 0.5 * (loss_t2k + loss_k2t)
    else:
        return loss_t2k

# --------------------- Collator ---------------------

class Collator:
    def __init__(self, tokenizer, max_length=384, device=None,
                 add_wrong_key_negs=False, wrong_key_ratio=0.5):
        self.tok = tokenizer
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.add_wrong = add_wrong_key_negs
        self.wrong_ratio = wrong_key_ratio

    def __call__(self, rows: List[Dict]):
        if self.add_wrong and len(rows) > 1:
            uniq_keys = sorted({r["key"] for r in rows})
            extra = []
            for r in rows:
                if r["label"] == 1 and random.random() < self.wrong_ratio:
                    choices = [k for k in uniq_keys if k != r["key"]]
                    if choices:
                        extra.append({"text": r["text"], "key": random.choice(choices), "label": 0})
            if extra:
                rows = rows + extra
                random.shuffle(rows)

        texts = [r["text"] for r in rows]
        keys  = [r["key"]  for r in rows]
        labels = torch.tensor([r["label"] for r in rows], dtype=torch.long, device=self.device)

        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        batch = {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
            "keys": keys,
            "labels": labels
        }
        return batch

# --------------------- Train / Eval ---------------------

def train_epoch(model, loader, optimizer, scheduler=None, lam_contrastive=0.0, grad_clip=1.0, epoch=1, total_epochs=1):
    model.train()
    tot, tot_loss, tot_acc = 0, 0.0, 0.0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for batch_idx, batch in enumerate(pbar):
        logits, (h_t, h_k) = model(batch["input_ids"], batch["attention_mask"], batch["keys"])
        loss_cls = bce_from_logits(logits, batch["labels"])
        loss = loss_cls

        if lam_contrastive > 0:
            loss_con = contrastive_infonce_sampled(
                h_t, h_k, batch["labels"], batch["keys"],
                temperature=0.07, neg_per_pos=8, symmetric=True
            )
            loss = loss + lam_contrastive * loss_con

        optimizer.zero_grad(); loss.backward()
        if grad_clip: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step(); 
        if scheduler: scheduler.step()

        probs = torch.sigmoid(logits).detach()
        acc = accuracy(probs, batch["labels"])
        n = batch["labels"].size(0)
        tot += n; tot_loss += loss.item() * n; tot_acc += acc * n
        
        current_loss = tot_loss / max(1, tot)
        current_acc = tot_acc / max(1, tot)
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    pbar.close()
    return {"loss": tot_loss/max(1,tot), "acc": tot_acc/max(1,tot)}

@torch.no_grad()
def evaluate(model, loader, desc="Evaluating"):
    model.eval()
    tot, tot_loss, tot_acc = 0, 0.0, 0.0
    
    pbar = tqdm(loader, desc=desc, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for batch in pbar:
        logits, (h_t, h_k) = model(batch["input_ids"], batch["attention_mask"], batch["keys"])
        loss = bce_from_logits(logits, batch["labels"])
        probs = torch.sigmoid(logits)
        acc = accuracy(probs, batch["labels"])
        n = batch["labels"].size(0)
        tot += n; tot_loss += loss.item() * n; tot_acc += acc * n
        
        current_loss = tot_loss / max(1, tot)
        current_acc = tot_acc / max(1, tot)
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}'
        })
    
    pbar.close()
    return {"loss": tot_loss/max(1,tot), "acc": tot_acc/max(1,tot)}

# --------------------- Inference ---------------------

@torch.no_grad()
def predict_proba(model, tokenizer, text: str, key: str, max_length=384, device=None) -> float:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device); attention_mask = enc["attention_mask"].to(device)
    logits, _ = model(input_ids, attention_mask, [key])
    return torch.sigmoid(logits).item()

# --------------------- CLI ---------------------
 
def main():
    ap = argparse.ArgumentParser("End-to-end key-conditioned watermark detector with feature-fusion")
    ap.add_argument("--train", type=str, default="Train_Main_llama2_varying_0.92.jsonl") # File name for train dataset
    ap.add_argument("--val",   type=str, default="Test_Main_llama2_varying_0.92.jsonl") # File name for test dataset
    ap.add_argument("--key_mode", type=str, default="hash_mlp", choices=["hash_mlp","direct"]) # Choose hash_mlp
    ap.add_argument("--model_name", type=str, default="roberta-large")
    ap.add_argument("--ffn_layer_idx", type=int, default=12, help="Which transformer layer FFN to reuse for Hash-MLP. 10-14 is middle layer for large models.")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train_backbone", action="store_true", default=True, help="Train the backbone")
    ap.add_argument("--lam_contrastive", type=float, default=0.15, help="Weight for InfoNCE text↔key loss (0 to disable)")
    ap.add_argument("--on_the_fly_wrong_key_negs", action="store_true", default=True, help="Add wrong key negatives on the fly")
    ap.add_argument("--wrong_key_ratio", type=float, default=0.5, help="Ratio of wrong key negatives to add")
    ap.add_argument("--save", type=str, default="detector.pt")
    ap.add_argument("--output_dir", type=str, default="trained_models_Main_llama2_varying_0.92", help="Base output directory") # Base output directory
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    args.save = os.path.join(args.output_dir, args.save)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_ds = PairDataset(args.train); val_ds = PairDataset(args.val)

    # Separate collators: train may add wrong-key negatives; val is clean
    train_collate = Collator(tok, max_length=args.max_length, device=device,
                             add_wrong_key_negs=args.on_the_fly_wrong_key_negs,
                             wrong_key_ratio=args.wrong_key_ratio)
    val_collate   = Collator(tok, max_length=args.max_length, device=device,
                             add_wrong_key_negs=False, wrong_key_ratio=0.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=train_collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=val_collate)

    model = Detector(model_name=args.model_name, key_mode=args.key_mode,
                     train_backbone=args.train_backbone, ffn_layer_idx=args.ffn_layer_idx).to(device)

    # optimizer: smaller LR for backbone if fine-tuning
    if args.train_backbone:
        bb_params = list(model.text.backbone.parameters())
        other = [p for n,p in model.named_parameters() if not n.startswith("text.backbone.")]
        optimizer = torch.optim.AdamW([
            {"params": bb_params, "lr": args.lr},
            {"params": other, "lr": args.lr * 5.0 if args.key_mode=="direct" else args.lr}
        ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = args.epochs * math.ceil(len(train_ds) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best = 0.0
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    print(f"Model: {args.model_name}, Key mode: {args.key_mode}")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}")
    print(f"Target directory: {args.output_dir}")
    print("=" * 80)
    
    for ep in range(1, args.epochs+1):
        print(f"\nEpoch {ep}/{args.epochs}")
        print("-" * 40)
        
        # Training
        tr = train_epoch(model, train_loader, optimizer, scheduler, 
                        lam_contrastive=args.lam_contrastive, epoch=ep, total_epochs=args.epochs)
        
        # Validation
        va = evaluate(model, val_loader, desc=f"Validating Epoch {ep}")
        
        # Print epoch summary
        print(f"\nEpoch {ep} Summary:")
        print(f"   Training   - Loss: {tr['loss']:.4f}, Accuracy: {tr['acc']:.4f}")
        print(f"   Validation - Loss: {va['loss']:.4f}, Accuracy: {va['acc']:.4f}")

        history["epoch"].append(ep)
        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])

        if va["acc"] > best:
            best = va["acc"]
            torch.save({"model": model.state_dict(),
                        "model_name": args.model_name,
                        "key_mode": args.key_mode,
                        "ffn_layer_idx": args.ffn_layer_idx}, args.save)
            print(f"New best validation accuracy: {va['acc']:.4f}")
            print(f"Model saved to: {args.save}")
        else:
            print(f"Best accuracy so far: {best:.4f}")

    # Save metrics JSON
    metrics_path = os.path.splitext(args.save)[0] + "_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Plot curves
    try:
        print("Generating training plots...")
        plt.figure(figsize=(12, 8))
        
        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(history["epoch"], history["train_loss"], label="Train Loss", marker='o')
        plt.plot(history["epoch"], history["val_loss"], label="Val Loss", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy plot
        plt.subplot(2, 2, 2)
        plt.plot(history["epoch"], history["train_acc"], label="Train Acc", marker='o')
        plt.plot(history["epoch"], history["val_acc"], label="Val Acc", marker='s')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Combined plot
        plt.subplot(2, 1, 2)
        plt.plot(history["epoch"], history["train_loss"], label="Train Loss", marker='o')
        plt.plot(history["epoch"], history["val_loss"], label="Val Loss", marker='s')
        plt.plot(history["epoch"], history["train_acc"], label="Train Acc", marker='^')
        plt.plot(history["epoch"], history["val_acc"], label="Val Acc", marker='v')
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Training Progress Overview")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = os.path.splitext(args.save)[0] + "_metrics.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {fig_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Best validation accuracy: {best:.4f}")
    print(f"Model saved to: {args.save}")
    print(f"Metrics saved to: {metrics_path}")
    if 'fig_path' in locals():
        print(f"Plot saved to: {fig_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
