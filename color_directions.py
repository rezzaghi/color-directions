# %% [markdown]
# # Do red / blue / green exist as directions in the residual stream?
# Qwen2.5-1.5B-Instruct, raw text. Run cell by cell (VS Code "Run Cell").

# %%
# ---------- 1. Setup ----------
import json
import random
from math import comb
from pathlib import Path

import torch as t
from torch import Tensor
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer

t.set_grad_enabled(False)
device = "cuda" if t.cuda.is_available() else "cpu"
HERE = Path(__file__).parent if "__file__" in globals() else Path.cwd()

model = HookedTransformer.from_pretrained_no_processing(
    "Qwen/Qwen2.5-1.5B-Instruct", dtype=t.bfloat16, device=device
)
tok = model.tokenizer
COLORS = ["red", "blue", "green"]

# %%
# ---------- 2. Data (written by data.py) ----------
def load(name: str) -> list[dict]:
    return json.load(open(HERE / "data" / f"{name}.json", encoding="utf-8"))

explicit, implicit, controls = load("explicit"), load("implicit"), load("controls")
print(len(explicit), len(implicit), len(controls))

# %%
# ---------- 3. Tokenize + find the target token ----------
# Target position = last token overlapping the target word's character span.
# Right padding (can't affect earlier tokens) and a leading <|endoftext|> as attention sink.
BOS_ID = tok.convert_tokens_to_ids("<|endoftext|>")

def tokenize(examples: list[dict]) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"], Int[Tensor, "batch"]]:
    texts = [ex["text"] for ex in examples]
    enc = tok(texts, padding=True, padding_side="right", return_offsets_mapping=True,
              add_special_tokens=False, return_tensors="pt")
    positions = []
    for i, ex in enumerate(examples):
        start = ex["text"].rindex(ex["target"])
        end = start + len(ex["target"])
        overlapping = [
            j for j, (a, b) in enumerate(enc.offset_mapping[i].tolist())
            if enc.attention_mask[i, j] and a < end and b > start
        ]
        positions.append(max(overlapping))
    bos = t.full((len(texts), 1), BOS_ID)
    ids = t.cat([bos, enc.input_ids], dim=1)
    mask = t.cat([t.ones_like(bos), enc.attention_mask], dim=1)
    return ids.to(device), mask.to(device), t.tensor(positions) + 1

for ex in explicit + implicit + controls:
    ids, _, pos = tokenize([ex])
    piece = tok.decode(ids[0, pos]).strip()
    assert piece and ex["target"].endswith(piece), (ex, piece)
print("target positions OK")

# %%
# ---------- 4. Cache the residual stream at the target token, every layer ----------
def get_resid_acts(examples: list[dict], batch_size: int = 32) -> Float[Tensor, "batch layer d_model"]:
    out = []
    for i in range(0, len(examples), batch_size):
        ids, mask, pos = tokenize(examples[i : i + batch_size])
        _, cache = model.run_with_cache(
            ids, attention_mask=mask, names_filter=lambda name: name.endswith("hook_resid_pre")
        )
        rows = t.arange(len(pos))
        acts = t.stack([cache["resid_pre", L][rows, pos] for L in range(model.cfg.n_layers)], dim=1)
        out.append(acts.float().cpu())
    return t.cat(out)

explicit_acts = get_resid_acts(explicit)
implicit_acts = get_resid_acts(implicit)
control_acts = get_resid_acts(controls)

# Sanity check: layer 0 is the raw embedding, so every " red" must be the same vector.
reds = explicit_acts[[i for i, ex in enumerate(explicit) if ex["label"] == "red"]]
print("layer 0 spread of ' red':", (reds[:, 0] - reds[0, 0]).abs().max().item())

t.save(
    {"explicit": explicit_acts, "implicit": implicit_acts, "controls": control_acts},
    HERE / "activations.pt",
)

# %%
# ---------- 5. Color directions from the explicit set ----------
acts = t.load(HERE / "activations.pt")
X_exp, X_imp, X_ctl = acts["explicit"], acts["implicit"], acts["controls"]
y_exp = t.tensor([COLORS.index(ex["label"]) for ex in explicit])
y_imp = t.tensor([COLORS.index(ex["label"]) for ex in implicit])

# Hold out 10 templates (split by template, so minimal pairs never straddle train/test).
test_templates = set(random.Random(0).sample(range(len(explicit) // 3), 10))
is_test = t.tensor([ex["group"] in test_templates for ex in explicit])

mu = t.stack([X_exp[~is_test & (y_exp == c)].mean(0) for c in range(3)])
grand = mu.mean(0)
dirs = mu - grand
dir_hat = dirs / dirs.norm(dim=-1, keepdim=True)

cos = lambda a, b: (a * b).sum(-1)
for L in [0, 7, 14, 21, 27]:
    print(f"layer {L:2d}  cos(red,blue)={cos(dir_hat[0, L], dir_hat[1, L]):+.2f}"
          f"  cos(red,green)={cos(dir_hat[0, L], dir_hat[2, L]):+.2f}"
          f"  cos(blue,green)={cos(dir_hat[1, L], dir_hat[2, L]):+.2f}")

def scores(X: Float[Tensor, "n layer d"], center: Float[Tensor, "layer d"]) -> Float[Tensor, "n 3 layer"]:
    return t.einsum("nld,cld->ncl", X - center, dir_hat)

def accuracy(X, y, center) -> Float[Tensor, "layer"]:
    return (scores(X, center).argmax(1) == y[:, None]).float().mean(0)

# %%
# ---------- 6. Test: held-out explicit, implicit objects, controls ----------
# Implicit vectors are centered on their own (color-balanced) mean, removing the
# noun-vs-adjective offset. The explicit-centered column shows the bias without it.
center_imp = X_imp.mean(0)
acc_exp = accuracy(X_exp[is_test], y_exp[is_test], grand)
acc_imp = accuracy(X_imp, y_imp, center_imp)
acc_imp_naive = accuracy(X_imp, y_imp, grand)

strength = lambda X, c: scores(X, c).norm(dim=1).mean(0)
ctl_ratio = strength(X_ctl, center_imp) / strength(X_imp, center_imp)

print("layer | explicit held-out | implicit | implicit, explicit-centered | control/implicit strength")
for L in range(model.cfg.n_layers):
    print(f"{L:5d} | {acc_exp[L]:17.2f} | {acc_imp[L]:8.2f} | {acc_imp_naive[L]:26.2f} | {ctl_ratio[L]:.2f}")

# %%
# ---------- 7. Best layer, object by object ----------
best = acc_imp.argmax().item()
objects = list(dict.fromkeys(ex["target"] for ex in implicit))
S = scores(X_imp, center_imp)[:, :, best]
n_right = 0
print(f"best layer = {best}  (sentence accuracy {acc_imp[best]:.2f})\n")
for obj in objects:
    idx = [i for i, ex in enumerate(implicit) if ex["target"] == obj]
    s = S[idx].mean(0)
    truth, pred = implicit[idx[0]]["label"], COLORS[s.argmax()]
    n_right += truth == pred
    print(f"{implicit[idx[0]]['group']:11s} {obj:20s} true={truth:5s} pred={pred:5s} {'ok' if truth == pred else 'XX'}"
          f"   scores r/b/g = {s[0]:+6.1f} {s[1]:+6.1f} {s[2]:+6.1f}")

# Binomial p-value vs guessing (optimistic: the layer was picked on this same data).
p = sum(comb(30, k) * (1 / 3) ** k * (2 / 3) ** (30 - k) for k in range(n_right, 31))
print(f"\nobjects correct: {n_right}/30   p = {p:.1e}")
