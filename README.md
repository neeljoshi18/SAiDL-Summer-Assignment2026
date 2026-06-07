# SAiDL Summer Assignment 2026

This is my submission for the SAiDL Summer 2026 Induction. I attempted the
Core ML compulsory task and the Sparsity & Optimization domain task.

---

## Structure

```


---

## Core ML — Transformer Experiments

Built a modular Transformer language model trained on WikiText-2. The entire
point of the code structure was making components swappable — one config
field changes the attention type, PE type, or conv variant, everything else
stays fixed. Ran 9 complete experiments (10 epochs each) and one incomplete
due to Colab session limits.

**Model:** 4 layers, d_model=256, 4 heads, ~16M params  
**Task:** Next-token prediction on WikiText-2  
**Hardware:** Google Colab T4  

### Attention Variants

| Method | PPL | Mem (MB) | tok/s |
|--------|-----|----------|-------|
| Baseline (standard) | 635.99 | 8,013 | 64,247 |
| Sliding Window (w=64) | **547.15** | 8,016 | 62,936 |
| Linear Attention | 601.43 | 9,725 | 47,955 |
| GQA (n_kv=2) | 599.80 | 8,010 | 65,268 |

Sliding window beat full attention on Wikipedia, which makes sense in
retrospect — predicting the next word mostly needs the surrounding paragraph,
not the whole article. Linear attention ended up *slower* and used more memory
than standard at seq_len=512. This isn't a bug — it only becomes cheaper when
sequences are long enough (probably ~2048+) that the T×T attention matrix
outweighs the per-token state cost from the kernel trick.

### Positional Encodings + Extrapolation

| PE | PPL@512 | PPL@1024 | PPL@2048 |
|----|---------|----------|----------|
| Sinusoidal | 605.67 | 766.28 | 2,516.35 |
| Relative | 273.32 | 280.82 | 293.33 |
| RoPE | **205.44** | 219.14 | 258.15 |
| ALiBi | 206.73 | **204.99** | **204.18** |

This was the most interesting result. RoPE and ALiBi are roughly 3× better
than sinusoidal at training length, and that gap just comes from changing the
PE formula — same model, same data, same training. Sinusoidal falls apart at
2048 tokens (PPL jumps to 2516) because the model never saw those positions
during training. ALiBi actually improves at longer sequences, which is exactly
what it was designed to do.

Biggest thing I learned here: positional encoding choice has a larger effect
on perplexity than any attention architecture choice. I didn't expect that
going in.

### Conv Hybrids

Both used Sliding Window + RoPE as the base.

| Config | PPL |
|--------|-----|
| Conv Before Attention | 218.71 |
| Gated Conv FFN | didn't finish (Colab ran out) |

Conv-before-attention confirmed that local conv features + attention are
complementary. Also learned an important lesson here: symmetric Conv1d padding
lets the model see future tokens (the conv window at position t reaches t+1),
which looks like a data leakage bug. The model hit PPL ~1.3 after 3 epochs,
which is obviously wrong. Fixed with left-only causal padding.

---

## Sparsity & Optimization — PEFT on CoLA

Compared LoRA, AdaLoRA, and SoRA fine-tuning DeBERTa-v3-base on the CoLA
grammatical acceptability task. Also implemented and compared proximal
gradient vs SGD subgradient for ℓ1-regularized sparse recovery.

**Metric:** MCC (Matthews Correlation Coefficient) — CoLA has 69/31 class
imbalance so accuracy is meaningless, MCC is the right metric.

### Task 1 Results

| Method | Best MCC | Trainable Params | Mean Eff. Rank |
|--------|----------|-----------------|----------------|
| LoRA (r=8) | 0.1557 | 296,450 (0.16%) | 6.81 / 8 |
| AdaLoRA (r=12→8) | 0.0901 | 665,522 (0.36%) | 6.32 / 8 |
| SoRA (r=8, λ=0.1) | **0.2294** | 444,194 (0.24%) | 6.81 / 8 |

AdaLoRA underperformed because its rank pruning (steps 200–1000) destabilises
training. CoLA only has 8k examples so the model can't recover from continuous
pruning perturbations — loss actually went up for 4 epochs during the pruning
window. LoRA's fixed-rank simplicity is more stable here.

SoRA never produced real gate sparsity (all gates stayed near 1.0) because
the task gradient was too strong to let regularisation push gates to zero.
Without sparsity it effectively becomes LoRA with extra per-dimension scaling,
and that extra flexibility helped performance.

### Task 2 Results

Proximal gradient vs SGD subgradient on a synthetic LASSO problem with known
6-sparse ground truth.

| Method | Exact zeros found | Loss |
|--------|-------------------|------|
| NumPy Proximal | 5–6 / 6 | 0.3802 |
| NumPy SGD | 0 / 6 | 0.3805 |
| PyTorch Proximal | 6 / 6 | 0.3786 |
| PyTorch SGD | 0 / 6 | 0.3789 |

Both methods get nearly identical objective values. The difference is entirely
structural — proximal produces exact zeros, SGD can't. This is because the
proximal operator analytically solves the ℓ1 subproblem (a closed-form
thresholding step), while SGD with a subgradient just oscillates around zero
without ever landing there.

### Task 3

Theoretical proposal for extending LoRA/SoRA to Mamba and xLSTM architectures.
Not implemented due to time constraints — see `task3/proposal.md`.

---

## Notes

- Everything ran on Google Colab T4 (free tier), so compute was limited
- Absolute perplexity numbers in Core ML are high because 10 epochs from
  random init isn't enough to converge — comparisons are what matter
- One experiment (Gated Conv FFN) didn't finish due to session limits
- For DeBERTa fine-tuning: lr must be ≤ 1e-4 and use_fast=False, otherwise
  the disentangled attention overflows and you get NaN loss
