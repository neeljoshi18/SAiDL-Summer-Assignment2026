# Task 3: Extending Adaptive Low-Rank Adaptation to Sequential Architectures

## Overview

LoRA and SoRA were designed with Transformer attention matrices as the primary target. Applying these methods to Mamba (SSM) and xLSTM requires rethinking which modules carry task-relevant information and how the sequential computation graph changes the adaptation dynamics.

---

## Why Methodology Must Change

In Transformers, the attention mechanism is fully parallelizable — all tokens interact simultaneously, and adapting Q/K/V projections changes which tokens attend to which. The gradient signal is dense and immediate.

In recurrent architectures (Mamba, xLSTM), computation is sequential. A weight update at time step t propagates through all future states h_{t+1}, h_{t+2}, ... via the recurrence. This means:

1. **The effective influence of an adapted layer spans the entire sequence**, not just one position.
2. **Recurrent weight adaptation has memory**: a low-rank update to a recurrent weight matrix changes every hidden state across the full sequence length.
3. **Gradient flow through time**: gradients of the loss w.r.t. early-sequence weights travel through many recurrent steps, potentially amplifying or vanishing. Lower rank adapters are more stable here.

---

## Mamba (Selective State Space Model)

### Architecture

Mamba's core selective SSM computes:
```
h_t = A(x_t) * h_{t-1} + B(x_t) * x_t
y_t = C(x_t) * h_t
```

The selectivity (input-dependence of A, B, C) is implemented through learned linear projections:
- `in_proj`: expands input dimension (d_model → 2 * d_inner)
- `x_proj`: projects for B, C, and Δ (time-step)
- `dt_proj`: projects Δ (controls state update rate)
- `out_proj`: contracts output (d_inner → d_model)

### Proposed LoRA Targets

**Primary: `in_proj` and `out_proj`**
- These are the input and output gates of the SSM block
- Analogous to `value_proj` and `out_proj` in Transformers
- Adapting these changes what information enters/exits the state without touching the recurrence dynamics

**Secondary: `dt_proj`**
- Δ controls how much to "forget" the previous state and "update" with new input
- Adapting this allows the model to learn task-appropriate temporal selectivity
- For grammaticality: longer dependencies (verb-subject agreement across clauses) may need different Δ than pre-trained language modeling

**Not recommended: `x_proj`**
- Projects to B and C simultaneously — changes in one direction affect the other
- Less clean adaptation target; better handled indirectly through `in_proj`

### Proposed LoRA Configuration
```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["in_proj", "out_proj", "dt_proj"],
    lora_dropout=0.1
)
```

### SoRA Adaptation
The gate vector `g` applied between A and B in SoRA's ΔW = B·diag(g)·A works naturally for all of Mamba's linear projections. For `dt_proj` specifically, sparse gates would learn which temporal update dimensions are task-relevant — components with g=0 retain pre-trained temporal dynamics, while nonzero gates adapt them.

---

## xLSTM (Extended Long Short-Term Memory)

### Architecture

xLSTM introduces two cell types:

**sLSTM (scalar memory, exponential gating):**
```
i_t = exp(w_i * x_t + r_i * h_{t-1} + b_i)   # input gate
f_t = exp(w_f * x_t + r_f * h_{t-1} + b_f)   # forget gate
o_t = sigmoid(w_o * x_t + r_o * h_{t-1} + b_o)  # output gate
c_t = f_t * c_{t-1} + i_t * z_t
h_t = o_t * tanh(c_t / max(|c_t|, 1))
```

**mLSTM (matrix memory, covariance update):**
```
C_t = f_t * C_{t-1} + i_t * v_t * k_t^T   # matrix state update
h_t = o_t * norm(C_t * q_t)                 # read from matrix state
```

### Proposed LoRA Targets

**sLSTM targets: gate weight matrices**
- `W_i`, `W_f`, `W_o` (input, forget, output gate projections from x_t)
- These control how information flows through time — most task-sensitive component
- Avoid adapting recurrent weights `R_i`, `R_f` directly (amplified via BPTT)
- If adapting `R`: use lower rank (r=2 or 4) and stronger dropout

**mLSTM targets: associative memory projections**
- `W_q`, `W_k`, `W_v` — direct analogs of Transformer Q/K/V
- mLSTM's matrix memory is essentially linear attention; LoRA on W_q/W_k/W_v is the natural adaptation
- Same intuition as Transformer LoRA: change what the model queries and what it stores

### Proposed LoRA Configuration
```python
# sLSTM layers
LoraConfig(r=8, target_modules=["W_i", "W_f", "W_o"])

# mLSTM layers  
LoraConfig(r=8, target_modules=["W_q", "W_k", "W_v"])
```

### Key Difference from Transformer Adaptation

In Transformers, all layers can be adapted with the same rank because gradients are dense and parallel. In xLSTM:
- Early layers have longer gradient paths — use smaller rank or stronger regularization
- mLSTM layers with matrix memory are more stable (gradient bounded by normalization) — can use larger rank
- sLSTM layers with exponential gates can have gradient spikes — use gradient clipping more aggressively (max_norm=0.5 instead of 1.0)

### SoRA Adaptation for xLSTM

For mLSTM's W_q/W_k/W_v, SoRA's gating applies directly — identical to Transformer SoRA. For sLSTM's gate matrices, the sparse gating is particularly meaningful: a gate value of zero means that rank component of the gate adaptation is turned off, preserving the pre-trained temporal dynamics for that dimension. This allows the model to selectively update only the gating patterns that need to change for grammaticality judgment.

---

## Summary Table

| Architecture | Primary LoRA Targets | Rationale | Special Considerations |
|-------------|---------------------|-----------|----------------------|
| Transformer | query, key, value, out_proj | Attention mechanism | Standard; stable |
| Mamba | in_proj, out_proj, dt_proj | Input gate + temporal | dt_proj for selectivity |
| xLSTM sLSTM | W_i, W_f, W_o | Gate matrices | Lower rank for recurrent |
| xLSTM mLSTM | W_q, W_k, W_v | Associative memory | Direct Transformer analog |

---

## Implementation Notes

For Mamba:
```bash
pip install mamba-ssm causal-conv1d  # requires CUDA 11.6+
# OR: use HuggingFace transformers which wraps Mamba
from transformers import MambaForCausalLM
```

For xLSTM:
```bash
git clone https://github.com/NX-AI/xlstm.git
cd xlstm && pip install -e .
```

Both can be wrapped with HuggingFace PEFT's LoraConfig by specifying the appropriate `target_modules` strings that match the module names in each architecture's implementation.
