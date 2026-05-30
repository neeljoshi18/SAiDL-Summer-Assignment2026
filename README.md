# SAiDL Summer Assignment 2026 — Sparsity & Optimization

**[Your Name] | SAiDL Summer Induction 2026**

This repository contains my submission for the **Sparsity & Optimization** domain track of the SAiDL Summer 2026 Induction Assignment.

---

## Track: Sparsity & Optimization

### What Was Done

A systematic comparison of three parameter-efficient fine-tuning (PEFT) methods — **LoRA**, **AdaLoRA**, and **SoRA** — applied to the CoLA grammatical acceptability task using DeBERTa-v3-base as the backbone. Additionally, a theoretical and empirical investigation of the optimizer-level mechanism (proximal gradient vs. SGD subgradient) that underlies SoRA's sparsity mechanism.

---

## Results Summary

| Method | Best Val MCC | Trainable Params | Mean Eff. Rank | Avg Epoch |
|--------|-------------|-----------------|----------------|-----------|
| LoRA (r=8) | 0.1557 | 296,450 (0.16%) | 6.809 / 8 | 181s |
| AdaLoRA (init_r=12 → target_r=8) | 0.0901 | 665,522 (0.36%) | 6.315 / 8 | 189s |
| **SoRA (r=8, λ=0.1)** | **0.2294** | 444,194 (0.24%) | 6.813 / 8 | 208s |

Primary metric: **MCC (Matthews Correlation Coefficient)** — used because CoLA has 69%/31% class imbalance. Accuracy is misleading; MCC = 0 for any model that predicts a single class.

---

## Repository Structure

```
SAiDL-Summer-Assignment2026/
├── README.md
├── report/
│   ├── main.tex              ← Full LaTeX report
│   └── main.pdf              ← Compiled PDF
├── task1/
│   ├── lora_cola.ipynb       ← LoRA training notebook (Colab)
│   ├── adalora_cola.ipynb    ← AdaLoRA training notebook (Colab)
│   ├── sora_cola.ipynb       ← SoRA training notebook (Colab)
│   └── results/
│       ├── lora_results.json
│       ├── adalora_results.json
│       └── sora_results.json
├── task2/
│   ├── numpy_implementation.py     ← Proximal vs SGD in NumPy
│   ├── pytorch_implementation.py   ← Proximal vs SGD in PyTorch
│   ├── numpy_comparison.png        ← Output plots
│   └── pytorch_comparison.png      ← Output plots
└── task3/
    └── proposal.md           ← Theoretical proposal for Mamba & xLSTM
```

---

## Task Descriptions

### Task 1 — LoRA vs AdaLoRA vs SoRA on CoLA

**Dataset:** CoLA (Corpus of Linguistic Acceptability), GLUE benchmark  
**Backbone:** `microsoft/deberta-v3-base` (184M parameters)  
**Metric:** MCC (primary), trainable parameters, effective rank, training time

**Key findings:**
- **SoRA > LoRA > AdaLoRA** on this small dataset
- AdaLoRA's rank pruning (steps 200–1000) destabilizes training — loss *increased* for 4 epochs before declining. CoLA is too small for the model to recover from adaptive pruning overhead.
- SoRA with λ=0.1 never induced true gate sparsity (all 288 gates stayed near 1.0). Task gradient defends all rank components. SoRA with zero sparsity acts as LoRA with additional per-dimension scaling → better performance.
- Effective rank analysis: LoRA uses 85% of its capacity on average (mean 6.81/8), but with high variance (4.4–7.8), motivating adaptive approaches on larger datasets.

**Setup:** Google Colab T4 GPU. All three methods trained with identical hyperparameters (lr=1e-4, AdamW eps=1e-6, gradient clipping, class weights [1.69, 0.71], linear warmup scheduler) for fair comparison.

---

### Task 2 — Proximal Gradient vs SGD Subgradient

**Problem:** `minimize 0.5 * ||Ag - b||^2 + λ * ||g||_1`  
with known sparse ground truth `g_true = [1.5, 0, 0, -0.8, 0, 0, 0.3, 0, 0, -1.2]` (6 zeros).

**SGD subgradient update (derived):**
```
g_new = g - η * (∂L_task/∂g + λ * sign(g))
```
where `sign(0) = 0` by convention.

**Proximal update:**
```
g_temp = g - η * ∂L_task/∂g
g_new  = sign(g_temp) * max(|g_temp| - λη, 0)
```

**Results:**

| Framework | Method | Near-zeros found | Final loss |
|-----------|--------|-----------------|-----------|
| NumPy | Proximal | 5 / 6 | 0.380198 |
| NumPy | SGD Subgradient | 0 / 6 | 0.380466 |
| PyTorch | Proximal | 6 / 6 | 0.378609 |
| PyTorch | SGD Subgradient | 0 / 6 | 0.378873 |

Both methods converge to nearly identical objective values. The difference is purely structural: proximal produces exact zeros, SGD cannot (components oscillate near zero with magnitude O(ηλ)).

**Why:** The proximal operator analytically solves `argmin_u ||u - g_temp||² + λη||u||₁`, whose closed-form solution is zero when `|g_temp| ≤ λη`. SGD's continuous update can only push toward zero asymptotically.

**Subgradient choice at zero:** Using `sign(0) = 0` vs `±1` does not close the gap. The fundamental limitation is the continuous gradient update, not the subgradient convention.

---

### Task 3 — Extension to Sequential Architectures (Theoretical Proposal)

Proposed LoRA/SoRA adaptation targets for Mamba SSM and xLSTM. See `task3/proposal.md` and the full discussion in the LaTeX report (Section 6).

---

## Running the Code

### Task 2 (local)
```bash
pip install numpy torch matplotlib
python task2/numpy_implementation.py
python task2/pytorch_implementation.py
```

### Task 1 (Google Colab recommended)
Open the notebooks in `task1/` on Google Colab with a T4 GPU runtime.

Required packages:
```
pip install torchao --upgrade
pip install transformers peft datasets evaluate scikit-learn accelerate
```

Important DeBERTa-v3 quirks:
- Always use `use_fast=False` in tokenizer
- Always load model with `torch_dtype=torch.float32`
- Learning rate must be ≤ 1e-4 (disentangled attention overflows at higher lr)
- AdaLoRA requires `model.update_and_allocate(global_step)` inside training loop

---

## Report

The full LaTeX report is in `report/main.tex` covering:
- Background on PEFT, DeBERTa, CoLA, MCC
- Mathematical formulations for LoRA, AdaLoRA, SoRA
- Experimental results with training curves, effective rank analysis
- Analysis of why each method performed as it did
- Task 2: derivation, implementation, and theoretical comparison
- Task 3: proposed methodology for Mamba and xLSTM

---

## Contact

For any queries about this submission, contact via SAiDL Slack.
