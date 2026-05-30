"""
Task 2 — NumPy Implementation
Proximal Gradient vs SGD Subgradient for l1-Regularized Sparse Recovery

Problem: minimize 0.5 * ||Ag - b||^2 + lambda * ||g||_1
Ground truth g has 6 exact zeros — we test whether each optimizer recovers sparsity.

SAiDL Summer Induction Assignment 2026 — Sparsity & Optimization Track
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Problem Setup ──────────────────────────────────────────────────────────────
n = 10
A = np.random.randn(20, n)
g_true = np.array([1.5, 0.0, 0.0, -0.8, 0.0, 0.0, 0.3, 0.0, 0.0, -1.2])
b = A @ g_true + 0.01 * np.random.randn(20)

lambda_reg = 0.1
lr = 0.01
n_steps = 500

# ── Helper Functions ───────────────────────────────────────────────────────────
def task_loss(g):
    residual = A @ g - b
    return 0.5 * np.dot(residual, residual)

def task_grad(g):
    return A.T @ (A @ g - b)

def total_loss(g):
    return task_loss(g) + lambda_reg * np.sum(np.abs(g))


def proximal_update(g, lr, lam):
    """
    Proximal gradient step for l1 regularization.

    Step 1: Gradient step on task loss only:
        g_temp = g - lr * grad_task(g)

    Step 2: Soft-thresholding (proximal operator for l1):
        g_new = sign(g_temp) * max(|g_temp| - lam * lr, 0)

    The soft-threshold analytically solves:
        argmin_u  0.5 * ||u - g_temp||^2 + lam * lr * ||u||_1
    This closed-form solution CAN produce exact zeros.
    """
    g_temp = g - lr * task_grad(g)
    return np.sign(g_temp) * np.maximum(np.abs(g_temp) - lam * lr, 0)


def sgd_subgradient_update(g, lr, lam):
    """
    SGD update with l1 subgradient.

    Subgradient of ||g_i||_1:
        d|g_i|/dg_i = sign(g_i),  with sign(0) = 0 by convention.

    Full update rule:
        g_new = g - lr * (grad_task(g) + lam * sign(g))

    NOTE: Cannot produce exact zeros.
    When g_i is near zero, it oscillates across zero each step
    because sign flips direction every time the threshold is crossed.
    """
    subgrad_l1 = np.sign(g)  # sign(0) = 0
    return g - lr * (task_grad(g) + lam * subgrad_l1)


# ── Run Both Optimizers ────────────────────────────────────────────────────────
g_prox = np.ones(n)
g_sgd  = np.ones(n)

hist_prox, hist_sgd = [], []
spar_prox, spar_sgd = [], []

for step in range(n_steps):
    g_prox = proximal_update(g_prox, lr, lambda_reg)
    g_sgd  = sgd_subgradient_update(g_sgd, lr, lambda_reg)

    hist_prox.append(total_loss(g_prox))
    hist_sgd .append(total_loss(g_sgd))

    spar_prox.append(int(np.sum(np.abs(g_prox) < 1e-6)))
    spar_sgd .append(int(np.sum(np.abs(g_sgd)  < 1e-6)))

# ── Print Results ──────────────────────────────────────────────────────────────
print("=" * 55)
print("PROXIMAL GRADIENT (NumPy)")
print("=" * 55)
print(f"Final g:       {np.round(g_prox, 4)}")
print(f"Exact zeros:   {int(np.sum(g_prox == 0.0))}")
print(f"Near-zeros:    {int(np.sum(np.abs(g_prox) < 1e-6))}")
print(f"Final loss:    {hist_prox[-1]:.6f}")

print()
print("=" * 55)
print("SGD SUBGRADIENT (NumPy)")
print("=" * 55)
print(f"Final g:       {np.round(g_sgd, 4)}")
print(f"Exact zeros:   {int(np.sum(g_sgd == 0.0))}")
print(f"Near-zeros:    {int(np.sum(np.abs(g_sgd) < 1e-6))}")
print(f"Final loss:    {hist_sgd[-1]:.6f}")

print()
print("=" * 55)
print("GROUND TRUTH")
print("=" * 55)
print(f"True g:        {g_true}")
print(f"True zeros:    6  (indices 1, 2, 4, 5, 7, 8)")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(hist_prox, label='Proximal', color='steelblue', lw=2)
axes[0].plot(hist_sgd,  label='SGD Subgradient', color='tomato', lw=2, ls='--')
axes[0].set_title('Total Loss (task + λ‖g‖₁)')
axes[0].set_xlabel('Step')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(spar_prox, label='Proximal', color='steelblue', lw=2)
axes[1].plot(spar_sgd,  label='SGD Subgradient', color='tomato', lw=2, ls='--')
axes[1].axhline(6, color='gray', ls=':', label='True sparsity (6)')
axes[1].set_title('Near-Zero Components (|g| < 1e-6)')
axes[1].set_xlabel('Step')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].bar(np.arange(n) - 0.2, np.abs(g_prox), width=0.35, alpha=0.8,
            label='Proximal', color='steelblue')
axes[2].bar(np.arange(n) + 0.2, np.abs(g_sgd),  width=0.35, alpha=0.8,
            label='SGD', color='tomato')
axes[2].bar(np.arange(n), np.abs(g_true), width=0.05,
            color='black', label='True g', zorder=5)
axes[2].set_title('Final |g| per Component')
axes[2].set_xlabel('Component index')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.suptitle('NumPy: Proximal Gradient vs SGD Subgradient on ℓ₁-Regularized Problem',
             fontsize=12)
plt.tight_layout()
plt.savefig('numpy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: numpy_comparison.png")
