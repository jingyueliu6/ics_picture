import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Generate a simple regression dataset
# ============================================================

np.random.seed(7)

N = 80

x = np.random.randn(N)
noise = 1.2 * np.random.randn(N)

# True relation:
# y = 2*x + 1 + noise
y = 2.0 * x + 1.0 + noise


# ============================================================
# 2. Loss and gradients
#
# Model:
#       y_hat = w*x + b
#
# Per-sample loss:
#       l_i = 1/2 (w*x_i + b - y_i)^2
# ============================================================

def full_loss(w, b):
    error = w * x + b - y
    return 0.5 * np.mean(error**2)


def full_gradient(w, b):
    error = w * x + b - y

    dw = np.mean(error * x)
    db = np.mean(error)

    return np.array([dw, db])


def sample_gradient(w, b, i):
    error = w * x[i] + b - y[i]

    dw = error * x[i]
    db = error

    return np.array([dw, db])


# ============================================================
# 3. Optimization settings
# ============================================================

theta0 = np.array([-3.5, -2.5])

lr_batch = 0.35
lr_sgd = 0.12

batch_steps = 16
sgd_steps = 100


# ============================================================
# 4. Batch Gradient Descent
# ============================================================

theta = theta0.copy()

batch_path = [theta.copy()]

for k in range(batch_steps):

    grad = full_gradient(theta[0], theta[1])

    theta = theta - lr_batch * grad

    batch_path.append(theta.copy())

batch_path = np.array(batch_path)


# ============================================================
# 5. Stochastic Gradient Descent
# ============================================================

theta = theta0.copy()

sgd_path = [theta.copy()]

rng = np.random.default_rng(10)

for k in range(sgd_steps):

    # randomly choose ONE sample
    i = rng.integers(N)

    grad = sample_gradient(theta[0], theta[1], i)

    theta = theta - lr_sgd * grad

    sgd_path.append(theta.copy())

sgd_path = np.array(sgd_path)


# ============================================================
# 6. Compute loss landscape
# ============================================================

w_values = np.linspace(-4.5, 4.5, 250)
b_values = np.linspace(-3.5, 4.5, 250)

W, B = np.meshgrid(w_values, b_values)

Z = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = full_loss(W[i, j], B[i, j])


# ============================================================
# 7. Find optimum
# ============================================================

X = np.column_stack([x, np.ones(N)])

theta_star = np.linalg.lstsq(X, y, rcond=None)[0]

w_star, b_star = theta_star


# ============================================================
# 8. Plot
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4.8),
    constrained_layout=True
)

levels = np.geomspace(
    Z.min() + 0.02,
    Z.max(),
    12
)


# ------------------------------------------------------------
# Batch GD
# ------------------------------------------------------------

ax = axes[0]

ax.contour(
    W, B, Z,
    levels=levels,
    linewidths=1.0
)

ax.plot(
    batch_path[:, 0],
    batch_path[:, 1],
    "-o",
    linewidth=2.2,
    markersize=4
)

ax.plot(
    w_star,
    b_star,
    marker="*",
    markersize=15
)

ax.plot(
    theta0[0],
    theta0[1],
    marker="o",
    markersize=8
)

ax.set_title(
    "Batch Gradient Descent",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(r"$w$", fontsize=14)
ax.set_ylabel(r"$b$", fontsize=14)

ax.text(
    0.03,
    0.04,
    "Uses all N samples\nfor every update",
    transform=ax.transAxes,
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        alpha=0.9
    )
)


# ------------------------------------------------------------
# SGD
# ------------------------------------------------------------

ax = axes[1]

ax.contour(
    W, B, Z,
    levels=levels,
    linewidths=1.0
)

ax.plot(
    sgd_path[:, 0],
    sgd_path[:, 1],
    "-o",
    linewidth=1.5,
    markersize=2.5
)

ax.plot(
    w_star,
    b_star,
    marker="*",
    markersize=15
)

ax.plot(
    theta0[0],
    theta0[1],
    marker="o",
    markersize=8
)

ax.set_title(
    "Stochastic Gradient Descent",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(r"$w$", fontsize=14)
ax.set_ylabel(r"$b$", fontsize=14)

ax.text(
    0.03,
    0.04,
    "Uses one sample\nfor every update",
    transform=ax.transAxes,
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        alpha=0.9
    )
)


# ============================================================
# 9. Same axes for fair comparison
# ============================================================

for ax in axes:

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-3.5, 4.5)

    ax.grid(alpha=0.15)

    ax.tick_params(labelsize=11)


# ============================================================
# 10. Save transparent PNG
# ============================================================

plt.savefig(
    "batch_vs_sgd.png",
    dpi=300,
    transparent=True,
    bbox_inches="tight"
)

plt.show()