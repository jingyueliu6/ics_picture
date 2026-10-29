import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Dataset
# ============================================================

np.random.seed(7)

N = 80

x = np.random.randn(N)
noise = 1.2 * np.random.randn(N)

y = 2.0 * x + 1.0 + noise


# ============================================================
# Model:
#
# y_hat = w*x + b
#
# loss_i = 1/2 (y_hat_i - y_i)^2
# ============================================================

def loss(w, b):
    error = w * x + b - y
    return 0.5 * np.mean(error**2)


def gradient(w, b, indices):

    xb = x[indices]
    yb = y[indices]

    error = w * xb + b - yb

    dw = np.mean(error * xb)
    db = np.mean(error)

    return np.array([dw, db])


# ============================================================
# General gradient descent
#
# batch_size = N  -> Batch GD
# batch_size = 8  -> Mini-Batch GD
# batch_size = 1  -> SGD
# ============================================================

def run_gd(batch_size, lr, steps, seed=1):

    rng = np.random.default_rng(seed)

    theta = np.array([-3.5, -2.5])

    path = [theta.copy()]

    for step in range(steps):

        if batch_size == N:

            # use the entire dataset
            indices = np.arange(N)

        else:

            # randomly sample a batch
            indices = rng.choice(
                N,
                size=batch_size,
                replace=False
            )

        grad = gradient(
            theta[0],
            theta[1],
            indices
        )

        theta = theta - lr * grad

        path.append(theta.copy())

    return np.array(path)


# ============================================================
# Run the three methods
# ============================================================

batch_path = run_gd(
    batch_size=N,
    lr=0.35,
    steps=25,
    seed=1
)

mini_path = run_gd(
    batch_size=8,
    lr=0.22,
    steps=50,
    seed=2
)

sgd_path = run_gd(
    batch_size=1,
    lr=0.10,
    steps=100,
    seed=3
)


# ============================================================
# Compute loss landscape
# ============================================================

w_values = np.linspace(-4.5, 4.5, 250)
b_values = np.linspace(-3.5, 4.5, 250)

W, B = np.meshgrid(w_values, b_values)

Z = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):

        Z[i, j] = loss(
            W[i, j],
            B[i, j]
        )


# ============================================================
# Exact least-squares optimum
# ============================================================

X = np.column_stack([
    x,
    np.ones(N)
])

theta_star = np.linalg.lstsq(
    X,
    y,
    rcond=None
)[0]

w_star, b_star = theta_star


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 4.5),
    constrained_layout=True
)

levels = np.geomspace(
    Z.min() + 0.02,
    Z.max(),
    12
)


methods = [
    (
        "Batch GD",
        batch_path,
        r"$B=N$",
        "Stable gradient\nExpensive update"
    ),

    (
        "Mini-Batch GD",
        mini_path,
        r"$B=8$",
        "Balanced variance\nBalanced cost"
    ),

    (
        "Stochastic GD",
        sgd_path,
        r"$B=1$",
        "Noisy gradient\nCheap update"
    )
]


for ax, (title, path, batch_text, description) in zip(
    axes,
    methods
):

    # Loss contours
    ax.contour(
        W,
        B,
        Z,
        levels=levels,
        linewidths=1.0
    )

    # Optimization path
    ax.plot(
        path[:, 0],
        path[:, 1],
        "-o",
        linewidth=1.8,
        markersize=3
    )

    # Initial point
    ax.plot(
        path[0, 0],
        path[0, 1],
        marker="o",
        markersize=8
    )

    # Optimum
    ax.plot(
        w_star,
        b_star,
        marker="*",
        markersize=15
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        r"$w$",
        fontsize=14
    )

    ax.set_ylabel(
        r"$b$",
        fontsize=14
    )

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-3.5, 4.5)

    ax.grid(alpha=0.15)

    # Batch-size label
    ax.text(
        0.05,
        0.94,
        batch_text,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top"
    )

    # Interpretation
    ax.text(
        0.05,
        0.05,
        description,
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.9
        )
    )


# ============================================================
# Save transparent image for PowerPoint
# ============================================================

plt.savefig(
    "batch_minibatch_sgd_comparison.png",
    dpi=300,
    transparent=True,
    bbox_inches="tight"
)

plt.show()