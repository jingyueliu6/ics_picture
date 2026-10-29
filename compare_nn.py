import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ============================================================
# 0. Reproducibility
# ============================================================
np.random.seed(4)
torch.manual_seed(4)

# ============================================================
# 1. Target function and training data
# ============================================================
def f_true(x):
    """
    Nonlinear target function.
    Contains both global and local variations.
    """
    return (
        0.55 * np.sin(1.4 * x)
        + 0.30 * np.sin(3.8 * x + 0.4)
        + 0.12 * x
    )


# Dense grid for plotting
x_plot = np.linspace(-3.0, 3.0, 800)
y_true = f_true(x_plot)

# Sparse training samples
n_train = 45
x_train = np.linspace(-3.0, 3.0, n_train)

noise_std = 0.04
y_train = f_true(x_train) + noise_std * np.random.randn(n_train)


# ============================================================
# 2. Sum of fixed basis functions
#    y = pi^T phi(x)
# ============================================================
def rbf_features(x, centers, width):
    """
    Gaussian radial basis functions.
    """
    x = np.asarray(x)[:, None]
    centers = np.asarray(centers)[None, :]

    Phi = np.exp(
        -0.5 * ((x - centers) / width) ** 2
    )

    # Add constant basis function
    Phi = np.column_stack([
        np.ones(len(x)),
        Phi
    ])

    return Phi


# Number of fixed basis functions
n_basis = 6

centers = np.linspace(-3.0, 3.0, n_basis)
width = 0.75

Phi_train = rbf_features(x_train, centers, width)
Phi_plot = rbf_features(x_plot, centers, width)

# Closed-form least-squares solution
pi_hat, *_ = np.linalg.lstsq(
    Phi_train,
    y_train,
    rcond=None
)

y_basis = Phi_plot @ pi_hat


# ============================================================
# 3. Neural network
# ============================================================
class MLP(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()

        layers = []
        input_dim = 1

        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_nn(hidden_sizes, epochs=8000, lr=5e-3):
    """
    Train an MLP on the same training data.
    """

    # Reset seed so comparison is reproducible
    torch.manual_seed(10 + sum(hidden_sizes))

    model = MLP(hidden_sizes)

    x_t = torch.tensor(
        x_train[:, None],
        dtype=torch.float32
    )

    y_t = torch.tensor(
        y_train[:, None],
        dtype=torch.float32
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    loss_fn = nn.MSELoss()

    for epoch in range(epochs):

        optimizer.zero_grad()

        prediction = model(x_t)

        loss = loss_fn(
            prediction,
            y_t
        )

        loss.backward()
        optimizer.step()

    # Prediction on dense plotting grid
    x_plot_t = torch.tensor(
        x_plot[:, None],
        dtype=torch.float32
    )

    model.eval()

    with torch.no_grad():
        y_pred = (
            model(x_plot_t)
            .cpu()
            .numpy()
            .squeeze()
        )

    return model, y_pred


# ============================================================
# 4. Train networks with different capacities
# ============================================================
model_2, y_nn_2 = train_nn([2])

model_10, y_nn_10 = train_nn([10])

model_10_10, y_nn_10_10 = train_nn([10, 10])


# ============================================================
# 5. Plotting helper
# ============================================================
def style_axis(ax):

    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.15, 1.15)

    ax.set_xlabel(r"$x$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)

    ax.tick_params(
        labelsize=11
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================
# 6. Create comparison figure
# ============================================================
fig, axes = plt.subplots(
    1,
    4,
    figsize=(15.5, 3.6),
    sharex=True,
    sharey=True
)

# ------------------------------------------------------------
# Panel 1: Fixed basis functions
# ------------------------------------------------------------
ax = axes[0]

ax.plot(
    x_plot,
    y_true,
    "--",
    linewidth=2.5,
    label="True function"
)

ax.scatter(
    x_train,
    y_train,
    s=24,
    zorder=3,
    label="Training data"
)

ax.plot(
    x_plot,
    y_basis,
    linewidth=3,
    label="Approximation"
)

ax.set_title(
    "Fixed basis functions\n6 Gaussian bases",
    fontsize=14
)

style_axis(ax)


# ------------------------------------------------------------
# Panel 2: Small NN
# ------------------------------------------------------------
ax = axes[1]

ax.plot(
    x_plot,
    y_true,
    "--",
    linewidth=2.5
)

ax.scatter(
    x_train,
    y_train,
    s=24,
    zorder=3
)

ax.plot(
    x_plot,
    y_nn_2,
    linewidth=3
)

ax.set_title(
    "Neural network\n[2]",
    fontsize=14
)

style_axis(ax)


# ------------------------------------------------------------
# Panel 3: Wider NN
# ------------------------------------------------------------
ax = axes[2]

ax.plot(
    x_plot,
    y_true,
    "--",
    linewidth=2.5
)

ax.scatter(
    x_train,
    y_train,
    s=24,
    zorder=3
)

ax.plot(
    x_plot,
    y_nn_10,
    linewidth=3
)

ax.set_title(
    "Neural network\n[10]",
    fontsize=14
)

style_axis(ax)


# ------------------------------------------------------------
# Panel 4: Deeper NN
# ------------------------------------------------------------
ax = axes[3]

ax.plot(
    x_plot,
    y_true,
    "--",
    linewidth=2.5
)

ax.scatter(
    x_train,
    y_train,
    s=24,
    zorder=3
)

ax.plot(
    x_plot,
    y_nn_10_10,
    linewidth=3
)

ax.set_title(
    "Neural network\n[10, 10]",
    fontsize=14
)

style_axis(ax)


# ============================================================
# 7. Shared legend
# ============================================================
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    frameon=False,
    fontsize=12
)

plt.tight_layout(
    rect=[0, 0.08, 1, 1]
)

plt.savefig(
    "function_approximation_comparison.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "function_approximation_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()