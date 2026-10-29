import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. Objective function
# ============================================================
def J(theta):
    return (theta - 2.0)**2 + 0.5


def grad_J(theta):
    return 2.0 * (theta - 2.0)


# ============================================================
# 2. Gradient descent
# ============================================================
theta0 = -2.0
eta = 0.20
n_steps = 7

thetas = [theta0]
losses = [J(theta0)]
gradients = [grad_J(theta0)]

theta = theta0

for k in range(n_steps):
    gradient = grad_J(theta)
    theta = theta - eta * gradient

    thetas.append(theta)
    losses.append(J(theta))
    gradients.append(grad_J(theta))

thetas = np.array(thetas)
losses = np.array(losses)
gradients = np.array(gradients)


# ============================================================
# 3. Create table
# ============================================================
table_data = []

for k in range(n_steps):
    theta_next = thetas[k] - eta * gradients[k]

    table_data.append({
        "Step k": k,
        "θ_k": thetas[k],
        "J(θ_k)": losses[k],
        "∇J(θ_k)": gradients[k],
        "θ_(k+1)": theta_next
    })

df = pd.DataFrame(table_data)

# Round for PPT
df_display = df.copy()

for col in ["θ_k", "J(θ_k)", "∇J(θ_k)", "θ_(k+1)"]:
    df_display[col] = df_display[col].map(
        lambda x: f"{x:.3f}"
    )

print("\nGradient descent iterations:\n")
print(df_display.to_string(index=False))

# ------------------------------------------------------------
# Copyable TSV
# ------------------------------------------------------------
print("\n\nCOPY THE FOLLOWING INTO POWERPOINT / EXCEL:\n")

print(
    df_display.to_csv(
        sep="\t",
        index=False
    )
)


# ============================================================
# 4. Loss landscape
# ============================================================
theta_grid = np.linspace(-2.5, 4.5, 600)
loss_grid = J(theta_grid)

fig, ax = plt.subplots(
    figsize=(9.5, 5.5)
)

# Loss landscape
ax.plot(
    theta_grid,
    loss_grid,
    linewidth=3,
    label=r"$J(\theta)$"
)

# Gradient descent trajectory
ax.plot(
    thetas,
    losses,
    "o--",
    linewidth=2.2,
    markersize=8,
    label="Gradient descent",
    zorder=4
)


# ============================================================
# 5. Labels
#    Only label early iterations individually.
#    Later points are grouped to avoid clutter.
# ============================================================

label_offsets = {
    0: (-5, 22),
    1: (-5, -35),
    2: (15, 20),
    3: (-20, -35),
}

for k in range(4):

    ax.annotate(
        rf"$\theta_{k}={thetas[k]:.2f}$",
        xy=(thetas[k], losses[k]),
        xytext=label_offsets[k],
        textcoords="offset points",
        ha="center",
        fontsize=11,
        arrowprops=dict(
            arrowstyle="-",
            linewidth=0.8
        )
    )


# ============================================================
# 6. Indicate later iterations without labeling every point
# ============================================================
ax.annotate(
    r"$\theta_4,\theta_5,\ldots \rightarrow \theta^\star$",
    xy=(thetas[5], losses[5]),
    xytext=(55, 45),
    textcoords="offset points",
    fontsize=11,
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.1
    )
)


# ============================================================
# 7. Optimum
# ============================================================
theta_star = 2.0
J_star = J(theta_star)

ax.scatter(
    theta_star,
    J_star,
    s=150,
    marker="*",
    zorder=6
)

ax.annotate(
    r"$\theta^\star=2$",
    xy=(theta_star, J_star),
    xytext=(55, 30),
    textcoords="offset points",
    fontsize=13,
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.2
    )
)


# ============================================================
# 8. Formatting
# ============================================================
ax.set_xlabel(
    r"Parameter $\theta$",
    fontsize=14
)

ax.set_ylabel(
    r"Loss $J(\theta)$",
    fontsize=14
)

# I would remove the title when inserting into PPT,
# because the slide already has a title.
# ax.set_title(
#     "Gradient Descent: Iterative Parameter Updates",
#     fontsize=16
# )

ax.tick_params(
    labelsize=11
)

# Grid
ax.grid(
    True,
    linestyle="--",
    linewidth=0.7,
    alpha=0.35
)

# Put grid behind curves
ax.set_axisbelow(True)

# Cleaner frame
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=False,
    fontsize=11,
    loc="upper right"
)

plt.tight_layout()

plt.savefig(
    "gradient_descent_steps.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "gradient_descent_steps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()