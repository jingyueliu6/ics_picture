import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ============================================================
# 1. Settings
# ============================================================

T = 20.0
dt = 0.02

t_eval = np.arange(0, T + dt, dt)

# Initial state: [angle, angular velocity]
x0 = np.array([1.2, 0.0])


# ============================================================
# 2. True dynamics
# ============================================================

g = 9.81
l_true = 1.0
c_true = 0.10


def true_dynamics(t, x):

    q, dq = x

    ddq = (
        -(g / l_true) * np.sin(q)
        - c_true * dq
    )

    return [dq, ddq]


# ============================================================
# 3. Learned dynamics
#
# Imagine these parameters were identified from data.
# They are close to the true system, but not perfect.
# ============================================================

l_model = 1.03
c_model = 0.09


def learned_dynamics(t, x):

    q, dq = x

    ddq = (
        -(g / l_model) * np.sin(q)
        - c_model * dq
    )

    return [dq, ddq]


# ============================================================
# 4. Free rollout
# ============================================================

true_sol = solve_ivp(
    true_dynamics,
    [0, T],
    x0,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-11
)

model_sol = solve_ivp(
    learned_dynamics,
    [0, T],
    x0,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-11
)

x_true = true_sol.y.T
x_model = model_sol.y.T


# ============================================================
# 5. Compute errors
# ============================================================

q_error = np.abs(
    x_model[:, 0] - x_true[:, 0]
)

dq_error = np.abs(
    x_model[:, 1] - x_true[:, 1]
)

# State-space Euclidean error
state_error = np.linalg.norm(
    x_model - x_true,
    axis=1
)


# ============================================================
# FIGURE 1
# Short-horizon vs long-horizon prediction
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4.0),
    constrained_layout=True
)

# ------------------------------------------------------------
# Short horizon
# ------------------------------------------------------------

short_time = 2.0
mask = t_eval <= short_time

axes[0].plot(
    t_eval[mask],
    x_true[mask, 0],
    linewidth=2.5,
    label="True dynamics"
)

axes[0].plot(
    t_eval[mask],
    x_model[mask, 0],
    "--",
    linewidth=2.5,
    label="Model rollout"
)

axes[0].set_title(
    "Short Horizon",
    fontsize=16,
    fontweight="bold"
)

axes[0].set_xlabel("Time [s]", fontsize=13)
axes[0].set_ylabel(r"Angle $q$ [rad]", fontsize=13)

axes[0].grid(alpha=0.2)


# ------------------------------------------------------------
# Long rollout
# ------------------------------------------------------------

axes[1].plot(
    t_eval,
    x_true[:, 0],
    linewidth=2.5,
    label="True dynamics"
)

axes[1].plot(
    t_eval,
    x_model[:, 0],
    "--",
    linewidth=2.5,
    label="Model rollout"
)

axes[1].set_title(
    "Long Rollout",
    fontsize=16,
    fontweight="bold"
)

axes[1].set_xlabel("Time [s]", fontsize=13)
axes[1].set_ylabel(r"Angle $q$ [rad]", fontsize=13)

axes[1].grid(alpha=0.2)


# ------------------------------------------------------------
# ONE shared legend
# ------------------------------------------------------------

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False,
    fontsize=11
)

plt.savefig(
    "pendulum_short_vs_rollout.png",
    dpi=300,
    transparent=True,
    bbox_inches="tight"
)

plt.show()