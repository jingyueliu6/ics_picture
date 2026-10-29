import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


# ============================================================
# 1. SETTINGS
# ============================================================

T = 20.0
dt = 0.02

t_eval = np.arange(0, T + dt, dt)

# Same initial condition as Slide 2
x0 = np.array([1.2, 0.0])

g = 9.81

COLOR_TRUE = "tab:blue"
COLOR_ONE = "tab:orange"
COLOR_ROLLOUT = "tab:green"

STYLE_TRUE = "-"
STYLE_ONE = "--"
STYLE_ROLLOUT = "-."
# ============================================================
# 2. TRUE DYNAMICS
#
# EXACTLY the same as Slide 2
# ============================================================

l_true = 1.0
c_true = 0.10


def true_dynamics(t, x):

    q, dq = x

    ddq = (
        -(g / l_true) * np.sin(q)
        - c_true * dq
    )

    return np.array([dq, ddq])


# ============================================================
# 3. TRUE TRAJECTORY
# ============================================================

true_sol = solve_ivp(
    true_dynamics,
    [0, T],
    x0,
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12
)

x_true = true_sol.y.T


# ============================================================
# 4. ONE-STEP MODEL
#
# IMPORTANT:
# These are EXACTLY the parameters used on Slide 2.
#
# We do NOT retrain this model here.
#
# Therefore the orange curve should be identical
# to the orange curve on Slide 2.
# ============================================================

l_one = 1.03
c_one = 0.09


# ============================================================
# 5. GENERAL LEARNED MODEL
#
# Both the one-step model and rollout model use
# exactly the same model structure.
#
# Only parameters l and c are different.
# ============================================================

def learned_dynamics(t, x, l, c):

    q, dq = x

    ddq = (
        -(g / l) * np.sin(q)
        - c * dq
    )

    return np.array([dq, ddq])


# ============================================================
# 6. RK4 STEP
#
# Used during rollout training.
# ============================================================

def rk4_step(x, l, c):

    def f(xx):
        return learned_dynamics(
            0.0,
            xx,
            l,
            c
        )

    k1 = f(x)

    k2 = f(
        x + 0.5 * dt * k1
    )

    k3 = f(
        x + 0.5 * dt * k2
    )

    k4 = f(
        x + dt * k3
    )

    return (
        x
        + dt / 6.0
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
    )


# ============================================================
# 7. ROLLOUT FUNCTION
# ============================================================

def rollout_model(
    x_initial,
    l,
    c,
    n_steps
):

    trajectory = np.zeros(
        (n_steps + 1, 2)
    )

    trajectory[0] = x_initial

    x = x_initial.copy()

    for k in range(n_steps):

        x = rk4_step(
            x,
            l,
            c
        )

        trajectory[k + 1] = x

    return trajectory


# ============================================================
# 8. ORIGINAL ONE-STEP MODEL ROLLOUT
#
# This reproduces Slide 2.
# ============================================================

x_one = rollout_model(
    x0,
    l_one,
    c_one,
    len(t_eval) - 1
)


# ============================================================
# 9. ROLLOUT TRAINING SETTINGS
#
# Train using multiple 4-second rollout windows.
#
# H = 200:
# 200 * 0.02 = 4 seconds
# ============================================================

H = 200

# Start a new training window every 1 second
window_stride = 50


# ============================================================
# 10. ROLLOUT LOSS
#
# Key difference:
#
# ONE-STEP:
#
#   true state -> model -> prediction
#   true state -> model -> prediction
#   true state -> model -> prediction
#
# ROLLOUT:
#
#   true state -> model -> prediction
#                         |
#                         v
#                       model
#                         |
#                         v
#                    prediction
#
# Predictions are recursively fed back into the model.
# ============================================================

def rollout_loss(theta):

    l, c = theta

    # Physical constraints
    if l <= 0.5 or l >= 1.5:
        return 1e10

    if c < 0.0 or c >= 0.5:
        return 1e10

    total_loss = 0.0
    total_weight = 0.0

    for start in range(
        0,
        len(x_true) - H - 1,
        window_stride
    ):

        # Only the INITIAL state is taken from data
        x_pred = x_true[start].copy()

        for k in range(1, H + 1):

            # Recursive model prediction
            x_pred = rk4_step(
                x_pred,
                l,
                c
            )

            # Ground-truth state
            target = x_true[
                start + k
            ]

            # ------------------------------------------------
            # Horizon weighting
            #
            # Later predictions receive slightly more weight.
            #
            # w = 0.5 at beginning
            # w = 1.0 at end
            # ------------------------------------------------

            weight = (
                0.5
                + 0.5 * k / H
            )

            error = (
                x_pred
                - target
            )

            total_loss += (
                weight
                * np.sum(error ** 2)
            )

            total_weight += weight

    return (
        total_loss
        / total_weight
    )


# ============================================================
# 11. TRAIN ROLLOUT MODEL
#
# Start from the SAME imperfect parameters shown
# on Slide 2.
# ============================================================

initial_guess = np.array([
    l_one,
    c_one
])


print(
    "\n===================================="
)

print(
    "ROLLOUT TRAINING"
)

print(
    "===================================="
)

print(
    f"Initial parameters:"
    f" l = {l_one:.4f},"
    f" c = {c_one:.4f}"
)


result_roll = minimize(
    rollout_loss,
    initial_guess,
    method="Nelder-Mead",
    options={
        "maxiter": 1500,
        "xatol": 1e-10,
        "fatol": 1e-12,
        "disp": True
    }
)


l_roll, c_roll = result_roll.x


print(
    "\nOptimized rollout parameters:"
)

print(
    f"l = {l_roll:.6f}"
)

print(
    f"c = {c_roll:.6f}"
)

print(
    f"Rollout loss = "
    f"{result_roll.fun:.8e}"
)


# ============================================================
# 12. FREE ROLLOUT OF ROLLOUT-TRAINED MODEL
# ============================================================

x_roll = rollout_model(
    x0,
    l_roll,
    c_roll,
    len(t_eval) - 1
)


# ============================================================
# 13. COMPUTE ERRORS
# ============================================================

error_one = np.linalg.norm(
    x_one - x_true,
    axis=1
)

error_roll = np.linalg.norm(
    x_roll - x_true,
    axis=1
)


# ============================================================
# 14. CUMULATIVE RMSE
# ============================================================

sq_error_one = np.sum(
    (x_one - x_true) ** 2,
    axis=1
)

sq_error_roll = np.sum(
    (x_roll - x_true) ** 2,
    axis=1
)


N = np.arange(
    1,
    len(t_eval) + 1
)


cum_rmse_one = np.sqrt(
    np.cumsum(
        sq_error_one
    ) / N
)


cum_rmse_roll = np.sqrt(
    np.cumsum(
        sq_error_roll
    ) / N
)


# ============================================================
# 15. PRINT PERFORMANCE
# ============================================================

print(
    "\n===================================="
)

print(
    "20 s FREE-ROLLOUT PERFORMANCE"
)

print(
    "===================================="
)

print(
    f"One-step model:"
    f" l={l_one:.4f},"
    f" c={c_one:.4f}"
)

print(
    f"Rollout model:"
    f"  l={l_roll:.4f},"
    f" c={c_roll:.4f}"
)

print()

print(
    f"Final state error"
    f" — One-step:"
    f" {error_one[-1]:.6f}"
)

print(
    f"Final state error"
    f" — Rollout:"
    f"  {error_roll[-1]:.6f}"
)

print()

print(
    f"20 s cumulative RMSE"
    f" — One-step:"
    f" {cum_rmse_one[-1]:.6f}"
)

print(
    f"20 s cumulative RMSE"
    f" — Rollout:"
    f"  {cum_rmse_roll[-1]:.6f}"
)


# ============================================================
# 16. FIGURE 1
#
# SHORT HORIZON + LONG ROLLOUT
#
# This should be the main figure for Slide 3.
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4.0),
    constrained_layout=True
)


# ------------------------------------------------------------
# Short horizon
# ------------------------------------------------------------

short_time = 2.0

mask = (
    t_eval <= short_time
)


axes[0].plot(
    t_eval[mask],
    x_true[mask, 0],
    color=COLOR_TRUE,
    linestyle=STYLE_TRUE,
    linewidth=2.7,
    label="True dynamics"
)

axes[0].plot(
    t_eval[mask],
    x_one[mask, 0],
    color=COLOR_ONE,
    linestyle=STYLE_ONE,
    linewidth=2.5,
    label="One-step model"
)

axes[0].plot(
    t_eval[mask],
    x_roll[mask, 0],
    color=COLOR_ROLLOUT,
    linestyle=STYLE_ROLLOUT,
    linewidth=2.5,
    label="Rollout-trained model"
)

axes[0].set_title(
    "Short Horizon",
    fontsize=16,
    fontweight="bold"
)


axes[0].set_xlabel(
    "Time [s]",
    fontsize=13
)


axes[0].set_ylabel(
    r"Angle $q$ [rad]",
    fontsize=13
)


axes[0].grid(
    alpha=0.2
)


# ------------------------------------------------------------
# Long rollout
# ------------------------------------------------------------

axes[1].plot(
    t_eval,
    x_true[:, 0],
    color=COLOR_TRUE,
    linestyle=STYLE_TRUE,
    linewidth=2.7,
    label="True dynamics"
)

axes[1].plot(
    t_eval,
    x_one[:, 0],
    color=COLOR_ONE,
    linestyle=STYLE_ONE,
    linewidth=2.5,
    label="One-step model"
)

axes[1].plot(
    t_eval,
    x_roll[:, 0],
    color=COLOR_ROLLOUT,
    linestyle=STYLE_ROLLOUT,
    linewidth=2.5,
    label="Rollout-trained model"
)


axes[1].set_title(
    "Long Rollout",
    fontsize=16,
    fontweight="bold"
)


axes[1].set_xlabel(
    "Time [s]",
    fontsize=13
)


axes[1].set_ylabel(
    r"Angle $q$ [rad]",
    fontsize=13
)


axes[1].grid(
    alpha=0.2
)


# ------------------------------------------------------------
# Shared legend
# ------------------------------------------------------------

handles, labels = (
    axes[0].get_legend_handles_labels()
)


fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    ncol=3,
    frameon=False,
    fontsize=11
)


plt.savefig(
    "pendulum_one_step_vs_rollout.png",
    dpi=300,
    transparent=True,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 17. FIGURE 2
#
# INSTANTANEOUS STATE ERROR
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 4.2)
)

ax.plot(
    t_eval,
    error_one,
    color=COLOR_ONE,
    linestyle=STYLE_ONE,
    linewidth=2.6,
    label="One-step model"
)

ax.plot(
    t_eval,
    error_roll,
    color=COLOR_ROLLOUT,
    linestyle=STYLE_ROLLOUT,
    linewidth=2.6,
    label="Rollout-trained model"
)

ax.set_title(
    "Prediction Error over Rollout Horizon",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "Rollout Time [s]",
    fontsize=13
)

ax.set_ylabel(
    r"State Error $\|\hat{s}_t-s_t\|_2$",
    fontsize=13
)

ax.grid(alpha=0.2)

ax.legend(
    frameon=False,
    fontsize=11
)

plt.tight_layout()

plt.savefig(
    "pendulum_rollout_error.png",
    dpi=300,
    transparent=True,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 18. FIGURE 3
#
# CUMULATIVE RMSE
#
# This is often easier to interpret than instantaneous error.
# ============================================================

ax.plot(
    t_eval,
    cum_rmse_one,
    color=COLOR_ONE,
    linestyle=STYLE_ONE,
    linewidth=2.6,
    label="One-step model"
)

ax.plot(
    t_eval,
    cum_rmse_roll,
    color=COLOR_ROLLOUT,
    linestyle=STYLE_ROLLOUT,
    linewidth=2.6,
    label="Rollout-trained model"
)