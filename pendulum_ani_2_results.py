import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================
# 1. Simulation settings
# ============================================================

T = 15.0
dt = 0.02

t_eval = np.arange(0, T + dt, dt)

x0 = np.array([1.2, 0.0])

g = 9.81


# ============================================================
# 2. TRUE dynamics
# ============================================================

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
# 3. ONE-STEP / ORIGINAL model
#
# EXACTLY the model from the previous animation.
# ============================================================

l_one = 1.06
c_one = 0.075


def model_dynamics(t, x, l, c):

    q, dq = x

    ddq = (
        -(g / l) * np.sin(q)
        - c * dq
    )

    return np.array([dq, ddq])


# ============================================================
# 4. Generate TRUE trajectory
# ============================================================

true_sol = solve_ivp(
    true_dynamics,
    [0, T],
    x0,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-11
)

x_true = true_sol.y.T


# ============================================================
# 5. RK4 model step
# ============================================================

def rk4_step(x, l, c):

    def f(xx):

        return model_dynamics(
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
# 6. Model rollout
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
# 7. ORIGINAL one-step model trajectory
#
# This must remain identical to your previous animation.
# ============================================================

x_one = rollout_model(
    x0,
    l_one,
    c_one,
    len(t_eval) - 1
)


# ============================================================
# 8. ROLLOUT TRAINING
# ============================================================

# 4-second rollout horizon
H = 200

# New training window every 1 second
window_stride = 50


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

        # Only use the true state at the START
        x_pred = x_true[start].copy()

        for k in range(1, H + 1):

            # Recursive prediction
            x_pred = rk4_step(
                x_pred,
                l,
                c
            )

            target = x_true[
                start + k
            ]

            # Slightly emphasize later prediction
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
                * np.sum(
                    error**2
                )
            )

            total_weight += weight

    return (
        total_loss
        / total_weight
    )


# ============================================================
# 9. Optimize rollout model
#
# Start from EXACTLY the original model.
# ============================================================

initial_guess = np.array([
    l_one,
    c_one
])


print(
    "\nTraining rollout model..."
)


result_roll = minimize(
    rollout_loss,
    initial_guess,
    method="Nelder-Mead",
    options={
        "maxiter": 1500,
        "xatol": 1e-10,
        "fatol": 1e-12
    }
)


l_roll, c_roll = result_roll.x


print("\n==============================")
print("MODEL PARAMETERS")
print("==============================")

print(
    f"True:"
    f"       l={l_true:.4f},"
    f" c={c_true:.4f}"
)

print(
    f"One-step:"
    f"   l={l_one:.4f},"
    f" c={c_one:.4f}"
)

print(
    f"Rollout:"
    f"    l={l_roll:.4f},"
    f" c={c_roll:.4f}"
)


# ============================================================
# 10. ROLLOUT-trained trajectory
# ============================================================

x_roll = rollout_model(
    x0,
    l_roll,
    c_roll,
    len(t_eval) - 1
)


# ============================================================
# 11. Endpoint positions
# ============================================================

q_true = x_true[:, 0]
q_one = x_one[:, 0]
q_roll = x_roll[:, 0]


# IMPORTANT:
# same physical geometry for ALL models
physical_length = l_true


true_px = (
    physical_length
    * np.sin(q_true)
)

true_py = (
    -physical_length
    * np.cos(q_true)
)


one_px = (
    physical_length
    * np.sin(q_one)
)

one_py = (
    -physical_length
    * np.cos(q_one)
)


roll_px = (
    physical_length
    * np.sin(q_roll)
)

roll_py = (
    -physical_length
    * np.cos(q_roll)
)


# ============================================================
# 12. State errors
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
# 13. Animation settings
# ============================================================

frame_skip = 3

frames = np.arange(
    0,
    len(t_eval),
    frame_skip
)


# About 1.6 seconds of visible history
trail_length = 80


# ============================================================
# 14. Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.0, 5.5),
    facecolor="white"
)

ax.set_facecolor("white")


# ============================================================
# 15. Animation update
# ============================================================

def update(frame):

    # --------------------------------------------------------
    # Clear EVERYTHING from previous frame
    # --------------------------------------------------------

    ax.clear()

    ax.set_facecolor("white")

    ax.set_aspect("equal")

    ax.set_xlim(
        -1.30,
        1.30
    )

    ax.set_ylim(
        -1.25,
        0.35
    )

    ax.axis("off")


    # ========================================================
    # TRAIL
    # ========================================================

    start = max(
        0,
        frame - trail_length
    )

    history = np.arange(
        start,
        frame
    )


    if len(history) > 0:

        n = len(history)

        # Old points are nearly invisible;
        # recent points are stronger.
        alphas = np.linspace(
            0.015,
            0.55,
            n
        )

        sizes = np.linspace(
            3,
            18,
            n
        )


        # ====================================================
        # TRUE trail
        # ====================================================

        true_colors = np.zeros(
            (n, 4)
        )

        true_colors[:, :] = [
            31/255,
            119/255,
            180/255,
            1
        ]

        true_colors[:, 3] = alphas


        ax.scatter(
            true_px[history],
            true_py[history],
            s=sizes,
            c=true_colors,
            edgecolors="none",
            zorder=2
        )


        # ====================================================
        # ONE-STEP trail
        # ====================================================

        one_colors = np.zeros(
            (n, 4)
        )

        one_colors[:, :] = [
            255/255,
            127/255,
            14/255,
            1
        ]

        one_colors[:, 3] = alphas


        ax.scatter(
            one_px[history],
            one_py[history],
            s=sizes,
            c=one_colors,
            marker="x",
            linewidths=1.3,
            zorder=3
        )


        # ====================================================
        # ROLLOUT trail
        # ====================================================

        roll_colors = np.zeros(
            (n, 4)
        )

        roll_colors[:, :] = [
            44/255,
            160/255,
            44/255,
            1
        ]

        roll_colors[:, 3] = alphas


        ax.scatter(
            roll_px[history],
            roll_py[history],
            s=sizes,
            c=roll_colors,
            marker=".",
            edgecolors="none",
            zorder=4
        )


    # ========================================================
    # PIVOT
    # ========================================================

    ax.plot(
        0,
        0,
        "o",
        color="black",
        markersize=8,
        zorder=30
    )


    # ========================================================
    # CURRENT TRUE PENDULUM
    # ========================================================

    tx = true_px[frame]
    ty = true_py[frame]


    ax.plot(
        [0, tx],
        [0, ty],
        color="tab:blue",
        linewidth=4.2,
        solid_capstyle="round",
        label="True dynamics",
        zorder=10
    )


    ax.plot(
        tx,
        ty,
        "o",
        color="tab:blue",
        markersize=15,
        zorder=11
    )


    # ========================================================
    # CURRENT ONE-STEP PENDULUM
    # ========================================================

    ox = one_px[frame]
    oy = one_py[frame]


    ax.plot(
        [0, ox],
        [0, oy],
        color="tab:orange",
        linewidth=3.0,
        linestyle="--",
        label="One-step model",
        zorder=12
    )


    ax.plot(
        ox,
        oy,
        "o",
        markerfacecolor="white",
        markeredgecolor="tab:orange",
        markeredgewidth=2.5,
        markersize=12,
        zorder=13
    )


    # ========================================================
    # CURRENT ROLLOUT PENDULUM
    # ========================================================

    rx = roll_px[frame]
    ry = roll_py[frame]


    ax.plot(
        [0, rx],
        [0, ry],
        color="tab:green",
        linewidth=3.0,
        linestyle="-.",
        label="Rollout-trained model",
        zorder=14
    )


    ax.plot(
        rx,
        ry,
        "o",
        markerfacecolor="white",
        markeredgecolor="tab:green",
        markeredgewidth=2.5,
        markersize=10,
        zorder=15
    )


    # ========================================================
    # TIME
    # ========================================================

    ax.text(
        0.035,
        0.95,
        f"t = {t_eval[frame]:.1f} s",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top"
    )


    # ========================================================
    # ERROR VALUES
    # ========================================================

    ax.text(
        0.035,
        0.885,
        (
            r"$e_{\mathrm{one}}$"
            f" = {error_one[frame]:.3f}"
        ),
        transform=ax.transAxes,
        fontsize=11,
        color="tab:orange",
        va="top"
    )


    ax.text(
        0.035,
        0.83,
        (
            r"$e_{\mathrm{roll}}$"
            f" = {error_roll[frame]:.3f}"
        ),
        transform=ax.transAxes,
        fontsize=11,
        color="tab:green",
        va="top"
    )


    # ========================================================
    # LEGEND
    # ========================================================

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=10
    )


# ============================================================
# 16. Create animation
# ============================================================

animation = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=40,
    blit=False,
    repeat=True
)


# ============================================================
# 17. Save GIF
# ============================================================

animation.save(
    "pendulum_one_step_vs_rollout.gif",
    writer=PillowWriter(
        fps=25
    ),
    dpi=120
)


plt.close(fig)


print(
    "\nSaved:"
    " pendulum_one_step_vs_rollout.gif"
)