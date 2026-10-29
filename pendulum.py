import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================
# 1. Simulation
# ============================================================

T = 15.0
dt = 0.02

t_eval = np.arange(0, T + dt, dt)

x0 = np.array([1.2, 0.0])

g = 9.81


# ============================================================
# 2. True dynamics
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
# 3. Learned model
#
# Slightly wrong dynamics
# ============================================================

l_model = 1.06
c_model = 0.075


def learned_dynamics(t, x):

    q, dq = x

    ddq = (
        -(g / l_model) * np.sin(q)
        - c_model * dq
    )

    return [dq, ddq]


# ============================================================
# 4. Simulate
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
# 5. Endpoint positions
# ============================================================

q_true = x_true[:, 0]
q_model = x_model[:, 0]

true_px = l_true * np.sin(q_true)
true_py = -l_true * np.cos(q_true)

# Same physical geometry for visualization
model_px = l_true * np.sin(q_model)
model_py = -l_true * np.cos(q_model)


# ============================================================
# 6. Animation settings
# ============================================================

frame_skip = 3

frames = np.arange(
    0,
    len(t_eval),
    frame_skip
)

# Around 1.6 seconds of history
trail_length = 80


# ============================================================
# 7. Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.2, 5.2),
    facecolor="white"
)

ax.set_facecolor("white")


# ============================================================
# 8. Update
# ============================================================

def update(frame):

    # Completely redraw frame
    ax.clear()

    ax.set_facecolor("white")

    ax.set_aspect("equal")

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.20, 0.30)

    ax.axis("off")


    # ========================================================
    # Trail
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

        # Old -> new
        alphas = np.linspace(
            0.02,
            0.65,
            n
        )

        sizes = np.linspace(
            4,
            22,
            n
        )

        # -----------------------------------------------
        # True trajectory
        # -----------------------------------------------

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


        # -----------------------------------------------
        # Predicted trajectory
        # -----------------------------------------------

        model_colors = np.zeros(
            (n, 4)
        )

        model_colors[:, :] = [
            255/255,
            127/255,
            14/255,
            1
        ]

        model_colors[:, 3] = alphas

        ax.scatter(
            model_px[history],
            model_py[history],
            s=sizes,
            c=model_colors,
            marker="x",
            linewidths=1.4,
            zorder=3
        )


    # ========================================================
    # Pivot
    # ========================================================

    ax.plot(
        0,
        0,
        "o",
        color="black",
        markersize=8,
        zorder=20
    )


    # ========================================================
    # Current TRUE pendulum
    # ========================================================

    tx = true_px[frame]
    ty = true_py[frame]

    ax.plot(
        [0, tx],
        [0, ty],
        color="tab:blue",
        linewidth=4,
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
    # Current MODEL pendulum
    # ========================================================

    mx = model_px[frame]
    my = model_py[frame]

    ax.plot(
        [0, mx],
        [0, my],
        color="tab:orange",
        linewidth=3,
        linestyle="--",
        label="Model prediction",
        zorder=12
    )

    ax.plot(
        mx,
        my,
        "o",
        markerfacecolor="white",
        markeredgecolor="tab:orange",
        markeredgewidth=2.5,
        markersize=12,
        zorder=13
    )


    # ========================================================
    # Time
    # ========================================================

    ax.text(
        0.04,
        0.94,
        f"t = {t_eval[frame]:.1f} s",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top"
    )


    # ========================================================
    # Legend
    # ========================================================

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=11
    )


# ============================================================
# 9. Animation
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
# 10. Save
# ============================================================

animation.save(
    "pendulum_endpoint_trail.gif",
    writer=PillowWriter(
        fps=25
    ),
    dpi=120
)

plt.close(fig)

print("Saved: pendulum_endpoint_trail.gif")