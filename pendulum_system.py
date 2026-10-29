import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================
# 1. Simulation settings
# ============================================================

T = 12.0
dt = 0.02

t_eval = np.arange(0, T + dt, dt)

# Initial condition
q0 = 1.15          # rad
dq0 = 0.0          # rad/s
x0 = np.array([q0, dq0])


# ============================================================
# 2. Pendulum parameters
# ============================================================

g = 9.81           # m/s^2
l = 1.0            # m
c = 0.10           # damping coefficient


# ============================================================
# 3. Equation of motion
# ============================================================

def pendulum_dynamics(t, x):

    q, dq = x

    ddq = -(g / l) * np.sin(q) - c * dq

    return [dq, ddq]


# ============================================================
# 4. Simulate
# ============================================================

sol = solve_ivp(
    pendulum_dynamics,
    [0, T],
    x0,
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12
)

q = sol.y[0]
dq = sol.y[1]


# ============================================================
# 5. Cartesian position of pendulum bob
# ============================================================

px = l * np.sin(q)
py = -l * np.cos(q)


# ============================================================
# 6. Animation settings
# ============================================================

# Skip simulation points to reduce GIF size
frame_skip = 3

frames = np.arange(
    0,
    len(t_eval),
    frame_skip
)

# Number of simulation samples shown in fading trail
trail_length = 100


# ============================================================
# 7. Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(5.6, 5.0),
    facecolor="white"
)

fig.subplots_adjust(
    left=0,
    right=1,
    bottom=0,
    top=1
)


# ============================================================
# 8. Colors
# ============================================================

BLUE = np.array([
    31 / 255,
    119 / 255,
    180 / 255
])

DARK = np.array([
    25 / 255,
    35 / 255,
    45 / 255
])


# ============================================================
# 9. Animation update
# ============================================================

def update(frame):

    # --------------------------------------------------------
    # Clear frame
    # --------------------------------------------------------

    ax.clear()

    ax.set_facecolor("white")
    ax.set_aspect("equal")

    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-1.25, 0.32)

    ax.axis("off")


    # ========================================================
    # Support
    # ========================================================

    # Horizontal ceiling/support
    ax.plot(
        [-0.22, 0.22],
        [0.16, 0.16],
        color=DARK,
        linewidth=5,
        solid_capstyle="round",
        zorder=20
    )

    # Small vertical connector
    ax.plot(
        [0, 0],
        [0.16, 0.04],
        color=DARK,
        linewidth=3,
        zorder=19
    )


    # ========================================================
    # Fading endpoint trajectory
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

        # Old points almost disappear,
        # recent points become stronger
        alphas = np.linspace(
            0.015,
            0.50,
            n
        )

        sizes = np.linspace(
            4,
            18,
            n
        )

        colors = np.zeros((n, 4))

        colors[:, 0] = BLUE[0]
        colors[:, 1] = BLUE[1]
        colors[:, 2] = BLUE[2]
        colors[:, 3] = alphas

        ax.scatter(
            px[history],
            py[history],
            s=sizes,
            c=colors,
            edgecolors="none",
            zorder=2
        )


    # ========================================================
    # Current position
    # ========================================================

    x = px[frame]
    y = py[frame]


    # ========================================================
    # Pendulum rod
    # ========================================================

    # Subtle shadow behind rod
    ax.plot(
        [0, x],
        [0.04, y],
        color=(0.75, 0.78, 0.82),
        linewidth=6,
        alpha=0.28,
        solid_capstyle="round",
        zorder=7
    )

    # Main rod
    ax.plot(
        [0, x],
        [0.04, y],
        color=DARK,
        linewidth=3.2,
        solid_capstyle="round",
        zorder=10
    )


    # ========================================================
    # Pivot
    # ========================================================

    # Outer ring
    ax.scatter(
        [0],
        [0.04],
        s=135,
        facecolor="white",
        edgecolor=DARK,
        linewidth=2.5,
        zorder=21
    )

    # Inner pivot
    ax.scatter(
        [0],
        [0.04],
        s=35,
        color=BLUE,
        zorder=22
    )


    # ========================================================
    # Pendulum bob
    # ========================================================

    # Soft shadow
    ax.scatter(
        [x + 0.018],
        [y - 0.018],
        s=610,
        color="black",
        alpha=0.10,
        edgecolors="none",
        zorder=11
    )

    # Main bob
    ax.scatter(
        [x],
        [y],
        s=560,
        color=BLUE,
        edgecolor=DARK,
        linewidth=2.0,
        zorder=12
    )

    # Highlight
    ax.scatter(
        [x - 0.025],
        [y + 0.028],
        s=70,
        color="white",
        alpha=0.55,
        edgecolors="none",
        zorder=13
    )


# ============================================================
# 10. Create animation
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
# 11. Save GIF
# ============================================================

animation.save(
    "pendulum_system.gif",
    writer=PillowWriter(
        fps=25
    ),
    dpi=130
)

plt.close(fig)

print("Saved: pendulum_system.gif")