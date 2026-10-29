import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================
# System parameters
# ============================================================
MU = 3.0

# Plot / simulation region
X1_LIM = (-3.0, 3.0)
X2_LIM = (-5.0, 5.0)

# Numerical simulation settings
T_FINAL = 30.0
CONVERGENCE_TOL = 0.10
ESCAPE_RADIUS = 20.0


# ============================================================
# Dynamics
#
#   x1_dot = x2
#   x2_dot = -x1 + mu * (x1^2 - 1) * x2
# ============================================================
def dynamics(t, x):
    x1, x2 = x

    dx1 = x2
    dx2 = -x1 + MU * (x1**2 - 1.0) * x2

    return [dx1, dx2]


# ============================================================
# Event: stop integration if trajectory escapes too far
# ============================================================
def escape_event(t, x):
    return ESCAPE_RADIUS - np.linalg.norm(x)


escape_event.terminal = True
escape_event.direction = -1


# ============================================================
# Simulate one initial condition
# ============================================================
def simulate(x0, t_final=T_FINAL):
    sol = solve_ivp(
        dynamics,
        [0.0, t_final],
        x0,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
        max_step=0.05,
        events=escape_event,
    )

    return sol


# ============================================================
# Decide whether an initial condition converges to the origin
# ============================================================
def converges_to_origin(x0):
    sol = simulate(x0)

    # If integration terminated because the trajectory escaped,
    # classify it as non-converging.
    if sol.status == 1:
        return False

    final_state = sol.y[:, -1]

    return np.linalg.norm(final_state) < CONVERGENCE_TOL


# ============================================================
# 1. Estimate the ROA numerically on a grid
# ============================================================
print("Estimating region of attraction...")

n_roa = 65

x1_roa = np.linspace(X1_LIM[0], X1_LIM[1], n_roa)
x2_roa = np.linspace(X2_LIM[0], X2_LIM[1], n_roa)

ROA = np.zeros((n_roa, n_roa))

for i, x2 in enumerate(x2_roa):
    for j, x1 in enumerate(x1_roa):

        x0 = np.array([x1, x2])

        if converges_to_origin(x0):
            ROA[i, j] = 1.0

    print(f"Grid row {i + 1:02d}/{n_roa} completed")


# ============================================================
# 2. Vector field
# ============================================================
n_field = 35

x1_field = np.linspace(X1_LIM[0], X1_LIM[1], n_field)
x2_field = np.linspace(X2_LIM[0], X2_LIM[1], n_field)

X1, X2 = np.meshgrid(x1_field, x2_field)

DX1 = X2
DX2 = -X1 + MU * (X1**2 - 1.0) * X2

speed = np.sqrt(DX1**2 + DX2**2)

# Avoid division by zero
speed[speed == 0] = 1.0

# Normalize vector field for cleaner streamlines
U = DX1 / speed
V = DX2 / speed


# ============================================================
# 3. Create figure
# ============================================================
fig, ax = plt.subplots(figsize=(9, 7))


# ------------------------------------------------------------
# Estimated ROA
# ------------------------------------------------------------
XX1, XX2 = np.meshgrid(x1_roa, x2_roa)

ax.contourf(
    XX1,
    XX2,
    ROA,
    levels=[0.5, 1.5],
    alpha=0.18,
)


# Approximate ROA boundary
ax.contour(
    XX1,
    XX2,
    ROA,
    levels=[0.5],
    linewidths=2.5,
)


# ------------------------------------------------------------
# Vector field
# ------------------------------------------------------------
ax.streamplot(
    X1,
    X2,
    U,
    V,
    density=1.1,
    linewidth=0.7,
    arrowsize=0.8,
)


# ============================================================
# 4. Example trajectories
# ============================================================

# Initial conditions chosen to show different behaviors.
initial_conditions = [
    [-0.4, 0.0],
    [0.5, 0.2],
    [-0.8, 0.6],
    [1.0, -0.5],
    [-1.2, -0.4],
    [1.4, 0.0],
    [1.8, 0.0],
    [-1.8, 0.0],
]


for x0 in initial_conditions:

    sol = simulate(np.array(x0))

    is_convergent = (
        sol.status != 1
        and np.linalg.norm(sol.y[:, -1]) < CONVERGENCE_TOL
    )

    if is_convergent:
        linestyle = "-"
        linewidth = 2.0
    else:
        linestyle = "--"
        linewidth = 1.8

    ax.plot(
        sol.y[0],
        sol.y[1],
        linestyle=linestyle,
        linewidth=linewidth,
    )

    # Mark initial condition
    ax.plot(
        x0[0],
        x0[1],
        "o",
        markersize=5,
    )


# ============================================================
# 5. Mark equilibrium
# ============================================================
ax.plot(
    0,
    0,
    "o",
    markersize=10,
    label="Stable equilibrium",
)


# ============================================================
# 6. Labels and formatting
# ============================================================
ax.set_xlim(X1_LIM)
ax.set_ylim(X2_LIM)

ax.set_xlabel(r"$x_1$", fontsize=16)
ax.set_ylabel(r"$x_2$", fontsize=16)

ax.set_title(
    rf"Phase Portrait and Numerical ROA Estimate ($\mu={MU}$)",
    fontsize=16,
)

ax.grid(alpha=0.2)

ax.text(
    0.03,
    0.96,
    "Shaded region: initial states converging to the origin",
    transform=ax.transAxes,
    va="top",
    fontsize=11,
)

ax.text(
    0.03,
    0.91,
    "Solid trajectories: converging",
    transform=ax.transAxes,
    va="top",
    fontsize=11,
)

ax.text(
    0.03,
    0.86,
    "Dashed trajectories: non-converging / escaping",
    transform=ax.transAxes,
    va="top",
    fontsize=11,
)

plt.tight_layout()


# ============================================================
# 7. Save high-resolution figures for PowerPoint
# ============================================================
plt.savefig(
    "roa_phase_portrait.png",
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    "roa_phase_portrait.pdf",
    bbox_inches="tight",
)

plt.show()

print("\nDone.")
print("Saved:")
print("  roa_phase_portrait.png")
print("  roa_phase_portrait.pdf")