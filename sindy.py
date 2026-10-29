import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from sklearn.linear_model import Lasso


# ============================================================
# 0. Reproducibility + output
# ============================================================

np.random.seed(42)

OUTPUT_DIR = "sindy_pendulum_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Plot settings
# ============================================================

plt.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 19,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_TRUE = "black"
COLOR_DATA = "tab:blue"
COLOR_SMOOTH = "tab:orange"
COLOR_STLSQ = "tab:blue"
COLOR_LASSO = "tab:green"
COLOR_BAD = "tab:red"


# ============================================================
# 2. Ground-truth pendulum
# ============================================================

T = 12.0
dt = 0.02

t_eval = np.arange(0.0, T + dt, dt)

q0 = 1.15
dq0 = 0.0

x0 = np.array([q0, dq0])

g = 9.81
l = 1.0
c = 0.10


def true_dynamics(t, x):
    q, dq = x

    ddq = (
        -(g / l) * np.sin(q)
        - c * dq
    )

    return [dq, ddq]


sol = solve_ivp(
    true_dynamics,
    [0, T],
    x0,
    t_eval=t_eval,
    rtol=1e-10,
    atol=1e-12
)

q_true = sol.y[0]
dq_true = sol.y[1]

ddq_true = (
    -(g / l) * np.sin(q_true)
    - c * dq_true
)


# ============================================================
# 3. Add measurement noise
# ============================================================

# Adjust this if you want the noisy-data problem
# to be more or less visible.
noise_std_q = 0.012

q_noisy = q_true + noise_std_q * np.random.randn(len(q_true))


# ============================================================
# 4. Smooth data and estimate derivatives
# ============================================================

# Window length must be odd.
# Larger window = more smoothing.
window_length = 31
polyorder = 3

q_smooth = savgol_filter(
    q_noisy,
    window_length=window_length,
    polyorder=polyorder
)

dq_est = savgol_filter(
    q_noisy,
    window_length=window_length,
    polyorder=polyorder,
    deriv=1,
    delta=dt
)

ddq_est = savgol_filter(
    q_noisy,
    window_length=window_length,
    polyorder=polyorder,
    deriv=2,
    delta=dt
)


# ============================================================
# 5. Candidate library
# ============================================================

feature_names = [
    "1",
    "q",
    "dq",
    "q^2",
    "q*dq",
    "dq^2",
    "sin(q)",
    "cos(q)",
    "q^3",
    "dq^3",
]


def build_library(q, dq):
    """
    Candidate function library Theta(X).

    Columns:
        1
        q
        dq
        q^2
        q*dq
        dq^2
        sin(q)
        cos(q)
        q^3
        dq^3
    """

    return np.column_stack([
        np.ones_like(q),
        q,
        dq,
        q**2,
        q * dq,
        dq**2,
        np.sin(q),
        np.cos(q),
        q**3,
        dq**3,
    ])


Theta = build_library(
    q_smooth,
    dq_est
)

y = ddq_est


# ============================================================
# 6. STLSQ
# ============================================================

def stlsq(Theta, y, threshold, max_iter=20):
    """
    Sequential Thresholded Least Squares.

    1. Least-squares fit
    2. Remove coefficients smaller than threshold
    3. Refit using surviving terms
    4. Repeat
    """

    coef, *_ = np.linalg.lstsq(
        Theta,
        y,
        rcond=None
    )

    active_previous = None

    for _ in range(max_iter):

        active = np.abs(coef) >= threshold

        # Avoid an empty model
        if not np.any(active):
            break

        if (
            active_previous is not None
            and np.array_equal(active, active_previous)
        ):
            break

        active_previous = active.copy()

        coef_new = np.zeros_like(coef)

        coef_active, *_ = np.linalg.lstsq(
            Theta[:, active],
            y,
            rcond=None
        )

        coef_new[active] = coef_active
        coef = coef_new

    return coef


# ============================================================
# 7. LASSO with feature normalization
# ============================================================

def fit_lasso(Theta, y, alpha):
    """
    LASSO should be applied after feature scaling because
    different library columns have different magnitudes.

    We scale each column by its L2 norm, fit LASSO,
    then transform coefficients back to original units.
    """

    scales = np.linalg.norm(
        Theta,
        axis=0
    )

    scales[scales < 1e-12] = 1.0

    Theta_scaled = Theta / scales

    model = Lasso(
        alpha=alpha,
        fit_intercept=False,
        max_iter=200000,
        tol=1e-10
    )

    model.fit(
        Theta_scaled,
        y
    )

    coef_original = model.coef_ / scales

    return coef_original


# ============================================================
# 8. Equation formatting
# ============================================================

LATEX_NAMES = {
    "1": "",
    "q": r"q",
    "dq": r"\dot q",
    "q^2": r"q^2",
    "q*dq": r"q\dot q",
    "dq^2": r"\dot q^2",
    "sin(q)": r"\sin q",
    "cos(q)": r"\cos q",
    "q^3": r"q^3",
    "dq^3": r"\dot q^3",
}


def format_equation_text(
    coef,
    tol=1e-6,
    decimals=3
):
    """
    Plain-text equation for terminal.
    """

    terms = []

    for c_i, name in zip(coef, feature_names):

        if abs(c_i) < tol:
            continue

        magnitude = abs(c_i)

        if name == "1":
            body = f"{magnitude:.{decimals}f}"
        else:
            body = f"{magnitude:.{decimals}f} {name}"

        terms.append(
            (np.sign(c_i), body)
        )

    if len(terms) == 0:
        return "ddq = 0"

    equation = "ddq = "

    for i, (sign, body) in enumerate(terms):

        if i == 0:

            if sign < 0:
                equation += "-"

            equation += body

        else:

            if sign >= 0:
                equation += " + "
            else:
                equation += " - "

            equation += body

    return equation


def format_equation_latex(
    coef,
    tol=1e-6,
    decimals=2
):
    """
    LaTeX equation for plots.
    """

    terms = []

    for c_i, name in zip(coef, feature_names):

        if abs(c_i) < tol:
            continue

        magnitude = abs(c_i)

        latex_name = LATEX_NAMES[name]

        if name == "1":
            body = rf"{magnitude:.{decimals}f}"
        else:
            body = (
                rf"{magnitude:.{decimals}f}"
                rf"\,{latex_name}"
            )

        terms.append(
            (np.sign(c_i), body)
        )

    if len(terms) == 0:
        return r"$\ddot q = 0$"

    equation = r"$\ddot q = "

    for i, (sign, body) in enumerate(terms):

        if i == 0:

            if sign < 0:
                equation += "-"

            equation += body

        else:

            if sign >= 0:
                equation += " + "
            else:
                equation += " - "

            equation += body

    equation += "$"

    return equation


# ============================================================
# 9. Prediction / error utilities
# ============================================================

def predict_derivative(
    coef,
    q,
    dq
):
    Theta_local = build_library(q, dq)

    return Theta_local @ coef


def derivative_rmse(coef):
    pred = predict_derivative(
        coef,
        q_smooth,
        dq_est
    )

    return np.sqrt(
        np.mean(
            (pred - ddq_true)**2
        )
    )


def active_terms(coef, tol=1e-6):
    return int(
        np.sum(
            np.abs(coef) > tol
        )
    )


# ============================================================
# 10. Rollout discovered model
# ============================================================

def discovered_rhs(t, x, coef):

    q, dq = x

    Theta_local = build_library(
        np.array([q]),
        np.array([dq])
    )

    ddq = float(
        Theta_local @ coef
    )

    return [
        dq,
        ddq
    ]


def rollout_model(coef):

    try:

        sol_model = solve_ivp(
            lambda t, x:
                discovered_rhs(
                    t,
                    x,
                    coef
                ),
            [0, T],
            x0,
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10
        )

        # Solver failure
        if (
            not sol_model.success
            or sol_model.y.shape[1] != len(t_eval)
        ):
            return None

        q_pred = sol_model.y[0]

        # Catch numerical blow-up
        if (
            np.any(~np.isfinite(q_pred))
            or np.max(np.abs(q_pred)) > 20
        ):
            return None

        return q_pred

    except Exception:
        return None


def rollout_rmse(coef):

    q_pred = rollout_model(coef)

    if q_pred is None:
        return np.inf

    return np.sqrt(
        np.mean(
            (q_pred - q_true)**2
        )
    )


# ============================================================
# 11. Hyperparameter sweeps
# ============================================================

stlsq_thresholds = [
    0.01,
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
    2.00,
]

lasso_alphas = [
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
]


stlsq_results = []
lasso_results = []


# ============================================================
# 12. Run STLSQ
# ============================================================

for threshold in stlsq_thresholds:

    coef = stlsq(
        Theta,
        y,
        threshold=threshold
    )

    d_rmse = derivative_rmse(coef)
    r_rmse = rollout_rmse(coef)
    n_active = active_terms(coef)

    stlsq_results.append({
        "parameter": threshold,
        "coef": coef,
        "derivative_rmse": d_rmse,
        "rollout_rmse": r_rmse,
        "active": n_active
    })


# ============================================================
# 13. Run LASSO
# ============================================================

for alpha in lasso_alphas:

    coef = fit_lasso(
        Theta,
        y,
        alpha=alpha
    )

    # Tiny numerical coefficients are visually irrelevant.
    coef[np.abs(coef) < 1e-7] = 0.0

    d_rmse = derivative_rmse(coef)
    r_rmse = rollout_rmse(coef)
    n_active = active_terms(coef)

    lasso_results.append({
        "parameter": alpha,
        "coef": coef,
        "derivative_rmse": d_rmse,
        "rollout_rmse": r_rmse,
        "active": n_active
    })


# ============================================================
# 14. Print ALL recovered equations
# ============================================================

print("\n")
print("=" * 75)
print("GROUND TRUTH")
print("=" * 75)

print(
    "ddq = -0.100 dq - 9.810 sin(q)"
)


print("\n")
print("=" * 75)
print("STLSQ RESULTS")
print("=" * 75)

for result in stlsq_results:

    threshold = result["parameter"]

    print(
        f"\nthreshold = {threshold:g}"
    )

    print(
        format_equation_text(
            result["coef"]
        )
    )

    print(
        f"active terms    = {result['active']}"
    )

    print(
        f"derivative RMSE = "
        f"{result['derivative_rmse']:.5f}"
    )

    print(
        f"rollout RMSE    = "
        f"{result['rollout_rmse']:.5f}"
    )


print("\n")
print("=" * 75)
print("LASSO RESULTS")
print("=" * 75)

for result in lasso_results:

    alpha = result["parameter"]

    print(
        f"\nalpha = {alpha:.0e}"
    )

    print(
        format_equation_text(
            result["coef"]
        )
    )

    print(
        f"active terms    = {result['active']}"
    )

    print(
        f"derivative RMSE = "
        f"{result['derivative_rmse']:.5f}"
    )

    print(
        f"rollout RMSE    = "
        f"{result['rollout_rmse']:.5f}"
    )


# ============================================================
# 15. Select best models
#
# We want good rollout accuracy but also a reasonably
# sparse equation.
#
# Penalize unnecessary active terms slightly.
# ============================================================

def selection_score(result):

    if not np.isfinite(
        result["rollout_rmse"]
    ):
        return np.inf

    sparsity_penalty = 0.003 * result["active"]

    return (
        result["rollout_rmse"]
        + sparsity_penalty
    )


best_stlsq = min(
    stlsq_results,
    key=selection_score
)

best_lasso = min(
    lasso_results,
    key=selection_score
)


print("\n")
print("=" * 75)
print("SELECTED MODELS")
print("=" * 75)

print("\nBest STLSQ")
print(
    f"threshold = "
    f"{best_stlsq['parameter']:g}"
)
print(
    format_equation_text(
        best_stlsq["coef"]
    )
)

print("\nBest LASSO")
print(
    f"alpha = "
    f"{best_lasso['parameter']:.0e}"
)
print(
    format_equation_text(
        best_lasso["coef"]
    )
)


# ============================================================
# 16. Figure 1
#
# Noisy measurements + smoothing
# ============================================================

fig, ax = plt.subplots(
    figsize=(10.5, 4.8)
)

# True
ax.plot(
    t_eval,
    q_true,
    color=COLOR_TRUE,
    linewidth=3.2,
    alpha=0.95,
    label="True trajectory",
    zorder=3
)

# Noisy measurements
ax.scatter(
    t_eval[::4],
    q_noisy[::4],
    s=22,
    color=COLOR_DATA,
    alpha=0.28,
    edgecolors="none",
    label="Noisy measurements",
    zorder=1
)

# Smoothed
ax.plot(
    t_eval,
    q_smooth,
    color=COLOR_SMOOTH,
    linewidth=2.3,
    linestyle="--",
    alpha=0.9,
    label="Smoothed signal",
    zorder=4
)

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$q$ [rad]")

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper right"
)

# Explicitly NO grid
ax.grid(False)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_measurements.png"
    ),
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)


# ============================================================
# 17. Helper: find result closest to requested parameter
# ============================================================

def closest_result(
    results,
    target
):
    return min(
        results,
        key=lambda r:
            abs(
                np.log10(r["parameter"])
                - np.log10(target)
            )
    )


# ============================================================
# 18. Figure 2
#
# STLSQ recovered equations
# ============================================================

# Three representative sparsity levels
stlsq_show = [
    closest_result(
        stlsq_results,
        0.01
    ),
    closest_result(
        stlsq_results,
        0.50
    ),
    closest_result(
        stlsq_results,
        2.00
    ),
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(14.0, 4.1)
)

for i, (ax, result) in enumerate(
    zip(
        axes,
        stlsq_show
    )
):

    threshold = result["parameter"]

    # Clean card
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header
    ax.text(
        0.5,
        0.87,
        rf"$\lambda_{{\rm th}}={threshold:g}$",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=COLOR_STLSQ
    )

    # Equation
    equation = format_equation_latex(
        result["coef"],
        decimals=2
    )

    ax.text(
        0.5,
        0.56,
        equation,
        ha="center",
        va="center",
        fontsize=18,
        wrap=True
    )

    # Metrics
    metric_text = (
        f"{result['active']} active terms\n"
        f"Rollout RMSE = "
        f"{result['rollout_rmse']:.3f}"
    )

    ax.text(
        0.5,
        0.23,
        metric_text,
        ha="center",
        va="center",
        fontsize=14,
        color="0.30",
        linespacing=1.5
    )

    # Border
    rect = plt.Rectangle(
        (0.03, 0.06),
        0.94,
        0.88,
        fill=False,
        linewidth=1.6,
        edgecolor=COLOR_STLSQ,
        alpha=0.55
    )

    ax.add_patch(rect)


fig.suptitle(
    "STLSQ: Thresholding Changes the Discovered Equation",
    fontsize=21,
    y=1.01
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_stlsq_equations.png"
    ),
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)


# ============================================================
# 19. Figure 3
#
# LASSO recovered equations
# ============================================================

lasso_show = [
    closest_result(
        lasso_results,
        1e-6
    ),
    closest_result(
        lasso_results,
        1e-4
    ),
    closest_result(
        lasso_results,
        1e-2
    ),
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(14.0, 4.1)
)

for ax, result in zip(
    axes,
    lasso_show
):

    alpha = result["parameter"]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header
    ax.text(
        0.5,
        0.87,
        rf"$\alpha={alpha:.0e}$",
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        color=COLOR_LASSO
    )

    # Equation
    equation = format_equation_latex(
        result["coef"],
        decimals=2
    )

    ax.text(
        0.5,
        0.56,
        equation,
        ha="center",
        va="center",
        fontsize=18,
        wrap=True
    )

    # Metrics
    metric_text = (
        f"{result['active']} active terms\n"
        f"Rollout RMSE = "
        f"{result['rollout_rmse']:.3f}"
    )

    ax.text(
        0.5,
        0.23,
        metric_text,
        ha="center",
        va="center",
        fontsize=14,
        color="0.30",
        linespacing=1.5
    )

    rect = plt.Rectangle(
        (0.03, 0.06),
        0.94,
        0.88,
        fill=False,
        linewidth=1.6,
        edgecolor=COLOR_LASSO,
        alpha=0.55
    )

    ax.add_patch(rect)


fig.suptitle(
    r"LASSO: $\alpha$ Controls Coefficient Shrinkage",
    fontsize=21,
    y=1.01
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_lasso_equations.png"
    ),
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)


# ============================================================
# 20. Find intentionally over-sparse model
#
# Used only as a visual teaching comparison.
# ============================================================

valid_stlsq = [
    r
    for r in stlsq_results
    if np.isfinite(
        r["rollout_rmse"]
    )
]

over_sparse = min(
    valid_stlsq,
    key=lambda r:
        (
            r["active"],
            -r["rollout_rmse"]
        )
)


# ============================================================
# 21. Rollouts
# ============================================================

q_stlsq = rollout_model(
    best_stlsq["coef"]
)

q_lasso = rollout_model(
    best_lasso["coef"]
)

q_sparse = rollout_model(
    over_sparse["coef"]
)


# ============================================================
# 22. Figure 4
#
# Clean rollout comparison
# ============================================================

fig, ax = plt.subplots(
    figsize=(10.8, 5.0)
)

# ------------------------------------------------------------
# True dynamics
# ------------------------------------------------------------

ax.plot(
    t_eval,
    q_true,
    color=COLOR_TRUE,
    linewidth=4.0,
    linestyle="-",
    alpha=0.95,
    label="True dynamics",
    zorder=10
)


# ------------------------------------------------------------
# Best STLSQ
# ------------------------------------------------------------

if q_stlsq is not None:

    ax.plot(
        t_eval,
        q_stlsq,
        color=COLOR_STLSQ,
        linewidth=2.7,
        linestyle="--",
        alpha=0.88,
        label=(
            "STLSQ "
            rf"($\lambda_{{th}}="
            f"{best_stlsq['parameter']:g}$)"
        ),
        zorder=8
    )


# ------------------------------------------------------------
# Best LASSO
# ------------------------------------------------------------

if q_lasso is not None:

    ax.plot(
        t_eval,
        q_lasso,
        color=COLOR_LASSO,
        linewidth=2.5,
        linestyle="-.",
        alpha=0.82,
        label=(
            "LASSO "
            rf"($\alpha="
            f"{best_lasso['parameter']:.0e}$)"
        ),
        zorder=7
    )


# ------------------------------------------------------------
# Over-sparse example
# ------------------------------------------------------------

# Only draw it if it is different from best STLSQ.
if (
    q_sparse is not None
    and over_sparse["parameter"]
    != best_stlsq["parameter"]
):

    ax.plot(
        t_eval,
        q_sparse,
        color=COLOR_BAD,
        linewidth=2.0,
        linestyle=":",
        alpha=0.45,
        label=(
            "Over-sparse "
            rf"($\lambda_{{th}}="
            f"{over_sparse['parameter']:g}$)"
        ),
        zorder=5
    )


ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$q$ [rad]")

ax.grid(False)

ax.legend(
    frameon=False,
    loc="upper right",
    ncol=2
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_rollout_comparison.png"
    ),
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)


# ============================================================
# 23. Save equations to text file
# ============================================================

equation_file = os.path.join(
    OUTPUT_DIR,
    "recovered_equations.txt"
)

with open(
    equation_file,
    "w"
) as f:

    f.write(
        "GROUND TRUTH\n"
    )

    f.write(
        "ddq = -0.100 dq - 9.810 sin(q)\n\n"
    )

    f.write(
        "=" * 70
        + "\nSTLSQ\n"
        + "=" * 70
        + "\n"
    )

    for result in stlsq_results:

        f.write(
            f"\nthreshold = "
            f"{result['parameter']:g}\n"
        )

        f.write(
            format_equation_text(
                result["coef"]
            )
            + "\n"
        )

        f.write(
            f"active terms = "
            f"{result['active']}\n"
        )

        f.write(
            f"derivative RMSE = "
            f"{result['derivative_rmse']:.6f}\n"
        )

        f.write(
            f"rollout RMSE = "
            f"{result['rollout_rmse']:.6f}\n"
        )

    f.write(
        "\n"
        + "=" * 70
        + "\nLASSO\n"
        + "=" * 70
        + "\n"
    )

    for result in lasso_results:

        f.write(
            f"\nalpha = "
            f"{result['parameter']:.0e}\n"
        )

        f.write(
            format_equation_text(
                result["coef"]
            )
            + "\n"
        )

        f.write(
            f"active terms = "
            f"{result['active']}\n"
        )

        f.write(
            f"derivative RMSE = "
            f"{result['derivative_rmse']:.6f}\n"
        )

        f.write(
            f"rollout RMSE = "
            f"{result['rollout_rmse']:.6f}\n"
        )


# ============================================================
# 24. Final summary
# ============================================================

print("\n")
print("=" * 75)
print("FILES SAVED")
print("=" * 75)

print(
    os.path.join(
        OUTPUT_DIR,
        "01_measurements.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "02_stlsq_equations.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "03_lasso_equations.png"
    )
)

print(
    os.path.join(
        OUTPUT_DIR,
        "04_rollout_comparison.png"
    )
)

print(
    equation_file
)