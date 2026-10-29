"""
same_data_different_bias.py

Teaching demo:
    Same Data, Different Inductive Bias

System:
    Conservative pendulum

Models:
    1. Baseline NN
       (q, q_dot) -> q_ddot

    2. Structured Lagrangian NN
       L_theta(q, q_dot)
           = 1/2 * m_theta * q_dot^2 - V_theta(q)

Both models:
    - use exactly the same sparse + noisy training data
    - are trained from acceleration supervision
    - are evaluated from exactly the same unseen initial condition

Outputs are automatically organized in:

same_data_different_bias_results/
    figures/
    animation/
    data/
"""

# ============================================================
# 0. Imports
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.integrate import solve_ivp


# ============================================================
# 1. Global settings
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 70)
print("Same Data, Different Inductive Bias")
print("=" * 70)
print("Device:", device)


# ============================================================
# 2. Output folders
# ============================================================

OUTPUT_DIR = "same_data_different_bias_results"

FIGURE_DIR = os.path.join(
    OUTPUT_DIR,
    "figures",
)

ANIMATION_DIR = os.path.join(
    OUTPUT_DIR,
    "animation",
)

DATA_DIR = os.path.join(
    OUTPUT_DIR,
    "data",
)


os.makedirs(
    FIGURE_DIR,
    exist_ok=True,
)

os.makedirs(
    ANIMATION_DIR,
    exist_ok=True,
)

os.makedirs(
    DATA_DIR,
    exist_ok=True,
)


print()
print("All outputs will be saved in:")
print(
    os.path.abspath(
        OUTPUT_DIR
    )
)


# ============================================================
# 3. Ground-truth pendulum
# ============================================================

g = 9.81
l = 1.0
m_true = 1.0


def true_acceleration(q):
    """
    Conservative pendulum:

        q_ddot = -(g/l) sin(q)
    """

    return -(g / l) * np.sin(q)


def pendulum_rhs(t, x):

    q = x[0]
    q_dot = x[1]

    q_ddot = true_acceleration(q)

    return [
        q_dot,
        q_ddot,
    ]


def simulate_true_system(
    q0,
    q_dot0,
    T=5.0,
    dt=0.02,
):

    N = int(
        np.round(T / dt)
    ) + 1

    t_eval = np.linspace(
        0.0,
        T,
        N,
    )

    sol = solve_ivp(
        pendulum_rhs,
        (0.0, T),
        [q0, q_dot0],
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-10,
    )

    q = sol.y[0]

    q_dot = sol.y[1]

    q_ddot = true_acceleration(q)

    return (
        t_eval,
        q,
        q_dot,
        q_ddot,
    )


# ============================================================
# 4. SAME sparse + noisy training data
# ============================================================

train_initial_conditions = [

    (-1.20,  0.00),

    (-0.90,  0.55),

    (-0.60, -0.70),

    (-0.30,  0.90),

    ( 0.30, -0.90),

    ( 0.60,  0.70),

    ( 0.90, -0.55),

    ( 1.20,  0.00),

]


TRAIN_T = 2.0

TRAIN_DT = 0.05

NOISE_STD = 0.30


rng = np.random.default_rng(
    SEED
)


X_list = []

Y_clean_list = []

Y_noisy_list = []


for q0, q_dot0 in train_initial_conditions:

    (
        t,
        q,
        q_dot,
        q_ddot,
    ) = simulate_true_system(
        q0,
        q_dot0,
        T=TRAIN_T,
        dt=TRAIN_DT,
    )

    q_ddot_noisy = (
        q_ddot
        +
        rng.normal(
            0.0,
            NOISE_STD,
            size=q_ddot.shape,
        )
    )

    X_i = np.column_stack(
        [
            q,
            q_dot,
        ]
    )

    Y_clean_i = q_ddot[:, None]

    Y_noisy_i = q_ddot_noisy[:, None]

    X_list.append(
        X_i
    )

    Y_clean_list.append(
        Y_clean_i
    )

    Y_noisy_list.append(
        Y_noisy_i
    )


X_np = np.concatenate(
    X_list,
    axis=0,
)

Y_clean_np = np.concatenate(
    Y_clean_list,
    axis=0,
)

Y_noisy_np = np.concatenate(
    Y_noisy_list,
    axis=0,
)


X_train = torch.tensor(
    X_np,
    dtype=torch.float32,
    device=device,
)

Y_train = torch.tensor(
    Y_noisy_np,
    dtype=torch.float32,
    device=device,
)


print()
print("Training samples:", len(X_train))

print(
    "q range:",
    X_np[:, 0].min(),
    "to",
    X_np[:, 0].max(),
)

print(
    "q_dot range:",
    X_np[:, 1].min(),
    "to",
    X_np[:, 1].max(),
)


# ============================================================
# 5. Plot training data
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.5, 5.5)
)

ax.scatter(
    X_np[:, 0],
    X_np[:, 1],
    s=15,
    alpha=0.5,
)

ax.set_xlabel(
    r"$q$ [rad]",
    fontsize=14,
)

ax.set_ylabel(
    r"$\dot q$ [rad/s]",
    fontsize=14,
)

ax.set_title(
    "Training data",
    fontsize=15,
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        FIGURE_DIR,
        "01_training_data.png",
    ),
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 6. Generic MLP
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=64,
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),

            nn.Tanh(),

            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )

        self.initialize_weights()

    def initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear,
            ):

                nn.init.xavier_uniform_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    def forward(
        self,
        x,
    ):

        return self.net(x)


# ============================================================
# 7. Baseline NN
# ============================================================

class BaselineNN(nn.Module):

    """
    Direct black-box dynamics:

        (q, q_dot)
              |
              v
             NN
              |
              v
           q_ddot
    """

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(2, 128),

            nn.Tanh(),

            nn.Linear(128, 128),

            nn.Tanh(),

            nn.Linear(128, 1),

        )

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear,
            ):

                nn.init.xavier_uniform_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    def acceleration(
        self,
        q,
        q_dot,
    ):

        x = torch.cat(
            [
                q,
                q_dot,
            ],
            dim=-1,
        )

        return self.net(x)


# ============================================================
# 8. Structured Lagrangian NN
# ============================================================

class StructuredLNN(nn.Module):

    """
    Structured Lagrangian model:

        L_theta(q, q_dot)
            =
            1/2 m_theta q_dot^2
            -
            V_theta(q)

    where

        m_theta > 0

    Dynamics:

        m_theta q_ddot
        +
        dV_theta/dq
        =
        0

    therefore

        q_ddot
        =
        -(1/m_theta) dV_theta/dq
    """

    def __init__(self):

        super().__init__()

        self.V_net = MLP(
            input_dim=1,
            output_dim=1,
            hidden_dim=64,
        )

        self.raw_mass = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

        self.m_min = 0.20

    def mass(self):

        return (
            F.softplus(
                self.raw_mass
            )
            +
            self.m_min
        )

    def potential(
        self,
        q,
    ):

        return self.V_net(q)

    def lagrangian(
        self,
        q,
        q_dot,
    ):

        kinetic = (
            0.5
            *
            self.mass()
            *
            q_dot**2
        )

        potential = self.potential(
            q
        )

        return (
            kinetic
            -
            potential
        )

    def acceleration(
        self,
        q,
        q_dot,
        create_graph=True,
    ):

        if not q.requires_grad:

            q = (
                q.clone()
                .detach()
                .requires_grad_(True)
            )

        V = self.potential(q)

        dV_dq = torch.autograd.grad(
            outputs=V.sum(),
            inputs=q,
            create_graph=create_graph,
            retain_graph=create_graph,
        )[0]

        q_ddot = (
            -dV_dq
            /
            self.mass()
        )

        return q_ddot


# ============================================================
# 9. Create models
# ============================================================

baseline = BaselineNN().to(
    device
)

structured_lnn = StructuredLNN().to(
    device
)


# ============================================================
# 10. Train Baseline NN
# ============================================================

def train_baseline(
    model,
    X,
    Y,
    epochs=4000,
    lr=1e-3,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-6,
    )

    history = []

    q = X[:, 0:1]

    q_dot = X[:, 1:2]


    for epoch in range(epochs):

        optimizer.zero_grad()

        prediction = model.acceleration(
            q,
            q_dot,
        )

        loss = torch.mean(
            (
                prediction
                -
                Y
            )**2
        )

        loss.backward()

        optimizer.step()

        history.append(
            loss.item()
        )

        if (
            epoch % 500 == 0
            or
            epoch == epochs - 1
        ):

            print(
                f"[Baseline] "
                f"epoch={epoch:4d} "
                f"loss={loss.item():.6e}"
            )

    return np.asarray(
        history
    )


# ============================================================
# 11. Train Structured LNN
# ============================================================

def train_structured_lnn(
    model,
    X,
    Y,
    epochs=4000,
    lr=1e-3,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-6,
    )

    history = []


    for epoch in range(epochs):

        optimizer.zero_grad()

        q = (
            X[:, 0:1]
            .clone()
            .detach()
            .requires_grad_(True)
        )

        q_dot = X[:, 1:2]

        prediction = model.acceleration(
            q,
            q_dot,
            create_graph=True,
        )

        loss = torch.mean(
            (
                prediction
                -
                Y
            )**2
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        history.append(
            loss.item()
        )

        if (
            epoch % 500 == 0
            or
            epoch == epochs - 1
        ):

            print(
                f"[Structured LNN] "
                f"epoch={epoch:4d} "
                f"loss={loss.item():.6e} "
                f"mass={model.mass().item():.4f}"
            )

    return np.asarray(
        history
    )


# ============================================================
# 12. Train both models
# ============================================================

print()
print("=" * 70)
print("Training Baseline NN")
print("=" * 70)


baseline_history = train_baseline(
    baseline,
    X_train,
    Y_train,
)


print()
print("=" * 70)
print("Training Structured Lagrangian NN")
print("=" * 70)


lnn_history = train_structured_lnn(
    structured_lnn,
    X_train,
    Y_train,
)


# ============================================================
# 13. Training loss
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 4.5)
)

ax.semilogy(
    baseline_history,
    linewidth=2.2,
    label="Baseline NN",
)

ax.semilogy(
    lnn_history,
    linewidth=2.2,
    label="Structured Lagrangian NN",
)

ax.set_xlabel(
    "Epoch",
    fontsize=13,
)

ax.set_ylabel(
    "Training MSE",
    fontsize=13,
)

ax.legend(
    frameon=False,
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        FIGURE_DIR,
        "02_training_loss.png",
    ),
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 14. Acceleration field
# ============================================================

q_grid_np = np.linspace(
    -1.4,
    1.4,
    400,
)

q_dot_grid_np = np.zeros_like(
    q_grid_np
)


q_grid = torch.tensor(
    q_grid_np[:, None],
    dtype=torch.float32,
    device=device,
)

q_dot_grid = torch.tensor(
    q_dot_grid_np[:, None],
    dtype=torch.float32,
    device=device,
)


acc_true_grid = true_acceleration(
    q_grid_np
)


baseline.eval()

with torch.no_grad():

    acc_base_grid = (
        baseline.acceleration(
            q_grid,
            q_dot_grid,
        )
        .cpu()
        .numpy()
        .squeeze()
    )


structured_lnn.eval()

q_grid_lnn = (
    q_grid
    .clone()
    .detach()
    .requires_grad_(True)
)

acc_lnn_grid = (
    structured_lnn.acceleration(
        q_grid_lnn,
        q_dot_grid,
        create_graph=False,
    )
    .detach()
    .cpu()
    .numpy()
    .squeeze()
)


fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.plot(
    q_grid_np,
    acc_true_grid,
    linewidth=3.0,
    label="Ground truth",
)

ax.plot(
    q_grid_np,
    acc_base_grid,
    "--",
    linewidth=2.4,
    label="Baseline NN",
)

ax.plot(
    q_grid_np,
    acc_lnn_grid,
    "-.",
    linewidth=2.4,
    label="Structured Lagrangian NN",
)

ax.scatter(
    X_np[:, 0],
    Y_noisy_np[:, 0],
    s=10,
    alpha=0.12,
    label="Training data",
)

ax.set_xlabel(
    r"$q$ [rad]",
    fontsize=14,
)

ax.set_ylabel(
    r"$\ddot q$ [rad/s$^2$]",
    fontsize=14,
)

ax.legend(
    frameon=False,
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        FIGURE_DIR,
        "03_acceleration_field.png",
    ),
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 15. NumPy acceleration wrappers
# ============================================================

def baseline_acceleration_numpy(
    q,
    q_dot,
):

    q_tensor = torch.tensor(
        [[q]],
        dtype=torch.float32,
        device=device,
    )

    q_dot_tensor = torch.tensor(
        [[q_dot]],
        dtype=torch.float32,
        device=device,
    )

    baseline.eval()

    with torch.no_grad():

        q_ddot = baseline.acceleration(
            q_tensor,
            q_dot_tensor,
        )

    return float(
        q_ddot.item()
    )


def lnn_acceleration_numpy(
    q,
    q_dot,
):

    q_tensor = torch.tensor(
        [[q]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )

    q_dot_tensor = torch.tensor(
        [[q_dot]],
        dtype=torch.float32,
        device=device,
    )

    structured_lnn.eval()

    q_ddot = structured_lnn.acceleration(
        q_tensor,
        q_dot_tensor,
        create_graph=False,
    )

    return float(
        q_ddot
        .detach()
        .item()
    )


# ============================================================
# 16. RK4 rollout
# ============================================================

def rollout(
    acceleration_function,
    x0,
    T=15.0,
    dt=0.01,
):

    N = int(
        np.round(
            T / dt
        )
    ) + 1

    t = np.linspace(
        0.0,
        T,
        N,
    )

    x = np.zeros(
        (N, 2),
        dtype=np.float64,
    )

    x[0] = x0


    def dynamics(state):

        q = state[0]

        q_dot = state[1]

        q_ddot = acceleration_function(
            q,
            q_dot,
        )

        return np.array(
            [
                q_dot,
                q_ddot,
            ]
        )


    for k in range(
        N - 1
    ):

        s = x[k]

        k1 = dynamics(s)

        k2 = dynamics(
            s
            +
            0.5
            *
            dt
            *
            k1
        )

        k3 = dynamics(
            s
            +
            0.5
            *
            dt
            *
            k2
        )

        k4 = dynamics(
            s
            +
            dt
            *
            k3
        )

        x[k + 1] = (
            s
            +
            dt
            /
            6.0
            *
            (
                k1
                +
                2 * k2
                +
                2 * k3
                +
                k4
            )
        )

    return (
        t,
        x,
    )


# ============================================================
# 17. SAME unseen test condition
# ============================================================

q0_test = 1.00

q_dot0_test = 0.20


TEST_T = 15.0

TEST_DT = 0.01


# Ground truth
(
    t_true,
    q_true,
    q_dot_true,
    q_ddot_true,
) = simulate_true_system(
    q0_test,
    q_dot0_test,
    T=TEST_T,
    dt=TEST_DT,
)


# Baseline
(
    t_base,
    x_base,
) = rollout(
    baseline_acceleration_numpy,
    [q0_test, q_dot0_test],
    T=TEST_T,
    dt=TEST_DT,
)


# Structured LNN
(
    t_lnn,
    x_lnn,
) = rollout(
    lnn_acceleration_numpy,
    [q0_test, q_dot0_test],
    T=TEST_T,
    dt=TEST_DT,
)


# ============================================================
# 18. Mechanical energy
# ============================================================

def physical_energy(
    q,
    q_dot,
):

    kinetic = (
        0.5
        *
        m_true
        *
        l**2
        *
        q_dot**2
    )

    potential = (
        m_true
        *
        g
        *
        l
        *
        (
            1.0
            -
            np.cos(q)
        )
    )

    return (
        kinetic
        +
        potential
    )


E_true = physical_energy(
    q_true,
    q_dot_true,
)

E_base = physical_energy(
    x_base[:, 0],
    x_base[:, 1],
)

E_lnn = physical_energy(
    x_lnn[:, 0],
    x_lnn[:, 1],
)


def relative_energy_drift(E):

    return (
        E
        -
        E[0]
    ) / E[0]


dE_true = relative_energy_drift(
    E_true
)

dE_base = relative_energy_drift(
    E_base
)

dE_lnn = relative_energy_drift(
    E_lnn
)


# ============================================================
# 19. Metrics
# ============================================================

q_rmse_base = np.sqrt(
    np.mean(
        (
            x_base[:, 0]
            -
            q_true
        )**2
    )
)

q_rmse_lnn = np.sqrt(
    np.mean(
        (
            x_lnn[:, 0]
            -
            q_true
        )**2
    )
)


max_energy_drift_base = np.max(
    np.abs(
        dE_base
    )
)

max_energy_drift_lnn = np.max(
    np.abs(
        dE_lnn
    )
)


print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"Baseline q RMSE: "
    f"{q_rmse_base:.6f} rad"
)

print(
    f"LNN q RMSE: "
    f"{q_rmse_lnn:.6f} rad"
)

print()

print(
    f"Baseline max energy drift: "
    f"{max_energy_drift_base:.6f}"
)

print(
    f"LNN max energy drift: "
    f"{max_energy_drift_lnn:.6f}"
)

print()

print(
    f"Learned LNN mass: "
    f"{structured_lnn.mass().item():.6f}"
)


# ============================================================
# 20. SAVE rollout data
# ============================================================

np.savez(
    os.path.join(
        DATA_DIR,
        "rollout_data.npz",
    ),

    t=t_true,

    q_true=q_true,

    qdot_true=q_dot_true,

    q_baseline=x_base[:, 0],

    qdot_baseline=x_base[:, 1],

    q_lnn=x_lnn[:, 0],

    qdot_lnn=x_lnn[:, 1],

    E_true=E_true,

    E_baseline=E_base,

    E_lnn=E_lnn,
)


# ============================================================
# 21. SAVE metrics
# ============================================================

metrics_path = os.path.join(
    DATA_DIR,
    "metrics.txt",
)


with open(
    metrics_path,
    "w",
) as f:

    f.write(
        "Same Data, Different Inductive Bias\n"
    )

    f.write(
        "=" * 50 + "\n\n"
    )

    f.write(
        f"Baseline q RMSE: "
        f"{q_rmse_base:.6f} rad\n"
    )

    f.write(
        f"LNN q RMSE: "
        f"{q_rmse_lnn:.6f} rad\n\n"
    )

    f.write(
        f"Baseline max energy drift: "
        f"{max_energy_drift_base:.6f}\n"
    )

    f.write(
        f"LNN max energy drift: "
        f"{max_energy_drift_lnn:.6f}\n\n"
    )

    f.write(
        f"Learned LNN mass: "
        f"{structured_lnn.mass().item():.6f}\n"
    )


# ============================================================
# 22. DIRECT TRAJECTORY COMPARISON
#
# THIS is probably the main figure for your PPT.
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 5.3)
)


ax.plot(
    t_true,
    q_true,
    linewidth=3.2,
    label="Ground truth",
)


ax.plot(
    t_base,
    x_base[:, 0],
    "--",
    linewidth=2.6,
    label="Baseline NN",
)


ax.plot(
    t_lnn,
    x_lnn[:, 0],
    "-.",
    linewidth=2.6,
    label="Structured Lagrangian NN",
)


ax.set_xlabel(
    "Time [s]",
    fontsize=15,
)


ax.set_ylabel(
    r"$q$ [rad]",
    fontsize=15,
)


ax.tick_params(
    labelsize=12,
)


ax.legend(
    frameon=False,
    fontsize=12,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        1.13,
    ),
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        FIGURE_DIR,
        "04_trajectory_comparison.png",
    ),
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# 23. PHASE PORTRAIT
# ============================================================

fig, ax = plt.subplots(
    figsize=(6.5, 6.0)
)


ax.plot(
    q_true,
    q_dot_true,
    linewidth=3.0,
    label="Ground truth",
)


ax.plot(
    x_base[:, 0],
    x_base[:, 1],
    "--",
    linewidth=2.5,
    label="Baseline NN",
)


ax.plot(
    x_lnn[:, 0],
    x_lnn[:, 1],
    "-.",
    linewidth=2.5,
    label="Structured Lagrangian NN",
)


ax.scatter(
    [q_true[0]],
    [q_dot_true[0]],
    s=70,
    zorder=10,
)


ax.set_xlabel(
    r"$q$ [rad]",
    fontsize=15,
)


ax.set_ylabel(
    r"$\dot q$ [rad/s]",
    fontsize=15,
)


ax.tick_params(
    labelsize=12,
)


ax.legend(
    frameon=False,
    fontsize=11,
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        FIGURE_DIR,
        "05_phase_portrait.png",
    ),
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# 24. ENERGY COMPARISON
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 4.8)
)


ax.plot(
    t_true,
    E_true,
    linewidth=3.0,
    label="Ground truth",
)


ax.plot(
    t_base,
    E_base,
    "--",
    linewidth=2.5,
    label="Baseline NN",
)


ax.plot(
    t_lnn,
    E_lnn,
    "-.",
    linewidth=2.5,
    label="Structured Lagrangian NN",
)


ax.set_xlabel(
    "Time [s]",
    fontsize=15,
)


ax.set_ylabel(
    "Mechanical energy",
    fontsize=15,
)


ax.tick_params(
    labelsize=12,
)


ax.legend(
    frameon=False,
    fontsize=12,
    ncol=3,
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        FIGURE_DIR,
        "06_energy_comparison.png",
    ),
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# 25. ENERGY DRIFT
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 4.8)
)


ax.plot(
    t_true,
    dE_true,
    linewidth=3.0,
    label="Ground truth",
)


ax.plot(
    t_base,
    dE_base,
    "--",
    linewidth=2.5,
    label="Baseline NN",
)


ax.plot(
    t_lnn,
    dE_lnn,
    "-.",
    linewidth=2.5,
    label="Structured Lagrangian NN",
)


ax.axhline(
    0,
    linewidth=1.0,
    alpha=0.4,
)


ax.set_xlabel(
    "Time [s]",
    fontsize=15,
)


ax.set_ylabel(
    r"Relative energy drift $(E-E_0)/E_0$",
    fontsize=14,
)


ax.tick_params(
    labelsize=12,
)


ax.legend(
    frameon=False,
    fontsize=12,
    ncol=3,
)


fig.tight_layout()


fig.savefig(
    os.path.join(
        FIGURE_DIR,
        "07_energy_drift.png",
    ),
    dpi=300,
    transparent=True,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# 26. PENDULUM ANIMATION
#
# IMPORTANT:
#
# All THREE trajectories are drawn in the SAME physical space.
#
# Therefore:
#
#   Ground truth
#   Baseline NN
#   Structured LNN
#
# can be directly compared visually.
# ============================================================

FPS = 30

ANIMATION_T = 12.0


# Only use first ANIMATION_T seconds
animation_mask = (
    t_true <= ANIMATION_T
)


t_anim_full = t_true[
    animation_mask
]


q_true_anim_full = q_true[
    animation_mask
]


q_base_anim_full = x_base[
    :len(t_anim_full),
    0
]


q_lnn_anim_full = x_lnn[
    :len(t_anim_full),
    0
]


# ============================================================
# Downsample for GIF
# ============================================================

target_frames = int(
    FPS
    *
    ANIMATION_T
)


stride = max(
    1,
    len(t_anim_full)
    //
    target_frames
)


frame_indices = np.arange(
    0,
    len(t_anim_full),
    stride,
)


print()
print(
    "Animation frames:",
    len(frame_indices),
)


# ============================================================
# Pendulum geometry
# ============================================================

def pendulum_position(q):

    x = (
        l
        *
        np.sin(q)
    )

    y = (
        -l
        *
        np.cos(q)
    )

    return (
        x,
        y,
    )


# ============================================================
# Create animation canvas
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.0, 6.2)
)


ax.set_xlim(
    -1.25,
    1.25,
)


ax.set_ylim(
    -1.20,
    0.30,
)


ax.set_aspect(
    "equal"
)


ax.axis(
    "off"
)


# ============================================================
# Pivot
# ============================================================

ax.scatter(
    [0],
    [0],
    s=80,
    zorder=10,
)


# ============================================================
# Ground truth artists
# ============================================================

rod_true, = ax.plot(
    [],
    [],
    linewidth=4.0,
    label="Ground truth",
)


bob_true, = ax.plot(
    [],
    [],
    "o",
    markersize=15,
)


# ============================================================
# Baseline artists
# ============================================================

rod_base, = ax.plot(
    [],
    [],
    "--",
    linewidth=3.0,
    label="Baseline NN",
)


bob_base, = ax.plot(
    [],
    [],
    "o",
    markersize=12,
)


# ============================================================
# Structured LNN artists
# ============================================================

rod_lnn, = ax.plot(
    [],
    [],
    "-.",
    linewidth=3.0,
    label="Structured Lagrangian NN",
)


bob_lnn, = ax.plot(
    [],
    [],
    "o",
    markersize=12,
)


# ============================================================
# Trails
# ============================================================

trail_true, = ax.plot(
    [],
    [],
    linewidth=1.5,
    alpha=0.30,
)


trail_base, = ax.plot(
    [],
    [],
    "--",
    linewidth=1.3,
    alpha=0.25,
)


trail_lnn, = ax.plot(
    [],
    [],
    "-.",
    linewidth=1.3,
    alpha=0.25,
)


# ============================================================
# Time label
# ============================================================

time_text = ax.text(
    0.5,
    0.94,
    "",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=15,
)


# ============================================================
# Legend
# ============================================================

ax.legend(
    loc="lower center",
    bbox_to_anchor=(
        0.5,
        -0.08,
    ),
    frameon=False,
    fontsize=11,
    ncol=3,
)


# ============================================================
# Trail length
# ============================================================

TRAIL_SECONDS = 0.8


trail_samples = int(
    TRAIL_SECONDS
    /
    TEST_DT
)


# ============================================================
# Animation update
# ============================================================

def update_animation(
    frame_number,
):

    idx = frame_indices[
        frame_number
    ]


    # --------------------------------------------------------
    # Current positions
    # --------------------------------------------------------

    x_true_now, y_true_now = pendulum_position(
        q_true_anim_full[idx]
    )


    x_base_now, y_base_now = pendulum_position(
        q_base_anim_full[idx]
    )


    x_lnn_now, y_lnn_now = pendulum_position(
        q_lnn_anim_full[idx]
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    rod_true.set_data(
        [
            0,
            x_true_now,
        ],
        [
            0,
            y_true_now,
        ],
    )


    bob_true.set_data(
        [x_true_now],
        [y_true_now],
    )


    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    rod_base.set_data(
        [
            0,
            x_base_now,
        ],
        [
            0,
            y_base_now,
        ],
    )


    bob_base.set_data(
        [x_base_now],
        [y_base_now],
    )


    # --------------------------------------------------------
    # Structured LNN
    # --------------------------------------------------------

    rod_lnn.set_data(
        [
            0,
            x_lnn_now,
        ],
        [
            0,
            y_lnn_now,
        ],
    )


    bob_lnn.set_data(
        [x_lnn_now],
        [y_lnn_now],
    )


    # ========================================================
    # Trails
    # ========================================================

    start_idx = max(
        0,
        idx - trail_samples,
    )


    q_true_trail = q_true_anim_full[
        start_idx:
        idx + 1
    ]


    q_base_trail = q_base_anim_full[
        start_idx:
        idx + 1
    ]


    q_lnn_trail = q_lnn_anim_full[
        start_idx:
        idx + 1
    ]


    x_true_trail = (
        l
        *
        np.sin(
            q_true_trail
        )
    )


    y_true_trail = (
        -l
        *
        np.cos(
            q_true_trail
        )
    )


    x_base_trail = (
        l
        *
        np.sin(
            q_base_trail
        )
    )


    y_base_trail = (
        -l
        *
        np.cos(
            q_base_trail
        )
    )


    x_lnn_trail = (
        l
        *
        np.sin(
            q_lnn_trail
        )
    )


    y_lnn_trail = (
        -l
        *
        np.cos(
            q_lnn_trail
        )
    )


    trail_true.set_data(
        x_true_trail,
        y_true_trail,
    )


    trail_base.set_data(
        x_base_trail,
        y_base_trail,
    )


    trail_lnn.set_data(
        x_lnn_trail,
        y_lnn_trail,
    )


    # ========================================================
    # Time
    # ========================================================

    time_text.set_text(
        f"t = {t_anim_full[idx]:.1f} s"
    )


    return (

        rod_true,
        bob_true,

        rod_base,
        bob_base,

        rod_lnn,
        bob_lnn,

        trail_true,
        trail_base,
        trail_lnn,

        time_text,
    )


# ============================================================
# Build animation
# ============================================================

animation = FuncAnimation(

    fig,

    update_animation,

    frames=len(
        frame_indices
    ),

    interval=(
        1000
        /
        FPS
    ),

    blit=True,
)


# ============================================================
# Save GIF
# ============================================================

gif_path = os.path.join(
    ANIMATION_DIR,
    "pendulum_comparison.gif",
)


print()
print("Saving animation...")

print(
    "This may take a little while."
)


writer = PillowWriter(
    fps=FPS,
)


animation.save(
    gif_path,
    writer=writer,
    dpi=120,
)


plt.close(fig)


# ============================================================
# 27. Finish
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)


print()
print(
    "Everything is inside ONE folder:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)


print()
print("Figures:")

for filename in sorted(
    os.listdir(
        FIGURE_DIR
    )
):

    print(
        "   ",
        filename,
    )


print()
print("Animation:")

for filename in sorted(
    os.listdir(
        ANIMATION_DIR
    )
):

    print(
        "   ",
        filename,
    )


print()
print("Data:")

for filename in sorted(
    os.listdir(
        DATA_DIR
    )
):

    print(
        "   ",
        filename,
    )


print()
print(
    "Main PPT figure:"
)

print(
    os.path.join(
        FIGURE_DIR,
        "04_trajectory_comparison.png",
    )
)


print()
print(
    "Main PPT animation:"
)

print(
    os.path.join(
        ANIMATION_DIR,
        "pendulum_comparison.gif",
    )
)