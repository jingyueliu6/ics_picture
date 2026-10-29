import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image
import os

# ============================================================
# 1. Numerical example
# ============================================================

x = np.array([1.0, 2.0])

W1 = np.array([
    [1.0, -0.5],
    [0.5,  1.0]
])

b1 = np.array([0.5, -0.5])

W2 = np.array([1.0, 1.0])
b2 = 0.0

y = 1.5


# ---------- Forward pass ----------
z1 = W1 @ x + b1
h = np.maximum(0.0, z1)       # ReLU
y_hat = W2 @ h + b2
J = 0.5 * (y_hat - y) ** 2


# ---------- Backward pass ----------
delta2 = y_hat - y

grad_W2 = delta2 * h

dJ_dh = W2 * delta2

relu_prime = (z1 > 0).astype(float)

delta1 = dJ_dh * relu_prime

grad_W1 = np.outer(delta1, x)


# ============================================================
# 2. Print values in terminal
# ============================================================

print("\n========== FORWARD ==========")
print("x =")
print(x)

print("\nW1 =")
print(W1)

print("\nb1 =")
print(b1)

print("\nz1 = W1 x + b1 =")
print(z1)

print("\nh = ReLU(z1) =")
print(h)

print("\nW2 =")
print(W2)

print("\ny_hat =", y_hat)
print("target y =", y)
print("J =", J)


print("\n========== BACKWARD ==========")
print("delta2 = dJ/dy_hat =", delta2)

print("\ngrad_W2 =")
print(grad_W2)

print("\ndJ/dh =")
print(dJ_dh)

print("\nReLU'(z1) =")
print(relu_prime)

print("\ndelta1 =")
print(delta1)

print("\ngrad_W1 =")
print(grad_W1)


# ============================================================
# 3. Figure setup
# ============================================================

fig, ax = plt.subplots(figsize=(16, 9))

# Fixed node positions
pos = {
    "x1": (1.6, 5.8),
    "x2": (1.6, 3.0),

    "h1": (6.3, 5.8),
    "h2": (6.3, 3.0),

    "yhat": (10.3, 4.4),
    "loss": (13.5, 4.4)
}

BLUE = "#3F8FD2"
LIGHT_BLUE = "#DCECF8"

RED = "#D9534F"
LIGHT_RED = "#F9DEDC"

GRAY = "#C8C8C8"
DARK = "#222222"

GREEN = "#55A868"
LIGHT_GREEN = "#E3F1E3"


# ============================================================
# 4. Drawing helpers
# ============================================================

def setup_canvas():
    """Reset and redraw the clean canvas."""

    ax.clear()

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)

    ax.axis("off")


def node(xy, text, active=False, backward=False):
    """Draw one network node."""

    if backward:
        edge = RED
        face = LIGHT_RED
    elif active:
        edge = BLUE
        face = LIGHT_BLUE
    else:
        edge = "#AAAAAA"
        face = "#F5F5F5"

    circle = Circle(
        xy,
        radius=0.48,
        facecolor=face,
        edgecolor=edge,
        linewidth=2.5
    )

    ax.add_patch(circle)

    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        fontsize=20
    )


def arrow(p1, p2, color=GRAY, lw=2.0, alpha=1.0):
    """Draw an arrow between two points."""

    a = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="->",
        mutation_scale=18,
        linewidth=lw,
        color=color,
        alpha=alpha,
        shrinkA=28,
        shrinkB=28
    )

    ax.add_patch(a)


def box(x, y, width, height, text,
        edgecolor=BLUE,
        facecolor=LIGHT_BLUE,
        fontsize=18):

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=2,
        edgecolor=edgecolor,
        facecolor=facecolor
    )

    ax.add_patch(patch)

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize
    )


def title(main, subtitle):
    ax.text(
        7.5,
        8.55,
        main,
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold"
    )

    ax.text(
        7.5,
        8.05,
        subtitle,
        ha="center",
        va="center",
        fontsize=18
    )


def draw_connections(color=GRAY, lw=2.0):

    # input -> hidden
    arrow(pos["x1"], pos["h1"], color, lw)
    arrow(pos["x1"], pos["h2"], color, lw)
    arrow(pos["x2"], pos["h1"], color, lw)
    arrow(pos["x2"], pos["h2"], color, lw)

    # hidden -> output
    arrow(pos["h1"], pos["yhat"], color, lw)
    arrow(pos["h2"], pos["yhat"], color, lw)

    # output -> loss
    arrow(pos["yhat"], pos["loss"], color, lw)


def draw_base_network(
    show_hidden=False,
    show_output=False,
    show_loss=False
):

    # Always show input
    node(pos["x1"], r"$x_1=1$", active=True)
    node(pos["x2"], r"$x_2=2$", active=True)

    # Connections
    draw_connections()

    # hidden
    if show_hidden:
        node(pos["h1"], r"$h_1=0.5$", active=True)
        node(pos["h2"], r"$h_2=2$", active=True)

    # output
    if show_output:
        node(pos["yhat"], r"$\hat{y}=2.5$", active=True)

    # loss
    if show_loss:
        node(pos["loss"], r"$J=0.5$", active=True)


# ============================================================
# 5. Individual animation frames
# ============================================================

def frame_0():
    setup_canvas()

    title(
        "Forward pass",
        "Start with the input and model parameters"
    )

    node(pos["x1"], r"$x_1=1$", active=True)
    node(pos["x2"], r"$x_2=2$", active=True)

    ax.text(
        1.6,
        7.0,
        r"$\mathbf{x}=[1,\;2]^\top$",
        ha="center",
        fontsize=21
    )

    box(
        3.1, 6.45, 3.2, 1.05,
        "$W_1$ = [ 1    -0.5 ]\n"
        "       [ 0.5   1   ]"
        "\n"
        r"$b_1=[0.5,\;-0.5]^\top$",
        fontsize=16
    )


def frame_1():
    setup_canvas()

    title(
        "Forward pass",
        r"Compute the pre-activation:  $\mathbf{z}_1=W_1\mathbf{x}+\mathbf{b}_1$"
    )

    node(pos["x1"], r"$x_1=1$", active=True)
    node(pos["x2"], r"$x_2=2$", active=True)

    # highlight input-hidden connections
    arrow(pos["x1"], pos["h1"], BLUE, 3)
    arrow(pos["x1"], pos["h2"], BLUE, 3)
    arrow(pos["x2"], pos["h1"], BLUE, 3)
    arrow(pos["x2"], pos["h2"], BLUE, 3)

    box(
        4.0, 0.75, 4.7, 1.75,
        "z₁ = W₁x + b₁\n\n"
        "W₁ = [ 1.0   -0.5 ]     x = [ 1 ]     b₁ = [ 0.5  ]\n"
        "     [ 0.5    1.0 ]         [ 2 ]          [ -0.5 ]\n\n"
        "z₁ = [ 0.5,  2.0 ]ᵀ",
        fontsize=15
    )

def frame_2():
    setup_canvas()

    title(
        "Forward pass",
        r"Apply the activation:  $\mathbf{h}=\mathrm{ReLU}(\mathbf{z}_1)$"
    )

    draw_base_network(show_hidden=True)

    ax.text(
        6.3,
        7.0,
        r"$\mathbf{z}_1=[0.5,\;2]^\top$",
        ha="center",
        fontsize=19
    )

    box(
        4.6, 1.0, 3.5, 1.05,
        r"$\mathbf{h}=\mathrm{ReLU}(\mathbf{z}_1)"
        r"=[0.5,\;2]^\top$",
        fontsize=18
    )

    ax.text(
        10.7,
        1.5,
        "Saved for backward",
        fontsize=16,
        color=GREEN,
        fontweight="bold"
    )

    ax.text(
        10.7,
        1.05,
        r"$\mathbf{x},\;\mathbf{z}_1,\;\mathbf{h}$",
        fontsize=19,
        color=GREEN
    )


def frame_3():
    setup_canvas()

    title(
        "Forward pass",
        r"Compute the prediction:  $\hat{y}=W_2\mathbf{h}+b_2$"
    )

    draw_base_network(
        show_hidden=True,
        show_output=True
    )

    # emphasize hidden -> output
    arrow(pos["h1"], pos["yhat"], BLUE, 3)
    arrow(pos["h2"], pos["yhat"], BLUE, 3)

    ax.text(
        8.2,
        6.6,
        r"$W_2=[1,\;1],\quad b_2=0$",
        ha="center",
        fontsize=19
    )

    box(
        7.8, 1.0, 4.0, 1.05,
        r"$\hat{y}=1(0.5)+1(2)=2.5$",
        fontsize=19
    )


def frame_4():
    setup_canvas()

    title(
        "Forward pass",
        "Compare the prediction with the target"
    )

    draw_base_network(
        show_hidden=True,
        show_output=True,
        show_loss=True
    )

    ax.text(
        13.5,
        5.5,
        r"$y=1.5$",
        ha="center",
        fontsize=20
    )

    box(
        10.5, 1.0, 4.0, 1.05,
        r"$J=\frac{1}{2}(\hat{y}-y)^2"
        r"=\frac{1}{2}(2.5-1.5)^2=0.5$",
        fontsize=18
    )


def frame_5():
    setup_canvas()

    title(
        "Backward pass",
        "Start from the loss and propagate the error backward"
    )

    draw_base_network(
        show_hidden=True,
        show_output=True,
        show_loss=True
    )

    # backward arrow: loss -> output
    arrow(
        pos["loss"],
        pos["yhat"],
        RED,
        4
    )

    box(
        10.3, 1.0, 4.0, 1.15,
        r"$\delta_2="
        r"\frac{\partial J}{\partial \hat{y}}"
        r"=\hat{y}-y=1$",
        edgecolor=RED,
        facecolor=LIGHT_RED,
        fontsize=19
    )


def frame_6():
    setup_canvas()

    title(
        "Backward pass",
        r"Use the saved hidden activation to compute $\nabla_{W_2}J$"
    )

    draw_base_network(
        show_hidden=True,
        show_output=True,
        show_loss=True
    )

    arrow(pos["loss"], pos["yhat"], RED, 4)

    arrow(pos["yhat"], pos["h1"], RED, 4)
    arrow(pos["yhat"], pos["h2"], RED, 4)

    # saved h
    box(
        4.5, 1.0, 3.5, 1.0,
        r"Saved forward value:"
        "\n"
        r"$\mathbf{h}=[0.5,\;2]^\top$",
        edgecolor=GREEN,
        facecolor=LIGHT_GREEN,
        fontsize=16
    )

    box(
        8.4, 1.0, 4.6, 1.0,
        r"$\nabla_{W_2}J"
        r"=\delta_2\mathbf{h}^\top"
        r"=[0.5,\;2]$",
        edgecolor=RED,
        facecolor=LIGHT_RED,
        fontsize=18
    )


def frame_7():
    setup_canvas()

    title(
        "Backward pass",
        "Propagate the same error signal into the hidden layer"
    )

    draw_base_network(
        show_hidden=True,
        show_output=True,
        show_loss=True
    )

    arrow(pos["loss"], pos["yhat"], RED, 3.5)
    arrow(pos["yhat"], pos["h1"], RED, 4)
    arrow(pos["yhat"], pos["h2"], RED, 4)

    box(
        7.0, 1.0, 5.0, 1.2,
        "∂J/∂h = W₂ᵀ δ₂\n"
        "= [ 1,  1 ]ᵀ",
        edgecolor=RED,
        facecolor=LIGHT_RED,
        fontsize=18
    )

def frame_8():
    setup_canvas()

    title(
        "Backward pass",
        "Pass through the activation derivative"
    )

    draw_base_network(
        show_hidden=True,
        show_output=True,
        show_loss=True
    )

    # Backward arrows
    arrow(pos["loss"], pos["yhat"], RED, 3)
    arrow(pos["yhat"], pos["h1"], RED, 4)
    arrow(pos["yhat"], pos["h2"], RED, 4)

    # --------------------------------------------------------
    # Saved forward value
    # --------------------------------------------------------
    box(
        3.3, 0.65, 4.2, 1.45,
        "Saved forward value:\n\n"
        r"$\mathbf{z}_1=[0.5,\;2]^\top$"
        "\n"
        r"$\mathrm{ReLU}'(\mathbf{z}_1)=[1,\;1]^\top$",
        edgecolor=GREEN,
        facecolor=LIGHT_GREEN,
        fontsize=15
    )

    # --------------------------------------------------------
    # Backpropagation through ReLU
    # --------------------------------------------------------
    box(
        8.0, 0.65, 5.5, 1.45,
        r"$\delta_1="
        r"\frac{\partial J}{\partial \mathbf{h}}"
        r"\odot"
        r"\mathrm{ReLU}'(\mathbf{z}_1)$"
        "\n\n"
        r"$=[1,\;1]^\top$",
        edgecolor=RED,
        facecolor=LIGHT_RED,
        fontsize=16
    )


def frame_9():
    setup_canvas()

    title(
        "Backward pass",
        r"Reuse the saved input to compute $\nabla_{W_1}J$"
    )

    draw_base_network(
        show_hidden=True,
        show_output=True,
        show_loss=True
    )

    # full red backward path
    arrow(pos["loss"], pos["yhat"], RED, 3)

    arrow(pos["yhat"], pos["h1"], RED, 3)
    arrow(pos["yhat"], pos["h2"], RED, 3)

    arrow(pos["h1"], pos["x1"], RED, 3)
    arrow(pos["h1"], pos["x2"], RED, 3)
    arrow(pos["h2"], pos["x1"], RED, 3)
    arrow(pos["h2"], pos["x2"], RED, 3)

    box(
        0.7, 0.7, 3.5, 1.0,
        r"Saved input:"
        "\n"
        r"$\mathbf{x}=[1,\;2]^\top$",
        edgecolor=GREEN,
        facecolor=LIGHT_GREEN,
        fontsize=16
    )

    box(
        5.0, 0.35, 5.6, 1.65,
        "∇W₁ J = δ₁ xᵀ\n\n"
        "= [ 1   2 ]\n"
        "  [ 1   2 ]",
        edgecolor=RED,
        facecolor=LIGHT_RED,
        fontsize=18
    )


def frame_10():
    setup_canvas()

    title(
        "Backpropagation",
        "One backward pass computes gradients for all parameters"
    )

    # draw complete network
    draw_base_network(
        show_hidden=True,
        show_output=True,
        show_loss=True
    )

    # forward path label
    ax.text(
        7.4,
        7.0,
        r"Forward: compute and save "
        r"$\mathbf{x},\mathbf{z}_1,\mathbf{h},\hat{y},J$",
        ha="center",
        fontsize=19,
        color=BLUE
    )

    # backward arrows
    arrow(pos["loss"], pos["yhat"], RED, 3)

    arrow(pos["yhat"], pos["h1"], RED, 3)
    arrow(pos["yhat"], pos["h2"], RED, 3)

    arrow(pos["h1"], pos["x1"], RED, 3)
    arrow(pos["h1"], pos["x2"], RED, 3)
    arrow(pos["h2"], pos["x1"], RED, 3)
    arrow(pos["h2"], pos["x2"], RED, 3)

    ax.text(
        7.4,
        1.75,
        "Backward: propagate one error signal and reuse saved forward values",
        ha="center",
        fontsize=19,
        color=RED,
        fontweight="bold"
    )

    box(
        4.1,
        0.35,
        6.8,
        0.9,
        "Backpropagation efficiently applies the chain rule\n"
        "to compute gradients for all parameters.",
        edgecolor=BLUE,
        facecolor=LIGHT_BLUE,
        fontsize=17
    )


# ============================================================
# 6. List all animation frames
# ============================================================

frames = [
    frame_0,
    frame_1,
    frame_2,
    frame_3,
    frame_4,
    frame_5,
    frame_6,
    frame_7,
    frame_8,
    frame_9,
    frame_10,
]


# ============================================================
# 7. Render frames manually
# ============================================================

from PIL import Image
import os

frame_dir = "backprop_frames"
os.makedirs(frame_dir, exist_ok=True)

frame_files = []

for i, draw_frame in enumerate(frames):

    draw_frame()

    filename = os.path.join(
        frame_dir,
        f"frame_{i:02d}.png"
    )

    fig.savefig(
        filename,
        dpi=120,
        bbox_inches="tight",
        facecolor="white"
    )

    frame_files.append(filename)

    print(f"Rendered frame {i + 1}/{len(frames)}")


# ============================================================
# 8. Build GIF
# ============================================================

images = [
    Image.open(filename).convert("RGB")
    for filename in frame_files
]

durations = [
    2500,
    3000,
    3000,
    3000,
    3000,
    3000,
    3500,
    3500,
    3500,
    4000,
    5000,
]

images[0].save(
    "backpropagation.gif",
    save_all=True,
    append_images=images[1:],
    duration=durations,
    loop=0
)

plt.close(fig)

print("\nSaved: backpropagation.gif")