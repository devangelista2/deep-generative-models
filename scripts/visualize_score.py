import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================
# Density: mixture of two isotropic Gaussians
# ============================================================

def gaussian_density(x, mean, sigma):
    diff = x - mean
    exponent = -0.5 * np.sum(diff**2, axis=-1) / sigma**2
    norm = 1.0 / (2.0 * np.pi * sigma**2)
    return norm * np.exp(exponent)


def mixture_density(x, means, sigmas, weights):
    p = np.zeros(x.shape[:-1])

    for w, m, s in zip(weights, means, sigmas):
        p += w * gaussian_density(x, m, s)

    return p


def score_function(x, means, sigmas, weights):
    """
    Score function:

        s(x) = grad_x log p(x)

    For a Gaussian mixture, this is a responsibility-weighted
    average of the score of each Gaussian component.
    """
    numerator = np.zeros_like(x)
    denominator = np.zeros(x.shape[:-1])

    for w, m, s in zip(weights, means, sigmas):
        p_k = w * gaussian_density(x, m, s)
        grad_log_p_k = -(x - m) / s**2

        numerator += p_k[..., None] * grad_log_p_k
        denominator += p_k

    return numerator / (denominator[..., None] + 1e-12)


# ============================================================
# Static plot
# ============================================================

def plot_score_field(
    filename="score_field.png",
    xlim=(-4, 4),
    ylim=(-4, 4),
    means=None,
    sigmas=None,
    weights=None,
):
    if means is None:
        means = np.array([[-2.0, -2.0], [2.0, 2.0]])

    if sigmas is None:
        sigmas = np.array([1.25, 0.5])

    if weights is None:
        weights = np.array([0.7, 0.2])

    xs = np.linspace(xlim[0], xlim[1], 450)
    ys = np.linspace(ylim[0], ylim[1], 450)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X, Y], axis=-1)

    p = mixture_density(grid, means, sigmas, weights)
    logp = np.log(p + 1e-12)

    qx = np.linspace(xlim[0], xlim[1], 17)
    qy = np.linspace(ylim[0], ylim[1], 17)
    QX, QY = np.meshgrid(qx, qy)
    qgrid = np.stack([QX, QY], axis=-1)

    S = score_function(qgrid, means, sigmas, weights)
    U = S[..., 0]
    V = S[..., 1]

    magnitude = np.sqrt(U**2 + V**2)
    U_plot = U / (magnitude + 1e-8)
    V_plot = V / (magnitude + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Make the axes fill the full figure, removing the border.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])

    ax.contour(
        X,
        Y,
        logp,
        levels=40,
        cmap="Blues",
        linewidths=1.8,
        alpha=0.75,
    )

    ax.quiver(
        QX,
        QY,
        U_plot,
        V_plot,
        angles="xy",
        scale_units="xy",
        scale=2.8,
        width=0.004,
        color="black",
        alpha=0.9,
    )

    ax.scatter(
        means[:, 0],
        means[:, 1],
        s=25,
        color="black",
        zorder=5,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.savefig(
        filename,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


# ============================================================
# GIF animation
# ============================================================

def make_score_trajectory_gif(
    filename="score_trajectory.gif",
    xlim=(-4, 4),
    ylim=(-4, 4),
    means=None,
    sigmas=None,
    weights=None,
    n_particles=500,
    n_steps=100,
    step_size=0.055,
    noise_scale=0.13,
    seed=7,
):
    if means is None:
        means = np.array([[-2.0, -2.0], [2.0, 2.0]])

    if sigmas is None:
        sigmas = np.array([1.25, 0.5])

    if weights is None:
        weights = np.array([0.7, 0.2])

    rng = np.random.default_rng(seed)

    particles = rng.uniform(
        low=[xlim[0], ylim[0]],
        high=[xlim[1], ylim[1]],
        size=(n_particles, 2),
    )

    trajectory = [particles.copy()]

    for _ in range(n_steps):
        score = score_function(particles, means, sigmas, weights)

        # Movement along the score, toward high-density regions.
        particles = particles + step_size * score

        # Noisy perturbation, similar to a Langevin-style update.
        particles = particles + np.sqrt(2.0 * step_size) * noise_scale * rng.normal(
            size=particles.shape
        )

        particles[:, 0] = np.clip(particles[:, 0], xlim[0], xlim[1])
        particles[:, 1] = np.clip(particles[:, 1], ylim[0], ylim[1])

        trajectory.append(particles.copy())

    xs = np.linspace(xlim[0], xlim[1], 450)
    ys = np.linspace(ylim[0], ylim[1], 450)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X, Y], axis=-1)

    p = mixture_density(grid, means, sigmas, weights)
    logp = np.log(p + 1e-12)

    fig, ax = plt.subplots(figsize=(8, 8))

    # This is the important part for removing the white border in the GIF.
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])

    ax.contour(
        X,
        Y,
        logp,
        levels=40,
        cmap="Blues",
        linewidths=1.6,
        alpha=0.65,
    )

    ax.scatter(
        means[:, 0],
        means[:, 1],
        s=35,
        color="black",
        zorder=5,
    )

    scat = ax.scatter(
        trajectory[0][:, 0],
        trajectory[0][:, 1],
        s=16,
        color="black",
        alpha=0.75,
        zorder=10,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.axis("off")

    title = ax.text(
        0.03,
        0.96,
        "",
        transform=ax.transAxes,
        fontsize=13,
        ha="left",
        va="top",
        color="black",
    )

    def update(frame):
        scat.set_offsets(trajectory[frame])
        title.set_text(f"Step {frame}")
        return scat, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(trajectory),
        interval=60,
        blit=True,
    )

    animation.save(
        filename,
        writer=PillowWriter(fps=18),
        savefig_kwargs={
            "bbox_inches": "tight",
            "pad_inches": 0,
            "facecolor": fig.get_facecolor(),
        },
    )

    plt.close(fig)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    means = np.array([
        [-2.0, -2.0],
        [2.0, 2.0],
    ])

    # The first attractor is broader and heavier.
    # The second attractor is smaller and sharper.
    sigmas = np.array([1.25, 0.5])
    weights = np.array([0.7, 0.2])

    plot_score_field(
        filename="score_field.png",
        means=means,
        sigmas=sigmas,
        weights=weights,
    )

    make_score_trajectory_gif(
        filename="score_trajectory.gif",
        means=means,
        sigmas=sigmas,
        weights=weights,
        n_particles=500,
        n_steps=100,
        step_size=0.055,
        noise_scale=0.13,
    )

    print("Saved score_field.png")
    print("Saved score_trajectory.gif")