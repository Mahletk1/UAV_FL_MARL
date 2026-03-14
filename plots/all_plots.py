import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # SimCode/
RESULTS_DIR = os.path.join(ROOT, "results")

# ============================================================
# Fixed config
# ============================================================
DATA_MODE = "dirichlet"
ENV_TAG = "highrise"
WIRELESS_ON = True
MAIN_K = 10

METHODS = ("rs", "rr", "pf", "bc", "marl_full")
LABELS = ("RS", "RR", "PF", "BC", "Proposed")

METHOD_COLORS = {
    "RS": "#7f7f7f",
    "RR": "#1f77b4",
    "PF": "#2ca02c",
    "BC": "#ff7f0e",
    "Proposed": "#9467bd",
}

ALTITUDE_METHOD = "marl_full"
UAVS_TO_PLOT = [0, 3, 2, 11, 19]

# ============================================================
# Helpers
# ============================================================
def load_log(path: str) -> dict:
    return np.load(path, allow_pickle=True).item()


def find_runs(method: str, exp_tag="main", env_tag=None, k_value=None):
    wtag = "True" if WIRELESS_ON else "False"
    env_part = "*" if env_tag is None else env_tag
    k_part = "*" if k_value is None else f"K{k_value}"

    pattern = os.path.join(
        RESULTS_DIR,
        exp_tag,
        f"{method}_{DATA_MODE}_{env_part}_{k_part}_wireless{wtag}_seed*.npy"
    )
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No logs found for pattern:\n{pattern}")
    return paths


def first_existing_key(d: dict, candidates: list[str], what: str) -> str:
    for k in candidates:
        if k in d:
            return k
    raise KeyError(
        f"Could not find a key for '{what}'.\n"
        f"Tried: {candidates}\n"
        f"Available keys:\n{list(d.keys())}"
    )


def stack_metric(logs: list[dict], key_candidates: list[str], what: str) -> np.ndarray:
    arrs = []
    for i, log in enumerate(logs):
        key = first_existing_key(log, key_candidates, what)
        arr = np.asarray(log[key], dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(
                f"Expected 1D array for '{what}', but run #{i} key '{key}' has shape {arr.shape}"
            )
        arrs.append(arr)

    T = min(a.shape[0] for a in arrs)
    arrs = [a[:T] for a in arrs]
    return np.stack(arrs, axis=0)


def mean_std(Y: np.ndarray):
    return Y.mean(axis=0), Y.std(axis=0)


def moving_average(x: np.ndarray, w: int = 5) -> np.ndarray:
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def plot_mean_std(ax, Y: np.ndarray, label: str, smooth_w: int = 1):
    mu, sd = mean_std(Y)
    mu_s = moving_average(mu, smooth_w)
    sd_s = moving_average(sd, smooth_w)
    x = np.arange(len(mu_s))
    color = METHOD_COLORS[label]

    ax.plot(x, mu_s, label=label, linewidth=2, color=color)
    ax.fill_between(x, mu_s - sd_s, mu_s + sd_s, alpha=0.18, color=color)


def try_get_per_uav_counts(log: dict) -> np.ndarray:
    direct_keys = [
        "selection_counts",
        "select_counts",
        "sel_counts",
        "uav_select_counts",
        "per_uav_selection_count",
    ]
    for k in direct_keys:
        if k in log:
            arr = np.asarray(log[k], dtype=np.float32)
            if arr.ndim == 1:
                return arr

    hist_keys = [
        "selected_mask",
        "selected_history",
        "selection_history",
        "selected_mask_history",
        "sel_history",
        "selected_masks",
    ]
    for k in hist_keys:
        if k in log:
            arr = np.asarray(log[k], dtype=np.float32)
            if arr.ndim == 2:
                return arr.sum(axis=0)

    raise KeyError(
        "Could not build per-UAV selection counts.\n"
        f"Available keys:\n{list(log.keys())}"
    )


def stack_per_uav_counts(logs: list[dict]) -> np.ndarray:
    arrs = [try_get_per_uav_counts(log) for log in logs]
    N = min(a.shape[0] for a in arrs)
    arrs = [a[:N] for a in arrs]
    return np.stack(arrs, axis=0)


# ============================================================
# Fig. 2 Accuracy vs rounds
# ============================================================
def make_fig2_accuracy():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    acc_keys = ["test_acc", "global_test_acc", "accuracy", "acc"]

    for method, label in zip(METHODS, LABELS):
        paths = find_runs(method, exp_tag="main", env_tag=ENV_TAG, k_value=MAIN_K)
        logs = [load_log(p) for p in paths]
        Y = stack_metric(logs, acc_keys, "test accuracy")
        plot_mean_std(ax, Y, label, smooth_w=1)
        print(f"[Fig2][{label}] runs: {len(paths)}")

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Global Test Accuracy (%)")
    ax.set_title(f"Fig. 2. Global Test Accuracy vs FL Rounds (K={MAIN_K})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3)

    out_path = os.path.join(RESULTS_DIR, f"fig2_acc_K{MAIN_K}_{ENV_TAG}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# Fig. 3 Upload success rate vs FL rounds
# ============================================================
def make_fig3_success():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    success_keys = [
        "success_rate",
        "sel_success",
        "selected_success",
        "selected_upload_success",
        "upload_success",
        "mean_sel_success",
    ]

    methods = METHODS
    labels = LABELS

    for method, label in zip(methods, labels):
        paths = find_runs(method, exp_tag="main", env_tag=ENV_TAG, k_value=MAIN_K)
        logs = [load_log(p) for p in paths]

        Y = stack_metric(logs, success_keys, "selected UAV upload success")

        plot_mean_std(ax, Y, label)

        print(f"[Fig3][{label}] runs: {len(paths)}")

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Successful Upload Ratio")

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=3)
    ax.set_ylim(0.0, 1.0)

    out_path = os.path.join(RESULTS_DIR, f"fig3_success_K{MAIN_K}_{ENV_TAG}.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
# ============================================================
# Fig. 4 Selection frequency per UAV
# ============================================================
def make_fig4_selection_hist():
    n_methods = len(METHODS)
    fig, axes = plt.subplots(
        nrows=n_methods, ncols=1, figsize=(8.0, 2.3 * n_methods), sharex=True
    )

    if n_methods == 1:
        axes = [axes]

    for ax, method, label in zip(axes, METHODS, LABELS):
        paths = find_runs(method, exp_tag="main", env_tag=ENV_TAG, k_value=MAIN_K)
        logs = [load_log(p) for p in paths]

        C = stack_per_uav_counts(logs)  # [S, N]
        mu = C.mean(axis=0)
        sd = C.std(axis=0)
        x = np.arange(len(mu))

        ax.bar(x, mu, alpha=0.9, color=METHOD_COLORS[label])
        ax.errorbar(x, mu, yerr=sd, fmt="none", capsize=2, linewidth=1, color="black")

        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.25)

        print(f"[Fig4][{label}] runs: {len(paths)}, UAVs: {len(mu)}")

    axes[-1].set_xlabel("UAV Index")
    fig.suptitle(f"Fig. 4. Per-UAV Selection Counts (K={MAIN_K})", y=0.995)

    out_path = os.path.join(RESULTS_DIR, f"fig4_selection_hist_K{MAIN_K}_{ENV_TAG}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Fig. 5 3D UAV trajectories
# ============================================================
def make_fig5_3d_trajectory():
    paths = find_runs(ALTITUDE_METHOD, exp_tag="main", env_tag=ENV_TAG, k_value=MAIN_K)
    logs = [load_log(p) for p in paths]
    log = logs[0]

    X = np.array(log["x_uav"])   # [T, N]
    Y = np.array(log["y_uav"])
    H = np.array(log["h_uav"])

    fig = plt.figure(figsize=(7.2, 5))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10(np.arange(len(UAVS_TO_PLOT)))

    for k, j in enumerate(UAVS_TO_PLOT):
        ax.plot(
            X[:, j],
            Y[:, j],
            H[:, j],
            linewidth=2,
            color=colors[k],
            label=f"UAV {k}"
        )

        ax.scatter(
            X[0, j], Y[0, j], H[0, j],
            color="#2ca02c",
            marker="o",
            s=45,
            edgecolor="black",
            linewidth=0.5
        )

        ax.scatter(
            X[-1, j], Y[-1, j], H[-1, j],
            color="#d62728",
            marker="x",
            s=55,
            linewidth=2
        )

    ax.scatter(
        0, 0, 25,
        color="black",
        marker="^",
        s=120,
        label="Base Station"
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title(f"Fig. 5. 3D UAV Trajectories (K={MAIN_K})")

    ax.set_xlim(-650, 650)
    ax.set_ylim(-650, 650)
    ax.set_zlim(0, 520)
    ax.view_init(elev=22, azim=-55)

    legend_markers = [
        Line2D([0], [0], marker='o', color="#2ca02c", markeredgecolor="black",
               linestyle='None', label='Start', markersize=6),
        Line2D([0], [0], marker='x', color="#d62728",
               linestyle='None', label='End', markersize=7),
    ]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_markers, labels + ["Start", "End"], fontsize=8)

    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, f"fig5_3d_traj_K{MAIN_K}_{ENV_TAG}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Fig. 6 Proposed only, different K
# ============================================================
def make_fig6_k_sweep():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    acc_keys = ["test_acc", "global_test_acc", "accuracy", "acc"]

    k_values = [2, 4, 6, 8, 10]
    final_mu = []
    final_sd = []

    for k in k_values:
        paths = find_runs("marl_full", exp_tag="k_sweep", env_tag=ENV_TAG, k_value=k)
        logs = [load_log(p) for p in paths]
        Y = stack_metric(logs, acc_keys, f"test accuracy K={k}")
        mu, sd = mean_std(Y)
        final_mu.append(mu[-1])
        final_sd.append(sd[-1])
        print(f"[Fig6][K={k}] runs: {len(paths)} final acc: {mu[-1]:.2f}")

    x = np.arange(len(k_values))
    ax.plot(x, final_mu, marker="o", linewidth=2, color=METHOD_COLORS["Proposed"], label="Proposed")
    ax.fill_between(
        x,
        np.array(final_mu) - np.array(final_sd),
        np.array(final_mu) + np.array(final_sd),
        alpha=0.18,
        color=METHOD_COLORS["Proposed"]
    )

    ax.set_xticks(x)
    ax.set_xticklabels(k_values)
    ax.set_xlabel("Number of Selected UAVs (K)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Fig. 6. Accuracy under Different Numbers of Selected UAVs")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    out_path = os.path.join(RESULTS_DIR, "fig6_k_sweep_proposed.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Fig. 7 Environment sweep
# ============================================================
def make_fig7_env_sweep():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    acc_keys = ["test_acc", "global_test_acc", "accuracy", "acc"]

    envs = ["suburban", "urban", "denseurban", "highrise"]
    methods = ("pf", "bc", "marl_full")
    labels = ("PF", "BC", "Proposed")

    x = np.arange(len(envs))
    width = 0.22

    for i, (method, label) in enumerate(zip(methods, labels)):
        final_mu = []
        final_sd = []

        for env in envs:
            paths = find_runs(method, exp_tag="env_sweep", env_tag=env, k_value=MAIN_K)
            logs = [load_log(p) for p in paths]
            Y = stack_metric(logs, acc_keys, f"{label} in {env}")
            mu, sd = mean_std(Y)
            final_mu.append(mu[-1])
            final_sd.append(sd[-1])

        ax.bar(x + i * width - width, final_mu, width=width,
               label=label, alpha=0.9, color=METHOD_COLORS[label])
        ax.errorbar(x + i * width - width, final_mu, yerr=final_sd,
                    fmt="none", capsize=2, linewidth=1, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_xlabel("Environment")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title(f"Fig. 7. Accuracy under Different Environments (K={MAIN_K})")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    out_path = os.path.join(RESULTS_DIR, "fig7_env_sweep.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    # make_fig2_accuracy()
    make_fig3_success()
    # make_fig4_selection_hist()
    # make_fig5_3d_trajectory()
    # make_fig6_k_sweep()
    # make_fig7_env_sweep()


if __name__ == "__main__":
    main()