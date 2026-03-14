import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, "results")

DATA_MODE = "dirichlet"
ENV_TAG = "highrise"
WIRELESS_ON = True

METHODS = ("rs", "rr", "pf", "bc", "marl_full")
LABELS = ("RS", "RR", "PF", "BC", "Proposed")

METHOD_COLORS = {
    "RS": "#7f7f7f",
    "RR": "#1f77b4",
    "PF": "#2ca02c",
    "BC": "#ff7f0e",
    "Proposed": "#9467bd",
}

UAVS_TO_PLOT = [0, 3, 2, 11, 19]
FIXED_K = 10   # used for Fig. 4, Fig. 5, Fig. 6, Fig. 7
K_COMPARE_VALUES = [4, 7, 10]

def load_log(path: str) -> dict:
    return np.load(path, allow_pickle=True).item()

def find_runs(method: str, exp_tag="k_compare", env_tag=None, k_value=None):
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

def plot_mean_std(ax, Y: np.ndarray, label: str):
    mu, sd = mean_std(Y)
    x = np.arange(len(mu))
    color = METHOD_COLORS[label]
    ax.plot(x, mu, label=label, linewidth=2, color=color)
    ax.fill_between(x, mu - sd, mu + sd, alpha=0.12, color=color)

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

    raise KeyError(f"Could not build per-UAV selection counts.\nAvailable keys:\n{list(log.keys())}")

def stack_per_uav_counts(logs: list[dict]) -> np.ndarray:
    arrs = [try_get_per_uav_counts(log) for log in logs]
    N = min(a.shape[0] for a in arrs)
    arrs = [a[:N] for a in arrs]
    return np.stack(arrs, axis=0)

def moving_average(x, w=3):
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")
# ============================================================
# Fig. 2 Proposed method under different K values
# ============================================================
def make_fig2_proposed_k_accuracy():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    acc_keys = ["test_acc", "global_test_acc", "accuracy", "acc"]

    for k in K_COMPARE_VALUES:
        paths = find_runs("marl_full", exp_tag="k_compare", env_tag=ENV_TAG, k_value=k)
        logs = [load_log(p) for p in paths]
        Y = stack_metric(logs, acc_keys, f"accuracy K={k}")

        mu, sd = mean_std(Y)
        x = np.arange(len(mu))
        ax.plot(x, mu, linewidth=2.5, label=f"K={k}")
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.12)

        print(f"[Fig2][K={k}] runs: {len(paths)}")

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Global Test Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    out_path = os.path.join(RESULTS_DIR, "fig2_proposed_k_accuracy.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# Fig. 3 Method comparison under K = 4, 7, 10
# Saved as 3 separate plots
# ============================================================
def make_fig3_k_method_comparison():
    acc_keys = ["test_acc", "global_test_acc", "accuracy", "acc"]

    panel_tags = {
        4: "(a) K=4",
        7: "(b) K=7",
        10: "(c) K=10",
    }

    for k in K_COMPARE_VALUES:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))

        for method, label in zip(METHODS, LABELS):
            paths = find_runs(method, exp_tag="k_compare", env_tag=ENV_TAG, k_value=k)
            logs = [load_log(p) for p in paths]
            Y = stack_metric(logs, acc_keys, f"accuracy K={k}")

            mu, sd = mean_std(Y)

            # optional small smoothing for mean only
            mu_s = moving_average(mu, w=2)

            x = np.arange(len(mu_s))

            ax.plot(x, mu_s, label=label, linewidth=2.5, color=METHOD_COLORS[label])
            ax.fill_between(
                x,
                mu_s - sd,
                mu_s + sd,
                alpha=0.12,
                color=METHOD_COLORS[label]
            )

            print(f"{label}, K={k}, num runs = {Y.shape[0]}, std final = {sd[-1]:.4f}")

        ax.set_xlabel("FL Round")
        ax.set_ylabel("Global Test Accuracy (%)")
        ax.grid(True, alpha=0.12)
        ax.legend(fontsize=8, ncol=2)

        ax.set_ylim(0, 1)
        
        # put (a) K=... under the plot
        ax.text(
            0.5, -0.18, panel_tags[k],
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=11
        )

        out_path = os.path.join(RESULTS_DIR, f"fig3_method_comparison_K{k}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"[Fig3][K={k}] saved to {out_path}")

# ============================================================
# Fig. 4 Upload success rate vs FL rounds at fixed K
# ============================================================
def make_fig4_success():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    success_keys = [
        "success_rate",
        "sel_success",
        "selected_success",
        "selected_upload_success",
        "upload_success",
        "mean_sel_success",
    ]

    for method, label in zip(METHODS, LABELS):
        paths = find_runs(method, exp_tag="k_compare", env_tag=ENV_TAG, k_value=FIXED_K)
        logs = [load_log(p) for p in paths]
        Y = stack_metric(logs, success_keys, "selected UAV upload success")
        plot_mean_std(ax, Y, label)
        print(f"[Fig4][{label}] runs: {len(paths)}")

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Successful Upload Ratio")
    ax.grid(True, alpha=0.12)
    ax.legend(fontsize=8, ncol=3)
    # ax.set_ylim(0.0, 1.0)

    out_path = os.path.join(RESULTS_DIR, f"fig4_success_K{FIXED_K}_{ENV_TAG}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# Fig. 5 Selection frequency per UAV at fixed K
# PF, BC, Proposed only
# ============================================================
def make_fig5_selection_hist():
    methods = ("pf", "bc", "marl_full")
    labels = ("PF", "BC", "Proposed")

    n_methods = len(methods)
    fig, axes = plt.subplots(
        nrows=n_methods, ncols=1, figsize=(8.0, 2.3 * n_methods), sharex=True
    )

    if n_methods == 1:
        axes = [axes]

    for ax, method, label in zip(axes, methods, labels):
        paths = find_runs(method, exp_tag="k_compare", env_tag=ENV_TAG, k_value=FIXED_K)
        logs = [load_log(p) for p in paths]

        C = stack_per_uav_counts(logs)
        mu = C.mean(axis=0)
        sd = C.std(axis=0)
        x = np.arange(len(mu))

        ax.bar(x, mu, alpha=0.9, color=METHOD_COLORS[label])
        ax.errorbar(x, mu, yerr=sd, fmt="none", capsize=2, linewidth=1, color="black")

        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.12)
        ax.text(0.01, 0.88, label, transform=ax.transAxes, fontsize=10, weight="bold")

        print(f"[Fig5][{label}] runs: {len(paths)}, UAVs: {len(mu)}")

    axes[-1].set_xlabel("UAV Index")

    out_path = os.path.join(RESULTS_DIR, f"fig5_selection_hist_K{FIXED_K}_{ENV_TAG}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# Fig. 6 3D UAV altitude trajectories at fixed K
# ============================================================
def make_fig6_3d_trajectory():
    paths = find_runs("marl_full", exp_tag="k_compare", env_tag=ENV_TAG, k_value=FIXED_K)
    logs = [load_log(p) for p in paths]
    log = logs[0]

    X = np.array(log["x_uav"])
    Y = np.array(log["y_uav"])
    H = np.array(log["h_uav"])

    fig = plt.figure(figsize=(7.2, 5))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10(np.arange(len(UAVS_TO_PLOT)))

    for k, j in enumerate(UAVS_TO_PLOT):
        ax.plot(X[:, j], Y[:, j], H[:, j], linewidth=2, color=colors[k], label=f"UAV {k}")
        ax.scatter(X[0, j], Y[0, j], H[0, j], color="#2ca02c", marker="o", s=45,
                   edgecolor="black", linewidth=0.5)
        ax.scatter(X[-1, j], Y[-1, j], H[-1, j], color="#d62728", marker="x", s=55, linewidth=2)

    ax.scatter(0, 0, 25, color="black", marker="^", s=120, label="Base Station")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude (m)")
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

    out_path = os.path.join(RESULTS_DIR, f"fig6_3d_traj_K{FIXED_K}_{ENV_TAG}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# Fig. 7 Accuracy under different environments
# PF, BC, Proposed at K = 10
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
            paths = find_runs(method, exp_tag="env_sweep", env_tag=env, k_value=10)
            logs = [load_log(p) for p in paths]
            Y = stack_metric(logs, acc_keys, f"{label} in {env}")
            mu, sd = mean_std(Y)
            final_mu.append(mu[-1])
            final_sd.append(sd[-1])

        ax.bar(x + i * width - width, final_mu, width=width,
               label=label, alpha=0.12, color=METHOD_COLORS[label])
        ax.errorbar(x + i * width - width, final_mu, yerr=final_sd,
                    fmt="none", capsize=2, linewidth=1, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.set_xlabel("Environment")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.grid(True, axis="y", alpha=0.12)
    ax.legend(fontsize=8)

    out_path = os.path.join(RESULTS_DIR, "fig7_env_sweep.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

def main():
    make_fig2_proposed_k_accuracy()
    make_fig3_k_method_comparison()
    make_fig4_success()
    make_fig5_selection_hist()
    # make_fig6_3d_trajectory()
    # make_fig7_env_sweep()

if __name__ == "__main__":
    main()