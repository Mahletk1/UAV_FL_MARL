import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.2,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "-",
    "savefig.dpi": 300,
})

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

LINESTYLES = {
    "RS": "--",
    "RR": "--",
    "PF": ":",
    "BC": "-.",
    "Proposed": "-",
}

LINEWIDTHS = {
    "RS": 1.8,
    "RR": 1.8,
    "PF": 2.0,
    "BC": 2.1,
    "Proposed": 2.8,
}

ALPHAS = {
    "RS": 0.95,
    "RR": 0.95,
    "PF": 0.95,
    "BC": 0.98,
    "Proposed": 1.0,
}

ABLATION_METHODS = ("bc", "marl_altitude_only", "marl_selection_only", "marl_full")
ABLATION_LABELS = ("BC", "MARL-AltOnly", "MARL-SelOnly", "Full MARL")

ABLATION_COLORS = {
    "BC": "#ff7f0e",
    "MARL-AltOnly": "#1f77b4",
    "MARL-SelOnly": "#2ca02c",
    "Full MARL": "#9467bd",
}

ABLATION_LINESTYLES = {
    "BC": "-.",
    "MARL-AltOnly": "--",
    "MARL-SelOnly": ":",
    "Full MARL": "-",
}

ABLATION_LINEWIDTHS = {
    "BC": 2.1,
    "MARL-AltOnly": 2.0,
    "MARL-SelOnly": 2.0,
    "Full MARL": 2.8,
}

UAVS_TO_PLOT = [2, 11, 19]
FIXED_K = 10
K_COMPARE_VALUES = [4, 7, 10]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_log(path: str) -> dict:
    return np.load(path, allow_pickle=True).item()

def find_runs(method: str, exp_tag="k_compare", env_tag=None, k_value=None, verbose=False):
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

    # Deduplicate by seed number, keep newest if duplicates exist
    by_seed = {}
    for p in paths:
        m = re.search(r"_seed(\d+)\.npy$", os.path.basename(p))
        if m is None:
            continue
        seed = int(m.group(1))
        if seed not in by_seed:
            by_seed[seed] = p
        else:
            if os.path.getmtime(p) > os.path.getmtime(by_seed[seed]):
                by_seed[seed] = p

    unique_paths = [by_seed[s] for s in sorted(by_seed.keys())]

    if verbose:
        print("\n[find_runs]")
        print("RESULTS_DIR =", RESULTS_DIR)
        print("pattern     =", pattern)
        print("matched raw =", len(paths))
        for p in paths:
            print("  raw:", os.path.basename(p))
        print("unique used =", len(unique_paths))
        for p in unique_paths:
            print("  use:", os.path.basename(p))

    return unique_paths

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

def tail_average(arr, tail=5):
    return np.mean(arr[-tail:])

def moving_average(x, w=3):
    if w <= 1:
        return x
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    x_pad = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(x_pad, kernel, mode="valid")

def plot_mean_std_clean(ax, Y, label, smooth_w=1, shade_only_proposed=True, mark_final=False, use_se=True):
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0)

    # Use standard error for cleaner paper bands
    spread = sd / np.sqrt(Y.shape[0]) if use_se else sd

    mu_s = moving_average(mu, w=smooth_w)
    spread_s = moving_average(spread, w=smooth_w)

    x = np.arange(len(mu_s))
    color = METHOD_COLORS[label]

    ax.plot(
        x, mu_s,
        label=label,
        color=color,
        linestyle=LINESTYLES[label],
        linewidth=LINEWIDTHS[label],
        alpha=ALPHAS[label],
        zorder=3 if label == "Proposed" else 2
    )

    if (not shade_only_proposed) or (label == "Proposed"):
        ax.fill_between(
            x,
            mu_s - spread_s,
            mu_s + spread_s,
            color=color,
            alpha=0.16 if label == "Proposed" else 0.05,
            zorder=1
        )

    if mark_final:
        tail = 5
        tail_point = np.mean(mu_s[-tail:])
        ax.scatter(
            x[-1], tail_point,
            color=color,
            s=30 if label != "Proposed" else 42,
            zorder=4
        )

    return mu_s, spread_s

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

# ------------------------------------------------------------
# Fig. 2
# ------------------------------------------------------------
def make_fig2_proposed_k_accuracy():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    acc_keys = ["test_acc", "global_test_acc", "accuracy", "acc"]

    k_colors = {
        4: "#4C78A8",
        7: "#F58518",
        10: "#54A24B",
    }

    for k in K_COMPARE_VALUES:
        paths = find_runs("marl_full", exp_tag="k_compare", env_tag=ENV_TAG, k_value=k, verbose=True)
        logs = [load_log(p) for p in paths]
        Y = stack_metric(logs, acc_keys, f"accuracy K={k}")

        mu = Y.mean(axis=0)
        se = Y.std(axis=0) / np.sqrt(Y.shape[0])

        mu_s = moving_average(mu, w=3)
        se_s = moving_average(se, w=3)

        x = np.arange(len(mu_s))

        ax.plot(x, mu_s, linewidth=2.5, label=f"K={k}", color=k_colors[k])
        ax.fill_between(x, mu_s - se_s, mu_s + se_s, alpha=0.10, color=k_colors[k])
        ax.scatter(x[-1], np.mean(mu_s[-5:]), s=32, color=k_colors[k], zorder=4)

        print(f"[Fig2][K={k}] runs: {len(paths)}")

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Global Test Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False)

    out_path = os.path.join(RESULTS_DIR, "fig2_proposed_k_accuracy.png")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()

# ------------------------------------------------------------
# Fig. 3
# ------------------------------------------------------------
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
            paths = find_runs(method, exp_tag="k_compare", env_tag=ENV_TAG, k_value=k, verbose=(label == "Proposed"))
            logs = [load_log(p) for p in paths]
            Y = stack_metric(logs, acc_keys, f"accuracy K={k}")

            mu_s, spread_s = plot_mean_std_clean(
                ax, Y, label,
                smooth_w=3,
                shade_only_proposed=False,
                mark_final=True,
                use_se=True
            )

            final_vals = Y[:, -1]
            tail_vals = np.array([tail_average(run, tail=5) for run in Y])

            print(f"{label}, K={k}")
            print("  per-seed final:", np.round(final_vals, 4))
            print("  per-seed tail-avg:", np.round(tail_vals, 4))
            print(f"  std final = {np.std(final_vals):.4f}")
            print(f"  std tail-avg = {np.std(tail_vals):.4f}")

        ax.set_xlabel("FL Round")
        ax.set_ylabel("Global Test Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right", ncol=2, frameon=False)

        ax.text(
            0.5, -0.18, panel_tags[k],
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=11
        )

        out_path = os.path.join(RESULTS_DIR, f"fig3_method_comparison_K{k}.png")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.show()

        print(f"[Fig3][K={k}] saved to {out_path}")

# ------------------------------------------------------------
# Fig. 4
# ------------------------------------------------------------
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

        plot_mean_std_clean(
            ax, Y, label,
            smooth_w=3,
            shade_only_proposed=True,
            mark_final=False,
            use_se=True
        )

        print(f"[Fig4][{label}] runs: {len(paths)}")

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Successful Upload Ratio")
    ax.set_ylim(0.0, 0.8)
    ax.legend(loc="upper left", ncol=2, frameon=False)

    out_path = os.path.join(RESULTS_DIR, f"fig4_success_K{FIXED_K}_{ENV_TAG}.png")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()

# ------------------------------------------------------------
# Fig. 5
# ------------------------------------------------------------
def make_fig5_selection_hist():
    methods = ("pf", "bc", "marl_full")
    labels = ("PF", "BC", "Proposed")

    n_methods = len(methods)
    fig, axes = plt.subplots(
        nrows=n_methods, ncols=1, figsize=(8.0, 2.25 * n_methods), sharex=True
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

        ax.bar(
            x, mu,
            width=0.82,
            alpha=0.92,
            color=METHOD_COLORS[label],
            edgecolor="none"
        )
        ax.errorbar(
            x, mu, yerr=sd,
            fmt="none",
            capsize=1.5,
            linewidth=0.8,
            color="black"
        )

        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.18)
        ax.text(0.01, 0.86, label, transform=ax.transAxes, fontsize=11, weight="bold")

        print(f"[Fig5][{label}] runs: {len(paths)}, UAVs: {len(mu)}")

    axes[-1].set_xlabel("UAV Index")

    out_path = os.path.join(RESULTS_DIR, f"fig5_selection_hist_K{FIXED_K}_{ENV_TAG}.png")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()

# ------------------------------------------------------------
# Fig. 6
# ------------------------------------------------------------
def make_fig6_3d_trajectory():
    paths = find_runs("marl_full", exp_tag="k_compare", env_tag=ENV_TAG, k_value=FIXED_K)
    logs = [load_log(p) for p in paths]
    log = logs[0]

    X = np.array(log["x_uav"])
    Y = np.array(log["y_uav"])
    H = np.array(log["h_uav"])

    fig = plt.figure(figsize=(7.4, 5.4))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10(np.arange(len(UAVS_TO_PLOT)))

    for k, j in enumerate(UAVS_TO_PLOT):
        ax.plot(
            X[:, j], Y[:, j], H[:, j],
            linewidth=1.8, alpha=0.9, color=colors[k], linestyle='-',
            label=f"UAV {k}"
        )

        ax.scatter(
            X[0, j], Y[0, j], H[0, j],
            color="#2ca02c", marker="o", s=34,
            edgecolor="black", linewidth=0.5, zorder=5
        )

        ax.scatter(
            X[-1, j], Y[-1, j], H[-1, j],
            color="#d62728", marker="x", s=42,
            linewidth=1.8, zorder=6
        )

    ax.scatter(0, 0, 25, color="black", marker="^", s=95, label="Base Station", zorder=7)

    ax.set_xlabel("X (m)", labelpad=8)
    ax.set_ylabel("Y (m)", labelpad=8)
    ax.set_zlabel("Altitude (m)", labelpad=8)

    ax.set_xlim(-650, 650)
    ax.set_ylim(-650, 650)
    ax.set_zlim(0, 520)
    ax.view_init(elev=22, azim=-55)

    ax.xaxis.pane.set_alpha(0.08)
    ax.yaxis.pane.set_alpha(0.08)
    ax.zaxis.pane.set_alpha(0.08)
    ax.grid(True)

    legend_markers = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=6, linestyle='None', label='Start'),
        Line2D([0], [0], marker='x', color="#d62728", markersize=7,
               linestyle='None', label='End'),
    ]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + legend_markers, labels + ["Start", "End"], loc="upper right", fontsize=8, frameon=True)

    out_path = os.path.join(RESULTS_DIR, f"fig6_3d_traj_K{FIXED_K}_{ENV_TAG}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

# ------------------------------------------------------------
# Fig. 7
# ------------------------------------------------------------
def make_fig7_env_sweep():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    acc_keys = ["test_acc", "global_test_acc", "accuracy", "acc"]

    envs = ["suburban", "urban", "denseurban", "highrise"]
    methods = ("pf", "bc", "marl_full")
    labels = ("PF", "BC", "Proposed")

    x = np.arange(len(envs))
    width = 0.22
    tail = 5

    for i, (method, label) in enumerate(zip(methods, labels)):
        final_mu = []
        final_sd = []

        for env in envs:
            paths = find_runs(method, exp_tag="env_sweep", env_tag=env, k_value=10)
            logs = [load_log(p) for p in paths]
            Y = stack_metric(logs, acc_keys, f"{label} in {env}")

            tail_vals = np.array([tail_average(run, tail=tail) for run in Y])

            final_mu.append(np.mean(tail_vals))
            final_sd.append(np.std(tail_vals))

        xpos = x + i * width - width

        ax.bar(
            xpos, final_mu, width=width, label=label,
            alpha=0.92, color=METHOD_COLORS[label],
            edgecolor="black", linewidth=0.4
        )

        ax.errorbar(
            xpos, final_mu, yerr=final_sd,
            fmt="none", capsize=2, linewidth=0.9, color="black"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Suburban", "Urban", "Dense Urban", "High-Rise"])
    ax.set_xlabel("Environment")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_ylim(0.55, 1.0)
    ax.grid(True, axis="y", alpha=0.18)
    ax.legend(loc="lower left", fontsize=8, frameon=False)

    out_path = os.path.join(RESULTS_DIR, "fig7_env_sweep.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    
def plot_ablation_mean_se(ax, Y, label, smooth_w=3, mark_final=True):
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0)
    se = sd / np.sqrt(Y.shape[0])

    mu_s = moving_average(mu, w=smooth_w)
    se_s = moving_average(se, w=smooth_w)

    x = np.arange(len(mu_s))
    color = ABLATION_COLORS[label]

    ax.plot(
        x, mu_s,
        label=label,
        color=color,
        linestyle=ABLATION_LINESTYLES[label],
        linewidth=ABLATION_LINEWIDTHS[label],
        zorder=3 if label == "Full MARL" else 2
    )

    ax.fill_between(
        x,
        mu_s - se_s,
        mu_s + se_s,
        color=color,
        alpha=0.14 if label == "Full MARL" else 0.06,
        zorder=1
    )

    if mark_final:
        tail_point = np.mean(mu_s[-5:])
        ax.scatter(
            x[-1], tail_point,
            color=color,
            s=42 if label == "Full MARL" else 30,
            zorder=4
        )

    return mu_s, se_s

# ------------------------------------------------------------
# Ablation Fig A: Accuracy vs rounds at K=10
# ------------------------------------------------------------
def make_ablation_accuracy():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    acc_keys = ["test_acc", "global_test_acc", "accuracy", "acc"]

    for method, label in zip(ABLATION_METHODS, ABLATION_LABELS):
        paths = find_runs(method, exp_tag="ablation_k10", env_tag=ENV_TAG, k_value=10, verbose=(label == "Full MARL"))
        logs = [load_log(p) for p in paths]
        Y = stack_metric(logs, acc_keys, f"ablation accuracy {label}")

        mu_s, se_s = plot_ablation_mean_se(ax, Y, label, smooth_w=3, mark_final=True)

        final_vals = Y[:, -1]
        tail_vals = np.array([tail_average(run, tail=5) for run in Y])

        print(f"{label}")
        print("  per-seed final:", np.round(final_vals, 4))
        print("  per-seed tail-avg:", np.round(tail_vals, 4))
        print(f"  std final = {np.std(final_vals):.4f}")
        print(f"  std tail-avg = {np.std(tail_vals):.4f}")

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Global Test Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", ncol=2, frameon=False)

    out_path = os.path.join(RESULTS_DIR, "ablation_k10_accuracy.png")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()

    print(f"[Ablation Accuracy] saved to {out_path}")
    
    
# ------------------------------------------------------------
# Ablation Fig B: Upload success vs rounds at K=10
# ------------------------------------------------------------
def make_ablation_success():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    success_keys = [
        "success_rate",
        "sel_success",
        "selected_success",
        "selected_upload_success",
        "upload_success",
        "mean_sel_success",
    ]

    for method, label in zip(ABLATION_METHODS, ABLATION_LABELS):
        paths = find_runs(method, exp_tag="ablation_k10", env_tag=ENV_TAG, k_value=10, verbose=False)
        logs = [load_log(p) for p in paths]
        Y = stack_metric(logs, success_keys, f"ablation success {label}")

        plot_ablation_mean_se(ax, Y, label, smooth_w=3, mark_final=False)

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Successful Upload Ratio")
    ax.set_ylim(0.0, 0.8)
    ax.legend(loc="upper left", ncol=2, frameon=False)

    out_path = os.path.join(RESULTS_DIR, "ablation_k10_success.png")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.show()

    print(f"[Ablation Success] saved to {out_path}")
    
    


def main():
    # make_fig2_proposed_k_accuracy()
    # make_fig3_k_method_comparison()
    # make_fig4_success()
    # make_fig5_selection_hist()
    # make_fig6_3d_trajectory()
    # make_fig7_env_sweep()
    make_ablation_accuracy()
    make_ablation_success()

if __name__ == "__main__":
    main()