# fig4_ablation.py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # SimCode/
RESULTS_DIR = os.path.join(ROOT, "results")

# Fixed config (prevents mixing runs)
DATA_MODE = "dirichlet"
ENV_TAG = "highrise"
WIRELESS_ON = True  # filenames use wirelessTrue/wirelessFalse

METHODS = (
    "random",
    "greedy_channel",
    "marl_selection_only",
    "marl_altitude_only",
    "marl_full",
)

LABELS = (
    "Random",
    "Greedy-Channel",
    "MARL (Selection-only)",
    "MARL (Altitude-only)",
    "Full MARL (MAPPO)",
)


def load_log(path: str) -> dict:
    return np.load(path, allow_pickle=True).item()


def find_runs(method: str) -> list[str]:
    wtag = "True" if WIRELESS_ON else "False"
    pattern = os.path.join(
        RESULTS_DIR,
        f"{method}_{DATA_MODE}_{ENV_TAG}_wireless{wtag}_seed*.npy"
    )
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No logs found for pattern:\n{pattern}")
    return paths


def stack_metric(logs: list[dict], key: str) -> np.ndarray:
    """Return Y of shape [S, T] aligned to shortest run."""
    arrs = []
    for i, l in enumerate(logs):
        if key not in l:
            raise KeyError(
                f"Missing key '{key}' in run #{i}. Available keys:\n{list(l.keys())}"
            )
        arrs.append(np.asarray(l[key], dtype=np.float32))

    T = min(a.shape[0] for a in arrs)
    arrs = [a[:T] for a in arrs]
    return np.stack(arrs, axis=0)


def plot_mean_std(ax, Y: np.ndarray, label: str):
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0)
    x = np.arange(mu.shape[0])
    ax.plot(x, mu, label=label)
    ax.fill_between(x, mu - sd, mu + sd, alpha=0.18)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.0))

    for m, lab in zip(METHODS, LABELS):
        paths = find_runs(m)
        logs = [load_log(p) for p in paths]
        Y = stack_metric(logs, "test_acc")
        plot_mean_std(ax, Y, lab)
        print(f"[{lab}] runs: {len(paths)}")

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Global Test Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    out_path = os.path.join(
        RESULTS_DIR,
        f"fig4_ablation_acc_{DATA_MODE}_{ENV_TAG}_wireless{WIRELESS_ON}.png"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()