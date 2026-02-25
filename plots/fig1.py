import os
import glob
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # simcode/
RESULTS_DIR = os.path.join(ROOT, "results")


def load_log(path: str) -> dict:
    return np.load(path, allow_pickle=True).item()


def find_runs(method: str, data_mode: str, env_tag: str, wireless_on: int) -> list[str]:
    # Your filenames use wirelessTrue / wirelessFalse
    wtag = "True" if int(wireless_on) == 1 else "False"
    pattern = os.path.join(
        RESULTS_DIR,
        f"{method}_{data_mode}_{env_tag}_wireless{wtag}_seed*.npy"
    )
    paths = sorted(glob.glob(pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No logs found for pattern:\n{pattern}")
    return paths


def stack_metric(logs: list[dict], key: str) -> np.ndarray:
    """Return Y of shape [S, T] aligned to shortest run."""
    arrs = [np.asarray(l[key], dtype=np.float32) for l in logs]
    T = min(a.shape[0] for a in arrs)
    arrs = [a[:T] for a in arrs]
    return np.stack(arrs, axis=0)


def plot_mean_std(ax, Y: np.ndarray, label: str):
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0)
    x = np.arange(mu.shape[0])
    ax.plot(x, mu, label=label)
    ax.fill_between(x, mu - sd, mu + sd, alpha=0.2)


def make_fig1(data_mode: str, env_tag: str, wireless_on: int,
              methods=("random", "greedy_channel", "marl_full"),
              labels=("Random", "Greedy-Channel", "MARL (MAPPO)"),
              out_path: str | None = None):

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

    # ---- Accuracy ----
    ax = axes[0]
    for m, lab in zip(methods, labels):
        paths = find_runs(m, data_mode, env_tag, wireless_on)
        logs = [load_log(p) for p in paths]
        Y = stack_metric(logs, "test_acc")
        plot_mean_std(ax, Y, lab)
        print(f"[{lab}] runs: {len(paths)}")
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Global Test Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # ---- Loss ----
    ax = axes[1]
    for m, lab in zip(methods, labels):
        paths = find_runs(m, data_mode, env_tag, wireless_on)
        logs = [load_log(p) for p in paths]
        Y = stack_metric(logs, "test_loss")
        plot_mean_std(ax, Y, lab)
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Global Test Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(
            RESULTS_DIR,
            f"fig1_acc_loss_{data_mode}_{env_tag}_wireless{wireless_on}.png"
        )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)


def auto_detect_config():
    """Detect (data_mode, env_tag, wireless_on) from any *_seed*.npy file.
       Supports method names with underscores (e.g., greedy_channel)
       and wirelessTrue/wirelessFalse or wireless1/wireless0.
    """
    files = sorted(
    f for f in os.listdir(RESULTS_DIR)
    if f.startswith("random_") and f.endswith(".npy") and "_seed" in f and "_wireless" in f
    )
    
    if not files:
        raise RuntimeError(
            f"No seed logs found in {RESULTS_DIR}.\n"
            "Expected files like: random_dirichlet_highrise_wirelessTrue_seed1.npy"
        )

    base = files[0].replace(".npy", "")
    parts = base.split("_")

    # Find tokens like wirelessTrue / wireless1 and seed5
    wireless_token = next(p for p in parts if p.startswith("wireless"))
    seed_token = next(p for p in parts if p.startswith("seed"))

    wireless_str = wireless_token.replace("wireless", "").lower()
    wireless_on = 1 if wireless_str in ["true", "1"] else 0

    # Everything between method and wireless token is: data_mode, env_tag
    widx = parts.index(wireless_token)

    # parts structure example:
    # ["greedy","channel","dirichlet","highrise","wirelessTrue","seed1"]
    # method = parts[:widx-2] -> ["greedy","channel"]
    # data_mode = parts[widx-2] -> "dirichlet"
    # env_tag   = parts[widx-1] -> "highrise"
    if widx < 3:
        raise RuntimeError(f"Unexpected filename format: {files[0]}")

    data_mode = parts[widx - 2]
    env_tag = parts[widx - 1]

    return data_mode, env_tag, wireless_on


if __name__ == "__main__":
    # Hardcode main-result scenario for Fig. 1 (camera-ready)
    data_mode = "dirichlet"
    env_tag = "highrise"
    wireless_on = True  # or True

    print(f"[Config] data_mode={data_mode}, env={env_tag}, wireless_on={wireless_on}")
    make_fig1(data_mode=data_mode, env_tag=env_tag, wireless_on=wireless_on)