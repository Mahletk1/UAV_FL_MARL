import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_key_from_summary(summary_path, key_substring, out_png):
    with open(summary_path, "r") as f:
        d = json.load(f)

    keys = [k for k in d.keys() if key_substring in k]
    if not keys:
        print(f"[Plot] No key containing '{key_substring}' found.")
        print("Available keys:", list(d.keys())[:10])
        return

    k = keys[0]
    arr = np.array(d[k], dtype=np.float32)  # [walltime, step, value]
    steps = arr[:, 1]
    values = arr[:, 2]

    plt.figure()
    plt.plot(steps, values)
    plt.xlabel("Training step")
    plt.ylabel(key_substring)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    print(f"[Plot] Saved {out_png} from key: {k}")


if __name__ == "__main__":
    run_dir = Path(r"C:\Users\USER\Desktop\conference\SimCode\light_mappo\results\MyEnv\MyEnv\mappo\check\run1\logs")
    summary_path = run_dir / "summary.json"

    plot_key_from_summary(summary_path, "average_episode_rewards", run_dir / "avg_reward.png")

    # Only works if you log these during training:
    plot_key_from_summary(summary_path, "mean_altitude", run_dir / "mean_altitude.png")
    plot_key_from_summary(summary_path, "mean_selected_snr", run_dir / "mean_selected_snr.png")
    plot_key_from_summary(summary_path, "unique_selected_ratio", run_dir / "unique_selected_ratio.png")
    plot_key_from_summary(summary_path, "mean_abs_dh", run_dir / "mean_abs_dh.png")