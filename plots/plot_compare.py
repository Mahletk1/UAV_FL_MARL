import os
import numpy as np
import matplotlib.pyplot as plt

# Directory of this script: .../SimCode/plots
HERE = os.path.dirname(os.path.abspath(__file__))

# Project root: .../SimCode
ROOT = os.path.abspath(os.path.join(HERE, ".."))

# Results dir: .../SimCode/results
RESULTS = os.path.join(ROOT, "results")

def load_log(path):
    d = np.load(path, allow_pickle=True).item()

    # ensure arrays (these always exist)
    d["round"] = np.array(d["round"])
    d["test_acc"] = np.array(d["test_acc"])
    d["test_loss"] = np.array(d["test_loss"])

    # optional arrays (exist if your main.py logged them)
    if "selected_mask" in d:
        d["selected_mask"] = np.array(d["selected_mask"])
    if "success_mask" in d:
        d["success_mask"] = np.array(d["success_mask"])

    return d

def plot_accuracy(logs, labels):
    plt.figure()
    for d, lab in zip(logs, labels):
        plt.plot(d["round"], d["test_acc"], label=lab)
    plt.xlabel("FL Round")
    plt.ylabel("Global Test Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_loss(logs, labels):
    plt.figure()
    for d, lab in zip(logs, labels):
        plt.plot(d["round"], d["test_loss"], label=lab)
    plt.xlabel("FL Round")
    plt.ylabel("Global Test Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
def plot_selection_counts_per_client_grouped(logs, labels,
                                             save_path="selection_counts_grouped.png",
                                             normalize=False):
    """
    Grouped bar chart: for each client index, show selection count for each method.

    normalize=False -> raw counts (# rounds selected)
    normalize=True  -> average selection rate (count / T)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # collect counts per method
    counts_list = []
    T_list = []

    for d, lab in zip(logs, labels):
        if "selected_mask" not in d:
            print(f"[WARN] {lab}: no selected_mask found.")
            return

        sel = np.array(d["selected_mask"], dtype=np.float32)  # [T, N]
        T, N = sel.shape
        counts = sel.sum(axis=0)  # [N]
        counts_list.append(counts)
        T_list.append(T)

    # sanity: same N across methods
    N = len(counts_list[0])
    for c in counts_list:
        if len(c) != N:
            raise ValueError("Different number of clients across logs.")

    # normalize if requested
    if normalize:
        counts_list = [c / (T + 1e-12) for c, T in zip(counts_list, T_list)]
        ylab = "Average selection rate (fraction of rounds)"
        title = "Average Selection Rate per Client (Grouped)"
    else:
        ylab = "Times selected"
        title = "Selection Count per Client (Grouped)"

    x = np.arange(N)
    m = len(labels)
    width = 0.8 / m  # total bar width ~0.8

    plt.figure(figsize=(12, 4))
    for j, (counts, lab) in enumerate(zip(counts_list, labels)):
        plt.bar(x + (j - (m - 1) / 2) * width, counts, width=width, label=lab)

    plt.xlabel("Client (UAV) index")
    plt.ylabel(ylab)
    plt.title(title)
    plt.xticks(x)  # show each client id; if N big, comment this out
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"Saved: {save_path}")

def plot_3d_uav_scatter_at_round(log, title="UAV 3D positions", r=0, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = np.array(log["x_uav"])  # [T, N]
    y = np.array(log["y_uav"])  # [T, N]
    h = np.array(log["h_uav"])  # [T, N]

    r = int(np.clip(r, 0, x.shape[0] - 1))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x[r], y[r], h[r], s=25)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("height (m)")
    ax.set_title(f"{title} — round {r}")

    # Optional: keep equal-ish XY scaling for nicer geometry feel
    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=15, azim=-120)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
    plt.show()
    
def plot_3d_uav_trajectories(log, uav_ids=None, title="UAV 3D trajectories", save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = np.array(log["x_uav"])  # [T, N]
    y = np.array(log["y_uav"])  # [T, N]
    h = np.array(log["h_uav"])  # [T, N]

    T, N = x.shape
    if uav_ids is None:
        uav_ids = list(range(min(5, N)))  # default: first 5

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")

    for i in uav_ids:
        i = int(i)
        ax.plot(x[:, i], y[:, i], h[:, i], linewidth=1.5, label=f"UAV {i}")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("height (m)")
    ax.set_title(title)
    ax.legend(loc="best")

    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=15, azim=-120)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
    plt.show()
    
def plot_3d_scatter_selected_success(log, title="UAV 3D (selection & success)", r=0, save_path=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = np.array(log["x_uav"])        # [T, N]
    y = np.array(log["y_uav"])        # [T, N]
    h = np.array(log["h_uav"])        # [T, N]
    sel = np.array(log["selected_mask"])  # [T, N]
    succ = np.array(log["success_mask"])  # [T, N]

    r = int(np.clip(r, 0, x.shape[0] - 1))

    mask_sel = sel[r].astype(bool)
    mask_succ = succ[r].astype(bool)

    mask_sel_succ = mask_sel & mask_succ        # selected & successful
    mask_sel_fail = mask_sel & (~mask_succ)     # selected but failed
    mask_not_sel  = ~mask_sel                   # not selected

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x[r, mask_not_sel],  y[r, mask_not_sel],  h[r, mask_not_sel],
               s=20, alpha=0.6, label="not selected")
    ax.scatter(x[r, mask_sel_fail], y[r, mask_sel_fail], h[r, mask_sel_fail],
               s=40, alpha=0.9, label="selected (failed)")
    ax.scatter(x[r, mask_sel_succ], y[r, mask_sel_succ], h[r, mask_sel_succ],
               s=50, alpha=1.0, label="selected (successful)")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("height (m)")
    ax.set_title(f"{title} — round {r}")
    ax.legend(loc="best")

    ax.set_box_aspect((1, 1, 0.6))
    ax.view_init(elev=15, azim=-120)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved: {save_path}")
    plt.show()
    
def main():
    # CHANGE these to match your saved filenames
    print("HERE:", HERE)
    print("ROOT:", ROOT)
    print("RESULTS:", RESULTS)
    print("FILES IN RESULTS:", os.listdir(RESULTS))
    
    paths = [
        os.path.join(RESULTS, "random_dirichlet_highrise_wirelessTrue.npy"),
        os.path.join(RESULTS, "greedy_channel_dirichlet_highrise_wirelessTrue.npy"),
        os.path.join(RESULTS, "marl_dirichlet_highrise_wirelessTrue.npy"),
    ]
    labels = ["Random", "Greedy-Channel", "MARL (MAPPO)"]

    logs = [load_log(p) for p in paths]

    # plot_accuracy(logs, labels)
    # plot_loss(logs, labels)
    
    # plot_selection_counts_per_client_grouped(
    #     logs, labels,
    #     save_path=os.path.join(RESULTS, "selection_counts_grouped.png"),
    #     normalize=False   # raw counts
    # )

    # # Optional: also save normalized rate version
    # plot_selection_counts_per_client_grouped(
    #     logs, labels,
    #     save_path=os.path.join(RESULTS, "selection_rate_grouped.png"),
    #     normalize=True
    # )
    
    # # ---- 3D visualizations for MARL (MAPPO) ----
    # r = 30  # pick any round to visualize (e.g., early/mid/late: 0, 30, 80)
    
    # plot_3d_uav_scatter_at_round(
    #     logs[1], title=labels[1], r=r,
    #     save_path=os.path.join(RESULTS, f"3d_scatter_{labels[2].replace(' ', '_')}_r{r}.png")
    # )
    
    # plot_3d_scatter_selected_success(
    #     logs[1], title=labels[1], r=65,
    #     save_path=os.path.join(RESULTS, "3d_selected_success_marl_r30.png")
    # )
    
    plot_3d_uav_trajectories(
        logs[2], uav_ids=[1, 3, 4],
        title=f"{labels[2]} — trajectories",
        save_path=os.path.join(RESULTS, f"3d_traj_{labels[2].replace(' ', '_')}.png")
    )

    plt.show()

if __name__ == "__main__":
    main()