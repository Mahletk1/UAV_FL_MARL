import os
import numpy as np
import matplotlib.pyplot as plt

# Directory of this script: .../SimCode/plots
HERE = os.path.dirname(os.path.abspath(__file__))

# Project root: .../SimCode
ROOT = os.path.abspath(os.path.join(HERE, ".."))

# Results dir: .../SimCode/results
RESULTS = os.path.join(ROOT, "results")

def main(log_path, show_only_uavs=None):
    d = np.load(log_path, allow_pickle=True).item()

    x = np.array(d["x_uav"])           # [T, N]
    y = np.array(d["y_uav"])           # [T, N]
    h = np.array(d["h_uav"])           # [T, N]
    sel = np.array(d["selected_mask"]) # [T, N]

    T, N = h.shape
    uavs = range(N) if show_only_uavs is None else show_only_uavs

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in uavs:
        # trajectory line
        ax.plot(x[:, i], y[:, i], h[:, i], linewidth=1)

        # selected points
        t_sel = np.where(sel[:, i] > 0.5)[0]
        ax.scatter(x[t_sel, i], y[t_sel, i], h[t_sel, i], s=10)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("h (m)")
    ax.set_title("UAV 3D Trajectories (selected points highlighted)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main("results/marl_dirichlet_highrise_wirelessTrue.npy", show_only_uavs=[0,1,2,3,4])