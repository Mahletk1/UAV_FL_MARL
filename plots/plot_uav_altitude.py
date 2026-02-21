import numpy as np
import matplotlib.pyplot as plt

def main(log_path, uav_ids=(0,1,2,3,4)):
    d = np.load(log_path, allow_pickle=True).item()
    h = np.array(d["h_uav"])  # [T, N]

    plt.figure()
    for i in uav_ids:
        plt.plot(h[:, i], label=f"UAV {i}")
    plt.xlabel("Round")
    plt.ylabel("Altitude (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main("results/marl_dirichlet_highrise_wirelessTrue.npy")