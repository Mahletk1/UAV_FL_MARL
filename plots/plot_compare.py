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
    # ensure arrays
    d["round"] = np.array(d["round"])
    d["test_acc"] = np.array(d["test_acc"])
    d["test_loss"] = np.array(d["test_loss"])
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

    plot_accuracy(logs, labels)
    plot_loss(logs, labels)


    plt.show()

if __name__ == "__main__":
    main()