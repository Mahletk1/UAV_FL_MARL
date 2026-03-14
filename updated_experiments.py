import subprocess
import sys
import os

PYTHON = sys.executable

if '__file__' in globals():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    SCRIPT_DIR = os.getcwd()

MAIN_FILE = os.path.join(SCRIPT_DIR, "main.py")

# Final paper runs
SEEDS = [1,2,3,4,5]

def run_cmd(arg_list):
    cmd = [PYTHON, "-u", MAIN_FILE] + arg_list
    print("\nRunning:", " ".join(cmd), flush=True)

    process = subprocess.Popen(
        cmd,
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)

# ============================================================
# 1) K-comparison runs for Fig. 2 and Fig. 3
# methods = RS, RR, PF, BC, Proposed
# env = highrise
# K = 4, 7, 10
# ============================================================
def run_k_compare():
    k_values = [4, 7, 10]

    methods = [
        ["--method", "random"],
        ["--method", "round_robin"],
        ["--method", "pf"],
        ["--method", "greedy_channel"],
        ["--method", "marl", "--marl_mode", "full"],
    ]

    for seed in SEEDS:
        for k in k_values:
            for method_args in methods:
                args = method_args + [
                    "--env", "highrise",
                    "--active_UE", str(k),
                    "--exp_tag", "k_compare",
                    "--seed", str(seed),
                ]
                run_cmd(args)

# ============================================================
# 2) Environment sweep for Fig. 7
# methods = PF, BC, Proposed
# fixed K = 10
# envs = suburban, urban, denseurban, highrise
# ============================================================
def run_env_sweep():
    envs = ["suburban", "urban", "denseurban", "highrise"]
    methods = [
        ["--method", "pf"],
        ["--method", "greedy_channel"],
        ["--method", "marl", "--marl_mode", "full"],
    ]

    for seed in SEEDS:
        for env in envs:
            for method_args in methods:
                args = method_args + [
                    "--env", env,
                    "--active_UE", "10",
                    "--exp_tag", "env_sweep",
                    "--seed", str(seed),
                ]
                run_cmd(args)
                
# ============================================================
# 3) Ablation study for K = 10
# Compare:
# - BC (fixed altitude + greedy selection)
# - MARL altitude only + greedy selection
# - MARL selection only + fixed altitude
# - Full MARL
# ============================================================
def run_ablation_k10():
    methods = [
        ["--method", "greedy_channel"],                         # BC
        ["--method", "marl", "--marl_mode", "altitude_only","--alt_only_selector", "random"],  # learned altitude only
        ["--method", "marl", "--marl_mode", "selection_only"], # learned selection only
        ["--method", "marl", "--marl_mode", "full"],           # full MARL
    ]

    for seed in SEEDS:
        for method_args in methods:
            args = method_args + [
                "--env", "highrise",
                "--active_UE", "10",
                "--exp_tag", "ablation_k10",
                "--seed", str(seed),
            ]
            run_cmd(args)
            
def run_ablation_k10_alt_random():
    methods = [
        [
            "--method", "marl",
            "--marl_mode", "selection_only",   # force random selection
        ],
    ]

    for seed in SEEDS:
        for method_args in methods:
            args = method_args + [
                "--env", "highrise",
                "--active_UE", "10",
                "--exp_tag", "ablation_k10",   # same folder as before
                "--seed", str(seed),
            ]
            run_cmd(args)

if __name__ == "__main__":
    # run_k_compare()
    # run_env_sweep()
    # run_ablation_k10()
    run_ablation_k10_alt_random()