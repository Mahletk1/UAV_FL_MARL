import subprocess
import sys
import os

PYTHON = sys.executable

if '__file__' in globals():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    SCRIPT_DIR = os.getcwd()

MAIN_FILE = os.path.join(SCRIPT_DIR, "main.py")

SEEDS = [1, 2]

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
# 1) Main comparison: Fig. 2, Fig. 3, Fig. 4, Fig. 5
# methods = RS, RR, PF, BC, Proposed
# fixed K=10, env=highrise
# ============================================================
def run_main_experiments():
    methods = [
        ["--method", "random"],
        ["--method", "round_robin"],
        ["--method", "pf"],
        ["--method", "greedy_channel"],
        ["--method", "marl", "--marl_mode", "full"],
    ]

    for seed in SEEDS:
        for method_args in methods:
            args = method_args + [
                "--env", "highrise",
                "--active_UE", "10",
                "--exp_tag", "main",
                "--seed", str(seed),
            ]
            run_cmd(args)

# ============================================================
# 2) K sweep: Fig. 6
# Proposed only, env=highrise
# ============================================================
def run_k_sweep():
    k_values = [2, 4, 6, 8, 10]

    for seed in SEEDS:
        for k in k_values:
            args = [
                "--method", "marl",
                "--marl_mode", "full",
                "--env", "highrise",
                "--active_UE", str(k),
                "--exp_tag", "k_sweep",
                "--seed", str(seed),
            ]
            run_cmd(args)

# ============================================================
# 3) Environment sweep: Fig. 7
# methods = PF, BC, Proposed
# fixed K=10
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

if __name__ == "__main__":
    run_main_experiments()
    # run_k_sweep()
    # run_env_sweep()