# marl/train/random_env_test.py
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from utils.options import args_parser
from UE_Selection.UAV_scenario import init_circular_xy_trajectory, init_predefined_height_trajectory

from marl.envs.uav_fl_env import UAVFLEnv


ENV_PARAMS = {

    'suburban': {
        'a': 4.88, 'b': 0.43,
        'eta1_db': 0.1, 'eta2_db': 21
    },
    'urban': {
        'a': 9.61, 'b': 0.16,
        'eta1_db': 1, 'eta2_db': 20
    },
    'denseurban': {
        'a': 12.08, 'b': 0.11,
        'eta1_db': 1.6, 'eta2_db': 23
    },
    'highrise': {
        'a': 27.23, 'b': 0.08,
        'eta1_db': 2.3, 'eta2_db': 34
    }
}
def main():
    args = args_parser()

    # Provide bounds to env through args (env reads these)
    args.h_min = 80
    args.h_max = 500

    traj_x, traj_y = init_circular_xy_trajectory(
        N=args.total_UE,
        T=args.round,
        R_mean=200.0,
        R_jitter=60.0,
        seed=42
    )
    traj_h_base = init_predefined_height_trajectory(
        N=args.total_UE,
        T=args.round,
        h_min=args.h_min,
        h_max=args.h_max,
        seed=getattr(args, "seed", 0)
    )

    env_cfg = ENV_PARAMS[args.env]  # {'a','b','eta1_db','eta2_db'}

    env = UAVFLEnv(
        args=args,
        traj_x=traj_x, traj_y=traj_y, traj_h_base=traj_h_base,
        env_params=env_cfg,
        selector_type="greedy_channel",
        max_dh=10.0,
        episode_len=20,          # short test episode
        run_full_fl=False        # keep fast
    )

    obs, state, info = env.reset(seed=0)
    print("reset:")
    print("  obs shape:", obs.shape)
    print("  state shape:", state.shape)

    for t in range(5):
        actions = np.random.uniform(-1, 1, size=(args.total_UE,)).astype(np.float32)
        obs, state, rew, done, info = env.step(actions)
        print(f"t={t} reward(mean)={rew.mean():.3f} success_rate={info['success_rate']:.3f} done={done}")
        if done:
            break


if __name__ == "__main__":
    main()
