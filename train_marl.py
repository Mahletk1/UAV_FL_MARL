# train_marl.py
import numpy as np
import torch

from utils.options import args_parser
from UE_Selection.UAV_scenario import init_circular_xy_trajectory,init_random_xy_trajectory,init_random_walk_xy_trajectory
from uav_marl_env import UAVScoreEnv
from mappo_agent import MAPPOAgent
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from utils.sampling_func import DataPartitioner
import numpy as np

def build_data_ratio_from_dataset(args):
    """
    Returns:
      data_ratio: np.ndarray shape [N], sum=1
      dict_users: dict {i: list_of_indices}  (optional, useful for debugging)
    """
    if args.dataset == "mnist":
        trans = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.MNIST("./data/mnist/", train=True, download=True, transform=trans)
    elif args.dataset == "cifar10":
        trans = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.CIFAR10("./data/cifar10/", train=True, download=True, transform=trans)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # IMPORTANT: match your FL code behavior:
    # NonIID uses args.iid (you set it to 'dirichlet') and args.alpha
    partition_obj = DataPartitioner(
        dataset_train,
        args.total_UE,
        NonIID=args.iid,   # e.g., 'dirichlet' or 'iid'
        alpha=args.alpha
    )
    dict_users, _ = partition_obj.use()

    sizes = np.array([len(dict_users[i]) for i in range(args.total_UE)], dtype=np.float32)
   
    data_ratio = sizes / (sizes.sum() + 1e-8)
    print(data_ratio)
    return data_ratio, dict_users


# Same env_params dict you already have in main.py
ENV_PARAMS = {
    'suburban': {'a': 4.88, 'b': 0.43, 'eta1_db': 0.1, 'eta2_db': 21},
    'urban': {'a': 9.61, 'b': 0.16, 'eta1_db': 1.0, 'eta2_db': 20},
    'denseurban': {'a': 12.08, 'b': 0.11, 'eta1_db': 1.6, 'eta2_db': 23},
    'highrise': {'a': 27.23, 'b': 0.08, 'eta1_db': 2.3, 'eta2_db': 34}
}

def main():
    args = args_parser()
    args.train_marl = True
    ep_return_hist = []
    rel_hist, fair_hist, small_hist, pdh_hist = [], [], [], []
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # trajectories for episode_len steps
    # traj_x, traj_y = init_circular_xy_trajectory(
    #     N=args.total_UE, T=args.episode_len, R_mean=200.0, R_jitter=40.0, seed=args.seed
    # )
    # traj_x, traj_y = init_random_xy_trajectory(
    #     N=args.total_UE,
    #     T=args.round,
    #     area_size=500.0,
    #     seed=args.seed
    # )

    traj_x, traj_y = init_random_walk_xy_trajectory(
         N=args.total_UE,
         T=args.episode_len,   # <-- FIX
         area_size=500.0,
         step_std=20.0,
         seed=args.seed
    )
    env = UAVScoreEnv(args, traj_x, traj_y, ENV_PARAMS[args.env])

    # Fake data_ratio for wireless-only training (replace with real dict_users ratios later)
    # For training stability: random but fixed distribution
    # sizes = np.random.randint(200, 2000, size=args.total_UE).astype(np.float32)
    # data_ratio = sizes / (sizes.sum() + 1e-8)
    # env.set_data_ratio(data_ratio)
    data_ratio, dict_users = build_data_ratio_from_dataset(args)
    env.set_data_ratio(data_ratio)
    
    print("Real data_ratio stats:",
          "min=", float(data_ratio.min()),
          "max=", float(data_ratio.max()),
          "mean=", float(data_ratio.mean())) 


    obs_dim = 6
    state_dim = args.total_UE * obs_dim

    agent = MAPPOAgent(args, obs_dim=obs_dim, state_dim=state_dim, device=device)

    for ep in range(args.marl_episodes):
        obs_n, state = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            dh, score, logp_sum, value = agent.act(obs_n, state)
            obs_n2, state2, reward, done, info = env.step(dh, score)

            agent.store(obs_n, state, dh, score, logp_sum, value, reward, done)

            obs_n, state = obs_n2, state2
            ep_reward += reward

        agent.update()
        avg_step_return = ep_reward / env.T
        ep_return_hist.append(avg_step_return)
        
        rel_hist.append(info["R_rel"])
        fair_hist.append(info["R_fair"])
        small_hist.append(info["R_small"])
        pdh_hist.append(info["P_up_non"])

        if (ep + 1) % 10 == 0:
            print(f"[EP {ep+1:04d}] reward={ep_reward/env.T:.3f} "
                  f"R_rel={info['R_rel']:.3f} R_fair={info['R_fair']:.3f} "
                  f"R_small={info['R_small']:.3f} mean_h={info['mean_h']:.1f}")

    agent.save("marl_policy.pt")
    print("Saved policy to marl_policy.pt")
    
    def moving_average(x, w=20):
        if len(x) < w:
            return x
        x = np.array(x, dtype=np.float32)
        return np.convolve(x, np.ones(w)/w, mode="valid")

    plt.figure()
    plt.plot(ep_return_hist, label="Avg reward/step")
    ma = moving_average(ep_return_hist, w=20)
    plt.plot(range(len(ep_return_hist)-len(ma), len(ep_return_hist)), ma, label="MA(20)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_reward.png", dpi=200)
    
    plt.figure()
    plt.plot(rel_hist, label="R_rel")
    plt.plot(fair_hist, label="R_fair")
    plt.plot(small_hist, label="R_small")
    plt.plot(pdh_hist, label="P_up_non")
    plt.xlabel("Episode")
    plt.ylabel("Metric")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_metrics.png", dpi=200)
    
    print("Saved plots: train_reward.png, train_metrics.png")

if __name__ == "__main__":
    main()
