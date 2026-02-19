# marl/train/train_mappo_phase1.py
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure project root is on path (for Spyder)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.options import args_parser
from UE_Selection.UAV_scenario import init_circular_xy_trajectory, init_predefined_height_trajectory
from marl.envs.uav_fl_env import UAVFLEnv
import copy
from torchvision import datasets, transforms

from utils.sampling_func import DataPartitioner
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.evaluation import test_model
from models.Nets import CNNMnist, CNN60K, ResNetCifar


# -------- Networks --------
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # learnable std

    def forward(self, obs):
        x = self.net(obs)
        mu = torch.tanh(self.mu(x))  # action in [-1,1]
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)
        return mu, std


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


# -------- Rollout buffer --------
class Buffer:
    def __init__(self):
        self.obs = []
        self.state = []
        self.act = []
        self.logp = []
        self.rew = []
        self.done = []
        self.val = []

    def add(self, obs, state, act, logp, rew, done, val):
        self.obs.append(obs)
        self.state.append(state)
        self.act.append(act)
        self.logp.append(logp)
        self.rew.append(rew)
        self.done.append(done)
        self.val.append(val)

    def clear(self):
        self.__init__()


def gae_advantages(rews, dones, vals, gamma=0.99, lam=0.95):
    """
    Inputs are lists length T:
      rews[t] scalar (team reward)
      dones[t] bool
      vals[t] scalar V(s_t)
    Returns:
      adv[T], ret[T]
    """
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    vals = np.array(vals + [0.0], dtype=np.float32)  # V(s_T)=0 bootstrap
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = rews[t] + gamma * vals[t+1] * nonterminal - vals[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + vals[:-1]
    return adv, ret


def main():
    args = args_parser()
    args.h_min = 80
    args.h_max = 500

    # ---- Build trajectories ----
    traj_x, traj_y = init_circular_xy_trajectory(
        N=args.total_UE, T=args.round, R_mean=200.0, R_jitter=60.0, seed=42
    )
    traj_h_base = init_predefined_height_trajectory(
        N=args.total_UE, T=args.round, h_min=args.h_min, h_max=args.h_max, seed=getattr(args, "seed", 0)
    )

    # ---------------- Build FL components (needed for run_full_fl=True) ----------------
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    if args.dataset == "mnist":
        trans = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.MNIST("./data/mnist/", train=True, download=True, transform=trans)
        dataset_test  = datasets.MNIST("./data/mnist/", train=False, download=True, transform=trans)
        args.num_channels = 1
        args.num_classes = 10
    
    elif args.dataset == "cifar10":
        trans = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset_train = datasets.CIFAR10("./data/cifar10/", train=True, download=True, transform=trans)
        dataset_test  = datasets.CIFAR10("./data/cifar10/", train=False, download=True, transform=trans)
        args.num_channels = 3
        args.num_classes = 10
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Partition data into N clients (UAVs)
    partition_obj = DataPartitioner(dataset_train, args.total_UE, NonIID=args.iid, alpha=args.alpha)
    dict_users, _ = partition_obj.use()
    
    # Global model
    if args.model == "cnn":
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == "cnn60k":
        net_glob = CNN60K(args=args).to(args.device)
    elif args.model == "resnet":
        net_glob = ResNetCifar(num_classes=args.num_classes).to(args.device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    net_glob.train()


    # ---- Env params (use same keys you use in args.env) ----
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
    env_cfg = ENV_PARAMS[args.env]
    env = UAVFLEnv(
        args=args,
        traj_x=traj_x, traj_y=traj_y, traj_h_base=traj_h_base,
        env_params=env_cfg,
        selector_type="greedy_channel",
        max_dh=10.0,
        episode_len=50,
        run_full_fl=True,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        dict_users=dict_users,
        net_glob=net_glob,
    )



    # ---- Dimensions ----
    obs0, state0, _ = env.reset(seed=0)
    N = obs0.shape[0]
    obs_dim = obs0.shape[1]
    state_dim = state0.shape[0]
    act_dim = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(obs_dim, act_dim).to(device)
    critic = Critic(state_dim).to(device)

    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)

    # PPO hyperparams
    gamma = 0.99
    lam = 0.95
    clip_eps = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    ppo_epochs = 5
    batch_size = 256

    buf = Buffer()

    num_episodes = 200

    for ep in range(num_episodes):
        obs_n, state, _ = env.reset(seed=ep)
        done = False

        # Rollout
        while not done:
            obs_t = torch.tensor(obs_n, dtype=torch.float32, device=device)          # (N, obs_dim)
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # (1, state_dim)

            with torch.no_grad():
                mu, std = actor(obs_t)  # (N,1), (1,)
                dist = torch.distributions.Normal(mu, std)
                act = dist.sample()
                logp = dist.log_prob(act).sum(dim=-1)  # (N,)
                # team value
                v = critic(state_t).item()

            act_np = act.cpu().numpy().astype(np.float32).reshape(N,)
            next_obs_n, next_state, rew_n, done, info = env.step(act_np)

            # team reward (same for all)
            team_rew = float(np.mean(rew_n))

            buf.add(
                obs=obs_n.copy(),
                state=state.copy(),
                act=act_np.copy(),
                logp=logp.cpu().numpy().copy(),
                rew=team_rew,
                done=done,
                val=v
            )

            obs_n, state = next_obs_n, next_state

        # Compute advantages/returns (team-based)
        adv, ret = gae_advantages(buf.rew, buf.done, buf.val, gamma=gamma, lam=lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Flatten data over time * agents for actor
        T = len(buf.rew)
        obs_arr = np.array(buf.obs, dtype=np.float32)            # (T, N, obs_dim)
        act_arr = np.array(buf.act, dtype=np.float32)            # (T, N)
        logp_arr = np.array(buf.logp, dtype=np.float32)          # (T, N)
        state_arr = np.array(buf.state, dtype=np.float32)        # (T, state_dim)

        # Actor dataset: (T*N, obs_dim), (T*N, act_dim), (T*N,), with per-time adv expanded to all agents
        obs_flat = obs_arr.reshape(T * N, obs_dim)
        act_flat = act_arr.reshape(T * N, 1)
        logp_flat = logp_arr.reshape(T * N)
        adv_flat = np.repeat(adv, N).astype(np.float32)

        # Critic dataset: (T, state_dim), ret (T,)
        states_t = torch.tensor(state_arr, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_flat, dtype=torch.float32, device=device)

        # Shuffle indices for actor
        idx = np.arange(T * N)
        np.random.shuffle(idx)

        # PPO updates
        for _ in range(ppo_epochs):
            # actor minibatches
            for start in range(0, T * N, batch_size):
                mb = idx[start:start + batch_size]
                mb_obs = torch.tensor(obs_flat[mb], dtype=torch.float32, device=device)
                mb_act = torch.tensor(act_flat[mb], dtype=torch.float32, device=device)
                mb_oldlogp = torch.tensor(logp_flat[mb], dtype=torch.float32, device=device)
                mb_adv = adv_t[mb]

                mu, std = actor(mb_obs)
                dist = torch.distributions.Normal(mu, std)
                newlogp = dist.log_prob(mb_act).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(newlogp - mb_oldlogp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                actor_loss = -(torch.min(surr1, surr2)).mean() - ent_coef * entropy

                opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                opt_actor.step()

            # critic update (use full batch, or mini-batch if you want)
            v_pred = critic(states_t)
            critic_loss = ((v_pred - ret_t) ** 2).mean()

            opt_critic.zero_grad()
            (vf_coef * critic_loss).backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            opt_critic.step()

        avg_rew = float(np.mean(buf.rew))
        print(f"EP {ep:03d} | avg team reward: {avg_rew:.3f}")

        buf.clear()

    # Save policy
    # --- robust saving (absolute path) ---
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    CKPT_DIR = os.path.join(PROJECT_ROOT, "marl", "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    actor_path = os.path.join(CKPT_DIR, "actor_phase1.pt")
    critic_path = os.path.join(CKPT_DIR, "critic_phase1.pt")
    
    torch.save(actor.state_dict(), actor_path)
    torch.save(critic.state_dict(), critic_path)
    
    print("Saved checkpoints to:")
    print("  ", actor_path)
    print("  ", critic_path)



if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-

