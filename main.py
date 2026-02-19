from utils.options import args_parser
from utils.sampling_func import DataPartitioner
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.Nets import CNNMnist,CNN60K
from models.evaluation import test_model
from UE_Selection.selectors import RandomSelector, GreedyChannelSelector, MARLSelector

import copy
import torch
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from UE_Selection.UAV_scenario import init_circular_xy_trajectory,init_predefined_height_trajectory,  init_altitudes #update_altitudes
from UE_Selection.atg_channel import elevation_angle, plos, snr_from_pathloss_db, avg_pathloss_db
from models.Nets import ResNetCifar
import os
import matplotlib.pyplot as plt

from marl.train.train_mappo_phase1 import Actor
from marl.utils.features import normalize_angle_deg, normalize_db, normalize_pos

def build_obs_main(x_uav, y_uav, h_uav, d, theta, P_LoS, PL_db, snr_db, prev_success,
                   h_min, h_max, use_agent_id=True):
    x = normalize_pos(x_uav, scale=500.0)
    y = normalize_pos(y_uav, scale=500.0)
    h = (h_uav - h_min) / (h_max - h_min + 1e-8)
    d_n = np.clip(d / 2000.0, 0.0, 1.0)
    th = normalize_angle_deg(theta)
    plos = np.clip(P_LoS, 0.0, 1.0)
    pl_db_n = normalize_db(PL_db, lo=60.0, hi=160.0)
    snr_db_n = normalize_db(snr_db, lo=-20.0, hi=40.0)
    prev = np.clip(prev_success, 0.0, 1.0)

    obs = np.stack([x, y, h, d_n, th, plos, pl_db_n, snr_db_n, prev], axis=1).astype(np.float32)

    if use_agent_id:
        N = obs.shape[0]
        agent_id = (np.arange(N, dtype=np.float32) / max(1, N - 1)).reshape(-1, 1)
        obs = np.concatenate([obs, agent_id], axis=1).astype(np.float32)

    return obs



def plot_uav_xy(x_uav, y_uav, x_bs=0.0, y_bs=0.0, round_id=None):
    plt.figure(figsize=(6,6))
    plt.scatter(x_uav, y_uav, c='blue', label='UAVs')
    plt.scatter([x_bs], [y_bs], c='red', marker='^', s=120, label='BS')

    for i in range(len(x_uav)):
        plt.text(x_uav[i]+3, y_uav[i]+3, str(i), fontsize=8)

    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    title = "UAV Positions (Top-Down View)"
    if round_id is not None:
        title += f" – Round {round_id}"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    args.use_agent_id = True

    actor = None
    if args.policy == "mappo":
        obs_dim = 10
        actor = Actor(obs_dim=obs_dim, act_dim=1)
        actor.load_state_dict(torch.load("marl/checkpoints/actor_phase1.pt",map_location=args.device,weights_only=True))
        actor.to(args.device)
        actor.eval()

    if args.dataset == 'mnist':
        trans = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans)


    elif args.dataset == 'cifar10':
        trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
        
        dataset_train = datasets.CIFAR10('./data/cifar10/', train=True, download=True, transform=trans)
        dataset_test = datasets.CIFAR10('./data/cifar10/', train=False, download=True, transform=trans)
        
        args.num_channels = 3
        args.num_classes = 10



    partition_obj = DataPartitioner(dataset_train, args.total_UE, NonIID=args.iid, alpha=args.alpha)
    dict_users, _ = partition_obj.use() #Each client gets indices of MNIST samples.

    if args.model == 'cnn':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn60k':
        net_glob = CNN60K(args=args).to(args.device)
    elif args.model == 'resnet':
        net_glob = ResNetCifar(num_classes=args.num_classes).to(args.device)
    else:
        raise ValueError("Unknown model type")
        
    net_glob.train()

    # ---------------- Logging ----------------
    log = {
        'round': [],
        'test_acc': [],
        'test_loss': [],
        'avg_pl_selected': [],
        'num_selected': [],
        'num_success': []
    }
# ---- Initialize scenario (outside the FL loop) ----
    x_bs, y_bs, h_bs = 0.0, 0.0, 20.0          # BS location
    h_min, h_max = 80, 500                # UAV altitude bounds
    args.h_min = h_min
    args.h_max = h_max


# ---- Predefined UAV trajectories (x, y, z) ----
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
        h_min=h_min,
        h_max=h_max,
        seed=args.seed if hasattr(args, 'seed') else 0
    )

    
    
    # Channel parameters (highrise urban example)
    env_cfg = ENV_PARAMS[args.env]
    a, b = env_cfg['a'], env_cfg['b']
    eta1_db, eta2_db = env_cfg['eta1_db'], env_cfg['eta2_db']
    
    print(f"[Environment] {args.env} | a={a}, b={b}, eta_LoS={eta1_db}, eta_NLoS={eta2_db}")

    
    fc = 2e9                  # 2 GHz
    # alpha = 2.0               # pathloss exponent
    # Transmit power and noise (dBm)
    P_tx_dbm = 30.0      # 100 mW UAV uplink
    noise_dbm = -97  # thermal noise + NF
    
    
    # ---------------- Selector ----------------
    if args.method == 'random':
        selector = RandomSelector()
    elif args.method == 'greedy_channel':
        selector = GreedyChannelSelector()
    elif args.method == 'marl':
        selector = MARLSelector(marl_agent=None)
    else:
        raise ValueError("Unknown selection method")
        
    prev_success = np.zeros(args.total_UE, dtype=np.float32)

    # initialize altitude for round 0
    h_uav = traj_h_base[0].copy()

    # ---- FL learning loop ----
    for r in range(args.round):
        x_uav = traj_x[r]
        y_uav = traj_y[r]
    
        # If fixed policy, override altitude for this round
        if args.policy == "fixed":
            h_uav = traj_h_base[r].copy()
    
        # 1) Compute channel using current h_uav (needed to build obs for MAPPO)
        theta, d = elevation_angle(x_bs, y_bs, h_bs, x_uav, y_uav, h_uav)
        P_LoS = plos(theta, a, b)
        PL_db = avg_pathloss_db(d, P_LoS, fc, eta1_db, eta2_db)
        snr_db = snr_from_pathloss_db(P_tx_dbm, PL_db, noise_dbm)
    
        # 2) Apply altitude control (random or MAPPO) AFTER we have channel features
        if args.policy == "random":
            dh = np.random.uniform(-1, 1, size=args.total_UE) * 10.0
            h_uav = np.clip(h_uav + dh, args.h_min, args.h_max)
    
        elif args.policy == "mappo":
            obs = build_obs_main(
                x_uav, y_uav, h_uav, d, theta, P_LoS, PL_db, snr_db,
                prev_success, args.h_min, args.h_max,
                use_agent_id=args.use_agent_id
            )

            obs_t = torch.tensor(obs, dtype=torch.float32, device=args.device)
    
            with torch.no_grad():
                mu, _ = actor(obs_t)  # [-1,1]
                dh = mu.detach().cpu().numpy().reshape(-1) * 10.0
    
            h_uav = np.clip(h_uav + dh, args.h_min, args.h_max)
    
            # 3) Recompute channel AFTER MAPPO changed altitude (this is what matters for comm/FL)
            theta, d = elevation_angle(x_bs, y_bs, h_bs, x_uav, y_uav, h_uav)
            P_LoS = plos(theta, a, b)
            PL_db = avg_pathloss_db(d, P_LoS, fc, eta1_db, eta2_db)
            snr_db = snr_from_pathloss_db(P_tx_dbm, PL_db, noise_dbm)
    
        # 4) Now compute success probability based on the FINAL snr_db
        if args.wireless_on:
            p_succ = (snr_db >= args.snr_th).astype(float)
        else:
            p_succ = np.ones_like(snr_db)

        # ---- DEBUG (put it HERE) ----
        print(f"\n[Round {r:02d}] Per-UAV Channel Stats:")
        print("UAV |   x (m)  |   y (m)  | Height (m) | Elevation (deg) |  P_LoS  |  PL_avg (dB) |  SNR (dB)")
        print("-" * 95)
        
        for i in range(args.total_UE):
            print(f"{i:3d} | "
                      f"{x_uav[i]:8.2f} | "
                      f"{y_uav[i]:8.2f} | "
                      f"{h_uav[i]:10.2f} | "
                      f"{theta[i]:15.2f} | "
                      f"{P_LoS[i]:7.3f} | "
                      f"{PL_db[i]:12.2f} | "
                      f"{snr_db[i]:9.2f}")
      
        # This is to plot the positions of the UAVs
        # plot_uav_xy(x_uav, y_uav, x_bs, y_bs, round_id=r)
        # Client selection
        idxs_users = selector.select(PL_db, args.active_UE)
        successful_users = [idx for idx in idxs_users if p_succ[idx] > 0.0]
        prev_success = np.zeros(args.total_UE, dtype=np.float32)
        
        for idx in successful_users:
            prev_success[idx] = 1.0

        # Local training
        w_locals = []
        for idx in successful_users:
           local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
           w, loss_local = local.train(net=copy.deepcopy(net_glob))
           w_locals.append(copy.deepcopy(w))

        if len(w_locals) > 0:
           w_glob = FedAvg(w_locals)
           net_glob.load_state_dict(w_glob)
        else:
           print(f"Round {r:02d} | No successful uploads (severe channel outage)")
         
        # Evaluation
        acc, loss = test_model(net_glob, dataset_test, args)

        # Logging
        log['round'].append(r)
        log['test_acc'].append(acc)
        log['test_loss'].append(loss)
        log['avg_pl_selected'].append(np.mean(PL_db[idxs_users]))
        log['num_selected'].append(len(idxs_users))
        log['num_success'].append(len(successful_users))

        print(f"Round {r:02d} | Method: {args.method} | Success: {len(successful_users)}/{args.active_UE} | "
              f"Test Acc: {acc:.2f}% | Test Loss: {loss:.4f}")

    

    # ---------------- Save Logs ----------------
    os.makedirs("results", exist_ok=True)
    
    data_mode = args.iid   # 'iid' or 'dirichlet'
    env_tag = args.env
    save_name = f"results/{args.method}_{data_mode}_{env_tag}_wireless{args.wireless_on}.npy"
    
    np.save(save_name, log)
    print(f"[Saved logs to {save_name}]")


    # ---------------- Plots ----------------
    plt.figure()
    plt.plot(log['round'], log['test_acc'], marker='o')
    plt.xlabel("FL Round")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_rounds.png")

    plt.figure()
    plt.plot(log['round'], log['test_loss'], marker='o')
    plt.xlabel("FL Round")
    plt.ylabel("Test Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_vs_rounds.png")

if __name__ == '__main__':
    main()
