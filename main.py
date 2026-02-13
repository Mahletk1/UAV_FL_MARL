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
from UE_Selection.UAV_scenario import init_uav_positions, init_altitudes #update_altitudes
from UE_Selection.atg_channel import elevation_angle, plos, snr_from_pathloss_db, avg_pathloss_db
from models.Nets import ResNetCifar
import os

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
    x_bs, y_bs, h_bs = 0.0, 0.0, 25.0          # BS location
    h_min, h_max = 50.0, 600.0                # UAV altitude bounds
    
    
    
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
        
    # ---- FL learning loop ----
    for r in range(args.round):
        x_uav, y_uav = init_uav_positions(args.total_UE)
        h_uav = init_altitudes(args.total_UE, h_min, h_max)
        # (Optional for now) random altitude change to simulate mobility
        # actions = np.random.uniform(-5, 5, size=args.total_UE)
        # h_uav = update_altitudes(h_uav, actions, h_min, h_max)
    
        # 1) Update ATG channel
        theta, d = elevation_angle(x_bs, y_bs, h_bs, x_uav, y_uav, h_uav)
        P_LoS = plos(theta, a, b)
        PL_db = avg_pathloss_db(d, P_LoS, fc, eta1_db, eta2_db)
        snr_db = snr_from_pathloss_db(P_tx_dbm, PL_db, noise_dbm)
        if args.wireless_on:
            p_succ = (snr_db >= args.snr_th).astype(float)
        else:
            p_succ = np.ones_like(snr_db)
        # ---- DEBUG (put it HERE) ----
        # print(f"\n[Round {r:02d}] Per-UAV Channel Stats:")
        # print("UAV |   x (m)  |   y (m)  | Height (m) | Elevation (deg) |  P_LoS  |  PL_avg (dB) |  SNR (dB)")
        # print("-" * 95)
        
        # for i in range(args.total_UE):
        #     print(f"{i:3d} | "
        #               f"{x_uav[i]:8.2f} | "
        #               f"{y_uav[i]:8.2f} | "
        #               f"{h_uav[i]:10.2f} | "
        #               f"{theta[i]:15.2f} | "
        #               f"{P_LoS[i]:7.3f} | "
        #               f"{PL_db[i]:12.2f} | "
        #               f"{snr_db[i]:9.2f}")
      
     
        # Client selection
        idxs_users = selector.select(PL_db, args.active_UE)
        successful_users = [idx for idx in idxs_users if p_succ[idx] > 0.0]
        
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
