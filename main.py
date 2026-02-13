from utils.options import args_parser
from utils.sampling_func import DataPartitioner
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.Nets import CNNMnist
from models.evaluation import test_model

import copy
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from UE_Selection.UAV_scenario import init_uav_positions, init_altitudes, update_altitudes
from UE_Selection.atg_channel import elevation_angle, plos, snr_from_pathloss_db, avg_pathloss_db

def main():
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    trans = transforms.Compose([transforms.ToTensor()])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans)
    dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans)

    partition_obj = DataPartitioner(dataset_train, args.total_UE, NonIID=args.iid, alpha=args.alpha)
    dict_users, _ = partition_obj.use() #Each client gets indices of MNIST samples.

    net_glob = CNNMnist(args=args).to(args.device)
    net_glob.train()

    acc_list = []
    loss_list = []
    
# ---- Initialize scenario (outside the FL loop) ----
    x_bs, y_bs, h_bs = 0.0, 0.0, 25.0          # BS location
    h_min, h_max = 50.0, 600.0                # UAV altitude bounds
    
    
    
    # Channel parameters (highrise urban example)
    # Channel parameters (DEBUG-FRIENDLY)
    a, b = 27.23, 0.08       # highrise-urban (Al-Hourani)
    fc = 2e9                  # 2 GHz
    # alpha = 2.0               # pathloss exponent
    # Transmit power and noise (dBm)
    P_tx_dbm = 30.0      # 100 mW UAV uplink
    noise_dbm = -97  # thermal noise + NF
    
    # Path loss parameters
    # alpha = 2.2
    eta1_db = 2.3     # LoS excess loss
    eta2_db = 34     # NLoS excess loss
    fc = 2e9            # 2 GHz
    
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
        p_succ = (snr_db >= 20.0).astype(float)   # threshold in dB
        print(p_succ)
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
      
     
    
        if args.mode == 'random_selection':
            # 2) Resource-constrained scheduling (M < N)
            idxs_users = np.random.choice(range(args.total_UE), args.active_UE, replace=False)
            successful_users = [idx for idx in idxs_users if p_succ[idx] > 0.0]
        
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
        
            success_count = len(successful_users)

        # elif args.mode == 'ideal':
        #     w_locals = []
        #     for idx in idxs_users:
        #         local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
        #         w, loss_local = local.train(net=copy.deepcopy(net_glob))
        #         w_locals.append(copy.deepcopy(w))
        
        #     w_glob = FedAvg(w_locals)
        #     net_glob.load_state_dict(w_glob)
        #     success_count = len(idxs_users)
        
        #evaluate after update (or no-update)
        acc, loss = test_model(net_glob, dataset_test, args)
        acc_list.append(acc)
        loss_list.append(loss)

        print(f"Round {r:02d} | Mode: {args.mode} | Success: {success_count}/{args.active_UE} | "f"Test Acc: {acc:.2f}% | Test Loss: {loss:.4f}")

    # ---- Plot Accuracy ----
    plt.figure()
    plt.plot(range(len(acc_list)), acc_list, marker='o')
    plt.xlabel("FL Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Global Model Accuracy vs Rounds (FedAvg)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_rounds.png")
    plt.show()

    # ---- Plot Loss ----
    plt.figure()
    plt.plot(range(len(loss_list)), loss_list, marker='o')
    plt.xlabel("FL Round")
    plt.ylabel("Test Loss")
    plt.title("Global Model Loss vs Rounds (FedAvg)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_vs_rounds.png")
    plt.show()

if __name__ == '__main__':
    main()
