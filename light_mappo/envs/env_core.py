# envs/env_core.py
import numpy as np
from UE_Selection.atg_channel import elevation_angle, plos, avg_pathloss_db, snr_from_pathloss_db, snr_rayleigh_from_pathloss_db
from UE_Selection.UAV_scenario import init_altitudes, init_random_walk_xy_trajectory
from torchvision import datasets, transforms
from utils1.sampling_func import DataPartitioner

ENV_PARAMS = {
    "suburban": {"a": 4.88, "b": 0.43, "eta1_db": 0.1, "eta2_db": 21},
    "urban": {"a": 9.61, "b": 0.16, "eta1_db": 1.0, "eta2_db": 20},
    "denseurban": {"a": 12.08, "b": 0.11, "eta1_db": 1.6, "eta2_db": 23},
    "highrise": {"a": 27.23, "b": 0.08, "eta1_db": 2.3, "eta2_db": 34},
}

class EnvCore(object):
    """
    light_mappo EnvCore API:
      reset() -> list length N, each obs shape (obs_dim,)
      step(actions) -> [obs_list, reward_list, done_list, info_list]
        where reward_list is length N and each element is [scalar]
    """

    def __init__(self):
        # -------- HARD-CODED SETTINGS (adjust as needed) --------
        self.agent_num = 20     # N = total_UE
        self.K = 10            # active_UE (Top-K)
        self.T = 100            # episode length

        self.h_min = 100.0
        self.h_max = 300.0
        self.dh_max = 10

        # Use same meaning as args.snr_th
        self.snr_th = 3

        self.env_params = ENV_PARAMS["highrise"]

        self.obs_dim = 6
        self.action_dim = 2  # (delta_h, score)

        # BS + channel settings
        self.x_bs, self.y_bs, self.h_bs = 0.0, 0.0, 25.0
        self.fc = 3.5e9
        self.P_tx_dbm = 23
        self.noise_dbm = -105

        # seed & mobility
        self.base_seed = 2
        self.episode_id = 0
        self.rng = np.random.default_rng(self.base_seed)

        # self.traj_x, self.traj_y = init_random_walk_xy_trajectory(
        #     N=self.agent_num, T=self.T, area_size=500.0, step_std=25.0, seed=self.seed_val
        # )

        # data ratio (replace with real dataset ratios if you want)
        trans = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.MNIST("../data/mnist", train=True, download=True, transform=trans)
        
        partition_obj = DataPartitioner(dataset_train, self.agent_num, NonIID="dirichlet", alpha=0.1)
        dict_users, _ = partition_obj.use()
        
        sizes = np.array([len(dict_users[i]) for i in range(self.agent_num)], dtype=np.float32)
        self.data_ratio = sizes / (sizes.sum() + 1e-8)
        print(self.data_ratio)
        # self.data_ratio = np.ones(self.agent_num, dtype=np.float32) / self.agent_num
        # print("this is dataset in env", self.data_ratio)
        # self.reset()

    def seed(self, seed):
        self.base_seed = int(seed)
        self.episode_id = 0
        self.rng = np.random.default_rng(self.base_seed)

    def reset(self):
        self.t = 0
        
        ep_seed = ((self.base_seed + self.episode_id) % 10) + 1
        # ep_seed = self.base_seed + self.episode_id
        self.episode_id += 1
        # print ("the seed used", ep_seed)
        # regenerate mobility per episode
        self.traj_x, self.traj_y = init_random_walk_xy_trajectory(
            N=self.agent_num, T=self.T, radius=500.0, step_std=25, seed=ep_seed )
        
        self.h = init_altitudes(self.agent_num, self.h_min, self.h_max, seed=ep_seed + 1).astype(np.float32)
        self.last_selected = np.zeros(self.agent_num, dtype=np.float32)
        self.sel_ema = np.zeros(self.agent_num, dtype=np.float32) 
        self.prev_delta_h = np.zeros(self.agent_num, dtype=np.float32)
        
        obs_n = self._build_obs()
        return [obs_n[i].astype(np.float32) for i in range(self.agent_num)]

    def step(self, actions):
        """
        actions: list length N, each action is shape (2,)
          action[i,0] -> delta_h control
          action[i,1] -> score control
        """
        a = np.asarray(actions, dtype=np.float32)  # [N,2]
    
        # ------------------------------------------------------------
        # 1) Decode policy outputs
        # ------------------------------------------------------------
        delta_h = np.clip(a[:, 0], -1.0, 1.0) * self.dh_max
        a1 = a[:, 1]
        scores = 0.5 * (np.tanh(a1) + 1.0)   # smooth score in [0,1]
    
        # ------------------------------------------------------------
        # 2) Current geometry at time t
        # ------------------------------------------------------------
        t_idx = min(self.t, self.traj_x.shape[0] - 1)
        x_uav = self.traj_x[t_idx]
        y_uav = self.traj_y[t_idx]
    
        a_env   = self.env_params["a"]
        b_env   = self.env_params["b"]
        eta1_db = self.env_params["eta1_db"]
        eta2_db = self.env_params["eta2_db"]
    
        # ------------------------------------------------------------
        # 4) Apply altitude update
        # ------------------------------------------------------------
        self.h = np.clip(self.h + delta_h, self.h_min, self.h_max).astype(np.float32)
    
        # ------------------------------------------------------------
        # 5) Reliability AFTER altitude update
        # ------------------------------------------------------------
        theta, d = elevation_angle(self.x_bs, self.y_bs, self.h_bs, x_uav, y_uav, self.h)
        P_LoS = plos(theta, a_env, b_env)
        PL_db = avg_pathloss_db(d, P_LoS, self.fc, eta1_db, eta2_db)
        snr_avg_db = snr_from_pathloss_db(self.P_tx_dbm, PL_db, self.noise_dbm)
    
        q_hard = (snr_avg_db >= self.snr_th).astype(np.float32)
        q_soft = 1.0 / (1.0 + np.exp(-(snr_avg_db - self.snr_th) / 2.0))
    
        # ------------------------------------------------------------
        # 6) Fairness deficit
        # ------------------------------------------------------------
        rho = 0.06
        p_star = self.K / float(self.agent_num)
        fair_def = np.clip(p_star - self.sel_ema, 0.0, 1.0).astype(np.float32)
    
        # ------------------------------------------------------------
        # 7) Score priority target
        #    fairness + data, gated by reliability BEFORE altitude move
        # ------------------------------------------------------------
        w_data = 0.4
        w_fair = 0.6
        priority = (w_data * self.data_ratio + w_fair * fair_def)
        priority = np.clip(priority, 0.0, 1.0)
        
    
        # ------------------------------------------------------------
        # 8) Top-K selection by score
        # ------------------------------------------------------------
        idx = np.argsort(scores)[-self.K:]
        selected = np.zeros(self.agent_num, dtype=np.float32)
        selected[idx] = 1.0
    
        # update fairness memory AFTER selection
        self.sel_ema = (1.0 - rho) * self.sel_ema + rho * selected
    
        # ------------------------------------------------------------
        # 9) Reward terms
        # ------------------------------------------------------------
    
        # (a) Score should reflect priority
        r_calib = - (scores - priority) ** 2
    
        # (b) Selected UAVs should be reliable
        r_rel_soft = selected * q_soft
        r_rel_hard = selected * (2.0 * q_hard - 1.0)
    
        # (c) Altitude action should improve reliability of selected UAVs
        # r_alt_gain = selected * (q_soft - q_soft_before)
    
        d_horizontal = np.sqrt((x_uav - self.x_bs) ** 2 + (y_uav - self.y_bs) ** 2)
        need = np.clip(d_horizontal / 600.0, 0.0, 1.0)
        h_target = self.h_min + need * (self.h_max - self.h_min)
        r_alt_gain = - ((self.h - h_target) / (self.h_max - self.h_min + 1e-8)) ** 2
           
        # ------------------------------------------------------------
        # 10) Final reward
        # ------------------------------------------------------------
        w_calib = 2.5
        w_rel_s = 1.5
        w_rel_h = 2
        w_alt   = 0.3
        # w_move  = 1.0
    
        reward_n = (
            w_calib * r_calib
            + w_rel_s * r_rel_soft
            + w_rel_h * r_rel_hard
            + w_alt   * r_alt_gain
        ).astype(np.float32)
    
        # ------------------------------------------------------------
        # 11) Update step / done
        # ------------------------------------------------------------
        self.last_selected = selected
        self.t += 1
        done = (self.t >= self.T)
    
        # ------------------------------------------------------------
        # 12) Build next obs
        # ------------------------------------------------------------
        obs_n = self._build_obs()
        obs_list = [obs_n[i].astype(np.float32) for i in range(self.agent_num)]
        rew_list = [[float(reward_n[i])] for i in range(self.agent_num)]
        done_list = [bool(done) for _ in range(self.agent_num)]
    
    
        info = {
            "mean_snr_db": float(np.mean(snr_avg_db)),
            "mean_h": float(np.mean(self.h)),
            "sel_success": float(np.mean(q_hard[idx])),
            "calib_mse": float(np.mean((scores - priority) ** 2)),
            "mean_priority_sel": float(np.mean(priority[idx])),
            "mean_alt_gain_sel": float(np.mean(r_alt_gain[idx])),
        }
        info_list = [info for _ in range(self.agent_num)]
    
        return [obs_list, rew_list, done_list, info_list]
    
    def _build_obs(self):
        t_idx = min(self.t, self.traj_x.shape[0] - 1)
        x_uav = self.traj_x[t_idx]
        y_uav = self.traj_y[t_idx]
    
        a_env    = self.env_params["a"]
        b_env    = self.env_params["b"]
        eta1_db  = self.env_params["eta1_db"]
        eta2_db  = self.env_params["eta2_db"]
    
        theta, d = elevation_angle(self.x_bs, self.y_bs, self.h_bs, x_uav, y_uav, self.h)
        P_LoS    = plos(theta, a_env, b_env)
        PL_db    = avg_pathloss_db(d, P_LoS, self.fc, eta1_db, eta2_db)
        snr_avg_db = snr_from_pathloss_db(self.P_tx_dbm, PL_db, self.noise_dbm)
    
        # normalizations
        h_norm = (self.h - self.h_min) / (self.h_max - self.h_min + 1e-8)
    
        d_max = np.sqrt((600.0**2 + 600.0**2) + (self.h_max - self.h_bs)**2)
        d_norm = d / (d_max + 1e-8)
    
        theta_norm = theta / 90.0
        snr_norm = np.clip((snr_avg_db + 20.0) / 60.0, 0.0, 1.0)
    
        obs_n = np.stack(
            [h_norm, d_norm, theta_norm, snr_norm, self.last_selected, self.data_ratio],
            axis=1
        ).astype(np.float32)
    
        return obs_n