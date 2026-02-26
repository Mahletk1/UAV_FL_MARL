# envs/env_core.py
import numpy as np
from UE_Selection.atg_channel import elevation_angle, plos, avg_pathloss_db, snr_from_pathloss_db
from UE_Selection.UAV_scenario import init_altitudes, init_random_walk_xy_trajectory

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
        self.K = 10             # active_UE (Top-K)
        self.T = 100            # episode length

        self.h_min = 100.0
        self.h_max = 500.0
        self.dh_max = 20.0

        # Use same meaning as args.snr_th
        self.snr_th = 20.0

        self.env_params = ENV_PARAMS["highrise"]

        self.obs_dim = 6
        self.action_dim = 2  # (delta_h, score)

        # BS + channel settings
        self.x_bs, self.y_bs, self.h_bs = 0.0, 0.0, 20.0
        self.fc = 2e9
        self.P_tx_dbm = 30.0
        self.noise_dbm = -97.0

        # seed & mobility
        self.seed_val = 2
        np.random.seed(self.seed_val)

        self.traj_x, self.traj_y = init_random_walk_xy_trajectory(
            N=self.agent_num, T=self.T, area_size=500.0, step_std=25.0, seed=self.seed_val
        )

        # data ratio (replace with real dataset ratios if you want)
        self.data_ratio = np.ones(self.agent_num, dtype=np.float32) / self.agent_num

        self.reset()

    def seed(self, seed):
        self.seed_val = int(seed)
        np.random.seed(self.seed_val)

    def reset(self):
        self.t = 0
        self.h = init_altitudes(self.agent_num, self.h_min, self.h_max, seed=self.seed_val).astype(np.float32)
        self.last_selected = np.zeros(self.agent_num, dtype=np.float32)

        obs_n = self._build_obs()
        return [obs_n[i].astype(np.float32) for i in range(self.agent_num)]

    def step(self, actions):
        """
        actions: list length N, each action is shape (2,)
          action[i,0] -> delta_h control (we interpret as normalized in [-1,1])
          action[i,1] -> score control (we interpret as normalized in [-1,1])
        """
        a = np.asarray(actions, dtype=np.float32)  # [N,2]

        # --- map policy outputs to your env variables ---
        # delta_h in meters, clipped by dh_max
        delta_h = np.clip(a[:, 0], -1.0, 1.0) * self.dh_max
        # score in [0,1]
        scores = (np.clip(a[:, 1], -1.0, 1.0) + 1.0) * 0.5

        # 1) altitude update
        self.h = np.clip(self.h + delta_h, self.h_min, self.h_max).astype(np.float32)

        # 2) compute A2G metrics at current t
        t_idx = min(self.t, self.traj_x.shape[0] - 1)
        x_uav = self.traj_x[t_idx]
        y_uav = self.traj_y[t_idx]

        a_env = self.env_params["a"]
        b_env = self.env_params["b"]
        eta1_db = self.env_params["eta1_db"]
        eta2_db = self.env_params["eta2_db"]

        theta, d = elevation_angle(self.x_bs, self.y_bs, self.h_bs, x_uav, y_uav, self.h)
        P_LoS = plos(theta, a_env, b_env)
        PL_db = avg_pathloss_db(d, P_LoS, self.fc, eta1_db, eta2_db)
        snr_db = snr_from_pathloss_db(self.P_tx_dbm, PL_db, self.noise_dbm)

        # q in {0,1} like your previous code
        q = (snr_db >= self.snr_th).astype(np.float32)
        q_hard = q.copy()

        # 3) BS selects Top-K by score
        idx = np.argsort(scores)[-self.K:]
        selected = np.zeros(self.agent_num, dtype=np.float32)
        selected[idx] = 1.0

        # 4) fairness/bias helper masks
        threshold = np.percentile(self.data_ratio, 50)
        small_mask = (self.data_ratio <= threshold).astype(np.float32)

        # 5) energy / altitude constraint (PER-AGENT, not mean)
        h_norm = (self.h - self.h_min) / (self.h_max - self.h_min + 1e-8)
        up = np.clip(delta_h, 0.0, None)
        P_eng_i = (0.2 * (h_norm ** 2) + 0.1 * (up / (self.dh_max + 1e-8)) ** 2).astype(np.float32)

        # -------- PER-AGENT reward that matches your macro terms --------
        # Reliability (selected should succeed)
        r_rel_i = selected * q  # [N]

        # Alignment (scores align with channel quality)
        # same expression you had, but not averaged
        r_align_i = scores * (2.0 * q - 1.0)  # [N]

        # Fairness: reward selecting someone not selected last time
        r_fair_i = selected * (1.0 - self.last_selected)  # [N]

        # Small-data inclusion: reward selecting "small" clients
        r_small_i = selected * small_mask  # [N]

        # You previously did: R_fair_macro = R_fair  # + R_small
        # So keep that default behavior:
        use_small = False
        r_fair_macro_i = r_fair_i + (r_small_i if use_small else 0.0)

        # Final per-agent reward:
        reward_n = (
            6 * (r_rel_i + 1.5 * r_align_i)
            + 1.2 * r_fair_macro_i
            - 1 * P_eng_i
        ).astype(np.float32)

        # -------- logging macros (same as your previous info dict) --------
        R_rel = float(np.mean(q[idx]))
        R_align = float(np.mean(scores * (2.0 * q - 1.0)))
        R_fair = float(np.mean(1.0 - self.last_selected[idx]))
        R_small = float(np.mean(small_mask[idx]))
        P_eng = float(np.mean(P_eng_i))

        # 6) update history + time
        self.last_selected = selected
        self.t += 1
        done = (self.t >= self.T)

        obs_n = self._build_obs()
        obs_list = [obs_n[i].astype(np.float32) for i in range(self.agent_num)]
        rew_list = [[float(reward_n[i])] for i in range(self.agent_num)]
        done_list = [bool(done) for _ in range(self.agent_num)]

        info = {
            "R_rel": R_rel,
            "R_align": R_align,
            "R_fair": R_fair,
            "R_small": R_small,
            "P_eng": P_eng,
            "mean_snr_db": float(np.mean(snr_db)),
            "mean_q_hard_selected": float(np.mean(q_hard[idx])),
            "mean_h": float(np.mean(self.h)),
        }
        info_list = [info for _ in range(self.agent_num)]
       
            
        return [obs_list, rew_list, done_list, info_list]

    def _build_obs(self):
        t_idx = min(self.t, self.traj_x.shape[0] - 1)
        x_uav = self.traj_x[t_idx]
        y_uav = self.traj_y[t_idx]

        a_env = self.env_params["a"]
        b_env = self.env_params["b"]
        eta1_db = self.env_params["eta1_db"]
        eta2_db = self.env_params["eta2_db"]

        theta, d = elevation_angle(self.x_bs, self.y_bs, self.h_bs, x_uav, y_uav, self.h)
        P_LoS = plos(theta, a_env, b_env)
        PL_db = avg_pathloss_db(d, P_LoS, self.fc, eta1_db, eta2_db)
        snr_db = snr_from_pathloss_db(self.P_tx_dbm, PL_db, self.noise_dbm)

        # Normalizations (same idea as your previous code)
        h_norm = (self.h - self.h_min) / (self.h_max - self.h_min + 1e-8)
        d_norm = d / (np.max(d) + 1e-8)
        theta_norm = theta / 90.0
        snr_norm = np.clip((snr_db + 20.0) / 60.0, 0.0, 1.0)

        obs_n = np.stack(
            [h_norm, d_norm, theta_norm, snr_norm, self.last_selected, self.data_ratio],
            axis=1
        ).astype(np.float32)
        return obs_n