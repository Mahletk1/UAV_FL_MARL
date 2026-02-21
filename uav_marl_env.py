# uav_marl_env.py
import numpy as np
from UE_Selection.atg_channel import elevation_angle, plos, avg_pathloss_db, snr_from_pathloss_db

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class UAVScoreEnv:
    """
    Wireless-only environment for Option B (Score-based scheduling + altitude control).
    Actions per UAV: (delta_h_i, score_i)
    BS selects Top-K by score each step.
    Reward = reliability + fairness + small-data - energy penalty
    """
    def __init__(self, args, traj_x, traj_y, env_params,
                 x_bs=0.0, y_bs=0.0, h_bs=20.0,
                 fc=2e9, P_tx_dbm=30.0, noise_dbm=-97.0):

        self.args = args
        self.traj_x = traj_x  # shape [T, N]
        self.traj_y = traj_y  # shape [T, N]
        self.env_params = env_params

        self.x_bs, self.y_bs, self.h_bs = x_bs, y_bs, h_bs
        self.fc = fc
        self.P_tx_dbm = P_tx_dbm
        self.noise_dbm = noise_dbm

        self.N = args.total_UE
        self.K = args.active_UE
        self.T = args.episode_len

        self.h_min = args.h_min
        self.h_max = args.h_max
        self.dh_max = args.delta_h_max

        # Per-UAV static info (set after you build dict_users)
        self.data_ratio = np.zeros(self.N, dtype=np.float32)

        self.reset()

    def set_data_ratio(self, data_ratio):
        self.data_ratio = data_ratio.astype(np.float32)

    def reset(self):
        self.t = 0
        # Start mid-altitude for stability
        # self.h = np.ones(self.N, dtype=np.float32) * (0.5 * (self.h_min + self.h_max))
        self.h = np.random.uniform(self.h_min, self.h_max, size=self.N).astype(np.float32)
        self.last_selected = np.zeros(self.N, dtype=np.float32)
        obs_n, state = self._build_obs_state()
        return obs_n, state

    def step(self, delta_h, scores):
        """
        delta_h: shape [N], meters (continuous)
        scores: shape [N], in [0,1]
        returns: obs_n_next, state_next, reward, done, info
        """
        # 1) altitude update (hard constraints)
        delta_h = np.clip(delta_h, -self.dh_max, self.dh_max).astype(np.float32)
        self.h = np.clip(self.h + delta_h, self.h_min, self.h_max).astype(np.float32)

        # 2) compute A2G metrics at current t
        t_idx = min(self.t, self.traj_x.shape[0] - 1)
        x_uav = self.traj_x[t_idx]
        y_uav = self.traj_y[t_idx]

        a = self.env_params["a"]
        b = self.env_params["b"]
        eta1_db = self.env_params["eta1_db"]
        eta2_db = self.env_params["eta2_db"]

        theta, d = elevation_angle(self.x_bs, self.y_bs, self.h_bs, x_uav, y_uav, self.h)
        P_LoS = plos(theta, a, b)
        PL_db = avg_pathloss_db(d, P_LoS, self.fc, eta1_db, eta2_db)
        snr_db = snr_from_pathloss_db(self.P_tx_dbm, PL_db, self.noise_dbm)

        # Smooth success probability (better learning signal than hard threshold)
        # q = _sigmoid((snr_db - self.args.snr_th) / self.args.snr_kappa).astype(np.float32)
        q = (snr_db >= self.args.snr_th).astype(np.float32)
       
        # score-weighted reliability over ALL UAVs
        eps = 1e-8
       
        
        
                # 3) BS selects Top-K by score
        idx = np.argsort(scores)[-self.K:]
        selected = np.zeros(self.N, dtype=np.float32)
        selected[idx] = 1.0
        
        # compute smallest half threshold once per episode
        threshold = np.percentile(self.data_ratio, 50)  # median
        
        small_mask = (self.data_ratio <= threshold).astype(np.float32)
        
        # 4) reward components
        R_rel = float(np.mean(q[idx]))
        R_fair = float(np.mean(1.0 - self.last_selected[idx]))
        R_small = float(np.mean(small_mask[idx]))
        # P_dh = float(np.mean((delta_h / (self.dh_max + 1e-8)) ** 2))
        R_score_rel = float(np.sum(scores * q) / (np.sum(scores) + eps))
        # --- movement penalty: penalize upward movement for NON-selected UAVs ---
        up = np.clip(delta_h, 0.0, None)  # positive part only
        non_selected = 1.0 - selected     # 1 if not selected
        
        P_up_non = float(np.mean((non_selected * (up / (self.dh_max + 1e-8))) ** 2))
        # penalize assigning high score to bad link
        P_mismatch = float(np.sum(scores * (1.0 - q)) / (np.sum(scores) + eps))
          
        
        reward = (
            7.0 * R_rel
          + 1.3 * R_fair
          + 0.7 * R_small
          + 1.5 * R_score_rel
          - 1.5 * P_mismatch
          - 1 * P_up_non
        )
        # reward = (
        #     6.0 * R_rel
        #   + 1.0 * R_fair
        #   + 0.7 * R_small
        #   + 1.5 * R_score_rel
        #   - 1.5 * P_mismatch
        #   - 0.8 * P_up_non
        # )
        # print("fairness", R_fair)
        # reward = 2 * R_rel + 1.5 * R_fair + 1 * R_small - 0.5 * P_dh

        # 5) update history + time
        self.last_selected = selected
        self.t += 1
        done = (self.t >= self.T)
        
        obs_n, state = self._build_obs_state()
        info = {
            "R_rel": R_rel, "R_fair": R_fair, "R_small": R_small, "P_up_non": P_up_non,
            "mean_snr_db": float(np.mean(snr_db)),
            "mean_h": float(np.mean(self.h))
        }
        return obs_n, state, reward, done, info

    def _build_obs_state(self):
        # If we are at t==T (done), clamp index to last valid for obs.
        t_idx = min(self.t, self.traj_x.shape[0] - 1)

        x_uav = self.traj_x[t_idx]
        y_uav = self.traj_y[t_idx]

        a = self.env_params["a"]
        b = self.env_params["b"]
        eta1_db = self.env_params["eta1_db"]
        eta2_db = self.env_params["eta2_db"]

        theta, d = elevation_angle(self.x_bs, self.y_bs, self.h_bs, x_uav, y_uav, self.h)
        P_LoS = plos(theta, a, b)
        PL_db = avg_pathloss_db(d, P_LoS, self.fc, eta1_db, eta2_db)
        snr_db = snr_from_pathloss_db(self.P_tx_dbm, PL_db, self.noise_dbm)

        # Normalizations (simple + stable)
        h_norm = (self.h - self.h_min) / (self.h_max - self.h_min + 1e-8)
        d_norm = d / (np.max(d) + 1e-8)
        theta_norm = theta / 90.0
        snr_norm = np.clip((snr_db + 20.0) / 60.0, 0.0, 1.0)

        obs_n = np.stack(
            [h_norm, d_norm, theta_norm, snr_norm, self.last_selected, self.data_ratio],
            axis=1
        ).astype(np.float32)

        # Centralized critic state = concat all agent obs
        state = obs_n.reshape(-1).astype(np.float32)
        return obs_n, state
