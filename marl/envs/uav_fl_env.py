# marl/envs/uav_fl_env.py
import copy
import numpy as np
import torch

from UE_Selection.atg_channel import elevation_angle, plos, avg_pathloss_db, snr_from_pathloss_db
from UE_Selection.selectors import RandomSelector, GreedyChannelSelector
from marl.utils.features import (
    normalize_angle_deg, normalize_db, normalize_pos, clip_altitude
)

# Optional (only if you want to run full FL inside env.step; slow!)
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.evaluation import test_model


class UAVFLEnv:
    """
    Phase 1 MARL environment:
      - Multi-agent continuous control of UAV altitudes (Δh per agent).
      - Client selection stays as baseline (random or greedy_channel).
      - Reward is cooperative global reward (CTDE-friendly).

    CTDE:
      - obs_n: local obs per agent (used by decentralized actor)
      - state: global state vector (used by centralized critic)
    """

    def __init__(
        self,
        args,
        traj_x, traj_y, traj_h_base,
        env_params,  # dict: a,b,eta1_db,eta2_db
        x_bs=0.0, y_bs=0.0, h_bs=20.0,
        fc=2e9,
        P_tx_dbm=30.0,
        noise_dbm=-97.0,
        selector_type="random",  # or "random"
        max_dh=10.0,                    # meters per step
        episode_len=None,               # default = args.round
        run_full_fl=True,              # keep False for fast MARL training
        dataset_train=None,
        dataset_test=None,
        dict_users=None,
        net_glob=None,
        reward_weights=None,            # optional dict to override weights
    ):
        self.args = args
        self.N = args.total_UE
        self.K = args.active_UE

        self.traj_x = traj_x
        self.traj_y = traj_y
        self.traj_h_base = traj_h_base

        self.x_bs, self.y_bs, self.h_bs = x_bs, y_bs, h_bs

        self.a = env_params["a"]
        self.b = env_params["b"]
        self.eta1_db = env_params["eta1_db"]
        self.eta2_db = env_params["eta2_db"]

        self.fc = fc
        self.P_tx_dbm = P_tx_dbm
        self.noise_dbm = noise_dbm

        self.max_dh = float(max_dh)
        self.h_min = getattr(args, "h_min", 80)
        self.h_max = getattr(args, "h_max", 500)

        self.T = int(episode_len) if episode_len is not None else int(args.round)

        # ✅ FIX: correct selector_type branching
        if selector_type == "random":
            self.selector = RandomSelector()
        elif selector_type == "greedy_channel":
            self.selector = GreedyChannelSelector()
        else:
            raise ValueError("selector_type must be 'random' or 'greedy_channel' for Phase 1.")

        # Full FL inside env (slow). Keep off for training, use for evaluation later.
        self.run_full_fl = bool(run_full_fl)
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dict_users = dict_users
        self.net_glob = net_glob

        # ---------- Reward weights ----------
        # IMPORTANT: to prevent "everyone at h_max", energy must matter.
        # To encourage unbiased model training, fairness must matter too.
        self.w_rel = 1.0     # reliability/soft success of selected clients
        self.w_link = 0.15   # smooth SNR shaping on selected clients
        self.w_div = 0.10    # discourage all-equal heights
        self.w_act = 0.05    # discourage large jumps
        self.w_fair = 0.50   # ✅ fairness bonus (entropy)
        self.w_energy = 1.00 # ✅ energy penalty (make it strong enough)
        self.w_ceiling = 0.10  # optional extra penalty for sitting at h_max
        self.w_fl = 1.00     # only used when run_full_fl=True

        # --- Energy proxy ---
        self.c_hover = 1.0   # scales (h_norm^2)
        self.c_climb = 0.5   # scales (|Δh|/max_dh)
        self.E_max = getattr(args, "E_max", None)  # optional hard budget (can be None)

        # Optional: smooth success prob instead of hard threshold
        self.use_smooth_psucc = bool(getattr(args, "use_smooth_psucc", True))
        self.snr_smooth_kappa = float(getattr(args, "snr_smooth_kappa", 1.5))  # dB

        # Optional: append agent id to obs (helps symmetry breaking)
        self.use_agent_id = bool(getattr(args, "use_agent_id", True))

        if reward_weights is not None:
            self.w_rel = float(reward_weights.get("w_rel", self.w_rel))
            self.w_link = float(reward_weights.get("w_link", self.w_link))
            self.w_div  = float(reward_weights.get("w_div",  self.w_div))
            self.w_act  = float(reward_weights.get("w_act",  self.w_act))
            self.w_fair = float(reward_weights.get("w_fair", self.w_fair))
            self.w_energy = float(reward_weights.get("w_energy", self.w_energy))
            self.w_ceiling = float(reward_weights.get("w_ceiling", self.w_ceiling))
            self.w_fl = float(reward_weights.get("w_fl", self.w_fl))

            self.c_hover  = float(reward_weights.get("c_hover",  self.c_hover))
            self.c_climb  = float(reward_weights.get("c_climb",  self.c_climb))

        # Internal state
        self.r = 0
        self.h_uav = None
        self.prev_success = np.zeros(self.N, dtype=np.float32)
        self.last_rel = 0.0
        self.last_mean_snr = 0.0

        # fairness tracking (selection and success history)
        self.select_count = np.zeros(self.N, dtype=np.float32)
        self.success_count = np.zeros(self.N, dtype=np.float32)

        # energy tracking
        self.energy_used = np.zeros(self.N, dtype=np.float32)

        # Cache last channel stats (for obs)
        self.theta = None
        self.d = None
        self.P_LoS = None
        self.PL_db = None
        self.snr_db = None
        self.p_succ = None

        # For FL improvement term (only used when run_full_fl=True)
        self.prev_test_acc = None
        self.prev_test_loss = None

    # ---------- Core helpers ----------
    def _compute_channel(self, x_uav, y_uav, h_uav):
        theta, d = elevation_angle(self.x_bs, self.y_bs, self.h_bs, x_uav, y_uav, h_uav)
        P_LoS = plos(theta, self.a, self.b)
        PL_db = avg_pathloss_db(d, P_LoS, self.fc, self.eta1_db, self.eta2_db)
        snr_db = snr_from_pathloss_db(self.P_tx_dbm, PL_db, self.noise_dbm)

        if getattr(self.args, "wireless_on", True):
            if self.use_smooth_psucc:
                # Smooth success probability in (0,1)
                kappa = max(1e-6, self.snr_smooth_kappa)
                p_succ = 1.0 / (1.0 + np.exp(-(snr_db - self.args.snr_th) / kappa))
                p_succ = p_succ.astype(np.float32)
            else:
                p_succ = (snr_db >= self.args.snr_th).astype(np.float32)
        else:
            p_succ = np.ones_like(snr_db, dtype=np.float32)

        self.theta, self.d, self.P_LoS, self.PL_db, self.snr_db, self.p_succ = (
            theta, d, P_LoS, PL_db, snr_db, p_succ
        )

    def _build_obs(self):
        """
        Local obs per agent i:
          [x_i, y_i, h_i, d_i, theta_i, P_LoS_i, PL_db_i, snr_db_i, prev_success_i, (optional agent_id)]
        """
        x = normalize_pos(self.x_uav, scale=500.0)
        y = normalize_pos(self.y_uav, scale=500.0)

        h = (self.h_uav - self.h_min) / (self.h_max - self.h_min + 1e-8)
        h = np.clip(h, 0.0, 1.0)

        d = np.clip(self.d / 2000.0, 0.0, 1.0)
        th = normalize_angle_deg(self.theta)
        plos = np.clip(self.P_LoS, 0.0, 1.0)

        pl_db_n = normalize_db(self.PL_db, lo=60.0, hi=160.0)
        snr_db_n = normalize_db(self.snr_db, lo=-20.0, hi=40.0)

        prev = np.clip(self.prev_success, 0.0, 1.0)

        obs = np.stack([x, y, h, d, th, plos, pl_db_n, snr_db_n, prev], axis=1).astype(np.float32)

        if self.use_agent_id:
            agent_id = (np.arange(self.N, dtype=np.float32) / max(1, self.N - 1)).reshape(-1, 1)
            obs = np.concatenate([obs, agent_id], axis=1).astype(np.float32)

        return obs

    def _build_state(self, obs_n):
        """
        Global state for centralized critic:
          flatten all obs + [round_norm, K_norm, last_mean_snr_norm, last_rel, energy_mean_norm]
        """
        round_norm = np.array([self.r / max(1, self.T - 1)], dtype=np.float32)
        K_norm = np.array([self.K / max(1, self.N)], dtype=np.float32)
        mean_snr_norm = np.array([np.clip((self.last_mean_snr + 20.0) / 60.0, 0.0, 1.0)], dtype=np.float32)
        rel = np.array([self.last_rel], dtype=np.float32)

        energy_mean = float(np.mean(self.energy_used))
        energy_mean_norm = np.array([np.clip(energy_mean / 10.0, 0.0, 1.0)], dtype=np.float32)

        state = np.concatenate(
            [obs_n.reshape(-1), round_norm, K_norm, mean_snr_norm, rel, energy_mean_norm],
            axis=0
        ).astype(np.float32)
        return state

    def _snr_norm(self, snr_db):
        return np.clip((snr_db + 20.0) / 60.0, 0.0, 1.0)

    def _fairness_entropy_norm(self):
        """
        Normalized entropy of selection distribution:
          p_j = select_count_j / sum(select_count)
          H_norm = H(p)/log(N) in [0,1]
        High => fair (unbiased participation over time)
        """
        s = float(np.sum(self.select_count))
        if s <= 1e-6:
            return 1.0  # no selections yet => treat as perfectly fair initially
        p = self.select_count / (s + 1e-12)
        H = -float(np.sum(p * np.log(p + 1e-12)))
        H_norm = H / max(1e-12, np.log(self.N))
        return float(np.clip(H_norm, 0.0, 1.0))

    # ---------- RL API ----------
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.r = 0
        self.prev_success = np.zeros(self.N, dtype=np.float32)
        self.last_rel = 0.0
        self.last_mean_snr = 0.0

        self.select_count[:] = 0.0
        self.success_count[:] = 0.0
        self.energy_used[:] = 0.0

        self.prev_test_acc = None
        self.prev_test_loss = None

        self.x_uav = self.traj_x[self.r].astype(np.float32)
        self.y_uav = self.traj_y[self.r].astype(np.float32)

        self.h_uav = self.traj_h_base[self.r].astype(np.float32)
        self.h_uav = clip_altitude(self.h_uav, self.h_min, self.h_max)

        self._compute_channel(self.x_uav, self.y_uav, self.h_uav)

        obs_n = self._build_obs()
        state = self._build_state(obs_n)
        info = {}
        return obs_n, state, info

    def step(self, actions):
        """
        actions: np.ndarray shape (N,) or (N,1), continuous in [-1,1].
        Interpret as Δh scaled by max_dh:
          Δh = clip(actions, -1, 1) * max_dh
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 2 and actions.shape[1] == 1:
            actions = actions[:, 0]
        assert actions.shape[0] == self.N, f"Expected actions for {self.N} agents, got {actions.shape}"

        a_clip = np.clip(actions, -1.0, 1.0)
        dh = a_clip * self.max_dh

        # Apply altitude change
        self.h_uav = clip_altitude(self.h_uav + dh, self.h_min, self.h_max)

        # ---------- Energy proxy per step ----------
        h_norm = (self.h_uav - self.h_min) / (self.h_max - self.h_min + 1e-8)
        h_norm = np.clip(h_norm, 0.0, 1.0)

        E_hover = self.c_hover * (h_norm ** 2)
        E_climb = self.c_climb * (np.abs(dh) / (self.max_dh + 1e-8))
        E_step = E_hover + E_climb
        self.energy_used += E_step

        # Extra penalty for sitting near ceiling (helps break "all 500m")
        ceiling_pen = float(np.mean((h_norm > 0.95).astype(np.float32)))

        # Advance time
        self.r += 1
        done = (self.r >= self.T)

        if not done:
            self.x_uav = self.traj_x[self.r].astype(np.float32)
            self.y_uav = self.traj_y[self.r].astype(np.float32)

        # Recompute channel at new (x,y,h)
        self._compute_channel(self.x_uav, self.y_uav, self.h_uav)

        # Baseline selection
        idxs_users = np.array(self.selector.select(self.PL_db, self.K), dtype=int)

        # ✅ Soft reliability score for selected (works with smooth p_succ)
        rel_score = float(np.mean(self.p_succ[idxs_users])) if len(idxs_users) > 0 else 0.0
        self.last_rel = rel_score

        # For logging "successful users" (optional hard decision)
        if self.use_smooth_psucc:
            success_mask = (self.p_succ[idxs_users] >= 0.5)
        else:
            success_mask = (self.p_succ[idxs_users] > 0.0)
        successful_users = idxs_users[success_mask].tolist()

        # Track selection/success counts
        self.select_count[idxs_users] += 1.0
        if len(successful_users) > 0:
            self.success_count[np.array(successful_users, dtype=int)] += 1.0

        mean_snr_sel = float(np.mean(self.snr_db[idxs_users])) if len(idxs_users) > 0 else 0.0
        self.last_mean_snr = mean_snr_sel

        # Update prev_success per agent (1 if selected+successful else 0)
        prev = np.zeros(self.N, dtype=np.float32)
        for idx in successful_users:
            prev[idx] = 1.0
        self.prev_success = prev

        # ---------- Optional: run full FL (slow) ----------
        fl_info = {}
        delta_acc = 0.0
        delta_loss = 0.0

        if self.run_full_fl:
            if self.net_glob is None or self.dataset_train is None or self.dict_users is None:
                raise ValueError("run_full_fl=True requires net_glob, dataset_train, dict_users (and preferably dataset_test).")

            w_locals = []
            for idx in successful_users:
                local = LocalUpdate(args=self.args, dataset=self.dataset_train, idxs=self.dict_users[idx])
                w, _loss_local = local.train(net=copy.deepcopy(self.net_glob))
                w_locals.append(copy.deepcopy(w))

            if len(w_locals) > 0:
                w_glob = FedAvg(w_locals)
                self.net_glob.load_state_dict(w_glob)

            if self.dataset_test is not None:
                acc, loss = test_model(self.net_glob, self.dataset_test, self.args)
                acc = float(acc); loss = float(loss)
                fl_info = {"test_acc": acc, "test_loss": loss}

                if self.prev_test_acc is not None:
                    delta_acc = acc - self.prev_test_acc
                if self.prev_test_loss is not None:
                    delta_loss = self.prev_test_loss - loss  # positive if loss decreased

                self.prev_test_acc = acc
                self.prev_test_loss = loss

        # ---------- Reward components ----------
        link_quality = float(np.mean(self._snr_norm(self.snr_db[idxs_users]))) if len(idxs_users) > 0 else 0.0
        diversity = float(np.std(h_norm))
        action_cost = float(np.mean(np.abs(dh)) / (self.max_dh + 1e-8))
        energy_cost = float(np.mean(E_step))

        # ✅ Fairness as entropy BONUS (higher entropy => less bias)
        fair_bonus = self._fairness_entropy_norm()

        # ✅ FL gain term (prefer delta_loss, smoother than delta_acc)
        fl_gain = float(delta_loss) if self.run_full_fl else 0.0

        reward = (
            self.w_rel * rel_score
            + self.w_link * link_quality
            + self.w_div  * diversity
            + self.w_fair * fair_bonus
            - self.w_act  * action_cost
            - self.w_energy * energy_cost
            - self.w_ceiling * ceiling_pen
            + self.w_fl * fl_gain
        )

        # Optional hard budget penalty
        if self.E_max is not None:
            over = (self.energy_used > float(self.E_max)).astype(np.float32)
            if np.any(over > 0.0):
                reward -= 0.5 * float(np.mean(over))

        reward_n = np.full((self.N,), reward, dtype=np.float32)

        obs_n = self._build_obs()
        state = self._build_state(obs_n)

        info = {
            "selected": idxs_users.tolist(),
            "num_success": int(len(successful_users)),
            "rel_score_selected": float(rel_score),
            "mean_snr_selected": float(mean_snr_sel),

            "reward_total": float(reward),
            "reward_rel": float(rel_score),
            "reward_link_quality": float(link_quality),
            "reward_diversity": float(diversity),
            "bonus_fair_entropy": float(fair_bonus),

            "pen_action": float(action_cost),
            "pen_energy": float(energy_cost),
            "pen_ceiling": float(ceiling_pen),

            "energy_mean_total": float(np.mean(self.energy_used)),
            "energy_max_total": float(np.max(self.energy_used)),

            "delta_acc": float(delta_acc),
            "delta_loss": float(delta_loss),

            "mean_h": float(np.mean(self.h_uav)),
            "std_h": float(np.std(self.h_uav)),
            "mean_pl_db": float(np.mean(self.PL_db)),
            "mean_snr_db": float(np.mean(self.snr_db)),

            **fl_info
        }

        return obs_n, state, reward_n, done, info
