# mappo_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

def _tanh_squash(sample, log_prob):
    """
    Apply tanh squash and log_prob correction.
    sample: [B, ...]
    log_prob: [B]
    """
    squashed = torch.tanh(sample)
    # correction: sum log(1 - tanh(x)^2)
    correction = torch.sum(torch.log(1.0 - squashed.pow(2) + 1e-6), dim=-1)
    return squashed, (log_prob - correction)

class Actor(nn.Module):
    """
    Shared actor for all UAVs.
    Input: obs_i (dim=6)
    Output: 2D tanh-squashed Gaussian action:
      a0 in [-1,1] -> delta_h = a0 * delta_h_max
      a1 in [-1,1] -> score = (a1+1)/2 in [0,1]
    """
    def __init__(self, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mu = nn.Linear(hidden, 2)         # mean for (dh_latent, score_latent)
        self.log_std = nn.Parameter(torch.zeros(2))  # shared log-std

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu(x)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        return mu, std

class Critic(nn.Module):
    """
    Centralized critic: input global state (N*obs_dim), output scalar V(s).
    """
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.state = []
        self.action = []
        self.logp_sum = []
        self.value = []
        self.reward = []
        self.done = []

    def add(self, obs, state, action, logp_sum, value, reward, done):
        self.obs.append(obs)
        self.state.append(state)
        self.action.append(action)
        self.logp_sum.append(logp_sum)
        self.value.append(value)
        self.reward.append(reward)
        self.done.append(done)

    def clear(self):
        self.__init__()

class MAPPOAgent:
    def __init__(self, args, obs_dim, state_dim, device):
        self.args = args
        self.device = device

        self.actor = Actor(obs_dim).to(device)
        self.critic = Critic(state_dim).to(device)

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=args.lr_critic)

        self.buf = RolloutBuffer()

        self.delta_h_max = args.delta_h_max

    @torch.no_grad()
    def act(self, obs_n, state):
        """
        obs_n: [N, obs_dim] numpy
        state: [state_dim] numpy
        returns:
          delta_h: [N] numpy (meters)
          scores: [N] numpy ([0,1])
          logp_sum: float (sum logprobs over all agents)
          value: float
        """
        obs = torch.tensor(obs_n, dtype=torch.float32, device=self.device)
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        mu, std = self.actor(obs)                        # [N,2]
        dist = Normal(mu, std)
        raw = dist.rsample()                             # [N,2]
        logp = dist.log_prob(raw).sum(dim=-1)           # [N]

        squashed, logp_corr = _tanh_squash(raw, logp)    # [N,2], [N]

        # map to env actions
        dh = squashed[:, 0] * self.delta_h_max
        score = (squashed[:, 1] + 1.0) * 0.5            # [0,1]

        value = self.critic(st).item()
        logp_sum = logp_corr.sum().item()

        return dh.cpu().numpy(), score.cpu().numpy(), logp_sum, value

    @torch.no_grad()
    def act_deterministic(self, obs_n, state):
        obs = torch.tensor(obs_n, dtype=torch.float32, device=self.device)
        st = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        mu, _ = self.actor(obs)
        squashed = torch.tanh(mu)                       # deterministic
        dh = squashed[:, 0] * self.delta_h_max
        score = (squashed[:, 1] + 1.0) * 0.5
        value = self.critic(st).item()
        return dh.cpu().numpy(), score.cpu().numpy(), value

    def store(self, obs_n, state, dh, score, logp_sum, value, reward, done):
        action = np.stack([dh, score], axis=1).astype(np.float32)  # [N,2]
        self.buf.add(
            obs=obs_n.astype(np.float32),
            state=state.astype(np.float32),
            action=action,
            logp_sum=float(logp_sum),
            value=float(value),
            reward=float(reward),
            done=bool(done)
        )

    def _compute_gae(self, rewards, values, dones):
        """
        rewards: [T]
        values: [T+1] (bootstrap)
        dones: [T]
        """
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.args.gamma * values[t+1] * mask - values[t]
            gae = delta + self.args.gamma * self.args.gae_lambda * mask * gae
            adv[t] = gae
        ret = adv + values[:-1]
        return adv, ret

    def update(self):
        # Convert buffer to tensors
        obs = torch.tensor(np.array(self.buf.obs), dtype=torch.float32, device=self.device)        # [T,N,obs_dim]
        state = torch.tensor(np.array(self.buf.state), dtype=torch.float32, device=self.device)    # [T,state_dim]
        action = torch.tensor(np.array(self.buf.action), dtype=torch.float32, device=self.device)  # [T,N,2]
        old_logp_sum = torch.tensor(np.array(self.buf.logp_sum), dtype=torch.float32, device=self.device)  # [T]
        rewards = np.array(self.buf.reward, dtype=np.float32)                                      # [T]
        dones = np.array(self.buf.done, dtype=np.bool_)                                            # [T]
        values = np.array(self.buf.value, dtype=np.float32)                                        # [T]

        # Bootstrap value for last state
        with torch.no_grad():
            last_v = self.critic(state[-1].unsqueeze(0)).item()
        values_boot = np.concatenate([values, np.array([last_v], dtype=np.float32)], axis=0)

        adv, ret = self._compute_gae(rewards, values_boot, dones)

        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        T, N, _ = obs.shape

        # Flatten time for minibatching
        idxs = np.arange(T)

        for _ in range(self.args.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, T, self.args.minibatch_size):
                mb = idxs[start:start + self.args.minibatch_size]
                if len(mb) == 0:
                    continue

                mb_obs = obs[mb]       # [B,N,obs_dim]
                mb_state = state[mb]   # [B,state_dim]
                mb_action = action[mb] # [B,N,2]
                mb_old_logp = old_logp_sum[mb]  # [B]
                mb_adv = adv_t[mb]     # [B]
                mb_ret = ret_t[mb]     # [B]

                # --- Recompute log prob under current actor ---
                B = mb_obs.shape[0]
                flat_obs = mb_obs.reshape(B * N, -1)  # [B*N, obs_dim]

                mu, std = self.actor(flat_obs)        # [B*N,2]
                dist = Normal(mu, std)

                # invert mapping approximately: our action stores (dh_meters, score_[0,1])
                # reconstruct squashed actions:
                dh_squash = torch.clamp(mb_action[..., 0] / (self.delta_h_max + 1e-8), -0.999, 0.999)
                sc_squash = torch.clamp(mb_action[..., 1] * 2.0 - 1.0, -0.999, 0.999)
                squash = torch.stack([dh_squash, sc_squash], dim=-1)  # [B,N,2]
                squash = squash.reshape(B * N, 2)

                # atanh to get pre-tanh values
                raw = 0.5 * torch.log((1 + squash) / (1 - squash + 1e-8))
                logp = dist.log_prob(raw).sum(dim=-1)  # [B*N]
                # tanh correction
                correction = torch.sum(torch.log(1.0 - torch.tanh(raw).pow(2) + 1e-6), dim=-1)
                logp_corr = logp - correction          # [B*N]

                logp_sum_new = logp_corr.reshape(B, N).sum(dim=1)    # [B]

                ratio = torch.exp(logp_sum_new - mb_old_logp)        # [B]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.args.ppo_clip, 1.0 + self.args.ppo_clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                v_pred = self.critic(mb_state)
                value_loss = ((v_pred - mb_ret) ** 2).mean()

                # Entropy bonus (encourage exploration)
                entropy = dist.entropy().sum(dim=-1).reshape(B, N).sum(dim=1).mean()
                loss = policy_loss + self.args.vf_coef * value_loss - self.args.entropy_coef * entropy

                self.opt_actor.zero_grad()
                self.opt_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()),
                                         self.args.max_grad_norm)
                self.opt_actor.step()
                self.opt_critic.step()

        self.buf.clear()

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor.eval()
        self.critic.eval()
