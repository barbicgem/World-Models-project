import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from intrinsic import CountBased, RND, RunningMeanStd

"""On-policy actor-critic agents (A2C, PPO) with optional exploration bonuses.

Both share BaseAgent: MLP actor (categorical or diagonal Gaussian) and critic,
GAE(lambda) advantages, running observation normalization, and an optional
intrinsic reward (RND or count-based) added to the extrinsic reward during
training and linearly annealed to zero.
"""


class RolloutBuffer:
    """On-policy rollout storage with GAE(lambda) advantage computation."""

    def __init__(self, gamma, lam):

        self.gamma = gamma
        self.lam = lam #lambda

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = None
        self.returns = None

    def add(self, state, action, log_prob, reward, value, done):

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, value):

        # GAE Advantage = sum of TD errors

        gae = 0
        T = len(self.rewards)
        advantages = [0]*T

        for t in range(T-1, -1, -1):
            if t == T-1:
                next_val = value
            else:
                next_val = self.values[t+1]
            next_nonterm = 1 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_val * next_nonterm - self.values[t]
            gae = delta + self.lam * self.gamma * next_nonterm * gae
            advantages[t] = gae

        returns = [a + v for a, v in zip(advantages, self.values)]

        self.advantages, self.returns = advantages, returns

    def get(self):

        states_t = torch.tensor(np.array(self.states), dtype = torch.float32)
        actions_t = torch.tensor(np.array(self.actions), dtype = torch.float32)
        log_probs_t = torch.tensor(self.log_probs, dtype = torch.float32)
        advantages_t = torch.tensor(self.advantages, dtype = torch.float32)
        returns_t = torch.tensor(self.returns, dtype = torch.float32)

        return states_t, actions_t, log_probs_t, advantages_t, returns_t

    def clear(self):

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []

class CriticNet(nn.Module):

    def __init__(self, obs_dim, hidden_dim=64, num_layers=2):
        super().__init__()

        layers = []
        in_dim = obs_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1)) # output head, want one value
        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs).squeeze(-1)

class ActorNet(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dim=64, num_layers=2, discrete=True):
        super().__init__()

        self.discrete = discrete

        layers = []
        in_dim = obs_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, act_dim)) # logits over all actions
        self.network = nn.Sequential(*layers)

        if not discrete:
            self.log_std = nn.Parameter(torch.zeros(act_dim))


    def forward(self, obs):

        out = self.network(obs)

        if self.discrete:
            return torch.distributions.Categorical(logits=out)

        return torch.distributions.Normal(out, self.log_std.exp())

class BaseAgent():

    def __init__(self, env, hidden_dim=64, num_layers=2, gamma=0.99, lam=0.95,
             lr=3e-4, value_coef=0.5, entropy_coef=0.01, device="cpu",
             explore="none", beta_start=0.5, beta_anneal_steps=0, rnd_lr=1e-3,
             normalize_obs=True):

        self.env = env
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lam = lam
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device

        # exploration bonus weight (used only when explore != "none")
        self.beta_start = beta_start
        self.beta_anneal_steps = beta_anneal_steps
        self.global_step = 0

        self.obs_dims = env.observation_space.shape[0]

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.discrete = True
            self.action_dim = env.action_space.n
        else:
            self.discrete = False
            self.action_dim = env.action_space.shape[0]

        # actor + critic
        self.actor = ActorNet(self.obs_dims, self.action_dim, hidden_dim,
                              num_layers, self.discrete).to(device)
        self.critic = CriticNet(self.obs_dims, hidden_dim, num_layers).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

        # rollout storage
        self.buffer = RolloutBuffer(gamma, lam)

        if explore == "rnd":
            self.intrinsic = RND(self.obs_dims, lr=rnd_lr, device=device)
        elif explore == "count":
            self.intrinsic = CountBased(self.obs_dims, device=device)
        else:
            self.intrinsic = None
        self.last_int_mean = 0.0   # diagnostics: mean intrinsic reward this rollout
        self.last_beta = 0.0

        # running observation normalizer (VecNormalize-style); also keeps the
        # differently-scaled context features comparable
        self.obs_rms = RunningMeanStd((self.obs_dims,)) if normalize_obs else None

    def _norm_obs(self, obs_t, update=False):
        if self.obs_rms is None:
            return obs_t
        if update:
            self.obs_rms.update(obs_t.detach().cpu().numpy().reshape(-1, self.obs_dims))
        mean = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=self.device)
        std = torch.as_tensor(self.obs_rms.std, dtype=torch.float32, device=self.device)
        return torch.clamp((obs_t - mean) / (std + 1e-8), -10.0, 10.0)

    def current_beta(self):
        """Exploration-bonus weight, linearly annealed to 0 over training."""
        if self.intrinsic is None or self.beta_start <= 0:
            return 0.0
        if self.beta_anneal_steps <= 0:
            return self.beta_start
        frac = max(0.0, 1.0 - self.global_step / self.beta_anneal_steps)
        return self.beta_start * frac

    def act(self, obs, train=True):

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            obs_n = self._norm_obs(obs_t, update=train)  # update stats only while training
            dist = self.actor(obs_n)
            value = self.critic(obs_n)

            if train:  # stochastic action sampling
                action = dist.sample()
            else:  # deterministic action sampling
                if self.discrete:
                    action = dist.probs.argmax(-1)
                else:
                    action = dist.mean

            log_prob = dist.log_prob(action) # returns per-dim log_probs
            if not self.discrete:
                log_prob = log_prob.sum(-1)

            action_out = action.item() if self.discrete else action.cpu().numpy()

        return action_out, log_prob.item(), value.item()

    def train(self, total_steps, rollout_len=2048, log_interval=10):

        obs, _ = self.env.reset()
        ep_return = 0.0
        ep_returns = []
        n_updates = max(1, total_steps // rollout_len)

        for update_i in range(1, n_updates + 1):

            # collect one rollout
            done = False
            for _ in range(rollout_len):
                action, log_prob, value = self.act(obs, train=True)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.buffer.add(obs, action, log_prob, reward, value, done)

                obs = next_obs
                ep_return += reward
                self.global_step += 1
                if done:
                    ep_returns.append(ep_return)
                    ep_return = 0.0
                    obs, _ = self.env.reset()

            # exploration bonus: add beta * normalized RND novelty to the reward
            # (extrinsic reward stays the logged metric; only training reward changes)
            if self.intrinsic is not None:
                states_t = torch.as_tensor(np.array(self.buffer.states),
                                           dtype=torch.float32, device=self.device)
                r_int = self.intrinsic.intrinsic_reward(states_t)
                beta = self.current_beta()
                self.last_int_mean = float(r_int.mean())
                self.last_beta = beta
                self.buffer.rewards = [re + beta * ri for re, ri
                                       in zip(self.buffer.rewards, r_int.tolist())]
                self.intrinsic.update(states_t)

            # bootstrap value of the state after the last step
            if done:
                last_value = 0.0
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    last_value = self.critic(self._norm_obs(obs_t)).item()

            self.buffer.compute_gae(last_value)
            self.update()

            if update_i % log_interval == 0 and ep_returns:
                recent = ep_returns[-10:]
                print(f"update {update_i}/{n_updates}  "
                      f"avg_return(last {len(recent)}) {sum(recent)/len(recent):.1f}")

        return ep_returns

    def evaluate(self, n_episodes=10):

        returns = []

        for _ in range(n_episodes):

            obs, _ = self.env.reset()
            done = False
            ep_return = 0.0

            while not done:

                action, _, _ = self.act(obs, train=False)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_return += reward

            returns.append(ep_return)

        return sum(returns) / len(returns)

class A2CAgent(BaseAgent):
    """Single gradient step per rollout."""

    def update(self):

        states, actions, old_log_probs, advantages, returns = self.buffer.get()

        if self.discrete:
            actions = actions.long()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalising advantages

        states = self._norm_obs(states)

        dist = self.actor(states)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        if not self.discrete:
            new_log_probs = new_log_probs.sum(-1)
            entropy = entropy.sum(-1)
        values = self.critic(states)

        # losses
        actor_loss  = -(new_log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_bonus = entropy.mean()
        total = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_bonus

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()
        self.buffer.clear()

class PPOAgent(BaseAgent):
    """Clipped surrogate objective, several epochs of minibatch updates per rollout."""

    def __init__(self, env, clip_ratio=0.2, n_epochs=10, minibatch_size=64, **kwargs):

        super().__init__(env, **kwargs)
        self.clip_ratio = clip_ratio
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size

    def update(self):

        states, actions, old_log_probs, advantages, returns = self.buffer.get()

        if self.discrete:
            actions = actions.long()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalising advantages

        states = self._norm_obs(states)

        T = states.shape[0]
        for _ in range(self.n_epochs):

            idx = torch.randperm(T)
            for start in range(0, T, self.minibatch_size):

                mb = idx[start : start + self.minibatch_size]

                dist = self.actor(states[mb])
                new_log_probs = dist.log_prob(actions[mb])
                entropy = dist.entropy()

                if not self.discrete:
                    new_log_probs = new_log_probs.sum(-1)
                    entropy = entropy.sum(-1)
                values = self.critic(states[mb])

                ratio = torch.exp(new_log_probs - old_log_probs[mb])
                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[mb]
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(values, returns[mb])
                total = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()
                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()

        self.buffer.clear()
