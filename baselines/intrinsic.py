"""Intrinsic-motivation modules for exploration bonuses: RND (Random Network
Distillation) and a count-based bonus over a discretized state grid.

A module exposes two methods used by the agent's train loop:
    intrinsic_reward(obs)  -> per-state novelty bonus (already normalized)
    update(obs)            -> trains the module's own networks, returns a loss
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    """Online mean/variance (Chan's parallel algorithm)."""

    def __init__(self, shape=(), eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = M2 / total
        self.count = total

    @property
    def std(self):
        return np.sqrt(self.var)


def _mlp(in_dim, hidden_dim, out_dim, n_hidden=2):
    layers = []
    d = in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
        d = hidden_dim
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class IntrinsicModule(nn.Module):
    """Interface for exploration-bonus modules."""

    def intrinsic_reward(self, obs):
        raise NotImplementedError

    def update(self, obs):
        raise NotImplementedError


class RND(IntrinsicModule):
    """Random Network Distillation.

    A fixed, randomly-initialized `target` net and a trained `predictor` net map
    observations to embeddings. The predictor learns to match the target on
    visited states, so its error is high on novel states -> exploration bonus.
    """

    def __init__(self, obs_dim, hidden_dim=128, out_dim=64, lr=1e-3, device="cpu"):
        super().__init__()
        self.device = device

        # fixed random target (never trained)
        self.target = _mlp(obs_dim, hidden_dim, out_dim).to(device)
        for p in self.target.parameters():
            p.requires_grad_(False)

        # trained predictor
        self.predictor = _mlp(obs_dim, hidden_dim, out_dim).to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # normalizers: obs (RND is scale-sensitive) and intrinsic reward
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.rew_rms = RunningMeanStd(())

    def _norm_obs(self, obs):
        mean = torch.as_tensor(self.obs_rms.mean, dtype=torch.float32, device=self.device)
        std = torch.as_tensor(self.obs_rms.std, dtype=torch.float32, device=self.device)
        return torch.clamp((obs - mean) / (std + 1e-8), -5.0, 5.0)

    @torch.no_grad()
    def intrinsic_reward(self, obs):
        """obs: (T, obs_dim) tensor -> (T,) normalized novelty bonus."""
        self.obs_rms.update(obs.detach().cpu().numpy())
        x = self._norm_obs(obs)
        err = ((self.predictor(x) - self.target(x)) ** 2).mean(dim=-1)  # (T,)
        self.rew_rms.update(err.detach().cpu().numpy())
        return err / (float(np.sqrt(self.rew_rms.var)) + 1e-8)

    def update(self, obs, n_steps=1):
        """Train the predictor to match the target on the visited states."""
        x = self._norm_obs(obs)
        target = self.target(x).detach()
        loss_val = 0.0
        for _ in range(n_steps):
            loss = ((self.predictor(x) - target) ** 2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_val = loss.item()
        return loss_val


class CountBased(IntrinsicModule):
    """Visit-count novelty over a discretized, normalized state grid.

    Bonus = 1/sqrt(N(bin)). Unlike RND, the signal does NOT collapse — rarely
    visited bins keep paying out — which makes it the right tool for low-dim
    state spaces (e.g. MountainCar's 2-D state). It does not scale to high-dim
    or pixel observations (bin space explodes); use RND there.
    """

    def __init__(self, obs_dim, bin_width=0.5, device="cpu"):
        super().__init__()
        self.device = device
        self.bin_width = bin_width
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.counts = {}

    def _keys(self, obs_np):
        norm = (obs_np - self.obs_rms.mean) / (self.obs_rms.std + 1e-8)
        binned = np.round(norm / self.bin_width).astype(np.int64)
        return [tuple(row) for row in binned]

    @torch.no_grad()
    def intrinsic_reward(self, obs):
        obs_np = obs.detach().cpu().numpy()
        self.obs_rms.update(obs_np)
        bonuses = []
        for key in self._keys(obs_np):
            self.counts[key] = self.counts.get(key, 0) + 1
            bonuses.append(1.0 / np.sqrt(self.counts[key]))
        return torch.as_tensor(bonuses, dtype=torch.float32, device=self.device)

    def update(self, obs):
        return 0.0  # no networks to train
