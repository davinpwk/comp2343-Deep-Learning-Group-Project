"""DQN building blocks for the transfer-learning experiment.

QNet, ReplayBuffer, DQNAgent. Lifted verbatim from
../archive/dqn.ipynb (cells 4, 7, 10) so the experiment uses the same
model the project's lab work was built around. Differences from the
notebook:

  - `set_epsilon` / `set_epsilon_schedule` helpers so the fine-tune loop
    can start from a custom (ε_start, ε_end, decay_steps) without
    rebuilding the agent.
  - `state_dict` / `load_state_dict` round-trips so checkpoints can move
    cleanly between phases (pretrain -> fine-tune -> eval).

Nothing about the architecture (MLP, hidden=[128,128]), Double-DQN
target rule, Huber loss, or update cadence has been changed.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------
class QNet(nn.Module):
    """Simple MLP: obs -> Q(s, a) per action. ReLU activations."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def linear_layers(self) -> list[nn.Linear]:
        """All nn.Linear modules in evaluation order. Used by freeze utils."""
        return [m for m in self.net if isinstance(m, nn.Linear)]


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Fixed-capacity uniform-sample replay buffer."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.append((
            np.asarray(obs, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_obs, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.stack(obs),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_obs),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent (Double-DQN target rule + Huber loss + grad clipping)
# ---------------------------------------------------------------------------
class DQNAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 500,
        hidden_sizes: list[int] = (128, 128),
        grad_clip: float = 10.0,
        device: str = "cpu",
    ) -> None:
        self.obs_dim            = obs_dim
        self.n_actions          = n_actions
        self.lr                 = lr
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.grad_clip          = grad_clip
        self.device             = torch.device(device)
        self.hidden_sizes       = list(hidden_sizes)

        self.online_net = QNet(obs_dim, n_actions, list(hidden_sizes)).to(self.device)
        self.target_net = QNet(obs_dim, n_actions, list(hidden_sizes)).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self._rebuild_optimizer()
        self.buffer = ReplayBuffer(buffer_capacity)
        self._steps = 0
        self._last_obs: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None

    # ---- training-side knobs --------------------------------------------

    def _rebuild_optimizer(self) -> None:
        """(Re)build Adam over the *trainable* online-net parameters.

        Called once at construction and again after any freezing change,
        so the optimizer never holds stale references to frozen tensors.
        """
        trainable = [p for p in self.online_net.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable, lr=self.lr)

    def freeze_first_layer(self) -> None:
        """Freeze the first nn.Linear layer's parameters in the online net.

        After freezing, the optimizer is rebuilt over the remaining
        trainable params so it doesn't carry references to frozen tensors.
        Idempotent.
        """
        linears = self.online_net.linear_layers()
        if not linears:
            raise ValueError("Online net has no Linear layers to freeze.")
        for p in linears[0].parameters():
            p.requires_grad = False
        self._rebuild_optimizer()

    def set_epsilon_schedule(self, eps_start: float, eps_end: float,
                             decay_steps: int) -> None:
        """Switch to a fresh (ε_start, ε_end, decay) schedule mid-training.

        Used by fine-tuning to start from a lower ε than pretraining and
        decay over a shorter horizon (see EXPERIMENT_PLAN §5).
        """
        self.epsilon = float(eps_start)
        self.epsilon_end = float(eps_end)
        self.epsilon_decay = (eps_start - eps_end) / max(1, decay_steps)

    # ---- act / learn ----------------------------------------------------

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        self._last_obs = np.asarray(obs, dtype=np.float32)
        if (not greedy) and random.random() < self.epsilon:
            action = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                x = torch.as_tensor(self._last_obs, device=self.device).unsqueeze(0)
                action = int(self.online_net(x).argmax(dim=1).item())
        self._last_action = action
        return action

    def store_transition(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self) -> dict:
        if len(self.buffer) < self.batch_size:
            return {}

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        obs_t      = torch.as_tensor(obs,      dtype=torch.float32, device=self.device)
        actions_t  = torch.as_tensor(actions,  dtype=torch.int64,   device=self.device)
        rewards_t  = torch.as_tensor(rewards,  dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t    = torch.as_tensor(dones,    dtype=torch.float32, device=self.device)

        B = obs_t.shape[0]
        all_q = self.online_net(obs_t)
        q_values = all_q[torch.arange(B, device=self.device), actions_t]

        with torch.no_grad():
            # Double-DQN: online net selects the action, target net values it.
            next_actions = self.online_net(next_obs_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_obs_t).gather(1, next_actions).squeeze(1)
            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            (p for p in self.online_net.parameters() if p.requires_grad),
            self.grad_clip,
        )
        self.optimizer.step()

        self._steps += 1
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        if self._steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {"loss": float(loss.item()), "epsilon": float(self.epsilon),
                "grad_steps": self._steps}

    # ---- checkpointing --------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "online":     self.online_net.state_dict(),
            "target":     self.target_net.state_dict(),
            "epsilon":    self.epsilon,
            "grad_steps": self._steps,
        }

    def load_state_dict(self, sd: dict, load_target: bool = True) -> None:
        self.online_net.load_state_dict(sd["online"])
        if load_target and "target" in sd:
            self.target_net.load_state_dict(sd["target"])
        else:
            self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        if "epsilon" in sd:
            self.epsilon = float(sd["epsilon"])
        if "grad_steps" in sd:
            self._steps = int(sd["grad_steps"])
