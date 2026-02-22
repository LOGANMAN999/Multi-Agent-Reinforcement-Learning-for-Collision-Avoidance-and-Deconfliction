"""rl_actor_critic_gru.py

Actor-Critic wrapper around the existing StudentGNNGRU policy.

Goal
----
Provide a *minimal, API-compatible* RL-ready network that:
  1) reuses the exact same GNN+GRU backbone you already trained (so you can
     initialize from your supervised `best.pt`),
  2) adds a value head for PPO/A2C-style methods,
  3) samples bounded actions via a tanh-squashed Gaussian.

This file is intentionally "trainer-agnostic": it doesn't implement PPO itself.
It just exposes the standard primitives you need when you build the trainer:
  - act(...) to sample actions + compute log-prob + value
  - evaluate_actions(...) for PPO updates

Expected inputs
---------------
`data` is a torch_geometric.data.Data object created by `rl_graph_obs.GraphObsBuilder`.
It must contain:
  - x         : (N, Dx)
  - edge_index: (2, E)
  - edge_attr : (E, De)
  - mask      : (N,) bool (True if agent is active)

Hidden state
------------
The GRU state is per-agent:
  h: (N, H_gru)
You should reset it at episode start.

Action scaling
--------------
Your supervised GRU training used targets y = actions / env.max_speed, i.e.
normalized to roughly [-1,1]. In RL we keep that convention *inside the policy*.
The environment step still needs physical velocities, so you will multiply by
env.max_speed outside (same as in StudentGNNGRUController).

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from pathlib import Path
import sys

# File is: repo/src/RL_stack/<this_file>.py
_THIS_DIR = Path(__file__).resolve().parent          # repo/src/RL_stack
_SRC_DIR = _THIS_DIR.parent                          # repo/src

# Make `controllers`, `sim_env`, `data_building`, etc. importable
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# (Optional) also allow importing other RL_stack files without package prefix
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from gnn_student_model_gru import StudentGNNGRU, StudentGNNGRUConfig



@dataclass
class ActorCriticConfig:
    """Config for the Actor-Critic wrapper."""

    # Must match your dataset/model dimensions.
    node_dim: int
    edge_dim: int

    # Backbone sizes (should match the supervised model you are initializing from).
    gnn_hidden_dim: int = 128
    gru_hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.0

    # Action space
    action_dim: int = 2
    # IMPORTANT: in your supervised GRU setup, outputs are *normalized* actions.
    # Keep max_speed=1.0 inside the policy; multiply by env.max_speed outside.
    max_speed: float = 1.0

    # Exploration (squashed Gaussian)
    init_log_std: float = -0.5  # exp(-0.5) ~ 0.61
    min_log_std: float = -5.0
    max_log_std: float = 2.0


def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Inverse tanh, numerically stable via clamping."""
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _squashed_gaussian_log_prob(
    raw_action: torch.Tensor,  # (N,A)
    mean: torch.Tensor,        # (N,A)
    log_std: torch.Tensor,     # (A,) or (N,A)
    squashed_action: torch.Tensor,  # tanh(raw_action)
    eps: float = 1e-6,
) -> torch.Tensor:
    """Log-prob under tanh-squashed diagonal Gaussian.

    Based on the standard change-of-variables correction:
      a = tanh(u),  u ~ N(mean, std)
      log pi(a) = log N(u; mean, std) - sum(log(1 - tanh(u)^2))

    Returns:
      logp: (N,) summed over action dims
    """
    std = torch.exp(log_std)
    dist = Normal(mean, std)
    # Sum over action dims.
    logp_u = dist.log_prob(raw_action).sum(dim=-1)
    # Jacobian correction.
    log_det = torch.log(1.0 - squashed_action.pow(2) + eps).sum(dim=-1)
    return logp_u - log_det


class GraphActorCriticGRU(nn.Module):
    """StudentGNNGRU backbone + value head + squashed Gaussian sampling."""

    def __init__(self, cfg: ActorCriticConfig):
        super().__init__()
        self.cfg = cfg

        student_cfg = StudentGNNGRUConfig(
            node_dim=cfg.node_dim,
            edge_dim=cfg.edge_dim,
            gnn_hidden_dim=cfg.gnn_hidden_dim,
            num_layers=cfg.num_layers,
            gru_hidden_dim=cfg.gru_hidden_dim,
            action_dim=cfg.action_dim,
            dropout=cfg.dropout,
            max_speed=cfg.max_speed,
        )
        self.actor = StudentGNNGRU(student_cfg)

        # Value head reads the GRU hidden state per agent.
        self.value_head = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.gru_hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.gru_hidden_dim, 1),
        )

        # Diagonal log-std parameter (shared across agents).
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), float(cfg.init_log_std)))

    # -------------------- checkpoint IO --------------------
    def load_supervised_student(self, ckpt_path: str | Path, *, strict: bool = True) -> None:
        """Load weights from a supervised StudentGNNGRU checkpoint.

        The checkpoint may be either:
          - a raw state_dict, or
          - a dict with key "model_state" (your supervised trainer format).

        This loads *only the actor* (StudentGNNGRU) weights. The critic/value head
        remains randomly initialized (as desired).
        """
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        self.actor.load_state_dict(state, strict=strict)

    def save(self, path: str | Path, *, extra: Optional[dict] = None) -> None:
        payload = {
            "actor_state": self.actor.state_dict(),
            "critic_state": self.value_head.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "cfg": self.cfg.__dict__,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, str(path))

    # -------------------- hidden state --------------------
    @torch.no_grad()
    def init_hidden(self, num_agents: int, device: Optional[torch.device] = None) -> torch.Tensor:
        dev = device if device is not None else next(self.parameters()).device
        return torch.zeros((num_agents, self.cfg.gru_hidden_dim), device=dev)

    # -------------------- action/value API --------------------
    def _actor_raw_mean_and_hnext(
        self,
        data,
        h_prev: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pre-tanh mean (raw), tanh(mean), and next hidden."""
        # Step the backbone to update h.
        actions_mean, h_next = self.actor.step(data, h_prev=h_prev, mask=mask)

        # Recover the pre-tanh mean corresponding to actions_mean.
        # In StudentGNNGRU: actions = max_speed * tanh(raw)
        # so tanh(raw) = actions / max_speed.
        # Since max_speed is 1.0 in normalized convention, this is simply actions.
        squashed_mean = actions_mean / float(self.cfg.max_speed)
        raw_mean = _atanh(squashed_mean)
        return raw_mean, squashed_mean, h_next

    def act(
        self,
        data,
        *,
        h_prev: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action.

        Returns:
          action: (N,A) in [-max_speed, max_speed] (here max_speed=1.0)
          logp  : (N,)  log-prob of the sampled action
          value : (N,)  state-value per agent
          h_next: (N,H) next hidden state
        """
        raw_mean, squashed_mean, h_next = self._actor_raw_mean_and_hnext(data, h_prev, mask)

        log_std = torch.clamp(self.log_std, self.cfg.min_log_std, self.cfg.max_log_std)

        if deterministic:
            raw_action = raw_mean
        else:
            eps = torch.randn_like(raw_mean)
            raw_action = raw_mean + torch.exp(log_std) * eps

        squashed_action = torch.tanh(raw_action)
        action = float(self.cfg.max_speed) * squashed_action

        # log-prob uses the squashed Gaussian formula.
        logp = _squashed_gaussian_log_prob(raw_action, raw_mean, log_std, squashed_action)

        # Value head reads h_next.
        value = self.value_head(h_next).squeeze(-1)

        # If a mask is provided, enforce inactive agents to have 0 action/logp/value.
        if mask is not None:
            m = mask.to(dtype=action.dtype)
            action = action * m.unsqueeze(-1)
            logp = logp * m
            value = value * m

        return action, logp, value, h_next

    def evaluate_actions(
        self,
        data,
        actions: torch.Tensor,  # (N,A) in [-max_speed,max_speed]
        *,
        h_prev: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given actions for PPO-style updates.

        Returns:
          logp    : (N,)
          entropy : (N,) (approx; entropy of underlying Gaussian)
          value   : (N,)
          h_next  : (N,H)
        """
        raw_mean, _, h_next = self._actor_raw_mean_and_hnext(data, h_prev, mask)

        log_std = torch.clamp(self.log_std, self.cfg.min_log_std, self.cfg.max_log_std)

        # Convert bounded action back to raw action via atanh.
        squashed_action = actions / float(self.cfg.max_speed)
        raw_action = _atanh(squashed_action)

        logp = _squashed_gaussian_log_prob(raw_action, raw_mean, log_std, torch.tanh(raw_action))

        # Underlying Gaussian entropy (sum over dims), broadcast over agents.
        std = torch.exp(log_std)
        dist = Normal(raw_mean, std)
        entropy = dist.entropy().sum(dim=-1)

        value = self.value_head(h_next).squeeze(-1)

        if mask is not None:
            m = mask.to(dtype=logp.dtype)
            logp = logp * m
            entropy = entropy * m
            value = value * m
        return logp, entropy, value, h_next
