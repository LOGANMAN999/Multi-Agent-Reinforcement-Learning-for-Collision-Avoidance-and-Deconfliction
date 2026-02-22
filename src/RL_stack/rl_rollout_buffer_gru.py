"""rl_rollout_buffer_gru.py

Recurrent rollout buffer for multi-agent PPO/A2C-style fine-tuning with a per-agent GRU policy.

This buffer is designed to mesh with your current repository conventions:

- Observations come from `rl_graph_obs.GraphObsBuilder.build(env)`, returning a PyG `Data`
  (x, edge_index, edge_attr, mask) and an `aux` dict.
- Actions/log-probs/values come from `rl_actor_critic_gru.GraphActorCriticGRU.act(...)`.
- Rewards + per-agent done flags come from `rl_rewards.RewardComputer.step(...)`.

Key requirements supported
--------------------------
1) Per-agent masking and per-agent termination
   - `mask_t[i]` indicates agent i was *active before acting* at time t.
   - `done_t[i]` indicates agent i terminated at the end of step t (goal or collision).
   - Advantage/return computation ignores inactive agents and does not bootstrap across done.

2) Variable edge counts over time
   - `edge_index_t` and `edge_attr_t` are stored as lists (E varies).

3) Recurrent minibatches (truncated BPTT)
   - Provides an iterator that yields contiguous sequences of length `seq_len`.
   - Each sequence includes the GRU hidden state at the sequence start (`h0`).

Notes
-----
- This is intentionally "trainer-agnostic": it does not implement PPO.
- To keep the first RL iteration robust, the buffer yields sequences as small python dicts.
  A trainer can loop over time steps inside each sequence and call
  `actor_critic.evaluate_actions(...)` to compute PPO losses.

Typical usage sketch (single-env)
---------------------------------
    buf = RolloutBufferGRU(horizon=T, num_agents=N, node_dim=Dx, edge_dim=De, hidden_dim=H)

    h = ac.init_hidden(N).to(device)
    prev_aux = None
    rewarder.reset(env)

    for t in range(T):
        data_t, aux = obs_builder.build(env)
        a, logp, v, h_next = ac.act(data_t.to(device), h_prev=h, mask=data_t.mask.to(device))
        env.step((a.cpu().numpy() * env.max_speed))
        r, done_agent, info = rewarder.step(env, aux, prev_aux, a.detach().cpu().numpy())
        buf.add(data_t, h, a, logp, v, r, done_agent)
        h = h_next
        prev_aux = aux

    # Bootstrap value from the last state (optional)
    data_T, aux_T = obs_builder.build(env)
    _, _, v_T, _ = ac.act(data_T.to(device), h_prev=h, mask=data_T.mask.to(device), deterministic=True)
    buf.compute_returns_and_advantages(gamma=0.99, lam=0.95, last_value=v_T)

    for batch in buf.iter_minibatches(seq_len=32, batch_size=4, device=device):
        ... PPO update ...

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from torch_geometric.data import Data
except Exception as e:  # pragma: no cover
    raise ImportError("torch_geometric is required for rl_rollout_buffer_gru.py") from e

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


@dataclass
class SequenceBatch:
    """One minibatch containing B sequences of length L.

    Shapes:
      x        : (B, L, N, Dx)
      mask     : (B, L, N) bool
      actions  : (B, L, N, A)
      logp_old : (B, L, N)
      values_old: (B, L, N)
      returns  : (B, L, N)
      adv      : (B, L, N)
      done     : (B, L, N) bool
      h0       : (B, N, H)

    Edges:
      edge_index[t][b]: (2, E_{t,b})
      edge_attr[t][b] : (E_{t,b}, De)
    """

    x: torch.Tensor
    mask: torch.Tensor
    actions: torch.Tensor
    logp_old: torch.Tensor
    values_old: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    done: torch.Tensor
    h0: torch.Tensor
    edge_index: List[List[torch.Tensor]]
    edge_attr: List[List[torch.Tensor]]


class RolloutBufferGRU:
    """Recurrent multi-agent rollout buffer (single environment instance).

    This buffer assumes a fixed `num_agents` for the duration of the rollout.
    If your env uses variable agent counts across episodes, create a new buffer per episode
    (or pad agents to a fixed N_max and keep mask accordingly).
    """

    def __init__(
        self,
        *,
        horizon: int,
        num_agents: int,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        action_dim: int = 2,
        store_on_cpu: bool = True,
    ) -> None:
        self.horizon = int(horizon)
        self.num_agents = int(num_agents)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.store_on_cpu = bool(store_on_cpu)

        self.reset()

    def reset(self) -> None:
        """Clear all stored rollout data."""
        T, N, Dx, H, A = self.horizon, self.num_agents, self.node_dim, self.hidden_dim, self.action_dim

        dev = torch.device("cpu") if self.store_on_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._t = 0

        self.x = torch.zeros((T, N, Dx), dtype=torch.float32, device=dev)
        self.mask = torch.zeros((T, N), dtype=torch.bool, device=dev)

        # Variable-size edge lists per time step
        self.edge_index: List[torch.Tensor] = [torch.empty((2, 0), dtype=torch.long, device=dev) for _ in range(T)]
        self.edge_attr: List[torch.Tensor] = [torch.empty((0, self.edge_dim), dtype=torch.float32, device=dev) for _ in range(T)]

        self.h = torch.zeros((T, N, H), dtype=torch.float32, device=dev)

        self.actions = torch.zeros((T, N, A), dtype=torch.float32, device=dev)
        self.logp = torch.zeros((T, N), dtype=torch.float32, device=dev)
        self.values = torch.zeros((T, N), dtype=torch.float32, device=dev)

        self.rewards = torch.zeros((T, N), dtype=torch.float32, device=dev)
        self.done = torch.zeros((T, N), dtype=torch.bool, device=dev)

        # Filled after calling compute_returns_and_advantages
        self.advantages = torch.zeros((T, N), dtype=torch.float32, device=dev)
        self.returns = torch.zeros((T, N), dtype=torch.float32, device=dev)

    @property
    def size(self) -> int:
        return self._t

    def add(
        self,
        data: Data,
        h_prev: torch.Tensor,  # (N,H)
        action: torch.Tensor,  # (N,A)
        logp: torch.Tensor,    # (N,)
        value: torch.Tensor,   # (N,)
        reward: np.ndarray | torch.Tensor,  # (N,)
        done: np.ndarray | torch.Tensor,    # (N,)
    ) -> None:
        """Add one timestep to the rollout."""
        if self._t >= self.horizon:
            raise RuntimeError(f"RolloutBufferGRU is full: t={self._t} horizon={self.horizon}")

        t = self._t
        dev = self.x.device

        # ---- observation ----
        x_t = data.x
        if x_t.shape != (self.num_agents, self.node_dim):
            raise ValueError(f"data.x has shape {tuple(x_t.shape)}; expected {(self.num_agents, self.node_dim)}")
        self.x[t].copy_(x_t.detach().to(device=dev, dtype=torch.float32))

        # mask field: prefer data.mask, fall back to last feature if absent
        if hasattr(data, "mask") and data.mask is not None:
            mask_t = data.mask
        else:
            # convention: last node feature is active
            mask_t = (x_t[:, -1] > 0.5)
        self.mask[t].copy_(mask_t.detach().to(device=dev, dtype=torch.bool))

        ei = data.edge_index.detach().to(device=dev, dtype=torch.long)
        ea = data.edge_attr.detach().to(device=dev, dtype=torch.float32)
        self.edge_index[t] = ei
        self.edge_attr[t] = ea

        # ---- recurrent state ----
        if h_prev.shape != (self.num_agents, self.hidden_dim):
            raise ValueError(f"h_prev has shape {tuple(h_prev.shape)}; expected {(self.num_agents, self.hidden_dim)}")
        self.h[t].copy_(h_prev.detach().to(device=dev, dtype=torch.float32))

        # ---- action + policy stats ----
        if action.shape != (self.num_agents, self.action_dim):
            raise ValueError(f"action has shape {tuple(action.shape)}; expected {(self.num_agents, self.action_dim)}")
        self.actions[t].copy_(action.detach().to(device=dev, dtype=torch.float32))

        self.logp[t].copy_(logp.detach().to(device=dev, dtype=torch.float32).reshape(self.num_agents))
        self.values[t].copy_(value.detach().to(device=dev, dtype=torch.float32).reshape(self.num_agents))

        # ---- reward + done ----
        if isinstance(reward, np.ndarray):
            r = torch.from_numpy(reward.astype(np.float32))
        else:
            r = reward.to(dtype=torch.float32)
        self.rewards[t].copy_(r.detach().to(device=dev).reshape(self.num_agents))

        if isinstance(done, np.ndarray):
            d = torch.from_numpy(done.astype(np.bool_))
        else:
            d = done.to(dtype=torch.bool)
        self.done[t].copy_(d.detach().to(device=dev).reshape(self.num_agents))

        self._t += 1

    # ------------------------- post-processing -------------------------

    @torch.no_grad()
    def compute_returns_and_advantages(
        self,
        *,
        gamma: float,
        lam: float,
        last_value: Optional[torch.Tensor] = None,  # (N,)
    ) -> None:
        """Compute GAE advantages and returns.

        Args:
            gamma: discount factor
            lam: GAE lambda
            last_value: value estimate at timestep T (after the final transition), shape (N,).
                        If None, we assume 0 (i.e., treat as terminal).
        """
        T = self.size
        if T == 0:
            return

        dev = self.x.device
        N = self.num_agents

        if last_value is None:
            next_value = torch.zeros((N,), device=dev, dtype=torch.float32)
        else:
            next_value = last_value.detach().to(device=dev, dtype=torch.float32).reshape(N)

        adv_next = torch.zeros((N,), device=dev, dtype=torch.float32)

        # Iterate backwards
        for t in reversed(range(T)):
            m = self.mask[t].to(dtype=torch.float32)  # active-before-action indicator
            done_t = self.done[t].to(dtype=torch.float32)

            # Do not bootstrap across a termination at the end of this step.
            nonterminal = (1.0 - done_t) * m

            # 1-step TD residual
            delta = (self.rewards[t] + gamma * next_value * nonterminal - self.values[t]) * m

            # GAE recursion
            adv = delta + gamma * lam * nonterminal * adv_next

            self.advantages[t].copy_(adv)
            self.returns[t].copy_(adv + self.values[t])

            # advance
            adv_next = adv
            next_value = self.values[t]

        # Zero out beyond filled horizon
        if T < self.horizon:
            self.advantages[T:].zero_()
            self.returns[T:].zero_()

    # ------------------------- minibatch iteration -------------------------

    def iter_minibatches(
        self,
        *,
        seq_len: int,
        batch_size: int,
        device: Optional[torch.device] = None,
        shuffle: bool = True,
        stride: Optional[int] = None,
    ) -> Iterator[SequenceBatch]:
        """Yield minibatches of contiguous sequences.

        Args:
            seq_len: unroll length (truncated BPTT length).
            batch_size: number of sequences per minibatch.
            device: device to move tensors to (e.g., cuda) when yielding.
            shuffle: whether to shuffle sequence start indices.
            stride: starting index step between candidate sequences.
                    Default: seq_len (non-overlapping segments).

        Yields:
            SequenceBatch with B sequences.
        """
        T = self.size
        if T == 0:
            return
        L = int(seq_len)
        if L <= 0:
            raise ValueError("seq_len must be positive")
        if T < L:
            raise ValueError(f"rollout length T={T} < seq_len={L}; use smaller seq_len or longer rollouts.")

        step = int(stride) if stride is not None else L
        starts = list(range(0, T - L + 1, step))
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(starts)

        # device handling
        out_dev = device if device is not None else self.x.device

        # Make batches
        for b0 in range(0, len(starts), batch_size):
            b_starts = starts[b0 : b0 + batch_size]
            B = len(b_starts)

            x = torch.stack([self.x[s : s + L] for s in b_starts], dim=0).to(out_dev)
            mask = torch.stack([self.mask[s : s + L] for s in b_starts], dim=0).to(out_dev)

            actions = torch.stack([self.actions[s : s + L] for s in b_starts], dim=0).to(out_dev)
            logp_old = torch.stack([self.logp[s : s + L] for s in b_starts], dim=0).to(out_dev)
            values_old = torch.stack([self.values[s : s + L] for s in b_starts], dim=0).to(out_dev)

            returns = torch.stack([self.returns[s : s + L] for s in b_starts], dim=0).to(out_dev)
            adv = torch.stack([self.advantages[s : s + L] for s in b_starts], dim=0).to(out_dev)
            done = torch.stack([self.done[s : s + L] for s in b_starts], dim=0).to(out_dev)

            h0 = torch.stack([self.h[s] for s in b_starts], dim=0).to(out_dev)

            # Edge lists as [t][b]
            edge_index: List[List[torch.Tensor]] = []
            edge_attr: List[List[torch.Tensor]] = []
            for t in range(L):
                edge_index.append([self.edge_index[s + t].to(out_dev) for s in b_starts])
                edge_attr.append([self.edge_attr[s + t].to(out_dev) for s in b_starts])

            # Normalize advantages over the *valid* entries (mask==1)
            # (Trainer may choose to re-normalize; this is a safe default.)
            with torch.no_grad():
                m = mask.to(dtype=torch.float32)
                denom = m.sum().clamp_min(1.0)
                mean = (adv * m).sum() / denom
                var = ((adv - mean) ** 2 * m).sum() / denom
                std = torch.sqrt(var + 1e-8)
                adv = (adv - mean) / (std + 1e-8)
                adv = adv * m  # keep inactive entries at 0

            yield SequenceBatch(
                x=x,
                mask=mask,
                actions=actions,
                logp_old=logp_old,
                values_old=values_old,
                returns=returns,
                advantages=adv,
                done=done,
                h0=h0,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

    # ------------------------- helper -------------------------

    def make_data_step(self, t: int, *, device: Optional[torch.device] = None) -> Data:
        """Reconstruct a PyG Data object for timestep t from stored tensors."""
        if t < 0 or t >= self.size:
            raise IndexError(f"t={t} out of range (size={self.size})")
        dev = device if device is not None else self.x.device
        return Data(
            x=self.x[t].to(dev),
            edge_index=self.edge_index[t].to(dev),
            edge_attr=self.edge_attr[t].to(dev),
            mask=self.mask[t].to(dev),
        )
