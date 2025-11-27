import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict, Any


class UpperGraphActor(nn.Module):
    """
    Graph-based Actor using a G2ANet-style message passing backbone.
    Decodes per-agent group logits (Phase 1: only group_id head).
    """
    def __init__(self, obs_dim: int, n_agents: int, n_groups: int, args):
        super().__init__()
        self.n_agents = n_agents
        self.n_groups = n_groups
        self.args = args
        hid = getattr(args, 'rnn_hidden_dim', 128)
        att = getattr(args, 'attention_dim', 64)

        # Encoding and per-agent GRU cell
        self.encoding = nn.Linear(obs_dim, hid)
        self.gru = nn.GRUCell(hid, hid)

        # Hard attention (bidirectional GRU over pairwise concat)
        self.hard_bi_gru = nn.GRU(hid * 2, hid, bidirectional=True)
        self.hard_encoding = nn.Linear(hid * 2, 2)

        # Soft attention projections
        self.q = nn.Linear(hid, att, bias=False)
        self.k = nn.Linear(hid, att, bias=False)
        self.v = nn.Linear(hid, att)

        # Decoder to group logits
        self.decoding = nn.Linear(hid + att, n_groups)
        
        # Initialize weights properly to avoid NaN
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with conservative values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRUCell):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, X: torch.Tensor, adj: Optional[torch.Tensor], hidden_state: torch.Tensor):
        """
        X: (batch_size * n_agents, obs_dim)
        adj: (n_agents, n_agents) or (batch_size, n_agents, n_agents) binary/weight matrix; if None treat fully-connected
        hidden_state: (batch_size, n_agents, hid)
        Returns: logits (batch_size * n_agents, n_groups), new_hidden (batch_size * n_agents, hid)
        """
        # Check for NaN in input
        if torch.isnan(X).any():
            print(f"[WARNING] NaN detected in observation input, replacing with zeros")
            X = torch.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        size = X.shape[0]  # batch_size * n_agents
        obs_enc = F.relu(self.encoding(X))
        # Clip encoded values to prevent overflow
        obs_enc = torch.clamp(obs_enc, -10.0, 10.0)
        
        h_in = hidden_state.reshape(-1, hidden_state.shape[-1])
        h_out = self.gru(obs_enc, h_in)
        # Clip hidden state
        h_out = torch.clamp(h_out, -10.0, 10.0)

        # Prepare hard attention inputs
        bs = size // self.n_agents
        h = h_out.reshape(bs, self.n_agents, -1)
        pair_inputs = []
        for i in range(self.n_agents):
            h_i = h[:, i]
            h_pairs = []
            for j in range(self.n_agents):
                if j == i:
                    continue
                h_pairs.append(torch.cat([h_i, h[:, j]], dim=-1))
            pair_inputs.append(torch.stack(h_pairs, dim=0))  # (n_agents-1, bs, 2*hid)
        pair_inputs = torch.stack(pair_inputs, dim=-2)  # (n_agents-1, bs, n_agents, 2*hid)
        pair_inputs = pair_inputs.view(self.n_agents - 1, -1, h_out.shape[-1] * 2)  # (n_agents-1, bs*n_agents, 2*hid)

        h0 = torch.zeros((2 * 1, size, h_out.shape[-1]), device=h_out.device)
        hard_out, _ = self.hard_bi_gru(pair_inputs, h0)
        hard_out = hard_out.permute(1, 0, 2).reshape(-1, h_out.shape[-1] * 2)
        hard_out = torch.clamp(hard_out, -10.0, 10.0)
        
        # Use temperature=1.0 instead of 0.01 for more stable Gumbel-Softmax
        hard_logits = self.hard_encoding(hard_out)
        hard_logits = torch.clamp(hard_logits, -10.0, 10.0)
        hard_weights = F.gumbel_softmax(hard_logits, tau=1.0, hard=False)[:, 1]
        hard_weights = hard_weights.view(bs, self.n_agents, 1, self.n_agents - 1)
        hard_weights = hard_weights.permute(1, 0, 2, 3)  # (n_agents, bs, 1, n_agents-1)

        # Soft attention
        q = self.q(h_out).reshape(bs, self.n_agents, -1)
        k = self.k(h_out).reshape(bs, self.n_agents, -1)
        v = F.relu(self.v(h_out)).reshape(bs, self.n_agents, -1)

        # Optional adjacency gating
        if adj is not None:
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).repeat(bs, 1, 1)
            # Build per-agent neighbor mask excluding self
            neighbor_masks = []
            for i in range(self.n_agents):
                mask_i = []
                for j in range(self.n_agents):
                    if j != i:
                        mask_i.append(adj[:, i, j])
                neighbor_masks.append(torch.stack(mask_i, dim=-1))  # (bs, n_agents-1)
            neighbor_masks = torch.stack(neighbor_masks, dim=1)  # (bs, n_agents, n_agents-1)

        x_list = []
        for i in range(self.n_agents):
            q_i = q[:, i].unsqueeze(1)  # (bs,1,att)
            k_i = torch.stack([k[:, j] for j in range(self.n_agents) if j != i], dim=0).permute(1, 2, 0)
            v_i = torch.stack([v[:, j] for j in range(self.n_agents) if j != i], dim=0).permute(1, 2, 0)
            score = torch.matmul(q_i, k_i)  # (bs,1,n_agents-1)
            scaled = score / np.sqrt(k_i.shape[1])
            soft_w = F.softmax(scaled, dim=-1)
            if adj is not None:
                gate = neighbor_masks[:, i].unsqueeze(1)  # (bs,1,n_agents-1)
                soft_w = soft_w * gate
            x_i = (v_i * soft_w * hard_weights[i]).sum(dim=-1)  # (bs,att)
            x_list.append(x_i)
        x = torch.stack(x_list, dim=1).reshape(-1, v.shape[-1])
        x = torch.clamp(x, -10.0, 10.0)
        
        final_in = torch.cat([h_out, x], dim=-1)
        final_in = torch.clamp(final_in, -10.0, 10.0)
        
        logits = self.decoding(final_in)
        # Clip logits to prevent extreme values
        logits = torch.clamp(logits, -20.0, 20.0)
        
        # Final NaN check
        if torch.isnan(logits).any():
            print(f"[ERROR] NaN in logits after forward pass! Resetting to uniform")
            logits = torch.zeros_like(logits)
        
        return logits, h_out


class UpperGraphCritic(nn.Module):
    """Graph Critic producing global value and per-agent local value."""
    def __init__(self, obs_dim: int, n_agents: int, args):
        super().__init__()
        self.n_agents = n_agents
        hid = getattr(args, 'rnn_hidden_dim', 128)
        att = getattr(args, 'attention_dim', 64)

        self.enc = nn.Linear(obs_dim, hid)
        self.gru = nn.GRUCell(hid, hid)
        self.q = nn.Linear(hid, att, bias=False)
        self.k = nn.Linear(hid, att, bias=False)
        self.v = nn.Linear(hid, att)
        self.readout_global = nn.Sequential(
            nn.Linear(hid + att, hid), nn.ReLU(), nn.Linear(hid, 1)
        )
        self.readout_local = nn.Sequential(
            nn.Linear(hid + att, hid), nn.ReLU(), nn.Linear(hid, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with conservative values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRUCell):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, X: torch.Tensor, adj: Optional[torch.Tensor], hidden_state: torch.Tensor):
        size = X.shape[0]
        bs = size // self.n_agents
        h_in = hidden_state.reshape(-1, hidden_state.shape[-1])
        h_out = self.gru(F.relu(self.enc(X)), h_in)
        h = h_out.reshape(bs, self.n_agents, -1)
        q = self.q(h_out).reshape(bs, self.n_agents, -1)
        k = self.k(h_out).reshape(bs, self.n_agents, -1)
        v = F.relu(self.v(h_out)).reshape(bs, self.n_agents, -1)
        x_list = []
        for i in range(self.n_agents):
            q_i = q[:, i].unsqueeze(1)
            k_i = torch.stack([k[:, j] for j in range(self.n_agents) if j != i], dim=0).permute(1, 2, 0)
            v_i = torch.stack([v[:, j] for j in range(self.n_agents) if j != i], dim=0).permute(1, 2, 0)
            score = torch.matmul(q_i, k_i)
            soft_w = F.softmax(score / np.sqrt(k_i.shape[1]), dim=-1)
            x_i = (v_i * soft_w).sum(dim=-1)
            x_list.append(x_i)
        x = torch.stack(x_list, dim=1).reshape(-1, v.shape[-1])
        final_in = torch.cat([h_out, x], dim=-1)
        V_local = self.readout_local(final_in).reshape(bs, self.n_agents)
        # Global readout as mean pooling then MLP
        global_feat = final_in.reshape(bs, self.n_agents, -1).mean(dim=1)
        V_global = self.readout_global(global_feat).reshape(bs)
        return V_global, V_local, h_out


class UpperGraphAC:
    """Minimal learner wrapper for UpperGraphActor/Critic with synchronous updates."""
    def __init__(self, args, hrl_controller, obs_dim: int, n_agents: int):
        n_groups = len(getattr(hrl_controller, 'channel_groups', []) or [])
        if n_groups <= 0:
            raise ValueError('UpperGraphAC requires valid channel_groups')
        self.args = args
        self.n_agents = n_agents
        self.n_groups = n_groups
        hid = getattr(args, 'rnn_hidden_dim', 128)
        self.actor = UpperGraphActor(obs_dim, n_agents, n_groups, args)
        self.critic = UpperGraphCritic(obs_dim, n_agents, args)
        self.actor_hidden = torch.zeros((1, n_agents, hid))
        self.critic_hidden = torch.zeros((1, n_agents, hid))
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=getattr(args, 'hrl_upper_lr', 2e-4))
        self.entropy_coef = getattr(args, 'hrl_upper_entropy', 0.01)
        # Entropy schedule (decay per update) and advantage normalization controls
        self.entropy_min = getattr(args, 'hrl_upper_entropy_min', 0.001)
        self.entropy_decay = getattr(args, 'hrl_upper_entropy_decay', 0.999)
        self.use_adv_norm = getattr(args, 'hrl_upper_adv_norm', True)
        self.adv_clip = getattr(args, 'hrl_upper_adv_clip', 5.0)
        self.beta = getattr(args, 'hrl_upper_beta', 0.7)
        self.memory = []  # store per period tuples
        # Decision histogram tracking: per-agent counts over groups
        self.decisions_hist = np.zeros((n_agents, n_groups), dtype=np.int64)
        self.decisions_count = 0
        self.hrl_controller = hrl_controller
        self.last_cluster_meta: Dict[str, Any] = {}

    def _pad_assignments(self, assignments: List[int]) -> List[int]:
        if len(assignments) >= self.n_agents:
            return assignments[:self.n_agents]
        if not assignments:
            return [0] * self.n_agents
        pad_val = assignments[-1]
        return assignments + [pad_val] * (self.n_agents - len(assignments))

    def _get_cluster_plan(self):
        plan = None
        if hasattr(self.hrl_controller, 'get_cluster_plan'):
            try:
                plan = self.hrl_controller.get_cluster_plan()
            except Exception:
                plan = None
        if plan is None:
            return None
        assigns = plan.get('assignments', [])
        plan = dict(plan)  # shallow copy
        plan['assignments'] = self._pad_assignments(list(assigns))
        return plan

    def _build_cluster_adj(self, adj: Optional[np.ndarray], plan: Optional[Dict[str, Any]]):
        if adj is None:
            base = np.ones((self.n_agents, self.n_agents), dtype=np.float32)
            np.fill_diagonal(base, 0.0)
        else:
            base = np.asarray(adj, dtype=np.float32)
            if base.shape != (self.n_agents, self.n_agents):
                base = np.resize(base, (self.n_agents, self.n_agents))
        if plan is None:
            return torch.tensor(base, dtype=torch.float32)
        assigns = plan.get('assignments', [0] * self.n_agents)
        cluster_adj = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j and assigns[i] == assigns[j]:
                    cluster_adj[i, j] = 1.0
        merged = base * cluster_adj
        return torch.tensor(merged, dtype=torch.float32)

    def decide_groups(self, obs: np.ndarray, adj: Optional[np.ndarray]) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """Return group_ids per agent, logp tensor, and logits."""
        bs = 1
        
        # Check for NaN/Inf in observation
        if np.isnan(obs).any() or np.isinf(obs).any():
            print(f"[WARNING] NaN/Inf in observation before decide_groups, clipping")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        
        X = torch.tensor(obs.reshape(bs * self.n_agents, -1), dtype=torch.float32)
        plan = self._get_cluster_plan()
        self.last_cluster_meta = {
            'plan': plan,
            'cluster_count': len(plan['clusters']) if plan and 'clusters' in plan else 0
        }
        A = self._build_cluster_adj(adj, plan)

        with torch.no_grad():
            logits, self.actor_hidden = self.actor(X, A, self.actor_hidden)
        
        # Additional safety check for logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[ERROR] NaN/Inf in logits! Resetting actor hidden state and using uniform distribution")
            # Reset hidden state
            self.actor_hidden = torch.zeros_like(self.actor_hidden)
            logits = torch.zeros(bs * self.n_agents, self.n_groups)
        
        try:
            pi = torch.distributions.Categorical(logits=logits.reshape(bs, self.n_agents, self.n_groups))
            actions = pi.sample()  # (bs, n_agents)
            logp = pi.log_prob(actions)  # (bs, n_agents)
        except Exception as e:
            print(f"[ERROR] Categorical distribution failed: {e}, using uniform random")
            # Fallback to uniform random
            actions = torch.randint(0, self.n_groups, (bs, self.n_agents))
            logp = torch.ones(bs, self.n_agents) * (-np.log(self.n_groups))
        
        group_ids = actions.squeeze(0).tolist()
        # Update decision histogram
        try:
            for i, gid in enumerate(group_ids):
                self.decisions_hist[i, int(gid)] += 1
            self.decisions_count += 1
        except Exception:
            pass
        return group_ids, logp.squeeze(0), logits

    def evaluate_values(self, obs: np.ndarray, adj: Optional[np.ndarray]):
        bs = 1
        X = torch.tensor(obs.reshape(bs * self.n_agents, -1), dtype=torch.float32)
        A = None
        if adj is not None:
            A = torch.tensor(adj, dtype=torch.float32)
        Vg, Vi, self.critic_hidden = self.critic(X, A, self.critic_hidden)
        return Vg.squeeze(0), Vi.squeeze(0)

    def store_period(self, obs, adj, logp, group_ids, reward_upper, done, logits=None):
        item = {'obs': obs, 'adj': adj, 'logp': logp.detach(), 'group_ids': group_ids, 'R': reward_upper, 'done': done}
        if logits is not None:
            item['logits'] = logits.detach() if hasattr(logits, 'detach') else logits
        self.memory.append(item)

    def update(self):
        if not self.memory:
            return {}
        # Single-period update (can be extended to multi-period minibatch)
        item = self.memory.pop(0)
        obs, adj, logp, R = item['obs'], item['adj'], item['logp'], item['R']
        # Convert reward to tensor and align device
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, dtype=torch.float32)
        device = next(self.actor.parameters()).device
        R = R.to(device)
        # Optional policy entropy from stored logits
        entropy = torch.tensor(0.0, device=device)
        if 'logits' in item and item['logits'] is not None:
            logits = item['logits']
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, dtype=torch.float32, device=device)
            else:
                logits = logits.to(device)
            bs = 1
            pi = torch.distributions.Categorical(logits=logits.reshape(bs, self.n_agents, self.n_groups))
            entropy = pi.entropy().mean()
        with torch.enable_grad():
            Vg, Vi = self.evaluate_values(obs, adj)
            Vg = Vg.to(device)
            Vi = Vi.to(device)
            # Advantage: blend global and local values
            A = R - (self.beta * Vi + (1 - self.beta) * Vg)
            # Advantage normalization (per-period across agents) with optional clipping
            A_norm = A
            A_std_val = torch.tensor(0.0, device=device)
            if self.use_adv_norm:
                A_mean = A.mean()
                A_std = A.std()
                A_std_val = A_std.detach()
                if torch.isfinite(A_std) and (A_std > 1e-6):
                    A_norm = (A - A_mean) / (A_std + 1e-6)
                else:
                    A_norm = A - A_mean
                if self.adv_clip is not None and self.adv_clip > 0:
                    A_norm = torch.clamp(A_norm, -self.adv_clip, self.adv_clip)
            # Policy loss with entropy regularization (scheduled coef)
            loss_pi = -(A_norm.detach() * logp).mean() - self.entropy_coef * entropy
            loss_v = F.mse_loss(Vg, R.mean()) + F.mse_loss(Vi, R)
            loss = loss_pi + loss_v
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
            self.optimizer.step()
        # Entropy coefficient decay (per update)
        try:
            self.entropy_coef = float(max(self.entropy_min, self.entropy_coef * self.entropy_decay))
        except Exception:
            pass
        # Detach recurrent hidden states to avoid backprop across periods
        self.actor_hidden = self.actor_hidden.detach()
        self.critic_hidden = self.critic_hidden.detach()
        stats = {
            'loss_pi': float(loss_pi.item()),
            'loss_v': float(loss_v.item()),
            'A_mean': float(A.mean().item()),
            'A_std': float(A_std_val.item()) if torch.is_tensor(A_std_val) else float(A_std_val),
            'entropy': float(entropy.item()),
            'entropy_coef': float(self.entropy_coef),
        }
        # Cache last stats for external logging (Runner/TensorBoard)
        self.last_stats = stats
        return stats

    def apply_to_controller(self, group_ids: List[int]):
        """Push group decisions to HRLController for fixed_channel_groups masking."""
        if hasattr(self.hrl_controller, 'apply_upper_group_decision'):
            self.hrl_controller.apply_upper_group_decision(group_ids)
        if self.last_cluster_meta:
            self.last_cluster_meta['applied_groups'] = list(group_ids)

    def get_decision_distribution(self, reset: bool = False):
        """Return per-agent group selection distribution and overall mean.
        If reset=True, clear the internal histogram counters after reading.
        """
        if self.decisions_count <= 0:
            per_agent = np.zeros_like(self.decisions_hist, dtype=np.float32)
            overall = np.zeros(self.n_groups, dtype=np.float32)
        else:
            per_agent = (self.decisions_hist / float(self.decisions_count)).astype(np.float32)
            overall = per_agent.mean(axis=0)
        if reset:
            self.decisions_hist[...] = 0
            self.decisions_count = 0
        return {'per_agent': per_agent, 'overall': overall}


class UpperLocalActor(nn.Module):
    """
    Distributed local actor: each agent outputs its group based on
    its own encoded state and an aggregated neighbor feature.
    Aggregator can be mean|max|sum|none controlled by args.hrl_local_agg.
    """
    def __init__(self, obs_dim: int, n_agents: int, n_groups: int, args):
        super().__init__()
        self.n_agents = n_agents
        self.n_groups = n_groups
        self.args = args
        hid = getattr(args, 'rnn_hidden_dim', 128)
        self.enc = nn.Linear(obs_dim, hid)
        self.gru = nn.GRUCell(hid, hid)
        # Local decoder uses concatenated [h_i, agg_i]
        self.head = nn.Sequential(
            nn.Linear(hid * 2, hid), nn.ReLU(), nn.Linear(hid, n_groups)
        )

    def _aggregate(self, h: torch.Tensor, adj: Optional[torch.Tensor], mode: str) -> torch.Tensor:
        """
        h: (bs, n_agents, hid)
        adj: (bs, n_agents, n_agents) or None
        returns agg: (bs, n_agents, hid)
        """
        bs, n_agents, hid = h.shape
        # Build neighbor tensor excluding self
        nbr = []
        for i in range(n_agents):
            others = [j for j in range(n_agents) if j != i]
            nbr_h = torch.stack([h[:, j] for j in others], dim=1)  # (bs, n_agents-1, hid)
            if adj is not None:
                # gate neighbors by adjacency
                gate = torch.stack([adj[:, i, j] for j in others], dim=1).unsqueeze(-1)  # (bs, n_agents-1, 1)
                nbr_h = nbr_h * gate
            if mode == 'mean':
                agg_i = nbr_h.mean(dim=1)
            elif mode == 'max':
                agg_i, _ = nbr_h.max(dim=1)
            elif mode == 'sum':
                agg_i = nbr_h.sum(dim=1)
            else:
                agg_i = torch.zeros((bs, hid), device=h.device, dtype=h.dtype)
            nbr.append(agg_i)
        return torch.stack(nbr, dim=1)  # (bs, n_agents, hid)

    def forward(self, X: torch.Tensor, adj: Optional[torch.Tensor], hidden_state: torch.Tensor):
        size = X.shape[0]
        bs = size // self.n_agents
        hid = hidden_state.shape[-1]
        x_enc = torch.relu(self.enc(X))
        h_in = hidden_state.reshape(-1, hid)
        h_out = self.gru(x_enc, h_in).reshape(bs, self.n_agents, hid)
        # Prepare adjacency with batch if provided
        A = None
        if adj is not None:
            A = adj
            if A.dim() == 2:
                A = A.unsqueeze(0).repeat(bs, 1, 1)
        mode = str(getattr(self.args, 'hrl_local_agg', 'mean')).lower()
        agg = self._aggregate(h_out, A, mode)  # (bs, n_agents, hid)
        final_in = torch.cat([h_out, agg], dim=-1).reshape(-1, hid * 2)
        logits = self.head(final_in)  # (bs*n_agents, n_groups)
        return logits, h_out.reshape(bs * self.n_agents, hid)


class UpperLocalCritic(nn.Module):
    """Local critic: per-agent value with neighbor aggregation; global value as mean of locals."""
    def __init__(self, obs_dim: int, n_agents: int, args):
        super().__init__()
        self.n_agents = n_agents
        hid = getattr(args, 'rnn_hidden_dim', 128)
        self.enc = nn.Linear(obs_dim, hid)
        self.gru = nn.GRUCell(hid, hid)
        self.v_head = nn.Sequential(
            nn.Linear(hid * 2, hid), nn.ReLU(), nn.Linear(hid, 1)
        )
        self.args = args

    def _aggregate(self, h: torch.Tensor, adj: Optional[torch.Tensor], mode: str) -> torch.Tensor:
        bs, n_agents, hid = h.shape
        nbr = []
        for i in range(n_agents):
            others = [j for j in range(n_agents) if j != i]
            nbr_h = torch.stack([h[:, j] for j in others], dim=1)
            if adj is not None:
                gate = torch.stack([adj[:, i, j] for j in others], dim=1).unsqueeze(-1)
                nbr_h = nbr_h * gate
            if mode == 'mean':
                agg_i = nbr_h.mean(dim=1)
            elif mode == 'max':
                agg_i, _ = nbr_h.max(dim=1)
            elif mode == 'sum':
                agg_i = nbr_h.sum(dim=1)
            else:
                agg_i = torch.zeros((bs, hid), device=h.device, dtype=h.dtype)
            nbr.append(agg_i)
        return torch.stack(nbr, dim=1)

    def forward(self, X: torch.Tensor, adj: Optional[torch.Tensor], hidden_state: torch.Tensor):
        size = X.shape[0]
        bs = size // self.n_agents
        hid = hidden_state.shape[-1]
        x_enc = torch.relu(self.enc(X))
        h_in = hidden_state.reshape(-1, hid)
        h_out = self.gru(x_enc, h_in).reshape(bs, self.n_agents, hid)
        A = None
        if adj is not None:
            A = adj
            if A.dim() == 2:
                A = A.unsqueeze(0).repeat(bs, 1, 1)
        mode = str(getattr(self.args, 'hrl_local_agg', 'mean')).lower()
        agg = self._aggregate(h_out, A, mode)
        final_in = torch.cat([h_out, agg], dim=-1)  # (bs, n_agents, 2*hid)
        V_local = self.v_head(final_in.reshape(-1, final_in.shape[-1])).reshape(bs, self.n_agents)
        V_global = V_local.mean(dim=1)
        return V_global, V_local, h_out.reshape(bs * self.n_agents, hid)


class UpperLocalAC:
    """Learner wrapper for UpperLocalActor/Critic with synchronous updates (per meta-period)."""
    def __init__(self, args, hrl_controller, obs_dim: int, n_agents: int):
        n_groups = len(getattr(hrl_controller, 'channel_groups', []) or [])
        if n_groups <= 0:
            raise ValueError('UpperLocalAC requires valid channel_groups')
        self.args = args
        self.n_agents = n_agents
        self.n_groups = n_groups
        hid = getattr(args, 'rnn_hidden_dim', 128)
        self.actor = UpperLocalActor(obs_dim, n_agents, n_groups, args)
        self.critic = UpperLocalCritic(obs_dim, n_agents, args)
        self.actor_hidden = torch.zeros((1, n_agents, hid))
        self.critic_hidden = torch.zeros((1, n_agents, hid))
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=getattr(args, 'hrl_upper_lr', 2e-4))
        self.entropy_coef = getattr(args, 'hrl_upper_entropy', 0.01)
        self.entropy_min = getattr(args, 'hrl_upper_entropy_min', 0.001)
        self.entropy_decay = getattr(args, 'hrl_upper_entropy_decay', 0.999)
        self.use_adv_norm = getattr(args, 'hrl_upper_adv_norm', True)
        self.adv_clip = getattr(args, 'hrl_upper_adv_clip', 5.0)
        self.beta = getattr(args, 'hrl_upper_beta', 0.7)
        self.memory = []
        self.decisions_hist = np.zeros((n_agents, n_groups), dtype=np.int64)
        self.decisions_count = 0
        self.hrl_controller = hrl_controller
        self.last_cluster_meta: Dict[str, Any] = {}

    def _pad_assignments(self, assignments: List[int]) -> List[int]:
        if len(assignments) >= self.n_agents:
            return assignments[:self.n_agents]
        if not assignments:
            return [0] * self.n_agents
        pad_val = assignments[-1]
        return assignments + [pad_val] * (self.n_agents - len(assignments))

    def _get_cluster_plan(self):
        plan = None
        if hasattr(self.hrl_controller, 'get_cluster_plan'):
            try:
                plan = self.hrl_controller.get_cluster_plan()
            except Exception:
                plan = None
        if plan is None:
            return None
        assigns = plan.get('assignments', [])
        plan = dict(plan)
        plan['assignments'] = self._pad_assignments(list(assigns))
        return plan

    def _build_cluster_adj(self, adj: Optional[np.ndarray], plan: Optional[Dict[str, Any]]):
        if adj is None:
            base = np.ones((self.n_agents, self.n_agents), dtype=np.float32)
            np.fill_diagonal(base, 0.0)
        else:
            base = np.asarray(adj, dtype=np.float32)
            if base.shape != (self.n_agents, self.n_agents):
                base = np.resize(base, (self.n_agents, self.n_agents))
        if plan is None:
            return torch.tensor(base, dtype=torch.float32)
        assigns = plan.get('assignments', [0] * self.n_agents)
        cluster_adj = np.zeros((self.n_agents, self.n_agents), dtype=np.float32)
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j and assigns[i] == assigns[j]:
                    cluster_adj[i, j] = 1.0
        merged = base * cluster_adj
        return torch.tensor(merged, dtype=torch.float32)

    def decide_groups(self, obs: np.ndarray, adj: Optional[np.ndarray]):
        bs = 1
        X = torch.tensor(obs.reshape(bs * self.n_agents, -1), dtype=torch.float32)
        plan = self._get_cluster_plan()
        self.last_cluster_meta = {
            'plan': plan,
            'cluster_count': len(plan['clusters']) if plan and 'clusters' in plan else 0
        }
        A = self._build_cluster_adj(adj, plan)
        logits, self.actor_hidden = self.actor(X, A, self.actor_hidden)
        pi = torch.distributions.Categorical(logits=logits.reshape(bs, self.n_agents, self.n_groups))
        actions = pi.sample()
        logp = pi.log_prob(actions)
        group_ids = actions.squeeze(0).tolist()
        try:
            for i, gid in enumerate(group_ids):
                self.decisions_hist[i, int(gid)] += 1
            self.decisions_count += 1
        except Exception:
            pass
        return group_ids, logp.squeeze(0), logits

    def evaluate_values(self, obs: np.ndarray, adj: Optional[np.ndarray]):
        bs = 1
        X = torch.tensor(obs.reshape(bs * self.n_agents, -1), dtype=torch.float32)
        plan = self.last_cluster_meta.get('plan') or self._get_cluster_plan()
        A = self._build_cluster_adj(adj, plan)
        Vg, Vi, self.critic_hidden = self.critic(X, A, self.critic_hidden)
        return Vg.squeeze(0), Vi.squeeze(0)

    def store_period(self, obs, adj, logp, group_ids, reward_upper, done, logits=None):
        item = {'obs': obs, 'adj': adj, 'logp': logp.detach(), 'group_ids': group_ids, 'R': reward_upper, 'done': done}
        if logits is not None:
            item['logits'] = logits.detach() if hasattr(logits, 'detach') else logits
        self.memory.append(item)

    def update(self):
        if not self.memory:
            return {}
        item = self.memory.pop(0)
        obs, adj, logp, R = item['obs'], item['adj'], item['logp'], item['R']
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, dtype=torch.float32)
        device = next(self.actor.parameters()).device
        R = R.to(device)
        entropy = torch.tensor(0.0, device=device)
        if 'logits' in item and item['logits'] is not None:
            logits = item['logits']
            if not isinstance(logits, torch.Tensor):
                logits = torch.tensor(logits, dtype=torch.float32, device=device)
            else:
                logits = logits.to(device)
            bs = 1
            pi = torch.distributions.Categorical(logits=logits.reshape(bs, self.n_agents, self.n_groups))
            entropy = pi.entropy().mean()
        with torch.enable_grad():
            Vg, Vi = self.evaluate_values(obs, adj)
            Vg = Vg.to(device)
            Vi = Vi.to(device)
            A = R - (self.beta * Vi + (1 - self.beta) * Vg)
            A_norm = A
            A_std_val = torch.tensor(0.0, device=device)
            if self.use_adv_norm:
                A_mean = A.mean()
                A_std = A.std()
                A_std_val = A_std.detach()
                if torch.isfinite(A_std) and (A_std > 1e-6):
                    A_norm = (A - A_mean) / (A_std + 1e-6)
                else:
                    A_norm = A - A_mean
                if self.adv_clip is not None and self.adv_clip > 0:
                    A_norm = torch.clamp(A_norm, -self.adv_clip, self.adv_clip)
            loss_pi = -(A_norm.detach() * logp).mean() - self.entropy_coef * entropy
            loss_v = F.mse_loss(Vg, R.mean()) + F.mse_loss(Vi, R)
            loss = loss_pi + loss_v
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
            self.optimizer.step()
        try:
            self.entropy_coef = float(max(self.entropy_min, self.entropy_coef * self.entropy_decay))
        except Exception:
            pass
        self.actor_hidden = self.actor_hidden.detach()
        self.critic_hidden = self.critic_hidden.detach()
        stats = {
            'loss_pi': float(loss_pi.item()),
            'loss_v': float(loss_v.item()),
            'A_mean': float(A.mean().item()),
            'A_std': float(A_std_val.item()) if torch.is_tensor(A_std_val) else float(A_std_val),
            'entropy': float(entropy.item()),
            'entropy_coef': float(self.entropy_coef),
        }
        self.last_stats = stats
        return stats

    def apply_to_controller(self, group_ids):
        if hasattr(self.hrl_controller, 'apply_upper_group_decision'):
            self.hrl_controller.apply_upper_group_decision(group_ids)

    def get_decision_distribution(self, reset: bool = False):
        if self.decisions_count <= 0:
            per_agent = np.zeros_like(self.decisions_hist, dtype=np.float32)
            overall = np.zeros(self.n_groups, dtype=np.float32)
        else:
            per_agent = (self.decisions_hist / float(self.decisions_count)).astype(np.float32)
            overall = per_agent.mean(axis=0)
        if reset:
            self.decisions_hist[...] = 0
            self.decisions_count = 0
        return {'per_agent': per_agent, 'overall': overall}
