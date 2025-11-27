import math
import copy
import os
import numpy as np
import torch


class AdaptiveClusterManager:
    """
    Lightweight自适应分簇器：根据 UAV 位置/邻接生成簇头与成员。
    - 目标：集中式 GAT 控制器可按簇下发资源掩码。
    - 算法：贪心选择簇头（可选 farthest/degree/random），再在半径内吸收邻近节点。
    """

    def __init__(self, args):
        self.args = args
        self.enabled = bool(getattr(args, 'hrl_cluster_enable', False))
        self.target_size = max(1, int(getattr(args, 'hrl_cluster_target_size', 6)))
        self.refresh_period = max(1, int(getattr(args, 'hrl_cluster_refresh', 5)))
        base_radius = getattr(args, 'graph_neighbor_threshold', getattr(args, 'comm_range_km', 1.0))
        self.radius = float(getattr(args, 'hrl_cluster_radius_km', base_radius))
        self.strategy = getattr(args, 'hrl_cluster_head_strategy', 'farthest')
        self.last_step = -1
        self.plan = None
        self.num_clusters = None

    def needs_update(self, step):
        if not self.enabled:
            return False
        if self.plan is None:
            return True
        return (step - self.last_step) >= self.refresh_period

    def _distance_matrix(self, positions):
        pos = np.asarray(positions, dtype=float)
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, 0.0)
        return dist

    def _pick_head(self, unassigned, dist, degrees, heads):
        if len(unassigned) == 0:
            return None
        if self.strategy == 'degree' and degrees is not None:
            return max(unassigned, key=lambda idx: degrees[idx])
        if self.strategy == 'farthest' and heads:
            def min_dist(idx):
                return min(dist[idx, h] for h in heads)

            return max(unassigned, key=min_dist)
        # fallback：取最小索引，稳定可重复
        return min(unassigned)

    def _estimate_cluster_count(self, n_nodes):
        return max(1, int(math.ceil(float(n_nodes) / float(self.target_size))))

    def _build_component_clusters(self, dist):
        n = dist.shape[0]
        mask = (dist <= self.radius).astype(np.int32)
        np.fill_diagonal(mask, 0)
        visited = set()
        clusters = []
        for idx in range(n):
            if idx in visited:
                continue
            stack = [idx]
            visited.add(idx)
            component = []
            while stack:
                v = stack.pop()
                component.append(v)
                neighbors = np.where(mask[v] > 0)[0]
                for nb in neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            component = sorted(component)
            while len(component) > self.target_size:
                clusters.append(component[:self.target_size])
                component = component[self.target_size:]
            if component:
                clusters.append(component)
        if not clusters:
            clusters = [list(range(n))]
        return clusters

    def _fallback_plan(self, n_nodes, step=0):
        k = self._estimate_cluster_count(n_nodes)
        clusters = [[] for _ in range(k)]
        assignments = []
        for idx in range(n_nodes):
            cid = idx % k
            clusters[cid].append(idx)
            assignments.append(cid)
        self.plan = {
            'clusters': clusters,
            'assignments': assignments,
            'heads': [c[0] if c else 0 for c in clusters],
            'step': step
        }
        self.num_clusters = k
        self.last_step = step
        return self.plan

    def update(self, snapshot, step=0):
        if not self.enabled:
            self.plan = None
            return None
        if not self.needs_update(step):
            return self.plan

        positions = None
        if snapshot is not None:
            positions = snapshot.get('positions', None)
        if positions is None:
            # fallback to evenly split clusters
            n_nodes = int(getattr(self.args, 'n_agents', 0))
            if n_nodes <= 0:
                self.plan = None
                return None
            return self._fallback_plan(n_nodes, step=step)

        positions = list(positions)
        n_nodes = len(positions)
        if n_nodes == 0:
            self.plan = None
            return None

        dist = self._distance_matrix(positions)
        adj = np.asarray(snapshot.get('adjacency', np.zeros((n_nodes, n_nodes))), dtype=float)
        degrees = adj.sum(axis=1) if adj.ndim == 2 else None

        clusters = self._build_component_clusters(dist)
        assignments = [-1] * n_nodes
        for cid, members in enumerate(clusters):
            for m in members:
                assignments[m] = cid
        for idx, cid in enumerate(assignments):
            if cid < 0:
                assignments[idx] = 0

        # 统计非空簇个数（至少1）
        non_empty = [c for c in clusters if len(c) > 0]
        if not non_empty:
            return self._fallback_plan(n_nodes, step=step)
        clusters = non_empty
        self.num_clusters = len(clusters)

        self.plan = {
            'clusters': clusters,
            'assignments': assignments,
            'heads': [c[0] if c else 0 for c in clusters],
            'step': step
        }
        self.last_step = step
        return self.plan

    def build_channel_groups(self, channel_num):
        if not self.plan or channel_num <= 0:
            return None
        clusters = self.plan.get('clusters', [])
        if not clusters:
            return None
        n_clusters = len(clusters)
        if n_clusters <= 0:
            return None
        groups = [[] for _ in range(n_clusters)]
        for ch in range(channel_num):
            groups[ch % n_clusters].append(ch)
        # 确保无空组：若有空则追加最后一个信道
        last_ch = channel_num - 1
        for g in groups:
            if not g:
                g.append(last_ch)
        self.num_clusters = n_clusters
        return groups

    def bootstrap(self, n_agents, channel_num):
        if not self.enabled or n_agents <= 0:
            return None
        plan = self._fallback_plan(n_agents, step=0)
        groups = self.build_channel_groups(channel_num)
        if not groups:
            return None
        return {'plan': plan, 'groups': groups, 'assignments': plan.get('assignments', [])}


class HRLController:
    """
    Hierarchical controller for optional action-space gating.
    Deprecation: 'prefer_clean_topk' is disabled for coordination scalability and returns None.
    Recommended: 'fixed_channel_groups' with optional rotation; supports 'mod' or 'custom' assignments.
    If gating would eliminate all currently available actions, returns None to avoid
    disrupting the training/execution flow.
    """

    def __init__(self, args, env=None):
        self.args = args
        self.env = env
        self.masking_mode = getattr(args, 'hrl_masking_mode', 'none')
        self.meta_period = getattr(args, 'hrl_meta_period', 20)
        # Prefer explicit HRL K, otherwise fall back to topk_clean_K if present
        self.topk_k = getattr(args, 'hrl_topk_K', 0) or getattr(args, 'topk_clean_K', None)
        # Fixed groups config (optional)
        self.channel_groups = getattr(args, 'hrl_channel_groups', None)
        self.group_assign_mode = getattr(args, 'hrl_group_assign_mode', 'mod')
        self.group_assignments = getattr(args, 'hrl_group_assignments', None)
        # Rotation options for fixed groups
        self.group_rotate = bool(getattr(args, 'hrl_group_rotate', False))
        self.group_rotate_period = int(getattr(args, 'hrl_group_rotate_period', self.meta_period))
        # Optional timeslot scheduling interface (Phase 2 placeholder)
        self.timeslot_enable = bool(getattr(args, 'hrl_timeslot_enable', False))
        self.timeslot_period = int(getattr(args, 'hrl_timeslot_period', self.meta_period))
        self.timeslot_allocator = None  # callable: (t, agent_num, graph_state) -> int or None
        try:
            if (self.channel_groups is None) and hasattr(args, 'hrl_channel_groups_json'):
                import json
                if getattr(args, 'hrl_channel_groups_json', ''):
                    self.channel_groups = json.loads(args.hrl_channel_groups_json)
            if (self.group_assignments is None) and hasattr(args, 'hrl_group_assignments_json'):
                import json
                if getattr(args, 'hrl_group_assignments_json', ''):
                    self.group_assignments = json.loads(args.hrl_group_assignments_json)
        except Exception:
            pass
        # Dimensions inferred from args lists to remain robust to configuration
        self.n_agents = int(getattr(args, 'n_agents', 0))
        self.ra_len = len(getattr(args, 'RA_action', [])) or 1
        self.pt_len = len(getattr(args, 'Pt_action', [])) or 1
        self.chan_len = len(getattr(args, 'FH_action', [])) or getattr(args, 'channel_num', 0) or 0
        self.n_actions = getattr(args, 'n_actions', self.chan_len * self.ra_len * self.pt_len)
        # Auto-generate groups when using fixed_channel_groups without explicit config
        if self.masking_mode == 'fixed_channel_groups' and (not self.channel_groups):
            auto_groups = self._auto_channel_groups_from_env()
            if auto_groups:
                self.channel_groups = auto_groups

        # Internal counters for potential periodic updates
        self._call_count = 0
        self._last_update_call = -1
        self._cached_mask = None
        # 自适应分簇管理（集中式簇头 GAT）
        self.cluster_manager = AdaptiveClusterManager(args)
        self.last_cluster_plan = None
        self.cluster_debug = bool(getattr(args, 'cluster_debug', False))
        self.cluster_debug_interval = max(1, int(getattr(args, 'cluster_debug_interval', 50)))
        if self.cluster_manager.enabled and (not self.channel_groups):
            bootstrap = self.cluster_manager.bootstrap(self.n_agents, self.chan_len)
            if bootstrap:
                self.channel_groups = bootstrap['groups']
                self.group_assignments = list(bootstrap.get('assignments', []))
                self.group_assign_mode = 'custom'
                self.masking_mode = 'fixed_channel_groups'
                self.last_cluster_plan = bootstrap.get('plan')

    def _compute_clean_channels_from_ewma(self, K):
        """Return indices of Top-K clean channels from env EWMA stats."""
        if self.env is None:
            return None
        try:
            moving_avg, ewma = self.env.get_interference_stats()
        except Exception:
            return None
        if ewma is None:
            return None
        ewma = np.asarray(ewma)
        if ewma.ndim == 0:
            return None
        k = K if K is not None and K > 0 else min(5, len(ewma))
        k = int(max(1, min(k, len(ewma))))
        # Smaller EWMA => cleaner channel
        clean_idx = np.argsort(ewma)[:k]
        return set(int(i) for i in clean_idx.tolist())

    def _build_mask_from_channel_set(self, allowed_channels):
        """Construct a length-n_actions mask allowing only actions on given channels."""
        if allowed_channels is None or self.chan_len <= 0:
            return None
        mask = np.ones(self.n_actions, dtype=np.float32)
        stride = self.ra_len * self.pt_len
        # Actions are enumerated as: channel * stride + mcs * pt_len + pt
        for a in range(self.n_actions):
            ch = a // stride
            if ch not in allowed_channels:
                mask[a] = 0.0
        return mask

    def _compute_channels_fixed_group(self, agent_num):
        groups = self.channel_groups
        if not groups or self.chan_len <= 0:
            return None
        try:
            n_groups = len(groups)
            if n_groups <= 0:
                return None
            if self.group_assign_mode == 'custom' and self.group_assignments:
                if agent_num is None or agent_num >= len(self.group_assignments):
                    return None
                g = int(self.group_assignments[agent_num])
            else:
                g_base = int(agent_num or 0) % n_groups
                # Optional rotation: periodically shift group assignment
                if self.group_rotate and self.group_rotate_period > 0:
                    rotation_epoch = int(self._call_count // max(1, self.group_rotate_period))
                    g = (g_base + rotation_epoch) % n_groups
                else:
                    g = g_base
            allowed = set(int(ch) for ch in groups[g])
            # clamp to valid channel range
            allowed = set([ch for ch in allowed if 0 <= ch < self.chan_len])
            return allowed if len(allowed) > 0 else None
        except Exception:
            return None

    def get_action_mask(self, agent_num=None, obs=None, q_value=None, avail_actions=None):
        """
        Return an action mask (length n_actions, 1/0) or None.
        - prefer_clean_topk: compute Top-K clean channels from EWMA and mask actions accordingly
        - fixed_channel_groups: restrict to preconfigured channel groups based on agent assignment
        - none or unknown mode: return None
        If mask would remove all currently available actions, return None.
        """
        self._call_count += 1

        if self.masking_mode == 'prefer_clean_topk':
            # Deprecated: Top-K gating is disabled for coordination scalability.
            return None
        elif self.masking_mode == 'fixed_channel_groups':
            allowed_channels = self._compute_channels_fixed_group(agent_num)
        else:
            return None

        mask = self._build_mask_from_channel_set(allowed_channels)
        if mask is None:
            return None

        # If avail_actions provided, ensure feasibility
        try:
            if avail_actions is not None:
                avail = avail_actions
                if torch.is_tensor(avail):
                    avail = avail.detach().cpu().numpy()
                avail = np.asarray(avail).reshape(-1)
                # q_value may be shape (1, n_actions); we flatten to match
                feasible = (avail > 0).astype(np.float32) * mask
                if feasible.sum() <= 0:
                    # Avoid eliminating all options: fallback to None
                    return None
        except Exception:
            # Be tolerant to shape mismatches; if anything goes wrong, skip masking
            return None

        return mask

    # --- Timeslot scheduling placeholder (Phase 2) ---
    def set_timeslot_allocator(self, allocator):
        """
        Register an external timeslot allocator.
        allocator signature: fn(t, agent_num, graph_state) -> int or None.
        """
        self.timeslot_allocator = allocator

    def get_timeslot_decision(self, agent_num=None, t=None, graph_state=None):
        """
        Placeholder for timeslot/rotation offset decision.
        Returns int offset in [0, self.timeslot_period-1] or None.
        No-op by default; does not affect masks.
        """
        if not self.timeslot_enable or self.timeslot_allocator is None:
            return None
        try:
            return self.timeslot_allocator(t, agent_num, graph_state)
        except Exception:
            return None

    def apply_upper_group_decision(self, group_ids):
        """
        Apply per-agent group assignments provided by an upper-layer policy.
        Switches to 'custom' assignment mode and updates internal mapping.
        """
        try:
            if group_ids is None:
                return False
            gids = list(int(g) for g in group_ids)
            # Optional: validate length
            n_agents = int(getattr(self.args, 'n_agents', len(gids)))
            if len(gids) < n_agents:
                # pad with modulo groups if shorter
                if self.channel_groups:
                    n_groups = len(self.channel_groups)
                    gids = gids + [i % max(1, n_groups) for i in range(n_agents - len(gids))]
                else:
                    gids = gids + [0 for _ in range(n_agents - len(gids))]
            elif len(gids) > n_agents:
                gids = gids[:n_agents]
            # Clamp to valid group range
            if self.channel_groups:
                n_groups = len(self.channel_groups)
                gids = [max(0, min(int(g), n_groups - 1)) for g in gids]
            self.group_assignments = gids
            self.group_assign_mode = 'custom'
            # Invalidate cached mask if any
            self._cached_mask = None
            return True
        except Exception:
            return False

    def build_upper_graph_state(self, positions=None, threshold=None, prev_adj=None):
        """
        Optional helper to construct graph/adjacency for upper-layer decisions.
        - positions: list/array of coordinates; if None, tries env positions.
        - threshold: distance threshold; if None, uses env default if available.
        Returns dict: { 'adj': np.ndarray, 'graph': optional }
        """
        try:
            from network.graph_utils import build_distance_graph, smooth_adjacent
        except Exception:
            return None
        try:
            graph_snapshot = None
            if positions is None and self.env is not None and hasattr(self.env, 'get_graph_snapshot'):
                graph_snapshot = self.env.get_graph_snapshot()
                if graph_snapshot is not None:
                    positions = [(p[0], p[1]) for p in graph_snapshot.get('positions', [])]
                    if threshold is None:
                        threshold = graph_snapshot.get('threshold')
                    adj_from_snapshot = graph_snapshot.get('adjacency')
                    if adj_from_snapshot is not None:
                        return {'adj': np.asarray(adj_from_snapshot, dtype=np.int32), 'graph': None}
            if positions is None:
                if self.env is not None and hasattr(self.env, 'get_positions'):
                    positions = self.env.get_positions()
            if positions is None:
                return None
            if threshold is None:
                threshold = getattr(self.args, 'hrl_graph_threshold', None) or getattr(self.env, 'neighbor_threshold', None) or 0.0
            adj, G = build_distance_graph(positions, threshold, directed=False, return_graph=True)
            if adj is None:
                return None
            adj_sm = smooth_adjacent(adj, alpha=0.8, prev_adj=prev_adj)
            return {'adj': adj_sm, 'graph': G}
        except Exception:
            return None

    def update_cluster_plan(self, graph_snapshot=None, step=0):
        """Update cluster-channel 分配，驱动 fixed_channel_groups 自动生成。"""
        if not (self.cluster_manager and self.cluster_manager.enabled):
            return False
        if graph_snapshot is None and self.env is not None and hasattr(self.env, 'get_graph_snapshot'):
            graph_snapshot = self.env.get_graph_snapshot()
        plan = self.cluster_manager.update(graph_snapshot, step=step)
        if plan is None:
            n_agents = int(getattr(self.args, 'n_agents', 0))
            if n_agents > 0:
                plan = self.cluster_manager._fallback_plan(n_agents, step=step)
            else:
                return False
        if plan is None:
            return False
        groups = self.cluster_manager.build_channel_groups(self.chan_len)
        if not groups and self.channel_groups:
            groups = self.channel_groups
        if not groups:
            return False
        assignments = plan.get('assignments', [])
        n_agents = int(getattr(self.args, 'n_agents', len(assignments)))
        if len(assignments) < n_agents:
            assignments = assignments + [assignments[-1] if assignments else 0] * (n_agents - len(assignments))
        elif len(assignments) > n_agents:
            assignments = assignments[:n_agents]
        self.channel_groups = groups
        self.group_assignments = [int(a) % max(1, len(groups)) for a in assignments]
        self.group_assign_mode = 'custom'
        self.masking_mode = 'fixed_channel_groups'
        self._cached_mask = None
        self.last_cluster_plan = plan
        if self.cluster_debug and (step % self.cluster_debug_interval == 0):
            clusters = plan.get('clusters', [])
            sizes = [len(c) for c in clusters]
            unique_ids = len(set(self.group_assignments)) if self.group_assignments else 0
            preview = clusters[:2]
            print(f"[ClusterDebug] step={step} clusters={len(sizes)} unique_groups={unique_ids} sizes={sizes} preview={preview}")
        return True

    def get_cluster_plan(self):
        if self.last_cluster_plan is None:
            return None
        return copy.deepcopy(self.last_cluster_plan)

    def _auto_channel_groups_from_env(self):
        try:
            if self.env is None:
                return None
            channel_num = int(getattr(self.env, 'channel_num', self.chan_len) or self.chan_len)
            sub_band = int(getattr(self.env, 'sub_band', 0) or 0)
            alg = getattr(self.args, 'alg', None)
            if channel_num <= 0:
                return None
            # Prefer 5-channel bands for no_mcs variants
            if alg in ('iql_no_mcs', 'qmix_no_mcs') and channel_num % 5 == 0:
                per_group = 5
                n_groups = channel_num // per_group
                return [list(range(g * per_group, (g + 1) * per_group)) for g in range(n_groups)]
            # Align with env sub_band if evenly divisible
            if sub_band and sub_band > 0 and channel_num % sub_band == 0:
                per_group = channel_num // sub_band
                return [list(range(g * per_group, (g + 1) * per_group)) for g in range(sub_band)]
            # Fallback: contiguous size-5 groups if possible
            if channel_num % 5 == 0:
                per_group = 5
                n_groups = channel_num // per_group
                return [list(range(g * per_group, (g + 1) * per_group)) for g in range(n_groups)]
            return None
        except Exception:
            return None
