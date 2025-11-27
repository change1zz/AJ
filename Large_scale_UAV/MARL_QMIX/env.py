"""
模块化通信抗干扰环境。

WirelessSimulator 负责底层物理建模（拓扑、路径损耗、干扰、SINR 等），
ComEnv 提供 RL 训练接口并与 Simulator 交互。

该文件兼容原 runner 调用方式，保留 get_state/get_obs/get_avail_actions 等 API。
"""
from __future__ import annotations

import copy
import math
import os
import random
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from node import init_G, calculate_lbs

try:
    from network.graph_utils import build_distance_graph, smooth_adjacent
except Exception:  # pragma: no cover - 可选依赖
    build_distance_graph = None
    smooth_adjacent = None


# ---------------------------------------------------------------------------
# 辅助缓存：拓扑数据
# ---------------------------------------------------------------------------


class TopologyCache:
    """按 sample_* 目录缓存拓扑信息，减少反复 IO。"""

    def __init__(self, dataset_dir: str, cache_size: int, lock_sample: bool, sample_name: str):
        self.dataset_dir = dataset_dir
        self.cache_size = cache_size
        self.lock_sample = lock_sample
        self.sample_name = sample_name
        self.cache: List[Dict] = []
        self._locked_entry: Optional[Dict] = None
        self._preload()

    def _preload(self) -> None:
        if not self.dataset_dir or not os.path.isdir(self.dataset_dir):
            return
        sample_dirs = [
            os.path.join(self.dataset_dir, d)
            for d in os.listdir(self.dataset_dir)
            if d.startswith("sample_") and os.path.isdir(os.path.join(self.dataset_dir, d))
        ]
        sample_dirs.sort()
        for path in sample_dirs:
            if len(self.cache) >= self.cache_size:
                break
            entry = self._load_entry(path)
            if entry:
                self.cache.append(entry)

    def _load_entry(self, sample_dir: str) -> Optional[Dict]:
        try:
            pos_path = os.path.join(sample_dir, "comm_pos3d.csv")
            if not os.path.isfile(pos_path):
                return None
            positions = []
            with open(pos_path, "r", newline="") as f:
                for row in csv.reader(f):
                    if len(row) < 3:
                        continue
                    try:
                        positions.append((float(row[0]), float(row[1]), float(row[2])))
                    except Exception:
                        continue
            if not positions:
                return None
            adj_path = os.path.join(sample_dir, "comm_adj.csv")
            adjacency = None
            if os.path.isfile(adj_path):
                mat = []
                with open(adj_path, "r", newline="") as f:
                    for row in csv.reader(f):
                        vals = []
                        for cell in row:
                            cell = cell.strip()
                            if not cell:
                                continue
                            try:
                                vals.append(int(float(cell)))
                            except Exception:
                                vals.append(0)
                        if vals:
                            mat.append(vals)
                if mat:
                    adjacency = mat
            return {"dir": sample_dir, "name": os.path.basename(sample_dir), "positions": positions, "adjacency": adjacency}
        except Exception:
            return None

    def sample(self) -> Optional[Dict]:
        if not self.cache:
            return None
        if self.lock_sample:
            if self._locked_entry is None:
                self._locked_entry = self._pick_entry()
            return self._locked_entry
        return self._pick_entry()

    def _pick_entry(self) -> Dict:
        if self.sample_name:
            for entry in self.cache:
                if entry["name"] == self.sample_name:
                    return entry
        return random.choice(self.cache)


# ---------------------------------------------------------------------------
# 干扰模型
# ---------------------------------------------------------------------------


class JammerModel:
    """负责生成干扰轮廓，支持多种样式与响应式逻辑。"""

    def __init__(self, args, channel_num: int):
        self.args = args
        self.channel_num = channel_num
        self.jam_power = [args.jam_origin_power, args.jam_origin_power]
        self.jam_sum_channel = [args.jam_sum_channel, args.jam_sum_channel]
        self.alt_prob = getattr(args, "alt_jam_occ_prob", 0.2)
        self.jam_change_time = int(getattr(args, "jam_change_time", 1))
        self.alt_change_time = int(getattr(args, "alt_jam_change_time", 5))
        self.trace_time = 0
        self.actions_history = None
        self.types = []
        self.jam_1 = np.zeros(channel_num, dtype=float)
        self.jam_2 = np.zeros(channel_num, dtype=float)

    def reset(self, jam_types: List[str], trace_span: int, nodes: int) -> None:
        self.types = jam_types
        self.trace_time = trace_span
        if "trace_jam" in jam_types and trace_span > 0:
            self.actions_history = np.zeros([trace_span + 1, nodes, 3])
        else:
            self.actions_history = None
        self.jam_1.fill(0.0)
        self.jam_2.fill(0.0)

    def update_history(self, actions_trans: np.ndarray) -> None:
        if self.actions_history is None:
            return
        for i in range(self.actions_history.shape[0] - 1):
            self.actions_history[i] = self.actions_history[i + 1]
        self.actions_history[self.trace_time] = np.array(actions_trans)

    def _generate_pattern(self, jam_type: str, time_step: int, jam_index: int) -> np.ndarray:
        num_channel = self.jam_sum_channel[jam_index - 1]
        result = np.zeros(self.channel_num)
        if num_channel <= 0:
            return result
        if jam_type == "narrow_band":
            result[:num_channel] = 1
            np.random.shuffle(result)
            return result
        if jam_type == "comb_band":
            start = random.randint(0, max(1, self.channel_num - num_channel))
            idx = np.linspace(start, self.channel_num - 1, num_channel, dtype=int)
            result[idx] = 1
            return result
        if jam_type == "wide_band":
            start = random.randint(0, self.channel_num - num_channel)
            result[start:start + num_channel] = 1
            return result
        if jam_type == "linear_sweep":
            interval = max(1, self.channel_num // num_channel)
            for i in range(num_channel):
                result[(time_step + i * interval) % self.channel_num] = 1
            return result
        if jam_type == "trace_jam" and self.actions_history is not None:
            trace = np.ones(self.channel_num)
            for agent_id in range(self.actions_history.shape[1]):
                ch0 = int(self.actions_history[0][agent_id][0])
                ch1 = int(self.actions_history[-1][agent_id][0])
                trace[ch0] += random.randint(1_000_000, 8_000_000)
                trace[ch1] += random.randint(1_000_000, 8_000_000)
            return 10 * np.log10(trace)
        return result

    def sample(self, time_step: int, graph) -> None:
        for idx in (1, 2):
            pattern = self._generate_pattern(self.types[idx - 1], time_step, idx)
            amp = np.random.uniform(59, self.jam_power[idx - 1], size=self.channel_num)
            if idx == 1:
                self.jam_1 = pattern * amp
            else:
                self.jam_2 = pattern * amp


# ---------------------------------------------------------------------------
# 物理仿真模块
# ---------------------------------------------------------------------------


class WirelessSimulator:
    """集中封装所有物理层计算，供 ComEnv 查询。"""

    def __init__(self, args):
        self.args = args
        self.nodes = int(args.nodes)
        self.channel_num = int(args.channel_num)
        self.jam_num = int(args.jam_num)
        self.sub_band = getattr(args, "sub_band", 8)
        self.channels_per_subband = getattr(args, "channels_per_subband", max(1, self.channel_num // max(1, self.sub_band)))
        self.power_level_num = getattr(args, "power_level_num", len(args.Pt_action))
        self.packet_num = args.packet_num
        self.bler_data = args.bler_data
        self.interf_hist: List[np.ndarray] = []
        self.interf_ewma = np.zeros(self.channel_num, dtype=float)
        self.jammer = JammerModel(args, self.channel_num)
        dataset_dir = getattr(args, "topology_dataset_dir", "")
        cache_size = int(getattr(args, "topology_dataset_cache_size", 64))
        self.topology_cache = TopologyCache(
            dataset_dir=dataset_dir,
            cache_size=cache_size,
            lock_sample=bool(getattr(args, "topology_dataset_lock_sample", False)),
            sample_name=str(getattr(args, "topology_dataset_sample", "") or "")
        )
        self.G = None
        self.graph_snapshot = None
        self._graph_prev_adj = None
        self.comm_neighbors: Dict[int, List[int]] = {}
        self.comm_adj = None
        self.channel = np.zeros(self.channel_num)
        self.avail_band = np.zeros([self.nodes, 2], dtype=int)
        self.avail_obs = np.array([list(range(self.sub_band)) for _ in range(self.nodes)], dtype=int)
        self.current_indices = np.zeros(self.nodes, dtype=int)
        self.target_node = np.zeros(self.nodes, dtype=int)
        self.last_actions = np.zeros([self.nodes, 3], dtype=int)
        self.receive_channel = np.zeros([self.nodes, self.channel_num])
        channel_bandwidth_hz = int(self.args.channel_bandwidth * 1e6)
        ndBm = -174 + 10 * math.log10(channel_bandwidth_hz) + getattr(self.args, "noise_figure", 7)
        self.N_mW = 10 ** (ndBm / 10)
        self.jam_types = random.sample(self.args.jam_type, 2)

    # -------------------- 拓扑与图 --------------------
    def _sample_topology(self):
        entry = self.topology_cache.sample()
        if not entry:
            self.G = init_G(self.nodes)
            return
        positions = entry["positions"]
        node_data = {i: positions[i] for i in range(self.nodes)}
        self.G = init_G(self.nodes, node_data=node_data)

    def _build_comm_topology(self):
        try:
            use_3d = all(("pos3d" in self.G.nodes[i]) for i in range(self.nodes))
            if use_3d:
                pos = np.array([
                    [float(self.G.nodes[i]["pos3d"][0]), float(self.G.nodes[i]["pos3d"][1]), float(self.G.nodes[i]["pos3d"][2])]
                    for i in range(self.nodes)
                ])
            else:
                pos = np.array([
                    [float(self.G.nodes[i]["pos"][0]), float(self.G.nodes[i]["pos"][1])]
                    for i in range(self.nodes)
                ])
            diff = pos[:, None, :] - pos[None, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dist, np.inf)
            R = float(getattr(self.args, "comm_range_km", 1.0))
            adj = (dist <= R).astype(int)
            adj = adj - np.eye(self.nodes, dtype=int)
            self.comm_adj = adj
            self.comm_neighbors = {i: list(np.where(adj[i] == 1)[0]) for i in range(self.nodes)}
        except Exception:
            self.comm_adj = np.ones((self.nodes, self.nodes), dtype=int) - np.eye(self.nodes, dtype=int)
            self.comm_neighbors = {i: [j for j in range(self.nodes) if j != i] for i in range(self.nodes)}

    def _update_graph_snapshot(self):
        if not getattr(self.args, "use_graph_obs", False):
            self.graph_snapshot = None
            return
        if self.G is None or build_distance_graph is None:
            self.graph_snapshot = None
            return
        pos = []
        for idx in range(self.nodes):
            attrs = self.G.nodes[idx]
            if "pos3d" in attrs:
                pos.append(tuple(map(float, attrs["pos3d"])))
            else:
                xy = attrs.get("pos", (0.0, 0.0))
                pos.append((float(xy[0]), float(xy[1]), 0.0))
        threshold = float(getattr(self.args, "graph_neighbor_threshold", getattr(self.args, "comm_range_km", 1.0)))
        xy = [(p[0], p[1]) for p in pos]
        adj, _ = build_distance_graph(xy, threshold, return_graph=False) if build_distance_graph else (None, None)
        if adj is not None:
            adj = np.asarray(adj, dtype=np.int32)
            alpha = float(getattr(self.args, "graph_obs_smooth_alpha", 0.8))
            if smooth_adjacent is not None:
                adj = smooth_adjacent(adj, alpha=alpha, prev_adj=self._graph_prev_adj)
            self._graph_prev_adj = adj
        self.graph_snapshot = {"positions": pos, "adjacency": adj, "threshold": threshold}

    # -------------------- 重置与观测 --------------------
    def reset(self, test: bool = False):
        self.channel = np.random.normal(-100, 10, size=self.channel_num)
        self.interf_hist.clear()
        self.interf_ewma[:] = 0.0
        self._sample_topology()
        self._build_comm_topology()
        self._update_graph_snapshot()
        for i in range(self.nodes):
            candidates = [v for v in range(self.nodes) if v != i]
            self.target_node[i] = random.choice(candidates)
        jam_types = random.sample(self.args.jam_type, 2) if not test else list(self.args.jam_type[:2])
        self.jam_types = jam_types
        self.jammer.reset(jam_types, trace_span=int(getattr(self.args, "trace_time", 0)), nodes=self.nodes)
        self.last_actions[:] = 0
        self._reset_sampling_indices()
        self.jammer.sample(0, self.G)

    def _reset_sampling_indices(self):
        for agent_id in range(self.nodes):
            self.current_indices[agent_id] = 0
            np.random.shuffle(self.avail_obs[agent_id])

    def build_state(self) -> np.ndarray:
        jam_view = np.maximum(self.jammer.jam_1, self.jammer.jam_2)
        state = np.concatenate([jam_view, self.channel])
        jam_type_vec = np.zeros(len(self.args.jam_type))
        for jt in self.jam_types:
            if jt in self.args.jam_type:
                jam_type_vec[self.args.jam_type.index(jt)] = 1
        geom = []
        for i in range(self.nodes + self.jam_num):
            pos = self.G.nodes[i]["pos"]
            geom.extend([float(pos[0]), float(pos[1])])
        state = np.concatenate([state, jam_type_vec, np.array(geom, dtype=float)])
        if self.args.alg in ("qmix", "qmix_attention", "qmix_no_mcs"):
            x_min, x_max = np.min(state), np.max(state)
            if x_max > x_min:
                state = (state - x_min) / (x_max - x_min)
        return state

    def _sample_band(self, agent_id: int) -> int:
        if self.current_indices[agent_id] == self.sub_band:
            self.current_indices[agent_id] = 0
            np.random.shuffle(self.avail_obs[agent_id])
        band = int(self.avail_obs[agent_id, self.current_indices[agent_id]])
        self.current_indices[agent_id] += 1
        return band

    def build_observation(self, time_step: int) -> np.ndarray:
        obs = np.zeros((self.nodes, 15), dtype=float)
        for agent_id in range(self.nodes):
            busy = np.zeros(self.sub_band, dtype=int)
            for j in range(self.channel_num):
                jammer_id1 = self.nodes
                jammer_id2 = self.nodes + 1
                jam1 = self.jammer.jam_1[j]
                jam2 = self.jammer.jam_2[j]
                if jam1 and jam2:
                    val = max(
                        jam1 - self.G.edges[agent_id, jammer_id1]["lbs"][j],
                        jam2 - self.G.edges[agent_id, jammer_id2]["lbs"][j]
                    ) + 33
                elif jam1:
                    val = jam1 - self.G.edges[agent_id, jammer_id1]["lbs"][j]
                elif jam2:
                    val = jam2 - self.G.edges[agent_id, jammer_id2]["lbs"][j]
                else:
                    val = self.channel[j]
                self.receive_channel[agent_id, j] = val
            if time_step == 0:
                self.avail_band[agent_id][0] = self._sample_band(agent_id)
                self.avail_band[agent_id][1] = self._sample_band(agent_id)
            else:
                self.avail_band[agent_id][0] = self.last_actions[agent_id][0] // 5
                self.avail_band[agent_id][1] = self._sample_band(agent_id)
                while self.avail_band[agent_id][1] == self.avail_band[agent_id][0]:
                    self.avail_band[agent_id][1] = self._sample_band(agent_id)
                for i in range(self.nodes):
                    if i != agent_id:
                        busy[int(self.last_actions[i][0] // 5)] += 1
            b0, b1 = self.avail_band[agent_id]
            obs[agent_id][0] = busy[b0]
            obs[agent_id][1:6] = self.receive_channel[agent_id][b0 * 5:b0 * 5 + 5]
            obs[agent_id][6] = busy[b1]
            obs[agent_id][7:12] = self.receive_channel[agent_id][b1 * 5:b1 * 5 + 5]
            obs[agent_id][12:14] = self.avail_band[agent_id]
            obs[agent_id][14] = self.G.edges[agent_id, self.target_node[agent_id]]["distance"]
        if self.args.alg in ("qmix", "qmix_attention", "qmix_no_mcs") and np.max(obs) > np.min(obs):
            x_min, x_max = np.min(obs), np.max(obs)
            obs = (obs - x_min) / (x_max - x_min)
        return obs

    # -------------------- 动作可用性 --------------------
    def get_avail_agent_actions(self, agent_id: int) -> np.ndarray:
        avail = np.zeros(self.args.n_actions, dtype=int)
        if self.args.alg in ("iql_no_mcs", "qmix_no_mcs"):
            for band in self.avail_band[agent_id]:
                start = band * 5
                end = start + 5
                for channel in range(start, end):
                    for pt in range(len(self.args.Pt_action)):
                        idx = channel * len(self.args.Pt_action) + pt
                        avail[idx] = 1
        else:
            span = self.channel_num // max(1, self.sub_band)
            for band in self.avail_band[agent_id]:
                start = int(band * span * len(self.args.RA_action) * len(self.args.Pt_action))
                end = int(start + span * len(self.args.RA_action) * len(self.args.Pt_action))
                avail[start:end] = 1
        return avail

    def get_avail_actions(self) -> List[np.ndarray]:
        return [self.get_avail_agent_actions(i) for i in range(self.nodes)]

    # -------------------- 物理步进 --------------------
    def _sinr_linear(self, agent_id: int, ch_idx: int, pt_dbm: float) -> float:
        target = int(self.target_node[agent_id])
        path_loss = self.G.edges[agent_id, target]["lbs"][ch_idx]
        signal = 10 ** (((pt_dbm + getattr(self.args, "gt", 3)) - path_loss) / 10)
        interf = self.N_mW
        if self.jammer.jam_1[ch_idx]:
            jammer_id = self.nodes
            loss = self.G.edges[agent_id, jammer_id]["lbs"][ch_idx]
            interf += 10 ** ((self.jammer.jam_1[ch_idx] - loss) / 10)
        if self.jammer.jam_2[ch_idx]:
            jammer_id = self.nodes + 1
            loss = self.G.edges[agent_id, jammer_id]["lbs"][ch_idx]
            interf += 10 ** ((self.jammer.jam_2[ch_idx] - loss) / 10)
        return signal / max(interf, 1e-12)

    def step_physics(self, actions_trans: np.ndarray, time_step: int) -> Dict[str, np.ndarray]:
        if getattr(self.args, "use_alt_jam_model", False) and (time_step % self.jammer.alt_change_time == 0):
            self._update_alt_jamming(actions_trans)
        elif time_step % self.jammer.jam_change_time == 0:
            self.jammer.sample(time_step, self.G)
        if getattr(self.args, "use_reactive_jam", False):
            self._update_reactive_jam(actions_trans)
        if getattr(self.args, "use_intelligent_jam", False):
            self._update_intelligent_jam(time_step)
        sinr_values = np.zeros(self.nodes, dtype=float)
        bler = np.zeros(self.nodes, dtype=float)
        change_flag = np.zeros(self.nodes, dtype=float)
        collision = np.zeros(self.nodes, dtype=float)
        for agent_id in range(self.nodes):
            ch = int(actions_trans[agent_id][0])
            mcs = int(actions_trans[agent_id][1])
            pt = int(actions_trans[agent_id][2])
            sinr = self._sinr_linear(agent_id, ch, self.args.Pt_action[pt])
            sinr_db = 10 * math.log10(max(sinr, 1e-12))
            sinr_values[agent_id] = sinr_db
            bler[agent_id] = self.calculate_bler(sinr_db, self.args.RA_action[mcs])
            if time_step > 0 and (
                actions_trans[agent_id][0] != self.last_actions[agent_id][0]
                or actions_trans[agent_id][1] != self.last_actions[agent_id][1]
            ):
                change_flag[agent_id] = 1
            neighbors = self.comm_neighbors.get(agent_id, [])
            for nb in neighbors:
                if int(actions_trans[nb][0]) == ch:
                    collision[agent_id] = 1
                    break
        self.last_actions = actions_trans.copy()
        self._update_interference_stats()
        return {"sinr_db": sinr_values, "bler": bler, "change_flag": change_flag, "collision": collision}

    def calculate_bler(self, sinr_db: float, code_mode: str) -> float:
        data = self.bler_data[str(code_mode)]
        idx = int(np.clip(round((sinr_db - (-10)) / 0.1), 0, len(data) - 1))
        return float(data[idx])

    def _update_alt_jamming(self, actions_trans: np.ndarray):
        usage = np.zeros(self.channel_num)
        stride = len(self.args.RA_action) * len(self.args.Pt_action)
        for act in actions_trans:
            ch = int(act[0])
            if 0 <= ch < self.channel_num:
                usage[ch] += 1
        num = max(1, int(self.channel_num * self.jammer.alt_prob))
        top = np.argsort(-usage)[:num]
        occ = np.zeros(self.channel_num)
        occ[top] = 1
        amp = np.random.uniform(59, self.args.jam_origin_power, size=self.channel_num)
        self.jammer.jam_1 = occ * amp
        self.jammer.jam_2 = occ * amp

    def _update_reactive_jam(self, actions_trans: np.ndarray):
        p = float(getattr(self.args, "reactive_jam_prob", 0.7))
        amp = float(getattr(self.args, "reactive_jam_amp_dbm", 64.0))
        self.jammer.jam_1 *= 0
        self.jammer.jam_2 *= 0
        for act in actions_trans:
            ch = int(act[0])
            if 0 <= ch < self.channel_num:
                if random.random() < p:
                    self.jammer.jam_1[ch] = amp
                if random.random() < p:
                    self.jammer.jam_2[ch] = amp

    def _update_intelligent_jam(self, time_step: int):
        change_T = int(getattr(self.args, "intel_jam_change_time", 10))
        if time_step % change_T != 0:
            return
        K = int(getattr(self.args, "intel_jam_topk", 5))
        amp = float(getattr(self.args, "reactive_jam_amp_dbm", 64.0))
        usage = np.zeros(self.channel_num)
        if hasattr(self, "last_actions"):
            for act in self.last_actions:
                ch = int(act[0])
                if 0 <= ch < self.channel_num:
                    usage[ch] += 1
        top = np.argsort(-usage)[:K]
        self.jammer.jam_1 *= 0
        self.jammer.jam_2 *= 0
        self.jammer.jam_1[top] = amp
        self.jammer.jam_2[top] = amp

    def _update_interference_stats(self):
        presence = ((self.jammer.jam_1 != 0) | (self.jammer.jam_2 != 0)).astype(float)
        L = int(getattr(self.args, "obs_lag_window_L", 10))
        if len(self.interf_hist) >= L:
            self.interf_hist.pop(0)
        self.interf_hist.append(presence.copy())
        beta = float(getattr(self.args, "ewma_beta", 0.7))
        self.interf_ewma = beta * self.interf_ewma + (1 - beta) * presence

    def get_interference_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.interf_hist:
            moving = np.zeros(self.channel_num)
        else:
            hist = np.stack(self.interf_hist, axis=0)
            moving = hist.mean(axis=0)
        return moving, self.interf_ewma.copy()

    def get_graph_snapshot(self):
        return copy.deepcopy(self.graph_snapshot)

    def get_positions(self):
        if self.G is None:
            return None
        return [tuple(map(float, self.G.nodes[i]["pos"])) for i in range(self.nodes)]


# ---------------------------------------------------------------------------
# RL 环境封装
# ---------------------------------------------------------------------------


class ComEnv:
    """对外暴露 RL 接口，内部委托 WirelessSimulator 执行物理更新。"""

    def __init__(self, args):
        self.arg = args
        self.sim = WirelessSimulator(args)
        self.obs = None
        self.state = None
        self.time_step = 0

    # --- 标准接口 ---
    def reset(self, test: bool = False):
        self.sim.reset(test=test)
        self.obs = self.sim.build_observation(time_step=0)
        self.state = self.sim.build_state()
        self.time_step = 0

    def step(self, actions, time_step: int):
        actions_trans = np.zeros([self.arg.n_agents, 3], dtype=int)
        for agent_id in range(self.arg.n_agents):
            channel_action = int(actions[agent_id]) // (len(self.arg.RA_action) * len(self.arg.Pt_action))
            mcs_action = (int(actions[agent_id]) % (len(self.arg.RA_action) * len(self.arg.Pt_action))) // len(self.arg.Pt_action)
            pt_action = (int(actions[agent_id]) % (len(self.arg.RA_action) * len(self.arg.Pt_action))) % len(self.arg.Pt_action)
            actions_trans[agent_id] = [channel_action, mcs_action, pt_action]
        physics = self.sim.step_physics(actions_trans, time_step)
        reward, avg_bler, avg_rate, switch_ratio, avg_power_dbm, collision_ratio = self._aggregate_reward(
            physics, actions_trans
        )
        self.obs = self.sim.build_observation(time_step + 1)
        self.state = self.sim.build_state()
        self.time_step = time_step + 1
        sinr_avg = float(np.mean(physics["sinr_db"])) if physics["sinr_db"].size > 0 else 0.0
        terminated = False
        return reward, avg_bler, avg_rate, switch_ratio, avg_power_dbm, collision_ratio, sinr_avg, terminated

    def get_obs(self, time_step: int):
        if self.obs is None:
            self.obs = self.sim.build_observation(time_step)
        return self.obs

    def get_state(self):
        if self.state is None:
            self.state = self.sim.build_state()
        return self.state

    def get_avail_actions(self):
        return self.sim.get_avail_actions()

    def get_env_info(self):
        return {
            "n_agents": self.arg.n_agents,
            "state_shape": self.get_state().shape[0],
            "obs_shape": self.get_obs(self.time_step).shape[1],
            "n_actions": self.arg.n_actions,
            "episode_limit": self.arg.episode_limit,
        }

    def get_graph_snapshot(self):
        return self.sim.get_graph_snapshot()

    def get_interference_stats(self):
        return self.sim.get_interference_stats()

    # --- reward 汇总 ---
    def _aggregate_reward(self, physics: Dict[str, np.ndarray], actions_trans: np.ndarray):
        packet_num = np.zeros(self.arg.n_agents, dtype=float)
        transmit_bler = np.zeros(self.arg.n_agents, dtype=float)
        transmit_power = np.zeros(self.arg.n_agents, dtype=float)
        reward = 0.0
        for agent_id in range(self.arg.n_agents):
            mcs = int(actions_trans[agent_id][1])
            pt = int(actions_trans[agent_id][2])
            channel = int(actions_trans[agent_id][0])
            packet_num[agent_id] = self.sim.packet_num[self.arg.RA_action[mcs]]
            transmit_bler[agent_id] = packet_num[agent_id] * physics["bler"][agent_id]
            transmit_power[agent_id] = 10 ** (self.arg.Pt_action[pt] / 10)
            reward += (
                (1 - physics["bler"][agent_id])
                * (1 - 0.1 * physics["change_flag"][agent_id])
                * math.log2(packet_num[agent_id])
                - self.arg.Pt_action[pt] * 0.03
            )
        avg_nodes = max(1, self.arg.n_agents)
        avg_rate = float(np.sum(packet_num) / avg_nodes)
        avg_bler = float(np.sum(transmit_bler) / max(np.sum(packet_num), 1e-6))
        avg_power_dbm = float(
            np.mean([self.arg.Pt_action[int(pt_idx)] for pt_idx in actions_trans[:, 2]])
        )
        switch_ratio = float(np.mean(physics["change_flag"]))
        collision_ratio = float(np.mean(physics["collision"]))
        return reward / avg_nodes, avg_bler, avg_rate, switch_ratio, avg_power_dbm, collision_ratio


# 与旧接口保持兼容
com_env = ComEnv
