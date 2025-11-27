import math

import numpy as np
import os
from common.rollout import RolloutWorker
from agent.agent import Agents, CommAgents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
from tensorboard import program
import csv

# Upper-layer ACs (Graph / Local)
try:
    from models.hierarchical.upper_graph_ac import UpperGraphAC, UpperLocalAC
except Exception:
    UpperGraphAC = None
    UpperLocalAC = None


class Runner:
    def __init__(self, env, args):
        self.env = env

        # if args.alg.find('commnet') > -1 or args.alg.find('g2anet') > -1:  # communication agent
        #     self.agents = CommAgents(args)
        #     self.rolloutWorker = CommRolloutWorker(env, self.agents, args)
        # else:  # no communication agent
        self.agents = Agents(args, env)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        # Initialize upper-level AC if enabled and fixed groups masking is used
        self.upper_ac = None
        if getattr(args, 'hrl_enable', False) and getattr(args, 'hrl_masking_mode', 'none') == 'fixed_channel_groups':
            mode = getattr(args, 'hrl_upper_mode', 'graph')
            try:
                if mode == 'local' and (UpperLocalAC is not None):
                    self.upper_ac = UpperLocalAC(args, self.agents.hrl_controller, args.obs_shape, args.n_agents)
                    print('[UpperLocalAC] Initialized (分布式本地决策模式)。')
                elif UpperGraphAC is not None:
                    self.upper_ac = UpperGraphAC(args, self.agents.hrl_controller, args.obs_shape, args.n_agents)
                    print('[UpperGraphAC] Initialized with G2ANet backbone (集中式图模式)。')
                else:
                    print('[Upper AC] 未找到可用实现（UpperGraphAC/UpperLocalAC 导入失败）。')
                # attach to rollout
                if self.upper_ac is not None:
                    self.rolloutWorker.upper_ac = self.upper_ac
            except Exception as e:
                print(f'[Upper AC] 初始化失败: {e}')
        if not args.evaluate and args.alg.find('coma') == -1 and args.alg.find('central_v') == -1 and args.alg.find(
                'reinforce') == -1 and args.alg != 'R-FH':  # these 3 algorithms are on-poliy
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.episode_rewards = []
        self.episode_BLER = []
        self.episode_trans_rate = []
        self.episode_collision = []
        self._cluster_metrics_history = []
        self.last_cluster_eval_metrics = {}

        # 用来保存plt和pkl
        self.save_path = self.args.result_dir + '/' + args.alg + '/'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _cluster_metrics_from_episode(self, episode, steps):
        metrics = {}
        if not getattr(self.args, 'hrl_cluster_enable', False):
            return metrics
        try:
            steps = int(max(1, steps))
            if 'cluster_count' in episode:
                counts = np.asarray(episode['cluster_count'], dtype=np.float32)
                counts = counts.reshape(counts.shape[0], counts.shape[1], -1)
                counts = counts[0, :steps, 0]
                if counts.size > 0:
                    valid = counts > 0
                    valid_counts = counts[valid]
                    if valid_counts.size > 0:
                        metrics['avg_clusters'] = float(valid_counts.mean())
                        metrics['max_clusters'] = float(valid_counts.max())
                    else:
                        metrics['avg_clusters'] = 0.0
                        metrics['max_clusters'] = 0.0
            if 'cluster_assign' in episode and steps > 1:
                assigns = np.asarray(episode['cluster_assign'], dtype=np.int32)
                assigns = assigns.reshape(assigns.shape[0], assigns.shape[1], -1)
                assigns = assigns[0, :steps, :]
                valid_mask = assigns >= 0
                prev = assigns[:-1]
                nxt = assigns[1:]
                valid_edge = valid_mask[:-1] & valid_mask[1:]
                diff = (prev != nxt) & valid_edge
                denom = max(1, int(valid_edge.sum()))
                metrics['cluster_switch_ratio'] = float(diff.sum() / denom)
                per_agent_valid = np.maximum(1, valid_edge.sum(axis=0))
                metrics['cluster_switch_agent_mean'] = float(
                    np.mean(diff.sum(axis=0) / per_agent_valid))
            if 'avg_clusters' in metrics and metrics['avg_clusters'] > 0:
                metrics['avg_members_per_cluster'] = float(self.args.n_agents / metrics['avg_clusters'])
            if hasattr(self.upper_ac, 'last_cluster_meta') and isinstance(self.upper_ac.last_cluster_meta, dict):
                meta = self.upper_ac.last_cluster_meta
                if 'cluster_count' in meta:
                    metrics['cluster_count_meta'] = float(meta['cluster_count'])
        except Exception:
            pass
        # Fallback: use controller channel groups when explicit指标缺失
        if 'avg_clusters' not in metrics:
            controller = getattr(self.agents, 'hrl_controller', None)
            if controller is not None:
                groups = getattr(controller, 'channel_groups', None)
                if groups:
                    metrics['avg_clusters'] = float(len(groups))
        if 'cluster_switch_ratio' not in metrics and metrics.get('avg_clusters', 0) > 0:
            metrics['cluster_switch_ratio'] = 0.0
        return metrics

    @staticmethod
    def _aggregate_cluster_metrics(metrics_list):
        if not metrics_list:
            return {}
        agg = {}
        for entry in metrics_list:
            for key, value in entry.items():
                if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                    continue
                agg.setdefault(key, []).append(float(value))
        return {k: float(np.mean(vs)) for k, vs in agg.items() if len(vs) > 0}

    def _log_cluster_metrics(self, writer, global_step, metrics, prefix):
        if not metrics or writer is None:
            return
        for key, value in metrics.items():
            if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                continue
            try:
                writer.add_scalar(f'Cluster/{prefix}_{key}', float(value), global_step=global_step)
            except Exception:
                pass

    def run(self):
        t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        path = 'result/tensorboard{}'.format(t)
        writer = SummaryWriter('result/tensorboard{}'.format(t))
        # Prepare CSV directory and file for evaluation metrics
        csv_dir = 'result/eval_csv{}'.format(t)
        os.makedirs(csv_dir, exist_ok=True)
        csv_file = os.path.join(csv_dir, 'metrics.csv')
        # 保存当前运行的参数与拓扑图的目录
        meta_dir = csv_dir
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', path])
        url = tb.launch()
        print(f"[TensorBoard] URL: {url}")

        # ---- 启动时输出所有参数并保存为JSON ----
        try:
            # 将 args（argparse Namespace）转为字典
            args_dict = {k: getattr(self.args, k) for k in dir(self.args) if not k.startswith('_') and not callable(getattr(self.args, k))}
            print('[RunConfig] 当前训练参数（部分关键项）:')
            # 精简打印，避免刷屏：挑选核心字段
            core_keys = ['alg', 'n_steps', 'episode_limit', 'n_agents', 'n_actions', 'evaluate_cycle', 'evaluate_epoch', 'train_steps', 'nodes', 'channel_num', 'cuda', 'use_sparse_comm_topology']
            for ck in core_keys:
                if ck in args_dict:
                    print(f'  - {ck}: {args_dict[ck]}')
            # 保存完整参数为JSON（处理 numpy/集合 等不可序列化类型）
            def _to_jsonable(x):
                try:
                    import numpy as np
                except Exception:
                    np = None
                # 基础类型
                if isinstance(x, (int, float, str, bool)) or x is None:
                    return x
                # numpy 特殊处理
                if np is not None:
                    if isinstance(x, np.ndarray):
                        return x.tolist()
                    if isinstance(x, (np.integer,)):
                        return int(x)
                    if isinstance(x, (np.floating,)):
                        return float(x)
                    if isinstance(x, (np.bool_,)):
                        return bool(x)
                # 容器类型
                if isinstance(x, (list, tuple, set)):
                    return [_to_jsonable(v) for v in list(x)]
                if isinstance(x, dict):
                    return {str(k): _to_jsonable(v) for k, v in x.items()}
                # 兜底：字符串化
                try:
                    return str(x)
                except Exception:
                    return '<unserializable>'

            import json
            with open(os.path.join(meta_dir, 'run_args.json'), 'w', encoding='utf-8') as f:
                json.dump({k: _to_jsonable(v) for k, v in args_dict.items()}, f, ensure_ascii=False, indent=2)
            print(f'[RunConfig] 训练参数已保存：{os.path.join(meta_dir, "run_args.json")}')
        except Exception as e:
            print(f'[RunConfig] 参数导出失败：{e}')

        # ---- 启动时导出通信拓扑图（若启用稀疏拓扑） ----
        try:
            # 先重置一次环境以确保位置与拓扑已构建
            self.env.reset(test=False)
            if getattr(self.args, 'use_sparse_comm_topology', False):
                # 从环境读取邻接与坐标（优先3D）
                comm_adj = getattr(self.env, 'comm_adj', None)
                use_3d = all(('pos3d' in self.env.G.nodes[i]) for i in range(self.env.nodes))
                if use_3d:
                    positions3d = [self.env.G.nodes[i]['pos3d'] for i in range(self.env.nodes)]
                else:
                    positions = [self.env.G.nodes[i]['pos'] for i in range(self.env.nodes)]
                rx = getattr(self.env, 'receive_node_coordinate', None)
                import matplotlib.pyplot as plt
                plt.close()
                if use_3d:
                    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                    import numpy as np
                    fig = plt.figure(figsize=(7, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    xs = [float(p[0]) for p in positions3d]
                    ys = [float(p[1]) for p in positions3d]
                    zs = [float(p[2]) for p in positions3d]
                    # 守护：若存在越界的Z值，绘图阶段进行硬裁剪并提示
                    try:
                        Lx, Ly, Lz = tuple(getattr(self.args, 'space_box_km', (5.0, 5.0, 2.0)))
                    except Exception:
                        Lx, Ly, Lz = 5.0, 5.0, 2.0
                    if len(zs) > 0:
                        z_min, z_max = min(zs), max(zs)
                        if z_min < 0.0 or z_max > Lz:
                            print(f"[Topology] 发现Z越界，已裁剪到[0,{Lz}]，原z_min={z_min:.3f}, z_max={z_max:.3f}")
                            import numpy as np
                            zs = list(np.clip(np.array(zs, dtype=float), 0.0, Lz))
                            positions3d = [(xs[i], ys[i], zs[i]) for i in range(len(zs))]
                    ax.scatter(xs, ys, zs, c='tab:blue', s=55, marker='o', label='UAV')
                    # 接收机（若仅2D坐标，则z=0）
                    if rx is not None and len(rx) in (2, 3):
                        rx_x, rx_y = float(rx[0]), float(rx[1])
                        rx_z = float(rx[2]) if len(rx) == 3 else 0.0
                        ax.scatter([rx_x], [rx_y], [rx_z], c='tab:green', s=80, marker='X', label='Receiver')
                    # 画边（普通边）
                    if comm_adj is not None:
                        import numpy as np
                        adj = np.array(comm_adj)
                        for i in range(adj.shape[0]):
                            for j in range(i + 1, adj.shape[1]):
                                if adj[i, j] == 1:
                                    x1, y1, z1 = positions3d[i]
                                    x2, y2, z2 = positions3d[j]
                                    z1 = max(0.0, min(Lz, float(z1)))
                                    z2 = max(0.0, min(Lz, float(z2)))
                                    ax.plot([x1, x2], [y1, y2], [z1, z2], color='gray', linewidth=0.8, alpha=0.7)
                    # 若存在桥接边，使用高亮显示
                    bridges = getattr(self.env, 'comm_bridge_edges', [])
                    for (i, j) in bridges:
                        x1, y1, z1 = positions3d[i]
                        x2, y2, z2 = positions3d[j]
                        z1 = max(0.0, min(Lz, float(z1)))
                        z2 = max(0.0, min(Lz, float(z2)))
                        ax.plot([x1, x2], [y1, y2], [z1, z2], color='tab:red', linewidth=1.2, alpha=0.85, linestyle='--')
                    # 样式与范围：使用空间盒与通信半径信息
                    # 已在前面解析 Lx, Ly, Lz；此处仅取通信范围
                    R = float(getattr(self.args, 'comm_range_km', 1.0))
                    ax.set_title(f'3D Distance-Threshold Topology (R={R:.2f} km, Alt 0-{Lz:.2f} km)')
                    ax.set_xlabel('X (km)')
                    ax.set_ylabel('Y (km)')
                    ax.set_zlabel('Z (km)')
                    # 先设定范围，再设定刻度，确保显示与数据一致
                    ax.set_xlim([-Lx/2.0, Lx/2.0])
                    ax.set_ylim([-Ly/2.0, Ly/2.0])
                    ax.set_zlim([0.0, Lz])
                    ax.set_zticks(np.linspace(0.0, Lz, 5))
                    ax.legend(loc='upper left')
                    # 保存图片
                    topo_path = os.path.join(meta_dir, 'comm_topology.png')
                    fig.savefig(topo_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f'[Topology] 稀疏通信拓扑图已导出：{topo_path}')
                    # 额外导出3D坐标以便排查：comm_pos3d.csv
                    try:
                        pos_path = os.path.join(meta_dir, 'comm_pos3d.csv')
                        with open(pos_path, 'w', newline='') as pf:
                            w = csv.writer(pf)
                            w.writerow(['x_km', 'y_km', 'z_km'])
                            for (x, y, z) in positions3d:
                                w.writerow([float(x), float(y), float(z)])
                        print(f'[Topology] 节点3D坐标已导出：{pos_path}')
                    except Exception:
                        pass
                else:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    xs = [p[0] for p in positions]
                    ys = [p[1] for p in positions]
                    ax.scatter(xs, ys, c='tab:blue', s=55, marker='o', label='UAV')
                    if rx is not None and len(rx) == 2:
                        ax.scatter([rx[0]], [rx[1]], c='tab:green', s=80, marker='X', label='Receiver')
                    if comm_adj is not None:
                        import numpy as np
                        adj = np.array(comm_adj)
                        for i in range(adj.shape[0]):
                            for j in range(i + 1, adj.shape[1]):
                                if adj[i, j] == 1:
                                    x1, y1 = positions[i]
                                    x2, y2 = positions[j]
                                    ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.8, alpha=0.7)
                    # 桥接边高亮（2D）
                    bridges = getattr(self.env, 'comm_bridge_edges', [])
                    for (i, j) in bridges:
                        x1, y1 = positions[i]
                        x2, y2 = positions[j]
                        ax.plot([x1, x2], [y1, y2], color='tab:red', linewidth=1.2, alpha=0.85, linestyle='--')
                    # 样式与范围：使用空间盒与通信半径信息
                    try:
                        Lx, Ly, _ = tuple(getattr(self.args, 'space_box_km', (5.0, 5.0, 2.0)))
                    except Exception:
                        Lx, Ly = 5.0, 5.0
                    R = float(getattr(self.args, 'comm_range_km', 1.0))
                    ax.set_title(f'Distance-Threshold Topology (R={R:.2f} km)')
                    ax.set_xlabel('X (km)')
                    ax.set_ylabel('Y (km)')
                    ax.set_xlim([-Lx/2.0, Lx/2.0])
                    ax.set_ylim([-Ly/2.0, Ly/2.0])
                    ax.legend(loc='upper right')
                    ax.grid(True, linestyle='--', alpha=0.3)
                    topo_path = os.path.join(meta_dir, 'comm_topology.png')
                    fig.savefig(topo_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f'[Topology] 稀疏通信拓扑图已导出：{topo_path}')
                # 同步保存邻接矩阵（CSV），便于分析
                try:
                    import csv
                    with open(os.path.join(meta_dir, 'comm_adj.csv'), 'w', newline='') as cf:
                        w = csv.writer(cf)
                        for row in comm_adj:
                            w.writerow(list(row))
                    print(f'[Topology] 邻接矩阵已导出：{os.path.join(meta_dir, "comm_adj.csv")}')
                except Exception:
                    pass
                # 计算并写入拓扑元数据（连通性、桥接边数、平均度等）到 run_args.json
                try:
                    import json
                    import numpy as np
                    adj = np.array(comm_adj)
                    degrees = adj.sum(axis=1)
                    avg_degree = float(degrees.mean())
                    k_guess = max(2, min(5, self.env.nodes - 1))
                    # 计算连通性（基于邻接矩阵的简单DFS/BFS）
                    n = adj.shape[0]
                    visited = np.zeros(n, dtype=bool)
                    comp_count = 0
                    for i in range(n):
                        if not visited[i]:
                            comp_count += 1
                            stack = [i]
                            visited[i] = True
                            while stack:
                                u = stack.pop()
                                nbrs = np.where(adj[u] == 1)[0]
                                for v in nbrs:
                                    if not visited[v]:
                                        visited[v] = True
                                        stack.append(v)
                    is_connected = (comp_count == 1)
                    bridge_edges = getattr(self.env, 'comm_bridge_edges', [])
                    bridge_edge_count = int(len(bridge_edges))
                    neighbor_threshold_km = float(getattr(self.env, 'neighbor_threshold', float(getattr(self.args, 'comm_range_km', 1.0))))
                    topo_meta = {
                        'nodes': int(self.env.nodes),
                        'avg_degree': avg_degree,
                        'degree_list': degrees.astype(int).tolist(),
                        'k_neighbors_guess': int(k_guess),
                        'min_degree_enforced': 2,
                        'is_connected': bool(is_connected),
                        'bridge_edge_count': bridge_edge_count,
                        'neighbor_threshold_km': neighbor_threshold_km,
                        'require_connected_flag': bool(getattr(self.args, 'comm_topology_require_connected', False)),
                        'enforce_min_degree_flag': bool(getattr(self.args, 'comm_topology_enforce_min_degree', False))
                    }
                    args_json_path = os.path.join(meta_dir, 'run_args.json')
                    # 若文件存在则合并拓扑元数据
                    if os.path.exists(args_json_path):
                        try:
                            with open(args_json_path, 'r', encoding='utf-8') as rf:
                                data = json.load(rf)
                        except Exception:
                            data = {}
                    else:
                        data = {}
                    data['topology'] = topo_meta
                    with open(args_json_path, 'w', encoding='utf-8') as wf:
                        json.dump(data, wf, ensure_ascii=False, indent=2)
                    print(f'[Topology] 拓扑元数据已写入：{args_json_path}')
                except Exception as e:
                    print(f'[Topology] 写入拓扑元数据失败：{e}')
            else:
                print('[Topology] 稀疏通信拓扑未启用（use_sparse_comm_topology=False），跳过拓扑图导出。')
        except Exception as e:
            print(f'[Topology] 导出失败：{e}')

        time_steps, train_steps, evaluate_steps = 0, 0, 0
        while time_steps < self.args.n_steps:
            # print('time_steps {}'.format(time_steps))
            if time_steps // self.args.evaluate_cycle > evaluate_steps:
                episode_reward, episode_BLER, episode_trans_rate, episode_sinr, episode_switch_ratio, episode_transmit_power, episode_collision = self.evaluate()
                self.episode_rewards.append(episode_reward)
                self.episode_BLER.append(episode_BLER)
                self.episode_trans_rate.append(episode_trans_rate)
                self.episode_collision.append(episode_collision)
                t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
                print('\n学习率: {}'.format(self.args.lr))
                print('评估轮次: {}，步数: {}，时间: {}'.format(evaluate_steps, time_steps, t))
                print('平均回报: ', episode_reward)
                print('平均BLER: ', episode_BLER)
                print('平均传输速率: ', episode_trans_rate)
                print('平均SINR(dB): ', episode_sinr)
                print('平均切换比例: ', episode_switch_ratio)
                print('平均发射功率(dBm): ', episode_transmit_power)
                print('平均碰撞比例: ', episode_collision)
                if getattr(self.args, 'hrl_cluster_enable', False):
                    train_summary = self._aggregate_cluster_metrics(self._cluster_metrics_history)
                    if train_summary:
                        print('[Cluster-Train] avg_clusters={:.2f} | switch_ratio={:.3f}'.format(
                            train_summary.get('avg_clusters', 0.0),
                            train_summary.get('cluster_switch_ratio', 0.0)))
                        self._log_cluster_metrics(writer, time_steps, train_summary, 'train_avg')
                    self._cluster_metrics_history.clear()
                writer.add_scalar('Reward', episode_reward, global_step=time_steps)
                writer.add_scalar('BLER', episode_BLER, global_step=time_steps)
                writer.add_scalar('Trans_rate', episode_trans_rate, global_step=time_steps)
                writer.add_scalar('SINR_dB', episode_sinr, global_step=time_steps)
                writer.add_scalar('Switch_ratio', episode_switch_ratio, global_step=time_steps)
                writer.add_scalar('Transmit_power', episode_transmit_power, global_step=time_steps)
                writer.add_scalar('collision_ratio', episode_collision, global_step=time_steps)
                # Upper-layer AC stats to TensorBoard, if available
                if (self.upper_ac is not None) and hasattr(self.upper_ac, 'last_stats') and isinstance(self.upper_ac.last_stats, dict):
                    for k, v in self.upper_ac.last_stats.items():
                        try:
                            writer.add_scalar(f'Upper/{k}', float(v), global_step=time_steps)
                        except Exception:
                            pass
                eval_cluster_metrics = getattr(self, 'last_cluster_eval_metrics', {})
                if eval_cluster_metrics:
                    print('[Cluster-Eval] avg_clusters={:.2f} | switch_ratio={:.3f}'.format(
                        eval_cluster_metrics.get('avg_clusters', 0.0),
                        eval_cluster_metrics.get('cluster_switch_ratio', 0.0)))
                    self._log_cluster_metrics(writer, time_steps, eval_cluster_metrics, 'eval')
                # Upper-layer decision distribution (per-agent & overall)
                if (self.upper_ac is not None) and hasattr(self.upper_ac, 'get_decision_distribution'):
                    try:
                        dist = self.upper_ac.get_decision_distribution(reset=True)
                        per_agent = dist.get('per_agent', None)
                        overall = dist.get('overall', None)
                        if per_agent is not None:
                            for i in range(per_agent.shape[0]):
                                for g in range(per_agent.shape[1]):
                                    writer.add_scalar(f'Upper/agent_{i}/group_{g}_prob', float(per_agent[i, g]), global_step=time_steps)
                        if overall is not None:
                            for g in range(overall.shape[0]):
                                writer.add_scalar(f'Upper/group_{g}_prob', float(overall[g]), global_step=time_steps)
                    except Exception:
                        pass
                # Append evaluation metrics to CSV (with header on first write)
                write_header = not os.path.exists(csv_file)
                try:
                    with open(csv_file, mode='a', newline='') as f:
                        w = csv.writer(f)
                        base_header = ['time_steps', 'Reward', 'BLER', 'Trans_rate', 'Switch_ratio', 'Transmit_power', 'collision_ratio']
                        row = [time_steps, episode_reward, episode_BLER, episode_trans_rate, episode_switch_ratio, episode_transmit_power, episode_collision]
                        # Optional: Upper-layer AC stats
                        upper_header = []
                        upper_row = []
                        if (self.upper_ac is not None) and hasattr(self.upper_ac, 'last_stats') and isinstance(self.upper_ac.last_stats, dict):
                            ls = self.upper_ac.last_stats
                            upper_header = ['Upper_loss_pi', 'Upper_loss_v', 'Upper_A_mean', 'Upper_A_std', 'Upper_entropy', 'Upper_entropy_coef']
                            upper_row = [
                                float(ls.get('loss_pi', float('nan'))),
                                float(ls.get('loss_v', float('nan'))),
                                float(ls.get('A_mean', float('nan'))),
                                float(ls.get('A_std', float('nan'))),
                                float(ls.get('entropy', float('nan'))),
                                float(ls.get('entropy_coef', float('nan'))),
                            ]
                        cluster_header = []
                        cluster_row = []
                        eval_cluster_metrics = getattr(self, 'last_cluster_eval_metrics', {})
                        if eval_cluster_metrics:
                            for key in sorted(eval_cluster_metrics.keys()):
                                cluster_header.append(f'Cluster_{key}')
                                cluster_row.append(eval_cluster_metrics[key])
                        if write_header:
                            # ?? SINR ????? BLER ??????????????
                            header = ['time_steps', 'Reward', 'BLER', 'SINR_dB', 'Trans_rate', 'Switch_ratio', 'Transmit_power', 'collision_ratio']
                            w.writerow(header + upper_header + cluster_header)
                        # ????????? 3 ???BLER ??????? SINR
                        w.writerow(row[:3] + [episode_sinr] + row[3:] + upper_row + cluster_row)
                except Exception as e:
                    print(f"[CSV] Failed to write metrics: {e}")

                evaluate_steps += 1
            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, episode_reward, steps, episode_BLER, episode_trans_rate, episode_sinr, episode_switch_ratio, episode_transmit_power, episode_collision \
                    = self.rolloutWorker.generate_episode(episode_idx)
                cluster_metrics = self._cluster_metrics_from_episode(episode, steps)
                if cluster_metrics:
                    self._cluster_metrics_history.append(cluster_metrics)
                    self._log_cluster_metrics(writer, time_steps + steps, cluster_metrics, 'train_step')
                episodes.append(episode)
                time_steps += steps
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            if self.args.alg.find('coma') > -1 or self.args.alg.find('central_v') > -1 or self.args.alg.find(
                    'reinforce') > -1:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    self.agents.train(mini_batch, train_steps)
                    train_steps += 1
        episode_reward, episode_BLER, episode_trans_rate, episode_sinr, episode_switch_ratio, episode_transmit_power, episode_collision = self.evaluate()

        self.episode_rewards.append(episode_reward)
        self.episode_BLER.append(episode_BLER)
        self.episode_trans_rate.append(episode_trans_rate)
        self.episode_collision.append(episode_collision)
        # self.plt(self.args.lr)

    def evaluate(self):
        episode_rewards = 0
        episode_BLERs = 0
        episode_trans_rates = 0
        episode_sinrs = 0
        episode_switch_ratios = 0
        episode_transmit_powers = 0
        episode_collisions = 0
        eval_cluster_metrics = []
        self.last_cluster_eval_metrics = {}
        for epoch in range(self.args.evaluate_epoch):
            episode, episode_reward, _, episode_BLER, episode_trans_rate, episode_sinr, episode_switch_ratio, episode_transmit_power, episode_collision = self.rolloutWorker.generate_episode(
                epoch, evaluate=True)
            episode_rewards += episode_reward
            episode_BLERs += episode_BLER
            episode_trans_rates += episode_trans_rate
            episode_sinrs += episode_sinr
            episode_switch_ratios += episode_switch_ratio
            episode_transmit_powers += episode_transmit_power
            episode_collisions += episode_collision
            metrics = self._cluster_metrics_from_episode(episode, self.args.episode_limit)
            if metrics:
                eval_cluster_metrics.append(metrics)
        # 安全分母保护，避免评估早期出现除零或 log(0)
        eval_epochs = max(1, self.args.evaluate_epoch)
        safe_trans_rates = episode_trans_rates if episode_trans_rates > 1e-9 else 1e-9
        safe_tx_power_mean = episode_transmit_powers / eval_epochs
        # 评估BLER采用加权平均：sum(packet_num * BLER) / sum(packet_num)，避免与碰撞混合导致负值
        self.last_cluster_eval_metrics = self._aggregate_cluster_metrics(eval_cluster_metrics)
        return episode_rewards / eval_epochs, \
               (episode_BLERs / safe_trans_rates), \
               (episode_trans_rates - episode_BLERs) / 3 / eval_epochs, \
               (episode_sinrs / eval_epochs), \
               episode_switch_ratios / eval_epochs, \
                safe_tx_power_mean, \
                episode_collisions / eval_epochs

    def test(self, test_epoch, jamming_type, distance_1, distance_2, jam_num, power):
        episode_rewards = 0
        episode_BLERs = 0
        episode_trans_rates = 0
        episode_switch_ratios = 0
        episode_transmit_powers = 0
        episode_collisions = 0

        for i in range(test_epoch):
            _, episode_reward, _, episode_BLER, episode_trans_rate, episode_switch_ratio, episode_transmit_power, episode_collision = self.rolloutWorker.generate_test_episode(
                distance_1, distance_2, jam_num, power, test_jamming_type=jamming_type)
            episode_rewards += episode_reward
            episode_BLERs += episode_BLER
            episode_trans_rates += episode_trans_rate
            episode_switch_ratios += episode_switch_ratio
            episode_transmit_powers += episode_transmit_power
            episode_collisions += episode_collision

        return episode_rewards / test_epoch, \
               episode_BLERs / episode_trans_rates, \
               (episode_trans_rates - episode_BLERs) / 3 / test_epoch, \
               episode_switch_ratios / test_epoch, \
               episode_transmit_powers / test_epoch, \
               episode_collisions / test_epoch

        # return episode_rewards / test_epoch, \
        #        episode_BLERs / episode_trans_rates - episode_collisions / test_epoch, \
        #        (episode_trans_rates - episode_BLERs) / 3 / test_epoch, \
        #        episode_switch_ratios / test_epoch, \
        #        10 * math.log10(episode_transmit_powers / test_epoch), \
        #        episode_collisions / test_epoch

    def plt(self, lr):
        plt.close()
        fig, axs = plt.subplots(3, figsize=(12, 15))
        axs[0].plot(range(len(self.episode_rewards)), self.episode_rewards)
        axs[0].set_title('Rewards'.format(lr))
        axs[0].set_xlabel('step*{}'.format(self.args.evaluate_cycle))
        axs[0].set_ylabel('Rewards')

        # 子图2
        axs[1].plot(range(len(self.episode_BLER)), self.episode_BLER)
        axs[1].set_title('BLER'.format(lr))
        axs[1].set_xlabel('step*{}'.format(self.args.evaluate_cycle))
        axs[1].set_ylabel('BLER')

        # 子图3
        axs[2].plot(range(len(self.episode_trans_rate)), self.episode_trans_rate)
        axs[2].set_title('Trans_rate'.format(lr))
        axs[2].set_xlabel('step*{}'.format(self.args.evaluate_cycle))
        axs[2].set_ylabel('Trans_rate')

        plt.savefig(self.save_path + '/learning_rate_{}.png'.format(lr), format='png')

        plt.show()
