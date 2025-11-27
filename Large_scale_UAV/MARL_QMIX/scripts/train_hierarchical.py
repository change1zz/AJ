"""
统一的分层抗干扰训练脚本：
- 上层：自适应分簇 + 簇头 GAT，集中式 graph AC。
- 下层：保留 QMIX 主流程。
- 自动读取 datasets/topologies/{nodes}_nodes 下最新连通样本，可用环境变量 MQ_TOPOLOGY_DIR 覆盖。
"""
import glob
import json
import os
import sys
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from runner import Runner  # noqa: E402
from env import com_env  # noqa: E402
from env_clean import ComEnvClean  # noqa: E402
from common.arguments import get_common_args, get_mixer_args, get_env_args  # noqa: E402


def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _repo_root() -> str:
    return os.path.dirname(_project_root())


def _resolve_dataset_dir(nodes: int) -> str:
    override = os.environ.get('MQ_TOPOLOGY_DIR')
    if override:
        path = os.path.abspath(override)
        if not os.path.isdir(path):
            raise FileNotFoundError(f'MQ_TOPOLOGY_DIR 不存在：{path}')
        return path

    base_dir = os.path.join(_repo_root(), '..', 'datasets', 'topologies', f'{nodes}_nodes')
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f'未找到拓扑数据集目录：{base_dir}')
    subdirs = [os.path.join(base_dir, name) for name in os.listdir(base_dir)
               if os.path.isdir(os.path.join(base_dir, name))]
    if not subdirs:
        raise FileNotFoundError(f'拓扑目录 {base_dir} 无 sample_xxx 数据')
    subdirs.sort()
    return subdirs[-1]


def _inspect_topology_meta(dataset_dir: str) -> Dict:
    pattern = os.path.join(dataset_dir, 'sample_*', 'run_args.json')
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            topo = data.get('topology', {})
            if topo:
                return topo
        except Exception:
            continue
    return {}


def _ensure_connected(meta: Dict) -> None:
    if meta and meta.get('is_connected') is False:
        raise RuntimeError('检测到非连通拓扑，请切换到连通数据集后再训练。')


def _apply_cluster_defaults(args, topo_meta: Optional[Dict]) -> None:
    args.hrl_enable = True
    args.hrl_cluster_enable = True
    args.hrl_masking_mode = 'fixed_channel_groups'
    args.hrl_group_assign_mode = 'custom'
    args.hrl_upper_mode = 'graph'
    args.hrl_meta_period = getattr(args, 'frame_slots', 10)
    # 簇头决策依赖图快照，强制开启 graph obs
    args.use_graph_obs = True
    if topo_meta:
        neighbor_threshold = topo_meta.get('neighbor_threshold_km')
        if neighbor_threshold:
            args.graph_neighbor_threshold = float(neighbor_threshold)
            args.hrl_cluster_radius_km = float(neighbor_threshold)


def _apply_memory_profile(args) -> str:
    """根据环境变量应用低内存配置，返回当前Profile名称。"""
    profile = os.environ.get('MQ_MEM_PROFILE', '').strip().lower()
    if not profile or profile == 'off':
        args.collect_interf_stats = True
        return 'off'

    def _cap(attr: str, limit: int):
        if hasattr(args, attr):
            current = getattr(args, attr)
            if current > limit:
                setattr(args, attr, limit)
                return current, limit
        return None

    summary = []
    if profile in ('lite', 'low', '1', 'true'):
        summary.append(_cap('buffer_size', 2500))
        summary.append(_cap('batch_size', 96))
        args.collect_interf_stats = False
        args.cluster_debug = False
    elif profile in ('micro', 'tiny', 'ultra'):
        summary.append(_cap('buffer_size', 1500))
        summary.append(_cap('batch_size', 64))
        summary.append(_cap('n_steps', 1_200_000))
        args.collect_interf_stats = False
        args.cluster_debug = False
    else:
        args.collect_interf_stats = True

    for item in filter(None, summary):
        before, after = item
        print(f"[train_hierarchical] 内存配置 {profile}: {before} -> {after}")
    print(f"[train_hierarchical] 内存配置模式：{profile}，interf_stats={'开' if args.collect_interf_stats else '关'}")
    return profile


def _select_env_class(args):
    env_override = os.environ.get('MQ_ENV_CLEAN')
    if env_override is not None:
        return ComEnvClean if env_override.strip() not in ('0', '', 'false', 'False') else com_env
    cfg = getattr(args, '_config', {})
    flag = cfg.get('general', {}).get('use_env_clean', True)
    return ComEnvClean if flag else com_env


def _print_config(args, dataset_dir: str, topo_meta: Dict) -> None:
    print('=' * 72)
    print('MARL_QMIX Hierarchical Training')
    print('=' * 72)
    print(f'Nodes             : {args.n_agents}')
    print(f'Steps             : {args.n_steps:,}')
    print(f'Episode Limit     : {args.episode_limit}')
    print(f'Topology Dataset  : {dataset_dir}')
    if topo_meta:
        print(f"Connected         : {topo_meta.get('is_connected', 'unknown')}")
        print(f"Avg Degree        : {topo_meta.get('avg_degree', 'n/a')}")
        print(f"Neighbor Radius   : {topo_meta.get('neighbor_threshold_km', 'n/a')} km")
    cluster_radius = getattr(args, 'hrl_cluster_radius_km', 'n/a')
    print(f'Cluster Radius    : {cluster_radius}')
    print(f'CUDA              : {args.cuda}')
    print('=' * 72)


def main():
    args = get_common_args()
    args.alg = 'qmix'
    args.n_agents = int(os.environ.get('MQ_NODES', args.n_agents))
    args.n_steps = int(os.environ.get('MQ_STEPS', args.n_steps))
    args.evaluate_cycle = int(os.environ.get('MQ_EVAL_CYCLE', args.evaluate_cycle))
    args.evaluate_epoch = int(os.environ.get('MQ_EVAL_EPOCH', args.evaluate_epoch))
    args = get_mixer_args(args)
    args = get_env_args(args)
    args.nodes = args.n_agents
    args.episode_limit = int(os.environ.get('MQ_EPISODE_LIMIT', args.episode_limit))
    args.frame_slots = int(os.environ.get('MQ_FRAME_SLOTS', args.frame_slots))
    args.state_shape = int(args.channel_num + 5 + 2 * (args.nodes + args.jam_num))
    args.n_actions = len(args.FH_action) * len(args.RA_action) * len(args.Pt_action)

    if getattr(args, 'topology_dataset_dir', None):
        dataset_dir = args.topology_dataset_dir
    else:
        dataset_dir = _resolve_dataset_dir(args.n_agents)
    topo_meta = _inspect_topology_meta(dataset_dir)
    _ensure_connected(topo_meta)
    args.topology_dataset_dir = dataset_dir
    args.use_sparse_comm_topology = True
    args.comm_topology_require_connected = True
    args.comm_topology_enforce_min_degree = True

    _apply_cluster_defaults(args, topo_meta)
    # Exploration & replay overrides for长训练稳定性
    args.epsilon = float(os.environ.get('MQ_EPSILON', args.epsilon))
    args.min_epsilon = float(os.environ.get('MQ_MIN_EPS', args.min_epsilon))
    anneal_steps_env = os.environ.get('MQ_EPS_ANNEAL_STEPS')
    anneal_steps_default = getattr(args, 'epsilon_anneal_steps', None)
    if anneal_steps_env is not None:
        anneal_steps = int(anneal_steps_env)
    elif anneal_steps_default is not None:
        anneal_steps = int(anneal_steps_default)
    else:
        anneal_steps = 400000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / max(1, anneal_steps)
    args.epsilon_anneal_scale = 'step'
    args.batch_size = int(os.environ.get('MQ_BATCH', 128))
    args.buffer_size = int(os.environ.get('MQ_BUFFER', 5000))
    # 干扰与分簇可通过环境变量快速实验
    if os.environ.get('MQ_ALT_ENABLE'):
        args.use_alt_jam_model = bool(int(os.environ['MQ_ALT_ENABLE']))
    args.alt_jam_occ_prob = float(os.environ.get('MQ_ALT_OCC', args.alt_jam_occ_prob))
    args.hrl_cluster_refresh = int(os.environ.get('MQ_CLUSTER_REFRESH', getattr(args, 'hrl_cluster_refresh', 5)))
    args.cluster_debug = bool(int(os.environ.get('MQ_CLUSTER_DEBUG', '0')))
    args.cluster_debug_interval = int(os.environ.get('MQ_CLUSTER_DEBUG_INT', 50))
    _apply_memory_profile(args)

    try:
        import torch
        args.cuda = bool(getattr(args, 'cuda', True) and torch.cuda.is_available())
        if args.cuda:
            print(f"[train_hierarchical] 使用 CUDA 设备：{torch.cuda.get_device_name(0)}")
        else:
            print('[train_hierarchical] CUDA 不可用，回退到 CPU。')
    except Exception as exc:
        print(f'[train_hierarchical] CUDA 检查失败，自动回退 CPU（原因：{exc}）')
        args.cuda = False

    _print_config(args, dataset_dir, topo_meta)
    EnvCls = _select_env_class(args)
    env = EnvCls(args)
    runner = Runner(env, args)
    runner.run()
    reward, BLER, trans_rate, sinr_db, switch_ratio, trans_power, collision = runner.evaluate()
    print('Final Reward:', reward)
    print('Final BLER:', BLER)
    print('Final Trans_rate:', trans_rate)
    print('Final SINR(dB):', sinr_db)
    print('Final Switch_ratio:', switch_ratio)
    print('Final Trans_power (dBm):', trans_power)
    print('Final Collision:', collision)


if __name__ == '__main__':
    main()
