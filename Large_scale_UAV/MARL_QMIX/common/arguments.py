import argparse
import os
from typing import Any, Dict, Optional

import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - tooling note
    raise RuntimeError("PyYAML is required to load configuration files. Please install with `pip install pyyaml`.") from exc


def _default_config_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, 'configs', 'default.yaml')


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = path or os.environ.get('MQ_CONFIG') or _default_config_path()
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


def _cfg(cfg, section, key, fallback=None):
    return cfg.get(section, {}).get(key, fallback)


def get_common_args():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--config', type=str, default=None, help='YAML config path (default: configs/default.yaml)')
    known, remaining = base_parser.parse_known_args()
    cfg = load_config(known.config)

    g = cfg.get('general', {})
    h = cfg.get('hrl', {})
    topo = cfg.get('topology', {})

    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument('--seed', type=int, default=g.get('seed', 123), help='random seed')
    parser.add_argument('--replay_dir', type=str, default=g.get('replay_dir', ''), help='absolute path to save the replay')
    parser.add_argument('--alg', type=str, default=g.get('alg', 'qmix'), help='algorithm')
    parser.add_argument('--n_agents', type=int, default=g.get('n_agents', 4))
    parser.add_argument('--n_actions', type=int, default=g.get('n_actions', 600))
    parser.add_argument('--state_shape', type=int, default=g.get('state_shape', 57))
    parser.add_argument('--obs_shape', type=int, default=g.get('obs_shape', 15))
    parser.add_argument('--n_steps', type=int, default=g.get('n_steps', 200000))
    parser.add_argument('--episode_limit', type=int, default=g.get('episode_limit', 100))
    parser.add_argument('--n_episodes', type=int, default=g.get('n_episodes', 1))
    parser.add_argument('--last_action', type=bool, default=g.get('last_action', True))
    parser.add_argument('--reuse_network', type=bool, default=g.get('reuse_network', False))
    parser.add_argument('--gamma', type=float, default=g.get('gamma', 0.99))
    parser.add_argument('--optimizer', type=str, default=g.get('optimizer', 'RMS'))
    parser.add_argument('--evaluate_cycle', type=int, default=g.get('evaluate_cycle', 20000))
    parser.add_argument('--evaluate_epoch', type=int, default=g.get('evaluate_epoch', 20))
    parser.add_argument('--model_dir', type=str, default=g.get('model_dir', './model'))
    parser.add_argument('--result_dir', type=str, default=g.get('result_dir', './result'))
    parser.add_argument('--load_model', type=bool, default=g.get('load_model', False))
    parser.add_argument('--evaluate', type=bool, default=g.get('evaluate', False))
    parser.add_argument('--cuda', type=bool, default=g.get('cuda', True))

    parser.add_argument('--hrl_enable', type=bool, default=h.get('enable', False))
    parser.add_argument('--hrl_meta_period', type=int, default=h.get('meta_period', 10))
    parser.add_argument('--hrl_masking_mode', type=str, default=h.get('masking_mode', 'none'))
    parser.add_argument('--hrl_topk_K', type=int, default=h.get('topk_K', 0))
    parser.add_argument('--hrl_upper_mode', type=str, default=h.get('upper_mode', 'graph'))
    parser.add_argument('--hrl_local_agg', type=str, default=h.get('local_agg', 'mean'))
    parser.add_argument('--hrl_upper_lr', type=float, default=h.get('upper_lr', 2e-4))
    parser.add_argument('--hrl_upper_entropy', type=float, default=h.get('upper_entropy', 0.01))
    parser.add_argument('--hrl_upper_entropy_min', type=float, default=h.get('upper_entropy_min', 1e-3))
    parser.add_argument('--hrl_upper_entropy_decay', type=float, default=h.get('upper_entropy_decay', 0.999))
    parser.add_argument('--hrl_upper_adv_norm', type=bool, default=h.get('upper_adv_norm', True))
    parser.add_argument('--hrl_upper_adv_clip', type=float, default=h.get('upper_adv_clip', 5.0))
    parser.add_argument('--hrl_upper_beta', type=float, default=h.get('upper_beta', 0.7))
    parser.add_argument('--hrl_cluster_enable', type=bool, default=h.get('cluster_enable', False))
    parser.add_argument('--hrl_cluster_target_size', type=int, default=h.get('cluster_target_size', 6))
    parser.add_argument('--hrl_cluster_refresh', type=int, default=h.get('cluster_refresh', 5))
    parser.add_argument('--hrl_cluster_radius_km', type=float, default=h.get('cluster_radius_km', 1.5))
    parser.add_argument('--hrl_cluster_head_strategy', type=str, default=h.get('cluster_head_strategy', 'farthest'))
    parser.add_argument('--hrl_channel_groups_json', type=str, default=h.get('channel_groups_json', ''))
    parser.add_argument('--hrl_group_assign_mode', type=str, default=h.get('group_assign_mode', 'mod'))
    parser.add_argument('--hrl_group_assignments_json', type=str, default=h.get('group_assignments_json', ''))
    parser.add_argument('--hrl_group_rotate', type=bool, default=h.get('group_rotate', False))
    parser.add_argument('--hrl_group_rotate_period', type=int, default=h.get('group_rotate_period', 50))
    parser.add_argument('--upper_log_every', type=int, default=h.get('upper_log_every', 20))

    parser.add_argument('--use_sparse_comm_topology', type=bool, default=topo.get('use_sparse_comm_topology', False))
    parser.add_argument('--comm_topology_require_connected', type=bool,
                        default=topo.get('comm_topology_require_connected', False))
    parser.add_argument('--comm_topology_enforce_min_degree', type=bool,
                        default=topo.get('comm_topology_enforce_min_degree', False))
    parser.add_argument('--topology_dataset_lock_sample', type=bool,
                        default=topo.get('topology_dataset_lock_sample', False))
    parser.add_argument('--topology_dataset_sample', type=str,
                        default=topo.get('topology_dataset_sample', ''))
    parser.add_argument('--topology_dataset_cache_size', type=int,
                        default=topo.get('topology_dataset_cache_size', 64))
    parser.add_argument('--topology_dataset_dir', type=str,
                        default=cfg.get('env', {}).get('topology_dataset_dir', ''))

    args = parser.parse_args(remaining)
    args._config = cfg
    return args


# arguments of coma
def get_coma_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.8
    args.anneal_epsilon = 0.00032
    args.min_epsilon = 0.15
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    cfg = getattr(args, '_config', {})
    m = cfg.get('mixer', {})
    args.rnn_hidden_dim = m.get('rnn_hidden_dim', 256)
    args.qmix_hidden_dim = m.get('qmix_hidden_dim', 32)
    args.two_hyper_layers = m.get('two_hyper_layers', True)
    args.hyper_hidden_dim = m.get('hyper_hidden_dim', 64)
    args.qtran_hidden_dim = m.get('qtran_hidden_dim', 64)
    args.lr = m.get('lr', 1e-4)
    args.epsilon = m.get('epsilon', 0.8)
    args.min_epsilon = m.get('min_epsilon', 0.15)
    anneal_steps = m.get('epsilon_anneal_steps', 100000)
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / max(1, anneal_steps)
    args.epsilon_anneal_scale = m.get('epsilon_anneal_scale', 'step')
    args.train_steps = m.get('train_steps', 1)
    args.batch_size = m.get('batch_size', 32)
    args.buffer_size = int(m.get('buffer_size', 1000))
    args.save_cycle = m.get('save_cycle', 30000)
    args.target_update_cycle = m.get('target_update_cycle', 400)
    args.lambda_opt = m.get('lambda_opt', 1)
    args.lambda_nopt = m.get('lambda_nopt', 1)
    args.grad_norm_clip = m.get('grad_norm_clip', 10)
    args.noise_dim = m.get('noise_dim', 16)
    args.lambda_mi = m.get('lambda_mi', 0.001)
    args.lambda_ql = m.get('lambda_ql', 1)
    args.entropy_coefficient = m.get('entropy_coefficient', 0.001)
    return args


def get_bler(args=None):
    cfg = getattr(args, '_config', {}) if args is not None else load_config()
    default_path = _default_config_path()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = cfg.get('bler', {}).get('data_path', 'Bler_data.csv')
    if not os.path.isabs(data_path):
        data_path = os.path.join(root, data_path)
    df = pd.read_csv(data_path)
    pattern = df['Bpsk-1/2']
    bpsk_1_2_ber = pattern[0:400]
    pattern = df['Qpsk-1/2']
    qpsk_1_2_ber = pattern[0:400]
    pattern = df['Qpsk-3/4']
    qpsk_3_4_ber = pattern[0:400]
    pattern = df['16Qam-1/2']
    qam16_1_2_ber = pattern[0:400]
    pattern = df['16Qam-3/4']
    qam16_3_4_ber = pattern[0:400]
    pattern = df['64Qam-2/3']
    qam64_2_3_ber = pattern[0:400]
    pattern = df['64Qam-3/4']
    qam64_3_4_ber = pattern[0:400]
    pattern = df['64Qam-5/6']
    qam64_5_6_ber = pattern[0:400]
    pattern = df['256Qam-3/4']
    qam256_3_4_ber = pattern[0:400]
    pattern = df['256Qam-5/6']
    qam256_5_6_ber = pattern[0:400]

    ber_data = {
        'bpsk_1_2': bpsk_1_2_ber,
        'qpsk_1_2': qpsk_1_2_ber,
        'qpsk_3_4': qpsk_3_4_ber,
        'qam16_1_2': qam16_1_2_ber,
        'qam16_3_4': qam16_3_4_ber,
        'qam64_2_3': qam64_2_3_ber,
        'qam64_3_4': qam64_3_4_ber,
        'qam64_5_6': qam64_5_6_ber,
        'qam256_3_4': qam256_3_4_ber,
        'qam256_5_6': qam256_5_6_ber,
    }
    return ber_data


def get_env_args(args):
    cfg = getattr(args, '_config', {})
    env = cfg.get('env', {})
    args.map = env.get('map', 'train_10_10_1.json')
    args.ber_threshold = env.get('ber_threshold', 1e-5)
    args.bler_data = get_bler(args)
    jam_dist = env.get('jam_distance', (5, 10))
    if isinstance(jam_dist, list):
        jam_dist = tuple(jam_dist)
    args.jam_distance = jam_dist
    args.all_bandwidth = env.get('all_bandwidth', 78)
    args.channel_bandwidth = env.get('channel_bandwidth', 1.95)
    args.sub_band = env.get('sub_band', 8)
    args.channels_per_subband = env.get('channels_per_subband', 5)
    if getattr(args, 'channel_num', None) is None:
        args.channel_num = env.get('channel_num', int(args.sub_band * args.channels_per_subband))
    args.jam_num = env.get('jam_num', 2)
    args.jam_origin_power = env.get('jam_origin_power', 80.0)
    args.jam_sum_channel = env.get('jam_sum_channel', 10)
    args.jam_type = env.get('jam_type', ['narrow_band', 'comb_band', 'wide_band', 'linear_sweep', 'trace_jam'])
    args.jam_change_time = env.get('jam_change_time', 5)
    args.use_alt_jam_model = env.get('use_alt_jam_model', True)
    args.alt_jam_change_time = env.get('alt_jam_change_time', 5)
    args.alt_jam_occ_prob = env.get('alt_jam_occ_prob', 0.2)
    args.use_reactive_jam = env.get('use_reactive_jam', False)
    args.reactive_jam_prob = env.get('reactive_jam_prob', 0.7)
    args.reactive_jam_amp_dbm = env.get('reactive_jam_amp_dbm', 64.0)
    args.use_intelligent_jam = env.get('use_intelligent_jam', False)
    args.intel_jam_change_time = env.get('intel_jam_change_time', 10)
    args.intel_jam_topk = env.get('intel_jam_topk', 5)
    args.log_topk_clean_channels = env.get('log_topk_clean_channels', getattr(args, 'log_topk_clean_channels', False))
    args.topk_clean_K = env.get('topk_clean_K', 5)
    args.topk_log_every_frames = env.get('topk_log_every_frames', 5)
    args.slot_ms = env.get('slot_ms', 1)
    args.frame_slots = env.get('frame_slots', 10)
    args.K = env.get('K', 5)
    args.alpha_leak = env.get('alpha_leak', 0.05)
    args.R_comm = env.get('R_comm', 250)
    args.obs_lag_window_L = env.get('obs_lag_window_L', 10)
    args.ewma_beta = env.get('ewma_beta', 0.7)
    args.nodes = env.get('nodes', args.n_agents)
    args.com_max_distance = env.get('com_max_distance', 10)
    args.space_box_km = tuple(env.get('space_box_km', (5.0, 5.0, 2.0)))
    args.comm_range_km = env.get('comm_range_km', 1.0)
    args.use_graph_obs = env.get('use_graph_obs', getattr(args, 'use_graph_obs', False))
    args.graph_neighbor_threshold = env.get('graph_neighbor_threshold', args.comm_range_km)
    args.graph_obs_smooth_alpha = env.get('graph_obs_smooth_alpha', 0.8)
    args.mobility_model = env.get('mobility_model', 'gaussian')
    args.mobility_sigma_km = env.get('mobility_sigma_km', 0.05)
    args.mobility_reflect = env.get('mobility_reflect', True)
    args.rebuild_comm_each_step = env.get('rebuild_comm_each_step', True)
    args.gt = env.get('gt', 3)
    args.noise_figure = env.get('noise_figure', 7)
    args.receive_Pth = env.get('receive_Pth', -100)
    args.RA_action = env.get('RA_action', ['qpsk_1_2', 'qam16_1_2', 'qam64_2_3', 'qam64_5_6', 'qam256_5_6'])
    if env.get('FH_action') is not None:
        args.FH_action = env.get('FH_action')
    else:
        args.FH_action = list(range(args.channel_num))
    args.Pt_action = env.get('Pt_action', [37, 40, 44])
    args.frame_bit_length = env.get('frame_bit_length', 1024)
    packet_defaults = env.get('packet_num', {})
    if packet_defaults:
        args.packet_num = {k: v for k, v in packet_defaults.items()}
    else:
        a = 1
        args.packet_num = {
            'qpsk_1_2': 6 * a,
            'qam16_1_2': 12 * a,
            'qam64_2_3': 24 * a,
            'qam64_5_6': 30 * a,
            'qam256_5_6': 40 * a
        }
    return args


# arguments of central_v
def get_centralv_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.8
    args.anneal_epsilon = 0.00032
    args.min_epsilon = 0.15
    args.epsilon_anneal_scale = 'episode'

    # lambda of td-lambda return
    args.td_lambda = 0.8

    # how often to save the model
    args.save_cycle = 5000

    # how often to update the target_net
    args.target_update_cycle = 500  # 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of central_v
def get_reinforce_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.8
    args.anneal_epsilon = 0.00032
    args.min_epsilon = 0.15
    args.epsilon_anneal_scale = 'episode'

    # how often to save the model
    args.save_cycle = 5000

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args


# arguments of coma+commnet
def get_commnet_args(args):
    if args.map == '3m':
        args.k = 2
    else:
        args.k = 3
    return args


def get_g2anet_args(args):
    args.attention_dim = 32
    args.hard = True
    return args
