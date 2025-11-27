"""
统计测试阶段动作选择比例

运行方式:
conda activate intelligent_AJ
python scripts\analyze_action_ratio.py --episodes 10 --config configs/default.yaml
"""
import argparse
import numpy as np
import torch

from common.arguments import get_common_args, get_mixer_args, get_env_args
from agent.agent import Agents
from common.rollout import RolloutWorker
from env_clean import ComEnvClean as EnvCls


def dotted_dict(d):
    class Obj:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    return Obj(**d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10, help='测试回合数')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    args_cli = parser.parse_args()

    args = get_common_args()
    args = get_mixer_args(args)
    args = get_env_args(args)

    env = EnvCls(args)
    agents = Agents(args, env)
    rollout_worker = RolloutWorker(env, agents, args)

    total_counts = np.zeros(args.n_actions, dtype=np.int64)
    total_steps = 0

    with torch.no_grad():
        for ep in range(args_cli.episodes):
            episode, _, step, _, _, _, _, _, _ = rollout_worker.generate_episode(evaluate=True)
            actions = episode['u'][0][:step]  # shape [T, n_agents, 1]
            onehot = episode['u_onehot'][0][:step]  # [T, n_agents, n_actions]
            total_counts += onehot.sum(axis=(0, 1))
            total_steps += actions.shape[0] * actions.shape[1]

    ratios = total_counts / max(1, total_counts.sum())
    print(f"累计步数: {total_steps}, 累计动作次数: {total_counts.sum()}")
    for idx, (cnt, ratio) in enumerate(zip(total_counts, ratios)):
        print(f"动作 {idx:4d}: 次数={cnt:8d}, 占比={ratio:.4f}")


if __name__ == '__main__':
    main()
