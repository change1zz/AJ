import copy
import random

import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time


def generate_sequences(length, num_sequences, min_value, max_value):
    sequences = []
    for _ in range(num_sequences):
        sequence = []
        for i in range(length):
            available_values = set(range(min_value, max_value+1)) - set(seq[i] for seq in sequences)
            value = random.choice(list(available_values))
            sequence.append(value)
        sequences.append(sequence)
    return sequences

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        self.fixed_FH = generate_sequences(50, self.n_agents, 0, len(self.args.FH_action) - 1)
        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        # self.step = 0
        self.last_action = None
        # Upper-layer AC integration (optional)
        self.upper_ac = None
        self._upper_last = None
        self._upper_reward_accum = 0.0
        self._upper_update_count = 0
        self._prev_adj = None
        self.collect_interf_stats = bool(getattr(self.args, 'collect_interf_stats', True))
        # print('Init RolloutWorker')

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False):
        # Episode buffers
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        record_interf = bool(self.collect_interf_stats)
        interf_stats = [] if record_interf else None
        record_clusters = bool(getattr(self.args, 'hrl_cluster_enable', False))
        cluster_assign = [] if record_clusters else None
        cluster_count = [] if record_clusters else None
        # Reset env (evaluation uses test flag)
        self.env.reset(test=evaluate)
        # reset update counter for logging
        self._upper_update_count = 0
        terminated = False
        step = 0
        episode_reward = 0  # cumulative rewards
        episode_BLER = 0
        episode_trans_rate = 0
        episode_switch_ratio = 0
        episode_transmit_power = 0
        episode_collision = 0
        episode_sinr = 0.0
        # onehot 上一动作缓存使用紧凑类型以减少内存（下游会转成 torch.float32）
        last_action = np.zeros((self.args.n_agents, self.args.n_actions), dtype=np.uint8)
        if self.agents.policy is not None:
            self.agents.policy.init_hidden(1)

        # epsilon (only annealed; Agents.choose_action is greedy in this repo)
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        # Main rollout
        while not terminated and step < self.episode_limit:
            state = self.env.get_state()
            obs = self.env.get_obs(step)

            # 更新上层簇头/GAT 所需的图信息
            if getattr(self.args, 'hrl_cluster_enable', False) and self.agents.hrl_controller is not None:
                snapshot = self.env.get_graph_snapshot() if hasattr(self.env, 'get_graph_snapshot') else None
                self.agents.hrl_controller.update_cluster_plan(snapshot, step=step)

            # 自适应分簇：记录最新簇划分（仅在启用时）
            if record_clusters:
                current_assign = np.full(self.n_agents, -1, dtype=np.int16)
                current_clusters = 0
                if self.agents.hrl_controller is not None:
                    plan = self.agents.hrl_controller.get_cluster_plan()
                    groups = getattr(self.agents.hrl_controller, 'channel_groups', None)
                    if plan is not None:
                        assigns = plan.get('assignments', [])
                        if len(assigns) < self.n_agents:
                            assigns = list(assigns) + [assigns[-1] if assigns else 0] * (self.n_agents - len(assigns))
                        elif len(assigns) > self.n_agents:
                            assigns = assigns[:self.n_agents]
                        current_assign = np.asarray(assigns, dtype=np.int16)
                        current_clusters = len(plan.get('clusters', [])) or (len(groups) if groups else 0)
                    elif groups:
                        current_clusters = len(groups)
                cluster_assign.append(current_assign)
                cluster_count.append([max(1, current_clusters)])

            # Upper-layer decision at meta-period boundaries
            if (self.upper_ac is not None) and getattr(self.args, 'hrl_enable', False) and \
               getattr(self.args, 'hrl_masking_mode', 'none') == 'fixed_channel_groups' and \
               (step % getattr(self.args, 'hrl_meta_period', 20) == 0):
                graph_state = None
                if (self.agents.hrl_controller is not None) and hasattr(self.agents.hrl_controller, 'build_upper_graph_state'):
                    graph_state = self.agents.hrl_controller.build_upper_graph_state(prev_adj=self._prev_adj)
                adj = graph_state['adj'] if (graph_state is not None and 'adj' in graph_state) else None
                group_ids, logp, logits = self.upper_ac.decide_groups(obs, adj)
                self.upper_ac.apply_to_controller(group_ids)
                self._upper_last = {'obs': obs.copy(), 'adj': adj.copy() if adj is not None else None, 'logp': logp, 'groups': group_ids, 'logits': logits}
                self._upper_reward_accum = 0.0
                self._prev_adj = adj

            # Choose actions for each agent
            actions, avail_actions, actions_onehot = [], [], []
            ra_length = len(self.args.RA_action)
            pt_length = len(self.args.Pt_action)
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                if self.agents.policy is not None:
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action)
                else:
                    # Fallback random-like policy when no learning policy
                    mcs_action = 3
                    pt_action = 1
                    action_FH = random.sample(range(len(self.args.FH_action)), self.args.nodes)
                    action = ((action_FH[agent_id] * ra_length * pt_length) + (mcs_action * pt_length) + pt_action)
                # onehot（uint8存储，训练时再转为float32）
                action_onehot = np.zeros(self.args.n_actions, dtype=np.uint8)
                action_onehot[action] = 1
                actions.append(np.int32(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            # Step environment（返回包含平均 SINR）
            reward, BLER, trans_rate, switch_ratio, transmit_power, collision, sinr_avg, terminated = self.env.step(actions, step)

            # Accumulate episode data
            o.append(obs)
            s.append(copy.deepcopy(state))
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            # 终止/填充标志使用 uint8，训练阶段转换为 float32
            terminate.append([np.uint8(terminated)])
            padded.append([np.uint8(0)])

            episode_reward += reward
            episode_BLER += BLER
            episode_trans_rate += trans_rate
            episode_sinr += sinr_avg
            episode_switch_ratio += switch_ratio
            episode_transmit_power += transmit_power
            episode_collision += collision

            # Upper-layer AC update & stats
            if self.upper_ac is not None and getattr(self.args, 'hrl_enable', False):
                self._upper_reward_accum += reward
                if (step > 0 and step % getattr(self.args, 'hrl_meta_period', 20) == 0) or terminated:
                    if self._upper_last is not None:
                        R_vec = np.full(self.n_agents, self._upper_reward_accum, dtype=np.float32)
                        self.upper_ac.store_period(self._upper_last['obs'], self._upper_last['adj'], self._upper_last['logp'], self._upper_last['groups'], R_vec, terminated, self._upper_last.get('logits'))
                        stats = self.upper_ac.update()
                        # 控制台日志频率：按更新次数间隔打印
                        self._upper_update_count += 1
                        log_every = int(getattr(self.args, 'upper_log_every', 5))
                        if isinstance(stats, dict) and stats and log_every > 0 and (self._upper_update_count % log_every == 0):
                            try:
                                print(f"[UpperGraphAC] Update stats: loss_pi={stats.get('loss_pi', float('nan')):.4f}, loss_v={stats.get('loss_v', float('nan')):.4f}, A_mean={stats.get('A_mean', float('nan')):.4f}, entropy={stats.get('entropy', float('nan')):.4f}")
                            except Exception:
                                print(f"[UpperGraphAC] Update stats: {stats}")
                        self._upper_reward_accum = 0.0

            # Interference EWMA/moving average stats（可选）
            need_stats = record_interf or getattr(self.args, 'log_topk_clean_channels', False)
            moving_avg = ewma = None
            if need_stats:
                moving_avg, ewma = self.env.get_interference_stats()
            if record_interf and interf_stats is not None and moving_avg is not None and ewma is not None:
                interf_stats.append(np.concatenate([moving_avg, ewma]))
            if getattr(self.args, 'log_topk_clean_channels', False) and ewma is not None and step > 0 and (step % getattr(self.args, 'frame_slots', 20) == 0):
                frame_slots = int(getattr(self.args, 'frame_slots', 20))
                frame_id = step // frame_slots
                every_frames = int(getattr(self.args, 'topk_log_every_frames', 5))
                if every_frames > 0 and (frame_id % every_frames == 0):
                    K = int(getattr(self.args, 'topk_clean_K', min(5, self.env.channel_num)))
                    clean_idx = np.argsort(ewma)[:K]
                    clean_vals = ewma[clean_idx]
                    print(f"[Frame {frame_id}] Top-{K} clean subcarriers (ewma): indices={clean_idx.tolist()}, values={np.round(clean_vals, 3).tolist()}")
                    q = [0.1, 0.25, 0.5, 0.75, 0.9]
                    qs = np.quantile(ewma, q).tolist()
                    print(f"[Frame {frame_id}] EWMA quantiles: q10={qs[0]:.3f}, q25={qs[1]:.3f}, median={qs[2]:.3f}, q75={qs[3]:.3f}, q90={qs[4]:.3f}, min={np.min(ewma):.3f}, max={np.max(ewma):.3f}")

            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # Append last obs/state and build nexts
        obs = self.env.get_obs(1)
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = self.env.get_avail_actions()
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape), dtype=np.float32))
            u.append(np.zeros([self.n_agents, 1], dtype=np.int32))
            s.append(np.zeros(self.state_shape, dtype=np.float32))
            r.append(np.array([0.], dtype=np.float32))
            o_next.append(np.zeros((self.n_agents, self.obs_shape), dtype=np.float32))
            s_next.append(np.zeros(self.state_shape, dtype=np.float32))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions), dtype=np.uint8))
            avail_u.append(np.zeros((self.n_agents, self.n_actions), dtype=np.uint8))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions), dtype=np.uint8))
            if record_interf and interf_stats is not None:
                interf_stats.append(np.zeros(self.env.channel_num * 2, dtype=np.float32))
            padded.append(np.array([1], dtype=np.uint8))
            terminate.append(np.array([1], dtype=np.uint8))

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy())
        if record_interf and interf_stats is not None:
            episode['interf_stats'] = interf_stats.copy()
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if evaluate and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
            self.env.save_replay()
        return episode, episode_reward, step, episode_BLER / step, episode_trans_rate / step, episode_sinr / step, episode_switch_ratio / step, episode_transmit_power / step, episode_collision / step

    def generate_test_episode(self, distance_1, distance_2, jam_num, power, test_jamming_type=None):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        record_interf = bool(self.collect_interf_stats)
        interf_stats = [] if record_interf else None
        evaluate = True
        if test_jamming_type is not None:
            self.env.reset_test(distance_1, distance_2, jam_num, power, test_jamming_type)
        terminated = False
        step = 0
        # reset update counter for logging
        self._upper_update_count = 0
        episode_reward = 0  # cumulative rewards
        episode_BLER = 0
        episode_trans_rate = 0
        episode_switch_ratio = 0
        episode_transmit_power = 0
        episode_collision = 0
        episode_sinr = 0.0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        if self.agents.policy is not None:
            self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        # sample z for maven
        if self.args.alg == 'maven':
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        ra_length = len(self.args.RA_action)
        pt_length = len(self.args.Pt_action)
        while not terminated and step < self.episode_limit:
            state = self.env.get_state()
            obs = self.env.get_obs(step)

            actions, avail_actions, actions_onehot = [], [], []
            # time_start = time.time()
            # for i in range(1000000):
            action_FH = random.sample(range(len(self.args.FH_action)), self.args.nodes)
            # time_end = time.time()
            # print((time_end - time_start) / 1000000)
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                if self.agents.policy is not None:
                    # time_start = time.time()
                    # for i in range(1000):
                    action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, avail_action)
                    # time_end = time.time()
                    # print(time_end - time_start)
                else:
                    mcs_action = random.randint(3, 4)
                    pt_action = 2
                    if self.args.alg == 'R-FH':
                        action = ((action_FH[agent_id] * ra_length * pt_length) + (mcs_action * pt_length) + pt_action)

                        time_start = time.time()
                        # for i in range(1000000):
                        action_FH = random.sample(range(len(self.args.FH_action)), self.args.nodes)
                        action = ((action_FH[agent_id] * ra_length * pt_length) + (mcs_action * pt_length) + pt_action)
                        # time_end = time.time()
                        # print((time_end - time_start) / 1000000)
                    elif self.args.alg == 'fix-FH':
                        action_fh = self.fixed_FH[agent_id][step // 4]
                        action = ((action_fh * ra_length * pt_length) + (mcs_action * pt_length) + pt_action)

                        # time_start = time.time()
                        # for i in range(1000000):
                        action_fh = self.fixed_FH[agent_id][step // 4]
                        action = ((action_fh * ra_length * pt_length) + (mcs_action * pt_length) + pt_action)
                        # time_end = time.time()
                        # print((time_end - time_start) / 1000000)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(np.int64(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, BLER, trans_rate, switch_ratio, transmit_power, collision, sinr_avg, terminated = self.env.step(actions, step)
            # win_tag = True if terminated and 'battle_won' in info and info['battle_won'] else False

            o.append(obs)
            s.append(copy.deepcopy(state))
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])

            episode_reward += reward
            # if wintag:
            #     episode_reward += 200
            episode_BLER += BLER
            episode_trans_rate += trans_rate
            # 测试路径不返回输出 SINR（此处不累积/不使用 sinr_avg）
            episode_switch_ratio += switch_ratio
            episode_transmit_power += transmit_power
            episode_collision += collision
            # accumulate upper reward
            if self.upper_ac is not None and getattr(self.args, 'hrl_enable', False):
                self._upper_reward_accum += reward
                # Update and learn at boundary or termination (one-step sync update)
                if (step > 0 and step % getattr(self.args, 'hrl_meta_period', 20) == 0) or terminated:
                    if self._upper_last is not None:
                        R_vec = np.full(self.n_agents, self._upper_reward_accum, dtype=np.float32)
                        self.upper_ac.store_period(self._upper_last['obs'], self._upper_last['adj'], self._upper_last['logp'], self._upper_last['groups'], R_vec, terminated, self._upper_last.get('logits'))
                        stats = self.upper_ac.update()
                        # 控制台日志频率：按更新次数间隔打印
                        self._upper_update_count += 1
                        log_every = int(getattr(self.args, 'upper_log_every', 5))
                        if isinstance(stats, dict) and stats and log_every > 0 and (self._upper_update_count % log_every == 0):
                            try:
                                print(f"[UpperGraphAC] Update stats: loss_pi={stats.get('loss_pi', float('nan')):.4f}, loss_v={stats.get('loss_v', float('nan')):.4f}, A_mean={stats.get('A_mean', float('nan')):.4f}, entropy={stats.get('entropy', float('nan')):.4f}")
                            except Exception:
                                print(f"[UpperGraphAC] Update stats: {stats}")
                        # reset accum
                        self._upper_reward_accum = 0.0
            need_stats = record_interf or getattr(self.args, 'log_topk_clean_channels', False)
            moving_avg = ewma = None
            if need_stats:
                moving_avg, ewma = self.env.get_interference_stats()
            if record_interf and interf_stats is not None and moving_avg is not None and ewma is not None:
                interf_stats.append(np.concatenate([moving_avg, ewma]))
            # 帧边界输出 Top-K 干扰最低子载波（基于 ewma）与分位数
            if getattr(self.args, 'log_topk_clean_channels', False) and ewma is not None and step > 0 and (step % getattr(self.args, 'frame_slots', 20) == 0):
                frame_slots = int(getattr(self.args, 'frame_slots', 20))
                frame_id = step // frame_slots
                every_frames = int(getattr(self.args, 'topk_log_every_frames', 5))
                if every_frames > 0 and (frame_id % every_frames == 0):
                    K = int(getattr(self.args, 'topk_clean_K', min(5, self.env.channel_num)))
                    clean_idx = np.argsort(ewma)[:K]
                    clean_vals = ewma[clean_idx]
                    print(f"[Frame {frame_id}] Top-{K} clean subcarriers (ewma): indices={clean_idx.tolist()}, values={np.round(clean_vals, 3).tolist()}")
                    # EWMA 干扰分位数日志
                    q = [0.1, 0.25, 0.5, 0.75, 0.9]
                    qs = np.quantile(ewma, q).tolist()
                    print(f"[Frame {frame_id}] EWMA quantiles: q10={qs[0]:.3f}, q25={qs[1]:.3f}, median={qs[2]:.3f}, q75={qs[3]:.3f}, q90={qs[4]:.3f}, min={np.min(ewma):.3f}, max={np.max(ewma):.3f}")
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        # last obs
        obs = self.env.get_obs(1)
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = self.env.get_avail_actions()
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            if record_interf and interf_stats is not None:
                interf_stats.append(np.zeros(self.env.channel_num * 2))
            if record_clusters:
                cluster_assign.append(np.zeros(self.n_agents, dtype=np.int16))
                cluster_count.append([0])
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy(),
                       )
        if record_interf and interf_stats is not None:
            episode['interf_stats'] = interf_stats.copy()
        if record_clusters and cluster_assign is not None:
            episode['cluster_assign'] = np.array(cluster_assign, dtype=np.int16).copy()
            episode['cluster_count'] = np.array(cluster_count, dtype=np.int16).copy()
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        return episode, episode_reward, step, episode_BLER / step, episode_trans_rate / step, episode_switch_ratio / step, episode_transmit_power / step, episode_collision / step
# RolloutWorker for communication
# class CommRolloutWorker:
#     def __init__(self, env, agents, args):
#         self.env = env
#         self.agents = agents
#         self.episode_limit = args.episode_limit
#         self.n_actions = args.n_actions
#         self.n_agents = args.n_agents
#         self.state_shape = args.state_shape
#         self.obs_shape = args.obs_shape
#         self.args = args
#
#         self.epsilon = args.epsilon
#         self.anneal_epsilon = args.anneal_epsilon
#         self.min_epsilon = args.min_epsilon
#         print('Init CommRolloutWorker')
#
#     @torch.no_grad()
#     def generate_episode(self, episode_num=None, evaluate_data=False):
#         if self.args.replay_dir != '' and evaluate_data and episode_num == 0:  # prepare for save replay
#             self.env.close()
#         o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
#         self.env.reset()
#         terminated = False
#         step = 0
#         episode_reward = 0
#         last_action = np.zeros((self.args.n_agents, self.args.n_actions))
#         self.agents.policy.init_hidden(1)
#         epsilon = 0 if evaluate_data else self.epsilon
#         if self.args.epsilon_anneal_scale == 'episode':
#             epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
#         while not terminated and step < self.episode_limit:
#             # time.sleep(0.2)
#             obs = self.env.get_obs()
#             state = self.env.get_state()
#             actions, avail_actions, actions_onehot = [], [], []
#
#             # get the weights of all actions for all agents
#             weights = self.agents.get_action_weights(np.array(obs), last_action)
#
#             # choose action for each agent
#             for agent_id in range(self.n_agents):
#                 avail_action = self.env.get_avail_agent_actions(agent_id)
#                 action = self.agents.choose_action(weights[agent_id], avail_action, epsilon)
#
#                 # generate onehot vector of th action
#                 action_onehot = np.zeros(self.args.n_actions)
#                 action_onehot[action] = 1
#                 actions.append(np.int(action))
#                 actions_onehot.append(action_onehot)
#                 avail_actions.append(avail_action)
#                 last_action[agent_id] = action_onehot
#
#             reward, terminated, info = self.env.step(actions)
#             o.append(obs)
#             s.append(state)
#             u.append(np.reshape(actions, [self.n_agents, 1]))
#             u_onehot.append(actions_onehot)
#             avail_u.append(avail_actions)
#             r.append([reward])
#             terminate.append([terminated])
#             padded.append([0.])
#             episode_reward += reward
#             step += 1
#             # if terminated:
#             #     time.sleep(1)
#             if self.args.epsilon_anneal_scale == 'step':
#                 epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
#         # last obs
#         obs = self.env.get_obs()
#         state = self.env.get_state()
#         o.append(obs)
#         s.append(state)
#         o_next = o[1:]
#         s_next = s[1:]
#         o = o[:-1]
#         s = s[:-1]
#         # get avail_action for last obs，because target_q needs avail_action in training
#         avail_actions = []
#         for agent_id in range(self.n_agents):
#             avail_action = self.env.get_avail_agent_actions(agent_id)
#             avail_actions.append(avail_action)
#         avail_u.append(avail_actions)
#         avail_u_next = avail_u[1:]
#         avail_u = avail_u[:-1]
#
#         # if step < self.episode_limit，padding
#         for i in range(step, self.episode_limit):
#             o.append(np.zeros((self.n_agents, self.obs_shape)))
#             u.append(np.zeros([self.n_agents, 1]))
#             s.append(np.zeros(self.state_shape))
#             r.append([0.])
#             o_next.append(np.zeros((self.n_agents, self.obs_shape)))
#             s_next.append(np.zeros(self.state_shape))
#             u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
#             avail_u.append(np.zeros((self.n_agents, self.n_actions)))
#             avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
#             padded.append([1.])
#             terminate.append([1.])
#
#         episode = dict(o=o.copy(),
#                        s=s.copy(),
#                        u=u.copy(),
#                        r=r.copy(),
#                        avail_u=avail_u.copy(),
#                        o_next=o_next.copy(),
#                        s_next=s_next.copy(),
#                        avail_u_next=avail_u_next.copy(),
#                        u_onehot=u_onehot.copy(),
#                        padded=padded.copy(),
#                        terminated=terminate.copy()
#                        )
#         # add episode dim
#         for key in episode.keys():
#             episode[key] = np.array([episode[key]])
#         if not evaluate_data:
#             self.epsilon = epsilon
#             # print('Epsilon is ', self.epsilon)
#         if evaluate_data and episode_num == self.args.evaluate_epoch - 1 and self.args.replay_dir != '':
#             self.env.save_replay()
#             self.env.close()
#         return episode, episode_reward, step
