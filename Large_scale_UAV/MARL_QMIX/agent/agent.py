import numpy as np
import torch
from torch.distributions import Categorical
import time


# Agent no communication
class Agents:
    def __init__(self, args, env=None):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'vdn':
            from policy.vdn import VDN
            self.policy = VDN(args)
        elif args.alg == 'iql' or args.alg == 'iql_no_mcs':
            from policy.iql import IQL
            self.policy = IQL(args)
        elif args.alg == 'qmix' or args.alg == 'qmix_no_mcs':
            from policy.qmix import QMIX
            self.policy = QMIX(args)
        elif args.alg == 'coma':
            from policy.coma import COMA
            self.policy = COMA(args)
        elif args.alg == 'qtran_alt':
            from policy.qtran_alt import QtranAlt
            self.policy = QtranAlt(args)
        elif args.alg == 'qtran_base':
            from policy.qtran_base import QtranBase
            self.policy = QtranBase(args)
        elif args.alg == 'maven':
            from policy.maven import MAVEN
            self.policy = MAVEN(args)
        elif args.alg == 'central_v':
            from policy.central_v import CentralV
            self.policy = CentralV(args)
        elif args.alg == 'reinforce':
            from policy.reinforce import Reinforce
            self.policy = Reinforce(args)
        elif args.alg == 'R-FH':
            self.policy = None
        elif args.alg == 'fix-FH':
            self.policy = None
        elif args.alg == 'qmix_attention':
            from policy.qmix_attention import QMIX_attention
            self.policy = QMIX_attention(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        self.env = env
        # HRL upper-level controller (skeleton; default disabled)
        self.hrl_controller = None
        if getattr(self.args, "hrl_enable", False):
            try:
                from controllers.hrl_controller import HRLController
                self.hrl_controller = HRLController(self.args, env=self.env)
                print("HRLController enabled")
            except Exception as e:
                print(f"HRLController init failed: {e}")
                self.hrl_controller = None

    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon=None):
        inputs = obs.copy()
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.
        inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # time_start = time.time()
        # for i in range(1000000):
        try:
            with torch.no_grad():
                q_value, new_hidden = self.policy.eval_rnn(inputs, hidden_state)
            # assign new hidden (move to policy device if necessary)
            try:
                self.policy.eval_hidden[:, agent_num, :] = new_hidden
            except Exception:
                # fallback: ensure device alignment
                self.policy.eval_hidden = self.policy.eval_hidden.cpu()
                self.policy.eval_hidden[:, agent_num, :] = new_hidden.cpu()
                if self.args.cuda:
                    self.policy.eval_hidden = self.policy.eval_hidden.cuda()
            q_value = q_value
        except RuntimeError as e:
            # Handle OOM: try clearing cache and retry once, otherwise fallback to CPU
            err_str = str(e)
            if 'out of memory' in err_str.lower():
                print('[WARN] CUDA OOM in choose_action, emptying cache and retrying on GPU')
                try:
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        q_value, new_hidden = self.policy.eval_rnn(inputs, hidden_state)
                    try:
                        self.policy.eval_hidden[:, agent_num, :] = new_hidden
                    except Exception:
                        self.policy.eval_hidden = self.policy.eval_hidden.cpu()
                        self.policy.eval_hidden[:, agent_num, :] = new_hidden.cpu()
                        if self.args.cuda:
                            self.policy.eval_hidden = self.policy.eval_hidden.cuda()
                except RuntimeError:
                    print('[WARN] Retry on GPU failed, falling back to CPU for this step')
                    # Move model to CPU temporarily
                    try:
                        model_device = next(self.policy.parameters()).device
                    except StopIteration:
                        model_device = torch.device('cuda' if self.args.cuda else 'cpu')
                    try:
                        self.policy.cpu()
                        inputs_cpu = inputs.cpu()
                        hidden_cpu = hidden_state.cpu()
                        with torch.no_grad():
                            q_value_cpu, new_hidden_cpu = self.policy.eval_rnn(inputs_cpu, hidden_cpu)
                        # move outputs back to original device
                        q_value = q_value_cpu.to(model_device)
                        new_hidden = new_hidden_cpu.to(model_device)
                        # restore model to original device
                        if self.args.cuda and model_device.type == 'cuda':
                            try:
                                self.policy.cuda()
                            except Exception:
                                pass
                        try:
                            self.policy.eval_hidden[:, agent_num, :] = new_hidden
                        except Exception:
                            # best-effort assignment
                            self.policy.eval_hidden = self.policy.eval_hidden.cpu()
                            self.policy.eval_hidden[:, agent_num, :] = new_hidden.cpu()
                            if self.args.cuda:
                                self.policy.eval_hidden = self.policy.eval_hidden.cuda()
                    except Exception as e2:
                        print(f'[ERROR] Fallback CPU evaluation failed: {e2}')
                        raise
            else:
                raise
        # time_end = time.time()
        # print((time_end - time_start)/1000000)

        q_value[avail_actions == 0.0] = - float("inf")
        # Optional HRL gating (skeleton)
        if self.hrl_controller is not None:
            mask = self.hrl_controller.get_action_mask(agent_num=agent_num, obs=obs, q_value=q_value, avail_actions=avail_actions)
            if mask is not None:
                if not torch.is_tensor(mask):
                    mask = torch.tensor(mask, dtype=q_value.dtype)
                # Align mask shape with q_value (1, n_actions)
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                if self.args.cuda:
                    mask = mask.cuda()
                q_value[mask == 0] = - float("inf")
        action = torch.argmax(q_value)


        return action

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """
        
        action = Categorical(prob).sample().long()
        return action

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != 'z':
                batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        # 周期性保存：与 CommAgents 保持一致，每 save_cycle 步保存一次
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)


# Agent for communication
class CommAgents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        alg = args.alg
        if alg.find('reinforce') > -1:
            from policy.reinforce import Reinforce
            self.policy = Reinforce(args)
        elif alg.find('coma') > -1:
            from policy.coma import COMA
            self.policy = COMA(args)
        elif alg.find('central_v') > -1:
            from policy.central_v import CentralV
            self.policy = CentralV(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init CommAgents')

    # 根据weights得到概率，然后再根据epsilon选动作
    def choose_action(self, weights, avail_actions, epsilon):
        weights = weights.unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # 可以选择的动作的个数
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(weights, dim=-1)
        # 在训练的时候给概率分布添加噪音
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        action = Categorical(prob).sample().long()
        return action

    def get_action_weights(self, obs, last_action):
        obs = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(last_action, dtype=torch.float32)
        inputs = list()
        inputs.append(obs)
        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            inputs.append(last_action)
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents))
        inputs = torch.cat([x for x in inputs], dim=1)
        if self.args.cuda:
            inputs = inputs.cuda()
            self.policy.eval_hidden = self.policy.eval_hidden.cuda()
        try:
            with torch.no_grad():
                weights, new_hidden = self.policy.eval_rnn(inputs, self.policy.eval_hidden)
            self.policy.eval_hidden = new_hidden
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print('[WARN] CUDA OOM in get_action_weights, emptying cache and retrying')
                try:
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        weights, new_hidden = self.policy.eval_rnn(inputs, self.policy.eval_hidden)
                    self.policy.eval_hidden = new_hidden
                except RuntimeError:
                    print('[WARN] Retry failed, moving model to CPU for this call')
                    model_device = next(self.policy.parameters()).device
                    try:
                        self.policy.cpu()
                        inputs_cpu = inputs.cpu()
                        hidden_cpu = self.policy.eval_hidden.cpu()
                        with torch.no_grad():
                            weights_cpu, new_hidden_cpu = self.policy.eval_rnn(inputs_cpu, hidden_cpu)
                        weights = weights_cpu.to(model_device)
                        self.policy.eval_hidden = new_hidden_cpu.to(model_device)
                        if self.args.cuda and model_device.type == 'cuda':
                            try:
                                self.policy.cuda()
                            except Exception:
                                pass
                    except Exception as e2:
                        print(f'[ERROR] CPU fallback failed in get_action_weights: {e2}')
                        raise
            else:
                raise
        weights = weights.reshape(self.args.n_agents, self.args.n_actions)
        return weights.cpu()

    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma在训练时也需要epsilon计算动作的执行概率
        # 每次学习时，各个episode的长度不一样，因此取其中最长的episode作为所有episode的长度
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)










