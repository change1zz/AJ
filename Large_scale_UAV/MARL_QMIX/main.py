from runner import Runner
from env import com_env
try:
    from env_test import com_env_test
except Exception:
    com_env_test = None

from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, \
    get_commnet_args, get_g2anet_args, get_env_args
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import random
from matplotlib.font_manager import FontProperties


def create_matrix(n, m, fill_value=0):
    # 使用列表理解创建一个 n 行 m 列的矩阵，并用 fill_value 填充
    return [[fill_value for _ in range(m)] for _ in range(n)]


def test_capacity(test_epoch, test_jamming_type, distance_1, distance_2, jam_num, power, alg, nodes_num=None):

    args = get_common_args()
    if nodes_num is not None:
        args.n_agents = nodes_num
    args = get_mixer_args(args)
    args = get_env_args(args)

    if alg == 'qmix':
        args.alg = 'qmix'
        args.rnn_hidden_dim = 256
    elif alg == 'iql':
        args.alg = 'iql'
        args.rnn_hidden_dim = 256
    elif alg == 'iql_no_mcs':
        args.alg = 'iql_no_mcs'
        args.rnn_hidden_dim = 256
    elif alg == 'R-FH':
        args.alg = 'R-FH'
    elif alg == 'fix-FH':
        args.alg = 'fix-FH'
    elif alg == 'qmix_no_mcs':
        args.alg = 'qmix_no_mcs'
        args.rnn_hidden_dim = 256
        args.two_hyper_layers = True
    elif alg == 'vdn':
        args.alg = 'vdn'
        args.rnn_hidden_dim = 256
    elif alg == 'qmix_attention':
        args.alg = 'qmix_attention'
        args.rnn_hidden_dim = 256
        args.two_hyper_layers = True


    env = com_env_test(args)
    # test_jamming_type = 0
    runner = Runner(env, args)
    reward, BLER, trans_rate, switch_ratio, trans_power, episode_collision = runner.test(test_epoch, test_jamming_type,
                                                                                         distance_1, distance_2,
                                                                                         jam_num, power)

    print('Reward is ', reward)
    print('BLER is :', BLER)
    print('Transmit_rate is :', trans_rate)
    print('Switch_ratio :', switch_ratio)
    print('Episode_transmit_power is :', trans_power)
    print('Episode_collision is :', episode_collision)

    if BLER < 0:
        BLER = 0
    success_ratio = round(100 - BLER * 100, 2)
    trans_rate = round(trans_rate, 2)
    switch_ratio = round(switch_ratio, 2)
    trans_power = round(trans_power, 2)
    return success_ratio, trans_rate, switch_ratio, trans_power


def data_save(data):
    # 保存为 CSV
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_path = '3.0/result/' + timestr
    os.makedirs(output_path, exist_ok=True)
    # 假设每个子列表对应不同的算法
    df_list = []

    # 假设每个子列表对应不同的样本
    for i in range(len(data['success_ratios_alg'])):
        for j in range(len(data['success_ratios_alg'][i])):
            row = {
                'sample': i + 1,
                'success_ratio': data['success_ratios_alg'][i][j],
                'trans_rate': data['trans_rates_alg'][i][j],
                'switch_ratio': data['switch_ratio_alg'][i][j],
                'power': data['power_alg'][i][j]
            }
            df_list.append(row)

    # 创建最终的长格式 DataFrame
    df_long = pd.DataFrame(df_list)

    df_long.to_csv(os.path.join(output_path, 'output.csv'), index=False, encoding='utf-8-sig')


def test_capacity_different_type():
    test_parameter = 'nodes'  # [channel_num, jam_type, nodes]
    test_epoch = 100
    algs = ['iql']  #['qmix', 'qmix_attention', 'qmix_no_mcs', 'iql', 'R-FH', 'fix-FH']
    # ['qmix_attention', 'qmix', 'iql', 'iql_no_mcs',  'R-FH', 'fix-FH']
    jamming_types = ['narrow_band', 'wide_band', 'linear_sweep', 'trace_jam', 'comb_band', 'mix_jam']

    test_jamming_num = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    test_nodes_num = [4, 5, 6, 7, 8]
    # args = get_common_args()
    # args = get_mixer_args(args)
    # args = get_env_args(args)
    distance_1 = 160
    distance_2 = 160
    power = [74, 74]
    if test_parameter == 'jam_type':
        success_ratios_alg = create_matrix(len(algs), len(test_jamming_num))
        trans_rates_alg = create_matrix(len(algs), len(test_jamming_num))
        switch_ratio_alg = create_matrix(len(algs), len(test_jamming_num))
        power_alg = create_matrix(len(algs), len(test_jamming_num))

    elif test_parameter == 'nodes':
        success_ratios_alg = create_matrix(len(algs), len(test_nodes_num))
        trans_rates_alg = create_matrix(len(algs), len(test_nodes_num))
        switch_ratio_alg = create_matrix(len(algs), len(test_nodes_num))
        power_alg = create_matrix(len(algs), len(test_nodes_num))

    else:  # test_parameter == 'channel_num'
        success_ratios_alg = create_matrix(len(algs), len(test_jamming_num))
        trans_rates_alg = create_matrix(len(algs), len(test_jamming_num))
        switch_ratio_alg = create_matrix(len(algs), len(test_jamming_num))
        power_alg = create_matrix(len(algs), len(test_jamming_num))

    # success_ratios, trans_rates = [], []
    for i in range(len(algs)):
        alg = algs[i]
        #  测试不同样式
        # if test_parameter == 'jam_type':
        #     jam_num = [15, 15]
        #     for j in range(len(jamming_types)):
        #         jamming_type = jamming_types[j]
        #         # if jamming_type == 'mix_jam':
        #         #     mix_jam = [['narrow_band', 'comb_band'], ['narrow_band', 'wide_band'],
        #         #                   ['narrow_band', 'linear_sweep'], ['narrow_band', 'trace_jam'],
        #         #                   ['comb_band', 'wide_band'], ['comb_band', 'linear_sweep'],
        #         #                   ['comb_band', 'trace_jam'], ['wide_band', 'linear_sweep'],
        #         #                   ['wide_band', 'trace_jam'], ['linear_sweep', 'trace_jam']]
        #         # random.shuffle(mix_jam)
        #         # jamming_type = mix_jam.pop()
        #         test_jamming_type = jamming_type
        #         success_ratio_evaluate, trans_rate_evaluate, switch_ratio_evaluate, power_evaluate = test_capacity(test_epoch, test_jamming_type, distance_1, distance_2, jam_num, power, alg)
        #
        #         success_ratios_alg[i][j] = success_ratio_evaluate
        #         trans_rates_alg[i][j] = trans_rate_evaluate
        #         switch_ratio_alg[i][j] = switch_ratio_evaluate
        #         power_alg[i][j] = power_evaluate

        #  测试不同干扰信道数
        if test_parameter == 'channel_num':
            for j in range(len(test_jamming_num)):
                jam_num = [test_jamming_num[j], test_jamming_num[j]]
                # if jamming_type == 'mix_jam':
                # if len(mix_jam) == 0:
                #     mix_jam = [['narrow_band', 'comb_band'], ['narrow_band', 'wide_band'],
                #                ['narrow_band', 'linear_sweep'], ['narrow_band', 'trace_jam'],
                #                ['comb_band', 'wide_band'], ['comb_band', 'linear_sweep'],
                #                ['comb_band', 'trace_jam'], ['wide_band', 'linear_sweep'],
                #                ['wide_band', 'trace_jam'], ['linear_sweep', 'trace_jam']]
                # jamming_type = mix_jam.pop()
                test_jamming_type = 'mix_jam'
                success_ratio_evaluate, trans_rate_evaluate, switch_ratio_evaluate, power_evaluate = test_capacity(test_epoch, test_jamming_type, distance_1, distance_2, jam_num, power, alg)

                success_ratios_alg[i][j] = success_ratio_evaluate
                trans_rates_alg[i][j] = trans_rate_evaluate
                switch_ratio_alg[i][j] = switch_ratio_evaluate
                power_alg[i][j] = power_evaluate

        #  测试不同干扰信道数(不同干扰样式)
        if test_parameter == 'jam_type':
            for j in range(len(test_jamming_num)):
                jam_num = [test_jamming_num[j], 0]
                # if jamming_type == 'mix_jam':
                # if len(mix_jam) == 0:
                #     mix_jam = [['narrow_band', 'comb_band'], ['narrow_band', 'wide_band'],
                #                ['narrow_band', 'linear_sweep'], ['narrow_band', 'trace_jam'],
                #                ['comb_band', 'wide_band'], ['comb_band', 'linear_sweep'],
                #                ['comb_band', 'trace_jam'], ['wide_band', 'linear_sweep'],
                #                ['wide_band', 'trace_jam'], ['linear_sweep', 'trace_jam']]
                # jamming_type = mix_jam.pop()

                test_jamming_type = 'trace_jam'
                success_ratio_evaluate, trans_rate_evaluate, switch_ratio_evaluate, power_evaluate = test_capacity(test_epoch, test_jamming_type, distance_1, distance_2, jam_num, power, alg)

                success_ratios_alg[i][j] = success_ratio_evaluate
                trans_rates_alg[i][j] = trans_rate_evaluate
                switch_ratio_alg[i][j] = switch_ratio_evaluate
                power_alg[i][j] = power_evaluate

        #  测试不同节点数量
        if test_parameter == 'nodes':
            for j in range(len(test_nodes_num)):
                nodes_num = test_nodes_num[j]

                jam_num=[15, 0]
                # if jamming_type == 'mix_jam':
                # if len(mix_jam) == 0:
                #     mix_jam = [['narrow_band', 'comb_band'], ['narrow_band', 'wide_band'],
                #                ['narrow_band', 'linear_sweep'], ['narrow_band', 'trace_jam'],
                #                ['comb_band', 'wide_band'], ['comb_band', 'linear_sweep'],
                #                ['comb_band', 'trace_jam'], ['wide_band', 'linear_sweep'],
                #                ['wide_band', 'trace_jam'], ['linear_sweep', 'trace_jam']]
                # jamming_type = mix_jam.pop()
                test_jamming_type = 'mix_jam'
                success_ratio_evaluate, trans_rate_evaluate, switch_ratio_evaluate, power_evaluate = test_capacity(test_epoch, test_jamming_type, distance_1, distance_2, jam_num, power, alg, nodes_num)

                success_ratios_alg[i][j] = success_ratio_evaluate
                trans_rates_alg[i][j] = trans_rate_evaluate
                switch_ratio_alg[i][j] = switch_ratio_evaluate
                power_alg[i][j] = power_evaluate


    data = {
        'success_ratios_alg': success_ratios_alg,
        'trans_rates_alg': trans_rates_alg,
        'switch_ratio_alg': switch_ratio_alg,
        'power_alg': power_alg
    }
    data_save(data)


if __name__ == '__main__':

    args = get_common_args()
    args = get_mixer_args(args)
    args = get_env_args(args)
    episode_reward, episode_BLER, episode_trans_rate, episode_switch_ratio, episode_trans_power = [], [], [], [], []

    for i in range(1):
        if not args.evaluate:
            env = com_env(args)
            runner = Runner(env, args)
            runner.run()
            reward, BLER, trans_rate, sinr_db, _, _, _ = runner.evaluate()
            episode_reward.append(reward)
            episode_BLER.append(BLER)
            episode_trans_rate.append(trans_rate)
            runner.agents.policy.save_model(i)

        else:
            # test_capacity(5, ['trace_jam', 'trace_jam'], 170, 260, 10, 5)
            test_capacity_different_type()
