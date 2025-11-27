# MARL_QMIX 快速指南：替代干扰模型与干扰统计日志

本项目在原协同通信仿真基础上，加入了可切换的“替代干扰模型”和干扰统计特征导出，并在采样过程中于帧边界打印 Top-K 干扰最低子载波（基于 EWMA）。本文档给出最小可用的运行方式与参数说明。

## 快速开始
- 训练并自动周期评估：
  - `python main.py`
- 固定步数训练（示例 50 万步，结束后做一次评估）：
  

两者都会使用 `common/arguments.py` 中的默认参数。你可以在代码中覆盖这些参数，或直接修改默认值。

## 关键功能
- 替代干扰模型（默认启用）：
  - 通过 `args.use_alt_jam_model=True` 切换至替代模型；该模型在每个通道上独立按伯努利采样占用，并随机幅度（受 `jam_power` 约束）。
  - 刷新周期由 `args.alt_jam_change_time` 控制（单位 ms，默认 5，与 `slot_ms=1` 对齐）。
- 干扰统计特征导出：
  - 每步提供 `moving_avg`（滑动窗口平均）与 `ewma`（指数加权移动平均），维度分别为 `channel_num`。
  - 在采样中以 `episode['interf_stats']` 返回为 `[T, 2 * channel_num]`，按 `[moving_avg, ewma]` 拼接。
- 帧边界 Top-K 干扰最低子载波日志：
  - 在采样中，当 `step % frame_slots == 0` 时打印基于 `ewma` 的低干扰子载波 Top-K：
    - 示例输出：`[Frame 3] Top-5 clean subcarriers (ewma): indices=[2, 7, 10, 15, 19], values=[0.012, 0.017, 0.021, 0.025, 0.031]`

## 参数位置与说明
- 文件：`common/arguments.py`，函数：`get_env_args(args)`
- 主要参数：
  - `use_alt_jam_model`：是否启用替代干扰模型（默认 `True`）。
  - `alt_jam_change_time`：替代模型的更新周期（ms，默认 `5`）。
  - `alt_jam_occ_prob`：每通道被干扰的概率（默认 `0.2`）。
  - `slot_ms`：时隙时长（默认 `1` ms）。
  - `frame_slots`：帧长度（以时隙数计，默认 `20`）。
  - `log_topk_clean_channels`：是否打印 Top-K 干扰最低子载波日志（默认 `True`）。
  - `topk_clean_K`：Top-K 的 K 值（默认 `5`）。
  - `obs_lag_window_L`：用于 `moving_avg` 的滑动窗口长度（默认 `20`）。
  - `ewma_beta`：EWMA 的平滑系数（默认 `0.7`，建议 `0.7~0.9`）。

## 如何覆盖参数
- 在脚本中覆盖：
  ```python
  from common.arguments import get_common_args, get_mixer_args, get_env_args
  from env import com_env
  from runner import Runner

  args = get_common_args()
  args = get_mixer_args(args)
  args = get_env_args(args)
  # 覆盖参数示例
  args.use_alt_jam_model = False          # 关闭替代干扰模型做对比
  args.alt_jam_occ_prob = 0.3             # 增大占用概率
  args.frame_slots = 10                   # 缩短帧长度
  args.log_topk_clean_channels = True     # 打开 Top-K 日志

  env = com_env(args)
  runner = Runner(env, args)
  runner.run()
  ```
- 或直接修改 `common/arguments.py` 中默认值。

## 运行时日志
- 训练循环会周期性打印评估指标（Reward/BLER/Transmit_rate/...）。
- 若开启 Top-K 日志（默认开启），在每个帧边界打印低干扰子载波索引与对应 `ewma` 值，便于观察与调参。

## 验证要点
- 模型切换互斥：`env.py` 中对原 `add_jam` 与替代模型的调用互斥，不会重复更新。
- 统计维度：`episode['interf_stats']` 形状为 `[T, 2 * channel_num]`，与 `obs/actions` 步数一致；填充步数按零向量补齐维度。
- 期望行为：随 `alt_jam_occ_prob` 增大，`ewma` 整体上行；Top-K 索引更偏向低 `ewma` 的子载波。

## 验证流程建议
- 单步健全性检查：设置较小训练步数和评估轮次，确认训练与评估能跑通、指标正常打印。
  - 示例：在脚本中覆盖 `args.n_steps=20000`、`args.evaluate_epoch=3`。
- Top-K 清洁子载波检查：保持 `log_topk_clean_channels=True`，观察在帧边界的 Top-K 日志是否随 `alt_jam_occ_prob` 增大而变化。
- 模型切换对比：分别运行 `use_alt_jam_model=True/False` 两组实验，比较 Reward/BLER/Transmit_rate 的趋势。
- 参数敏感性：对 `alt_jam_occ_prob`（如 0.1/0.2/0.3）与 `alt_jam_change_time`（如 5/10）做组合试验，验证统计与性能的稳定性。

## 批量运行示例
- 使用脚本自动扫参并汇总指标：
  - `python MARL_QMIX\scripts\sweep_alt_jam.py`
- 脚本要点：
  - 默认对 `occ_prob=[0.10,0.20,0.30]` 与 `change_time=[5,10]` 的笛卡尔积进行训练与评估。
  - 每次运行结束打印一行汇总：`Reward/BLER/TransRate/SwitchRatio/TxPower/Collision`。
  - 可在脚本中调整 `n_steps` 与 `evaluate_epoch` 的值以缩短验证时间。

## HRL 掩码使用说明
- 开关与模式：
  - `hrl_enable`：是否启用分层强化学习的上层掩码（默认 `False`）。
  - `hrl_masking_mode`：掩码模式，当前支持 `none | fixed_channel_groups | prefer_clean_topk（已弃用）`（默认 `none`）。
  - `hrl_topk_K`：已弃用；`prefer_clean_topk` 掩码已禁用，参数不生效。
  - `hrl_meta_period`：元决策刷新周期（用于固定分组的轮转协调参考周期）。
- 弃用说明（prefer_clean_topk）：
  - 基于 EWMA 的 Top-K“偏好干净信道”掩码在大规模集群组网下无法保证协调性，现已禁用；当设置为该模式时控制器直接返回 `None`（不应用掩码）。
  - 仍保留帧边界的 Top-K“清洁子载波”日志以便观察环境干扰统计，但不参与动作约束。

- 固定分组（协同避碰）模式（推荐）：
  - 将 `hrl_masking_mode` 设为 `fixed_channel_groups`，为每个智能体限定一个预定义的信道集合，避免同簇内碰撞，并可选启用时间轮转降低长期碰撞概率。
  - 参数：
    - `hrl_channel_groups_json`：JSON 字符串，信道分组列表，如 `[[0,1,2],[3,4,5]]`。
    - `hrl_group_assign_mode`：`mod`（默认，按智能体编号取模分组）或 `custom`。
    - `hrl_group_assignments_json`：当为 `custom` 时提供智能体到分组的映射，如 `[0,0,1,1]`。
    - `hrl_group_rotate`：是否启用固定分组的轮转协调（默认 `False`）。
    - `hrl_group_rotate_period`：轮转周期（步），如 `50` 表示每 50 步对分组偏移一次。
  - 自动分组（默认）：
    - 原理与原因：环境将可用频带按分组暴露，`no_mcs` 变体使用固定的每 5 信道一组的索引与观察切片（如 `last_actions//5` 与 `obs[..., avail_band*5:avail_band*5+5]`），而带 MCS 的 `iql/qmix` 则按 `channel_num/sub_band` 等分。因此自动分组优先在 `iql_no_mcs/qmix_no_mcs` 下采用“每 5 个信道一组”，否则在 `channel_num % sub_band == 0` 时按子带均分。
    - 使用方法：不提供 `hrl_channel_groups_json` 时，系统会按上述规则自动生成连续分组；若既不能按子带整除、又不是 5 的倍数，则需要显式提供 `hrl_channel_groups_json`。
    - 推荐配置：
      - `no_mcs` 场景：将 `channel_num` 设为 `5*N`，如 `40`；自动分组将生成 `N` 组，每组 5 信道。
      - 带 MCS 场景：确保 `channel_num % sub_band == 0`，如 `channel_num=48, sub_band=8 → 每组 6 信道`。
    - 排错与验证：
      - 运行时可打印 `hrl_controller.channel_groups` 检查自动分组是否符合预期。
      - 若掩码叠加后无可行动作，控制器会自动回退不应用掩码；此时应检查 `channel_num/sub_band/分组` 与环境 `avail_actions` 的带宽限制是否相符。
    - 示例：`channel_num=40, env.sub_band=8` → 自动生成 8 组，每组 5 个信道：`[[0,1,2,3,4],[5,6,7,8,9],...,[35,36,37,38,39]]`。
  - 轮转协调（可选）：
    - 当 `hrl_group_rotate=True` 时，以 `group_rotate_period` 为周期做分组偏移：`g = (agent_id % n_groups + floor(call_count / period)) % n_groups`，实现时间维度上的分簇轮换。
    - 适用于静态分组场景降低长期碰撞风险；如需精细控制，可保持 `hrl_group_rotate=False` 并使用 `custom` 映射。
  - 启用示例：
    ```python
    args.hrl_enable = True
    args.hrl_masking_mode = 'fixed_channel_groups'
    # 自动分组（推荐）：不设 hrl_channel_groups_json，由控制器按环境规则生成
    # 显式分组示例：
    # args.hrl_channel_groups_json = '[[0,1,2],[3,4,5]]'
    args.hrl_group_assign_mode = 'mod'   # 或 'custom' + assignments
    # args.hrl_group_assignments_json = '[0,0,1,1]'
    # 可选轮转协调：
    args.hrl_group_rotate = True
    args.hrl_group_rotate_period = 50
    env = com_env(args); runner = Runner(env, args); runner.run()
    ```
  - 机制：根据智能体编号映射到其分组，掩码仅允许该分组内的信道动作；若与 `avail_actions` 叠加后无可行动作，则自动回退不应用掩码。
  - 动作映射说明：动作按 `channel * (len(RA_action) * len(Pt_action)) + mcs * len(Pt_action) + pt` 枚举；掩码仅基于 `channel` 分量约束，其他分量不变。
  - mod 映射说明：`group = agent_id % len(channel_groups)`，智能体按编号循环分配到各分组。
    - 例如：`channel_groups = [[0,1,2],[3,4,5]]` 且 `n_agents=6` 时：
      - agent 0→group 0；agent 1→group 1；agent 2→group 0；agent 3→group 1；agent 4→group 0；agent 5→group 1。
    - 适合“静态分簇”的协同避碰；若需精细控制，请使用 `custom` 并提供 `hrl_group_assignments_json`。

## 常见问题
- 未见 Top-K 日志：确认 `log_topk_clean_channels=True` 且 `frame_slots` 设置合理；日志位于采样循环的帧边界。
- 切换至原干扰模型：将 `use_alt_jam_model=False`，替代模型关闭后沿用原 `jam_type` 与 `jam_change_time` 逻辑。

## 参考文件
- `env.py`：替代干扰模型实现与干扰统计导出。
- `common/rollout.py`：采样循环中统计填充与 Top-K 日志。
- `DEVLOG.md`：开发日志，包含参数更新、实现细节与验证建议。

## 上层图策略（UpperGraphAC）
- 概述：在固定分组掩码模式下，自动启用基于 G2ANet 的上层 Actor-Critic（UpperGraphAC），周期性（`hrl_meta_period`）在边界处做分组决策并同步更新，决策通过 `HRLController.apply_upper_group_decision` 应用于动作掩码以实现协同避碰。
- 开启条件：`hrl_enable=True` 且 `hrl_masking_mode='fixed_channel_groups'`，并且存在 `models/hierarchical/upper_graph_ac.py`；Runner 会自动初始化并在采样循环中调用。
- 关键参数：
  - `hrl_meta_period`：元决策周期（步），在此周期边界处做上层决策与同步更新。
  - `hrl_upper_lr`：上层 AC 的学习率（默认 `2e-4`）。
  - `hrl_upper_entropy`：策略熵系数（默认 `0.01`）。
  - `hrl_upper_beta`：优势融合系数，平衡全局/局部价值（默认 `0.7`）。
  - `rnn_hidden_dim`、`attention_dim`：G2ANet 骨干的隐藏维度与注意力维度（默认 `128`/`64`）。
- 最小示例（评估模式）：
  ```python
  from common.arguments import get_common_args, get_mixer_args, get_env_args
  from env import com_env
  from runner import Runner

  args = get_common_args(); args = get_mixer_args(args); args = get_env_args(args)
  args.alg = 'qmix_no_mcs'
  args.hrl_enable = True
  args.hrl_masking_mode = 'fixed_channel_groups'
  args.hrl_group_assign_mode = 'mod'
  args.hrl_meta_period = 5
  args.evaluate = True
  args.evaluate_epoch = 1
  env = com_env(args); runner = Runner(env, args)
  print(runner.evaluate())
  ```
- 冒烟测试：
  - 运行 `python MARL_QMIX\tests\upper_ac_smoke_test.py`
  - 覆盖项包括：前向分组与控制器集成、单周期同步更新、Runner.evaluate 集成于元周期边界的决策与更新。### 自适应分簇 + 簇头 GAT
- `use_graph_obs`：开启后环境会在 `env.get_graph_snapshot()` 中输出 UAV 位置与平滑邻接矩阵，上层控制器和 GAT Actor 会复用这些信息。
- `hrl_cluster_enable`：启动自适应簇头调度，`hrl_cluster_target_size / hrl_cluster_radius_km / hrl_cluster_refresh` 控制簇大小、聚合半径与刷新周期。簇划分会映射为 `fixed_channel_groups` 掩码，并在 TensorBoard/CSV 里新增 `Cluster/*` 指标。
- 运行过程中可在日志中看到 `Cluster-Train` 和 `Cluster-Eval` 行，分别代表训练期间的平均簇数、切换率以及评估阶段的簇稳定性；所有指标也会自动写入 `result/eval_csv*/metrics.csv` 和 TensorBoard。
