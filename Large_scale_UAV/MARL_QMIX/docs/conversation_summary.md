# MARL_QMIX 项目阶段性总结（科研论文体裁）

## 1. 引言  
为应对动态大规模机载网络中的自适应抗干扰需求，本阶段围绕 MARL_QMIX 平台构建了“上层自适应分簇 + 簇头 GAT 协同 / 下层 QMIX 控制”的层次化体系。目标是在保持原有多智能体协同优势的基础上，引入可扩展的图信息建模与簇级资源调控机制。

## 2. 系统设计  
1. **分层结构**：上层集中式 GAT 负责簇头决策，善用 `use_graph_obs` 提供的邻接信息；下层继续采用 QMIX 聚合策略，确保动作值分解与协同训练能力。  
2. **环境与控制器**：`env.py` 暴露标准化 `graph_snapshot` 接口，`HRLController` 内嵌 `AdaptiveClusterManager`，并在经验池中同步 `cluster_assign/cluster_count`，形成“状态+簇信息”的混合采样。  
3. **上层 AC 模型**：`models/hierarchical/upper_graph_ac.py` 引入簇感知邻接矩阵，缓存 `last_cluster_meta`，便于 Runner、TensorBoard 输出簇指标。

## 3. 训练与监测  
1. Runner 在训练/评估阶段记录 Reward、BLER、SINR、Switch_ratio、Transmit_power、Collision 等核心指标，并新增 `Cluster-Train/Eval` 曲线，将数据同时写入 TensorBoard 与 `result/eval_csv*/metrics.csv`。  
2. 清理冗余脚本与 `env_backup.py`，保留论文/数据集/日志，实现可控的版本管理；仓库已初始化并纳入 `.gitignore` 约束。  
3. 在 `intelligent_AJ` 环境运行时统一设置 `PYTHONIOENCODING=utf-8`，通过 `scripts/start_train_with_tb.py` 便捷启动训练与 TensorBoard（默认 http://localhost:6006）。

## 4. 2025-11-21 调试成果  
1. **自适应分簇升级**：`AdaptiveClusterManager` 改为“连通分量 + 目标簇大小”策略，自动计算簇数量，解决长期固定为 3 的问题。  
2. **调参接口**：`scripts/train_hierarchical.py` 支持 `MQ_EPSILON/MQ_MIN_EPS/MQ_EPS_ANNEAL_STEPS/MQ_BATCH/MQ_BUFFER/MQ_CLUSTER_REFRESH/MQ_CLUSTER_DEBUG/MQ_ALT_*` 等环境变量，实现探索率、缓冲区、簇刷新周期、干扰模式与调试输出的快速切换。  
3. **指标验证**：TensorBoard 的 `Cluster/train_step_cluster_count_meta` 已出现 5、6 等动态值，控制台 `ClusterDebug` 信息也展示不同簇大小/成员组合，证实簇划分随拓扑实时变化。  
4. **实验可重复性**：通过环境变量覆写与标准化日志体系，可在无需修改源代码的情况下快速复现实验配置，便于后续论文撰写、消融研究和长周期训练。

## 5. 结论  
工程升级已完成从环境建模、HRL 控制器到训练监测的全链路改造，形成可扩展的分层抗干扰训练基线。后续可直接在该版本上开展长时训练、干扰场景对比实验以及论文数据收集。***

## 6. 拓扑可控与审计
为便于论文实验锁定一致的通信拓扑，新增了 `topology_dataset_lock_sample` 与 `topology_dataset_sample` 参数，可在训练期间固定某个 `sample_xxx`；脚本 `scripts/eval_topologies.py` 能遍历数据集、输出连通性与度分布统计，为拓扑筛选和复现实验提供客观依据。
