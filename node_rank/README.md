# 当前模型：逐节点双视图 + 几何 BAN 靶点排序

2026-11-17 按用户提供的最终架构图部署。初始化检查点为 `node_rank/checkpoints/untrained_node_rank.pt`，固定保留为 **0 训练步**。官方 ESM2 和 PBCNet2.0 保留预训练并冻结；新投影、注意力传播、双视图融合、BAN 和输出层从该初始化开始训练。

这是按照所选架构编写的新任务网络。BAN 采用低秩双线性注意力思想并加入几何约束，属于本项目实现，不是直接下载的官方 BAN/DrugBAN 评分头，也不是 PBCNet 原来的相对亲和力输出头。

## 2026-11-18 PDBbind 重训

已完成本地 PDBbind 2020 refined 的 50 轮重训，共 11500 次参数更新。任务目录为 `node_rank/runs/pdbbind2020_refined_20261118/`；2629 个候选中 2623 个通过 PBC 特征质控，按化合物分组划分为 1836 训练、395 验证、392 保留测试。最佳验证模型为 `model/best_validation.pt`，来自第 13 轮。保留集中的部分复合物已经过测试；完整保留集尚未评估。

`python -m node_rank.train_pdbbind` 使用标准化实验亲和力监督无量纲 score，并只按验证误差选最佳权重。`rank-predict` 现在默认使用该已训练模型。该权重并没有使用下文 `rank-train` 的同化合物成对偏好损失。训练及恢复见[任务说明](runs/pdbbind2020_refined_20261118/README.md)。

## 当前默认用法：对一个化合物排多个靶点

每个候选靶点提供一份与其受体 PDB 对齐的配体姿势 SDF 和对应蛋白序列。清单中每一行代表同一化合物与一个不同靶点；用 `target_name` 可为序列指定易读名称。没有实验标签也可以排名。参见[已运行的真实输入清单](../evaluation/model3_target_ranking_20261118/structure_manifest.json)。

```bash
bash dti.sh rank-candidate-features --manifest inputs/my_candidates/structure_manifest.json --output inputs/my_candidates/candidate_features.json
bash dti.sh rank-predict --features inputs/my_candidates/candidate_features.json --output-dir node_rank/predictions/my_candidates --device cpu
```

第一步使用与当前检查点相同的冻结特征策略；新蛋白序列需要本地 CUDA ESM 环境。第二步可在 CPU 或 GPU 运行，输出 `scores.csv`，包含 `compound_id`、`target_id`、可选的 `target_name`、`pair_id`、`rank`、`score`、`candidate_count` 和 `status`。每个化合物单独按原始分数从高到低排列，完全相同的分数并列（例如 1、1、3）。不足两个不同靶点、同一配对重复提供姿势或策略不匹配时会报错。目标身份按精确蛋白序列区分，若多个构建体代表同一生物学靶点，应先选一个代表。训练、评估中使用的实验标签不作为对外预测值。`predict` 和 `evaluate-pdbbind` 是当前 `rank-predict` 的别名。可直接查看[谷氨酸的 12 靶点排名](../evaluation/model3_target_ranking_20261118/ranking/ranked_targets.csv)。

## 图中模块与实际代码

| 模块 | 本地实现和作用 |
|---|---|
| ESM2 序列 S | 官方 ESM2 3B 每残基 2560 维；510 残基窗口覆盖全序列、保留各行，去掉特殊标记。长序列不同窗口间没有 ESM 上下文，后续序列邻接连接窗口边界。 |
| PBCNet 小分子 L、口袋 P | 官方冻结 TensorNet 对一个有结合姿势的复合物编码，分别保留配体和口袋每个原子的 96 维输出；在原来的配体求和读出之前截取。 |
| 特征投影与构图 | 三种节点各投影为 128 维；序列相邻边、分子化学键、原子空间邻近边、配体—口袋接触边，以及口袋原子到序列残基的映射边。 |
| 2 层／4 层注意力传播 | 两个独立分支读取同一初始节点图。距离径向基编码、边类型、正值可学习接触偏置参与注意力；接触边与其他入边共同竞争权重。 |
| 双视图融合 | 每种节点类型各有一组可学习的两视图 softmax 权重；逐节点融合，保留 S′、L′、P′。这是深度视图融合，不是全局／局部评分门控。 |
| 全局读出 | 分别平均 S′、L′、P′，拼接并投影为 128 维 `z_global`，避免节点数较多的类型仅靠数量占优。 |
| 几何 BAN | 只在距离不超过 5 Å 的 L′×P′ 原子对上计算低秩双线性注意力；4 个注意力分支、每个低秩维度 32，加入距离编码，汇总为 128 维 `z_BAN`。掩码外原子对注意力为零。 |
| 最终评分 | `[z_global; z_BAN]` 为 256 维，经过**一个 Linear(256,1)** 输出 score。没有最终 MLP、sigmoid 或 Adaptive Gating。 |
| 排序 | 同一化合物的候选靶点按 score 降序排列，输出 score、rank、candidate_count。并列分数使用并列名次；不同化合物分别排序。 |

默认新增任务参数共 **1,365,983**。官方编码器参数不包含在任务检查点中。内部信息传播包含前馈变换；“单层线性”特指最后从拼接特征到 score 的输出层。

原子局部空间邻接默认 4.5 Å，跨分子接触默认 5 Å，完整残基口袋裁剪半径 8 Å。序列节点不伪造三维坐标。后续几何模块使用距离，因此对整个复合物共同旋转、平移保持不变。它不是构象生成器，也不优化结合姿势。

## 输入和输出

每个化合物—靶点配对必须提供：

- 配体在该蛋白口袋中的三维结合姿势 SDF，以及相同分子的带立体化学 SMILES。
- 与配体处于同一坐标系的受体 PDB；必要时明确 `protein_chain`。
- 该靶点蛋白序列。清单直接使用 `protein_sequence` 字符串；当前入口不单独解析 FASTA。
- 唯一的 `sample_id` 或 `pair_id`。

仅有游离配体 SDF 或蛋白序列，不能产生这里需要的三维接触关系。一个化合物预测多个靶点时，每个靶点都需要各自的受体和对应结合姿势；不能把第一个靶点的坐标直接用于其他蛋白。

清单为 JSON 数组，格式与既有 `evaluation/pdbbind_unseen5_20261116/dataset/structure_manifest.json` 一致。SDF/PDB 相对路径以清单所在目录为基准。只接受每个化合物—靶点一个明确选定的姿势，不自动选择最高分姿势。化合物身份按规范化、保留立体化学和电荷的 SMILES 分组，靶点身份按序列 SHA256 分组；因此需要统一化合物质子化和盐形式，避免同一研究对象被拆成多个组。

结构与序列映射使用原始蛋白链的序列比对、PDB 链/残基编号/插入码和逐原子坐标检查；多解比对或无法匹配的蛋白残基被拒绝。只有明确允许保留的非蛋白异质原子可标记为 -1，不构造虚假的序列映射边。

**score 是模型学到的相对排序值，不是亲和力单位，也不是结合或治疗概率。** 随机任务参数产生的 score 没有筛选意义；正式预测默认拒绝未训练检查点。注意力权重同样不是物理相互作用能量或因果证据。

## WSL 使用

在 WSL 项目目录运行：

```bash
cd /mnt/c/baidunetdiskdownload/3
bash dti.sh status
bash dti.sh verify --device cuda
bash dti.sh rank-verify-workflow
```

重新初始化时选择新的文件名，原文件不会覆盖：

```bash
bash dti.sh init --output node_rank/checkpoints/another_initialization.pt --seed 20261118
```

下列旧格式特征包用于显式的成对训练流程；当前默认权重的候选靶点请使用上方的 `rank-candidate-features` 分片索引：

```bash
bash dti.sh rank-features \
  --manifest inputs/my_complexes/structure_manifest.json \
  --output inputs/my_complexes/node_features.pt
```

提取器顺序运行 PBCNet 兼容环境与 ESM2 环境，释放前一进程后再加载后一模型，适配当前 8 GB 显存。`.parts` 目录保留可核验的阶段缓存；中断后同一输入、代码和配置可重新运行。已完成的输出文件不会覆盖。原来的平均 ESM 和汇总 PBC 缓存不能转换为这些逐节点特征。

## 训练排序模型需要什么标签

训练器使用成对排序损失 `softplus(-(score_better - score_worse))`。训练样本只在**同一个化合物、同一 comparison_group、同一 endpoint**内比较不同靶点。每组至少两条不相同的可比标签；数值越大表示越应该排在前面。不把数据库缺失关系当作负样本。

标签文件为 CSV，列为：

```text
pair_id,relevance,comparison_group,endpoint
```

`pair_id` 必须逐条对应特征清单，`relevance` 是有限数值，`comparison_group` 由研究者明确确认可比较的实验条件，`endpoint` 表示标签类型。pKd、pKi 等只有在数据可比时才能用于形成排序偏好；同端点名称本身不能保证实验条件可比。所有特征样本必须有标签，训练/验证/测试材料应在进入这个命令之前明确分开。当前训练入口不自动划分、调参或做性能验证，也不自动把亲和力数值作为本模型输出。

```bash
bash dti.sh rank-train \
  --features inputs/my_training/node_features.pt \
  --labels inputs/my_training/ranking_labels.csv \
  --dry-run

bash dti.sh rank-train \
  --features inputs/my_training/node_features.pt \
  --labels inputs/my_training/ranking_labels.csv \
  --run-dir node_rank/runs/new_ranking_run \
  --epochs 20 --lr 0.0001 --device cuda
```

训练从该新架构的随机初始化检查点开始；所有新任务模块参与更新，ESM2/PBCNet 冻结。运行目录必须是新目录，保存训练记录和每轮 `last.pt`。当前入口不提供断点续训，避免把首轮训练与续训混淆；需要续训时应另行增加明确的优化器/随机状态恢复流程。训练误差不代表泛化表现。单个比较逐次反传，在同一比较组内累积梯度后更新，默认每组最多随机抽取 64 个比较。

现有 `selected_training_v2` 是 **1000 个不同化合物、1000 个配对、770 条蛋白序列**，每个化合物只有一个靶点，潜在的同化合物跨靶点比较数为 **0**。它可以支持其他监督目标，但不能直接用于这个跨靶点排序训练器。审计入口：

```bash
bash dti.sh rank-audit --manifest inputs/pdbbind_1000/selected_training_v2/structure_manifest.json
```

## 预测与已完成的验证

若另行完成 `rank-train` 成对训练，可显式指定其权重和配套特征包：

```bash
bash dti.sh rank-predict \
  --checkpoint node_rank/runs/new_ranking_run/last.pt \
  --features inputs/my_candidates/node_features.pt \
  --output-dir node_rank/predictions/new_candidates \
  --device cuda --save-attention
```

输出 `scores.csv`、`summary.json`；可选 `contact_attention.pt` 保存原子对、距离和注意力。检查点与特征必须具有相同的官方编码器及预处理策略。不兼容的旧架构检查点会被拒绝。

架构部署阶段曾执行模型 CPU/CUDA 数值和梯度检查、临时合成比较的单步训练与回载，以及五个真实复合物的完整特征提取和前向检查。合成训练权重位于临时目录并已清除；这些早期记录对应零训练步的初始化模型。

- `verification/model_cpu.json`、`model_cuda_final.json`：模型功能检查。
- `verification/workflow.json`：标签约束、排序、检查点、防止误用未训练结果等集成检查。
- `verification/feature_mapping_final.json`、`feature_mapping_and_parity_final.json`：结构映射及新节点特征重新聚合与旧官方输出的一致性。
- `verification/real5_features_final.pt`：真实的逐节点特征。
- `verification/real5_forward/summary.json`：新模型的完整运行检查，结果明确标记 `UNTRAINED_DIAGNOSTIC_ONLY`。

以上五个真实输入的检查属于 2026-11-17 架构部署阶段，各自仅有一个候选靶点、rank 都是 1，不能用于评价跨靶点排序能力。当前的多候选示例见[12 靶点测试说明](../evaluation/model3_target_ranking_20261118/README.md)。

## 保留的旧版本

2026-11-18 前一重训准备任务已将旧运行权重及缓存移至 `maintenance/pdbbind_retrain_20261118/retired_artifacts/`，详情见该维护目录的清单。原始结构、`fresh_dti/checkpoints/untrained_dti.pth` 和官方预训练编码器保留。旧版代码仍在 `fresh_dti/`；默认 `status/init/verify` 已指向本版本。TCMBank 隔离目录不被新流程读取。
