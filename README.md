# model2

`model2` 在 `model1` 的基础上增加了独立配体构象库和基于 SMILES 的靶点排序入口。

## Git 仓库内容与运行环境

仓库包含模型代码、最佳评分权重、续训权重、初始化权重、训练指标和架构图。PBCNet2.0 的固定版本及上游来源记录位于 [`third_party/PBCNet2.0`](third_party/PBCNet2.0)。[PBCNet2.0 上游仓库](https://github.com/YuJie-0202/PBCNet2.0)

PDBbind 数据、特征缓存和本地评估保存在原工作区。下面引用 `evaluation/` 的命令用于带有这些本地文件的工作区。新克隆可直接以自己的 SMILES 生成构象，提供受体结构和口袋清单后可运行靶点排序。原 PDBbind 筛选面板依赖原项目的锁定划分和结构目录。

`DTI_VENV` 和 `PBCNET_VENV` 分别指定 WSL 中的模型运行环境和 PBCNet 兼容环境。`rank-from-smiles` 可通过 `--docking-python` 指定安装了 RDKit、Meeko 和 Vina 的 Python，并通过 `--esm-model` 指定本地 ESM2 权重目录。新靶点的 ESM2 特征提取使用 CUDA。

## 生成构象库

在本目录的 WSL 中运行：

```bash
bash dti.sh conformer-library \
  --smiles '[NH3+][C@@H](CCC(=O)O)C(=O)O' \
  --output-dir predictions/glutamate_conformers \
  --num-conformers 32 --rmsd-threshold 0.75 --seed 20261125
```

ETKDGv3 默认尝试生成 32 个构象。使用对齐后、考虑原子对称性的重原子 RMSD 做完全链接聚类，默认簇内任意两个构象的距离都不超过 0.75 Å。每簇选取与其他成员 RMSD 之和最小的真实成员，按原始坐标写入 `representatives.sdf`。`conformer_manifest.json` 记录种子、RDKit 版本、簇成员及代表编号。SMILES 应给定单一分子的立体化学和所需的质子化状态。

## 用同一批代表构象给多个靶点排序

```bash
bash dti.sh rank-from-smiles \
  --smiles '[NH3+][C@@H](CCC(=O)O)C(=O)O' \
  --targets evaluation/conformer_cluster_demo/targets.json \
  --output-dir predictions/glutamate_targets --exhaustiveness 2 --device cpu
```

靶点清单是 JSON 数组，至少提供两个不同的蛋白序列。每条记录需要 `id`、`protein_sequence`、`protein_pdb`、`box_center` 和 `box_size`，可选 `target_name`、`protein_chain` 和 `receptor_pdbqt`。搜索盒使用受体坐标系的 XYZ，单位为 Å。结构相对路径从清单所在目录解析。受体准备遇到组氨酸质子化歧义时，可提供已在同一坐标系下准备好的 PDBQT。命令默认使用 `~/.venvs/docking_pb` 内的 Meeko 和 Vina，未缓存的蛋白序列仍由现有 CUDA ESM2 流程提取特征。

每次调用只生成一份构象库。每个受体复用该库，在独立搜索盒内只做刚性旋转和平移。Vina 在每个靶点内选出一个姿势供现有模型评分，`ranking/scores.csv` 按原始 `score` 降序列出同一化合物的候选靶点。`run.json` 记录姿势来源的构象簇及各靶点的处理状态，`poses/` 保存生成的姿势和日志。至少两个靶点成功放置后才进入多靶点排序。

示例结果见 [`validation/ranking/scores.csv`](evaluation/conformer_cluster_demo/validation/ranking/scores.csv)。4io2 和 5ogb 成功打分，2a5s 和 3fuz 因原始受体结构存在质子化歧义而跳过。示例用于验证流程是否能运行，排序性能需要另行评估。

## 已有复合物的评分和训练权重

默认评分权重为 `node_rank/runs/pdbbind2020_refined_20261118/model/best_validation.pt`（第 13 轮，2,990 次更新）。同目录有第 50 轮的续训权重 `last.pt`、训练配置及指标，`node_rank/checkpoints` 保留初始化。PBCNet2.0 的冻结权重和源码也在本副本内。

在本目录的 WSL 中运行 `bash dti.sh status` 可加载本地默认权重。已验证的候选数据位于 `evaluation/model3_target_ranking_20261118/`，包含同一化合物在 12 个靶点上的结构与模型特征，可用以下命令输出原始分数和名次。

```bash
bash dti.sh rank-predict --features evaluation/model3_target_ranking_20261118/candidate_features.json --output-dir predictions/example --device cpu
```

`verification/ranking/scores.csv` 是备份时的核验结果，`verification/backup_validation.json` 记录与原模型对照的结论。`SNAPSHOT_SHA256.json` 记录加入构象流程前的文件校验值。

本例打分使用副本内的特征。给新分子提取特征仍沿用本机 WSL Python 环境与 `/home/dti/models/esm2_3b`。`provenance/` 内的训练清单和索引保留原始字节，完整 PDBbind 结构与训练缓存继续位于原项目 `inputs/`。发起新训练时，可显式引用原始训练和验证索引，并将输出写到本副本的新运行目录。归档里的旧训练配置记录的是原路径。

模型加载器校验架构源码的哈希。调整 `model.py` 或 `graph.py` 后，需使用与新结构对应的权重。老权重保留在这里作为比较基准。
