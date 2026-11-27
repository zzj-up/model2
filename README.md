# Model 2

Model 2 用两条独立的 2 层和 4 层几何注意力分支，编码独立的配体构象和受体口袋。扩散模块对配体重原子到口袋锚点的距离矩阵加高斯噪声，由网络学习按时间步预测噪声。反向采样输出单位为 Å 的距离矩阵。距离导出可用于后续坐标拟合；本版本的输出是距离矩阵与靶点分数。

共享的几何编码器还有一个单独的评分头。训练用亲和力标签监督该头，推理只输出无量纲 `score`，对同一个化合物的靶点按分数降序排列。每个靶点的分数为各簇代表构象分数的平均值。

## 公开权重

PDBbind 重训已完成 50 轮。默认推理加载第 33 轮的最佳验证权重 [best_validation.pt](node_rank/runs/pdbbind2020_distance_diffusion/best_validation.pt)。第 50 轮的 [last.pt](node_rank/runs/pdbbind2020_distance_diffusion/last.pt) 可与原训练配置和数据缓存配合续训。[训练指标](node_rank/runs/pdbbind2020_distance_diffusion/metrics.json) 和 [训练汇总](node_rank/runs/pdbbind2020_distance_diffusion/training_summary.json) 记录了这次训练的数据划分和检查点。训练数据、特征缓存和本地评估保存在原工作区。

## PDBbind 重训

在 WSL 的 `model2` 目录运行：

```bash
DISTANCE_RUN_DIR=node_rank/runs/pdbbind2020_distance_diffusion_new \
  bash run_distance_pdbbind.sh --epochs 50 --device cuda
bash dti.sh status node_rank/runs/pdbbind2020_distance_diffusion_new/training_status.json
```

默认沿用父项目中按化合物分组的锁定划分。1,836 个复合物用于训练，395 个用于验证，392 个留出。预处理记录每个集合的通过和失败数。PDBbind 数据在本地，`PDBBIND_SPLITS_DIR` 可指向原工作区的 `splits_v2`。训练使用独立生成的 ETKDGv3 配体构象。受体结构先独立生成候选口袋，再用晶体标签确定哪一个候选对应已知结合位置。模型使用该口袋和独立构象作为条件，晶体配体坐标用于生成距离监督。此设置评估的是指定口袋内的距离预测。

`DTI_VENV` 可覆盖默认的 WSL Python 环境 `~/.venvs/dti`。`DISTANCE_CACHE_DIR`、`DISTANCE_RUN_DIR` 和 `DISTANCE_PREP_WORKERS` 分别配置缓存、运行目录和数据处理进程数。重新训练时使用尚未创建的运行目录。中断后保留对应目录内的 `run_config.json` 和 `last.pt`，用相同的运行目录及参数加 `--resume` 继续。

## 给同一个化合物的候选靶点排序

```bash
bash dti.sh rank-from-smiles \
  --smiles '[NH3+][C@@H](CCC(=O)O)C(=O)O' \
  --targets path/to/targets.json \
  --output-dir predictions/glutamate_targets --device cuda
```

靶点清单是包含至少两条不同蛋白序列的 JSON 数组。每条记录需提供 `id`、`protein_sequence`、`protein_pdb`、`box_center` 和 `box_size`，可选 `target_name` 与 `protein_chain`。口袋盒使用受体坐标系，单位为 Å。结构路径相对于清单解析。

一次筛选为化合物生成一套 ETKDGv3 构象，按对齐后且考虑原子对称性的重原子 RMSD 聚类。每簇保留一个真实构象，供所有靶点复用。`ranking/scores.csv` 输出每个靶点的分数和名次。加 `--export-distances` 可为每个靶点的最高分簇导出距离矩阵和相应的口袋锚点坐标。

仅需构象库时可运行 `bash dti.sh conformer-library --smiles '<SMILES>' --output-dir predictions/conformers`。默认尝试 32 个构象，簇内 RMSD 阈值为 0.75 Å。
