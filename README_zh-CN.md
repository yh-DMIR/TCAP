# TCAP

[English](README.md) | [简体中文](README_zh-CN.md)

**目标条件非对称参数投影（Target-Conditioned Asymmetric Parameter Projection）** 用于选择适合目标表格分类任务的检查点。从共享基础模型出发，对目标任务适应一次后，TCAP 根据参数位移对源任务检查点排序。默认直接使用 top-1，也可根据验证集 log-loss 从 top-k 候选中选择。

支持 **TabICL-v2** 和 **TabPFN-v3**。**Built with PriorLabs-TabPFN.**

## 安装

需要 Python 3.10+ 和 PyTorch。在仓库根目录安装所需后端：

```bash
python -m pip install ".[tabicl]"
# 或使用 TabPFN：
python -m pip install ".[tabpfn]"
```

自行准备分类 CSV 与基础检查点。后端包固定为 `tabicl==2.1.1` 和 `tabpfn==8.5.0`，请使用对应的 v2/v3 分类检查点。若后端已通过源码安装，可直接执行 `python -m pip install .` 安装 TCAP。

## 快速开始

每个任务使用一个 CSV。标签列默认为 `y`；若没有 `y`，则使用最后一列，也可用 `--label COLUMN` 指定。支持不超过 10 类的分类任务。

### 1. 适应任务

所有源任务与目标任务从同一个基础检查点出发，使用相同设置适应。以下示例使用 Bash：

```bash
BASE=/path/to/tabicl-classifier-v2-20260212.ckpt

for task in source_a source_b source_c source_d source_e target; do
  tcap adapt \
    --backend tabicl \
    --base "$BASE" \
    --data "data/${task}.csv" \
    --task-key "$task" \
    --output "artifacts/tabicl/${task}"
done
```

每个任务目录包含检查点、`delta.npy` 和 `task.json`。每次运行请使用新的输出目录。

使用 TabPFN-v3 时，所有任务改用 `--backend tabpfn` 并指定同一个 TabPFN-v3 分类检查点，再为生成的文件单独建立检查点库。

### 2. 建立检查点库

保存以下内容为 `bank.json`。相对路径以该文件所在目录为起点：

```json
[
  "artifacts/tabicl/source_a/task.json",
  "artifacts/tabicl/source_b/task.json",
  "artifacts/tabicl/source_c/task.json",
  "artifacts/tabicl/source_d/task.json",
  "artifacts/tabicl/source_e/task.json"
]
```

### 3. 预测

默认选择得分最高的外部检查点，执行一次最终预测。

```bash
tcap predict \
  --target artifacts/tabicl/target/task.json \
  --bank bank.json \
  --data data/target.csv \
  --output predictions/top1
```

可选：召回五个候选，再根据验证集 log-loss 选择。

```bash
tcap predict \
  --target artifacts/tabicl/target/task.json \
  --bank bank.json \
  --data data/target.csv \
  --rerank --top-k 5 \
  --output predictions/top5_val
```

使用与适应时相同的目标 CSV，预测其中预先留出的查询行。输出为 `selection.json` 和 `predictions.npz`；后者包含 `probabilities`、`classes`、`predictions` 及从零开始的 `row_indices`。

## 选择选项

| 选项 | 作用 |
|---|---|
| 默认 | top-1 选择，排除目标自身检查点 |
| `--rerank --top-k K` | 验证 top-k 候选，默认 `k=5` |
| `--include-self` | 直接模式允许目标自身检查点参与 |
| `--no-include-self` | 重排模式排除目标自身检查点 |
| `--mu 0.1` | 模块权重平滑系数 |
| `--tiny-energy-fallback` | 总目标模块能量不大于 `1e-12` 时使用均匀权重；默认关闭 |
| `--n-estimators 8` | 验证及最终预测的估计器数 |
| `--device cuda` | 运行设备；CPU 执行使用 `cpu` |

重排模式默认将目标自身检查点加入候选池，由 TCAP 得分决定是否进入 top-k，不强制召回。按验证 log-loss 选择，分数相同时依次比较验证准确率与 TCAP 名次。符合条件的候选数量不得少于 k。

## 方法与默认设置

任务位移定义为 `delta_i = theta_i - theta_0`。根据目标模块的均方位移计算权重，再以 `mu=0.1` 平滑。TCAP 使用加权投影评分：

```text
energy[m] = ||delta_target[m]||² / dim[m]
weight[m] = (1 - mu) * energy[m] / sum(energy) + mu / number_of_modules
score = sum_m weight[m] * <delta_target[m], delta_source[m]>
        / (sqrt(sum_m weight[m] * ||delta_target[m]||²) + 1e-12)
```

评分不使用源位移范数归一化。TabICL-v2 使用 15 个参数模块，TabPFN-v3 使用 34 个。候选必须共享相同的后端、基础检查点与适应设置。

默认对所有正的模块总能量按上述公式归一化，包括不大于 `1e-12` 的情况。总能量为零时，权重归一化无定义，因此明确报错。如需在总能量不大于 `1e-12`（包括零）时使用均匀权重，可为 `tcap predict` 添加 `--tiny-energy-fallback`。Python 接口 `target_weights`、`tcap_score`、`rank_candidates` 和 `predict_task` 均可设置 `tiny_energy_fallback=True`，默认值为 `False`。`selection.json` 记录该设置。开启回退后，零目标位移得到全零评分，按任务 key 确定性打破并列。

| 设置 | 默认值 |
|---|---|
| 分层划分 | 64% 训练 / 8% 早停 / 8% 重排 / 20% 查询 |
| 上下文 | 重排使用 72%；最终预测使用 80% |
| 优化器 | AdamW，学习率 `1e-5`，权重衰减 `0.01` |
| 训练 | 最多 30 轮，早停指标 ROC-AUC，patience 为 8 |
| 批大小 | 4；CUDA OOM 后从头以 2、1 重试 |
| 估计器视图数 | 训练及早停为 32；预测为 8 |
| 随机种子 | 42 |

四个子集互不重叠。留出查询标签仅用于分层划分，不传入检查点选择或预测。每个子集均须包含所有类别。

## 测试

```bash
python -m pip install ".[test]"
python -m pytest -q
```

## 致谢

基于 [TabICL](https://github.com/soda-inria/tabicl) 和 [TabPFN](https://github.com/PriorLabs/TabPFN) 构建。署名与上游许可证见[第三方说明](THIRD_PARTY_NOTICES.md)和 [licenses/](licenses/)。
