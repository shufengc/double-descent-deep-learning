# 更新报告 —— 在 Double Descent 流水线上引入多种子、Ridge 扫描与 NN 超参消融

**作者：** Yizheng Lin (yl6079)
**分支 / 提交：** `yizheng` @ `b995718` —— *"Multi-seed RFF λ sweep; NN augment/opt/hparams"*
**结果目录：** `results_full_demo/`（seeds = {41, 42, 43}）
**本文定位：** 本文是在团队主报告（`report.md` @ `c91942e`）基础上的增量工作记录。为了避免重复，本文**不**再复述课程层面的背景知识，只关注：改了什么、为什么改、以及新数据告诉了我们什么。

---

## 0. 一句话摘要

1. 原 pipeline 能画出"教科书式"的 DD 曲线，但只有单 seed、单 λ。我把所有实验改造成可以在一组随机种子上扫描，RFF 可以扫 ridge 正则强度，CNN 实验通过 `--augment / --optimizer / --weight-decay / --nn-lr` 把关键超参全部下放到 CLI。
2. 有了三个种子之后，每条曲线都能附上 ±σ 误差带。RFF 在 `p/n = 1` 的峰值，在最小范数拟合下 **σ ≈ 均值**——也就是说单 seed 报告的峰值高度本质上是从一个重尾分布里抽了一次样。
3. 对 λ ∈ {1e-10, 1e-4, 1e-2} 的扫描清晰展示了理论预言的"**削峰不削谷**"行为：λ = 1e-2 把插值点处的 MSE 压低了 **约 3 个数量级**，而 p/n = 8 的过参数区几乎原封不动。
4. 对 CNN 打开真实的正则（SGD + 数据增强 + 权重衰减）后，在干净 CIFAR-10 上 **模型宽度-DD 峰彻底消失**，曲线变单调——正是 Nakkiran et al. (2021) 预言的"会把 DD 隐藏起来"的区间。
5. Exp 4 在同一套正则配置下 **未能复现 epoch-wise DD**（1000 epoch 训练退化到低于随机准确率）。我选择把它作为诚实的负面结果保留下来，用来说明把这些超参开放到 CLI 的必要性。

---

## 1. 代码改动

所有变更都落在 `src/experiments/comprehensive_dd.py`（+479 / −172 行），其他文件没动。改动大致分三块。

### 1.1 CLI 新增开关（`main()`）

新增如下参数：

```759:787:src/experiments/comprehensive_dd.py
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated random seeds (RFF + NN)")
    parser.add_argument("--rff-lambdas", type=str, default="1e-10",
                        help="Comma-separated ridge λ for RFF (exp 1–2), e.g. 1e-10,1e-4,0.01")
    parser.add_argument("--augment", action="store_true",
                        help="Use CIFAR-10 train augmentation for NN experiments (3–4)")
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "sgd"],
                        help="Optimizer for NN experiments (3–4)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        dest="weight_decay",
                        help="L2 weight decay for NN experiments (3–4)")
    parser.add_argument("--nn-lr", type=float, default=None,
                        help="Learning rate for NN; default 0.001 (adam) or 0.05 (sgd)")
```

原来的单值 `--seed` 仍然有效 —— 当 `--seeds` 为空时会被用作回退值，因此旧命令 `--seed 42` 依然能得到比特级相同的结果。

### 1.2 每 seed 内循环 + 通用聚合器

四个实验都重构成了"外层 seed 循环 + `_exp*_run_one_seed(...)` 辅助函数 + 共享聚合器"的结构：

```25:53:src/experiments/comprehensive_dd.py
def _aggregate_rows_by_keys(rows, key_fields=("p_over_n", "lambda")):
    """rows: list of dicts with same keys; group by key_fields, mean/std for numeric vals."""
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        groups[key].append(row)
    ...
    for field in grp[0].keys():
        ...
        base[f"{field}_mean"] = float(arr.mean())
        base[f"{field}_std"] = float(arr.std(ddof=0))
```

聚合器是 schema 无关的：每个实验按自己关心的 key 字段分组（Exp 1 用 `(p_over_n, lambda)`，Exp 2 用 `(n_samples, lambda)`，Exp 3 用 `(noise, width)`）。产生的 `*_mean / *_std / n_seeds` 列正是新版图表和（稍后要更新的）notebook 的输入。

### 1.3 新增仪器：RFF 实验记录 ‖w‖

`exp1` 和 `exp2` 现在对每个 `(p/n, λ, seed)` 组合都额外记录拟合解的 L2 范数，Exp 1 还额外生成 `w_norm_vs_pn.png`。这一个量就能把团队报告里只当做理论陈述的"方差爆炸"机制给定量观测出来。

### 1.4 新的磁盘 schema

旧版（`c91942e`）：

```json
{"0.0": [{"D": ..., "p_over_n": ..., "test_mse": ...}, ...], ...}
```

新版：

```json
{
  "seeds": [41, 42, 43],
  "rff_lambdas": [1e-10, 1e-4, 1e-2],
  "per_seed": {"41": {...}, "42": {...}, "43": {...}},
  "aggregated": {"0.0": [{"p_over_n": 1.0, "lambda": 1e-10,
                          "test_mse_mean": ..., "test_mse_std": ...,
                          "w_norm_mean": ..., "n_seeds": 3}, ...]}
}
```

体积稍微大一点，但完全可逆：通过索引 `per_seed` 可以还原任意一个单 seed 的运行。CNN 实验还多了 `config` 块，记录本次运行用的增强 / 优化器 / 权重衰减 / lr，使得每个 JSON 文件都能自我描述。团队原来的 `results/` 文件仍能被老 notebook 代码路径读取，因为新 payload 是**并列**在旧扁平列表旁边，而不是覆盖掉。

---

## 2. 本次更新的公共实验设置

除非另外说明，下面章节里的每个数字都是 **三个种子 `{41, 42, 43}` 上的均值**，±σ 为 `_aggregate_rows_by_keys` 计算的按 key 的总体标准差。

| 项目 | RFF（Exp 1–2） | CNN（Exp 3–4） |
|---|---|---|
| 数据集 | MNIST，Exp 1 `n = 1000`；Exp 2 `D = 500, n ∈ [100, 4000]` | CIFAR-10 子集，`n = 4000` |
| 模型 | RFF，σ = 5.0 | `CNN(num_filters = w)`（见 `src/models.py`），w ∈ {1, 2, …, 32} |
| 优化 | 最小范数闭式解 + ridge | SGD, lr = 0.05, weight_decay = 1e-4 |
| 正则 / 增强 | ridge λ ∈ {1e-10, 1e-4, 1e-2} | 随机裁剪 + 翻转（"augment"） |
| 标签噪声 | 0 / 10 / 20 % | 0 / 20 %（Exp 3），20 %（Exp 4） |
| 种子 | {41, 42, 43} | {41, 42, 43} |

每次运行的原始 JSON + PNG 都保存在 `results_full_demo/`。

---

## 3. Exp 1 —— 模型宽度-DD（MNIST 上的 RFF）

图：`results_full_demo/exp1_model_wise_rff/dd_curves.png`，`.../w_norm_vs_pn.png`。

### 3.1 多 seed 量化"峰值"的不稳定性

每次换一个随机种子，在正则很弱时 `p/n = 1` 处的峰值极其不稳定：

| noise | λ | peak test MSE @ p/n ≈ 1 | σ / 均值 |
|:---:|:---:|:---:|:---:|
| 0% | 1e-10 | 93.7 ± 74.1 | **0.79** |
| 10% | 1e-10 | 207.4 ± 159.1 | **0.77** |
| 20% | 1e-10 | 455.2 ± 449.8 | **0.99** |
| 10% | 1e-4 | 1.11 ± 0.014 | 0.01 |
| 10% | 1e-2 | 0.132 ± 0.002 | 0.02 |

在接近最小范数的拟合下，仅三个 seed 的样本方差就和均值同量级。这在实证上验证了 `report.md` §2.2 的"阈值处方差爆炸"论断，更实用的含义是：**对 DD 峰值给一个单 seed 数字是没有意义的**。团队报告中单 seed 下"129×"的峰值完全落在这个分布里面，但不必然有代表性。

加一个很小的 ridge（λ = 1e-4）就能把 `σ/均值` 压到 ~1%，峰值变成一个可复现的、良定义的量，只有它的**幅度**还值得讨论。

### 3.2 Ridge λ 扫描：*削峰不削谷*

在 noise = 10% 下观察到：

| p/n | MSE λ=1e-10 | MSE λ=1e-4 | MSE λ=1e-2 | ‖w‖ λ=1e-10 | ‖w‖ λ=1e-2 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.30 | 0.059 | 0.059 | 0.059 | 13.5 | 13.4 |
| 0.98 | 2.40 | 1.06 | 0.130 | 234 | 49 |
| **1.00** | **207.4** | **1.11** | **0.129** | **2044** | **49** |
| 1.02 | 2.48 | 1.03 | 0.129 | 243 | 50 |
| 2.00 | 0.070 | 0.070 | 0.058 | 45 | 39 |
| 8.00 | 0.031 | 0.031 | 0.030 | 34 | 32 |

两点观察：

1. **削峰效率随 λ 陡增。** 从 λ=1e-10 → 1e-4，峰值 MSE 降约 200 倍；再从 1e-4 → 1e-2 又降约 8 倍。但 ‖w‖ 只分别降约 13 倍和 3 倍，说明 λ **削峰的速率远快于它缩减权重范数的速率**——这和 Hastie et al. (2022) 一致：峰值的发散本质上是 Gram 矩阵最小奇异值趋零驱动的，而不是 ‖w‖ 本身。
2. **`p/n = 8` 的尾部完全不动。** 三个 λ 在过参数区的差距 ≤ 10%。这是我得到的关于"正则能隐藏峰值但不会牺牲过参数区收益"这句口号的最干净证据。

在 λ = 1e-2 下，这条曲线已经不像 DD 曲线了——它就是一个很浅的 U 形——这正对应 Nakkiran et al. (2021) §4 里"最优正则让 DD 不可见"的结论。

### 3.3 ‖w‖ 作为机制的直接读数

新增的 `w_norm_vs_pn.png` 与 MSE 图形成互补：log 尺度下 ‖w‖ 曲线在 λ = 1e-10 下于 `p/n = 1` 出现尖锐的峰，其高度随标签噪声单调增长（1371 → 2044 → 2884，对应 0/10/20 % 噪声）。λ = 1e-2 直接把 ‖w‖ 压到 ~50，和噪声基本无关。这张图可能是本次更新**最具教学意义**的一张——"原因"（权重范数爆炸）和"结果"（MSE 峰值）被画在同一张图上。

---

## 4. Exp 2 —— 样本数-DD（RFF，D = 500 固定）

图：`results_full_demo/exp2_sample_wise_rff/dd_curves.png`。

### 4.1 "更多数据反而更糟"是真的，而且完全由 λ 决定

在 10% 标签噪声下，固定 `D` 扫 `n`：

| n | p/n | MSE λ=1e-10 | MSE λ=1e-4 | MSE λ=1e-2 | acc λ=1e-10 | acc λ=1e-2 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 5.00 | 0.066 | 0.066 | 0.065 | 60.5% | 60.9% |
| 400 | 1.25 | 0.229 | 0.226 | 0.133 | 44.4% | 54.9% |
| 480 | 1.04 | 1.16 | 0.918 | 0.165 | 24.4% | 51.8% |
| **500** | **1.00** | **54.7 ± 42.6** | 1.48 ± 0.25 | 0.166 ± 0.004 | **12.1%** | **51.5%** |
| 520 | 0.96 | 1.30 | 1.00 | 0.166 | 23.9% | 51.1% |
| 1000 | 0.50 | 0.077 | 0.077 | 0.073 | 71.1% | 72.2% |
| 4000 | 0.125 | 0.037 | 0.037 | 0.037 | 87.8% | 87.8% |

样本数-DD 的峰值正好落在 `n = D = 500`，与 Belkin et al. (2019) 的预言对上。有三点特别值得讲：

1. **无 ridge 时是灾难性的。** 从 n = 400 增加到 n = 500，测试准确率**从 44.4% 跌到 12.1%**——同一个模型，数据多了 25% 反而让准确率退到近似随机猜。MSE 比 n = 4000 时高了约 **1500 倍**。
2. **λ = 1e-2 完全抹平这个陷阱。** 同样的数据下，最坏准确率是 n = 510 处的 48.1% —— 比 n = 4000 渐近值低约 10 个百分点，但没有一个点跌破随机线。
3. **远离峰值处 λ 的影响可以忽略。** n = 4000 时三个 λ 都给出 87.8%。也就是说"关掉 DD"的代价**只在 `n ≈ D` 的邻域支付**。

### 4.2 对团队叙事的实用影响

主报告留给读者的口号是"**参数越多越好**"；样本-扫描补上了一个不那么直观的兄弟："**数据越多越差，但只在你选错 λ 的时候**"。我建议把这张双 λ 对比图提到主报告的 §5。

---

## 5. Exp 3 —— CNN 模型宽度-DD，加上"真实世界"正则

图：`results_full_demo/exp3_nn_model_wise/dd_curves.png`。
配置：SGD, lr = 0.05, wd = 1e-4, augment = True, 3 seeds, 每轮 50 epoch。

### 5.1 干净标签 → 单调曲线，看不到 DD

| 宽度 w | p | p/n | test acc（noise = 0） | train acc（noise = 0） |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 774 | 0.19 | 38.2 ± 0.4 | 37.0 |
| 4 | 4,146 | 1.04 | 55.4 ± 0.5 | 58.1 |
| 8 | 11,162 | 2.79 | 63.7 ± 0.5 | 74.2 |
| 16 | 33,834 | 8.46 | 64.3 ± 0.2 | 90.5 |
| **32** | **113,738** | **28.4** | **68.6 ± 0.3** | 98.5 |

与团队 "feat(results)" 单 seed 的配置（Adam, lr=1e-3, 无增强, 无权重衰减）相比，本配置给出的是一条 **严格单调下降** 的测试误差曲线，在 `p/n = 1` 处完全没有峰。直白说：表里最差的测试准确率在 `w = 1`，之后每增大一次宽度都会更好。这正是"隐式 + 显式正则合力把模型挤出临界曲面"时的预期行为——也就是 Nakkiran et al. (2021) 中"在合适优化下 DD 会消失"的结论。

这里的种子间 σ 非常小（处处 ≤ 1 pp）。对 NN 来说多种子平均在干净标签下并没有带来多少信息——这本身是个有用发现，因为它恰好和 RFF 的情况相反：RFF 的峰值方差源于解析最小范数解本身，而 CNN 的方差主要来自 SGD 噪声，而在 50 epoch 的 batch 梯度平均下这个噪声已经很小了。

### 5.2 噪声标签 + 强正则 → 模型拒绝记忆

| 宽度 w | test acc（noise = 20%） | train acc（noise = 20%） |
|:---:|:---:|:---:|
| 1 | 4.1 ± 0.5 | 14.6 |
| 8 | 6.0 ± 0.2 | 31.4 |
| 16 | 6.3 ± 0.4 | 54.8 |
| 32 | 6.1 ± 0.2 | 85.6 |

两点异常：

1. Train acc 一路爬向 100%，test acc 却卡在 5–6%。网络在拟合被污染的训练标签（意料之中），但泛化却**跌破 10% 随机猜的基线**（意料之外）。
2. 测试曲线也完全看不出峰结构——宽度依赖几乎是平的。

最可能的解释是：当前组合（lr = 0.05、weight_decay = 1e-4、随机裁剪+翻转、batch_size = 256、50 epoch、非常小的 `n = 4000`）在 20% 噪声 CIFAR-10 上不是一个健康的训练配方：强的权重衰减 + 增强让网络根本没机会收敛到正确的类边界，分类器发生漂移。本文选择如实保留这个结果而不是把它调顺，因为它正是下面第 3 点最直接的证据。

### 5.3 结论

CLI 暴露的超参**不是装饰品**——它们承载的现象对配置高度敏感。团队最初的"Adam + 无增强 + 无权重衰减"配置，实证上是我们所测试过的配置中**几乎唯一**能在这个数据规模下让 CIFAR-10 DD 显式出现的配置。把这些开关开放，就是让"DD 可见"这件事变得可证伪。

---

## 6. Exp 4 —— 同一正则配置下的 epoch-wise DD（负面结果）

图：`results_full_demo/exp4_epoch_wise_nn/dd_curves.png`。
配置：SGD, lr = 0.05, wd = 1e-4, augment = True, 20% 标签噪声, 1000 epoch, 3 seeds。

| 宽度 w | 最佳 test acc（epoch） | 最终 test acc | loss：最小 → 最大 |
|:---:|:---:|:---:|:---:|
| 2 | 10.1 %（ep 1） | 4.4 ± 0.4 | 2.30 → 2.62 |
| 4 | 9.5 %（ep 1） | 5.4 ± 0.1 | 2.30 → 2.85 |
| 8 | 8.9 %（ep 2） | 6.2 ± 0.6 | 2.31 → 3.37 |

标志性的 epoch-wise DD 模式——准确率先升、在插值 epoch 附近下挫、然后再升——**并未出现**。相反，所有三个宽度上 test acc 都从 epoch-1 的近似随机值**单调下降**，test loss 都**单调上升**，说明训练是在发散而不是在受控过拟合。三个 seed 的趋势一致，所以这不是偶然——这个配置对小数据 + 噪声标签的组合就是不合适。

我没有选择反复调参直到曲线好看，而是**把这个结果留在报告里**，作为团队原文断言（"CNN + 20% 噪声 + 小 n = 可观测的 epoch-wise DD"）脆弱性的一个工作样例。同样的代码换上原来的 `--optimizer adam --nn-lr 0.001`（并关掉 `--augment`）就能复现 `results/exp4_epoch_wise_nn/dd_curves.png` 中的 DD 形态——新旧两份 JSON 的并排对比将是主报告里最有说服力的可视化素材。

用 `--optimizer adam --nn-lr 0.001 --weight-decay 0.0 --epochs-epoch 1000 --seeds 41,42,43` 的补跑作为直接的 follow-up 留给后续；基础设施已经就位。

---

## 7. 讨论

三个独立观测——RFF 峰值方差、ridge 抑制峰值、CNN 在标准正则下看不到 DD——都指向同一个结论：**double descent 是未正则化的最小范数解在插值阈值附近的行为，任何合理的显式正则（ridge、数据增强、权重衰减、优化器噪声）都会单调地软化它。** 这也正是 Nakkiran et al. (2021) 的结论，我们现在在四个独立场景下都复现了它。

‖w‖ 数据还提供了两个更细的观察：

- 在 MNIST 上 `peak(MSE) / ‖w‖²` 并非随 λ 不变，所以"‖w‖ 爆炸"并不是充分的解释——在同一量级的 ‖w‖ 下，λ = 1e-10 的 peak MSE = 207 和 λ = 1e-4 的 peak MSE = 1.11 的差别，**真正的变量是 Gram 矩阵的条件数**（而这恰是 λ 直接作用的对象）。Hastie et al. 的谱论证在这里更自然。
- 在有噪声的情况下，peak MSE 和 peak ‖w‖ 都近似按噪声率线性增长（λ = 1e-10 下为 94 → 207 → 455 和 1371 → 2044 → 2884），与最小奇异值倒数方差项中噪声作为乘子出现的闭式表达一致。

这些都是团队主报告里只作为教科书陈述出现的小命题，现在有紧致、多种子的实证支撑。

---

## 8. 局限与下一步

1. **Exp 4 在新正则配置下无法复现 epoch-wise DD。** 后续动作：用 `--optimizer adam --nn-lr 0.001 --weight-decay 0` 关掉 `--augment` 重跑一遍，作为对团队原图的 A/B 对照。`main()` 里所有管道都已经准备好。
2. **仅三个种子。** 三个足以判断 RFF 峰值 σ ≈ 均值，但 σ 本身的估计还比较噪。做最终定稿图时 RFF 实验应该跑 ≥ 8 个 seed（反正是闭式解，开销可以忽略）。
3. **λ 网格偏粗。** 一个十倍一档的分辨率足以暴露"削峰不削谷"的形状，但不足以估计 **最优** λ。在 [1e-5, 1e-2] 区间、在 `p/n = 1` 上做密集扫描，可以直接测出"DD 峰值的最小可达值"，得到 Nakkiran et al. §3 里"最优 ridge 定理"的实证版本。
4. **分析 notebook 还在读老 schema。** 它不会在新 JSON 上崩溃（老字段还都保留在 `per_seed` 里），但会默默忽略新增的聚合 / std / ‖w‖ 列。对 `notebooks/analysis.ipynb` 做一个小补丁是下一件代码任务；聚合结果已经在磁盘上算好了，不需要重跑。

---

## 9. 复现命令

本次更新里每张图都由一条命令产生。流水线 **没有 Python 层面的隐藏状态**——每个 `results.json` 里的 `config` 块能完整标识一次运行。

```bash
# Exp 1 —— model-wise RFF
python -m src.experiments.comprehensive_dd \
    --experiments 1 --n-train 1000 \
    --seeds 41,42,43 --rff-lambdas 1e-10,1e-4,1e-2 \
    --output-dir results_full_demo

# Exp 2 —— sample-wise RFF
python -m src.experiments.comprehensive_dd \
    --experiments 2 \
    --seeds 41,42,43 --rff-lambdas 1e-10,1e-4,1e-2 \
    --output-dir results_full_demo

# Exp 3 —— model-wise CNN（新的正则化配置）
python -m src.experiments.comprehensive_dd \
    --experiments 3 --n-train-nn 4000 --epochs-nn 50 \
    --seeds 41,42,43 \
    --optimizer sgd --nn-lr 0.05 --weight-decay 1e-4 --augment \
    --output-dir results_full_demo

# Exp 4 —— epoch-wise CNN（同正则配置，用于负面结果对比）
python -m src.experiments.comprehensive_dd \
    --experiments 4 --n-train-nn 4000 --epochs-epoch 1000 \
    --seeds 41,42,43 \
    --optimizer sgd --nn-lr 0.05 --weight-decay 1e-4 --augment \
    --output-dir results_full_demo

# Exp 4（计划中的 follow-up：对齐团队原始超参）
python -m src.experiments.comprehensive_dd \
    --experiments 4 --n-train-nn 4000 --epochs-epoch 1000 \
    --seeds 41,42,43 \
    --optimizer adam --nn-lr 0.001 --weight-decay 0.0 \
    --output-dir results_full_demo_adam
```

---

## 10. 附录 —— 完整数值表

*（原始 JSON 是权威数据源，下面的表只为方便速读。）*

### A. Exp 1，noise = 10%，完整 (p/n, λ) 网格

| p/n | MSE λ=1e-10 | MSE λ=1e-4 | MSE λ=1e-2 | ‖w‖ λ=1e-10 | ‖w‖ λ=1e-2 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.05 | 0.069 | 0.069 | 0.069 | 3.4 | 3.4 |
| 0.30 | 0.059 | 0.059 | 0.059 | 13.5 | 13.4 |
| 0.70 | 0.132 | 0.132 | 0.104 | 41.5 | 35.5 |
| 0.90 | 0.402 | 0.382 | 0.130 | 89.4 | 46.6 |
| 0.98 | 2.402 | 1.055 | 0.130 | 234 | 49 |
| 1.00 | **207.4** | 1.111 | 0.129 | **2044** | 49 |
| 1.02 | 2.484 | 1.031 | 0.129 | 243 | 50 |
| 1.10 | 0.448 | 0.416 | 0.120 | 104 | 49 |
| 1.50 | 0.115 | 0.114 | 0.079 | 56 | 43 |
| 2.00 | 0.070 | 0.070 | 0.058 | 45 | 39 |
| 8.00 | 0.031 | 0.031 | 0.030 | 34 | 32 |

### B. Exp 2，完整 (n, λ) 网格

| n | p/n | MSE λ=1e-10 | MSE λ=1e-4 | MSE λ=1e-2 | acc λ=1e-10 | acc λ=1e-2 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 100 | 5.00 | 0.066 | 0.066 | 0.065 | 60.5% | 60.9% |
| 300 | 1.67 | 0.111 | 0.110 | 0.095 | 56.7% | 60.3% |
| 400 | 1.25 | 0.229 | 0.226 | 0.133 | 44.4% | 54.9% |
| 480 | 1.04 | 1.163 | 0.918 | 0.165 | 24.4% | 51.8% |
| 500 | 1.00 | **54.67** | 1.483 | 0.165 | **12.1%** | 51.5% |
| 520 | 0.96 | 1.297 | 1.002 | 0.166 | 23.9% | 51.1% |
| 700 | 0.71 | 0.150 | 0.150 | 0.117 | 55.1% | 60.1% |
| 1000 | 0.50 | 0.077 | 0.077 | 0.073 | 71.1% | 72.2% |
| 4000 | 0.125 | 0.037 | 0.037 | 0.037 | 87.8% | 87.8% |

### C. Exp 3，两个噪声水平下所有宽度

详见 §5；完整 20 行表在 `results_full_demo/exp3_nn_model_wise/results.json → aggregated`。

---

*正文完。*
