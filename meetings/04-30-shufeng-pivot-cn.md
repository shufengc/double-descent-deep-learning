# 4-30 提议 — DD-Recovery 当 paper headline + 周五前并行收尾（致 Yusheng / Zhengda / Yizheng）

各位，今晚一个比较紧的提议——关系到 paper 定位 + 这周五前各自任务安排。请抽空看一下，今晚到明天上午给个 reply。

## 1. 4-28 push 改了游戏规则

我今天才正式 push 上 shufeng 的 DD-Recovery（fractional-k ResNet, k∈{0.0625..2.0}, RTX 5090, 2000 epochs, n-slice 验证）跑出来 NN 端 model-wise DD：

**24.9% → 49.0% → 55.4%，peak 在 k≈0.1875，n-slice 验证 peak 随 n 右移。**

这是我们组**唯一一个干净复现 Nakkiran NN DD** 的实验。

Yusheng 昨晚那份 review（4 个 extension A/B/C/D）写的时候还没看到这次 push，所以他的定位是 "NN 没有 working DD，靠拓展去补"。我今天那 4 条 baseline 全跑完 push 上去了（commit `c34dcb8`），**但跑完之后我自己的判断：4-28 这一波应该当 paper headline，4 个 extension 退到 supporting**。

## 2. paper 重新定位

- **Headline** = "fractional-k ResNet recovers Nakkiran model-wise DD where literal ResNet18 at n=4000 cannot"
- A/B/C/D 当 supporting。**Person C 的 falsification（Adam/SGD 在 p/n≥2.8 都炸到 5–7%）反而支持 DD-Recovery 的论点**：optimizer 是次要变量、EMC 才是主轴。
- Yusheng 的 "Reproduction vs Extensions" 分层照旧，但 Extensions 那一层的核心从 §6.5–6.8 改成 DD-Recovery（升到 §6 主线）。

关于 Yusheng 之前说的 "和 lecture 重合 ~40%"——我核对了 11 份 lecture PDF + report §2.3，估计基本准确（bias-variance L1–L3 / NTK L7–L8 / Rademacher L9 / ridge L10 都直接复述了）。但 **4-28 + Person C falsification + Zhengda Exp8 + Yizheng OOD/curriculum 这部分是 60% 真延伸**，不是 lecture recap。**把 paper 重心放在这 60%，重合度问题就消失**。

## 3. 时间线（slide 上确认）

- **5/4 (周一)**：第一个 presentation slot, 10 min, 10 slides, **+3% EC**
- **5/8 (周二)**, **5/13 (周二)**：附加 slot
- **5/14 (周四)**：**paper due（~15 页，最重要的 deliverable）**

paper 不是 5/4 due，是 5/14。**我抢 5/4 的 EC slot**——deck 围绕 DD-Recovery headline 做。如果你们也想抢可以拼组（10 min 4 个人 alternation），不抢 5/4 我自己讲。

## 4. 关键提议：所有实验 **周五（5/1）EOD 前** 全做完

理由：5/2–5/3 周末做 deck，5/5–5/13 写 paper。**实验拖到下周一定写不完 paper**。
我给每个人安排了一条**不重叠**、**贴合各自之前最强工作**的深挖任务：

### Shufeng（我自己，今晚启动）
- **sample-wise NN DD**：fractional-k 上 sweep n ∈ {1k, 2k}（n=4k/8k 已有，复用）× k ∈ {0.0625, 0.125, 0.25, 0.5, 1.0} × 2 seeds = 20 runs，5090 上跑 ~3h，今晚 launch、明早收。
- 输出：`figures/samplewise_nn_dd.png` + report §6.9。

### Zhengda（建议）
- **NN spectral mechanism**：你 Exp8 的 RFF condition-number 是机理层最强的一块。**周五前**在我的 fractional-k ResNet checkpoints 上做一个 NN 版的对照——empirical-NTK Gram 矩阵 / Jacobian eigenspectrum，画 effective-rank vs k。假设：effective rank 在 k≈0.1875 spike，对应 RFF 在 p/n=1 的 condition-number spike。
- 我明早把 checkpoints 路径丢群里。计算预算 5090 上 3–5h，跟我的 sweep 错开时间共用一台机器。
- 输出：`src/experiments/exp_nn_spectral.py` + `figures/nn_effective_rank_vs_k.png` + report §6.10。
- 这是 paper 第二个 headline figure。如果你愿意做这一节我们合署。

### Yizheng（建议）
- **fractional-k 上的 epoch-wise / early-stop dynamics**：你 Supp3 early-stop CNN 跑出 38–58% recovery 是 NN 端最干净的 salvage。**周五前**把同一套早停诊断套到 fractional-k 上——k ∈ {0.125, 0.1875, 0.5} × 2 seeds × 2000 epochs，eval 每 25 个 epoch，画 test_acc(epoch) 三条曲线、标 early-stop optimum。
- 回答的问题："early stopping 是否在 fractional-k 上同样 kill peak？还是只在 over-parameterized 区域 kill？"
- 算力：5090 ~2h（6 runs × 20min），跟我的 sweep 错开。
- 输出：`src/experiments/exp_epochwise_fractionalk.py` + `figures/fractionalk_epochwise.png` + report §6.11。

### Yusheng（建议）
- **§2 Background 重写 + lecture-mapping table**：你写 PDF 的那两份 logic 分析能力组里最强，让你做 paper 最关键的理论 framing 是 ROI 最高的安排。**周五前**：
  1. 重写 report §2，加一节 "为什么 parameter count 是错的轴"，formal 引 Nakkiran §3 / Belkin benign overfitting / Bartlett norm-based bounds。
  2. 加一段 formal definition of double descent，区分 model-wise / sample-wise / epoch-wise。
  3. §2.4 加一张 lecture-mapping table（每个 experiment 对应到 L1–L11 的哪一节），把你之前 Reproduction vs Extensions 的 relabel 推到全 report。
- 没有 compute，纯写作 + 引文。预计 ~6h focused work。
- 这是 paper 引言部分最重要的一块，**Person D 的 bounds critique 没有你的 §2 framing 就立不住**。

## 5. 错峰 5090 使用

我今晚 22:00 launch sample-wise，预计跑到明早 02:00–03:00。Zhengda 上午 9:00 起可用，下午 14:00 起 Yizheng 用。Yusheng 不需要 5090。我会在群里同步 ssh alias 状态。

## 6. 总结

**周五（5/1）EOD 前每个人跑完 / 写完自己那块 → 周末（5/2–3）我集成 deck → 5/4 我演讲（拿 EC，paper 大头放后面） → 5/5–13 全员写 paper → 5/14 提交。**

中间不停。哪怕周末也要持续推进。

---

直接微信回复 ack 就行（包括 "我接 / 我换 / 我做不完"），我今晚就需要知道 Zhengda 和 Yizheng 接不接，明早我把 checkpoint 路径 + 数据准备好丢群里。

— Shufeng, 4/30 晚
