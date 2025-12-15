# bvb_predictor

[阅读英文版](README.md)

## 摘要

本仓库实现了一个轻量级、可复现的管道，用于足球**概率性精确比分预测**。我们通过一个带有球队/联赛嵌入与时间感知赛前特征的神经网络来预测进球分布，并输出完整的比分概率矩阵（在有界网格内）及其导出的胜/平/负概率。

主要设计目标：

- **无前瞻性泄漏**：所有赛前滚动统计数据均使用一步偏移计算。
- **非平稳性处理**：通过时间衰减加权和可选的微调阶段来强调最近的比赛。
- **实用推理**：生成完整的比分分布（在有界网格内）和推导出的胜/平/负概率。

## 方法

### 问题表述

给定一场比赛 $(home, away, t)$，预测期望进球数 $\mu_{home}$ 和 $\mu_{away}$，并在目标网格 $[0,\dots,G]$ 上得到有界的精确比分分布。

除了基线的独立泊松比分模型外，我们还支持：

- **Negative Binomial (NB)**：用于建模过度离散（方差 > 均值）
- **Dixon-Coles (DC)**：用于低比分区域的相关性修正

当前训练/推理默认使用 **NB + DC**，并按联赛学习可训练参数：

- $\rho_{league}$：Dixon-Coles 相关参数
- $\alpha_{league}$：Negative Binomial dispersion 参数

#### NB+DC 比分分布（有界网格）

我们采用 Negative Binomial 的均值-离散度参数化（均值 $\mu$，dispersion $\alpha$）：

$$
P_{NB}(X=k\mid \mu,\alpha)=\frac{\Gamma(k+\alpha^{-1})}{\Gamma(\alpha^{-1})\,k!}\left(\frac{\alpha^{-1}}{\alpha^{-1}+\mu}\right)^{\alpha^{-1}}\left(\frac{\mu}{\alpha^{-1}+\mu}\right)^k.
$$

Dixon-Coles 在低比分区域使用修正因子 $\tau(h,a)$：

$$
\tau(h,a)=
\begin{cases}
1-\mu_{home}\mu_{away}\rho, & (h,a)=(0,0)\\
1+\mu_{home}\rho, & (h,a)=(0,1)\\
1+\mu_{away}\rho, & (h,a)=(1,0)\\
1-\rho, & (h,a)=(1,1)\\
1, & \text{其他情况.}
\end{cases}
$$

在有界网格 $h,a\in[0,\dots,G]$ 上，联合分布定义为：

$$
P(H=h,A=a) \propto \tau(h,a)\,P_{NB}(H=h\mid\mu_{home},\alpha)\,P_{NB}(A=a\mid\mu_{away},\alpha),
$$

并在网格内归一化。

### 模型

该模型在 `src/bvb_predictor/models/poisson_mlp.py` 中以 `TeamPoissonScoreModel` 的形式实现：

- 球队嵌入：`home_team_id`，`away_team_id`
- 联赛嵌入：`league_id`
- MLP 骨干（ReLU + BatchNorm + Dropout）
- 双头 `Softplus` 输出以确保 $\mu>0$
- 按联赛可学习的 $\rho$（DC）与 $\alpha$（NB）

训练目标是 **NB+DC 负对数似然（NLL）**。启用时，使用**时间衰减**加权变体。

### 时间感知学习 (A+B+C)

为了反映足球表现会随时间漂移的特点：

- **(A) 时间衰减加权**：在*训练集*上分配样本权重 $w=\exp(-\Delta days/\tau)$。
- **(B) 微调**：在主训练阶段之后，可选地在训练集内最后 $N$ 个赛季上进行微调。
- **(C) 显式时间特征**：包含赛季和休息日信号。

## 数据

### 原始数据获取

我们目前支持 `scripts/data/fetch_football_data.py` 中的 `football-data.co.uk` 德甲（`D1`）数据。

原始文件存储在：

```text
data/raw/football-data.co.uk/D1/
```

### 清理和标准化

处理脚本 `scripts/data/build_processed_matches.py` 生成：

```text
data/processed/matches.csv
data/processed/report.json
```

处理后的最小模式是：

```text
match_id,date,league,season,home_team,away_team,home_score,away_score,odds_home,odds_draw,odds_away
```

注意：

- 该脚本包含 `--repair` 模式，用于规范化格式错误的 CSV 行（填充/修剪尾随空列），同时保留比赛行。

## 特征工程

`src/bvb_predictor/features/league_rolling.py` 实现了 `build_league_features`：

- 滚动式赛前球队统计数据（偏移）：L1/L3/L5 的进球数/失球数/净胜球，并包含主场专属/客场专属拆分。
- 指数滑动平均（EMA）特征（GF/GA/GD）。
- 休息日特征：`home_rest_days`，`away_rest_days`（自上次比赛以来的天数）。
- 赛季特征：`season_start_year`。
- 赔率衍生信号：
  - 隐含概率 `prob_home/prob_draw/prob_away`
  - 返还率 `odds_overround`
  - 可用性标志 `odds_available`

赔率/概率特征支持两种模式：

- 若存在原始赔率（`odds_home/draw/away`），则归一化得到 `prob_*`。
- 若缺少赔率但提供了 `prob_*`（例如由 odds 模型预测得到），则直接使用并视作可用。

所有特征均按时间顺序计算，并使用 `shift(1)` 来防止未来数据泄露。

## 训练

### 脚本入口点

- 训练：`scripts/train.py`
- 赔率模型训练（可选）：`scripts/train_odds.py`
- 推理：`scripts/predict.py`
- 便捷包装器：
  - `run_train.sh`
  - `run_predict.sh`

### 运行训练

编辑 `run_train.sh` 设置路径和超参数，然后运行：

```bash
./run_train.sh
```

生成物：

- `artifacts/model.pt`（模型权重 + 编码器 + 归一化统计数据）
- `artifacts/odds.pt`（可选：赔率模型权重 + 编码器 + 归一化统计数据）
- `artifacts/metrics.json`（评估摘要）

### 指标

`scripts/train.py` 报告：

- 验证/测试集上的 NB+DC NLL
- $\mu$ 与实际进球数的 MAE
- 胜/平/负准确率（从预测比分矩阵导出）
- 精确比分 top-1 和 top-k 命中率

当存在赔率模型时，训练支持可选的 **odds mix**：

- 将一部分训练样本的 `prob_home/draw/away` 替换为赔率模型预测的概率。
- 用于降低推理时“用户不提供真实赔率”的分布偏移。

## 推理

编辑 `run_predict.sh` 并运行：

```bash
./run_predict.sh
```

预测器返回一个 JSON，其中包含：

- `mu_home`，`mu_away`
- `rho`，`alpha`
- `topk_scores`（$[0,G]$ 内最可能的比分线）
- `wdl`（主队/平局/客队概率）

推理支持**任意比赛日期**：历史行会自动截断到目标日期之前的数据。

推理支持可选的 **odds 模型**：

- 当用户不提供原始赔率且提供 `--odds-model artifacts/odds.pt` 时，推理会先预测 `prob_home/draw/away` 并作为 pseudo-odds 特征参与比分预测。

## GPU / device

所有 torch 入口脚本支持 `--device`：

- `auto`（默认）：可用则用 CUDA
- `cpu`
- `cuda` / `cuda:0` / ...

包装脚本也提供 `DEVICE` 变量：

- `run_train.sh`：为 `train_odds.py` 与 `train.py` 透传 `--device`
- `run_predict.sh`：为 `predict.py` 透传 `--device`

## 仓库结构

```text
src/bvb_predictor/
  data/
    dataset.py
  features/
    league_rolling.py
  models/
    poisson_mlp.py
  utils/
    score_prob.py

scripts/
  train.py
  train_odds.py
  predict.py
  data/
    fetch_football_data.py
    build_processed_matches.py

run_train.sh
run_predict.sh
```

## 限制和未来工作

- 泊松独立性假设可能低估主客队进球之间的相关性。
- 特征集有意保持最小；加入更丰富的协变量（xG、阵容、伤病）应能改善校准。
- 数据源覆盖目前主要集中在德甲；扩展到欧洲赛事（欧冠/欧联）和多联赛训练是自然的下一步。
