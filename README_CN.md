# bvb_predictor

[阅读英文版](README.md)

## 摘要

本仓库实现了一个轻量级、可复现的管道，用于足球**概率性精确比分预测**。我们将主队和客队进球建模为条件独立的泊松变量，其速率分别为 $\lambda_{home}$ 和 $\lambda_{away}$，并通过一个带有球队嵌入和时间感知的赛前特征的神经网络进行预测。

主要设计目标：

- **无前瞻性泄漏**：所有赛前滚动统计数据均使用一步偏移计算。
- **非平稳性处理**：通过时间衰减加权和可选的微调阶段来强调最近的比赛。
- **实用推理**：生成完整的比分分布（在有界网格内）和推导出的胜/平/负概率。

## 方法

### 问题表述

给定一场比赛 $(home, away, t)$，预测 $\lambda_{home}$ 和 $\lambda_{away}$。然后我们得到比分分布：

$$
P(H=h, A=a) = \text{Poisson}(h;\lambda_{home})\;\text{Poisson}(a;\lambda_{away}),\quad h,a \in [0,\dots,G].
$$

### 模型

该模型在 `src/bvb_predictor/models/poisson_mlp.py` 中以 `TeamPoissonScoreModel` 的形式实现：

- 球队嵌入：`home_team_id`，`away_team_id`
- 联赛嵌入：`league_id`
- MLP 骨干（ReLU + BatchNorm + Dropout）
- 双头 `Softplus` 输出以确保 $\lambda>0$

训练目标是泊松负对数似然（NLL）。启用时，使用**时间衰减**加权变体。

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

- 滚动式赛前球队统计数据（偏移）：L5/L10 的进球数/失球数/净胜球，以及主场专属/客场专属的 L10。
- 休息日特征：`home_rest_days`，`away_rest_days`（自上次比赛以来的天数）。
- 赛季特征：`season_start_year`。
- 赔率衍生信号：
  - 隐含概率 `prob_home/prob_draw/prob_away`
  - 返还率 `odds_overround`
  - 可用性标志 `odds_available`

所有特征均按时间顺序计算，并使用 `shift(1)` 来防止未来数据泄露。

## 训练

### 脚本入口点

- 训练：`scripts/train.py`
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
- `artifacts/metrics.json`（评估摘要）

### 指标

`scripts/train.py` 报告：

- 验证/测试集上的泊松 NLL
- $\lambda$ 与实际进球数的 MAE
- 胜/平/负准确率（从预测比分矩阵导出）
- 精确比分 top-1 和 top-k 命中率

## 推理

编辑 `run_predict.sh` 并运行：

```bash
./run_predict.sh
```

预测器返回一个 JSON，其中包含：

- `lambda_home`，`lambda_away`
- `topk_scores`（$[0,G]$ 内最可能的比分线）
- `wdl`（主队/平局/客队概率）

推理支持**任意比赛日期**：历史行会自动截断到目标日期之前的数据。

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
