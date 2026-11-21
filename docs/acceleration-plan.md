# trop 加速计划

本文档汇总对两个外部项目的调研结论，并据此给出 `trop` Rust 核心的分阶段加速计划：

- [leoliu0/fereg](https://github.com/leoliu0/fereg)：高维固定效应回归（MIT），Rust 核心；
- [justinwaddy/TROP](https://github.com/justinwaddy/TROP)：另一个 Stata 版 TROP 实现（GPL-3.0），纯 Mata。

> 状态：**计划文档，尚未实现任何代码改动**。各项预期收益来自代码阅读与复杂度分析，尚无实测；每一阶段落地时都须附带基准测试（`run_bench.sh` / `rust/benches`）与数值一致性测试。

---

## 1. 背景：trop 的耗时在哪里

TROP 只有两组固定效应（αᵢ、βₜ），面板为平衡的 N×T（N、T 通常为几十到几千），因此：

| 环节 | 位置 | 单次成本 | 重复次数 |
|---|---|---|---|
| α/β 更新（一次 Gauss–Seidel 扫描） | `estimation.rs` `estimate_model_impl` 外层循环 | O(NT) | 外层迭代数 |
| L 的 FISTA 近端步（SVD 软阈值） | 同上，内层循环（`MAX_INNER_ITER`=10 / `MAX_INNER_ITER_HIGH`=50） | O(NT·min(N,T)) | 外层 × 内层 |
| 协变量 WLS（重建 X_w、X'WX） | 同上 Step 1b | O(NTK²) | 外层迭代数 |
| λ_nn = ∞ 的 joint 求解 | `solve_joint_no_lowrank` | **O(NT·(N+T)²)**（稠密虚拟变量矩阵 + LAPACK 最小二乘） | 每次调用 |
| 以上整体 | twostep 逐处理单元格 × LOOCV 网格 × bootstrap 次数 | | |

结论：瓶颈是 **SVD 的总次数**（外层×内层嵌套）、**`solve_joint_no_lowrank` 的稠密设计矩阵**，以及调用次数的乘积，而不是固定效应本身。

## 2. 外部项目的可借鉴点

### 2.1 fereg

fereg 的求解器"精确消去最大的一组固定效应，剩余系统在按单元格压缩的充分统计量上用预条件共轭梯度（PCG）求解"。

- **不能整体照搬**：PCG 面向千万级观测、≥3 组、连通结构差的固定效应；TROP 只有两组 FE，直接精确求解即可，无需迭代。
- **可借鉴**：①精确消去一组 FE + 小规模 Schur 补直接求解；②权重固定时充分统计量（X'WX 等）只算一次。

### 2.2 justinwaddy/TROP

| | justinwaddy/TROP | 本仓库 |
|---|---|---|
| 默认估计对象 | `group(time)`：按队列/块合并（快，近似） | 逐处理单元格（忠于论文 Algorithm 2） |
| FE 求解 | 预计算 (N+T)² Gram 逆 `Ginv`，每步 FISTA 精确求解 μ/α/β/τ；**单层循环** | 每轮一次 Gauss–Seidel，外/内**双层嵌套** |
| SVD | 每步 `fullsvd` | Gram 技巧，更快 |
| λ_nn 网格 | 从大到小 + 热启动，`Ginv` 全路径复用 | LOOCV 在观测间热启动，λ_nn 路径未复用 |
| CV 收敛准则 | 仅看 τ 相对变化 < 1e-6 | 全部参数最大绝对变化 < tol |
| CV 规模 | 默认 200 次安慰剂抽样；LOOCV 可 `cells(#)` 抽样 | 全部控制单元格 LOOCV |
| 并行 | 无 | rayon（bootstrap、LOOCV） |
| 大 N | 受 O((N+T)³) 求逆限制 | 更好 |
| 功能 | 无协变量、须平衡无缺失、无 survey、无 estat/predict | 全 |

它"快"的主要原因是默认少算（块合并 + 小规模 CV），但**单层 FISTA + 每步精确 FE** 是真正更优的算法结构，值得移植。

## 3. 分阶段计划

所有改动以"**数值结果与现实现一致（在 tol 内）**"为验收前提；改变估计对象的选项必须是可选项，默认行为保持忠于论文。

### 阶段 1：零风险的精确等价改写（优先）

**1a. `solve_joint_no_lowrank` 去虚拟变量化**
- 现状：构造 NT×(1+(N−1)+(T−1)+K) 稠密矩阵 + LAPACK lstsq。
- 改为：
  1. 有协变量时先按 Frisch–Waugh–Lovell 对 y 与 X 做加权双向去均值（用 1b 的精确两向求解）；
  2. 解 K×K 的 γ（Cholesky，失败回退 SVD，保留现有条件数上报）；
  3. 对残差精确求解 α、β：消去 α 后得到 T×T 的 Schur 补系统 `S β = r`，Cholesky 直接求解（T > N 时对称地消去 β）；
  4. 处理 rank deficiency / 零权重单元，与现有归一化约定（α₁=β₁=0 或现有截距约定）对齐。
- 复杂度：O(NT(K+1) + NTK² + min(N,T)³)，N=1000、T=50 时预计从 ~10¹⁰–10¹¹ flops 降到 ~10⁷。
- 测试：与当前 LAPACK 实现在随机面板（含缺失、零权重行列、协变量、近共线）上比对 τ、α、β、γ（差 < 1e-10 相对）。

**1b. 加权双向 FE 精确求解器（公共组件）**
- 新增 `fn solve_twoway_weighted(r, w, ...) -> (alpha, beta)`：给定残差 R 与权重 W，一次性精确求解 min Σ w(R−α−β)²。
- 实现：αᵢ 由 β 的闭式表达消去 → T×T（或 N×N，取小者）Schur 补 Cholesky；Schur 矩阵只依赖 W，**在一次 `estimate_model` 调用内只分解一次**，之后每次求解 O(NT + min(N,T)²)。
- 不连通 / 单行单列零权重的处理与现有 `unit_has_obs` / `time_has_obs` 语义一致。

**1c. 协变量充分统计量缓存**
- `estimate_model_impl` Step 1b：X'WX 及其 Cholesky 因子移出循环，仅一次；每轮只计算 X'W·残差，避免每轮重建 NT×K 的 `x_w`。
- 同时缓存 `x·γ` 的结果（当前在 Step 1 与 Step 2 各算一次）。

### 阶段 2：算法结构改进（需验证收敛与一致性）

**2a. 单层 FISTA：每步精确 FE**（借鉴 justinwaddy/TROP，预期收益最大）
- 把 `estimate_model_impl` 与 `solve_joint_with_lowrank` 的"外层 GS + 内层 FISTA"改为单层：
  每次迭代 = 动量外推 L̂ → 用 1b 精确求解 α、β（、γ）→ 梯度步 → SVD 软阈值 → 自适应重启。
- 目标函数与不动点不变（对 (α,β,γ) 精确最小化后对 L 做近端梯度 = 对 profiled 目标做 FISTA）。
- 保留外部契约：`converged`、`n_iters` 语义，`e(converged)==1 ⇒ 残差 < tol` 的现有测试。
- 验收：在 README 的 7 个基准数据集设定与 proptest 随机面板上，τ 与现实现差 < tol；记录 SVD 总调用次数与墙钟时间。

**2b. λ_nn 路径热启动**
- LOOCV / CV 网格中，对固定 (λ_unit, λ_time) 按 λ_nn 从大到小遍历，用上一个解热启动 L；1b 的 Schur 分解与 1c 的 X'WX 因子在同一权重下跨 λ_nn 复用。
- 与现有"观测间热启动"（`loocv.rs` `warm_alpha/beta/l`）组合使用。

**2c. 外层不动点加速（可选）**
- 若 2a 后仍有慢收敛区间（小 λ_nn，如 PWT），评估对迭代序列做 Anderson(m=5) 或 Irons–Tuck 加速，并带安全回退（目标函数不降则丢弃外推）。

### 阶段 3：可选的快速模式（改变计算量，不改默认）

**3a. CV 阶段宽松收敛**：新增选项（如 `cvtol()`），CV 中只要求 τ 的相对变化 < cvtol 即停止；最终点估计仍按 `tol()` 完整收敛。
**3b. LOOCV 单元格抽样**：新增 `cvcells(#)` 与种子，随机抽取部分控制单元格计算 LOOCV 目标，适用于大面板。
**3c. 块合并估计对象**：新增 `group(cell|time)`（默认 `cell`），`time` 时按"同一连续处理期 + 同处理时点"合并为块求解，块中点计算时间距离。需在 help 中明确这是近似估计对象。

### 阶段 4：SVD 与并行

- **截断 / 随机化 SVD**：软阈值后秩通常很低，用上一迭代的秩 r 估计做 rank-(r+p) 随机化 SVD（带过采样与幂迭代），若第 r+p 个奇异值仍高于阈值则回退到现有完整/Gram 路径，保证精确性。
- **并行覆盖**：检查 twostep 逐单元格循环与 CV 网格点是否都已用 rayon 并行；嵌套并行时控制线程池避免超额订阅。

## 4. 验证与基准

- **一致性测试**（每阶段必需）：新旧实现在以下情形比对 τ/α/β/γ/L：随机面板（proptest）、含 NaN、零权重行/列、交错与开关式处理、协变量近共线、λ_nn ∈ {0, 小, 大, ∞}。
- **基准**：扩展 `rust/benches`，覆盖 (N,T) ∈ {(50,40), (111,48), (500,50), (2000,30)}，分别记录 joint / twostep / LOOCV / bootstrap 的墙钟时间与 SVD 调用次数；在 PR 中报告前后对比。
- **Stata 端**：用 `run_bench.sh` 在四个平台插件上复测；`estat` / `predict` 结果不变。

## 5. 建议的落地顺序

| 顺序 | 项目 | 风险 | 预期收益 |
|---|---|---|---|
| 1 | 1b 精确两向 FE 求解器 | 低 | 为后续铺路 |
| 2 | 1a `solve_joint_no_lowrank` 去虚拟变量 | 低 | λ_nn=∞ 路径数量级提速 |
| 3 | 1c 协变量充分统计量缓存 | 低 | 有协变量时显著 |
| 4 | 2a 单层 FISTA | 中 | SVD 总次数下降，主路径最大收益 |
| 5 | 2b λ_nn 热启动 | 低 | LOOCV 显著 |
| 6 | 4 截断 SVD | 中 | 大面板显著 |
| 7 | 3a–3c 快速模式 | 低（可选项） | 用户自选 |
| 8 | 2c 不动点加速 | 中 | 视 2a 后表现而定 |

每一项单独成 PR，附一致性测试与基准结果。

## 6. 许可

fereg 为 MIT，justinwaddy/TROP 为 GPL-3.0，本仓库为 AGPL-3.0。上述均为通用数值线性代数技巧，计划自行实现，不直接复制外部代码。
