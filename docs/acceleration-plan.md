# trop 加速计划

本文档汇总对以下外部项目的调研结论，并据此给出 `trop` Rust 核心的分阶段加速计划：

- **fereg**（注意：存在两个同名但互相独立的包）：
  - [johannes-schmieder/fereg](https://github.com/johannes-schmieder/fereg)：Johannes F. Schmieder 的 Stata 命令 `fereg`/`feppml`，对应论文 Schmieder (2026), *fereg: Fast High-Dimensional Fixed-Effects Regressions in Stata*（[NBER w36017](https://www.nber.org/papers/w36017)、[CEPR DP22170](https://cepr.org/publications/dp22170)）。"精确消去 + 约简系统（reduced-system）"路线的原始设计来源。
  - [leoliu0/fereg](https://github.com/leoliu0/fereg)：Leo Liu 的 Stata/Python/R 包（MIT），Rust 核心，独立实现了"精确消去最大一组 FE + 在单元格压缩充分统计量上做 PCG"。
- [reisportela/xhdfe-xfe](https://github.com/reisportela/xhdfe-xfe)：xhdfe（MIT），C++/OpenMP/CUDA 多组 HDFE 回归，集成了多条求解路线与精度校验/回退机制。
- [justinwaddy/TROP](https://github.com/justinwaddy/TROP)：另一个 Stata 版 TROP 实现（GPL-3.0），纯 Mata。

> 更正：本文档早先版本把 NBER 论文（Schmieder）与 leoliu0/fereg 混为一谈，二者是不同的包。

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

### 2.1 fereg（Schmieder；leoliu0）

两者核心思路相同：精确消去规模最大的一组固定效应，对剩余的约简（Schur 补）系统做预条件共轭梯度（PCG）。leoliu0 版在"按单元格压缩的充分统计量"上运行。二者的基准数据均为作者自报。

- **不能整体照搬**：PCG 面向千万级观测、≥3 组、连通结构差的固定效应；TROP 只有两组 FE，直接精确求解即可，无需迭代。
- **可借鉴**：①精确消去一组 FE + 小规模 Schur 补直接求解；②权重固定时充分统计量（X'WX 等）只算一次。

### 2.2 xhdfe-xfe

xhdfe 面向千万级观测与多组 FE，路线包括 Irons–Tuck 加速 MAP（参考 fixest）、图预条件 MLSMR / 加性 Schwarz（来自 Fischer–Schröder 的 `within`）、以及沿用 Schmieder fereg 设计的约简系统 PCG；并引用 Somaini & Wolak (2016) 的双向 FE 算法。

- **可借鉴**：
  1. 约简 Schur 系统 —— 即本计划 1b；Somaini & Wolak (2016) 提供了双向 FE 的直接文献依据。
  2. Irons–Tuck 加速 —— 即本计划 2c。
  3. **"快速路径 + 精度校验 + 回退"模式** —— 用于阶段 4 的截断 SVD：校验第 r+p 个奇异值是否低于阈值，不满足则回退完整 SVD。
  4. **补偿求和 + 确定性并行归约** —— 本身不提速，但保证结果不随线程数变化，是扩大 rayon 并行而不改变结果的前提（见阶段 0 / 阶段 4）。
- **不适用**：图预条件 LSMR / Schwarz / PCG（两组 FE 直接精确解即可）；GPU（N×T 太小，传输开销抵消收益）；singleton 剔除（平衡面板不存在）；并行 Stata 数据拷贝（`stata_bridge.c` 逐值 `SF_vdata` 在 TROP 规模下不是瓶颈）。

### 2.3 justinwaddy/TROP

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

### 阶段 0：逐位一致的实现级优化（最先做）

本阶段只改实现、不改浮点运算顺序，验收标准为新旧实现输出**逐位相同**（`to_bits()` 比较）。以下各项已核对当前代码中尚未实现；已有的距离缓存（`UnitDistanceCache`）、LOOCV 分数缓存与剪枝、观测间热启动、LTO/`codegen-units=1` 不在此列。

**0a. 循环不变量外提**（`estimation.rs` `estimate_model_impl`）
- `w_norm` 在每轮外层迭代中用 `from_shape_fn` 重建，但在整个调用内是常数 → 循环外计算一次。
- `r`、`r_target`、`r_masked` 每轮重新分配 → 预分配缓冲区复用。
- 有协变量时 `x·γ` 在 Step 1 与 Step 2 各算一次 → 每轮算一次并缓存。
- `solve_joint_with_lowrank` 同样处理。

**0b. 缓存友好的访问顺序**
- 数组按 (t, i) 行优先存储，α 更新的内层循环沿 t 跨步访问。改为外层 t、内层 i、用 `num[i]` 向量累加；每个 αᵢ 的求和仍按 t 递增，结果逐位相同。

**0c. 去掉热循环中的边界检查**
- 将 `[[t, i]]` 索引改为切片 / `ndarray::Zip` / `as_slice()`，便于自动向量化；不启用 fast-math，求和顺序不变。

**0d. 合并 FISTA 内循环的多次遍历**
- 动量外推、梯度步、重启内积、`max_abs_diff_2d` 收敛判断当前各遍历 NT 一次，合并为 1–2 次遍历（保持各求和的累加顺序）。

**0e. 相同权重矩阵的求解去重**（预期为 LOOCV 中最大的单项收益）
- twostep 中目标 (i, t) 的拟合只依赖于（估计 mask，权重矩阵）。权重 = θ(t) ⊗ ω(i, t)：
  - λ_unit = 0 → ω 为常数，所有相同 t 的处理单元格共享同一次求解；
  - λ_unit = λ_time = 0 → 全部处理单元格共享一次求解；
  - 默认 LOOCV 网格通常含 0，影响面大。
- 实现：以（mask 哈希，权重矩阵位模式哈希）为键缓存 (α, β, L, γ, n_iters, converged)，命中时直接复用；键冲突时做完整比较。

**0f. bootstrap 复用单位距离**
- `bootstrap.rs` 每次重抽样都重建 `UnitDistanceCache`（约 702、1219 行）。重抽样面板中两单位的距离等于其来源单位在原样本中的距离（重复抽到同一单位时距离为 0），因此在全样本上建一次缓存，再做下标映射即可。

**0g. 确定性并行归约**（为阶段 4 扩大并行做准备）
- 所有并行汇总（ATT、LOOCV 分数、bootstrap 统计量）改为"按固定下标顺序归并"，使结果与线程数无关；如需更高精度，可选用补偿（Kahan / double-double）求和——此项会改变末位，须单独开关并记录。

### 阶段 1：精确等价改写（结果在舍入误差内一致）

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
- 文献依据：Somaini & Wolak (2016), "An algorithm to estimate the two-way fixed effects model", *Journal of Econometric Methods* 5(1): 143–152；Schmieder fereg 的约简系统路线。
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

- **剔除第 t 期的单位距离 O(1) 计算**：缓存每对单位的平方差总和与有效期数，计算剔除第 t 期的距离时减去该期那一项，避免每次遍历 T。减法带来舍入级差异（非逐位一致），须在容差内验收。
- **热启动的子空间 SVD**：用上一迭代的奇异向量做块 Krylov / 子空间迭代，配合下述校验与回退。
- **截断 / 随机化 SVD**：软阈值后秩通常很低，用上一迭代的秩 r 估计做 rank-(r+p) 随机化 SVD（带过采样与幂迭代），若第 r+p 个奇异值仍高于阈值则回退到现有完整/Gram 路径，保证精确性。
- **并行覆盖**：检查 twostep 逐单元格循环与 CV 网格点是否都已用 rayon 并行；嵌套并行时控制线程池避免超额订阅。

## 4. 验证与基准

- **逐位一致测试**（阶段 0）：对同一输入比较新旧实现所有输出的 `f64::to_bits()`，并在 1/2/4/8 线程下重复。
- **一致性测试**（阶段 1 起）：新旧实现在以下情形比对 τ/α/β/γ/L：随机面板（proptest）、含 NaN、零权重行/列、交错与开关式处理、协变量近共线、λ_nn ∈ {0, 小, 大, ∞}。
- **基准**：扩展 `rust/benches`，覆盖 (N,T) ∈ {(50,40), (111,48), (500,50), (2000,30)}，分别记录 joint / twostep / LOOCV / bootstrap 的墙钟时间与 SVD 调用次数；在 PR 中报告前后对比。
- **Stata 端**：用 `run_bench.sh` 在四个平台插件上复测；`estat` / `predict` 结果不变。

## 5. 建议的落地顺序

| 顺序 | 项目 | 风险 | 预期收益 |
|---|---|---|---|
| 0 | 0a–0f 逐位一致的实现级优化（单独一个 PR） | 极低 | 常数倍提速；0e 在 LOOCV 中可能很大 |
| 0′ | 0g 确定性并行归约 | 低 | 为扩大并行铺路 |
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

两个 fereg 与 xhdfe-xfe 均为 MIT，justinwaddy/TROP 为 GPL-3.0，本仓库为 AGPL-3.0。上述均为通用数值线性代数技巧，计划自行实现，不直接复制外部代码。
