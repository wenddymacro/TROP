# trop

**Stata 三重稳健面板估计量**

[![Stata 17+](https://img.shields.io/badge/Stata-17%2B-blue.svg)](https://www.stata.com/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Version: 1.2.0](https://img.shields.io/badge/Version-1.2.0-green.svg)]()
[![Build](https://github.com/gorgeousfish/TROP/actions/workflows/build-plugins.yml/badge.svg)](https://github.com/gorgeousfish/TROP/actions/workflows/build-plugins.yml)
![Platforms](https://img.shields.io/badge/platforms-macOS%20|%20Linux%20|%20Windows-blue)

![trop](image/image.png)

## 概述

`trop` 为 Stata 实现了 Athey, Imbens, Qu 与 Viviano (2025) 提出的**三重稳健面板（Triply RObust Panel, TROP）估计量**。该估计量结合三个组件——**单位权重**、**时间权重**与**核范数正则化低秩回归调整**——在具有复杂处理分配模式的面板数据中估计处理组平均处理效应（ATT）。

在基于七个真实数据集校准的半合成模拟中（论文表 1），TROP 在 **21 个设定中的 20 个**取得最低 RMSE，全面优于 DID、SC、SDID、MC 与 DIFP 估计量：

| 数据集 | 结果变量 | 处理 | N | T | TROP | SDID | SC | DID | MC | DIFP |
|--------|----------|------|---|---|------|------|----|-----|-----|------|
| CPS | log-wage | 最低工资 | 50 | 40 | **1.00** | 1.14 | 1.44 | 1.91 | 1.26 | 1.22 |
| CPS | urate | 最低工资 | 50 | 40 | **1.00** | 1.05 | 1.11 | 1.89 | 1.10 | 1.09 |
| PWT | log-GDP | 民主化 | 111 | 48 | **1.00** | 1.44 | 1.59 | 7.85 | 1.76 | 1.54 |
| Germany | GDP | 随机 | 17 | 44 | **1.00** | 1.46 | 2.82 | 3.58 | 1.56 | 2.46 |
| Basque | GDP | 随机 | 18 | 43 | **1.00** | 1.02 | 4.55 | 9.11 | 1.70 | 2.47 |
| Smoking | packs pc | 随机 | 39 | 31 | **1.00** | 1.22 | 1.48 | 2.16 | 1.14 | 1.45 |
| Boatlift | log-wage | 随机 | 44 | 19 | **1.00** | 1.34 | 1.62 | 1.35 | 1.04 | 1.62 |

<sub>标准化 RMSE 取自 Athey et al. (2025) 表 1。完整结果覆盖 7 个数据集的 21 个设定。</sub>

**主要特性：**

- **三重稳健估计** — 当单位权重、时间权重或回归调整中任一组件消除偏差时即可实现渐近无偏（定理 5.1）
- **两种估计方法** — Twostep（逐观测，异质性处理效应；算法 2）与 Joint（加权最小二乘，同质性处理效应）
- **留一交叉验证** — 通过坐标循环 LOOCV 进行调优参数的数据驱动选择（算法 1）
- **Bootstrap 推断** — 分层单位块 Bootstrap 用于方差估计与置信区间构建（算法 3）
- **一般性处理分配模式** — 支持交错采用、切换处理和任意二元处理矩阵
- **全面的后估计诊断** — 13 个 `estat` 子命令（含三重稳健偏差分解、事件研究、前趋势检验与表格导出）与 11 种 `predict` 类型
- **协变量调整** — 时不变协变量 X_i'γ（论文 6.2 节公式 14）与自动 WLS 投影
- **调查设计支持** — 分层、PSU 聚类与有限总体校正（FPC），通过 Rao-Wu 重缩放 Bootstrap
- **高性能后端** — 核心计算通过 Rust 编译插件实现，无需外部依赖

## 核心概念

### 三重稳健性

TROP 估计量结合三个组件，分别针对不同来源的混杂：

| 组件 | 作用 | 控制参数 |
|------|------|----------|
| **单位权重** $\omega_j$ | 提升与处理单位相似的控制单位权重 | $\lambda_{\text{unit}}$ |
| **时间权重** $\theta_s$ | 提升接近处理时期的时间段权重 | $\lambda_{\text{time}}$ |
| **低秩因子模型** $\mathbf{L}$ | 捕获未观测的交互固定效应 | $\lambda_{nn}$ |

核心洞察在于：估计量的偏差受三个不平衡项**乘积**的约束（定理 5.1）。只要任一组件成功消除偏差，整体偏差即消失——这就是"三重"稳健的含义。这一乘积形式的界严格优于 DID、SC 或 SDID 所依赖的加法形式界。

### 特殊情形

TROP 框架嵌套了现有估计量：

| 参数设定 | 等价方法 |
|----------|----------|
| $\lambda_{nn} = \infty$，均匀权重 | **DID / TWFE** |
| 均匀权重，$\lambda_{nn} < \infty$ | **矩阵补全 (MC)** |
| $\lambda_{nn} = \infty$，特定单位/时间权重 | **SC / SDID** |

## 环境要求

- Stata 17.0 或更高版本
- 无需额外安装其他 Stata 包
- 已包含预编译插件（macOS ARM64/Intel、Windows x64）

### 支持平台

| 平台 | 插件文件 | 状态 |
|------|----------|------|
| macOS Apple Silicon (ARM64) | `trop_macos_arm64.plugin` | ✅ 预编译 |
| macOS Intel (x86-64) | `trop_macos_x64.plugin` | ✅ 预编译 |
| Windows x86-64 | `trop_windows_x64.plugin` | ✅ 预编译 |
| Linux x86-64 | `trop_linux_x64.plugin` | ✅ 预编译 |

## 安装

### 方式 A：从 GitHub 安装（推荐）

```stata
net install trop, from("https://raw.githubusercontent.com/gorgeousfish/TROP/main") replace
```

自动安装以下内容：
- 所有命令和帮助文件
- 预编译 Mata 库
- 对应平台的预编译插件

### 方式 B：本地安装

如已下载或克隆本仓库：

```stata
net install trop, from("/path/to/TROP") replace
```

### 验证安装

```stata
trop, version
trop_check
```

## 快速开始与示例

```stata
* 安装
net install trop, from("https://raw.githubusercontent.com/gorgeousfish/TROP/main") replace

* 加载示例数据
trop_data cps_logwage

* 估计
trop y d, panelvar(id) timevar(t) method(twostep) fixedlambda(0.5 0 0.01)
```

以下所有示例使用 **CPS log-wage 数据集** — 50 个美国州 × 40 年（1979–2018）的州级对数工资数据，其中 `d` 标记最低工资上调生效的州-年观测。这是 Athey et al. (2025) 的七个基准数据集之一。

**数据集：** N = 50, T = 40, 2,000 个观测。`y` = 州级对数工资；`d` = 最低工资处理（8 个处理州-年单元格，0.4%）；`id` = 州标识；`t` = 年份。

### 示例 1：固定超参数（Twostep）

使用论文推荐的 CPS log-wage 参数值（Athey et al. 2025 表 S.1）：

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t) fixedlambda(0.1 0 0.9)
```

输出：

```
------------------------------------------------------------------------------
Triply Robust Panel Estimator                   Number of obs      =    2,000
Method: twostep                                 Number of units    =       50
                                                Time periods       =       40
                                                Treated obs        =        8
                                                Bootstrap reps     =      200
------------------------------------------------------------------------------
             |      ATT       Std. err.    t     P>|t|    [95% conf. interval]
-------------+----------------------------------------------------------------
           d |   0.031406     0.014098      2.23  0.061    0.004771    0.056458
------------------------------------------------------------------------------
Lambda: time = 0.100, unit = 0.000, nn = 0.900 (fixed)
Convergence: Yes (5 iterations)
------------------------------------------------------------------------------
```

### 示例 2：LOOCV 自动选择超参数

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t)
```

输出（LOOCV 通过坐标下降网格搜索选择最优超参数）：

```
------------------------------------------------------------------------------
Triply Robust Panel Estimator                   Number of obs      =    2,000
Method: twostep                                 Number of units    =       50
                                                Time periods       =       40
                                                Treated obs        =        8
                                                Bootstrap reps     =      200
------------------------------------------------------------------------------
             |      ATT       Std. err.    t     P>|t|    [95% conf. interval]
-------------+----------------------------------------------------------------
           d |   0.034188     ...           ...   ...     ...         ...
------------------------------------------------------------------------------
Lambda: time = 0.500, unit = 5.000, nn = 1.000 (LOOCV, Q = 3.4717)
Convergence: Yes (... iterations)
------------------------------------------------------------------------------
```

**注意：** 在 N = 50, T = 40 的面板上运行 LOOCV 需要对每个 D = 0 的单元格求和（论文公式 5），可能耗时 20–40 分钟。当不需要完整网格搜索时，请使用 `fixedlambda()`。

### 示例 3：Bootstrap 推断

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t) ///
    fixedlambda(0.1 0 0.9) bootstrap(200) seed(42)
```

输出：

```
------------------------------------------------------------------------------
Triply Robust Panel Estimator                   Number of obs      =    2,000
Method: twostep                                 Number of units    =       50
                                                Time periods       =       40
                                                Treated obs        =        8
                                                Bootstrap reps     =      200
------------------------------------------------------------------------------
             |      ATT       Std. err.    t     P>|t|    [95% conf. interval]
-------------+----------------------------------------------------------------
           d |   0.031406     0.014098      2.23  0.061    0.004771    0.056458
------------------------------------------------------------------------------
Lambda: time = 0.100, unit = 0.000, nn = 0.900 (fixed)
Convergence: Yes (5 iterations)
------------------------------------------------------------------------------
```

### 示例 4：Joint 方法

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t) ///
    method(joint) fixedlambda(0.1 0 0.9)
```

输出：

```
------------------------------------------------------------------------------
Triply Robust Panel Estimator                   Number of obs      =    2,000
Method: joint                                   Number of units    =       50
                                                Time periods       =       40
                                                Treated obs        =        8
                                                Bootstrap reps     =      200
------------------------------------------------------------------------------
             |      ATT       Std. err.    t     P>|t|    [95% conf. interval]
-------------+----------------------------------------------------------------
           d |   0.031406     0.014097      2.23  0.061    0.004771    0.056458
------------------------------------------------------------------------------
Lambda: time = 0.100, unit = 0.000, nn = 0.900 (fixed)
Convergence: Yes (2 iterations)
Global intercept (mu):   5.154320
------------------------------------------------------------------------------
```

### 示例 5：后估计工作流

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t) fixedlambda(0.1 0 0.9)

* 预测反事实结果和处理效应
predict y0_hat, y0
predict te_hat, te

* 诊断
estat summarize
estat weights
estat factors
```

`estat summarize` 输出：

```
------------------------------------------------------------------------------
Estimation sample summary
------------------------------------------------------------------------------
  Number of observations:        2000    (balanced panel)
  Number of units (N):             50
  Number of periods (T):           40
  Missing rate:                   0.0%

Treatment structure:
  Treated observations:             8    (  0.4%)
  Control observations:          1992    ( 99.6%)
  Treated units:                    8    ( 16.0%)
  Treated periods:                  1    (  2.5%)
  Pattern:                   multiple_treated_simultaneous

Outcome variable (y):
  Mean:          5.925
  Std. Dev:      0.444
  Min:           4.798
  Max:           6.806
  p25:           5.584
  p75:           6.298

Estimation details:
  Method:        twostep (Algorithm 2 default)
  Outcome var:   y
  Treatment var: d
  Panel var:     id
  Time var:      t
------------------------------------------------------------------------------
```

若需查看 LOOCV 诊断，请不使用 `fixedlambda()` 运行：

```stata
trop_data cps_logwage
trop y d, panelvar(id) timevar(t)
estat loocv
```

### 示例 6：独立 Bootstrap（后估计）

```stata
trop_data cps_logwage

* 先估计不含 bootstrap（更快的迭代）
trop y d, panelvar(id) timevar(t) fixedlambda(0.1 0 0.9) bootstrap(0)

* 再单独添加 bootstrap 推断
trop_bootstrap, nreps(200) seed(42)
```

输出：

```
------------------------------------------------------------
TROP Bootstrap Inference Results
------------------------------------------------------------
ATT estimate:       0.031406
Bootstrap SE:       0.014098
95% CI:       [   -0.001929,     0.064742]
p-value:              0.0612

Bootstrap reps:       200
Valid reps:           200
------------------------------------------------------------
```

### 示例 7：PWT Log-GDP 面板（111 国家 × 48 年）

对于大面板，使用 Penn World Tables 民主化数据集：

```stata
trop_data pwt_loggdp

* 论文的 PWT 超参数。大面板配合极小 lambda_nn
* 可能需要大量迭代；使用 maxiter(1000) 以确保更紧的收敛。
trop y d, panelvar(id) timevar(t) fixedlambda(0.4 0.3 0.006) maxiter(1000) bootstrap(0)
```

输出：

```
------------------------------------------------------------------------------
Triply Robust Panel Estimator                   Number of obs      =    5,328
Method: twostep                                 Number of units    =      111
                                                Time periods       =       48
                                                Treated obs        =       29
------------------------------------------------------------------------------
             |      ATT
-------------+----------------------------------------------------------------
           d |  -0.016024
------------------------------------------------------------------------------
Lambda: time = 0.400, unit = 0.300, nn = 0.006 (fixed)
Convergence: No (1000 iterations)
Note: SE/CI require bootstrap(); re-run with bootstrap(200).
------------------------------------------------------------------------------
```

> **收敛说明。** 当 `lambda_nn` 极小（如 0.006）时，低秩因子矩阵 L 可能有许多非零奇异值，交替最小化在大面板上收敛较慢。严格的 `tol(1e-6)` 默认值可能无法在 `maxiter(1000)` 内达到；但点估计在小数点后三位已经稳定。为获得更快的完全收敛估计，可以 (i) 增大 `lambda_nn`（如 `fixedlambda(0.4 0.3 0.1)` 约 110 次迭代收敛，τ = -0.006672）或 (ii) 放宽容差 `tol(1e-4)`。大面板上使用 `bootstrap(0)` 进行快速探索性分析，然后通过 `trop_bootstrap` 或重新运行 `bootstrap(200)` 获取推断结果。

**可用数据集**（通过 `trop_data` 加载）：

| 文件 | 描述 | N | T |
|------|------|---|---|
| `cps_logwage.dta` | CPS 州级对数工资（最低工资处理） | 50 | 40 |
| `cps_urate.dta` | CPS 州级失业率（最低工资处理） | 50 | 40 |
| `pwt_loggdp.dta` | Penn World Tables 对数 GDP（民主化转型） | 111 | 48 |
| `germany_gdp.dta` | Abadie & Gardeazabal (2003) 西德 GDP | 17 | 44 |
| `basque_gdp.dta` | Abadie (2003) 巴斯克地区 GDP | 18 | 43 |
| `smoking_packs.dta` | 加州 Prop 99 香烟消费 | 39 | 31 |

> `germany_gdp`、`basque_gdp` 和 `smoking_packs` 的 `d = 0`（全程无处理）——它们是原始结果面板，用于半合成模拟（如 Athey et al. 2025 表 1 所用）。运行 `trop` 前需自行指定处理指示变量。

## 推荐工作流

```
步骤 1: 验证     →  trop_validate depvar treatvar, panelvar() timevar()
步骤 2: 估计     →  trop depvar treatvar, panelvar() timevar()
步骤 3: 诊断     →  estat summarize / estat weights / estat loocv / estat factors
步骤 4: 推断     →  trop_bootstrap, nreps(1000)    — 或 —    步骤 2 中使用 bootstrap()
```

**步骤 1 — 验证数据结构。** `trop_validate` 在估计前检查面板平衡性、处理模式、缺失数据和最小样本要求。此步骤也会在 `trop` 内部自动执行。

```stata
trop_validate y d, panelvar(id) timevar(t)
```

**步骤 2 — 估计 ATT。** 选择 `method(twostep)`（默认）用于异质性效应，或 `method(joint)` 用于更快的同质性估计。LOOCV 自动选择超参数。

| 方法 | 适用场景 | 速度 |
|------|----------|------|
| `twostep` | 异质性处理效应；一般性处理分配 | 较慢 |
| `joint` | 同质性处理效应；仅同时采用 | 较快 |

**步骤 3 — 诊断结果。**

| 子命令 | 检查内容 |
|--------|----------|
| `estat summarize` | 面板维度、处理模式、平衡性 |
| `estat weights` | 权重是否集中于少数单位/时期 |
| `estat loocv` | 收敛情况、失败率、选定的 lambda |
| `estat factors` | 有效秩、前几个奇异值的解释方差 |

**步骤 4 — 进行推断。** 使用内联 `bootstrap()` 或后估计 `trop_bootstrap`。Bootstrap 在处理层内对单位进行重抽样（算法 3）。

```stata
* 内联方式（同时运行估计 + bootstrap）
trop y d, panelvar(id) timevar(t) bootstrap(1000) seed(42)

* 后估计方式（使用已存储的估计结果）
trop_bootstrap, nreps(1000) seed(42)
```

## 教程

交互式 Jupyter Notebook 教程作为辅助材料包含在内：

```stata
trop_data 10_trop_stata, type(notebook)
```

下载的 `10_trop_stata.ipynb` 文件涵盖数据生成、两种方法的估计、所有 `estat` 诊断、预测及真实数据 CPS 示例。

## 命令列表

| 命令             | 描述                                     |
| ---------------- | ---------------------------------------- |
| `trop`           | 主估计命令（twostep 或 joint 方法）      |
| `trop_bootstrap` | 独立 bootstrap 推断（后估计）            |
| `trop_validate`  | 面板数据结构验证                         |
| `trop_check`     | 环境与安装检查                           |

后估计命令（在 `trop` 之后可用）：

| 命令             | 描述                                     |
| ---------------- | ---------------------------------------- |
| `estat`          | 诊断调度器（13 个子命令；见下文）         |
| `predict`        | 预测调度器（11 种类型；见下文）          |

## 选项

### trop 选项

**必需选项：**

| 选项                | 描述                   |
| ------------------- | ---------------------- |
| `panelvar(varname)` | 单位（面板）标识变量   |
| `timevar(varname)`  | 时间标识变量           |

**可选选项：**

| 选项                         | 描述                                                       | 默认值     |
| ---------------------------- | ---------------------------------------------------------- | ---------- |
| `method(string)`             | 估计方法：`twostep` / `joint`（或别名 `local` / `global`） | `twostep`  |
| `grid_style(string)`         | Lambda 网格样式：`default`、`fine` 或 `extended`          | `default`  |
| `lambda_time_grid(numlist)`  | 用户指定的 λ_time 网格                                     | 自动       |
| `lambda_unit_grid(numlist)`  | 用户指定的 λ_unit 网格                                     | 自动       |
| `lambda_nn_grid(numlist)`    | 用户指定的 λ_nn 网格                                       | 自动       |
| `fixedlambda(numlist)`       | 固定 (λ_time λ_unit λ_nn)；跳过 LOOCV                     | —          |
| `tol(real)`                  | 迭代估计的收敛容差                                         | `1e-6`     |
| `maxiter(integer)`           | 最大迭代次数                                               | `500`      |
| `bootstrap(integer)`         | Bootstrap 重复次数（0 = 不执行推断）                       | `200`      |
| `bsvariance(string)`         | Bootstrap 方差分母：`sample` (1/(B-1)) 或 `paper` (1/B, 算法 3) | `sample`   |
| `cimethod(string)`           | 主置信区间方法：`percentile`（算法 3 步骤 6）、`t` 或 `normal` | `bootstrap > 0` 时为 `percentile`，否则为 `t` |
| `seed(integer)`              | 随机数生成器种子                                           | `42`       |
| `level(cilevel)`             | 置信区间的置信水平                                         | `c(level)` |
| `verbose`                    | 显示详细诊断输出                                           | 关闭       |

**网格样式说明：**
- `default` — 6 × 6 × 5 = 180 网格组合，每个坐标下降循环 17 次评估
- `fine` — 7 × 7 × 7 = 343 组合，每个循环 21 次评估（中等精度）
- `extended` — 14 × 16 × 19 = 4,256 组合，每个循环 49 次评估（更精细搜索，更慢；含 DID/TWFE 角点）

**网格注意事项：**
- `lambda_time_grid()` 和 `lambda_unit_grid()` 必须为有限非负数列；Stata 缺失值（`.`）会在解析时被拒绝。
- `lambda_nn_grid()` 和 `fixedlambda()` 的第三个位置接受 `.` 表示 +∞（DID/TWFE 角点，L ≡ 0）。`default` 网格**不**包含此角点；使用 `grid_style(extended)` 或在自定义 `lambda_nn_grid()` 中添加 `.` 以允许 LOOCV 选择"无因子结构"机制（经典 DID / 合成控制）。
- **大面板性能：** 落在开区间 `(0, 0.1)` 的 `lambda_nn` 值（尤其是 `default` 网格里的 `0.01`）评估成本最高：`0 < lambda_nn < 0.1` 会走 FISTA 求解（内层上限抬到 50，每次内层迭代做一次完整的 T×N SVD），而 `lambda_nn = 0` 与 `lambda_nn ≥ 0.1` 只走廉价的闭式路径。在约 8,300 个控制格的规模下，单个此类内点候选实测每格比 `lambda_nn = 0` 慢数万倍（约 12,600 毫秒/格 对 约 0.2 毫秒/格），即单个候选就可能耗时数十小时。大面板建议自定义网格避开该开区间，例如 `lambda_nn_grid(0 1 10)`。

**其他可选选项：**

| 选项                         | 描述                                                       | 默认值     |
| ---------------------------- | ---------------------------------------------------------- | ---------- |
| `covariates(varlist)`        | 时不变协变量 X_i'γ 调整（论文 6.2 节公式 14）         | —          |
| `twostep_loocv(string)`      | Twostep LOOCV 策略：`cycling`（默认）或 `exhaustive`      | `cycling`  |
| `joint_loocv(string)`        | Joint LOOCV 策略：`cycling` 或 `exhaustive`（默认）      | `exhaustive` |
| `vlevel(integer)`            | 详细程度 (0-4): 0=静默, 1=简略, 2=详细, 3=调试, 4=跟踪  | `0`        |
| `singleunit(string)`         | 单 PSU 分层处理：`skip`（省略）、`centered`（总均值校正）  | `skip`     |
| `strata(varname)`            | Rao-Wu Bootstrap 的分层变量                               | —          |
| `psu(varname)`               | 初级抽样单位变量                                           | —          |
| `fpc(varname)`               | 有限总体校正变量                                           | —          |
| `nest`                       | 声明 PSU 嵌套于分层内                                     | 关闭       |
| `notiming`                   | 抑制耗时显示                                               | 关闭       |

### trop_bootstrap 选项

| 选项              | 描述                                                       | 默认值     |
| ----------------- | ---------------------------------------------------------- | ---------- |
| `nreps(integer)`  | Bootstrap 重复次数                                         | `1000`     |
| `level(real)`     | 置信水平（百分比，10–99.99）                               | `c(level)` |
| `seed(integer)`   | 随机数生成器种子                                           | `42`       |
| `maxiter(integer)`| 每次重复的最大迭代数                                       | `500`      |
| `tol(real)`       | 每次重复的收敛容差                                         | `1e-6`     |
| `verbose`         | 显示进度信息                                               | 关闭       |

## 存储结果

### 标量

*核心估计：*

| 标量            | 描述                                                       |
| --------------- | ---------------------------------------------------------- |
| `e(att)`        | ATT 点估计                                                 |
| `e(se)`         | Bootstrap 标准误                                           |
| `e(t)`          | t 统计量 (att/se)                                          |
| `e(pvalue)`     | 主置信区间的双侧 p 值                                     |
| `e(ci_lower)`   | 主置信区间下界（由 `cimethod()` 从下列三组候选中选定）     |
| `e(ci_upper)`   | 主置信区间上界                                             |
| `e(df_r)`       | `max(1, N_1 - 1)`，其中 `N_1 = e(N_treated_units)`；`N_1 < 2` 时缺失（回退到正态） |
| `e(mu)`         | 全局截距（仅 joint；twostep 时缺失）                      |

*置信区间候选：*

启用 bootstrap 时，以下三组候选对均写入 `e()`，下游代码可不重新估计即切换 `cimethod()`。

| 标量                        | 描述                                                     |
| --------------------------- | -------------------------------------------------------- |
| `e(ci_lower_t)` / `e(ci_upper_t)` | 使用 `e(se)` 和 t(`e(df_r)`) 参考分布的 t 包裹置信区间 |
| `e(pvalue_t)`               | t 包裹的双侧 p 值                                       |
| `e(ci_lower_normal)` / `e(ci_upper_normal)` | 使用 `e(se)` 和 N(0,1) 的正态包裹置信区间 |
| `e(pvalue_normal)`          | 正态包裹的双侧 p 值                                     |
| `e(ci_lower_percentile)` / `e(ci_upper_percentile)` | Bootstrap 经验 CDF 的百分位置信区间（算法 3 步骤 6） |

*调优参数：*

| 标量             | 描述                                  |
| ---------------- | ------------------------------------- |
| `e(lambda_time)` | 选定的 λ_time                         |
| `e(lambda_unit)` | 选定的 λ_unit                         |
| `e(lambda_nn)`   | 选定的 λ_nn                           |
| `e(loocv_score)` | 最优 LOOCV 得分 Q(λ̂)                |

*样本信息：*

| 标量                  | 描述                                                           |
| --------------------- | -------------------------------------------------------------- |
| `e(N_units)`          | 面板单位数 (N)                                                 |
| `e(N_periods)`        | 时间期数 (T)                                                   |
| `e(N_obs)`            | 总观测数                                                       |
| `e(N_treat)`          | 处理单位-时期**单元格**数 (W=1)；`e(N_treated_obs)` 的旧别名  |
| `e(N_treated)`        | `e(tau)` 的长度 = 处理单元格数；等于 `e(N_treated_obs)`       |
| `e(N_treated_obs)`    | 处理单位-时期**单元格**数 (W=1 计数)                          |
| `e(N_treated_units)`  | 曾受处理的**单位**数 (N_1)；算法 3 bootstrap 的聚类计数       |
| `e(N_control)`        | 控制组观测数                                                   |
| `e(N_control_units)`  | 从未受处理的单位数 (N_0)                                       |
| `e(T_treat_periods)`  | 处理期数                                                       |
| `e(bootstrap_reps)`   | Bootstrap 重复次数                                             |

*收敛：*

| 标量               | 描述                                     |
| ------------------ | ---------------------------------------- |
| `e(n_iterations)`  | 迭代次数                                 |
| `e(converged)`     | 收敛指示器 (1/0)                         |
| `e(n_obs_estimated)` | 成功估计的观测数（仅 twostep）         |
| `e(n_obs_failed)`  | 失败的观测数（twostep，若 > 0）          |

*LOOCV 诊断：*

| 标量                     | 描述                                          |
| ------------------------ | --------------------------------------------- |
| `e(loocv_n_valid)`       | 有效 LOOCV 评估次数                           |
| `e(loocv_n_attempted)`   | 尝试 LOOCV 评估次数（= 所有 D=0 单元格，论文公式 5） |
| `e(loocv_fail_rate)`     | LOOCV 失败率                                  |
| `e(loocv_used)`          | 是否执行了 LOOCV (1/0)                        |
| `e(seed)`                | 使用的随机数种子                              |

*网格信息：*

| 标量                   | 描述                                          |
| ---------------------- | --------------------------------------------- |
| `e(n_lambda_time)`     | λ_time 网格值数量                             |
| `e(n_lambda_unit)`     | λ_unit 网格值数量                             |
| `e(n_lambda_nn)`       | λ_nn 网格值数量                               |
| `e(n_grid_combinations)` | 笛卡尔网格总组合数                          |
| `e(n_grid_per_cycle)`  | 每个坐标下降循环的网格评估次数                |

*其他：*

| 标量                    | 描述                                  |
| ----------------------- | ------------------------------------- |
| `e(balanced)`           | 平衡面板指示器 (1/0)                  |
| `e(miss_rate)`          | 缺失数据比率                          |
| `e(alpha_level)`        | 置信区间的显著性水平                  |
| `e(effective_rank)`     | 因子矩阵的有效秩                      |
| `e(n_bootstrap_valid)`  | 有效 bootstrap 重复次数               |
| `e(data_validated)`     | 数据验证指示器 (1/0)                  |
| `e(loocv_rmse)`           | LOOCV RMSE = sqrt(Q(λ̂) / n_valid)    |
| `e(condition_number)`     | WLS 设计矩阵条件数                    |
| `e(bootstrap_fail_rate)`  | Bootstrap 失败率 (0 到 1)             |
| `e(n_covariates)`         | 协变量数量（无时为 0）                |
| `e(deff_weights)`         | pweight 的 Kish 设计效应              |

### 宏

| 宏                     | 描述                                          |
| ---------------------- | --------------------------------------------- |
| `e(cmd)`               | `"trop"`                                      |
| `e(cmdline)`           | 完整命令行                                    |
| `e(method)`            | `"twostep"` 或 `"joint"`                     |
| `e(grid_style)`        | `"default"`、`"fine"`、`"extended"` 或 `"custom"`      |
| `e(depvar)`            | 因变量名称                                    |
| `e(treatvar)`          | 处理变量名称                                  |
| `e(panelvar)`          | 面板变量名称                                  |
| `e(timevar)`           | 时间变量名称                                  |
| `e(vcetype)`           | `"Bootstrap"` 或 `""`                        |
| `e(bsvariance)`        | 实际使用的 Bootstrap 方差分母：`sample` 或 `paper` |
| `e(cimethod)`          | 主置信区间方法：`percentile`、`t` 或 `normal`；降级时为 `"percentile->t"` |
| `e(estat_cmd)`         | `"trop_estat"`                                |
| `e(treatment_pattern)` | 处理分配模式描述                              |
| `e(twostep_loocv)`     | Twostep LOOCV 策略：`cycling` 或 `exhaustive` |
| `e(joint_loocv)`       | Joint LOOCV 策略：`cycling` 或 `exhaustive`   |
| `e(covariates)`        | 空格分隔的协变量名称                          |
| `e(spec_string)`       | 用于重现的设定字符串                          |
| `e(strata_var)`        | 分层变量（仅调查设计）                        |
| `e(psu_var)`           | PSU 变量（仅调查设计）                        |
| `e(fpc_var)`           | FPC 变量（仅调查设计）                        |
| `e(bootstrap_type)`    | Bootstrap 类型：`standard` 或 `rao_wu`        |

### 矩阵

| 矩阵                     | 描述                                                                |
| ------------------------ | ------------------------------------------------------------------- |
| `e(b)`                   | 系数向量 (1×1, ATT)                                                |
| `e(V)`                   | 方差-协方差矩阵 (1×1；需要 bootstrap)                              |
| `e(alpha)`               | 单位固定效应 (N×1)；行名为估计样本中 `e(panelvar)` 排序后的唯一值（已清洗为合法 Stata 矩阵标识符） |
| `e(beta)`                | 时间固定效应 (T×1)；行名为估计样本中 `e(timevar)` 排序后的唯一值（已清洗为合法 Stata 矩阵标识符） |
| `e(factor_matrix)`       | 低秩因子矩阵 L (T×N)                                              |
| `e(tau)`                 | 逐单元格处理效应 (N_treated×1)；两种方法均填充。对 `joint`，向量携带复制的标量 `tau`，因此 `mean(e(tau)) == e(att)` 对两种方法均精确成立 |
| `e(tau_matrix)`          | 处理效应排列为 T×N 面板形状矩阵，未处理单元格为 `.`（当面板元数据可用时） |
| `e(converged_by_obs)`    | 逐处理单元格收敛标志（`1` 收敛，`0` 达到 `maxiter()`，`-1` 求解器错误）；仅 twostep |
| `e(n_iters_by_obs)`      | 逐处理单元格迭代次数；仅 twostep                                   |
| `e(bootstrap_estimates)` | Bootstrap 分布 (B×1；需要 bootstrap)                               |
| `e(theta)`               | 时间权重（仅 twostep）                                             |
| `e(omega)`               | 单位权重（仅 twostep）                                             |
| `e(delta_time)`          | 时间权重（仅 joint）                                               |
| `e(delta_unit)`          | 单位权重（仅 joint）                                               |
| `e(lambda_time_grid)`    | Lambda 时间网格值                                                  |
| `e(lambda_unit_grid)`    | Lambda 单位网格值                                                  |
| `e(lambda_nn_grid)`      | Lambda 核范数网格值                                                |
| `e(gamma)`               | 协变量系数 (1×p；仅使用 `covariates()` 时)                        |
| `e(lambda_grid)`         | Lambda 网格笛卡尔积 (K×3)                                         |
| `e(cv_curve)`            | 网格点的 LOOCV 得分 (K×4)                                         |

## 后估计

### estat 子命令

| 子命令              | 缩写         | 描述                                                         |
| ------------------- | ------------ | ------------------------------------------------------------ |
| `estat summarize`   | `sum`        | 样本结构与处理分配                                           |
| `estat vce`         |              | 方差-协方差矩阵显示                                         |
| `estat sensitivity` | `sens`       | 超参数敏感性分析                                             |
| `estat weights`     | `weight`     | 单位权重与时间权重诊断                                       |
| `estat bootstrap`   | `boot`       | Bootstrap 分布诊断                                           |
| `estat loocv`       |              | LOOCV 超参数选择诊断                                         |
| `estat factors`     |              | 因子矩阵 (L) SVD 分析                                       |
| `estat triplerob`   | `trip`       | 定理 5.1 三重稳健偏差界分解 (`|Δᵘ|₂ · |Δᵗ|₂ · |B|_*`)     |
| `estat distance`    | `dist`       | 单位距离分布诊断                                             |
| `estat mht`         |              | 多重假设检验校正                                             |
| `estat eventstudy`  | `es`         | 事件研究动态处理效应                                         |
| `estat pretrend`    | `pretest`    | 前趋势检验（所有处理前效应 = 0）                             |
| `estat table`       |              | 导出结果为格式化表格（LaTeX/Markdown/CSV）                   |

### predict 类型

在 `trop` 之后，使用 `predict newvar, type` 生成预测：

| 类型             | 描述                                    |
| ---------------- | --------------------------------------- |
| `y0`             | 反事实结果 Y(0) **[默认]**              |
| `y1`             | 反事实结果 Y(1)                         |
| `te`             | 处理效应（仅处理观测）                  |
| `residuals`      | 残差 Y - Y(0) - τ·W                    |
| `fitted`         | 拟合值 Ŷ = Y(0) + τ·W                  |
| `alpha`          | 单位固定效应                            |
| `beta`           | 时间固定效应                            |
| `mu`             | 全局截距（仅 joint）                    |
| `xb`             | 线性预测（等价于 `y0`）               |
| `att`            | 处理效应（`te` 的别名）               |
| `counterfactual` | 反事实 Y(0)（`y0` 的别名）            |

## 方法论

### TROP 估计量

TROP 估计量将控制潜在结果建模为 $Y_{it}(0) = \alpha_i + \beta_t + L_{it} + \epsilon_{it}$，其中 $\alpha_i$ 为单位固定效应，$\beta_t$ 为时间固定效应，$L_{it}$ 为低秩因子成分，$\epsilon_{it}$ 为特质噪声。

对于每个处理单位-时间对 $(i,t)$，估计量通过求解加权核范数惩罚回归来预测反事实结果（论文公式 2）：

$$(\hat{\alpha}, \hat{\beta}, \hat{\mathbf{L}}) = \arg\min_{\alpha, \beta, \mathbf{L}} \sum_{j=1}^{N} \sum_{s=1}^{T} \theta_s^{i,t} \omega_j^{i,t} (1-W_{js})(Y_{js} - \alpha_j - \beta_s - L_{js})^2 + \lambda_{nn} \|\mathbf{L}\|_*$$

其中权重呈指数衰减（公式 3）：

$$\theta_s^{i,t} = \exp(-\lambda_{\text{time}} \cdot |t - s|), \qquad \omega_j^{i,t} = \exp(-\lambda_{\text{unit}} \cdot \text{dist}_{-t}^{\text{unit}}(j, i))$$

单位距离度量共同控制期上结果差异的 RMSE：

$$\text{dist}_{-t}^{\text{unit}}(j,i) = \left(\frac{\sum_{u} \mathbf{1}\{u \neq t\}(1-W_{iu})(1-W_{ju})(Y_{iu}-Y_{ju})^2}{\sum_{u} \mathbf{1}\{u \neq t\}(1-W_{iu})(1-W_{ju})}\right)^{1/2}$$

处理效应为 $\hat{\tau}_{it} = Y_{it} - \hat{\alpha}_i - \hat{\beta}_t - \hat{L}_{it}$。

这一框架将 DID、SC、MC 和 SDID 均作为特殊情形。当 $\lambda_{nn} = \infty$ 且 $\omega_j = \theta_s = 1$ 时，恢复 DID/TWFE 估计量。当 $\omega_j = \theta_s = 1$ 且 $\lambda_{nn} < \infty$ 时，恢复 MC 估计量。当 $\lambda_{nn} = \infty$ 配合特定单位和时间权重时，恢复 SC 和 SDID。

### 三重稳健性质

偏差满足乘积形式界（定理 5.1）：

$$\left|\mathbb{E}[\hat{\tau} - \tau \mid \mathbf{L}]\right| \leq \|\Delta^{\mathbf{u}}(\omega, \Gamma)\|_2 \times \|\Delta^{\mathbf{t}}(\theta, \Lambda)\|_2 \times \|B\|_*$$

其中 $\Delta^{\mathbf{u}}$ 为单位不平衡，$\Delta^{\mathbf{t}}$ 为时间不平衡，$B$ 刻画回归调整的误设定。当以下三个条件中**任意一个**成立时，估计量具有一致性（推论 1）：

1. 单位因子载荷平衡 ($\|\Delta^{\mathbf{u}}\|_2 \approx 0$)
2. 时间因子载荷平衡 ($\|\Delta^{\mathbf{t}}\|_2 \approx 0$)
3. 回归调整设定正确 ($\|B\|_* \approx 0$)

这一*乘积*形式的偏差界严格优于 DID、SC 和 SDID 所依赖的*加法*形式界，赋予 TROP 更强的稳健性。

### 调优参数选择

三元组 $(\lambda_{\text{time}}, \lambda_{\text{unit}}, \lambda_{nn})$ 通过留一交叉验证选择，最小化（公式 5）：

$$Q(\lambda) = \sum_{i=1}^{N} \sum_{t=1}^{T} (1 - W_{it})(\hat{\tau}_{it}(\lambda))^2$$

这等价于选择在控制观测上预测控制潜在结果的样本外平方误差最小的调优参数。网格搜索使用坐标下降（算法 1）：轮流优化每个参数同时固定另两个在其最新选定值，循环直至收敛。

### 估计方法

- **Twostep**（算法 2）：逐观测估计，允许异质性处理效应。对每个处理对 $(i,t)$，模型如同 $(i,t)$ 是唯一处理观测一样进行拟合。ATT 为 $\hat{\tau} = \frac{1}{\sum_{i,t} W_{it}} \sum_{i,t} W_{it} \hat{\tau}_{it}$。
- **Joint**（注释 6.1）：加权最小二乘，估计单一标量处理效应 $\tau$，假设所有处理单位-时间对具有同质性效应。使用所有处理观测共享的全局权重。当同质性假设成立时计算效率更高。

### Bootstrap 推断

方差估计遵循算法 3：分层单位块 Bootstrap，分别对 $N_0$ 个控制单位和 $N_1$ 个处理单位进行有放回重抽样。对于每次重复 $b = 1, \ldots, B$，在重抽样数据上重复完整估计过程（含 LOOCV，若适用）以获得 $\hat{\tau}^{(b)}$。Bootstrap 方差为

$$\hat{V}_{\tau} = \frac{1}{B - 1} \sum_{b=1}^{B} (\hat{\tau}^{(b)} - \bar{\hat{\tau}})^2$$

（默认使用 Bessel 校正的样本方差）。论文原始的总体方差分母 $1/B$ 可通过 `bsvariance(paper)` 选择；在 $B = 200$ 时两者差异不超过 0.5%。

**参考分布。** 由于算法 3 对*单位*进行重抽样，控制小样本自由度的聚类计数为 $N_1 = $ `e(N_treated_units)`。因此 `trop` 在 $N_1 \geq 2$ 时使用 $t(N_1 - 1)$ 分布，否则回退到 $\mathcal{N}(0,1)$。`e(df_r)` 为 `max(1, N_1 - 1)` 或缺失（正态回退）。使用处理*单元格*数推导的 `df` 会在 $T_\text{post}$ 增长时膨胀显著性。

**三组置信区间，一组为主。** 每次 Bootstrap 运行产生三组置信区间候选：

1. **百分位置信区间** — Bootstrap 经验 CDF 的 $\alpha/2$ 和 $1-\alpha/2$ 分位数（算法 3 步骤 6）。
2. **t 包裹置信区间** — `att ± invttail(df_r, α/2) · se`。
3. **正态包裹置信区间** — `att ± invnormal(1 − α/2) · se`。

`cimethod(percentile | t | normal)` 选项指定哪组提升为主置信区间 `e(ci_lower)` / `e(ci_upper)`。当 `bootstrap > 0` 时默认为 `percentile`（论文推荐的无分布假设区间）；当 `bootstrap(0)` 与 `cimethod(percentile)` 同时使用时，解析器降级到 `t` 并在 `e(cimethod)` 中记录为 `"percentile->t"`。三组候选对始终持久化在 `e()` 上，下游分析者可不重新估计即切换 `cimethod()`。

## 架构

本包采用四层设计以兼顾性能与数值精度：

```
┌─────────────────────────────────────────────────┐
│  Stata 用户界面  (trop.ado)                      │
│  - 语法解析、选项处理                            │
├─────────────────────────────────────────────────┤
│  Mata 接口层                                     │
│  - 输入验证与数据转换                            │
│  - e() 结果存储                                  │
├─────────────────────────────────────────────────┤
│  C 桥接插件                                      │
│  - 指针转换、错误码映射                          │
├─────────────────────────────────────────────────┤
│  Rust 核心                                       │
│  - 距离矩阵、权重计算                            │
│  - LOOCV 网格搜索、SVD 估计                      │
│  - 分层单位块 Bootstrap                          │
└─────────────────────────────────────────────────┘
```

所有数值计算（LOOCV、SVD、Bootstrap）均在 Rust 中执行以获得速度和精度。Mata 层处理数据验证与结果存储。用户无需 Rust 工具链——预编译插件已包含在内。

## 数值鲁棒性设计选择

以下记录了 `trop` 中九项实现选择。每一项均由回归测试固定，使未来重构不会悄然退化；且每一项均由第一性原理关切驱动而非个人偏好：

1. **FISTA 自适应重启已禁用** (`rust/src/estimation.rs`)。核范数近端求解器**不**使用 O'Donoghue & Candès (2015) 的单调梯度重启方案。虽然重启准则 `⟨y_k − x_k, x_k − x_{k−1}⟩ > 0` 理论上能消除动量振荡，但在小面板上触发过于激进，阻止收敛。Python 参考实现（`diff-diff` v3.1.1）同样不使用重启，故我们禁用以维持数值一致性。测试 `tests/test_fista_restart_stability.do` 验证 FISTA 求解器在多种 `lambda_nn` 值下无需重启仍保持稳定。

2. **加权最小二乘步骤使用 LAPACK `dgelsd`** (`rust/src/estimation.rs`)。基于 SVD 的最小范数求解器在设计矩阵因权重向量置零整行/列而秩亏时，返回 Moore–Penrose 伪逆解。SVD 截断容差 `rcond` 为 `max(ε · max(m, n), 1e-12)` — 下限在最小基准面板（Basque N = 17, 西德 N = 16）上稳定 $\hat\alpha / \hat\beta$ 而不扰动 $\hat\tau$。由 `tests/test_dgelsd_rank_deficient_wls.do` 固定。

3. **`UnitDistanceCache`** (`rust/src/distance.rs`)。成对 $\sum_u (Y_{iu} − Y_{ju})^2$ 和预先计算一次；每个留-$t$-外距离 $\text{dist}_{-t}(j, i)$ 随后为 O(1) 减法而非 O(T) 重扫。缓存等价性由 `tests/test_unit_distance_cache_equivalence.do` 验证至 < 10⁻¹⁰。

4. **确定性 LOOCV 平局打破规则** (`rust/src/loocv.rs`, `better_candidate`)。当两个 `(lambda_time, lambda_unit, lambda_nn)` 三元组的得分差异在 `TIE_TOL = 1 × 10⁻¹⁰` 以内时，`trop` 优先选择更大的 `lambda_nn`，然后更小的 `lambda_time`，然后更小的 `lambda_unit`。ULP 级别的 BLAS 差异否则可能在不同平台间翻转 `argmin Q(λ)`。由 `tests/test_loocv_tiebreak_determinism.do` 固定。

5. **推断参考分布** (`ado/trop.ado`, `mata/trop_ereturn_store.mata`)。Bootstrap 以分层方式重抽样单位（算法 3 步骤 3），因此控制小样本参考自由度的聚类计数为 $N_1$——曾受处理的单位数——而非处理单元格数。`trop` 因此在 $N_1 \geq 2$ 时使用 $t(N_1 - 1)$，否则回退到标准正态。主置信区间在启用 Bootstrap 时默认为论文指定的百分位区间；`cimethod()` 从三组候选（percentile, t, normal）中重新选定主对。由 `tests/test_inference_df_is_treated_units.do` 和 `tests/test_cimethod_option.do` 固定。

6. **Joint 方法的同时采用守卫** (`rust/src/loocv.rs::check_simultaneous_adoption`)。Joint 估计量的全局权重矩阵 $\delta$ 依赖于共享的 `treated_periods` 计数；仅当每个处理单位在相同时期 $T_1$ 进入处理并保持至面板结束时才有良好定义（论文注释 6.1）。Stata 前端已拒绝交错 $D$ 用于 `method(joint)`；作为纵深防御，每个 Joint C-ABI 入口（`stata_estimate_joint`、`stata_bootstrap_trop_variance_joint`、`stata_loocv_grid_search_joint`、`stata_loocv_cycling_search_joint` 及 `_weighted` 变体）现在对交错/非吸收 $D$ 短路返回 `TropError::InvalidDimension`，而非静默地错误计算 $T_1$。由 `rust/src/loocv.rs` 中五个单元测试固定。

7. **λ_nn = 0 的封闭形式** (`rust/src/estimation.rs`)。当 $\lambda_{nn}=0$ 时，论文公式 2 化简为 $\hat{L}_{t,i} = Y_{t,i} - \hat\alpha_i - \hat\beta_t$（在加权支撑 $W>0$ 上），而 $\hat{L}$ 在支撑外（$W=0$）不可识别。实现因此在有效单元格上将 $\hat{L}$ 设为封闭形式残差，在无效单元格上保留前一迭代值；debug 构建后条件验证后者不变式。由 `estimation::tests::test_lambda_nn_zero_closed_form_preserves_invalid_cells` 固定。

8. **LOOCV 和 Bootstrap 统一的 5% 失败率阈值** (`mata/trop_ereturn_store.mata`, `mata/trop_rust_interface.mata`)。`_trop_display_bootstrap_warnings` 和 Mata 端 `check_loocv_fail_rate()` 现在在 **5%** 时发出警告，在 **50%** 时以 `rc ∈ {498, 504}` 中止。5% 阈值足够紧，以至于在约 1,000 个 `D=0` 单元格的面板上，大约 50 个失败的留一拟合会浮出水面——这一量级的失败率可能将选定的 `(lambda_time, lambda_unit, lambda_nn)` 偏离 `Q(λ)` 的真实 argmin（论文公式 5）。Bootstrap 阶段相同阈值意味着 200 次重复中 11 次失败不再隐藏在历史的 10% 门槛之后。两个失败率均通过 `e(loocv_fail_rate)` / `e(bootstrap_fail_rate)` 暴露供下游诊断。由 `tests/test_bootstrap_fail_rate_threshold.do`、`tests/test_ereturn_fail_rate_coverage.do` 和 `tests/test_loocv_fail_rate_threshold.do` 固定。

9. **`e(alpha)` 和 `e(beta)` 的原始 ID 行名** (`ado/_trop_attach_idnames.ado`)。估计完成后，`e(alpha)` (N×1) 和 `e(beta)` (T×1) 的矩阵行名被重写为估计样本中用户提供的 `panelvar` / `timevar` 排序后的唯一值。由于 `egen ... = group()` 和 `levelsof` 均返回排序后的唯一值，`e(alpha)` 的第 $i$ 行对应第 $i$ 个唯一面板标识符；插件索引 1..N 仍可通过标量访问（`e(alpha)[i, 1]`）获取。标识符已清洗为合法 Stata 矩阵名（字母/数字/下划线，≤ 32 字符，非数字开头），因此数字 ID 如 `1989, 1990, ...` 在 `matrix list e(beta)` 中显示为 `_1989, _1990, ...`。由 `tests/test_alpha_beta_rownames.do` 固定。

**当前版本范围之外。** 时变协变量 $X_{it}\beta$（当前实现支持时不变协变量 $X_i'\gamma$，见论文 6.2 节公式 14，但不支持完整面板时变回归量）；以及 `method(joint)` 下的切换处理模式。这些超出当前范围，计划在未来版本中实现。

## 已知限制

### 功能限制

| 限制 | 说明 | 状态 |
|------|------|------|
| 协变量维度约束 | 要求 p < min(N, T)，其中p为协变量数 | 设计约束 |
| 时变协变量 | 不支持 X_{it}β 形式的时变协变量调整 | 计划中 |
| 交错处理（Joint方法）| `method(joint)` 要求所有处理单位同期进入处理 | 论文约束 |
| 交错处理（Twostep方法）| `method(twostep)` 隐式允许交错但缺乏完整理论保证 | 需谨慎使用 |

### 性能与内存

| 约束 | 说明 | 缓解方案 |
|------|------|----------|
| LOOCV计算复杂度 | 大面板(N>200, T>50)可能耗时较长 | 使用 `fixedlambda()` 或 `grid_style(default)` |
| Bootstrap内存消耗 | B次迭代需要 O(B·N·T) 内存 | 减少 `bootstrap()` 次数 |
| 距离矩阵存储 | O(N²) 单位距离矩阵 | 面板规模控制在 N<500 |

## 故障排除

### 常见错误

| 错误码 | 含义 | 解决方案 |
|--------|------|----------|
| 3 | 无效的处理分配 | 检查处理变量是否为0/1二值变量 |
| 4 | 无有效控制组观测 | 确保数据中存在处理前的控制观测 |
| 5 | 估计未收敛 | 尝试增加 `maxiter()` 或放松 `tol()` |
| 8 | 面板结构无效 | 检查面板标识变量和时间变量的唯一性 |
| 12 | SVD分解失败 | 检查数据中是否存在完全共线的协变量 |
| 13 | 单PSU分层 | 使用 `singleunit(centered)` 选项 |

### 数值稳定性问题

**症状**：LOOCV失败率高(>10%)或估计未收敛

**诊断步骤**：
1. 检查 `e(loocv_fail_rate)` — 若 >0.10 表示网格搜索质量受损
2. 检查 `e(condition_number)` — 若 >1e10 表示设计矩阵病态
3. 检查 `e(effective_rank)` — 低秩可能表示因子模型过度拟合

**解决方案**：
- 减少协变量数量或检查多重共线性
- 使用 `grid_style(default)` 代替 `grid_style(extended)`
- 考虑对极端值进行winsorize处理

### 性能优化建议

| 场景 | 建议 |
|------|------|
| 大面板(N>100, T>30) | 使用 `fixedlambda()` 跳过LOOCV |
| Bootstrap耗时 | 减少 `bootstrap(200)` 初步探索，确认后增加至500 |
| 内存不足 | 减少面板规模或分子样本估计 |

## 第三方命令集成

### estout / esttab 集成

`trop` 按照 Stata 标准将结果存储在 `e()` 中。如需通过 `estout` / `esttab` 导出结果，可构建兼容的系数向量：

```stata
* 运行 TROP 估计
trop y d, panelvar(id) timevar(t) fixedlambda(0.1 0 0.9) bootstrap(200) seed(42)

* 方式 1：直接使用 e(b) 和 e(V)（已经是 1x1 矩阵）
eststo trop_model

* 方式 2：添加自定义标量以构建更丰富的表格
estadd scalar att = e(att)
estadd scalar se = e(se)
estadd scalar pvalue = e(pvalue)
estadd scalar ci_lo = e(ci_lower)
estadd scalar ci_hi = e(ci_upper)
estadd local method = e(method)

* 导出为 LaTeX
esttab trop_model, stats(att se pvalue ci_lo ci_hi method) ///
    title("TROP Estimation Results")

* 导出为 CSV
esttab trop_model using results.csv, stats(att se pvalue) csv replace
```

### coefplot 集成

`trop` 的估计结果可通过 `coefplot` 进行可视化：

```stata
* 标准系数图（绘制 e(b) 配合 e(V)）
trop y d, panelvar(id) timevar(t) fixedlambda(0.1 0 0.9) bootstrap(200) seed(42)
coefplot, title("TROP ATT Estimate")
```

对于事件研究图，使用内置的 `estat eventstudy` 命令：

```stata
* 使用内置绘图的事件研究
estat eventstudy, graph

* 或提取数据用于自定义 coefplot 格式
estat eventstudy, nograph
matrix es = r(event_effects)
```

### 多模型比较

```stata
* 比较 twostep 与 joint 方法
trop y d, panelvar(id) timevar(t) fixedlambda(0.1 0 0.9) bootstrap(200) seed(42)
eststo twostep_model

trop y d, panelvar(id) timevar(t) method(joint) fixedlambda(0.1 0 0.9) bootstrap(200) seed(42)
eststo joint_model

esttab twostep_model joint_model, ///
    mtitles("Twostep" "Joint") ///
    stats(att se pvalue N_obs)
```

## 性能调优

### Bootstrap 并行计算

TROP 的 Bootstrap 方差估计使用 Rust 后端的 Rayon 并行库，默认利用全部可用 CPU 核心。

**线程控制：** 通过环境变量 `RAYON_NUM_THREADS` 控制并行线程数：

```stata
* 在调用 trop 前设置（限制为 4 线程）
shell export RAYON_NUM_THREADS=4

* 或在 shell 启动脚本中设置
* ~/.bashrc: export RAYON_NUM_THREADS=4
```

**性能参考：**

| 面板规模 | Bootstrap(B) | 大致耗时 |
|----------|--------------|----------|
| 小面板 (N<50, T<20) | 200 | < 30 秒 |
| 中等面板 (N=50-200, T=20-50) | 500 | 2-10 分钟 |
| 大面板 (N>200, T>50) | 200 | 10-60 分钟 |

**内存估算：** 约需 `8 x N x T x B` 字节。例如 N=100, T=50, B=500 约需 ~200 MB。

**大面板使用建议：**
- 使用 `fixedlambda()` 跳过 LOOCV（最耗时的步骤）
- 先用 `bootstrap(200)` 进行探索性分析
- 确认结果稳定后增加至 `bootstrap(500)` 或 `bootstrap(1000)`
- 监控 `e(loocv_fail_rate)` — 高失败率表示数值计算困难

## 参考文献

Athey, S., Imbens, G., Qu, Z., & Viviano, D. (2025). Triply robust panel estimators. *arXiv preprint arXiv:2508.21536*.

## 作者

**Stata 实现：**

- **蔡宣宇 (Xuanyu Cai)**，澳门城市大学
  邮箱：[xuanyuCAI@outlook.com](mailto:xuanyuCAI@outlook.com)
- **许文立 (Wenli Xu)**，澳门城市大学
  邮箱：[wlxu@cityu.edu.mo](mailto:wlxu@cityu.edu.mo)

**方法论：**

- **Susan Athey**，斯坦福大学
- **Guido Imbens**，斯坦福大学
- **Zhaonan Qu**，哥伦比亚大学
- **Davide Viviano**，哈佛大学

## 许可证

AGPL-3.0 许可证。详见 [LICENSE](LICENSE)。

## 引用

如在已发表的研究中使用本包，请同时引用方法论论文与 Stata 实现：

**APA 格式：**

> Cai, X., & Xu, W. (2025). *trop: Stata module for Triply Robust Panel estimation* [Computer software]. GitHub. https://github.com/gorgeousfish/TROP
>
> Athey, S., Imbens, G., Qu, Z., & Viviano, D. (2025). Triply robust panel estimators. *arXiv preprint arXiv:2508.21536*.

**BibTeX：**

```bibtex
@software{trop2025stata,
  title={trop: Stata module for Triply Robust Panel estimation},
  author={Xuanyu Cai and Wenli Xu},
  year={2025},
  version={1.2.0},
  url={https://github.com/gorgeousfish/TROP}
}

@article{athey2025triply,
  title={Triply robust panel estimators},
  author={Athey, Susan and Imbens, Guido and Qu, Zhaonan and Viviano, Davide},
  journal={arXiv preprint arXiv:2508.21536},
  year={2025}
}
```

## 另见

- 原始论文：Athey, Imbens, Qu & Viviano: https://arxiv.org/abs/2508.21536
- 相关 Stata 包：[`sdid`](https://github.com/Daniel-Pailanir/sdid)（合成 DID）、[`diddesign`](https://github.com/gorgeousfish/diddesign)（Double DID）
