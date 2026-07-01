# AI Workload Simulate

AI Workload Simulate 是一个面向大模型推理/训练工作负载的仿真项目，目标是在**不同芯片配置**、**不同网络拓扑（topology）**、**不同并行策略**下，对主流模型的性能进行建模、分析与对比。

当前项目主要采用 **Roofline** 方法进行建模，用于评估计算与带宽约束下的理论性能边界，并输出细粒度的仿真结果，帮助分析系统瓶颈与优化方向。

## 项目目标

本项目聚焦以下问题：

- 不同芯片配置对大模型性能的影响
- 不同网络拓扑对通信开销和整体吞吐的影响
- 不同并行方式下的扩展性与瓶颈分析
- 不同主流大模型在给定系统配置下的性能对比
- 基于 Roofline 模型的快速性能估计与算子级分析

## 关注模型

当前重点覆盖以下主流模型：

- DeepSeek V3
- DeepSeek V4
- Qwen-235B
- Qwen-3.5

后续可扩展到更多开源或闭源大模型架构。

## 仿真维度

项目主要从以下几个维度进行组合仿真：

### 1. 芯片配置

包括但不限于：

- 单卡计算能力
- HBM / 显存带宽
- 显存容量
- 卡间互联带宽
- 芯片数量与集群规模

### 2. 网络拓扑配置

包括但不限于：

- 单机多卡互联
- 跨机互联
- Fat-tree / Ring / Mesh / 其他定制拓扑
- 不同拓扑下的带宽、时延与拥塞影响

### 3. 并行方式

支持分析不同并行策略下的性能表现，例如：

- Tensor Parallel (TP)
- Pipeline Parallel (PP)
- Data Parallel (DP)
- Expert Parallel (EP)
- Sequence Parallel (SP)
- 多种并行策略的混合并行

## 建模方法

当前项目直接采用 **Roofline 建模**。

Roofline 模型通过结合：

- 峰值算力
- 峰值带宽
- 算术强度（Arithmetic Intensity）

来估计算子或整体工作负载的性能上界，从而帮助判断当前瓶颈主要受限于：

- **计算能力**
- **存储/访存带宽**
- **通信带宽或通信时延**

这种方法适合在系统设计初期快速评估不同硬件与并行配置组合的相对优劣。

## 当前输出结果

每次仿真会输出以下信息：

### 核心性能指标

- 单卡时延
- 单卡吞吐
- 端到端性能估计
- 不同并行策略下的性能对比结果

### 算子级细节

- 单卡算子详细执行时延
- 算子级瓶颈分析
- 算子输入 shape
- 算子输出 shape

### 系统分析信息

- 计算开销拆解
- 访存开销拆解
- 通信开销拆解
- 关键瓶颈定位

## 推荐项目目录结构

项目建议按 **配置输入** 与 **仿真实现** 两条主线组织：

- `configs/`：放实验输入、系统参数、模型负载描述、并行策略配置
- `src/`：放仿真引擎、Roofline 建模逻辑、分析代码和 CLI

推荐目录如下：

```text
AI-simulate/
├── README.md
├── configs/
│   ├── system/          # 芯片、节点、互联、集群等系统配置
│   ├── workload/        # 模型、batch size、seq len、shape 等负载描述
│   ├── strategy/        # TP / PP / DP / EP / SP / 混合并行策略配置
│   └── experiments/     # 一次完整仿真实验的组合配置
├── src/
│   └── ai_simulate/
│       ├── cli/         # 命令行入口与参数解析
│       ├── core/        # 核心抽象、公共数据结构
│       ├── system/      # 芯片、内存、互联、网络拓扑建模
│       ├── workload/    # 模型结构、算子列表、shape 推导
│       ├── strategy/    # 并行切分、映射与通信策略逻辑
│       ├── roofline/    # Roofline 模型实现与性能上界估计
│       ├── simulator/   # 仿真主流程、调度与结果汇总
│       ├── analysis/    # 结果分析、瓶颈拆解、报表生成
│       └── utils/       # 通用工具函数
├── examples/            # 示例配置与示例运行
├── results/             # 仿真输出结果
├── tests/
│   ├── unit/            # 单元测试
│   └── integration/     # 集成测试
├── docs/                # 设计文档、建模说明
├── notebooks/           # 分析与可视化 notebook
└── scripts/             # 辅助脚本
```

### 为什么这样划分

这样划分的核心目的是避免 `configs/` 和 `src/` 出现“同名镜像重复”：

- `configs/system/` 对应的是**系统输入参数**
- `src/ai_simulate/system/` 对应的是**系统建模实现代码**
- `configs/workload/` 对应的是**模型与 shape 的工作负载描述**
- `src/ai_simulate/workload/` 对应的是**工作负载解析与算子建模逻辑**
- `configs/strategy/` 对应的是**实验使用的并行配置**
- `src/ai_simulate/strategy/` 对应的是**并行策略实现与分析逻辑**

也就是说：

- `configs/` 关注 **输入与实验组合**
- `src/` 关注 **实现与计算逻辑**

### 首个 GB300 实验配置示例

当前仓库已经补充了一条最小的 GB300 基线实验配置链路：

- 模型：DeepSeek V3
- 模式：推理
- 精度：FP8
- 硬件规模：GB300 NVL72 全 72 卡
- 请求形状：BS1 / input 4096 / output 512
- 并行策略：TP8 × PP9 × DP1

对应配置文件为：

- `configs/system/chip_config/GB300.json`
- `configs/system/topo_config/GB300.json`
- `configs/workload/deepseek_v3_inference_fp8_bs1_in4k_out512.json`
- `configs/strategy/gb300_tp8_pp9_dp1.json`
- `configs/experiments/gb300_deepseek_v3_inference_fp8_tp8_pp9_bs1_in4k_out512.json`

其中：

- `system/` 负责描述 GB300 单卡能力与 72 卡互联参数
- `workload/` 负责描述模型、精度、batch 和输入输出长度
- `strategy/` 负责描述 TP / PP / DP 并行切分
- `experiments/` 负责把前三类配置组合成一个完整实验入口

更多约定可参考 `configs/README.md`。

## 适用场景

本项目适用于：

- 大模型系统架构设计评估
- 芯片与集群配置选型
- 并行策略设计与对比
- 模型部署前的性能预估
- 算子级热点分析与优化方向研判

## 项目特点

- 面向超大模型工作负载仿真
- 同时关注硬件、网络与并行策略
- 基于 Roofline 的快速建模方法
- 支持输出从整机到算子级的细粒度结果
- 便于横向比较不同模型和系统配置

## 后续规划

后续可逐步补充：

- 更精细的通信建模
- 更真实的算子执行时间校准
- 不同模型结构的参数化描述
- 自动化配置扫描与结果汇总
- 可视化分析报表
- 与真实测量结果对齐验证
