# configs/system

这里存放系统侧输入配置，当前分成两类：

- `chip_config/`：单卡硬件能力
- `topo_config/`：多卡互联参数

## 当前字段约定

### 1. chip_config

用于 Roofline 建模，当前只保留最核心字段：

- 不同精度下的峰值算力（如 fp4 / fp8 / fp16 / fp32）
- HBM 容量
- HBM 带宽

### 2. topo_config

用于通信建模，当前只保留最核心字段：

- 卡数
- 单卡互联带宽
- 静态时延

## 当前示例

- `chip_config/GB300.json`：GB300 单卡算力与 HBM 参数
- `topo_config/GB300.json`：GB300 NVL72 的卡数、互联带宽与静态时延

## 说明

当前配置优先服务第一版仿真，目标是先保证字段简单、直接、够用。后续如果需要更细的系统建模，再逐步补充更复杂的拓扑或硬件字段。
