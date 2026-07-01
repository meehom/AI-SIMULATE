# configs

这里存放仿真实验的输入配置，按 4 类组织：

- `system/`：系统侧输入，如单卡算力、HBM、互联带宽、静态时延
- `workload/`：工作负载输入，如模型、模式、精度、batch、输入输出长度
- `strategy/`：并行策略输入，如 TP / PP / DP 配置
- `experiments/`：一次完整实验的组合入口，负责引用前三类配置

## 最小组合关系

一个 experiment 由下面 3 类配置组合而成：

- system
- workload
- strategy

推荐让 experiment 文件只负责“引用”，不要重复拷贝底层参数。

## 当前 GB300 基线示例

本仓库当前补充的首个基线实验为：

- 模型：DeepSeek V3
- 模式：推理
- 精度：FP8
- 规模：GB300 NVL72 全 72 卡
- 请求形状：BS1 / input 4096 / output 512
- 并行策略：TP8 × PP9 × DP1

对应文件：

- `system/chip_config/GB300.json`
- `system/topo_config/GB300.json`
- `workload/deepseek_v3_inference_fp8_bs1_in4k_out512.json`
- `strategy/gb300_tp8_pp9_dp1.json`
- `experiments/gb300_deepseek_v3_inference_fp8_tp8_pp9_bs1_in4k_out512.json`

## 命名建议

建议把关键信息直接写入文件名，便于后续做 sweep 和横向比较，例如：

- workload：`<model>_<mode>_<precision>_bs<batch>_in<input>_out<output>.json`
- strategy：`<system>_tp<tp>_pp<pp>_dp<dp>.json`
- experiment：`<system>_<model>_<mode>_<precision>_tp<tp>_pp<pp>_bs<batch>_in<input>_out<output>.json`

## experiment 示例

```json
{
  "name": "gb300_deepseek_v3_inference_fp8_tp8_pp9_bs1_in4k_out512",
  "system": {
    "chip_config": "system/chip_config/GB300.json",
    "topo_config": "system/topo_config/GB300.json"
  },
  "workload_config": "workload/deepseek_v3_inference_fp8_bs1_in4k_out512.json",
  "strategy_config": "strategy/gb300_tp8_pp9_dp1.json"
}
```

当前目标是先建立一套简单、可扩展、可组合的配置输入约定；后续如果仿真器需要更多字段，再逐步扩展各类 schema。
