# Elastic AFD 与上游启动参数、特性对照

审查日期：2026-09-22。运行基线固定为 vLLM **0.26.0**、AFD 上游
**8bb14be66d9f940b1b132b141d986214c0210353**。同时查询了上游 main
**a1c9e06987a4fc7d191ba64afe9cf7634b98d3c9**：多出的 10 个提交主要涉及
NPU 模型、算子和构建；下述 GPU DeepSeek-V2-Lite README 与 DP2/TP1
graph+DBO 启动脚本内容未变。本次不升级运行基线。

## 推荐示例实际使用什么

以 [GPU recipe](../recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/README.md)
及其 [DP2/TP1 graph+DBO 脚本](../recipe/gpu/P2pNcclAFDConnector/deepseek_v2_lite/prefill_decode_colocation/2a2f_graph_dbo_dp2tp1.sh)
为准。示例参数不等于适用于所有负载的性能建议。

| 项目 | 上游静态 AFD 示例 | 当前弹性路径 |
| --- | --- | --- |
| 运行时 | vLLM 0.26、MRV1、CUDA P2pNccl | 相同 |
| EP / EPLB | 开 EP；不开 EPLB | 相同，显式校验 |
| DBO | 开启；decode threshold=2、prefill=12；双 ubatch | 支持；本轮 threshold=2/2，确保短请求触发 prefill DBO |
| 图 | `FULL_DECODE_ONLY`；prefill eager | 相同 |
| 编译 | 脚本不显式设置 `mode`，沿用 vLLM 配置 | 本轮显式 `mode=0`，保留 full CUDA Graph；mode=3 未验，mode=1 被拒绝 |
| 图尺寸 | capture sizes=[64]、max capture=64 | 本轮 [1,2,4,8,16]、max capture=16；64 未验 |
| 批次 | max sequences=64、max batched tokens=64 | 本轮 16、2048；未验证示例的长 prompt 分块行为 |
| 拓扑 | DP2/TP1、DP1/TP2、DP2/TP2 等脚本 | 目前只允许 TP1、A DP>=2、F DP>=1、A ranks>=F ranks |
| 启动方式 | A/F 分别启动，各自配置 | 一个 Attention `vllm serve`，EEP 管理 A，插件 Ray actors 管理 F |
| 弹性附加参数 | 无 | `--enable-elastic-ep --data-parallel-backend ray --distributed-executor-backend uni`、显式 A worker、endpoint plugin |
| KV 预算 | 示例按常规显存规划 | 必须显式 `--kv-cache-memory-bytes`；当前每 A 1 GiB，扩缩清 KV |

图模式不要求启用 `torch.compile`。`mode=0` 只关闭编译，不能把这次运行
描述成 eager decode。反过来，`--enable-dbo` 也不保证每个 batch 都拆分：
低负载会走单批 fallback，验收需要同时覆盖两条路径。

本轮上述小尺寸图配置已通过四类扩缩与 85/85 次推理：每个存活 A 都有
真实双批图 replay，F 有两 stage 图执行，保留 worker 的旧图在重新捕获前
全部释放。vLLM `async_scheduling` 实际保持开启；它与下文被禁止的 AFD
`async_dp` 是不同功能。

## 上游已有，而弹性版明确没有接入

| 能力 | 上游证据 / 范围 | 当前限制与原因 |
| --- | --- | --- |
| TP>1 | GPU recipe 有 DP1/TP2 和 DP2/TP2 | 配置拒绝。A 的 GPU bundle 获取与 F 每 actor 一 GPU、uni executor 都按 TP1 实现；需要真正扩展角色 executor/placement 和 EEP 切组验证，不能只删校验 |
| 1A1F / A DP=1 | 静态、PD recipe 都有 | 弹性要求 A DP>=2；使用 DPLBAsyncMPClient/native EEP 的当前接入范围，没有单 A 启动和跨 1/2 的完整控制路径 |
| PD 分离 / NIXL KV transfer | 2P1A1F eager/graph+DBO recipe；A 是 KV consumer，F 不接 KV transfer | `kv_transfer_config` 被拒绝。当前重建 KV、清缓存，尚未处理迁移中的远端 KV 握手、请求路由和在途传输 |
| Ascend 同步 AFD | CAMP2p、DBO、ACL Graph，另有较新的 NPU 模型支持 | 弹性仅 GPU。缺 stateless HCCL EEP group 的创建/切换/销毁资格验证及对应 worker 生命周期适配 |
| Ascend CAM async | 上游有实验性 async 路径及自管 MoE 双阶段；最新 main 改用插件自建 routed-only 算子 | 弹性不接 async connector。该路径自身也不支持 native DBO+图，不能直接套本轮 GPU 方案 |
| ModelRunner V2 静态路径 | 实际代码、模块文档及 E2E 有 CUDA V2 eager/graph，NPU V2 仅有限验证 | 弹性要求 MRV1。上游 V2 校验本身拒绝 elastic EP 和 DBO；根 README 仍写 V2 不支持，与细粒度文档不同，应以具体路径为准 |

## 允许配置，但尚不能算弹性验证通过

| 项目 | 当前证据 / 下一步所需 |
| --- | --- |
| 编译模式 3、官方 [64] 捕获尺寸、prefill threshold=12 | 配置允许。仍需编译、warmup、重捕获和真实请求验证；本轮不代表所有图尺寸都实际 replay |
| `compute_gate_on_attention=true` | DeepSeek CUDA 静态 AFD 支持，弹性未禁；本轮为默认 false。需要验证 router logits 在两 stage 和重连后的传输；不能推广到上游本就禁止 A-side gate 的 Qwen |
| 非整除 A/F，例如 3A2F | connector 和 ElasticTopology 允许；本轮最多四 GPU，只验 2A2F/2A1F/3A1F。3A1F 是整除，不能作为 3A2F 的硬件证据 |
| 多机 GPU | 上游记录 DeepSeek-V2-Lite 2A2F 跨机 TCP/EFA；弹性只有单机验证。需要 Ray placement、跨机 rendezvous、通讯和扩缩后的请求检查 |
| 其他模型与量化 | 上游有 DeepSeek 系列、Qwen3/3.5/3.6 等适配，各有边界。弹性仅 DeepSeek-V2-Lite-Chat BF16 实测；模型未被统一白名单阻止不等于已支持其全部状态和权重布局 |
| 长上下文、chunked prefill、prefix cache | 默认机制未关闭；短请求功能用例不证明长 prompt 分块或缓存命中。扩缩会清 prefix/KV cache，不保留请求状态 |
| 多轮稳定性、扩缩时持续流量、精度与吞吐 | 当前是排空后变更、每阶段短请求验收。尚无多轮 soak、并发请求跨 resize、GSM8K 回归和 DBO 加速比证据 |

## 不能算成弹性版遗漏的通用功能

- GPU 静态 AFD 本来就只支持 `FULL_DECODE_ONLY`；prefill full graph、
  `FULL_AND_PIECEWISE` 等不是删掉弹性校验即可获得的能力。
- GPU P2pNccl 本来就要求 A ranks>=F ranks，且不支持 AFD async DP。
- native DBO+图只支持两个 ubatch，不能任意设置微批数量。
- CUDA Attention remote experts 不支持 EPLB，A 上没有可让 EPLB 重排的远端专家权重。
- LoRA、speculation、PP/CP、sleep、多 API/外部 LB 在当前弹性入口被拒绝。
  不能据此声称这些在所有静态模型上都已支持；例如 Qwen 静态路径也明确
  拒绝 LoRA、speculation、PP、EPLB 等组合。
- 按需加载、KV 保留、无停顿切换、失败回滚仍是当前设计主动排除的优化，
  不属于这次 DBO 图生命周期修复。

## 后续优先级

1. 对齐 GPU DP2/TP1 recipe 的参数：mode3、capture64、threshold2/12，
   加长 prompt 与多轮 A/F 扩缩，再做固定数据集精度回归。
2. TP2：这是已有 GPU 推荐启动脚本与当前弹性实现之间最直接的能力差距。
3. 真正的 3A2F 非整除、多机、A-side gate，按部署需要分别验收。
4. PD 分离和 NPU 弹性各自需要独立生命周期设计；保留现有 EEP 接口与 A/F
   编排分工，不因补功能重新开发一套 EEP 状态机。

本次图+DBO硬件记录见 [ELASTIC_AFD_VALIDATION.md](ELASTIC_AFD_VALIDATION.md)。
