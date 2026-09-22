# Elastic AFD development build

This implements the GPU TP=1 path of the EEP-reuse design against vLLM
**0.26.0 / 568afb3a13806beb53bb2e6bd518269357b237c0** and afd-plugin
**8bb14be66d9f940b1b132b141d986214c0210353**. CPU tests and source review pass.
On H20, commit **00c03bbe** passed initial **2A1F eager startup and a real
completion**, and **78b4638** passed **2A2F → 2A1F → 2A2F F-only resizing**
with real completions at all three stages. Commit **130346b** also passed
**4A2F → 2A2F → 4A2F A-only resizing**, including requests completed by the
new A ranks. Commit **03c87ad** passed the four-GPU mixed-role CUDA graph
cycle **2A2F → 2A1F → 3A1F → 2A1F → 2A2F**, including real-request graph
replay on every A at each stage. Accuracy qualification remains pending.

## Implemented flow

- The existing `/scale_elastic_ep` route defaults to Attention. `role=ffn`
  changes only the FFN pool. The integer is the selected role's **DP size**.
- A uses the native EEP client, `CoreEngineActorManager`,
  `reinitialize_distributed`, `ElasticEPScalingState`, notifications, KV budget
  synchronization, and routing updates. No EEP state machine is copied.
- The worker executor replaces expert mapping/weight migration/EPLB operations
  with paired no-ops and actual checkpoint reloads. Retained A rebuild their
  runner and workspace, then initialize KV through the native allocator path.
- F uses separate Ray actors with a UniProcExecutor and the existing background
  FFN loop. F-only scaling releases every F role, resizes the actor pool, and
  reconstructs each model against the new F DP/EP group. It does not reload
  only the weights of an old model object.
- Every change drains requests, reaches native DP pause consensus, sends STOP
  over the existing metadata channel, joins F, and releases graphs before
  communication groups. Both roles connect concurrently; F starts receiving
  before A profile/warmup/capture. Service resumes only after this completes.
- A scaling waits for **every target EngineCore** to report
  `eep_scaling_state is None`. Native `RECONFIGURE_FINISHED` alone is too early.
- The official endpoint plugin's `init_state` runs the same cross-role finish
  before HTTP starts. Initial/new A defer connector/profile/warmup and stay
  paused after native READY.

The old M1 implementation informed F ownership, role-config isolation, and
repeatable stop/restart. The implementation does not carry over the global
ActorManager hierarchy, FFN EngineCore/ZMQ identities, exception-driven stop,
in-place expert reload, or hard-coded graph pools.

## Current boundaries

GPU only, `P2pNcclAFDConnector`, MRV1, A DP >= 2, F DP >= 1, A ranks >= F
ranks, TP=PP=PCP=DCP=1. Existing GPU non-divisible mappings are retained.
One API process and internal load balancing are required. Specify Ray DP and
the `uni` role executor separately. DBO, speculative decoding, LoRA, sleep,
KV transfer and KV retention are outside this build. KV memory is fixed
explicitly. Model/precision qualification starts with DeepSeek-V2-Lite.

Initial eager 2A1F inference, separate eager A/F shrink/expand cycles and
the four-GPU mixed-role CUDA graph cycle passed. The graph qualification
uses `FULL_DECODE_ONLY`, compilation mode 0, capture sizes 1/2/4/8/16 and
the launch settings below. It does not qualify other compilation modes,
all capture sizes' replay, model accuracy or repeated-cycle stability.

**Ascend is not implemented as an elastic backend in this build.** Static NPU
AFD remains available. Elastic configuration fails early with the stateless
HCCL qualification requirement. The pinned Ascend worker/platform lacks the
EEP device-group support needed for the shared A state machine; this needs
the N0 group-creation/collective/switch/destruction test before an NPU worker
implementation can be accepted. Receiving STOP is supported by the existing
NPU metadata loop, but that alone does not provide NPU elasticity.

Concurrent resize is rejected using the existing scaling gate. There is no
rollback/retry/recovery system. A failure after communication teardown needs
a service restart; this build makes no failure-availability guarantee.

## Candidate launch on an isolated Ray GPU cluster

Install the pinned vLLM runtime and this plugin checkout on every Ray node.
Enable Ray's Dashboard HTTP service: native EEP scale-up calls
`ray.util.state.list_nodes()`. On a single node, bind it to loopback and an
explicit free port, then verify `list_nodes()` before starting the model.
The A scale-up validation failed with `--include-dashboard=false` and passed
after enabling that service, without a product code change.
The complete candidate sequence peaks at six available GPUs (4 A + 2 F).
The initial topology needs three. Use the same model checkpoint on each node.
The code uses the actual GPU assigned to each Ray placement, including A
placements when F is colocated. It does not reserve whole nodes for a role.

```bash
export VLLM_PLUGINS=afd,afd_elastic
export VLLM_USE_V2_MODEL_RUNNER=0

vllm serve "$MODEL" \
  --served-model-name elastic-afd \
  --worker-cls afd_plugin.v1.worker.AFDAttentionWorker \
  --enable-expert-parallel --enable-elastic-ep \
  --data-parallel-backend ray --distributed-executor-backend uni \
  --data-parallel-size 2 --data-parallel-size-local 2 \
  --tensor-parallel-size 1 --enforce-eager \
  --max-model-len 2048 --max-num-seqs 32 --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 1073741824 \
  --additional-config '{"afd":{"role":"attention","connector":"P2pNcclAFDConnector","num_attention_ranks":2,"num_ffn_ranks":1},"afd_elastic":{"ffn_tensor_parallel_size":1}}'
```

The 1 GiB KV budget above is an example, not a measured memory recommendation.
Allow room for the largest tested topology, communication buffers and captures.
Do not set `--enable-eplb`. The plugin narrowly exempts its explicit attention
worker from EPLB and async-EPLB/NIXL prerequisites; native EEP keeps its checks.
The AFD rendezvous host and ports are distributed from F0's actual node.

### CUDA graph configuration

For decode CUDA graphs, remove `--enforce-eager` and use:

```bash
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}' \
--max-cudagraph-capture-size 16 --max-num-seqs 16
```

Compilation mode 0 disables `torch.compile`; it still installs the native
full CUDA graph wrapper. Prefill remains eager. The plugin rejects other
CUDA graph modes and rejects compilation mode 1 (`STOCK_TORCH_COMPILE`) for
elastic graphs: vLLM 0.26's MRV1 loader returns before installing the A graph
wrapper in that mode. Compilation mode 3 is accepted by configuration, but
needs separate hardware qualification.

Graph capture sizes and the explicit KV budget do not impose a total GPU
memory limit. Leave room for weights, communication buffers, workspace and
graph pools, including the F=1 topology's complete expert weights.

Four GPUs can exercise both roles with
`2A2F → 2A1F → 3A1F → 2A1F → 2A2F`. Start with F DP size 2, then request
F=1, A=3, A=2 and F=2 through the same endpoint below. A DP size 3 is legal:
A holds remote expert proxies, while F=1 retains all expert weights.

Candidate scale chain, issuing the next call only after the preceding succeeds:

```bash
curl -f http://localhost:8000/scale_elastic_ep -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size":4,"drain_timeout":300}'
curl -f http://localhost:8000/scale_elastic_ep -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size":2,"role":"ffn","drain_timeout":300}'
curl -f http://localhost:8000/scale_elastic_ep -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size":2,"drain_timeout":300}'
curl -f http://localhost:8000/scale_elastic_ep -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size":1,"role":"ffn","drain_timeout":300}'
```

`POST /is_scaling_elastic_ep` remains available while an AFD resize is active.
Other requests receive the native scaling middleware's 503 response.

## Validation and remaining work

CPU tests are in `tests/unit/elastic`. They cover real orchestration methods
with device/actor doubles, API routing, the gate and drain boundary, STOP wire
compatibility, workspace/KV ordering, and the zero-new-remote actor regression.
These tests cannot establish NCCL progress, model accuracy or memory safety.

The first hardware startup found a MessagePack boundary bug: nested worker RPC
arguments restored `AFDConfig` as a dictionary. The client now sends a plain
mapping on both A/F paths, and workers restore it with the existing validated
config parser. The fix has real MessagePack regression coverage and passed
the H20 startup/completion check. See [ELASTIC_AFD_VALIDATION.md](ELASTIC_AFD_VALIDATION.md).

The subsequent F-only run kept both A actor IDs/PIDs and their placements
unchanged, removed one F actor/placement on shrink, and created a new F
actor/placement on expansion. Both resize calls returned HTTP 200, followed
by `is_scaling_elastic_ep=false` and a successful completion. The measured
API durations were 17.644 s and 28.024 s, including model reload and warmup.
This is one eager functional smoke cycle, not a latency or accuracy benchmark.

The A-only run issued 16 concurrent fixed requests after each topology was
ready. All 48 succeeded. Per-engine request-success counters increased by
4/4/4/4, then 8/8, then 4/4/4/4, demonstrating that both newly added A ranks
processed requests. The two F actors retained their identities throughout.
The run did not send requests during either resize.

The four-GPU graph run issued 16 concurrent requests at each of five stages;
all 80 succeeded. Test-only worker extensions observed successful captures,
actual graph replay inside each A's real-request `execute_model` (excluding
dummy batches), F replay during the request window, and retirement of each
retained worker's old graphs before new capture. The A=3 stage completed
6/5/5 requests across the three engines, including the newly added A.
This was one cycle with requests between resizes. See the validation record
for exact configuration, sampled memory and evidence boundaries.

Further hardware qualification requires:

1. Fixed request/accuracy comparison against static AFD and native EEP baselines
   on the chosen model/runtime; F expert/weight coverage checks.
2. Same-topology STOP/restart and the larger
   `2A1F → 4A1F → 4A2F → 2A2F → 2A1F` mixed-role chain.
3. Repeated resizing, concurrent-traffic drain, and long-lived memory and
   communication-group leak checks.
4. Other graph capture sizes and compilation mode 3.
5. NPU N0 and implementation, which are not covered by GPU results.

The compatibility patches record their pinned source and AFD differences.
Source review closed two concrete bugs: workspace reset on A runner reload,
and upstream `[-0:]` accidentally waiting on old remote actors. Upgrade work
must recopy upstream seams and reapply marked changes; this build does not
implement newer `prepare_elastic_ep`/`commit_elastic_ep` APIs.
