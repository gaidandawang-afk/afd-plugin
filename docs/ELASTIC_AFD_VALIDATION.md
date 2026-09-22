# Elastic AFD implementation validation — 2026-09-21–22

CPU checks, initial GPU startup, separate eager A-only and F-only cycles,
and one mixed-role CUDA graph cycle passed. Broader elastic acceptance
remains pending.

## Source review

Architecture, worker lifecycle and simplicity were reviewed independently,
then the findings were checked again after fixes:

- Closed: resetting an A runner destroyed the workspace manager. Reinitialize
  one workspace slot before constructing/loading the replacement runner.
- Closed: native EEP's `[-0:]` selected all old remote actors when only local
  actors were added. Waiting and starting now share the same new-actor list.
- Closed: removed a redundant resize lock and automatic failure teardown;
  use the existing scaling gate and retain the agreed no-recovery scope.
- Closed: ordinary EEP delegates to the native HTTP handler before the new
  AFD size/timeout validation.
- Closed: endpoint `attach_router` precedes installation of native scaling
  middleware in v0.26. Middleware replacement now runs in `init_state`, after
  `build_app` and before ASGI startup. The API fixture follows that real order.
- Closed on hardware: nested worker RPC arguments decode as dictionaries.
  Both A/F paths now send mappings and restore `AFDConfig` at the worker
  boundary. Real MessagePack round-trip tests first reproduced the failure,
  then passed along with the subsequent startup run.
- Checked: A continues to use native EEP notifications, KV synchronization,
  state progression and routing; the F branch bypasses A scale/stat updates.
- Checked: the EPLB/NIXL prerequisite exemptions are limited to the explicit
  AFD attention worker. EEP's other validation remains.

## Local checks

Environment: Windows, Python 3.11, local editable plugin install; no torch,
vLLM, Ray device runtime, CUDA or torch-npu runtime installed.

Targeted command:

```text
python -X utf8 -m pytest tests/unit/elastic \
  tests/unit/compat/patches/test_config_validation.py \
  tests/unit/compat/patches/test_engine_core.py \
  tests/unit/package/test_package.py tests/unit/config \
  -o addopts='' --tb=short -q
```

Result: **129 passed, 15 skipped**, including **54 new elastic tests**. The
15 skipped tests require the unavailable vLLM runtime. Two deprecation
warnings come from the local FastAPI/Starlette test client dependencies.

The 2026-09-22 graph configuration change (`90ef183`) additionally ran
`tests/unit/v1/worker/test_cuda_graph.py` with the command above:
**153 passed, 15 skipped**. Added elastic checks accept `FULL_DECODE_ONLY`
with compilation modes 0 and 3, reject unsupported graph modes, and reject
`STOCK_TORCH_COMPILE`, whose MRV1 loader bypasses the A full-graph wrapper.
Changed Python files passed Ruff check/format and `git diff --check`.

The new elastic tests exercise production orchestration/patch methods with
upstream and device doubles. Metadata STOP round-tripping uses the production
encoder/decoder. HTTP tests use FastAPI. Workspace/KV and connector-release
regressions exercise the actual worker method bodies without importing CUDA.
They are control/lifecycle evidence, not proof of distributed execution.

Ruff check/format on changed Python files, compileall, editable installation,
the installed `afd_elastic` endpoint entry point and `git diff --check` are
also checked before the development commit.

## Wider suite limitations

The complete unit suite cannot be collected locally: two existing modules
import torch unconditionally (`test_dsv4_async_validation.py` and
`test_base_factory.py`). With those two modules excluded, seven tests fail in
unchanged test helpers: one completion-overlap test, two POSIX signal cleanup
tests (`os.killpg`), two pidfd tests (`os.pidfd_open`), and two POSIX path-shape
assertions. These were not converted to Windows or relaxed for this change.
The full suite is therefore **not green** in this environment.

## Initial GPU startup — passed

On 2026-09-21, run `elastic-smoke-20260921.startup004` verified product commit
`00c03bbe185848b674134a39034249744b053254` on H20:

- DeepSeek-V2-Lite-Chat BF16, 2A1F, TP=1, MRV1 eager, Ray DP + uni;
  EP enabled, EPLB disabled, fixed 1 GiB KV and GPU utilization 0.35.
- vLLM 0.26.0, torch 2.11.0+cu130 and task-local Ray 2.48.0. The wheel's
  generated commit label is `gffd46bfab`; all 2,034 shared Python sources
  matched the pinned v0.26.0 tag after line-ending normalization. This does
  not claim the wheel itself was built from commit `568afb3a`.
- The isolated Ray cluster could assign only GPUs 4–7. Two A actors and one
  F actor were alive; new GPU processes appeared only on 4, 5 and 6. GCS
  container PIDs and NVML host PIDs were recorded without a direct namespace
  mapping, so the exact A0/A1 device order is not asserted.
- Health, model listing and a real completion returned HTTP 200. The prompt
  `The capital of France is` generated 16 tokens beginning with ` Paris.`.
- The runner exited 0 with both assertions passing. Task-owned service/Ray
  processes were removed, independently checked absent, and GPU memory
  returned to the recorded preflight levels; existing workloads were retained.

Earlier runs retained separate failure evidence: Ray Unix socket path length,
a shared `VLLM_PORT` conflicting between A actors, then the RPC config bug
fixed in `00c03bbe`. The successful recipe uses a short Ray temp directory
and does not export a common `VLLM_PORT` to the actors.

The sibling `afd_agent` project retains the report and raw evidence under
`work/elastic-smoke-20260921/`, with successful artifacts in
`artifacts/startup004/output/`. This was a short startup/inference check;
no resize request or accuracy benchmark was run.

## GPU F-only resize — passed

On 2026-09-21, run `elastic-ffn-20260921.resize001.1789982600668499500`
verified commit `78b463872538c25b32abe61a49c526e214ac3d27` on GPUs 4–7.
The model/runtime and eager TP=1 settings matched the startup check above;
the initial F DP size was 2. No product code change was needed for this run.

The same service executed `2A2F → 2A1F → 2A2F` sequentially. Both calls used
`POST /scale_elastic_ep` with `role=ffn`, `drain_timeout=300`, and target DP
sizes 1 and 2. Each returned HTTP 200, followed by a scaling-status response
of false. Durations below are client-observed HTTP wall time, including
reconfiguration/reload/warmup, not a separately instrumented pause duration.

| Stage | Resize API time | Completion time | Result |
| --- | --- | --- | --- |
| Initial 2A2F | — | 1.845 s | HTTP 200, 16 generated tokens |
| Shrink to 2A1F | 17.644 s | 1.567 s | HTTP 200, 16 generated tokens |
| Expand to 2A2F | 28.024 s | 0.991 s | HTTP 200, 16 generated tokens |

- Both A actor IDs and container PIDs (2371808, 2371809) remained unchanged,
  as did their Ray placement groups.
- Shrink retained F PID 2372702, marked F PID 2372703 DEAD and its placement
  group REMOVED. Expansion retained 2372702 and added F PID 2373630 with a
  new CREATED placement group. Snapshots showed alive F counts 2 → 1 → 2.
- Logs showed F role/group reconstruction and checkpoint loading. All three
  fixed requests (`temperature=0`, `seed=0`, `max_tokens=16`) produced the
  same text beginning ` Paris.`. The test required nonempty completions,
  not bitwise equality; this is not an independent accuracy oracle or proof
  that every expert was exercised.
- GPU 7 memory returned to its preflight level after shrink, then increased
  when the new F actor was created. After final task cleanup, all four cards
  returned to their recorded preflight memory levels. The owned-process
  cleanup inventory was empty; existing workloads were retained.

All three stage assertions and the runner passed (exit 0). Raw API responses,
GCS actor/placement snapshots, GPU observations and logs are retained in the
sibling `afd_agent` project's
`work/elastic-ffn-20260921/artifacts/resize001/output/`. This was one functional
cycle with requests between resizes, not concurrent traffic, repeated-cycle
stress, a peak-memory measurement, or an accuracy benchmark.

## GPU A-only resize — passed

Run `elastic-attention-20260921.resize002.1789984248908461000` verified
commit `130346bb3f27204a911257c49b865ef464cf5754` with GPUs 0–7 available
to the isolated Ray cluster. The same model/runtime and eager TP=1 settings
were used, starting with 4 A and 2 F. No product code change was needed.

The same service executed `4A2F → 2A2F → 4A2F`. Calls to
`POST /scale_elastic_ep` omitted `role` (the default is attention), with
target sizes 2 and 4 and `drain_timeout=300`. Both returned HTTP 200 and
subsequent scaling status was false. Client-observed API wall times were
19.426 s for shrink and 53.904 s for expansion, including reconstruction
and warmup; these are single-run observations on shared GPUs.

| Stage | Successful requests | Per-A request-success counter increase |
| --- | --- | --- |
| Initial 4A2F | 16/16 | 4, 4, 4, 4 |
| Shrink to 2A2F | 16/16 | 8, 8; removed ranks unchanged |
| Expand to 4A2F | 16/16 | 4, 4, 4, 4 |

Each phase sent 16 concurrent fixed requests with `temperature=0`, `seed=0`,
and `max_tokens=64`; all returned HTTP 200, nonempty text and 64 generated
tokens. Before/after `/metrics` snapshots establish that the newly added
A ranks 2 and 3 also completed requests. Counters were compared within each
phase, since native scale-up recreated the API's metric logger counters.

A PIDs 2384333/2384334 persisted. Shrink removed 2384335/2384336 and their
two placement groups; expansion created 2386869/2386870 with new groups.
F PIDs 2385838/2385839, actor IDs and placement groups remained unchanged.
Native EEP logs recorded reconfiguration and checkpoint reload. Scale success
passes the product's all-engine `afd_eep_complete` barrier; no independent
external sampling of EngineCore state was added.

The prior `resize001` passed initial/shrink inference but failed expansion:
native `add_dp_placement_groups()` uses `ray.util.state.list_nodes()`, which
requires the Ray Dashboard HTTP service. The harness had disabled it.
`resize002` enabled Dashboard on loopback port 6129 and verified `list_nodes()`
before model startup. The product commit was unchanged; both runs' evidence
is retained. Both were cleaned up by exact task ownership.

The successful runner exited 0. The owned-process inventory and independent
cleanup check were empty, ports were released and all eight cards returned
to this run's preflight memory levels. Original workloads were retained.
Raw results are under the sibling `afd_agent` project's
`work/elastic-attention-20260921/artifacts/resize002/output/`; the failed
run is under `artifacts/resize001/`. This establishes functional eager
A scaling and inference, not accuracy or availability during resizing.

## GPU mixed-role CUDA graph cycle — passed

On 2026-09-22, run
`elastic-graph-20260922.graph001.1790043070249727000` verified commit
`03c87ad533fbaf91cfde0ecf81701f7d08c5afa4` (product Python unchanged from
`90ef183`) on **GPUs 4–7 only**, with a half-card memory allowance.
The runtime/model matched the eager runs: H20, DeepSeek-V2-Lite-Chat BF16,
vLLM 0.26.0, torch 2.11.0+cu130, Ray 2.48.0, MRV1, TP=1, EP enabled,
EPLB disabled, fixed 1 GiB KV and GPU utilization 0.35.

The service omitted `--enforce-eager` and explicitly used compilation
mode **0**, `FULL_DECODE_ONLY`, capture sizes **[1,2,4,8,16]**, max capture
16, max sequences 16 and max batched tokens 2048. Compilation mode 0
disables `torch.compile`, while retaining CUDA graph capture/replay.
Prefill remains eager. Ray Dashboard HTTP and its `list_nodes()` check
were enabled before model startup.

| Stage | Resize API wall time | Successful requests | Per-A request counter increase |
| --- | --- | --- | --- |
| Initial 2A2F | — | 16/16 | 8, 8 |
| F shrink to 2A1F | 18.509 s | 16/16 | 8, 8 |
| A expand to 3A1F | 42.704 s | 16/16 | 6, 5, 5 |
| A shrink to 2A1F | 12.376 s | 16/16 | 8, 8 |
| F expand to 2A2F | 25.664 s | 16/16 | 8, 8 |

Each stage sent 16 concurrent fixed requests using the same prompt and
parameters as the eager A test. All **80** returned HTTP 200 and **64**
generated tokens. The new A (container PID 2394193) completed five requests.
The four resize calls returned HTTP 200 and scaling status became false.
These times include reconstruction and capture; they are single-run HTTP
observations, not performance benchmarks or isolated pause measurements.

The test-only `graph_probe.GraphProbe` was loaded through the official
`--worker-extension-cls` option. It wrapped successful CUDAGraph capture,
replay and reset calls and observed graph GC using weak references, without
editing remote product/vLLM sources. Evidence established:

- Every live A executed actual `CUDAGraph.replay()` inside `execute_model`
  with scheduled request tokens at each stage. A thread-local scope excluded
  nested `_dummy_run` calls, so idle DP participation did not count as real
  request replay.
- Every live F executed graph replay during each request window. F metadata
  does not independently identify real versus dummy tokens; this is combined
  with A replay and completed-request evidence, not a separate token-origin
  claim for F.
- Every live worker held captured graphs before inference. Every retained
  worker reset or garbage-collected all old live graphs before its first
  new capture after each resize. Removed actors were verified DEAD and their
  placement groups REMOVED; interpreter exit was not counted as graph GC.
- F resizing retained both A actor IDs/PIDs and placement groups; A resizing
  retained F identity and placement. Expansion created new actors/groups,
  and model placement counts matched every target topology.

A conservative monitor sampled **all compute-process memory** on the four
selected GPUs, including loading/capture/resizing. Physical memory was
97,871 MiB per card, so the limit was **48,935.5 MiB** per card. Across
164 samples (actual intervals 1.136–2.659 s), observed peaks were:

| GPU | Sampled peak MiB |
| --- | --- |
| 4 | 6,568 |
| 5 | 6,568 |
| 6 | 34,008 |
| 7 | 19,968 |

The maximum was **33.21 GiB / 34.75% of physical memory**, below the
47.79 GiB half-card allowance. No guard violation occurred. Summing all
selected-card compute processes conservatively bounds task usage at the
sampled times; this is not exact NVML host-PID/container-PID attribution,
an allocator quota, or a measurement of sub-sample transient peaks.

All five stage assertions passed. Task-owned processes were removed and
independently checked absent. The four cards returned to their original
20/18/18/20 MiB baseline; ports 6180–6239 were verified bindable. Raw results
remain in the sibling `afd_agent`
project's `work/elastic-graph-20260922/artifacts/graph001/output/`, with
the executed case in `skills/test-service/cases/elastic-graph-smoke/` and
report `reports/ELASTIC-GRAPH-2026-09-22.md` committed as `552d08f` in that
validation repository.

This qualifies one bounded four-GPU cycle. It does not establish replay of
every captured size, compilation mode 3, model accuracy, resizing under
concurrent traffic, long-running stability or NPU graph elasticity.

## Eager DBO mixed-role cycle — 2026-09-22

Product commit **652f661870ccccfb1a54445fc8cc6581721862ff** adds eager DBO
configuration support and restores two workspace slots when EEP reloads A.
It changes two product files: 8 added / 3 removed lines. The EEP state machine,
role actors, STOP protocol and shared pause/reconnect sequence are unchanged.

Run `elastic-dbo-20260922.eager001.1790048614112815500` passed on GPU 4–7,
using the pinned vLLM 0.26 runtime, DeepSeek-V2-Lite-Chat BF16, TP1, Ray2.48,
MRV1, explicit 1 GiB KV, max sequences16, max batched tokens2048 and memory
utilization0.35. It enabled `--enforce-eager --enable-dbo` with decode and
prefill token thresholds both2. Async scheduling remained enabled.

| Stage | Scale HTTP seconds | Concurrent request increments per active A |
| --- | ---: | --- |
| Initial 2A2F | — | 8 / 8 |
| F shrink: 2A1F | 19.182 | 8 / 8 |
| A expand: 3A1F | 43.986 | 6 / 5 / 5 |
| A shrink: 2A1F | 13.127 | 8 / 8 |
| F expand: 2A2F | 24.781 | 8 / 8 |

Each stage completed16 concurrent requests and one subsequent low-traffic
request: **85/85 HTTP200**, all64 completion tokens, with the expected `Paris`
answer prefix. These checks are a functional smoke test, not accuracy or
performance qualification. There was no inference during resize.

An official worker-extension probe recorded successful eager `_run_ubatches`
inside real scheduled A requests, excluding nested dummy work. Every active A,
including newly created rank2, recorded both prefill and decode with stages0/1,
positive token counts, current DP size and no cached graphs. Every F completed
two-stage forward during the request window. F metadata does not independently
identify token origin. Each topology also passed a real-request single-batch
fallback check. Actor/PG identity and release assertions passed for both roles.

Across191 conservative samples, all-compute-process memory peaks on GPUs4/5/6/7
were6700/6700/33270/19614MiB, below48935.5MiB per card. The guard is sampled
protection, not an allocator quota. Cleanup left no task processes or GPU compute
processes; the cards returned to18MiB each and ports6240–6299 were bindable.

Targeted CPU checks: **95 passed, 4 skipped**, covering elastic configuration,
worker lifecycle, compatibility validation, Attention runner and graph policy.
Probe CPU checks verify request attribution, dummy exclusion and original
return/exception behavior. Python lint/format and `git diff --check` pass.

The sibling `afd_agent` repository contains the executed case at
`skills/test-service/cases/elastic-dbo-smoke/`, report
`reports/ELASTIC-DBO-2026-09-22.md`, and raw evidence at
`work/elastic-dbo-20260922/artifacts/eager001/output/`.
Elastic DBO with graphs remains rejected pending the separate graph lifecycle
implementation and hardware qualification.

## Unverified / not implemented

- The larger six-GPU mixed A/F chain and same-topology STOP/restart.
- Long-running repeated resizing, concurrent-traffic drain and F expert coverage.
- Accuracy, sub-sample memory peaks, communication-group leak checks and exact
  pause duration.
- Other CUDA graph capture sizes and compilation mode 3.
- Elastic DBO with CUDA graphs; repeated-cycle DBO stability and accuracy.
- NPU EEP stateless HCCL qualification and NPU elastic implementation.

The required hardware sequence and candidate launch are in
[ELASTIC_AFD.md](ELASTIC_AFD.md).
