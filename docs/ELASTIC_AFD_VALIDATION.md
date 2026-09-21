# Elastic AFD implementation validation — 2026-09-21

CPU checks, initial GPU startup and one eager F-only shrink/expand cycle
passed. A resizing and broader elastic acceptance remain pending.

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

## Unverified / not implemented

- A shrink/expand and new-A request distribution; the complete mixed A/F chain.
- Long-running repeated resizing, concurrent-traffic drain and F expert coverage.
- Accuracy, peak memory, communication-group leak checks and exact pause duration.
- CUDA graph capture/replay after role changes.
- NPU EEP stateless HCCL qualification and NPU elastic implementation.

The required hardware sequence and candidate launch are in
[ELASTIC_AFD.md](ELASTIC_AFD.md).
