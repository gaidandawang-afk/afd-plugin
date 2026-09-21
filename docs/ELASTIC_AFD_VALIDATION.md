# Elastic AFD implementation validation — 2026-09-21

This is a development checkpoint, not hardware acceptance.

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

Result: **127 passed, 15 skipped**, including **52 new elastic tests**. The
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

## Unverified / not implemented

- Real Ray resource assignment and repeated NCCL rendezvous.
- Eager inference after all four A/F resize directions; new-A request
  distribution and F expert coverage.
- Accuracy, peak memory, group/actor cleanup and pause duration on hardware.
- CUDA graph capture/replay after role changes.
- NPU EEP stateless HCCL qualification and NPU elastic implementation.

The required hardware sequence and candidate launch are in
[ELASTIC_AFD.md](ELASTIC_AFD.md). No remote GPU was used for these checks.
