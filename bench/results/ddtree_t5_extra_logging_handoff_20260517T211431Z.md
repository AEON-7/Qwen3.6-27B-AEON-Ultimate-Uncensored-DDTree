# DDTree T5 extra logging handoff

Scope: continue the clean `mt-ddtree-qwen36-branch-state-isolation` investigation with a reproducible live smoke harness and compare the stable DFlash baseline to unsafe `ddtree-full`.

## Branch / worktree

- Branch: `mt-ddtree-qwen36-branch-state-isolation`
- Worktree: `/tmp/ddtree-qwen36-branch-state-isolation`
- Added harness: `bench/scripts/ddtree_live_capability_smoke.py`
- No public PR, issue, or external publishing performed.

## Baseline check: stable DFlash v4

Container: existing `qwen36-aeon-xs-dflash` service (`ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4`).

Artifact: `bench/results/qwen36_dflash_v4_baseline_live_smoke_20260517T210848Z.json`

Results with thinking disabled for the short capability probes:

- prose: OK, visible content, `finish_reason=stop`
- math `17*23`: OK, returned `391`
- guided JSON: OK, parseable schema JSON
- tool call: OK, OpenAI `message.tool_calls[]` emitted

This confirms the smoke harness itself is not causing the guided-output/tool-call failures.

## Unsafe DDTree-full probe

Container: temporary `ddtree-full-extra-logging` from `ghcr.io/aeon-7/vllm-aeon-ultimate-ddtree:qwen36-v5-m53-experimental`, launched with:

- `MODEL_DIR=/models/xs`
- `DFLASH_DIR=/models/dflash-drafter`
- `PROFILE=benchmark`
- `MAX_MODEL_LEN=1024`
- `MAX_NUM_BATCHED_TOKENS=1024`
- `MAX_NUM_SEQS=1`
- `GPU_MEMORY_UTILIZATION=0.70`
- `DDTREE_UNSAFE_FULL_BRANCH_RESEARCH=1`
- Docker host flags: `--gpus all --ipc host --network host --security-opt label=disable`

Artifacts:

- `bench/results/qwen36_ddtree_full_extra_logging_live_smoke_20260517T211431Z.json`
- local log file: `bench/results/qwen36_ddtree_full_extra_logging_live_logs_20260517T211431Z.log` (ignored by git as `.log`, kept on disk)

Observed results:

- prose: HTTP 200, but quality collapsed into repetition and hit `finish_reason=length`:
  `The horizon bled gold and the the the horizon, the horizon, ...`
- math `17*23`: HTTP 200, returned `391`
- guided JSON: HTTP 500 after EngineCore fatal error
- tool call: HTTP 500 because the engine was already dead

## What the extra logging taught us

The failure is now more specific than the earlier generic quality-collapse observation.

Key log sequence:

1. The first DDTree request successfully built and handed off real tree metadata:
   - `DDTree proposer built payloads=1 first_nodes=15 first_parents=[-1, 0, 1, 2, 3, 4, 5, 6]`
   - `DDTree runner installed live parent metadata payloads=1 parent_shape=(1, 16)`
   - `DDTree refreshed cached attention metadata builder=GDNAttentionMetadataBuilder ... parent_shape=(1, 16)`
   - `DDTree refreshed cached attention metadata builder=FlashAttentionMetadataBuilder ... parent_shape=(1, 16)`

2. During the later guided/tool-capability phase, the runner reused a pending proposer payload:
   - `DDTree rehydrated pending proposer payloads=1 for attention metadata (single-use)`
   - immediately followed by `Using DDTree Triton conv/GDN parent-state replay`

3. The fatal crash occurred inside the fused GDN update path, not in request parsing or the OpenAI API layer:
   - `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `_forward_core`, line 1940
   - `fused_sigmoid_gating_delta_rule_update(...)`
   - `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered`

Interpretation: unsafe full-branch mode has two separate failure signatures:

- **Coherence failure:** free-form prose can return HTTP 200 while repeating/collapsing, so success status is insufficient.
- **State/payload lifecycle failure:** guided/tool requests can drive the full-branch path into a stale or mismatched parent-state replay window. The smoking-gun line is the `rehydrated pending proposer payloads=1 ... (single-use)` immediately before Triton GDN replay and CUDA illegal memory access.

This points the next investigation at the pending proposer payload lifecycle and GDN parent metadata freshness around guided decoding/tool-call requests, not at the generic chat API, Docker launch flags, or DFlash baseline.

## Cleanup

- Removed the failed `ddtree-full-extra-logging` container.
- Restarted the pre-existing `qwen36-aeon-xs-dflash` service and verified `/v1/models` is ready again.

## Recommended next step

Do not try another broad fix. Add targeted instrumentation/assertions around the full-branch path:

1. Log request id + scheduler step id on every pending proposer payload save/rehydrate/consume.
2. Assert that a rehydrated payload request id matches the active request id and that guided/tool decode windows are not using a payload from a prior request.
3. Log compact parent tensor width, `num_spec_decodes`, and GDN `spec_state_indices_tensor` shape immediately before `Using DDTree Triton conv/GDN parent-state replay`.
4. Re-run only the guided JSON probe first, because it is now the fastest crash reproducer.
