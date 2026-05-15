# Agent Notes

This repository tracks the experimental DDTree-on-vLLM path for Qwen3.6 AEON
Ultimate on DGX Spark / GB10. Treat the stable DFlash path as production and
this repository as a research lab.

## Working Rules

- Keep DDTree work reproducible. If you add a milestone patch, update the
  milestone list in `container/qwen36-v5-ddtree-experimental/README.md`.
- Preserve the full serving capability surface: multimodal input, reasoning
  parsing, OpenAI-compatible tool calls, JSON/structured output behavior, and
  the ModelOpt NVFP4 GB10 CUTLASS path.
- Do not claim DDTree is production-faster until non-flat branch commit,
  branch-state GDN replay, DFlash context compaction, and fused branch attention
  all pass quality and benchmark checks.
- Keep raw benchmark artifacts under `bench/results/` with timestamped names.
- Do not commit model weights, API keys, generated caches, or local runtime
  logs.

## Suggested Validation Flow

1. Run the small full-attention proving ground first:
   `Qwen/Qwen2.5-0.5B-Instruct`.
2. Verify token-by-token parity against ordinary decoding.
3. Move to Qwen3.6 only after parent metadata, accepted-branch commit, and
   branch masks are correct on the small model.
4. Benchmark Qwen3.6 with natural prompt categories: coding, math, reasoning,
   prose, natural language, and extraction/JSON.
5. Report TTFT, TPOT, median decode, peak decode, acceptance rate, and quality
   notes together.

## Production Reminder

For reliable Qwen3.6 serving today, use:

```text
ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4
```

The DDTree image is published so researchers can reproduce and extend the work,
not because the tree path is complete.
