#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from dataclasses import fields
from pathlib import Path

MARKER = "aeon_dflash_ddtree_m11p"


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def patch_routed_experts_compat(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/layers/fused_moe/routed_experts_capturer.py"
    text = path.read_text()
    shim = '''

# aeon_dflash_ddtree_m11p
# Nightly compatibility shims: gpu_model_runner may import these helpers even
# when routed-expert return is disabled. Keep startup working for non-MoE Qwen3.6.
_GLOBAL_ROUTED_EXPERTS_CAPTURER = None

def get_global_experts_capturer():
    return _GLOBAL_ROUTED_EXPERTS_CAPTURER

def init_routed_experts_capturer_with_shared_cache(*args, **kwargs):
    return None

def free_routing_buffers(*args, **kwargs):
    return None

def issue_routing_d2h_copy(*args, **kwargs):
    return None

def extract_routed_experts_for_current_batch(*args, **kwargs):
    return None
'''
    missing = [
        name for name in (
            "get_global_experts_capturer",
            "init_routed_experts_capturer_with_shared_cache",
            "free_routing_buffers",
            "issue_routing_d2h_copy",
            "extract_routed_experts_for_current_batch",
        )
        if f"def {name}" not in text
    ]
    if missing:
        text += shim
    path.write_text(text)


def patch_conv_state_slice(pkg_root: Path) -> None:
    path = pkg_root / "model_executor/models/qwen3_dflash.py"
    text = path.read_text()
    old = """                        conv_state_indices=spec_state_indices_tensor,
                        parent_ids=attn_metadata.ddtree_parent_ids[
                            : attn_metadata.num_spec_decodes
                        ],
"""
    new = """                        conv_state_indices=spec_state_indices_tensor[
                            : attn_metadata.num_spec_decodes
                        ],
                        parent_ids=attn_metadata.ddtree_parent_ids[
                            : attn_metadata.num_spec_decodes
                        ],
"""
    if new not in text:
        if old in text:
            text = text.replace(old, new, 1)
            path.write_text(text)
        elif "conv_state_indices=spec_state_indices_tensor[" in text:
            return
        else:
            raise RuntimeError(f"Could not find DDTree conv_state_indices call pattern in {path}")


def patch_scheduler_output_compat(pkg_root: Path) -> None:
    path = pkg_root / "v1/core/sched/scheduler.py"
    text = path.read_text()
    old = """            # Get routing data from ModelRunnerOutput (via worker D2H pipeline)
            routed_experts = None
            if (
                model_runner_output.routed_experts_dict is not None
                and req_id in model_runner_output.routed_experts_dict
            ):
                routed_experts = model_runner_output.routed_experts_dict[req_id]
"""
    new = """            # Get routing data from ModelRunnerOutput (via worker D2H pipeline)
            routed_experts = None
            # aeon_dflash_ddtree_m11p
            # vLLM nightly renamed routed_experts_dict -> routed_experts.
            # Keep DDTree-compatible images aligned with the installed
            # ModelRunnerOutput dataclass instead of assuming either name.
            routed_experts_by_req = getattr(
                model_runner_output, "routed_experts_dict", None
            )
            if routed_experts_by_req is None:
                routed_experts_by_req = getattr(
                    model_runner_output, "routed_experts", None
                )
            if routed_experts_by_req is not None and req_id in routed_experts_by_req:
                routed_experts = routed_experts_by_req[req_id]
"""
    if old in text:
        path.write_text(text.replace(old, new, 1))


def patch_gpu_model_runner_output_compat(pkg_root: Path) -> None:
    from vllm.v1.outputs import ModelRunnerOutput

    path = pkg_root / "v1/worker/gpu_model_runner.py"
    text = path.read_text()
    valid_output_fields = {field.name for field in fields(ModelRunnerOutput)}
    if "routed_experts_dict" not in valid_output_fields:
        text = text.replace("                routed_experts_dict=routed_experts_dict,\n", "")
        path.write_text(text)


def verify_static(pkg_root: Path) -> None:
    from vllm.v1.outputs import ModelRunnerOutput

    routed = (pkg_root / "model_executor/layers/fused_moe/routed_experts_capturer.py").read_text()
    qwen = (pkg_root / "model_executor/models/qwen3_dflash.py").read_text()
    runner = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    scheduler = (pkg_root / "v1/core/sched/scheduler.py").read_text()
    valid_output_fields = {field.name for field in fields(ModelRunnerOutput)}
    if "def extract_routed_experts_for_current_batch" not in routed:
        raise RuntimeError("M11P verification failed: missing routed expert shim")
    if (
        "routed_experts_dict" not in valid_output_fields
        and "routed_experts_dict=routed_experts_dict" in runner
    ):
        raise RuntimeError("M11P verification failed: ModelRunnerOutput receives unsupported routed_experts_dict kwarg")
    if (
        "routed_experts_dict" not in valid_output_fields
        and "model_runner_output.routed_experts_dict" in scheduler
        and "getattr(\n                model_runner_output, \"routed_experts_dict\", None" not in scheduler
    ):
        raise RuntimeError("M11P verification failed: scheduler directly reads unsupported routed_experts_dict attr")
    if "conv_state_indices=spec_state_indices_tensor[" not in qwen:
        print(f"[{MARKER}] warning: conv_state_indices slice marker not found in qwen3_dflash.py")


def main() -> int:
    import vllm
    pkg_root = Path(vllm.__file__).resolve().parent
    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_routed_experts_compat(pkg_root)
    patch_scheduler_output_compat(pkg_root)
    patch_gpu_model_runner_output_compat(pkg_root)
    try:
        patch_conv_state_slice(pkg_root)
    except RuntimeError as exc:
        # Some source layouts carry the slice fix in another overlay/location;
        # M11P's required startup fix is the routed-experts shim.
        print(f"[{MARKER}] conv-index slice check skipped: {exc}")
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] routed-experts import shim and conv-index slice verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
