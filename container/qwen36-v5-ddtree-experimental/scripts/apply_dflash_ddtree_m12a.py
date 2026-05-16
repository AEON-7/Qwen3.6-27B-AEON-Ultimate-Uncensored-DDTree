#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MARKER = "aeon_dflash_ddtree_m12a"


def replace_exact(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Could not find expected text in {path}:\n{old}")
    path.write_text(text.replace(old, new, 1))


def clear_python_caches(pkg_root: Path) -> None:
    for pyc in pkg_root.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
    for pycache in pkg_root.rglob("__pycache__"):
        shutil.rmtree(pycache, ignore_errors=True)


def patch_tree_sample_materialization(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """                # aeon_dflash_ddtree_m8u
                # Diagnostic visibility for quality bring-up: log the compact
""",
        """                if os.environ.get("DDTREE_SYNC_AFTER_TREE_SAMPLE", "0") == "1":
                    # aeon_dflash_ddtree_m12a
                    # Keep the ordering/materialization diagnostic available as
                    # an explicit benchmark knob. It reproduced the high-
                    # acceptance branchguard path used by the M72 proof run.
                    torch.cuda.current_stream().synchronize()
                if os.environ.get("DDTREE_MATERIALIZE_TREE_SAMPLE", "0") == "1":
                    # aeon_dflash_ddtree_m12a
                    tree_sample.output_token_ids.detach().cpu().tolist()

                # aeon_dflash_ddtree_m8u
                # Diagnostic visibility for quality bring-up: log the compact
""",
    )


def patch_full_branch_prefix_guard(pkg_root: Path) -> None:
    path = pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py"
    replace_exact(
        path,
        """    if (
        os.environ.get("DDTREE_FULL_BRANCH_COMMIT", "0") == "1"
        and os.environ.get("DDTREE_ALLOW_BRANCH_STATE_COMPACTION", "0") == "1"
        and os.environ.get("DDTREE_UNSAFE_FULL_BRANCH_RESEARCH", "0") == "1"
    ):
""",
        """    if (
        os.environ.get("DDTREE_FULL_BRANCH_COMMIT", "0") == "1"
        and os.environ.get("DDTREE_ALLOW_BRANCH_STATE_COMPACTION", "0") == "1"
        and os.environ.get("DDTREE_UNSAFE_FULL_BRANCH_RESEARCH", "0") == "1"
    ):
        if (
            os.environ.get("DDTREE_FULL_BRANCH_REQUIRE_PREFIX", "0") == "1"
            and accepted_compact
            and flat_prefix_len == 0
        ):
            # aeon_dflash_ddtree_m12a
            # Do not commit root-sibling states unless there is a flat accepted
            # prefix to anchor vLLM's normal postprocess path. Return only a
            # safe bonus token in that case.
            safe_bonus_parent = 0
            emitted = [accepted_tokens[0]]
            return emitted, [], safe_bonus_parent

""",
    )


def verify_static(pkg_root: Path) -> None:
    runner = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    sampler = (pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py").read_text()
    for needle, text in (
        ("DDTREE_MATERIALIZE_TREE_SAMPLE", runner),
        ("DDTREE_SYNC_AFTER_TREE_SAMPLE", runner),
        ("DDTREE_FULL_BRANCH_REQUIRE_PREFIX", sampler),
        (MARKER, runner + sampler),
    ):
        if needle not in text:
            raise RuntimeError(f"Static {MARKER} verification failed: missing {needle}")


def main() -> int:
    root_override = os.environ.get("VLLM_PACKAGE_ROOT")
    if root_override:
        pkg_root = Path(root_override).resolve()
    else:
        import vllm
        pkg_root = Path(vllm.__file__).resolve().parent
    print(f"[{MARKER}] vLLM package root: {pkg_root}")
    patch_tree_sample_materialization(pkg_root)
    patch_full_branch_prefix_guard(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] branchguard/materialization controls installed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
