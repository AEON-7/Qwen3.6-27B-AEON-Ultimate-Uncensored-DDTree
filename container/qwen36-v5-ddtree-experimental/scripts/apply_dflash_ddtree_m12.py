#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MARKER = "aeon_dflash_ddtree_m12"


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
                    # aeon_dflash_ddtree_m12
                    # Keep the ordering/materialization diagnostic available as
                    # an explicit benchmark knob. It reproduced the high-
                    # acceptance branchguard path used by the M72 proof run.
                    torch.cuda.current_stream().synchronize()
                if os.environ.get("DDTREE_MATERIALIZE_TREE_SAMPLE", "0") == "1":
                    # aeon_dflash_ddtree_m12
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
            # aeon_dflash_ddtree_m12
            # Do not commit root-sibling states unless there is a flat accepted
            # prefix to anchor vLLM's normal postprocess path. Return only a
            # safe bonus token in that case.
            safe_bonus_parent = 0
            emitted = [accepted_tokens[0]]
            return emitted, [], safe_bonus_parent

""",
    )


def patch_batched_branch_state_mirror(pkg_root: Path) -> None:
    path = pkg_root / "v1/worker/gpu_model_runner.py"
    replace_exact(
        path,
        """                        for target_block_id in target_conv_blocks:
                            if target_block_id >= conv_state.shape[0]:
                                continue
                            for offset in dict.fromkeys(offsets):
                                if offset < 0 or offset + conv_width > state_len:
                                    continue
                                conv_state[
                                    target_block_id,
                                    offset : offset + conv_width,
                                ].copy_(branch_conv_state)
""",
        """                        if (
                            os.environ.get("DDTREE_BATCH_BRANCH_CONV_MIRROR", "0") == "1"
                            and os.environ.get("DDTREE_BROADCAST_BRANCH_CONV", "1") == "1"
                        ):
                            # aeon_dflash_ddtree_m12
                            # Batch all target compact-row blocks per offset.
                            # Use indexed assignment rather than copy_ on an
                            # advanced-indexing temporary so the cache tensor is
                            # actually updated.
                            valid_blocks = [
                                int(block)
                                for block in target_conv_blocks
                                if 0 <= int(block) < conv_state.shape[0]
                            ]
                            if valid_blocks:
                                block_idx = torch.tensor(
                                    valid_blocks,
                                    dtype=torch.long,
                                    device=conv_state.device,
                                )
                                for offset in dict.fromkeys(offsets):
                                    if offset < 0 or offset + conv_width > state_len:
                                        continue
                                    conv_state[
                                        block_idx,
                                        offset : offset + conv_width,
                                    ] = branch_conv_state.unsqueeze(0).expand(
                                        len(valid_blocks), *branch_conv_state.shape
                                    )
                        else:
                            for target_block_id in target_conv_blocks:
                                if target_block_id >= conv_state.shape[0]:
                                    continue
                                for offset in dict.fromkeys(offsets):
                                    if offset < 0 or offset + conv_width > state_len:
                                        continue
                                    conv_state[
                                        target_block_id,
                                        offset : offset + conv_width,
                                    ].copy_(branch_conv_state)
""",
    )
    replace_exact(
        path,
        """                        for target_block_id in target_conv_blocks:
                            if target_block_id >= conv_state.shape[0]:
                                continue
                            for offset in dict.fromkeys(offsets):
                                if offset < 0 or offset + conv_width > state_len:
                                    continue
                                conv_state[
                                    target_block_id,
                                    :,
                                    offset : offset + conv_width,
                                ].copy_(branch_conv_state)
""",
        """                        if (
                            os.environ.get("DDTREE_BATCH_BRANCH_CONV_MIRROR", "0") == "1"
                            and os.environ.get("DDTREE_BROADCAST_BRANCH_CONV", "1") == "1"
                        ):
                            # aeon_dflash_ddtree_m12
                            valid_blocks = [
                                int(block)
                                for block in target_conv_blocks
                                if 0 <= int(block) < conv_state.shape[0]
                            ]
                            if valid_blocks:
                                block_idx = torch.tensor(
                                    valid_blocks,
                                    dtype=torch.long,
                                    device=conv_state.device,
                                )
                                for offset in dict.fromkeys(offsets):
                                    if offset < 0 or offset + conv_width > state_len:
                                        continue
                                    conv_state[
                                        block_idx,
                                        :,
                                        offset : offset + conv_width,
                                    ] = branch_conv_state.unsqueeze(0).expand(
                                        len(valid_blocks), -1, -1
                                    )
                        else:
                            for target_block_id in target_conv_blocks:
                                if target_block_id >= conv_state.shape[0]:
                                    continue
                                for offset in dict.fromkeys(offsets):
                                    if offset < 0 or offset + conv_width > state_len:
                                        continue
                                    conv_state[
                                        target_block_id,
                                        :,
                                        offset : offset + conv_width,
                                    ].copy_(branch_conv_state)
""",
    )
    replace_exact(
        path,
        """                    for target_block_id in dict.fromkeys(target_ssm_blocks):
                        if (
                            target_block_id < ssm_state.shape[0]
                            and src_block_id != target_block_id
                        ):
                            ssm_state[target_block_id].copy_(src_ssm_state)
                    compacted += 1
""",
        """                    if os.environ.get("DDTREE_BATCH_BRANCH_SSM_MIRROR", "0") == "1":
                        # aeon_dflash_ddtree_m12
                        valid_ssm_blocks = [
                            int(block)
                            for block in dict.fromkeys(target_ssm_blocks)
                            if 0 <= int(block) < ssm_state.shape[0]
                            and int(block) != src_block_id
                        ]
                        if valid_ssm_blocks:
                            ssm_block_idx = torch.tensor(
                                valid_ssm_blocks,
                                dtype=torch.long,
                                device=ssm_state.device,
                            )
                            ssm_state[ssm_block_idx] = src_ssm_state.unsqueeze(0).expand(
                                len(valid_ssm_blocks), *src_ssm_state.shape
                            )
                    else:
                        for target_block_id in dict.fromkeys(target_ssm_blocks):
                            if (
                                target_block_id < ssm_state.shape[0]
                                and src_block_id != target_block_id
                            ):
                                ssm_state[target_block_id].copy_(src_ssm_state)
                    compacted += 1
""",
    )


def verify_static(pkg_root: Path) -> None:
    runner = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    sampler = (pkg_root / "v1/spec_decode/ddtree_runtime_sampler.py").read_text()
    for needle, text in (
        (MARKER, runner + sampler),
        ("DDTREE_MATERIALIZE_TREE_SAMPLE", runner),
        ("DDTREE_SYNC_AFTER_TREE_SAMPLE", runner),
        ("DDTREE_FULL_BRANCH_REQUIRE_PREFIX", sampler),
        ("DDTREE_BATCH_BRANCH_CONV_MIRROR", runner),
        ("DDTREE_BATCH_BRANCH_SSM_MIRROR", runner),
        ("conv_state[\n                                        block_idx,", runner),
        ("ssm_state[ssm_block_idx] =", runner),
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
    patch_batched_branch_state_mirror(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] M72 branchguard and batched branch-state controls installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
