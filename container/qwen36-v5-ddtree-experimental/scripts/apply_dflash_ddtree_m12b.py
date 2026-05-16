#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MARKER = "aeon_dflash_ddtree_m12b"


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
                            # aeon_dflash_ddtree_m12b
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
                            # aeon_dflash_ddtree_m12b
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
                        # aeon_dflash_ddtree_m12b
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
    text = (pkg_root / "v1/worker/gpu_model_runner.py").read_text()
    for needle in (
        MARKER,
        "DDTREE_BATCH_BRANCH_CONV_MIRROR",
        "DDTREE_BATCH_BRANCH_SSM_MIRROR",
        "conv_state[\n                                        block_idx,",
        "ssm_state[ssm_block_idx] =",
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
    patch_batched_branch_state_mirror(pkg_root)
    clear_python_caches(pkg_root)
    verify_static(pkg_root)
    print(f"[{MARKER}] batched branch conv/SSM mirror installed")
    return 0

if __name__ == "__main__":
    sys.exit(main())
