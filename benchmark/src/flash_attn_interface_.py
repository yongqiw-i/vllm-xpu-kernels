# SPDX-License-Identifier: Apache-2.0
from importlib import metadata
from pathlib import Path
import site
import sys
from typing import Optional

import torch

try:
    from xattention import (
        flash_attn_varlen_func as pip_flash_attn_varlen_func,
        flash_attn_with_kvcache as pip_flash_attn_with_kvcache,
    )
    PIP_FLASH_ATTN_UNAVAILABLE_REASON = None
except ImportError as e:
    pip_flash_attn_varlen_func = None
    pip_flash_attn_with_kvcache = None
    PIP_FLASH_ATTN_UNAVAILABLE_REASON = str(e)

DEFAULT_FA_VERSION = 2


def _has_vllm_fa2_varlen_fwd() -> bool:
    return hasattr(torch.ops._vllm_fa2_C, "varlen_fwd")


def _find_installed_vllm_fa2_extension() -> Optional[Path]:
    candidates = ("vllm-xpu-kernels", "vllm_xpu_kernels")
    for package_name in candidates:
        try:
            dist = metadata.distribution(package_name)
        except metadata.PackageNotFoundError:
            continue
        for file in dist.files or ():
            file_name = str(file)
            if "_vllm_fa2_C" in file_name and file_name.endswith(".so"):
                path = Path(dist.locate_file(file))
                if path.exists():
                    return path

    # Editable installs and some wheel installers do not expose generated
    # extension files through ``Distribution.files``. Search Python install
    # roots directly, while preferring site/dist-packages over the source
    # checkout that launched this benchmark.
    install_roots = []
    try:
        install_roots.extend(site.getsitepackages())
    except AttributeError:
        pass
    try:
        install_roots.append(site.getusersitepackages())
    except AttributeError:
        pass
    install_roots.extend(
        entry for entry in sys.path
        if "site-packages" in entry or "dist-packages" in entry)

    seen = set()
    for root in install_roots:
        if not root:
            continue
        root_path = Path(root).resolve()
        if root_path in seen:
            continue
        seen.add(root_path)
        package_dir = root_path / "vllm_xpu_kernels"
        for pattern in ("_vllm_fa2_C*.so", "*_vllm_fa2_C*.so"):
            for path in sorted(package_dir.glob(pattern)):
                if path.is_file():
                    return path
    return None


def _load_vllm_fa2_extension():
    if _has_vllm_fa2_varlen_fwd():
        return None

    # Prefer the binary from the installed pip distribution.  When this
    # benchmark is launched from the source checkout, the checkout's
    # ``vllm_xpu_kernels`` package shadows site-packages and may not contain
    # the compiled extension even though the wheel is installed.
    extension_path = _find_installed_vllm_fa2_extension()
    pip_error = None
    if extension_path is not None:
        try:
            torch.ops.load_library(str(extension_path))
        except OSError as e:
            pip_error = f"load_library({extension_path}) failed: {e}"
        else:
            if _has_vllm_fa2_varlen_fwd():
                return None
            pip_error = (
                f"loaded {extension_path}, but "
                "_vllm_fa2_C.varlen_fwd is absent")

    # Fall back to an in-place/local extension only when no usable pip binary
    # was found.
    try:
        from vllm_xpu_kernels import _vllm_fa2_C  # noqa: F401
    except ImportError as e:
        import_error = e
    else:
        if _has_vllm_fa2_varlen_fwd():
            return None
        import_error = RuntimeError(
            "imported vllm_xpu_kernels._vllm_fa2_C, but varlen_fwd was "
            "not registered")

    if _has_vllm_fa2_varlen_fwd():
        return None
    if pip_error is not None:
        return f"{pip_error}; local import failed: {import_error}"
    return str(import_error)


FA2_UNAVAILABLE_REASON = _load_vllm_fa2_extension()
FA2_AVAILABLE = FA2_UNAVAILABLE_REASON is None


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _normalize_flash_attn_result(result, return_softmax_lse: bool):
    if isinstance(result, tuple):
        return result[:2] if return_softmax_lse else result[0]
    return result


def flash_attn_varlen_func_pip(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: Optional[list[int]] = None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux: Optional[torch.Tensor] = None,
    num_splits_kv: Optional[int] = None,
    is_mix_batch: bool = True,
    start_event: Optional[torch.Event] = None,
    end_event: Optional[torch.Event] = None,
    device: str = "xpu",
    host_kv_lens: Optional[torch.Tensor] = None,
):
    """Benchmark adapter for the installed pip flash_attn XPU kernels."""
    if pip_flash_attn_varlen_func is None or pip_flash_attn_with_kvcache is None:
        raise RuntimeError(
            "pip flash_attn XPU kernels are unavailable: "
            f"{PIP_FLASH_ATTN_UNAVAILABLE_REASON}")
    if dropout_p != 0.0:
        raise NotImplementedError("pip flash_attn benchmark uses dropout_p=0")
    if alibi_slopes is not None:
        raise NotImplementedError("pip flash_attn benchmark does not use ALiBi")
    if out is not None:
        raise NotImplementedError("pip flash_attn benchmark does not pass out")
    if fa_version != DEFAULT_FA_VERSION:
        raise NotImplementedError("pip flash_attn benchmark expects FA2 shapes")
    assert cu_seqlens_k is not None or seqused_k is not None, \
        "cu_seqlens_k or seqused_k must be provided"
    assert cu_seqlens_k is None or seqused_k is None, \
        "cu_seqlens_k and seqused_k cannot be provided at the same time"

    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    if start_event is not None:
        start_event.record()
    if block_table is not None and max_seqlen_q == 1:
        result = pip_flash_attn_with_kvcache(
            q,
            k,
            v,
            qv=q_v,
            cache_seqlens=seqused_k,
            page_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            softmax_scale=softmax_scale,
            sinks=s_aux,
            causal=causal,
            window_size=real_window_size,
            softcap=softcap,
            scheduler_metadata=scheduler_metadata,
            num_splits=num_splits_kv if num_splits_kv is not None else num_splits,
            return_softmax_lse=return_softmax_lse,
        )
    else:
        if block_table is not None:
            if cu_seqlens_k is None:
                assert seqused_k is not None
                cu_seqlens_k = torch.nn.functional.pad(
                    seqused_k.to(device=q.device, dtype=torch.int32),
                    (1, 0)).cumsum(dim=0, dtype=torch.int32)
            seqused_k = None
        result = pip_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            causal=causal,
            qv=q_v,
            block_table=block_table,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=real_window_size,
            softcap=softcap,
            sinks=s_aux,
            num_splits=num_splits,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            return_softmax_lse=return_softmax_lse,
        )
    if end_event is not None:
        end_event.record()
    return _normalize_flash_attn_result(result, return_softmax_lse)

def _as_int32_device_tensor(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.int32)
    return torch.tensor(x, device=device, dtype=torch.int32)


def _kv_tile_from_block_size(block_size: int) -> int:
    # Mirror of flash_api.cpp get_num_splits() / TileShapeQK<1>.
    if block_size == 16:
        return 16
    if block_size == 32:
        return 32
    return 64


def _min_blocks_for_split(kv_tile: int) -> int:
    # Mirror of XeFMHAFwdSplitKVKernel::kMinBlocksForSplit /
    # ReduceSplitK::kMinBlocksForSplit. Below this threshold a sequence is
    # processed as a single split for numerical stability.
    return 32 if kv_tile <= 64 else 128


def _infer_num_xe_cores(device: torch.device) -> int:
    # Prefer runtime query; fall back to 20 (Xe2/BMG default).
    try:
        props = torch.xpu.get_device_properties(device)
    except Exception:
        return 20
    slices = (getattr(props, "gpu_slices", None)
              or getattr(props, "num_slices", None)
              or getattr(props, "slices", None))
    subslices = (getattr(props, "gpu_subslices_per_slice", None)
                 or getattr(props, "num_subslices_per_slice", None)
                 or getattr(props, "subslices_per_slice", None))
    if slices is not None and subslices is not None:
        return max(1, int(slices) * int(subslices))
    return 20


def build_decode_split_plan(
    kv_lens,
    kv_tile: int,
    num_kv_splits: int,
    num_xe_cores: int,
    num_heads_kv: int,
):
    """Produce (splits_per_seq, work_list) for the compact-grid decode kernel.

    Inputs
    ------
    kv_lens         : per-seq KV length in tokens (list/tensor, on host).
    kv_tile         : KV tile width in tokens (must equal the kernel's
                      get<1>(TileShapeQK{})).
    num_kv_splits   : global cap on per-seq split count (buffer dim).
    num_xe_cores    : Xe-core count used for the target-WG heuristic.
    num_heads_kv    : KV heads (workload is sliced across heads_kv heads).

    Returns
    -------
    splits_per_seq  : int32 cpu tensor [batch], splits[i] >= 1.
    work_list       : int32 cpu tensor [total_wgs, 4] of
                      (seq_idx, kv_tile_start, kv_tile_count, split_idx).

    Guarantees
    ----------
    - sum(splits_per_seq) == work_list.size(0) == total_wgs
    - For every emitted work item, kv_tile_count >= 1
    - For each seq, the work items partition [0, kv_tiles) exactly once
    - splits_per_seq[i] <= num_kv_splits (so Oaccum/exp_sums/max_logits
      buffer indexing is safe)
    - splits_per_seq[i] folds in {single-split heuristic, balanced
      assignment, hard cap}; the kernel never needs to second-guess it.
    """
    if isinstance(kv_lens, torch.Tensor):
        kv_lens_list = kv_lens.to(dtype=torch.int32, device="cpu").tolist()
    else:
        kv_lens_list = [int(v) for v in kv_lens]

    tiles_per_seq = [max(1, (kv + kv_tile - 1) // kv_tile)
                     for kv in kv_lens_list]
    total_tiles = sum(tiles_per_seq)

    # Target: ~2x oversubscription of Xe cores per kv head, minimum 4 tiles
    # per WG so split-K overhead stays amortized.
    min_wgs = max(1, num_xe_cores * 2 // max(1, num_heads_kv))
    target_tiles_per_wg = max(4, total_tiles // min_wgs)

    # Mirror of the kernel's is_single_split heuristic: avoid split-reduce
    # for short sequences (numerical stability + overhead).
    min_blocks_for_split = _min_blocks_for_split(kv_tile)

    splits_per_seq = []
    work_items = []
    for i, n_tiles in enumerate(tiles_per_seq):
        if (n_tiles <= target_tiles_per_wg
                or n_tiles < min_blocks_for_split
                or num_kv_splits <= 1):
            n_splits = 1
        else:
            n_splits = ((n_tiles + target_tiles_per_wg - 1)
                        // target_tiles_per_wg)
            # Cap to the static buffer dim AND to n_tiles, so every emitted
            # work item has kv_tile_count >= 1.
            n_splits = min(n_splits, num_kv_splits, n_tiles)
        splits_per_seq.append(n_splits)

        # Even partitioning: first `rem` splits get base+1 tiles, the rest
        # get base. Guarantees count >= 1 since n_splits <= n_tiles.
        base, rem = divmod(n_tiles, n_splits)
        start = 0
        for s in range(n_splits):
            count = base + (1 if s < rem else 0)
            work_items.append([i, start, count, s])
            start += count

    splits_t = torch.tensor(splits_per_seq, dtype=torch.int32)
    work_t = torch.tensor(work_items, dtype=torch.int32)
    return splits_t, work_t

# vllm_xpu_kernel,main,
#   https://github.com/vllm-project/vllm-xpu-kernels/tree/3cf991b
def flash_attn_varlen_func_CalKernelTime(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: Optional[list[int]] = None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # FA3 Only
    scheduler_metadata=None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    # Version selector
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux: Optional[torch.Tensor] = None,
    num_splits_kv: Optional[int] = None,
    is_mix_batch: bool = True,
    start_event: Optional[torch.Event] = None,
    end_event: Optional[torch.Event] = None,
    device: str = "xpu",
    host_kv_lens: Optional[torch.Tensor] = None,
):
    """
    FlashAttention interface for variable-length sequences, with optional
    paged KV cache support.

    Args:
        q, k, v: Query, key, value tensors.
        max_seqlen_q: Maximum query sequence length in the batch.
        cu_seqlens_q: Cumulative sequence lengths for queries.
        max_seqlen_k: Maximum key/value sequence length in the batch.
        cu_seqlens_k: Cumulative sequence lengths for keys/values when not
            using paged KV cache.
        seqused_k: Number of tokens used per sequence when using paged KV.
        block_table: Optional block table for paged KV cache.
        num_splits: Backend-specific split parameter (non-KV specific),
            typically used to control work partitioning in some FA versions.
        num_splits_kv: Optional number of splits applied to KV **blocks**
            when using paged KV cache. This is forwarded to the underlying
            C++ FlashAttention op as its ``num_splits`` parameter; the split
            unit is KV blocks, not individual tokens or pages.
        fa_version: FlashAttention backend version selector.
    """
    if not _has_vllm_fa2_varlen_fwd():
        raise RuntimeError(
            "pip vllm-xpu-kernels FA2 extension is unavailable: "
            f"{FA2_UNAVAILABLE_REASON}")

    assert cu_seqlens_k is not None or seqused_k is not None, \
        "cu_seqlens_k or seqused_k must be provided"
    assert cu_seqlens_k is None or seqused_k is None, \
        "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert block_table is None or seqused_k is not None, \
        "when enable block_table, seqused_k is needed"
    assert block_table is not None or cu_seqlens_k is not None, \
        "when block_table is disabled, cu_seqlens_k is needed"

    if softmax_scale is None:
        softmax_scale = q.shape[-1]**(-0.5)
    if k_descale is not None:
        assert sum(k_descale.stride()) == 0 and \
            k_descale.dtype == torch.float32, \
            "k_descale must be view of single float32 scalar tensor"
    if v_descale is not None:
        assert sum(v_descale.stride()) == 0 and \
            v_descale.dtype == torch.float32, \
            "v_descale must be view of single float32 scalar tensor"
    # custom op does not support non-tuple input
    real_window_size: tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)

    if fa_version == 2:
        if scheduler_metadata is not None and q_descale is not None \
            and k_descale is not None and v_descale is not None:
            raise NotImplementedError(
                "FA2 does not support scheduler_metadata, q_descale, "
                "k_descale, v_descale")
        if num_splits > 1:
            raise NotImplementedError("FA2 does not support num_splits > 1")
        if q_descale is not None:
            raise NotImplementedError("FA2 does not support q_descale")
        if scheduler_metadata is not None:
            raise NotImplementedError(
                "FA2 does not support scheduler_metadata")
        if (k_descale is not None
                and v_descale is None) or (k_descale is None
                                           and v_descale is not None):
            raise NotImplementedError(
                "FA2 only supports both KV cache descaled")

        # Compute per-seq splits and work_list on host, upload to device.
        # Only enable for decode (max_seqlen_q == 1) with paged KV cache,
        # multi-seq batches, and global num_splits_kv > 1.
        splits_per_seq_dev = None
        work_list_dev = None
        if (block_table is not None and host_kv_lens is not None
                and num_splits_kv is not None and num_splits_kv > 1
                and max_seqlen_q == 1):
            block_size = k.size(1)
            kv_tile = _kv_tile_from_block_size(block_size)
            num_xe_cores = _infer_num_xe_cores(q.device)
            num_heads_kv = k.size(2)
            splits_cpu, work_list_cpu = build_decode_split_plan(
                host_kv_lens,
                kv_tile=kv_tile,
                num_kv_splits=num_splits_kv,
                num_xe_cores=num_xe_cores,
                num_heads_kv=num_heads_kv,
            )
            if work_list_cpu.numel() > 0:
                splits_per_seq_dev = splits_cpu.to(
                    device=q.device, non_blocking=True)
                work_list_dev = work_list_cpu.to(
                    device=q.device, non_blocking=True)

        if start_event is not None:
            start_event.record()
        out, softmax_lse = torch.ops._vllm_fa2_C.varlen_fwd(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
            # still wants it so we pass all zeros
            dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
            seqused_k,
            None,
            block_table,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            k_descale,
            v_descale,
            softmax_scale,
            s_aux,
            False,
            causal,
            real_window_size[0],
            real_window_size[1],
            softcap,
            return_softmax_lse and dropout_p > 0,
            None,
            num_splits_kv,
            is_mix_batch,
            splits_per_seq_dev,
            work_list_dev,
        )
        if end_event is not None:
            end_event.record()
    else:
        raise NotImplementedError("not support yet")
    return (out, softmax_lse) if return_softmax_lse else (out)
