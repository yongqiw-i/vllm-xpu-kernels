# SPDX-License-Identifier: Apache-2.0
from importlib import metadata
from pathlib import Path
from typing import Optional

import torch

try:
    from flash_attn.flash_attn_interface_xpu import (
        flash_attn_varlen_func as xattn_flash_attn_varlen_func,
        flash_attn_with_kvcache as xattn_flash_attn_with_kvcache,
    )
    XATTN_FLASH_ATTN_UNAVAILABLE_REASON = None
except ImportError as e:
    xattn_flash_attn_varlen_func = None
    xattn_flash_attn_with_kvcache = None
    XATTN_FLASH_ATTN_UNAVAILABLE_REASON = str(e)

# get_scheduler_metadata is optional: older xattn builds don't export it.
# When absent, xattn falls back to its internal kernel-side auto split-K
# heuristic (scheduler_metadata=None passed to flash_attn_with_kvcache).
try:
    from flash_attn.flash_attn_interface_xpu import (
        get_scheduler_metadata as xattn_get_scheduler_metadata,
    )
except ImportError:
    xattn_get_scheduler_metadata = None

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
    return None


def _load_vllm_fa2_extension():
    if _has_vllm_fa2_varlen_fwd():
        return None
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

    extension_path = _find_installed_vllm_fa2_extension()
    if extension_path is None:
        return str(import_error)
    try:
        torch.ops.load_library(str(extension_path))
    except OSError as e:
        return f"{import_error}; load_library({extension_path}) failed: {e}"
    if not _has_vllm_fa2_varlen_fwd():
        return f"loaded {extension_path}, but _vllm_fa2_C.varlen_fwd is absent"
    return None


FA2_UNAVAILABLE_REASON = _load_vllm_fa2_extension()
FA2_AVAILABLE = FA2_UNAVAILABLE_REASON is None


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _normalize_flash_attn_result(result, return_softmax_lse: bool):
    if isinstance(result, tuple):
        return result[:2] if return_softmax_lse else result[0]
    return result


def flash_attn_varlen_func_xattn(
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
    if xattn_flash_attn_varlen_func is None or xattn_flash_attn_with_kvcache is None:
        raise RuntimeError(
            "pip flash_attn XPU kernels are unavailable: "
            f"{XATTN_FLASH_ATTN_UNAVAILABLE_REASON}")
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

    # Build the xattn scheduler_metadata on the host/C++ side *before* the
    # timed region (start_event/end_event), so the per-call split-K plan is
    # not charged to the kernel when provider timing is collected. Mirrors
    # the build_decode_split_plan path of flash_attn_varlen_func_CalKernelTime.
    # Opt-in: only for the paged decode path with an explicit num_splits_kv > 1.
    # If the caller pre-computed scheduler_metadata (e.g. once outside a timed
    # loop, via build_xattn_decode_scheduler_metadata), we use it as-is.
    if (scheduler_metadata is None
            and block_table is not None
            and max_seqlen_q == 1
            and seqused_k is not None
            and num_splits_kv is not None):
        scheduler_metadata = build_xattn_decode_scheduler_metadata(
            q, k, v, max_seqlen_q, max_seqlen_k, cu_seqlens_q, seqused_k,
            num_splits_kv=num_splits_kv, causal=causal,
            window_size=real_window_size, softcap=softcap)

    if start_event is not None:
        start_event.record()
    if block_table is not None and max_seqlen_q == 1:
        result = xattn_flash_attn_with_kvcache(
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
        result = xattn_flash_attn_varlen_func(
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

        # Ceil-div partitioning: MUST match the kernel's Reduce stage which
        # uses ceil_div(windowed_k_blocks, seq_num_kv_splits) to compute
        # num_blocks_per_split and breaks when i * num_blocks_per_split >=
        # windowed_k_blocks (paged_decode_kernel.hpp:687,804). If we used
        # balanced divmod here, FMHA would write splits that the reducer
        # never reads -> silent data loss / numerical error.
        num_blocks_per_split = (n_tiles + n_splits - 1) // n_splits
        for s in range(n_splits):
            start = s * num_blocks_per_split
            count = min(n_tiles - start, num_blocks_per_split)
            if count <= 0:
                break
            work_items.append([i, start, count, s])

    splits_t = torch.tensor(splits_per_seq, dtype=torch.int32)
    work_t = torch.tensor(work_items, dtype=torch.int32)
    return splits_t, work_t


def build_xattn_decode_scheduler_metadata(
    q,
    k,
    v,
    max_seqlen_q,
    max_seqlen_k,
    cu_seqlens_q,
    seqused_k,
    num_splits_kv: int,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
):
    """Pre-compute the xattn scheduler_metadata tensor for the paged decode path.

    This is the xattn counterpart of build_decode_split_plan: it produces the
    scheduler plan on the host/C++ side so the kernel can be called with a
    ready-made metadata tensor. Call this *outside* the timed region and pass
    the result via ``scheduler_metadata=`` so the per-iter get_scheduler_metadata
    GPU kernel is not charged to the kernel timing.

    Returns None when the xattn package is unavailable or seqused_k is None.
    ``num_splits_kv <= 0`` selects the xattn auto heuristic (delegates to
    num_splits_heuristic inside get_scheduler_metadata), matching the
    production default (num_splits=0). ``num_splits_kv > 0`` is used as the
    cap.
    """
    if (xattn_get_scheduler_metadata is None
            or seqused_k is None
            or num_splits_kv is None):
        return None
    batch_size = cu_seqlens_q.size(0) - 1
    page_size = k.size(1)
    return xattn_get_scheduler_metadata(
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=q.size(2),
        num_heads_kv=k.size(2),
        headdim=q.size(-1),
        cache_seqlens=seqused_k,
        qkv_dtype=q.dtype,
        headdim_v=v.size(-1),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=None,
        cache_leftpad=None,
        page_size=page_size,
        max_seqlen_k_new=0,
        causal=causal,
        window_size=window_size,
        attention_chunk=0,
        has_softcap=softcap > 0,
        num_splits=num_splits_kv,
        pack_gqa=None,
        sm_margin=0,
    )


def _vllm_get_num_splits(
    num_xe_cores: int,
    batch_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    max_seqlen_k: int,
    block_size: int,
) -> int:
    """Python port of the vllm kernel's get_num_splits host heuristic.

    Mirrors csrc/flash_attn/flash_api.cpp:get_num_splits (pure-host CPU
    computation, no GPU calls). Used when num_splits_kv <= 0 (auto) so the
    vllm host plan can self-tune its split cap — apple-to-apple with xattn's
    get_scheduler_metadata(num_splits=0) which delegates to its own
    num_splits_heuristic.
    """
    if block_size == 16:
        kv_tile, sg_per_wg, policy_split_cap = 16, 1, 16
    elif block_size == 32:
        kv_tile, sg_per_wg, policy_split_cap = 32, 2, 32
    else:
        kv_tile, sg_per_wg, policy_split_cap = 64, 4, 64

    kv_tiles = (max_seqlen_k + kv_tile - 1) // kv_tile
    if kv_tiles < 16:
        return 1

    num_wg_slots = num_xe_cores * 4 // sg_per_wg
    wgs_per_split = batch_size * num_heads_kv
    if wgs_per_split >= num_wg_slots and kv_tiles < 64:
        return 1

    splits_par = max(1, (4 * num_wg_slots + wgs_per_split - 1) // wgs_per_split)
    splits_bw = max(1, kv_tiles // 12)
    splits = max(splits_par, splits_bw)

    red_work = max(1, batch_size * num_heads_q)
    red_cap = max(2, 128 * num_xe_cores // red_work)
    splits = min(splits, red_cap)

    max_splits_tiles = max(1, kv_tiles // 4)
    return max(1, min(splits, max_splits_tiles, 32, policy_split_cap))


def build_vllm_decode_split_plan(q, k, host_kv_lens, num_splits_kv: int):
    """Pre-compute the vllm decode split-K plan (splits_per_seq, work_list).

    vllm counterpart of build_xattn_decode_scheduler_metadata: produces the
    host-side per-seq split plan and uploads it to the device once, so the
    kernel can be called with ready-made tensors. Call this *outside* the
    timed region and pass the result via ``splits_per_seq_dev=`` /
    ``work_list_dev=`` so build_decode_split_plan + the H2D copies are not
    re-run (and thus not timed) on every iteration — apple-to-apple with the
    xattn pre-compute path.

    ``num_splits_kv <= 0`` selects the auto heuristic (Python port of the
    kernel's get_num_splits), matching the production default (vllm engine
    passes num_splits_kv=None -> kernel auto). This is the apple-to-apple
    counterpart of xattn's get_scheduler_metadata(num_splits=0).

    Returns (None, None, 0) when host_kv_lens is None or the plan is empty.
    Otherwise returns (splits_per_seq_dev, work_list_dev, effective_cap) where
    effective_cap is the actual num_splits value used (either the caller's
    cap or the auto-computed one) — this MUST be forwarded to the C++ kernel
    as num_splits so it allocates matching tmp_out/max_logits/exp_sums buffers.
    """
    if host_kv_lens is None:
        return None, None, 0
    block_size = k.size(1)
    kv_tile = _kv_tile_from_block_size(block_size)
    num_xe_cores = _infer_num_xe_cores(q.device)
    num_heads_kv = k.size(2)
    num_heads_q = q.size(1)
    batch_size = host_kv_lens.size(0) if isinstance(host_kv_lens, torch.Tensor) \
        else len(host_kv_lens)
    if num_splits_kv is None or num_splits_kv <= 0:
        # auto: mirror the kernel's get_num_splits host heuristic
        max_seqlen_k = int(host_kv_lens.max()) if isinstance(host_kv_lens, torch.Tensor) \
            else max(host_kv_lens)
        num_splits_kv = _vllm_get_num_splits(
            num_xe_cores, batch_size, num_heads_q, num_heads_kv,
            max_seqlen_k, block_size)
    if num_splits_kv <= 1:
        return None, None, 0
    splits_cpu, work_list_cpu = build_decode_split_plan(
        host_kv_lens,
        kv_tile=kv_tile,
        num_kv_splits=num_splits_kv,
        num_xe_cores=num_xe_cores,
        num_heads_kv=num_heads_kv,
    )
    if work_list_cpu.numel() == 0:
        return None, None, 0
    # actual_max_split is the real max split count across seqs. The C++ kernel
    # uses num_splits to size tmp_out/max_logits/exp_sums buffers AND to decide
    # split vs non-split path. When actual_max == 1 (all seqs single-split),
    # we must NOT pass the host plan — the kernel's non-split path does not
    # expect splits_per_seq/work_list, and passing them with num_splits=1
    # corrupts dispatch -> device lost. Return None so the wrapper passes None
    # to C++, letting the kernel take its non-split fast path cleanly.
    actual_max_split = int(splits_cpu.max().item())
    if actual_max_split <= 1:
        return None, None, 0
    return (splits_cpu.to(device=q.device, non_blocking=True),
            work_list_cpu.to(device=q.device, non_blocking=True),
            actual_max_split)


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
    splits_per_seq_dev: Optional[torch.Tensor] = None,
    work_list_dev: Optional[torch.Tensor] = None,
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
        splits_per_seq_dev / work_list_dev: pre-computed decode split-K plan
            (from build_vllm_decode_split_plan). If provided, the host-side
            plan is not rebuilt, so the metadata preparation stays out of the
            timed region.
    """
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
        # multi-seq batches, and num_splits_kv set. If the caller pre-computed
        # the plan (build_vllm_decode_split_plan, outside the timed loop), use
        # it as-is so the host plan + H2D copies are not re-run per call.
        # effective_num_splits_kv is the actual cap used for the plan — it MUST
        # be forwarded to the C++ kernel as num_splits so it allocates matching
        # tmp_out/max_logits/exp_sums buffers. When no host plan is built
        # (splits_per_seq_dev stays None), pass None to C++ so the kernel runs
        # its own get_num_splits auto heuristic.
        effective_num_splits_kv = num_splits_kv
        if (splits_per_seq_dev is None and work_list_dev is None
                and block_table is not None and host_kv_lens is not None
                and num_splits_kv is not None
                and max_seqlen_q == 1):
            splits_per_seq_dev, work_list_dev, effective_num_splits_kv = \
                build_vllm_decode_split_plan(
                    q, k, host_kv_lens, num_splits_kv=num_splits_kv)
        # If a host plan exists, C++ must receive an explicit positive split
        # count (optional<int> num_splits). Passing 0 is NOT auto in C++;
        # it's a literal 0 and can crash. Derive from splits_per_seq when the
        # caller pre-computed a plan but left num_splits_kv at 0/None.
        if splits_per_seq_dev is not None:
            if effective_num_splits_kv is None or effective_num_splits_kv <= 0:
                effective_num_splits_kv = int(splits_per_seq_dev.max().item())
            c_num_splits = int(effective_num_splits_kv)
        else:
            c_num_splits = None

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
            c_num_splits,
            is_mix_batch,
            splits_per_seq_dev,
            work_list_dev,
        )
        if end_event is not None:
            end_event.record()
    else:
        raise NotImplementedError("not support yet")
    return (out, softmax_lse) if return_softmax_lse else (out)


def flash_attn_varlen_func_vllm(
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
    splits_per_seq_dev: Optional[torch.Tensor] = None,
    work_list_dev: Optional[torch.Tensor] = None,
):
    """vLLM provider adapter for the FA2 varlen C kernel.

    Mirrors flash_attn_varlen_func_CalKernelTime: the per-seq decode split-K
    plan (build_decode_split_plan) is computed on the host *before* the
    start_event/end_event bracket, so the metadata preparation is not charged
    to the kernel when provider timing is collected (same rationale as
    kernel-benchmark's op-level event bracketing). The plan is opt-in: only
    the paged decode path (block_table set, max_seqlen_q == 1) with an
    explicit host_kv_lens and num_splits_kv > 1 engages it. Pass
    splits_per_seq_dev / work_list_dev (from build_vllm_decode_split_plan) to
    supply a pre-computed plan and skip the in-function rebuild — this is the
    apple-to-apple counterpart of passing a pre-computed scheduler_metadata to
    the xattn adapter.
    """
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
        # multi-seq batches, and num_splits_kv set. The plan is built before
        # start_event so it stays outside the timed region. If the caller
        # pre-computed it (build_vllm_decode_split_plan, outside the timed
        # loop), use it as-is.
        # effective_num_splits_kv is the actual cap used for the plan — it MUST
        # be forwarded to the C++ kernel as num_splits so it allocates matching
        # tmp_out/max_logits/exp_sums buffers. When no host plan is built
        # (splits_per_seq_dev stays None), pass None to C++ so the kernel runs
        # its own get_num_splits auto heuristic.
        effective_num_splits_kv = num_splits_kv
        if (splits_per_seq_dev is None and work_list_dev is None
                and block_table is not None and host_kv_lens is not None
                and num_splits_kv is not None
                and max_seqlen_q == 1):
            splits_per_seq_dev, work_list_dev, effective_num_splits_kv = \
                build_vllm_decode_split_plan(
                    q, k, host_kv_lens, num_splits_kv=num_splits_kv)
        if splits_per_seq_dev is not None:
            if effective_num_splits_kv is None or effective_num_splits_kv <= 0:
                effective_num_splits_kv = int(splits_per_seq_dev.max().item())
            c_num_splits = int(effective_num_splits_kv)
        else:
            c_num_splits = None

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
            c_num_splits,
            is_mix_batch,
            splits_per_seq_dev,
            work_list_dev,
        )
        if end_event is not None:
            end_event.record()
    else:
        raise NotImplementedError("not support yet")
    return (out, softmax_lse) if return_softmax_lse else (out)
