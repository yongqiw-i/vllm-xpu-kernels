# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

# isort: off
import csv
import gc
import math
import os
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def _suppress_fd_stderr():
    """Temporarily redirect fd 2 (C-level stderr) to /dev/null.

    The Intel ITT/USDT library prints "USDT:... ActivityProfilerController.cpp
    profiler_start/stop" lines directly to fd 2 on every torch.profiler
    enter/exit. Python-level sys.stderr redirects and ITT_LOG_LEVEL don't
    catch these because they bypass the Python stdio layer.
    """
    save_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(save_fd, 2)
        os.close(devnull_fd)
        os.close(save_fd)

# Pin this process to a single physical GPU at the Level-Zero driver level
# (not just "current device" inside PyTorch) *before* torch/xpu is ever
# touched. On a shared multi-GPU host, an unpinned process can have tensors
# land on different dies across runs, or be scheduled alongside other
# processes' work on sibling GPUs -- both indistinguishable from a real
# kernel regression when only looking at wall-clock latency. Respect an
# externally-exported ZE_AFFINITY_MASK (e.g. set by a CI runner) if present.
os.environ.setdefault("ZE_AFFINITY_MASK", "0")

import torch
import triton

from utils import (bootstrap_benchmark_env, ensure_save_path_exists,
                   extract_attention_profiled_us)

bootstrap_benchmark_env(__file__)

from benchmark.src.flash_attn_interface_ import (
    build_vllm_decode_split_plan, build_xattn_decode_scheduler_metadata,
    flash_attn_varlen_func_vllm, flash_attn_varlen_func_xattn)
from benchmark.src.get_model_config import (
    gen_cutlass_flash_attn_decode_correctness_configs as
    gen_correctness_config)
from benchmark.src.get_model_config import (
    gen_cutlass_flash_attn_decode_perf_configs as gen_perf_configs)
from tests.flash_attn.test_flash_attn_varlen_func import ref_paged_attn
from tests.utils import parse_args, seed_everything
from benchmark.presets import get_hardware_preset
# isort: on

DEVICE = "xpu"
_BENCHMARK_RESULT_CACHE = {}
_BENCHMARK_INPUT_CACHE = {}
_PROFILE_LAYER_STATE = {}
_PROFILE_LAYER_WARNED = set()

# Per-seq decode split-K plan cap. 0 = auto: each provider's host heuristic
# decides the cap (vllm: _vllm_get_num_splits, port of the kernel's
# get_num_splits; xattn: num_splits_heuristic inside get_scheduler_metadata).
# This matches the production default (vllm engine / xattn callers pass
# num_splits=0/None -> kernel auto). Both providers still pre-compute the
# host plan (split per seq / scheduler_metadata) once, outside the timed
# loop, so the plan is delivered ready-made — apple-to-apple.
DECODE_NUM_SPLITS_KV = 0


def clear_xpu_cache():
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


def timed_median_us(run_fn, num_indices, warmup_frac=0.4, min_warmup=10,
                    max_blocks=5, min_block_size=4, target_block_us=2000.0):
    """Time ``run_fn(i % num_indices)`` and return a noise-robust
    per-call latency in microseconds.

    A single aggregate elapsed_time() over one long timed span (the
    original approach) silently absorbs any drift/outliers within that
    window. Splitting into several fixed-size timed blocks and taking the
    median (an earlier iteration of this helper) helps, but on this box
    fast (tens-of-us) decode kernels still showed large, *bimodal* noise
    (e.g. 30us vs 130us) even across blocks. Root-caused via direct
    experiment to two things:
      1. Small blocks -- if a block's total wall time is only slightly
         longer than one GPU clock ramp-up/idle-to-boost transition, the
         *whole block's average* gets skewed by that one-time cost. Fix:
         size blocks by a *target wall-clock duration* (calibrated from a
         quick probe) instead of a fixed iteration count, so the steady
         -state portion dominates the transition.
      2. ``torch.xpu.synchronize()`` between blocks fully drains the
         device queue, which lets the GPU drop to an idle/low-power clock
         state; the *next* block then pays a fresh ramp-up tax. Fix:
         queue all blocks back-to-back and synchronize only once, after
         the last block, so the GPU never goes idle mid-measurement.
    A residual ramp can still land in the first post-warmup block, so we
    additionally drop it from the statistics once we have >=3 blocks.

    Returns (median_us, stdev_us, per_block_us) so callers can also
    surface the noise level alongside the point estimate.
    """
    warmup_n = max(min_warmup, int(num_indices * warmup_frac))
    warmup_n = min(warmup_n, max(min_warmup, num_indices - min_block_size))

    # Calibrate per-call latency on the tail of the warm-up so we can size
    # blocks by target wall-clock duration rather than a fixed count.
    calib_n = max(1, min(warmup_n, 5))
    for i in range(warmup_n - calib_n):
        run_fn(i % num_indices)
    torch.xpu.synchronize()
    calib_start = torch.xpu.Event(enable_timing=True)
    calib_end = torch.xpu.Event(enable_timing=True)
    calib_start.record()
    for i in range(warmup_n - calib_n, warmup_n):
        run_fn(i % num_indices)
    calib_end.record()
    torch.xpu.synchronize()
    per_call_us = max(
        0.1, calib_start.elapsed_time(calib_end) * 1000.0 / calib_n)

    block_size = max(min_block_size, round(target_block_us / per_call_us))
    n_blocks = max(1, max_blocks)

    # Queue every block back-to-back with only event markers in between;
    # a single final synchronize() avoids the idle-gap clock reset that a
    # per-block torch.xpu.synchronize() would otherwise cause.
    events = []
    idx = warmup_n
    for _ in range(n_blocks):
        start_event = torch.xpu.Event(enable_timing=True)
        start_event.record()
        for i in range(idx, idx + block_size):
            run_fn(i % num_indices)
        end_event = torch.xpu.Event(enable_timing=True)
        end_event.record()
        events.append((start_event, end_event))
        idx += block_size
    torch.xpu.synchronize()
    per_block_us = [start_event.elapsed_time(end_event) * 1000.0 / block_size
                    for start_event, end_event in events]

    # The first post-warmup block is the likeliest to still catch a
    # residual clock transition; drop it once we have enough blocks left
    # to still get a meaningful median.
    stats_blocks = per_block_us[1:] if len(per_block_us) >= 3 else per_block_us

    sorted_blocks = sorted(stats_blocks)
    mid = len(sorted_blocks) // 2
    if len(sorted_blocks) % 2 == 1:
        median_us = sorted_blocks[mid]
    else:
        median_us = (sorted_blocks[mid - 1] + sorted_blocks[mid]) / 2.0
    mean_us = sum(stats_blocks) / len(stats_blocks)
    var_us = sum((x - mean_us) ** 2
                for x in stats_blocks) / max(1, len(stats_blocks) - 1)
    stdev_us = var_us ** 0.5
    return median_us, stdev_us, per_block_us


def timed_device_us(fn, num_indices, iters=200, warmup=20):
    """Measure per-call GPU kernel self-time via torch.profiler.

    This is the apple-to-apple timing layer: it sums only attention-kernel
    device time from the profiler table and excludes non-attention GPU ops
    launched by wrapper logic (asserts, maybe_contiguous stride checks,
    empty_like pool allocs, metadata prep, etc.).

    Ported from kernel-benchmark's ``profile_device_us`` (common.py). ``fn``
    is invoked as ``fn(i % num_indices)``. Runs two independent profile
    passes and returns ``(mean_of_two, half_diff, meta)`` where
    ``meta['layer']`` is the selected profiling layer and
    ``meta['top_ops']`` is the selected top-op list. The half-diff between
    the two passes stands in for stdev so the caller's
    ``stdev > 0.1 * us`` noise alarm still fires when the two passes disagree
    by more than ~20% (profiler yields a single aggregate per pass, so a
    real per-block stdev is unavailable).
    """
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "XPU"):
        activities.append(ProfilerActivity.XPU)

    n = iters - warmup
    assert n > 0, f"iters ({iters}) must exceed warmup ({warmup})"

    def _one_pass():
        for i in range(warmup):
            fn(i % num_indices)
        torch.xpu.synchronize()
        with _suppress_fd_stderr(), profile(activities=activities) as prof:
            for i in range(n):
                fn(i % num_indices)
            torch.xpu.synchronize()
        return prof

    us_samples = []
    top_ops = []
    layers = []
    for _ in range(2):
        evs = _one_pass().key_averages()
        total, matched_top, all_top, layer = extract_attention_profiled_us(evs,
                                                                           n)
        if total <= 0:
            attrs = [a for a in dir(evs[0]) if "time" in a.lower()] \
                if len(evs) else []
            print("  [warn] attention profiler time read as 0; available "
                  f"*time* attrs: {attrs}")
            if all_top:
                print("  [warn] top device ops (us/iter): "
                      + ", ".join(f"{name}={us:.1f}" for name, us in all_top))
            return 0.0, 0.0, {"layer": "none", "top_ops": []}
        us_samples.append(total)
        top_ops = matched_top
        layers.append(layer)

    median_us = sum(us_samples) / len(us_samples)
    stdev_us = abs(us_samples[0] - us_samples[1]) / 2.0
    selected_layer = layers[0] if layers and all(
        layer == layers[0] for layer in layers) else "mixed"
    return median_us, stdev_us, {
        "layer": selected_layer,
        "top_ops": top_ops,
    }


def _track_profile_layer(config, provider_prefix, profile_meta):
    if provider_prefix not in ("vllm_xpu_kernels", "xattention"):
        return
    if not isinstance(profile_meta, dict):
        return
    layer = profile_meta.get("layer")
    if not layer:
        return

    key = tuple(config)
    state = _PROFILE_LAYER_STATE.setdefault(key, {})
    state[provider_prefix] = layer
    if ("vllm_xpu_kernels" in state and "xattention" in state
            and state["vllm_xpu_kernels"] != state["xattention"]
            and key not in _PROFILE_LAYER_WARNED):
        print("  [warn] profiling layer mismatch for this config: "
              f"vllm_xpu_kernels={state['vllm_xpu_kernels']}, "
              f"xattention={state['xattention']}",
              flush=True)
        _PROFILE_LAYER_WARNED.add(key)


def calculate_memory_usage(q_len_sum, kv_len_sum, num_heads, head_size,
                           query_dtype, kv_dtype, output_dtype):
    # Memory for query, key and value caches, and output
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    query_memory = q_len_sum * num_query_heads * head_size * \
        torch.tensor([], dtype=query_dtype).element_size()
    kv_cache_memory = 2 * kv_len_sum * num_kv_heads * \
        head_size * torch.tensor([], dtype=kv_dtype).element_size()
    output_memory = q_len_sum * num_query_heads * head_size * \
        torch.tensor([], dtype=output_dtype).element_size()
    # Convert to GB
    return (query_memory + kv_cache_memory + output_memory) / (1000**3)


def _safe_ratio(numerator, denominator):
    if (not math.isfinite(numerator) or not math.isfinite(denominator)
            or denominator == 0):
        return float("nan")
    return numerator / denominator


def _benchmark_cache_key(config, provider, iterations):
    return (tuple(config), provider, iterations)


def _cache_result(cache_key, value):
    _BENCHMARK_RESULT_CACHE[cache_key] = value
    return value


def append_speedup_average_row(save_path, plot_name):
    if not save_path:
        return
    csv_path = Path(save_path) / f"{plot_name}.csv"
    if not csv_path.exists():
        return

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return
        rows = [
            row for row in reader
            if row.get(fieldnames[0]) != "Average Speedup"
        ]

    speedup_cols = [name for name in fieldnames if "Speedup" in name]
    if not rows or not speedup_cols:
        return

    avg_row = {name: "" for name in fieldnames}
    avg_row[fieldnames[0]] = "Average Speedup"
    for col in speedup_cols:
        values = []
        for row in rows:
            try:
                value = float(row[col])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if values:
            avg_row[col] = f"{sum(values) / len(values):.6f}"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(avg_row)
    print(f"Wrote speedup averages to {csv_path}")


def make_decode_with_paged_kv_input(config):
    seq_lens, num_heads, head_size, block_size, \
    output_dtype, _, num_blocks, _, q_dtype, is_sink = config
    # if num_heads == (16, 1) and head_size == 256:
    #     pytest.skip("skip test cases that may run out of SLM.")
    num_seqs = int(seq_lens.split(",")[0])
    query_lens = list(map(lambda x: int(x), seq_lens.split(",")[1].split("+")))
    kv_lens = list(map(lambda x: int(x), seq_lens.split(",")[2].split("+")))
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=output_dtype)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=output_dtype)
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)

    seq_k = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32)
    sink = None
    if is_sink:
        sink = torch.randn(num_query_heads, dtype=output_dtype)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None  #noqa: F841
    k_descale = None  #noqa: F841
    v_descale = None  #noqa: F841
    if q_dtype is not None:
        # QKV are drawn from N(0, 1): no need for a fp8 scaling factor
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)

        scale_shape = (num_seqs, num_kv_heads)
        q_descale = torch.ones(scale_shape, dtype=torch.float32)  #noqa: F841
        k_descale = torch.ones(scale_shape, dtype=torch.float32)  #noqa: F841
        v_descale = torch.ones(scale_shape, dtype=torch.float32)  #noqa: F841
    return maybe_quantized_query, maybe_quantized_key_cache, \
        maybe_quantized_value_cache, max_query_len, cu_query_lens, \
            max_kv_len, seq_k, scale, block_tables, sink, query, \
                key_cache, value_cache, query_lens, kv_lens


def calculate_diff_decode_paged_kv(config):
    _, _, _, _, _, _, _, _, q_dtype, _ = config
    maybe_quantized_query, maybe_quantized_key_cache, \
        maybe_quantized_value_cache, max_query_len, cu_query_lens, \
        max_kv_len, seq_k, scale, block_tables, sink, query, \
        key_cache, value_cache, query_lens, kv_lens = \
        make_decode_with_paged_kv_input(config)

    def run_op(op):
        return op(maybe_quantized_query,
                  maybe_quantized_key_cache,
                  maybe_quantized_value_cache,
                  max_query_len,
                  cu_query_lens,
                  max_kv_len,
                  seqused_k=seq_k,
                  softmax_scale=scale,
                  causal=False,
                  block_table=block_tables,
                  window_size=(-1, -1),
                  s_aux=sink,
                  host_kv_lens=seq_k,
                  num_splits_kv=DECODE_NUM_SPLITS_KV)

    output = run_op(flash_attn_varlen_func_vllm)
    pip_output = None
    pip_error = None
    try:
        pip_output = run_op(flash_attn_varlen_func_xattn)
    except Exception as e:
        pip_error = e

    ref_output = ref_paged_attn(query=query,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                query_lens=query_lens,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                casual=False,
                                is_paged=True,
                                sink=sink,
                                window_size_left=-1,
                                window_size_right=-1)
    atol, rtol = 1e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    vllm_matches = True
    try:
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)
    except AssertionError as e:
        vllm_matches = False
        print("❌ vllm_xpu_kernels differs from reference, ", config,
              " error: ", e)

    if pip_output is None:
        print("Skipping xattention correctness for unsupported "
              f"shape: {pip_error}")
        if vllm_matches:
            print("✅ vllm_xpu_kernels implementation matches, ", config)
        return

    xattention_matches = True
    try:
        torch.testing.assert_close(pip_output, ref_output, atol=atol,
                                   rtol=rtol)
    except AssertionError as e:
        xattention_matches = False
        print("❌ xattention differs from reference, ", config,
              " error: ", e)

    if vllm_matches and xattention_matches:
        print("✅ All implementations match, ", config)


def benchmark_decode_with_paged_kv(seq_lens, num_heads, head_size, block_size,
                                   output_dtype, soft_cap, num_blocks,
                                   fa_versions, q_dtype, is_sink, provider,
                                   iterations):
    config = (seq_lens, num_heads, head_size, block_size, output_dtype,
              soft_cap, num_blocks, fa_versions, q_dtype, is_sink)
    cache_key = _benchmark_cache_key(config, provider, iterations)
    if cache_key in _BENCHMARK_RESULT_CACHE:
        return _BENCHMARK_RESULT_CACHE[cache_key]

    def run_provider(provider_name):
        provider_cache_key = _benchmark_cache_key(config, provider_name,
                                                  iterations)
        if provider_cache_key in _BENCHMARK_RESULT_CACHE:
            return _BENCHMARK_RESULT_CACHE[provider_cache_key]
        return benchmark_decode_with_paged_kv(
            seq_lens=seq_lens,
            num_heads=num_heads,
            head_size=head_size,
            block_size=block_size,
            output_dtype=output_dtype,
            soft_cap=soft_cap,
            num_blocks=num_blocks,
            fa_versions=fa_versions,
            q_dtype=q_dtype,
            is_sink=is_sink,
            provider=provider_name,
            iterations=iterations)

    if provider == "xattention_Time_Speedup":
        vllm_time = run_provider("vllm_xpu_kernels")
        try:
            xattention_time = run_provider("xattention")
        except Exception:
            return _cache_result(cache_key, float("nan"))
        return _cache_result(cache_key, _safe_ratio(vllm_time,
                                                   xattention_time))
    if provider == "xattention_Bandwidth_Speedup":
        vllm_bw = run_provider("vllm_xpu_kernels_memBandwidth")
        try:
            xattention_bw = run_provider("xattention_memBandwidth")
        except Exception:
            return _cache_result(cache_key, float("nan"))
        return _cache_result(cache_key, _safe_ratio(xattention_bw, vllm_bw))
    if provider == "xattention_MBU_Speedup":
        vllm_mbu = run_provider("vllm_xpu_kernels_MBU")
        try:
            xattention_mbu = run_provider("xattention_MBU")
        except Exception:
            return _cache_result(cache_key, float("nan"))
        return _cache_result(cache_key, _safe_ratio(xattention_mbu, vllm_mbu))

    input_cache_key = (tuple(config), iterations)
    if input_cache_key not in _BENCHMARK_INPUT_CACHE:
        if _BENCHMARK_INPUT_CACHE:
            _BENCHMARK_INPUT_CACHE.clear()
            clear_xpu_cache()
        maybe_quantized_query, maybe_quantized_key_cache, \
            maybe_quantized_value_cache, max_query_len, cu_query_lens, \
            max_kv_len, seq_k, scale, _, sink, _, \
            _, _, _, _ = make_decode_with_paged_kv_input(config)

        num_seqs = int(seq_lens.split(",")[0])
        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        queries = [
            torch.rand_like(maybe_quantized_query) for _ in range(iterations)
        ]
        block_tables_list = [
            torch.randint(0,
                          num_blocks,
                          (num_seqs, max_num_blocks_per_seq),
                          dtype=torch.int32)
            for _ in range(iterations)
        ]
        _BENCHMARK_INPUT_CACHE[input_cache_key] = (
            maybe_quantized_query, maybe_quantized_key_cache,
            maybe_quantized_value_cache, max_query_len, cu_query_lens,
            max_kv_len, seq_k, scale, sink, queries, block_tables_list)

    maybe_quantized_query, maybe_quantized_key_cache, \
        maybe_quantized_value_cache, max_query_len, cu_query_lens, \
        max_kv_len, seq_k, scale, sink, \
        queries, block_tables_list = _BENCHMARK_INPUT_CACHE[input_cache_key]

    provider_name = provider.replace("vllm_xpu_kernels", "vllm-xpu-kernels")
    print(f"Running config: {seq_lens, num_heads, head_size, \
                              block_size, output_dtype, soft_cap, num_blocks, \
                              fa_versions, q_dtype, \
                              is_sink}, Provider: {provider_name}",
          flush=True)
    assert iterations > 10, \
    "Number of iterations should be greater than 10 to allow warmup + " \
    "multiple timed blocks (see timed_median_us)"

    use_xattention = provider.startswith("xattention")
    provider_prefix = "xattention" if use_xattention else "vllm_xpu_kernels"

    flash_op = flash_attn_varlen_func_xattn \
        if use_xattention else flash_attn_varlen_func_vllm

    # Pre-compute the per-provider metadata once per config, outside the timed
    # loop, so neither provider pays any metadata-prep cost inside the timed
    # region — apple-to-apple. xattn: get_scheduler_metadata (launches a GPU
    # kernel, must be pre-computed). vllm: build_decode_split_plan + H2D copies
    # (pure-host + small async copies, pre-computed for parity).
    xattn_sched = None
    vllm_splits = None
    vllm_work = None
    if use_xattention:
        xattn_sched = build_xattn_decode_scheduler_metadata(
            maybe_quantized_query, maybe_quantized_key_cache,
            maybe_quantized_value_cache, max_query_len, max_kv_len,
            cu_query_lens, seq_k, num_splits_kv=DECODE_NUM_SPLITS_KV,
            causal=False, window_size=(-1, -1))
        if xattn_sched is None:
            print(f"  [warn] {provider_prefix}: scheduler_metadata pre-compute "
                  f"returned None (xattn package lacks get_scheduler_metadata "
                  f"or num_splits_kv<=1) -> falling back to xattn kernel "
                  f"auto-split; this config is NOT measuring the host-plan "
                  f"path.", flush=True)
    else:
        vllm_splits, vllm_work, vllm_cap = build_vllm_decode_split_plan(
            maybe_quantized_query, maybe_quantized_key_cache, seq_k,
            num_splits_kv=DECODE_NUM_SPLITS_KV)
        if vllm_splits is None:
            print(f"  [warn] {provider_prefix}: split plan pre-compute "
                  f"returned None (host_kv_lens missing, num_splits_kv<=1, "
                  f"or empty work_list) -> falling back to vllm kernel "
                  f"auto-split; this config is NOT measuring the host-plan "
                  f"path.", flush=True)

    if use_xattention:
        def _run(index):
            flash_op(queries[index],
                     maybe_quantized_key_cache,
                     maybe_quantized_value_cache,
                     max_query_len,
                     cu_query_lens,
                     max_kv_len,
                     seqused_k=seq_k,
                     softmax_scale=scale,
                     causal=False,
                     block_table=block_tables_list[index],
                     window_size=(-1, -1),
                     s_aux=sink,
                     num_splits_kv=DECODE_NUM_SPLITS_KV,
                     scheduler_metadata=xattn_sched)
    else:
        def _run(index):
            flash_op(queries[index],
                     maybe_quantized_key_cache,
                     maybe_quantized_value_cache,
                     max_query_len,
                     cu_query_lens,
                     max_kv_len,
                     seqused_k=seq_k,
                     softmax_scale=scale,
                     causal=False,
                     block_table=block_tables_list[index],
                     window_size=(-1, -1),
                     s_aux=sink,
                     num_splits_kv=DECODE_NUM_SPLITS_KV,
                     splits_per_seq_dev=vllm_splits,
                     work_list_dev=vllm_work)

    time_us, time_stdev_us, profile_meta = timed_device_us(_run, iterations)
    _track_profile_layer(config, provider_prefix, profile_meta)
    if time_stdev_us > 0.1 * time_us:
        print(f"  [warn] {provider_prefix} timing noisy: "
              f"median={time_us:.1f}us stdev={time_stdev_us:.1f}us "
              f"({100 * time_stdev_us / time_us:.0f}% of median)",
              flush=True)
    ms = time_us / 1000.0
    memory_load_GB = calculate_memory_usage(
        cu_query_lens[-1].item(),
        seq_k.sum().item(),
        num_heads,
        head_size,
        queries[0].dtype,
        maybe_quantized_key_cache.dtype,
        output_dtype)
    measured_bw = memory_load_GB / (ms / 1000)
    hardware_presets = get_hardware_preset(torch.xpu.get_device_name())
    mbu = float("nan")
    if hardware_presets is not None:
        peak_bw = hardware_presets["memory_bandwidth_GBs"]
        mbu = (measured_bw / peak_bw) * 100

    _cache_result(_benchmark_cache_key(config, provider_prefix, iterations),
                 time_us)
    _cache_result(_benchmark_cache_key(
        config, f"{provider_prefix}_memBandwidth", iterations), measured_bw)
    _cache_result(_benchmark_cache_key(config, f"{provider_prefix}_MBU",
                                      iterations), mbu)
    clear_xpu_cache()
    return _BENCHMARK_RESULT_CACHE[cache_key]


def get_benchmark_decode_with_paged_kv(iterations=50):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "seq_lens", "num_heads", "head_size", "block_size",
                "output_dtype", "soft_cap", "num_blocks", "fa_versions",
                "q_dtype", "is_sink"
            ],
            x_vals=[tuple(c) for c in configs],
            line_arg="provider",
            line_vals=["vllm_xpu_kernels", "vllm_xpu_kernels_memBandwidth",
                       "vllm_xpu_kernels_MBU", "xattention",
                       "xattention_memBandwidth", "xattention_MBU",
                       "xattention_Time_Speedup",
                       "xattention_Bandwidth_Speedup",
                       "xattention_MBU_Speedup"],
            line_names=[
                "vllm-xpu-kernels(us)",
                "vllm-xpu-kernels_memBandwidth(GB/s)",
                "vllm-xpu-kernels_MBU (%)", "xattention(us)",
                "xattention_memBandwidth(GB/s)", "xattention_MBU (%)",
                "Time Speedup", "bandwidth Speedup", "MBU Speedup"
            ],
            styles=[("blue", "-"), ("purple", "-"), ("red", "-"),
                    ("cyan", "-"), ("orange", "-"), ("brown", "-"),
                    ("black", "--"), ("green", "--"), ("pink", "--")],
            ylabel="Latency (us)",
            plot_name="flash-attn-decode",
            args={},
        ))
    def benchmark(seq_lens, num_heads, head_size, block_size, output_dtype,
                  soft_cap, num_blocks, fa_versions, q_dtype, is_sink,
                  provider):
        try:
            return benchmark_decode_with_paged_kv(seq_lens=seq_lens,
                                                  num_heads=num_heads,
                                                  head_size=head_size,
                                                  block_size=block_size,
                                                  output_dtype=output_dtype,
                                                  soft_cap=soft_cap,
                                                  num_blocks=num_blocks,
                                                  fa_versions=fa_versions,
                                                  q_dtype=q_dtype,
                                                  is_sink=is_sink,
                                                  provider=provider,
                                                  iterations=iterations)
        except Exception as e:
            if (not provider.startswith("xattention")
                    or provider.endswith("_Speedup")):
                raise
            print(f"Skipping {provider} for unsupported xattention "
                  f"shape: {e}")
            clear_xpu_cache()
            config = (seq_lens, num_heads, head_size, block_size, output_dtype,
                      soft_cap, num_blocks, fa_versions, q_dtype, is_sink)
            cache_key = _benchmark_cache_key(config, provider, iterations)
            return _cache_result(cache_key, float("nan"))

    return benchmark


def filter_configs(configs):
    new_configs = []
    for config in configs:
        if (config[1] == (16, 1) and config[2] == 256) or \
           (config[3] == 128 and config[6] == 32768 and config[2] >= 192):
            print("Skipping config due to potential OOM: ", config)
            continue
        new_configs.append(config)
    return new_configs


def _mk_cfg(seq_lens, num_heads, block_size, name, head_size=128,
            dtype=torch.bfloat16, num_blocks=2048):
    # 11-tuple matching make_decode_with_paged_kv_input contract.
    return (seq_lens, num_heads, head_size, block_size, dtype, None,
            num_blocks, 2, None, False, name)


# Format: seq_lens="B,1+1+...,kv0+kv1+...", num_heads=(q, kv), block_size, name
BATCH_DECODE_CONFIGS = [
    # Uniform KV
    _mk_cfg("32," + "+".join(["1"] * 32) + "," + "+".join(["512"] * 32),
            (32, 2), 64, "32x512_uniform_(32,2)"),
    _mk_cfg("32," + "+".join(["1"] * 32) + "," + "+".join(["4096"] * 32),
            (32, 2), 64, "32x4096_uniform_(32,2)"),
    # Mixed KV (key optimization target)
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
            (32, 2), 64, "8xmixed_128-16384_(32,2)"),
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
            (32, 4), 64, "8xmixed_128-16384_(32,4)"),
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
            (32, 8), 64, "8xmixed_128-16384_(32,8)"),
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
            (40, 8), 64, "8xmixed_128-16384_(40,8)"),
    # Skewed (mostly short + one long)
    _mk_cfg("8,1+1+1+1+1+1+1+1,256+256+256+256+256+256+256+16384",
            (32, 2), 64, "8xskewed_256-16384_(32,2)"),
    # All short
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+128+256+256+512+512+1024+1024",
            (32, 2), 64, "8xshort_128-1024_(32,2)"),
    # Realistic vLLM-like
    _mk_cfg(
        "16," + "+".join(["1"] * 16) + ","
        + "+".join(["256", "512", "1024", "1024",
                    "2048", "2048", "4096", "4096",
                    "4096", "8192", "8192", "8192",
                    "16384", "16384", "16384", "16384"]),
        (32, 2), 64, "16xrealistic_mixed_(32,2)"),
    # block_size=128
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
            (32, 2), 128, "8xmixed_128-16384_bs128_(32,2)"),
    # MHA
    _mk_cfg("32," + "+".join(["1"] * 32) + "," + "+".join(["512"] * 32),
            (16, 16), 64, "32x512_uniform_(16,16)_MHA"),
    _mk_cfg("8,1+1+1+1+1+1+1+1,128+256+512+1024+2048+4096+8192+16384",
            (16, 16), 64, "8xmixed_128-16384_(16,16)_MHA"),
]


def benchmark_batch_decode(config, iterations=200, use_xattention=False):
    """Benchmark a single batch decode config with GPU-event timing."""
    (seq_lens, num_heads, head_size, block_size, dtype, soft_cap,
     num_blocks, fa_versions, q_dtype, is_sink, name) = config

    full_config = (seq_lens, num_heads, head_size, block_size, dtype,
                   soft_cap, num_blocks, fa_versions, q_dtype, is_sink)
    (maybe_quantized_query, maybe_quantized_key_cache,
     maybe_quantized_value_cache, max_query_len, cu_query_lens,
     max_kv_len, seq_k, scale, block_tables, sink, _,
     _, _, _, _) = make_decode_with_paged_kv_input(full_config)

    num_seqs = int(seq_lens.split(",")[0])
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    queries = [torch.rand_like(maybe_quantized_query)
               for _ in range(iterations)]
    bt_list = [torch.randint(0, num_blocks,
                             (num_seqs, max_num_blocks_per_seq),
                             dtype=torch.int32)
               for _ in range(iterations)]
    flash_op = flash_attn_varlen_func_xattn \
        if use_xattention else flash_attn_varlen_func_vllm

    # Pre-compute per-provider metadata once per config (outside the timed
    # loop) so neither provider pays metadata-prep cost inside timing —
    # apple-to-apple. xattn: get_scheduler_metadata (GPU kernel). vllm:
    # build_decode_split_plan + H2D (pure-host + small async copies).
    xattn_sched = None
    vllm_splits = None
    vllm_work = None
    if use_xattention:
        xattn_sched = build_xattn_decode_scheduler_metadata(
            maybe_quantized_query, maybe_quantized_key_cache,
            maybe_quantized_value_cache, max_query_len, max_kv_len,
            cu_query_lens, seq_k, num_splits_kv=DECODE_NUM_SPLITS_KV,
            causal=False, window_size=(-1, -1))
        if xattn_sched is None:
            print("  [warn] xattention: scheduler_metadata pre-compute "
                  "returned None (xattn package lacks get_scheduler_metadata "
                  "or num_splits_kv<=1) -> falling back to xattn kernel "
                  "auto-split; this config is NOT measuring the host-plan "
                  "path.", flush=True)
    else:
        vllm_splits, vllm_work, vllm_cap = build_vllm_decode_split_plan(
            maybe_quantized_query, maybe_quantized_key_cache, seq_k,
            num_splits_kv=DECODE_NUM_SPLITS_KV)
        if vllm_splits is None:
            print("  [warn] vllm_xpu_kernels: split plan pre-compute "
                  "returned None (host_kv_lens missing, num_splits_kv<=1, "
                  "or empty work_list) -> falling back to vllm kernel "
                  "auto-split; this config is NOT measuring the host-plan "
                  "path.", flush=True)

    if use_xattention:
        def _run(i):
            flash_op(
                queries[i], maybe_quantized_key_cache,
                maybe_quantized_value_cache,
                max_query_len, cu_query_lens, max_kv_len,
                seqused_k=seq_k, softmax_scale=scale,
                causal=False, block_table=bt_list[i],
                window_size=(-1, -1), s_aux=sink,
                num_splits_kv=DECODE_NUM_SPLITS_KV,
                scheduler_metadata=xattn_sched)
    else:
        def _run(i):
            flash_op(
                queries[i], maybe_quantized_key_cache,
                maybe_quantized_value_cache,
                max_query_len, cu_query_lens, max_kv_len,
                seqused_k=seq_k, softmax_scale=scale,
                causal=False, block_table=bt_list[i],
                window_size=(-1, -1), s_aux=sink,
                num_splits_kv=DECODE_NUM_SPLITS_KV,
                splits_per_seq_dev=vllm_splits,
                work_list_dev=vllm_work)

    # Warm-up + median-of-blocks timing (see timed_median_us docstring for
    # why a single aggregate window is noise-prone on shared multi-GPU hosts).
    avg_us, stdev_us, profile_meta = timed_device_us(_run, iterations)
    _track_profile_layer(full_config, "xattention" if use_xattention else
                         "vllm_xpu_kernels", profile_meta)
    if stdev_us > 0.1 * avg_us:
        print(f"  [warn] batch_decode timing noisy: median={avg_us:.1f}us "
              f"stdev={stdev_us:.1f}us ({100 * stdev_us / avg_us:.0f}% of "
              f"median)", flush=True)

    # KV bandwidth (K + V, bf16 -> 2 bytes)
    kv_lens = list(map(int, seq_lens.split(",")[2].split("+")))
    kv_bytes = sum(kv_lens) * num_heads[1] * head_size * 2 * 2
    bw_gbs = (kv_bytes / 1e9) / (avg_us / 1e6)

    clear_xpu_cache()
    return avg_us, bw_gbs


if __name__ == "__main__":

    args = parse_args()
    seed = 1234
    seed_everything(seed)
    iterations = 100
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")

    configs = gen_correctness_config()
    configs = filter_configs(configs)
    for config in configs:
        try:
            calculate_diff_decode_paged_kv(config)
        except Exception as e:
            print("Error in config: ", config, " error: ", e)
        clear_xpu_cache()

    configs = gen_perf_configs()
    configs = filter_configs(configs)
    benchmark = get_benchmark_decode_with_paged_kv(iterations=iterations)
    save_path = ensure_save_path_exists(args.save_path)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=save_path)
    append_speedup_average_row(save_path, "flash-attn-decode")

    # ================================================================
    # Batch Decode Benchmark (per-seq adaptive split-K evaluation)
    # ================================================================
    print("\n" + "=" * 80)
    print("Batch Decode Benchmark (per-seq adaptive split-K)")
    print("=" * 80)
    hdr = (f"{'config':<40} | {'batch':>5} {'kv_sum':>7} | "
           f"{'vllm_us':>9} {'xattention_us':>13} {'speedup':>7} | "
           f"{'vllm_bw':>9} {'xattention_bw':>13}")
    print(hdr)
    print("-" * 80)

    for cfg in BATCH_DECODE_CONFIGS:
        name = cfg[-1]
        seq_lens = cfg[0]
        num_seqs = int(seq_lens.split(",")[0])
        kv_lens = list(map(int, seq_lens.split(",")[2].split("+")))
        kv_sum = sum(kv_lens)
        try:
            avg_us, bw_gbs = benchmark_batch_decode(cfg, iterations=200)
        except Exception as e:
            print(f"{name:<40} | {num_seqs:>5} {kv_sum:>7} | "
                  f"{'ERROR':>9} {str(e)[:20]}")
            clear_xpu_cache()
            continue

        try:
            xattention_avg_us, xattention_bw_gbs = benchmark_batch_decode(
                cfg, iterations=200, use_xattention=True)
            speedup = avg_us / xattention_avg_us
            xattention_us_text = f"{xattention_avg_us:>13.1f}"
            speedup_text = f"{speedup:>7.2f}"
            xattention_bw_text = f"{xattention_bw_gbs:>13.1f}"
        except Exception as e:
            print(f"Skipping batch xattention for unsupported shape "
                  f"{name}: {e}")
            xattention_us_text = f"{float('nan'):>13.1f}"
            speedup_text = f"{float('nan'):>7.2f}"
            xattention_bw_text = f"{float('nan'):>13.1f}"
        print(f"{name:<40} | {num_seqs:>5} {kv_sum:>7} | "
              f"{avg_us:>9.1f} {xattention_us_text} {speedup_text} | "
              f"{bw_gbs:>9.1f} {xattention_bw_text}")
        clear_xpu_cache()

    print("=" * 80)
