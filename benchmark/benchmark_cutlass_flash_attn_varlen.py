# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402

# isort: off
import csv
import gc
import math
from pathlib import Path

import torch
import triton

from utils import bootstrap_benchmark_env, ensure_save_path_exists

bootstrap_benchmark_env(__file__)

from benchmark.src.flash_attn_interface_ import (
    flash_attn_varlen_func_pip)
from benchmark.src.get_model_config import (
    gen_cutlass_flash_attn_varlen_correctness_configs as
    gen_correctness_config)
from benchmark.src.get_model_config import (
    gen_cutlass_flash_attn_varlen_perf_configs as gen_perf_configs)
from tests.flash_attn.test_flash_attn_varlen_func import ref_paged_attn
from tests.utils import parse_args, seed_everything
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func
from benchmark.presets import get_hardware_preset
# isort: on

DEVICE = "xpu"
_BENCHMARK_RESULT_CACHE = {}
_BENCHMARK_INPUT_CACHE = {}


def clear_xpu_cache():
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


def calculate_flops(num_query_heads, query_lens, kv_lens, head_size,
                    is_causal):
    total = 0
    for sq, sk in zip(query_lens, kv_lens):
        effective_sk = sk * 0.5 if is_causal else sk
        total += 4 * num_query_heads * sq * effective_sk * head_size
    return total


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


def make_varlen_with_paged_kv_input(config):
    num_seqs, query_lens, kv_lens, num_heads, head_size, \
        block_size, window_size, output_dtype, _, num_blocks, \
        _, q_dtype, is_sink, is_causal, is_paged, kv_dtype = config
    query_lens = query_lens.split(",")
    query_lens = [int(x) for x in query_lens]
    kv_lens = kv_lens.split(",")
    kv_lens = [int(x) for x in kv_lens]
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
    if is_paged:
        key_cache = torch.randn(num_blocks,
                                block_size,
                                num_kv_heads,
                                head_size,
                                dtype=output_dtype)
    else:
        key_cache = torch.randn(sum(kv_lens),
                                num_query_heads,
                                head_size,
                                dtype=output_dtype)
    value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)
    cu_kv_lens = torch.tensor([0] + kv_lens,
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
    scale_shape = (num_seqs, num_kv_heads)
    is_fp8_query = q_dtype is not None
    if is_fp8_query:
        q_descale = (torch.abs(query).max() / 200).to(torch.float32)
        maybe_quantized_query = (query / q_descale).to(q_dtype)
    is_fp8kv = kv_dtype is not None
    if is_fp8kv:
        k_descale = (torch.abs(key_cache).max() / 200).to(torch.float32)
        v_descale = (torch.abs(value_cache).max() / 200).to(torch.float32)
        maybe_quantized_key_cache = (key_cache / k_descale).to(kv_dtype)
        maybe_quantized_value_cache = (value_cache / v_descale).to(kv_dtype)
    return (maybe_quantized_query, maybe_quantized_key_cache,
            maybe_quantized_value_cache, max_query_len, cu_query_lens,
            max_kv_len, cu_kv_lens, seq_k, q_descale, k_descale, v_descale,
            scale, is_causal, block_tables, window_size, sink, scale_shape,
            query, query_lens, kv_lens, is_fp8kv, is_fp8_query,
            max_num_blocks_per_seq)


def calculate_diff_varlen_paged_kv(config):
    _, _, _, _, _, _, window_size, output_dtype, _, _, _, \
        q_dtype, _, is_causal, is_paged, kv_dtype = config
    maybe_quantized_query, maybe_quantized_key_cache, \
        maybe_quantized_value_cache, \
        max_query_len, cu_query_lens, max_kv_len, cu_kv_lens, \
        seq_k, q_descale, k_descale, v_descale, scale, is_causal, \
        block_tables, window_size, sink, scale_shape, query, \
        query_lens, kv_lens, is_fp8kv, is_fp8_query, _ = \
    make_varlen_with_paged_kv_input(config)

    def run_op(op):
        if is_paged:
            return op(maybe_quantized_query,
                      maybe_quantized_key_cache,
                      maybe_quantized_value_cache,
                      max_query_len,
                      cu_query_lens,
                      max_kv_len,
                      seqused_k=seq_k,
                      q_descale=q_descale.expand(scale_shape)
                      if q_descale is not None else None,
                      k_descale=k_descale.expand(scale_shape)
                      if k_descale is not None else None,
                      v_descale=v_descale.expand(scale_shape)
                      if v_descale is not None else None,
                      softmax_scale=scale,
                      causal=is_causal,
                      block_table=block_tables,
                      window_size=window_size,
                      s_aux=sink)
        return op(maybe_quantized_query,
                  maybe_quantized_key_cache,
                  maybe_quantized_value_cache,
                  max_query_len,
                  cu_query_lens,
                  max_kv_len,
                  cu_seqlens_k=cu_kv_lens,
                  q_descale=q_descale.expand(scale_shape)
                  if q_descale is not None else None,
                  k_descale=k_descale.expand(scale_shape)
                  if k_descale is not None else None,
                  v_descale=v_descale.expand(scale_shape)
                  if v_descale is not None else None,
                  softmax_scale=scale,
                  causal=is_causal,
                  block_table=None,
                  window_size=window_size,
                  s_aux=sink)

    output = run_op(flash_attn_varlen_func)
    pip_output = None
    pip_error = None
    try:
        pip_output = run_op(flash_attn_varlen_func_pip)
    except Exception as e:
        pip_error = e

    ref_output = ref_paged_attn(query=query,
                                key_cache=maybe_quantized_key_cache,
                                value_cache=maybe_quantized_value_cache,
                                query_lens=query_lens,
                                kv_lens=kv_lens,
                                block_tables=block_tables,
                                scale=scale,
                                casual=is_causal,
                                is_paged=is_paged,
                                sink=sink,
                                q_descale=q_descale,
                                k_descale=k_descale,
                                v_descale=v_descale,
                                window_size_left=window_size[0],
                                window_size_right=window_size[1],
                                is_fp8kv=is_fp8kv,
                                is_fp8_query=is_fp8_query,
                                dtype=output_dtype)
    atol, rtol = 1e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    if window_size[0] != -1 or window_size[1] != -1:
        atol, rtol = 1.5e-2, 1.5e-2
    if kv_dtype is not None:
        atol, rtol = 1.5e-2, 1.5e-2
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


def benchmark_varlen_with_paged_kv(num_seqs,
                                   query_lens,
                                   kv_lens,
                                   num_heads,
                                   head_size,
                                   block_size,
                                   window_size,
                                   output_dtype,
                                   soft_cap,
                                   num_blocks,
                                   fa_versions,
                                   q_dtype,
                                   is_sink,
                                   is_causal,
                                   is_paged,
                                   kv_dtype,
                                   provider,
                                   iterations=50,
                                   verbose=True):
    config = (num_seqs, query_lens, kv_lens, num_heads, head_size, block_size,
              window_size, output_dtype, soft_cap, num_blocks, fa_versions,
              q_dtype, is_sink, is_causal, is_paged, kv_dtype)
    cache_key = _benchmark_cache_key(config, provider, iterations)
    if cache_key in _BENCHMARK_RESULT_CACHE:
        return _BENCHMARK_RESULT_CACHE[cache_key]

    def run_provider(provider_name):
        provider_cache_key = _benchmark_cache_key(config, provider_name,
                                                  iterations)
        if provider_cache_key in _BENCHMARK_RESULT_CACHE:
            return _BENCHMARK_RESULT_CACHE[provider_cache_key]
        return benchmark_varlen_with_paged_kv(
            num_seqs=num_seqs,
            query_lens=query_lens,
            kv_lens=kv_lens,
            num_heads=num_heads,
            head_size=head_size,
            block_size=block_size,
            window_size=window_size,
            output_dtype=output_dtype,
            soft_cap=soft_cap,
            num_blocks=num_blocks,
            fa_versions=fa_versions,
            q_dtype=q_dtype,
            is_sink=is_sink,
            is_causal=is_causal,
            is_paged=is_paged,
            kv_dtype=kv_dtype,
            provider=provider_name,
            iterations=iterations,
            verbose=verbose)

    if provider == "xattention_Time_Speedup":
        vllm_time = run_provider("vllm_xpu_kernels")
        try:
            xattention_time = run_provider("xattention")
        except Exception:
            return _cache_result(cache_key, float("nan"))
        return _cache_result(cache_key, _safe_ratio(vllm_time,
                                                   xattention_time))
    if provider == "xattention_TFLOPS_Speedup":
        vllm_tflops = run_provider("vllm_xpu_kernels_TFLOPS")
        try:
            xattention_tflops = run_provider("xattention_TFLOPS")
        except Exception:
            return _cache_result(cache_key, float("nan"))
        return _cache_result(cache_key, _safe_ratio(xattention_tflops,
                                                   vllm_tflops))
    if provider == "xattention_MFU_Speedup":
        vllm_mfu = run_provider("vllm_xpu_kernels_MFU")
        try:
            xattention_mfu = run_provider("xattention_MFU")
        except Exception:
            return _cache_result(cache_key, float("nan"))
        return _cache_result(cache_key, _safe_ratio(xattention_mfu, vllm_mfu))

    input_cache_key = (tuple(config), iterations)
    if input_cache_key not in _BENCHMARK_INPUT_CACHE:
        if _BENCHMARK_INPUT_CACHE:
            _BENCHMARK_INPUT_CACHE.clear()
            clear_xpu_cache()
        maybe_quantized_query, maybe_quantized_key_cache, \
            maybe_quantized_value_cache, \
            max_query_len, cu_query_lens, max_kv_len, cu_kv_lens, \
            seq_k, q_descale, k_descale, v_descale, scale, is_causal, \
            _, window_size, sink, scale_shape, _, \
            query_lens, kv_lens, _, _, max_num_blocks_per_seq = \
                make_varlen_with_paged_kv_input(config)

        queries = [
            torch.rand_like(maybe_quantized_query) for _ in range(iterations)
        ]
        block_tables_list = None
        if is_paged:
            block_tables_list = [
                torch.randint(0,
                              num_blocks,
                              (num_seqs, max_num_blocks_per_seq),
                              dtype=torch.int32)
                for _ in range(iterations)
            ]
        _BENCHMARK_INPUT_CACHE[input_cache_key] = (
            maybe_quantized_key_cache, maybe_quantized_value_cache,
            max_query_len, cu_query_lens, max_kv_len, cu_kv_lens, seq_k,
            q_descale, k_descale, v_descale, scale, is_causal, window_size,
            sink, scale_shape, query_lens, kv_lens, queries,
            block_tables_list)

    maybe_quantized_key_cache, maybe_quantized_value_cache, \
        max_query_len, cu_query_lens, max_kv_len, cu_kv_lens, seq_k, \
        q_descale, k_descale, v_descale, scale, is_causal, window_size, \
        sink, scale_shape, query_lens, kv_lens, queries, \
        block_tables_list = _BENCHMARK_INPUT_CACHE[input_cache_key]
    num_query_heads = num_heads[0]

    if verbose:
        provider_name = provider.replace("vllm_xpu_kernels",
                                         "vllm-xpu-kernels")
        print(f"Running config: {num_seqs, query_lens, kv_lens, \
                                  num_heads, head_size, block_size, \
                                  window_size, output_dtype, soft_cap, \
                                  num_blocks, fa_versions, q_dtype, is_sink, \
                                  is_causal, is_paged, kv_dtype}, \
                                  Provider: {provider_name}",
              flush=True)
    assert iterations > 5, \
    "Number of iterations should be greater than 5 to account for warmup"

    use_xattention = provider.startswith("xattention")
    provider_prefix = "xattention" if use_xattention else "vllm_xpu_kernels"

    flash_op = flash_attn_varlen_func_pip \
        if use_xattention else flash_attn_varlen_func

    def run_flash(index):
        if is_paged:
            assert block_tables_list is not None
            return flash_op(queries[index],
                            maybe_quantized_key_cache,
                            maybe_quantized_value_cache,
                            max_query_len,
                            cu_query_lens,
                            max_kv_len,
                            seqused_k=seq_k,
                            q_descale=q_descale.expand(scale_shape)
                            if q_descale is not None else None,
                            k_descale=k_descale.expand(scale_shape)
                            if k_descale is not None else None,
                            v_descale=v_descale.expand(scale_shape)
                            if v_descale is not None else None,
                            softmax_scale=scale,
                            causal=is_causal,
                            block_table=block_tables_list[index],
                            window_size=window_size,
                            s_aux=sink)
        return flash_op(queries[index],
                        maybe_quantized_key_cache,
                        maybe_quantized_value_cache,
                        max_query_len,
                        cu_query_lens,
                        max_kv_len,
                        cu_seqlens_k=cu_kv_lens,
                        q_descale=q_descale.expand(scale_shape)
                        if q_descale is not None else None,
                        k_descale=k_descale.expand(scale_shape)
                        if k_descale is not None else None,
                        v_descale=v_descale.expand(scale_shape)
                        if v_descale is not None else None,
                        softmax_scale=scale,
                        causal=is_causal,
                        block_table=None,
                        window_size=window_size,
                        s_aux=sink)

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    for index in range(5):
        run_flash(index)
    start_event.record()
    for index in range(5, iterations):
        run_flash(index)
    end_event.record()
    torch.xpu.synchronize()

    ms = start_event.elapsed_time(end_event) / (iterations - 5)
    time_us = 1000 * ms
    flops = calculate_flops(num_query_heads, query_lens, kv_lens, head_size,
                            is_causal)
    tflops = flops / (ms / 1000) / 1e12
    hardware_presets = get_hardware_preset(torch.xpu.get_device_name())
    mfu = float("nan")
    if hardware_presets is not None:
        peak_tflops = hardware_presets["tflops"]
        mfu = (tflops / peak_tflops) * 100

    _cache_result(_benchmark_cache_key(config, provider_prefix, iterations),
                  time_us)
    _cache_result(_benchmark_cache_key(config, f"{provider_prefix}_TFLOPS",
                                       iterations), tflops)
    _cache_result(_benchmark_cache_key(config, f"{provider_prefix}_MFU",
                                       iterations), mfu)
    clear_xpu_cache()
    return _BENCHMARK_RESULT_CACHE[cache_key]


def get_benchmark_varlen_with_paged_kv(iterations=50):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "num_seqs", "query_lens", "kv_lens", "num_heads", "head_size",
                "block_size", "window_size", "output_dtype", "soft_cap",
                "num_blocks", "fa_versions", "q_dtype", "is_sink", "is_causal",
                "is_paged", "kv_dtype"
            ],
            x_vals=[tuple(c) for c in configs],
            line_arg="provider",
            line_vals=["vllm_xpu_kernels", "vllm_xpu_kernels_TFLOPS",
                       "vllm_xpu_kernels_MFU", "xattention",
                       "xattention_TFLOPS", "xattention_MFU",
                       "xattention_Time_Speedup",
                       "xattention_TFLOPS_Speedup",
                       "xattention_MFU_Speedup"],
            line_names=[
                "vllm-xpu-kernels(us)", "vllm-xpu-kernels_TFLOPS",
                "vllm-xpu-kernels_MFU (%)", "xattention(us)",
                "xattention_TFLOPS", "xattention_MFU (%)",
                "Time Speedup", "TFLOPS Speedup", "MFU Speedup"
            ],
            styles=[("blue", "-"), ("purple", "-"), ("red", "-"),
                    ("cyan", "-"), ("orange", "-"), ("brown", "-"),
                    ("black", "--"), ("green", "--"), ("pink", "--")],
            ylabel="Latency (us)",
            plot_name="flash-attn-varlen",
            args={},
        ))
    def benchmark(num_seqs, query_lens, kv_lens, num_heads, head_size,
                  block_size, window_size, output_dtype, soft_cap, num_blocks,
                  fa_versions, q_dtype, is_sink, is_causal, is_paged, kv_dtype,
                  provider):
        try:
            return benchmark_varlen_with_paged_kv(
                num_seqs=num_seqs,
                query_lens=query_lens,
                kv_lens=kv_lens,
                num_heads=num_heads,
                head_size=head_size,
                block_size=block_size,
                window_size=window_size,
                output_dtype=output_dtype,
                soft_cap=soft_cap,
                num_blocks=num_blocks,
                fa_versions=fa_versions,
                q_dtype=q_dtype,
                is_sink=is_sink,
                is_causal=is_causal,
                is_paged=is_paged,
                kv_dtype=kv_dtype,
                provider=provider,
                iterations=iterations,
            )
        except Exception as e:
            if (not provider.startswith("xattention")
                    or provider.endswith("_Speedup")):
                raise
            print(f"Skipping {provider} for unsupported xattention "
                  f"shape: {e}")
            clear_xpu_cache()
            config = (num_seqs, query_lens, kv_lens, num_heads, head_size,
                      block_size, window_size, output_dtype, soft_cap,
                      num_blocks, fa_versions, q_dtype, is_sink, is_causal,
                      is_paged, kv_dtype)
            cache_key = _benchmark_cache_key(config, provider, iterations)
            return _cache_result(cache_key, float("nan"))

    return benchmark


def filter_configs(configs):
    new_configs = []
    for config in configs:
        if (config[5] == 128 and config[9] == 32768 and config[4] >= 192) or \
            (config[6][0] != -1 or config[6][1] != -1):
            print("Skipping config due to potential OOM: ", config)
            continue
        new_configs.append(config)
    return new_configs

if __name__ == "__main__":

    args = parse_args()
    seed = 4242
    seed_everything(seed)
    iterations = 50
    torch.set_default_device("xpu")
    torch.xpu.set_device("xpu:0")

    configs = gen_correctness_config()
    configs = filter_configs(configs)

    for config in configs:
        try:
            calculate_diff_varlen_paged_kv(config)
        except Exception as e:
            print("Error in config: ", config, " error: ", e)
        clear_xpu_cache()

    configs = gen_perf_configs()
    configs = filter_configs(configs)
    benchmark = get_benchmark_varlen_with_paged_kv(iterations=iterations)
    save_path = ensure_save_path_exists(args.save_path)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=save_path)
    append_speedup_average_row(save_path, "flash-attn-varlen")
