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
    flash_attn_varlen_func_CalKernelTime as flash_attn_varlen_func,
    flash_attn_varlen_func_pip)
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


def clear_xpu_cache():
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


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
                  s_aux=sink)

    output = run_op(flash_attn_varlen_func)
    pip_output = None
    pip_error = None
    try:
        pip_output = run_op(flash_attn_varlen_func_pip)
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
            maybe_quantized_key_cache, maybe_quantized_value_cache,
            max_query_len, cu_query_lens, max_kv_len, seq_k, scale, sink,
            queries, block_tables_list)

    maybe_quantized_key_cache, maybe_quantized_value_cache, \
        max_query_len, cu_query_lens, max_kv_len, seq_k, scale, sink, \
        queries, block_tables_list = _BENCHMARK_INPUT_CACHE[input_cache_key]

    provider_name = provider.replace("vllm_xpu_kernels", "vllm-xpu-kernels")
    print(f"Running config: {seq_lens, num_heads, head_size, \
                              block_size, output_dtype, soft_cap, num_blocks, \
                              fa_versions, q_dtype, \
                              is_sink}, Provider: {provider_name}",
          flush=True)
    assert iterations > 5, \
    "Number of iterations should be greater than 5 to account for warmup"

    use_xattention = provider.startswith("xattention")
    provider_prefix = "xattention" if use_xattention else "vllm_xpu_kernels"

    flash_op = flash_attn_varlen_func_pip \
        if use_xattention else flash_attn_varlen_func

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    for index in range(5):
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
                s_aux=sink)
    start_event.record()
    for index in range(5, iterations):
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
                s_aux=sink)
    end_event.record()
    torch.xpu.synchronize()

    ms = start_event.elapsed_time(end_event) / (iterations - 5)
    time_us = 1000 * ms
    memory_load_GB = calculate_memory_usage(
        cu_query_lens[-1].item(),
        seq_k.sum().item(),
        num_heads,
        head_size,
        queries[5].dtype,
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
    flash_op = flash_attn_varlen_func_pip \
        if use_xattention else flash_attn_varlen_func

    def _run(i):
        flash_op(
            queries[i], maybe_quantized_key_cache,
            maybe_quantized_value_cache,
            max_query_len, cu_query_lens, max_kv_len,
            seqused_k=seq_k, softmax_scale=scale,
            causal=False, block_table=bt_list[i],
            window_size=(-1, -1), s_aux=sink)

    # Warmup
    for i in range(min(10, iterations)):
        _run(i)
    torch.xpu.synchronize()

    # Timed
    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    measured = iterations - 10
    start_event.record()
    for i in range(10, iterations):
        _run(i)
    end_event.record()
    torch.xpu.synchronize()

    avg_us = start_event.elapsed_time(end_event) * 1000.0 / measured

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
