# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""Dense MLA decode correctness and performance comparison.

This compares the vllm-xpu-kernels dense MLA decode path against xattention's
``flash_attn_with_kvcache`` dense MLA decode path. PyTorch is used as the
correctness reference; vllm-xpu-kernels is the baseline for speedup reporting.
"""

import argparse
import csv
import gc
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import triton

from utils import bootstrap_benchmark_env, ensure_save_path_exists

bootstrap_benchmark_env(__file__)

from benchmark.presets import get_hardware_preset  # noqa: E402
from tests.flash_attn.test_mla_decode import (  # noqa: E402
    _make_inputs,
    _mla_decode_via_varlen,
    _mla_decode_via_xattention,
    _ref_mla_decode,
)

DEVICE = "xpu"
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_QK = KV_LORA_RANK + QK_ROPE_HEAD_DIM
DTYPE = torch.bfloat16
_BENCHMARK_RESULT_CACHE = {}
_BENCHMARK_INPUT_CACHE = {}


@dataclass(frozen=True)
class MLADecodeConfig:
    name: str
    query_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    num_heads_q: int
    block_size: int
    num_blocks: int = 0

    @property
    def batch(self) -> int:
        return len(self.query_lens)

    @property
    def max_seqlen_q(self) -> int:
        return max(self.query_lens)

    @property
    def max_seqlen_k(self) -> int:
        return max(self.kv_lens)

    @property
    def effective_num_blocks(self) -> int:
        if self.num_blocks:
            return self.num_blocks
        needed = (self.max_seqlen_k + self.block_size - 1) // self.block_size
        return max(256, needed * self.batch * 2)

    @property
    def query_lens_str(self) -> str:
        return "+".join(map(str, self.query_lens))

    @property
    def kv_lens_str(self) -> str:
        return "+".join(map(str, self.kv_lens))


PERF_CONFIGS = [
    # Single-request decode: short, medium, long context.
    MLADecodeConfig("1x64_h1_bs64", (1,), (64,), 1, 64, 256),
    MLADecodeConfig("1x140_h8_bs64", (1,), (140,), 8, 64, 256),
    MLADecodeConfig("1x1024_h8_bs64", (1,), (1024,), 8, 64, 512),
    MLADecodeConfig("1x4096_h8_bs64", (1,), (4096,), 8, 64, 2048),
    MLADecodeConfig("1x8192_h8_bs64", (1,), (8192,), 8, 64, 2048),
    MLADecodeConfig("1x16384_h8_bs64", (1,), (16384,), 8, 64, 4096),

    # Head-count sweep. qgroup>8 is intentionally omitted because the
    # head_size_qk=576 MLA path is limited to q_packed<=8 on Intel Xe.
    MLADecodeConfig("1x4096_h1_bs64", (1,), (4096,), 1, 64, 2048),
    MLADecodeConfig("1x4096_h3_bs64", (1,), (4096,), 3, 64, 2048),
    MLADecodeConfig("1x4096_h8_bs128", (1,), (4096,), 8, 128, 2048),

    # Batched realistic decode shapes.
    MLADecodeConfig("4xmixed_h8_bs64", (1, 1, 1, 1),
                    (128, 512, 2048, 8192), 8, 64, 2048),
    MLADecodeConfig("8xmixed_h8_bs64", (1, 1, 1, 1, 1, 1, 1, 1),
                    (128, 256, 512, 1024, 2048, 4096, 8192, 16384), 8, 64,
                    4096),
    MLADecodeConfig("8xmixed_h8_bs128", (1, 1, 1, 1, 1, 1, 1, 1),
                    (128, 256, 512, 1024, 2048, 4096, 8192, 16384), 8, 128,
                    4096),
    MLADecodeConfig("8xshort_h8_bs64", (1, 1, 1, 1, 1, 1, 1, 1),
                    (128, 128, 256, 256, 512, 512, 1024, 1024), 8, 64,
                    2048),
    MLADecodeConfig("8xshort_h8_bs128", (1, 1, 1, 1, 1, 1, 1, 1),
                    (128, 128, 256, 256, 512, 512, 1024, 1024), 8, 128,
                    2048),
    MLADecodeConfig("16xrealistic_h8_bs64", (1,) * 16,
                    (256, 512, 1024, 1024, 2048, 2048, 4096, 4096, 4096,
                     8192, 8192, 8192, 16384, 16384, 16384, 16384), 8, 64,
                    8192),
    MLADecodeConfig("32x512_h8_bs64", (1,) * 32, (512,) * 32, 8, 64, 2048),
    MLADecodeConfig("64x512_h8_bs64", (1,) * 64, (512,) * 64, 8, 64, 4096),
]

# Keep correctness coverage in lock-step with benchmark problem shapes.
CORRECTNESS_CONFIGS = PERF_CONFIGS


def clear_xpu_cache():
    torch.xpu.empty_cache()
    torch.xpu.synchronize()
    gc.collect()


def calculate_memory_usage(q_len_sum, kv_len_sum, num_heads, head_qk, head_v,
                           query_dtype, kv_dtype, output_dtype):
    # Match the MHA benchmark's logical traffic model: query, K/V, and output.
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    query_memory = q_len_sum * num_query_heads * head_qk * \
        torch.tensor([], dtype=query_dtype).element_size()
    kv_cache_memory = kv_len_sum * num_kv_heads * (head_qk + head_v) * \
        torch.tensor([], dtype=kv_dtype).element_size()
    output_memory = q_len_sum * num_query_heads * head_v * \
        torch.tensor([], dtype=output_dtype).element_size()
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


def _config_tuple(config: MLADecodeConfig):
    return (config.name, config.block_size, list(config.query_lens),
            list(config.kv_lens), config.num_heads_q)


def _benchmark_tuple(config: MLADecodeConfig):
    return (
        config.name,
        config.batch,
        config.query_lens_str,
        config.kv_lens_str,
        sum(config.kv_lens),
        config.num_heads_q,
        1,
        HEAD_QK,
        KV_LORA_RANK,
        config.block_size,
        config.effective_num_blocks,
        str(DTYPE).replace("torch.", ""),
    )


def _parse_lens(lens: str):
    return tuple(map(int, str(lens).split("+")))


def _config_from_benchmark_args(config, batch, query_lens, kv_lens, kv_sum,
                                num_heads_q, num_heads_kv, head_qk, head_v,
                                block_size, num_blocks, dtype):
    del batch, kv_sum, num_heads_kv, head_qk, head_v, dtype
    return MLADecodeConfig(
        str(config),
        _parse_lens(query_lens),
        _parse_lens(kv_lens),
        int(num_heads_q),
        int(block_size),
        int(num_blocks),
    )


def make_inputs(config: MLADecodeConfig, seed: int = 0):
    return _make_inputs(
        batch=config.batch,
        query_lens=list(config.query_lens),
        kv_lens=list(config.kv_lens),
        num_heads_q=config.num_heads_q,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_size=config.block_size,
        num_blocks=config.effective_num_blocks,
        dtype=DTYPE,
        seed=seed,
    )


def run_vllm(config: MLADecodeConfig, inputs):
    q_nope, q_pe, cache, cu_q, sk, bt = inputs
    return _mla_decode_via_varlen(
        q_nope,
        q_pe,
        cache,
        bt,
        cu_q,
        sk,
        max_seqlen_q=config.max_seqlen_q,
        max_seqlen_k=config.max_seqlen_k,
        softmax_scale=HEAD_QK**-0.5,
    )


def run_xattention(config: MLADecodeConfig, inputs):
    q_nope, q_pe, cache, cu_q, sk, bt = inputs
    return _mla_decode_via_xattention(
        q_nope,
        q_pe,
        cache,
        bt,
        cu_q,
        sk,
        max_seqlen_q=config.max_seqlen_q,
        max_seqlen_k=config.max_seqlen_k,
        softmax_scale=HEAD_QK**-0.5,
    )


def run_ref(config: MLADecodeConfig, inputs):
    q_nope, q_pe, cache, cu_q, sk, bt = inputs
    return _ref_mla_decode(
        q_nope,
        q_pe,
        cache,
        bt,
        cu_q,
        sk,
        softmax_scale=HEAD_QK**-0.5,
        causal=False,
    )


def check_correctness(config: MLADecodeConfig) -> bool:
    inputs = make_inputs(config)
    ref = run_ref(config, inputs)
    ok = True
    for name, fn in (("vllm_xpu_kernels", run_vllm),
                     ("xattention", run_xattention)):
        try:
            torch.testing.assert_close(fn(config, inputs),
                                       ref,
                                       atol=2e-2,
                                       rtol=2e-2)
        except AssertionError as e:
            ok = False
            print(f"❌ {name} differs from reference, {_config_tuple(config)} "
                  f"error: {e}")
        else:
            print(f"✅ {name} implementation matches, {_config_tuple(config)}")
    clear_xpu_cache()
    return ok


def benchmark_provider(config: MLADecodeConfig, provider: str,
                       iterations: int) -> float:
    benchmark_config = _benchmark_tuple(config)
    cache_key = _benchmark_cache_key(benchmark_config, provider, iterations)
    if cache_key in _BENCHMARK_RESULT_CACHE:
        return _BENCHMARK_RESULT_CACHE[cache_key]

    def run_provider(provider_name):
        provider_cache_key = _benchmark_cache_key(benchmark_config,
                                                  provider_name, iterations)
        if provider_cache_key in _BENCHMARK_RESULT_CACHE:
            return _BENCHMARK_RESULT_CACHE[provider_cache_key]
        return benchmark_provider(config, provider_name, iterations)

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

    input_cache_key = (benchmark_config, iterations)
    if input_cache_key not in _BENCHMARK_INPUT_CACHE:
        if _BENCHMARK_INPUT_CACHE:
            _BENCHMARK_INPUT_CACHE.clear()
            clear_xpu_cache()

        q_nope, q_pe, cache, cu_q, sk, bt = make_inputs(config)
        queries = [
            (torch.rand_like(q_nope), torch.rand_like(q_pe))
            for _ in range(iterations)
        ]
        block_tables = [
            torch.randint(0,
                          config.effective_num_blocks,
                          bt.shape,
                          dtype=torch.int32,
                          device=bt.device) for _ in range(iterations)
        ]
        _BENCHMARK_INPUT_CACHE[input_cache_key] = (
            cache,
            cu_q,
            sk,
            queries,
            block_tables,
        )

    cache, cu_q, sk, queries, block_tables = \
        _BENCHMARK_INPUT_CACHE[input_cache_key]

    provider_name = provider.replace("vllm_xpu_kernels", "vllm-xpu-kernels")
    print(f"Running config: {_config_tuple(config)}, "
          f"Provider: {provider_name}",
          flush=True)
    assert iterations > 5, \
        "Number of iterations should be greater than 5 to account for warmup"

    use_xattention = provider.startswith("xattention")
    provider_prefix = "xattention" if use_xattention else "vllm_xpu_kernels"
    fn = run_xattention if use_xattention else run_vllm

    def _run(index):
        q_nope, q_pe = queries[index]
        return fn(config, (q_nope, q_pe, cache, cu_q, sk,
                          block_tables[index]))

    for index in range(5):
        _run(index)
    torch.xpu.synchronize()

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)
    start_event.record()
    for index in range(5, iterations):
        _run(index)
    end_event.record()
    torch.xpu.synchronize()
    ms = start_event.elapsed_time(end_event) / (iterations - 5)
    avg_us = 1000 * ms
    memory_load_GB = calculate_memory_usage(
        cu_q[-1].item(),
        sk.sum().item(),
        (config.num_heads_q, 1),
        HEAD_QK,
        KV_LORA_RANK,
        queries[5][0].dtype,
        cache.dtype,
        DTYPE,
    )
    measured_bw = memory_load_GB / (ms / 1000)
    hardware_presets = get_hardware_preset(torch.xpu.get_device_name())
    mbu = float("nan")
    if hardware_presets is not None:
        peak_bw = hardware_presets["memory_bandwidth_GBs"]
        mbu = (measured_bw / peak_bw) * 100

    _cache_result(_benchmark_cache_key(benchmark_config, provider_prefix,
                                      iterations), avg_us)
    _cache_result(_benchmark_cache_key(
        benchmark_config, f"{provider_prefix}_memBandwidth", iterations),
        measured_bw)
    _cache_result(_benchmark_cache_key(benchmark_config, f"{provider_prefix}_MBU",
                                      iterations), mbu)
    clear_xpu_cache()
    return _BENCHMARK_RESULT_CACHE[cache_key]


def get_benchmark_mla_dense_decode(iterations=100):

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=[
                "config", "batch", "query_lens", "kv_lens", "kv_sum",
                "num_heads_q", "num_heads_kv", "head_qk", "head_v",
                "block_size", "num_blocks", "dtype"
            ],
            x_vals=[_benchmark_tuple(c) for c in PERF_CONFIGS],
            line_arg="provider",
            line_vals=[
                "vllm_xpu_kernels",
                "vllm_xpu_kernels_memBandwidth",
                "vllm_xpu_kernels_MBU",
                "xattention",
                "xattention_memBandwidth",
                "xattention_MBU",
                "xattention_Time_Speedup",
                "xattention_Bandwidth_Speedup",
                "xattention_MBU_Speedup",
            ],
            line_names=[
                "vllm-xpu-kernels(us)",
                "vllm-xpu-kernels_memBandwidth(GB/s)",
                "vllm-xpu-kernels_MBU (%)",
                "xattention(us)",
                "xattention_memBandwidth(GB/s)",
                "xattention_MBU (%)",
                "Time Speedup",
                "bandwidth Speedup",
                "MBU Speedup",
            ],
            styles=[("blue", "-"), ("purple", "-"), ("red", "-"),
                    ("cyan", "-"), ("orange", "-"), ("brown", "-"),
                    ("black", "--"), ("green", "--"), ("pink", "--")],
            ylabel="Latency (us)",
            plot_name="mla-dense-decode",
            args={},
        ))
    def benchmark(config, batch, query_lens, kv_lens, kv_sum, num_heads_q,
                  num_heads_kv, head_qk, head_v, block_size, num_blocks, dtype,
                  provider):
        mla_config = _config_from_benchmark_args(
            config, batch, query_lens, kv_lens, kv_sum, num_heads_q,
            num_heads_kv, head_qk, head_v, block_size, num_blocks, dtype)
        try:
            return benchmark_provider(mla_config, provider, iterations)
        except Exception as e:
            if (not provider.startswith("xattention")
                    or provider.endswith("_Speedup")):
                raise
            print(f"Skipping {provider} for unsupported xattention "
                  f"shape: {e}")
            clear_xpu_cache()
            cache_key = _benchmark_cache_key(_benchmark_tuple(mla_config),
                                             provider, iterations)
            return _cache_result(cache_key, float("nan"))

    return benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark dense MLA decode: vllm-xpu-kernels vs xattention")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--save-path", type=str, default="")
    parser.add_argument("--skip-correctness", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.xpu.is_available():
        raise SystemExit("XPU not available")

    torch.set_default_device(DEVICE)
    torch.xpu.set_device("xpu:0")

    if not args.skip_correctness:
        failures = 0
        for config in CORRECTNESS_CONFIGS:
            failures += 0 if check_correctness(config) else 1
        if failures:
            raise SystemExit(f"{failures} correctness config(s) failed")

    print("\n" + "=" * 80)
    print("Dense MLA Decode Benchmark")
    print("=" * 80)
    benchmark = get_benchmark_mla_dense_decode(iterations=args.iterations)
    save_path = ensure_save_path_exists(args.save_path) if args.save_path else ""
    if save_path:
        benchmark.run(print_data=True, save_path=save_path)
        append_speedup_average_row(save_path, "mla-dense-decode")
    else:
        benchmark.run(print_data=True)


if __name__ == "__main__":
    main()
