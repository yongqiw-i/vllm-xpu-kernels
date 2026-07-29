# SPDX-License-Identifier: Apache-2.0

import benchmark.src.get_model_config as model_configs


GEMMA4_MODEL = "google/gemma-4-26B-A4B-it"
GEMMA4_CONFIG = {
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 256,
    "num_global_key_value_heads": 2,
    "global_head_dim": 512,
    "sliding_window": 1024,
}


def test_get_attention_configs_covers_global_and_sliding_shapes():
    assert model_configs.get_attention_configs(GEMMA4_CONFIG) == [
        (16, 2, 512, (-1, -1), True),
        (16, 8, 256, (1023, 0), False),
    ]


def test_attention_config_without_global_shape_remains_full_only():
    model_config = {
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "sliding_window": 32768,
    }
    assert model_configs.get_attention_configs(model_config) == [
        (40, 8, 128, (-1, -1), None),
    ]


def test_gemma4_generates_both_attention_perf_cases(monkeypatch):
    monkeypatch.setattr(model_configs, "attention_model_lists",
                        [GEMMA4_MODEL])
    monkeypatch.setattr(model_configs, "get_model_config",
                        lambda model, tp_size: GEMMA4_CONFIG)

    varlen_configs = (
        model_configs.gen_cutlass_flash_attn_varlen_perf_configs())
    decode_configs = model_configs.gen_cutlass_flash_attn_decode_perf_configs()

    assert varlen_configs
    assert decode_configs
    assert any(config[3:7] == ((16, 2), 512, 64, (-1, -1))
               and config[13] is True
               for config in varlen_configs)
    assert any(config[3:7] == ((16, 8), 256, 64, (1023, 0))
               and config[13] is False
               for config in varlen_configs)
    assert any(config[1:4] == ((16, 2), 512, 64)
               and config[10] == (-1, -1)
               for config in decode_configs)
    assert any(config[1:4] == ((16, 8), 256, 64)
               and config[10] == (1023, 0)
               for config in decode_configs)
