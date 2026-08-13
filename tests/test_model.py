from __future__ import annotations

import copy
import math

import pytest
import torch

from llm_pipeline.config import DEFAULT_CONFIG
from llm_pipeline.model import build_attention_bias, build_model
from llm_pipeline.training import evaluate_loss


def tiny_config() -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["model"].update(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        max_seq_len=16,
        attention_backend="eager",
        gradient_checkpointing=False,
    )
    config["hardware"].update(device="cpu", num_workers=0)
    config["train"]["mixed_precision"] = "fp32"
    return config


def cognitive_tiny_config() -> dict:
    config = tiny_config()
    cognitive = config["cognitive_architecture"]
    cognitive["enabled"] = True
    cognitive["workspace"].update(enabled=True, bottleneck_size=8, every_n_layers=1)
    cognitive["predictive_coding"].update(enabled=True, loss_weight=0.05)
    cognitive["homeostasis"].update(enabled=True, target_rms=0.05, loss_weight=0.001)
    return config


def test_forward_loss_is_finite_and_zero_label_batches_fail() -> None:
    torch.manual_seed(3)
    model = build_model(tiny_config())
    input_ids = torch.randint(4, 64, (2, 10))
    output = model(input_ids, labels=input_ids)
    assert output["logits"].shape == (2, 10, 64)
    assert math.isfinite(output["loss"].item())
    assert output["num_loss_tokens"].item() == 18

    with pytest.raises(ValueError, match="zero supervised"):
        model(input_ids, labels=torch.full_like(input_ids, -100))


def test_generate_uses_local_rng_without_mutating_global_state() -> None:
    torch.manual_seed(101)
    model = build_model(tiny_config()).eval()
    prompt = torch.tensor([[1, 7, 8]])
    global_before = torch.random.get_rng_state().clone()

    first = model.generate(
        prompt,
        max_new_tokens=6,
        eos_id=-1,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        generator=torch.Generator().manual_seed(17),
    )
    global_after = torch.random.get_rng_state()
    second = model.generate(
        prompt,
        max_new_tokens=6,
        eos_id=-1,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        generator=torch.Generator().manual_seed(17),
    )

    assert torch.equal(global_before, global_after)
    assert torch.equal(first, second)


def test_disabled_mtp_does_not_allocate_unused_projection_heads() -> None:
    config = tiny_config()
    assert len(build_model(config).mtp_heads) == 0
    config["mtp"]["enabled"] = True
    assert len(build_model(config).mtp_heads) == config["mtp"]["num_future_tokens"]


def test_public_qk_norm_count_and_opt_in_gate_parameter_delta() -> None:
    production = copy.deepcopy(DEFAULT_CONFIG)
    production["model"].update(
        vocab_size=32000,
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        max_seq_len=2048,
        qk_norm=True,
        attention_output_gate=False,
    )
    with torch.device("meta"):
        baseline = build_model(production)
        baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    assert baseline_count == 303_353_856

    gated = tiny_config()
    gated["model"]["attention_output_gate"] = True
    baseline_tiny = tiny_config()
    baseline_tiny["model"]["attention_output_gate"] = False
    gated_count = sum(parameter.numel() for parameter in build_model(gated).parameters())
    baseline_tiny_count = sum(parameter.numel() for parameter in build_model(baseline_tiny).parameters())
    head_dim = gated["model"]["hidden_size"] // gated["model"]["num_attention_heads"]
    expected_delta = gated["model"]["num_layers"] * gated["model"]["num_attention_heads"] * (head_dim + 1)
    assert gated_count - baseline_tiny_count == expected_delta
    first_gate = build_model(gated).layers[0].attn.output_gate
    assert first_gate is not None
    assert torch.count_nonzero(first_gate.weight).item() == 0
    torch.testing.assert_close(first_gate.bias, torch.full_like(first_gate.bias, 2.0))


def test_disabled_gate_and_pattern_preserve_legacy_state_and_logits() -> None:
    explicit = tiny_config()
    explicit["model"].update(qk_norm=False, attention_output_gate=False)
    explicit["model"]["sliding_window"].update(enabled=False, layer_pattern=[])
    legacy = copy.deepcopy(explicit)
    legacy["model"].pop("attention_output_gate")
    legacy["model"]["sliding_window"].pop("layer_pattern")

    torch.manual_seed(8)
    explicit_model = build_model(explicit).eval()
    torch.manual_seed(8)
    legacy_model = build_model(legacy).eval()
    assert explicit_model.state_dict().keys() == legacy_model.state_dict().keys()
    assert not any("output_gate" in key for key in explicit_model.state_dict())

    input_ids = torch.randint(4, 64, (1, 8))
    torch.testing.assert_close(explicit_model(input_ids)["logits"], legacy_model(input_ids)["logits"])


def test_attention_output_gate_keeps_kv_cache_exact() -> None:
    torch.manual_seed(9)
    config = tiny_config()
    config["model"]["attention_output_gate"] = True
    model = build_model(config).eval()
    input_ids = torch.randint(4, 64, (1, 9))

    full = model(input_ids)["logits"]
    prefix = model(input_ids[:, :5], use_cache=True)
    suffix = model(input_ids[:, 5:], past_key_values=prefix["past_key_values"], use_cache=True)["logits"]

    torch.testing.assert_close(suffix, full[:, 5:], atol=1e-5, rtol=1e-5)


def hybrid_tiny_config() -> dict:
    config = tiny_config()
    config["model"]["num_layers"] = 5
    config["model"]["sliding_window"].update(
        enabled=True,
        window_size=4,
        layer_pattern=["full", "sliding", "sliding", "sliding"],
    )
    return config


def test_hybrid_attention_repeats_exact_full_sliding_schedule() -> None:
    model = build_model(hybrid_tiny_config())
    assert [layer.attn.window for layer in model.layers] == [None, 4, 4, 4, None]


def test_hybrid_attention_cache_matches_full_forward() -> None:
    torch.manual_seed(10)
    model = build_model(hybrid_tiny_config()).eval()
    input_ids = torch.randint(4, 64, (1, 10))
    full = model(input_ids)["logits"]
    prefix = model(input_ids[:, :6], use_cache=True)
    suffix = model(input_ids[:, 6:], past_key_values=prefix["past_key_values"], use_cache=True)["logits"]
    torch.testing.assert_close(suffix, full[:, 6:], atol=1e-5, rtol=1e-5)


def test_hybrid_attention_keeps_packed_documents_isolated() -> None:
    torch.manual_seed(11)
    model = build_model(hybrid_tiny_config()).eval()
    first = torch.tensor([[7, 8, 9, 20, 21, 22]])
    second = torch.tensor([[40, 41, 42, 20, 21, 22]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]])
    document_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
    first_logits = model(first, position_ids=position_ids, document_ids=document_ids)["logits"]
    second_logits = model(second, position_ids=position_ids, document_ids=document_ids)["logits"]
    torch.testing.assert_close(first_logits[:, 3:], second_logits[:, 3:], atol=1e-5, rtol=1e-5)


def test_kv_cache_matches_full_causal_forward() -> None:
    torch.manual_seed(4)
    model = build_model(tiny_config()).eval()
    input_ids = torch.randint(4, 64, (1, 8))
    full = model(input_ids)["logits"]
    prefix = model(input_ids[:, :5], use_cache=True)
    suffix = model(input_ids[:, 5:], past_key_values=prefix["past_key_values"], use_cache=True)["logits"]
    torch.testing.assert_close(suffix, full[:, 5:], atol=1e-5, rtol=1e-5)


def test_packed_documents_cannot_attend_across_boundary() -> None:
    torch.manual_seed(5)
    model = build_model(tiny_config()).eval()
    first = torch.tensor([[7, 8, 20, 21]])
    second = torch.tensor([[40, 41, 20, 21]])
    position_ids = torch.tensor([[0, 1, 0, 1]])
    document_ids = torch.tensor([[0, 0, 1, 1]])
    first_logits = model(first, position_ids=position_ids, document_ids=document_ids)["logits"]
    second_logits = model(second, position_ids=position_ids, document_ids=document_ids)["logits"]
    torch.testing.assert_close(first_logits[:, 2:], second_logits[:, 2:], atol=1e-5, rtol=1e-5)


def test_empty_evaluation_loader_is_an_error() -> None:
    config = tiny_config()
    model = build_model(config)
    loader = torch.utils.data.DataLoader([])
    with pytest.raises(RuntimeError, match="zero supervised tokens"):
        evaluate_loss(model, loader, torch.device("cpu"), config)


def test_prefix_block_attention_is_bidirectional_inside_block_but_not_across_future_blocks() -> None:
    bias = build_attention_bias(
        attention_mask=None,
        document_ids=None,
        q_len=6,
        kv_len=6,
        past_len=0,
        dtype=torch.float32,
        device=torch.device("cpu"),
        sliding_window=None,
        attention_mode="prefix_block",
        prefix_lengths=torch.tensor([2]),
        block_size=2,
    )
    visible = bias.eq(0)[0, 0]
    assert visible[2, 3]  # same noisy block, including the token to the right
    assert not visible[3, 4]  # a later block cannot leak backward
    assert visible[4, 3]  # a later block can use an earlier completed block
    assert not visible[1, 2]  # clean prefix remains ordinary token-causal


def test_same_token_loss_and_hidden_state_capture_are_explicit() -> None:
    model = build_model(tiny_config())
    input_ids = torch.randint(4, 64, (2, 7))
    labels = torch.full_like(input_ids, -100)
    labels[:, [2, 5]] = input_ids[:, [2, 5]]
    output = model(
        input_ids,
        labels=labels,
        attention_mode="bidirectional",
        loss_mode="same_token",
        return_hidden_states=True,
    )
    assert output["num_loss_tokens"].item() == 4
    assert math.isfinite(output["loss"].item())
    assert len(output["hidden_states"]) == tiny_config()["model"]["num_layers"] + 1


def test_causal_workspace_cache_matches_full_forward() -> None:
    torch.manual_seed(6)
    model = build_model(cognitive_tiny_config()).eval()
    input_ids = torch.randint(4, 64, (1, 9))
    full = model(input_ids)["logits"]
    prefix = model(input_ids[:, :5], use_cache=True)
    suffix_output = model(
        input_ids[:, 5:],
        past_key_values=prefix["past_key_values"],
        past_workspace_states=prefix["workspace_states"],
        use_cache=True,
    )
    suffix = suffix_output["logits"]
    torch.testing.assert_close(suffix, full[:, 5:], atol=1e-5, rtol=1e-5)
    assert all(state is not None and state[0].dtype == torch.float32 for state in suffix_output["workspace_states"])
    with pytest.raises(ValueError, match="requires past_workspace_states"):
        model(input_ids[:, 5:], past_key_values=prefix["past_key_values"], use_cache=True)


def test_cognitive_objectives_are_finite_and_observable() -> None:
    torch.manual_seed(7)
    config = cognitive_tiny_config()
    model = build_model(config)
    input_ids = torch.randint(4, 64, (2, 10))
    output = model(input_ids, labels=input_ids, return_hidden_states=True)
    assert math.isfinite(output["loss"].item())
    assert math.isfinite(output["predictive_loss"].item())
    assert math.isfinite(output["homeostatic_loss"].item())
    assert len(output["workspace_activations"]) == config["model"]["num_layers"]
    assert output["workspace_activations"][0].shape == (2, 10, 8)
