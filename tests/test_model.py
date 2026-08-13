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


def test_disabled_mtp_does_not_allocate_unused_projection_heads() -> None:
    config = tiny_config()
    assert len(build_model(config).mtp_heads) == 0
    config["mtp"]["enabled"] = True
    assert len(build_model(config).mtp_heads) == config["mtp"]["num_future_tokens"]


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
