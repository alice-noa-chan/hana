from __future__ import annotations

import copy
import json

import torch

from llm_pipeline.config import DEFAULT_CONFIG
from llm_pipeline.experiments import (
    ActivationExperiment,
    RuntimePatchController,
    build_masked_diffusion_batch,
    hybrid_generate,
)
from llm_pipeline.model import build_model


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


def test_masked_diffusion_batch_only_labels_corrupted_positions() -> None:
    torch.manual_seed(13)
    input_ids = torch.arange(1, 17).view(2, 8)
    labels = input_ids.clone()
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    batch = build_masked_diffusion_batch(input_ids, labels, attention, 63, 4, 0.5)
    supervised = batch["labels"].ne(-100)
    assert supervised.any(dim=1).all()
    assert torch.equal(batch["input_ids"].eq(63), supervised)
    assert torch.equal(batch["labels"][supervised], input_ids[supervised])
    assert batch["prefix_lengths"].remainder(4).eq(0).all()


def test_activation_monitor_records_and_intervenes(tmp_path) -> None:
    config = tiny_config()
    config["experiments"].update(enabled=True)
    config["experiments"]["activation_monitor"].update(
        enabled=True,
        modules=["layers.0"],
        every_n_calls=1,
        max_records=10,
        output_file="activations.jsonl",
    )
    config["experiments"]["interventions"] = [{"module": "layers.0", "kind": "zero", "start_step": 0, "end_step": 0}]
    model = build_model(config).eval()
    monitor = ActivationExperiment(model, config, tmp_path)
    monitor.set_step(0)
    model(torch.randint(4, 64, (1, 6)))
    monitor.close()
    row = json.loads((tmp_path / "activations.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["interventions"] == ["zero"]
    assert row["post"]["rms"] == 0
    assert row["delta_rms"] > 0


def test_activation_experiment_reports_only_current_stochastic_interventions(tmp_path) -> None:
    config = tiny_config()
    config["experiments"].update(enabled=True)
    config["experiments"]["activation_monitor"]["enabled"] = False
    config["experiments"]["interventions"] = [
        {"module": "layers.0", "kind": "noise", "value": 0.1, "start_step": 2, "end_step": 3}
    ]
    experiment = ActivationExperiment(build_model(config), config, tmp_path)

    assert experiment.has_active_stochastic_interventions() is False
    experiment.set_step(2)
    assert experiment.has_active_stochastic_interventions() is True
    experiment.set_step(4)
    assert experiment.has_active_stochastic_interventions() is False
    experiment.close()


def test_runtime_patch_is_shape_safe_and_does_not_rewrite_source_config() -> None:
    config = tiny_config()
    config["experiments"]["runtime_patches"] = [
        {
            "at_step": 2,
            "changes": {
                "train.learning_rate": 0.0001,
                "model.embedding_dropout": 0.25,
                "hybrid_diffusion.loss_weight": 0.7,
            },
        }
    ]
    model = build_model(config)

    class Optimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 0.0003}]

    class Scheduler:
        def __init__(self) -> None:
            self.base_lr = 0.0003
            self.min_lr = 0.00003

    controller = RuntimePatchController(config)
    optimizer = Optimizer()
    scheduler = Scheduler()
    assert controller.apply_due(1, optimizer, scheduler, model) == []
    changes = controller.apply_due(2, optimizer, scheduler, model)
    assert len(changes) == 3
    assert optimizer.param_groups[0]["lr"] == 0.0001
    assert model.embed_dropout.p == 0.25
    assert controller.value("hybrid_diffusion.loss_weight", 0.25) == 0.7
    assert config["hybrid_diffusion"]["loss_weight"] == 0.25


def test_hybrid_generation_resolves_every_mask_and_emits_trace() -> None:
    torch.manual_seed(17)
    model = build_model(tiny_config()).eval()
    prompt = torch.tensor([[1, 10, 11]])
    trace: list[dict] = []
    output = hybrid_generate(
        model,
        prompt,
        max_new_tokens=7,
        eos_id=2,
        mask_id=63,
        block_size=3,
        denoise_steps=2,
        ar_warmup_tokens=1,
        temperature=0,
        top_p=1,
        top_k=0,
        repetition_penalty=1,
        suppress_ids={63},
        trace=trace,
    )
    assert prompt.size(1) <= output.size(1) <= prompt.size(1) + 7
    assert not output.eq(63).any()
    assert trace
    assert {row["phase"] for row in trace}.issubset({"ar", "diffusion"})


def test_hybrid_generation_uses_local_rng_without_mutating_global_state() -> None:
    torch.manual_seed(103)
    model = build_model(tiny_config()).eval()
    prompt = torch.tensor([[1, 10, 11]])
    global_before = torch.random.get_rng_state().clone()

    def generate(seed: int) -> torch.Tensor:
        return hybrid_generate(
            model,
            prompt,
            max_new_tokens=7,
            eos_id=-1,
            mask_id=63,
            block_size=3,
            denoise_steps=2,
            ar_warmup_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            suppress_ids={63},
            generator=torch.Generator().manual_seed(seed),
        )

    first = generate(19)
    global_after = torch.random.get_rng_state()
    second = generate(19)

    assert torch.equal(global_before, global_after)
    assert torch.equal(first, second)
