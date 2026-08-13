from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from llm_pipeline.config import DEFAULT_CONFIG
from llm_pipeline.inference import (
    GenerationResult,
    TextGenerator,
    _DecodedPhase,
    apply_reasoning_control,
    build_inference_prompt,
    reasoning_token_budget,
)


class FakeTokenizer:
    bos_id = 1
    eos_id = 2
    vocab_size = 64
    special_ids = frozenset({2, 8, 9, 10, 11, 12})

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if text == "\n<reasoning:off>\n":
            ids = [7, 9]
        else:
            ids = [7] if text == "\n" else [20 + index for index, _ in enumerate(text)]
        return [self.bos_id, *ids, self.eos_id] if add_special_tokens else ids

    def decode(self, ids: list[int]) -> str:
        pieces = {31: "private scratchpad", 41: "final answer"}
        return " ".join(pieces[token] for token in ids if token in pieces)

    def piece_to_id(self, piece: str) -> int:
        return {"<reasoning:off>": 9}[piece]


class FakeLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


class FakeActivationExperiment:
    def has_active_stochastic_interventions(self) -> bool:
        return False

    def flush(self) -> None:
        return None


class RecordingActivationExperiment:
    def __init__(self) -> None:
        self.records: list[dict[str, str]] = []
        self.max_records = 100
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1

    def has_active_stochastic_interventions(self) -> bool:
        return False


class FakeMemory:
    def __init__(self) -> None:
        self.retrieve_calls: list[list[float]] = []
        self.observe_calls: list[tuple[str, str, list[float], float]] = []
        self.state = {"episodes": [], "gists": []}

    def retrieve(self, embedding: list[float]) -> list[dict[str, Any]]:
        self.retrieve_calls.append(embedding)
        return []

    def render(self, records: list[dict[str, Any]]) -> str:
        assert records == []
        return ""

    def observe(
        self,
        prompt: str,
        answer: str,
        embedding: list[float],
        surprise: float,
    ) -> bool:
        self.observe_calls.append((prompt, answer, embedding, surprise))
        return True


class StubTextGenerator(TextGenerator):
    def __init__(self, config: dict[str, Any], log_dir: Path) -> None:
        self.config = config
        self.logger = FakeLogger(log_dir)
        self.device = torch.device("cpu")
        self.tokenizer = FakeTokenizer()
        self.memory = None
        self.activation_experiment = FakeActivationExperiment()
        self._warning_count = 0
        self._saved_reasoning_records = []
        self.reasoning_output_ids = (31, 2)
        self.calls: list[dict[str, Any]] = []

    def _decode(
        self,
        rendered_prompt: str,
        settings: dict[str, Any],
        *,
        phase: str,
        allow_token_trace: bool,
        reserve_new_tokens: int = 0,
        allow_reasoning_boundary: bool = False,
        generator: torch.Generator | None = None,
    ) -> _DecodedPhase:
        self.calls.append(
            {
                "kind": "prompt",
                "prompt": rendered_prompt,
                "phase": phase,
                "reserve": reserve_new_tokens,
                "allow_boundary": allow_reasoning_boundary,
                "trace": allow_token_trace,
                "generator_seed": generator.initial_seed() if generator is not None else None,
            }
        )
        if phase == "reasoning":
            return _DecodedPhase(
                text="private scratchpad",
                input_ids=torch.tensor([[1, 10]], dtype=torch.long),
                token_ids=self.reasoning_output_ids,
                trace=None,
            )
        return _DecodedPhase(
            text="final answer",
            input_ids=torch.tensor([[1, 12]], dtype=torch.long),
            token_ids=(41,),
            trace=[{"phase": "ar", "token_ids": [41]}] if allow_token_trace else None,
        )

    def _decode_from_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        settings: dict[str, Any],
        *,
        phase: str,
        allow_token_trace: bool,
        allow_reasoning_boundary: bool = False,
        generator: torch.Generator | None = None,
    ) -> _DecodedPhase:
        self.calls.append(
            {
                "kind": "ids",
                "ids": input_ids.tolist()[0],
                "phase": phase,
                "max_new_tokens": max_new_tokens,
                "trace": allow_token_trace,
                "generator_seed": generator.initial_seed() if generator is not None else None,
            }
        )
        return _DecodedPhase(
            text="final answer",
            input_ids=input_ids,
            token_ids=(41,),
            trace=[{"phase": "ar", "token_ids": [41]}] if allow_token_trace else None,
        )


class MaxTextGenerator(StubTextGenerator):
    def __init__(self, config: dict[str, Any], log_dir: Path, answers: list[str], selector: str = "A") -> None:
        super().__init__(config, log_dir)
        self.answers = iter(answers)
        self.selector = selector
        self.scratchpads: list[str] = []

    def _selector_answer(self, answer: str, max_tokens: int) -> str:
        return " ".join(answer.split()[:max_tokens])

    def _decode(
        self,
        rendered_prompt: str,
        settings: dict[str, Any],
        *,
        phase: str,
        allow_token_trace: bool,
        reserve_new_tokens: int = 0,
        allow_reasoning_boundary: bool = False,
        generator: torch.Generator | None = None,
    ) -> _DecodedPhase:
        self.calls.append(
            {
                "kind": "prompt",
                "prompt": rendered_prompt,
                "phase": phase,
                "reserve": reserve_new_tokens,
                "allow_boundary": allow_reasoning_boundary,
                "trace": allow_token_trace,
                "generator_seed": generator.initial_seed() if generator is not None else None,
            }
        )
        if phase == "reasoning_selector":
            return _DecodedPhase(
                text=self.selector,
                input_ids=torch.tensor([[1, 15]], dtype=torch.long),
                token_ids=(50,),
                trace=None,
            )
        scratchpad = f"private scratchpad {len(self.scratchpads)}"
        self.scratchpads.append(scratchpad)
        scratchpad_id = 31 + len(self.scratchpads) - 1
        self.tokenizer.decode = lambda ids: " ".join(
            {
                31: "private scratchpad 0",
                32: "private scratchpad 1",
                33: "private scratchpad 2",
                41: "first answer",
                42: "second answer",
                43: "third answer",
            }.get(token, "")
            for token in ids
        ).strip()
        return _DecodedPhase(
            text=scratchpad,
            input_ids=torch.tensor([[1, 10]], dtype=torch.long),
            token_ids=(scratchpad_id, 2),
            trace=None,
        )

    def _decode_from_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        settings: dict[str, Any],
        *,
        phase: str,
        allow_token_trace: bool,
        allow_reasoning_boundary: bool = False,
        generator: torch.Generator | None = None,
    ) -> _DecodedPhase:
        answer = next(self.answers)
        token_id = {"first answer": 41, "second answer": 42, "third answer": 43}.get(answer, 41)
        self.calls.append(
            {
                "kind": "ids",
                "ids": input_ids.tolist()[0],
                "phase": phase,
                "max_new_tokens": max_new_tokens,
                "trace": allow_token_trace,
                "answer": answer,
                "generator_seed": generator.initial_seed() if generator is not None else None,
            }
        )
        return _DecodedPhase(
            text=answer,
            input_ids=input_ids,
            token_ids=(token_id,),
            trace=[{"phase": "ar", "token_ids": [token_id]}] if allow_token_trace else None,
        )


def reasoning_config() -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["reasoning"]["max_reasoning_tokens"] = 8
    config["inference"].update(reasoning_mode="medium", max_new_tokens=3)
    return config


def max_reasoning_config() -> dict[str, Any]:
    config = reasoning_config()
    config["reasoning"].update(
        modes=["off", "low", "medium", "high", "max"],
        mode_budget_ratios={"off": 0.0, "low": 0.25, "medium": 0.5, "high": 0.75, "max": 1.0},
        test_time_compute={
            "enabled": True,
            "mode": "max",
            "candidates": 3,
            "candidate_temperature": 0.8,
            "candidate_top_p": 0.95,
            "candidate_top_k": 20,
            "selector_max_new_tokens": 1,
            "selector_candidate_max_tokens": 8,
        },
    )
    config["tokenizer"]["special_tokens"]["reasoning_max"] = "<reasoning:max>"
    config["inference"]["reasoning_mode"] = "max"
    return config


def test_generation_result_keeps_backward_compatible_defaults() -> None:
    result = GenerationResult(answer="answer", reasoning_mode="off")

    assert result.reasoning_tokens == 0
    assert result.reasoning_trace is None
    assert result.candidate_count == 1
    assert result.reasoning_compute_tokens == 0
    assert result.answer_compute_tokens == 0
    assert result.selector_compute_tokens == 0
    assert result.selector_used is False


def test_reasoning_control_is_adjacent_to_the_assistant_cue() -> None:
    config = reasoning_config()

    prompt = build_inference_prompt(config, "synthetic question")

    assert prompt.endswith("<reasoning:medium>\n<assistant>\n")
    assert not prompt.startswith("<reasoning:medium>")
    assert reasoning_token_budget(config, "low") == 2
    assert reasoning_token_budget(config, "medium") == 4
    assert reasoning_token_budget(config, "high") == 6
    assert apply_reasoning_control("<assistant>\n", config, "off") == "<assistant>\n"


def test_hidden_reasoning_uses_two_phases_and_returns_only_the_answer(tmp_path: Path) -> None:
    config = reasoning_config()
    trace_file = tmp_path / "final-token-trace.jsonl"
    config["inference"]["token_trace_file"] = str(trace_file)
    generator = StubTextGenerator(config, tmp_path)

    result = generator.generate_result(prompt="synthetic question")

    assert result == GenerationResult(
        answer="final answer",
        reasoning_mode="medium",
        reasoning_tokens=1,
        reasoning_trace=None,
        reasoning_compute_tokens=2,
        answer_compute_tokens=1,
    )
    assert generator.generate(prompt="synthetic question") == "final answer"
    first_reasoning, first_final = generator.calls[:2]
    assert first_reasoning["phase"] == "reasoning"
    assert first_reasoning["reserve"] == 5
    assert first_reasoning["allow_boundary"] is True
    assert first_final["ids"] == [1, 10, 31, 7, 9]
    assert first_final["phase"] == "inference"
    assert json.loads(trace_file.read_text(encoding="utf-8"))["phase"] == "ar"
    assert not (tmp_path / "reasoning_trace.jsonl").exists()


def test_reasoning_exposure_and_persistence_are_explicit(tmp_path: Path) -> None:
    config = reasoning_config()
    config["reasoning"].update(expose_reasoning_trace=True, save_reasoning_trace=True)
    generator = StubTextGenerator(config, tmp_path)

    result = generator.generate_result(prompt="synthetic question")

    assert result.answer == "final answer"
    assert result.reasoning_trace == "private scratchpad"
    saved = json.loads((tmp_path / "reasoning_trace.jsonl").read_text(encoding="utf-8"))
    assert saved == {
        "reasoning_mode": "medium",
        "reasoning_tokens": 1,
        "reasoning_trace": "private scratchpad",
    }


def test_naturally_generated_reasoning_boundary_is_not_duplicated(tmp_path: Path) -> None:
    config = reasoning_config()
    generator = StubTextGenerator(config, tmp_path)
    generator.reasoning_output_ids = (31, 7, 9)

    result = generator.generate_result(prompt="synthetic question")

    assert result.answer == "final answer"
    assert result.reasoning_compute_tokens == 3
    assert generator.calls[1]["ids"] == [1, 10, 31, 7, 9]


def test_reasoning_instruction_can_be_language_localized_privately(tmp_path: Path) -> None:
    config = reasoning_config()
    instruction_file = tmp_path / "reasoning.txt"
    instruction_file.write_text("Use this private language-specific reasoning instruction.", encoding="utf-8")
    config["reasoning"]["scratchpad_instruction_file"] = str(instruction_file)
    generator = StubTextGenerator(config, tmp_path)

    generator.generate_result(prompt="synthetic question")

    assert "Use this private language-specific reasoning instruction." in generator.calls[0]["prompt"]


def test_off_mode_uses_one_answer_phase(tmp_path: Path) -> None:
    config = reasoning_config()
    config["inference"]["reasoning_mode"] = "off"
    generator = StubTextGenerator(config, tmp_path)

    result = generator.generate_result(prompt="synthetic question")

    assert result.reasoning_tokens == 0
    assert result.reasoning_trace is None
    assert result.reasoning_compute_tokens == 0
    assert result.answer_compute_tokens == 1
    assert len(generator.calls) == 1
    assert generator.calls[0]["phase"] == "inference"
    assert generator.calls[0]["prompt"].endswith("<assistant>\n")


def test_zero_effective_budget_falls_back_to_answer_only_mode(tmp_path: Path) -> None:
    config = reasoning_config()
    config["reasoning"]["max_reasoning_tokens"] = 1
    config["inference"]["reasoning_mode"] = "low"
    generator = StubTextGenerator(config, tmp_path)

    result = generator.generate_result(prompt="synthetic question")

    assert result.reasoning_mode == "off"
    assert result.reasoning_tokens == 0
    assert result.candidate_count == 1
    assert result.selector_used is False
    assert len(generator.calls) == 1
    assert generator.calls[0]["prompt"].endswith("<assistant>\n")
    assert "<reasoning:low>" not in generator.calls[0]["prompt"]


def test_max_reasoning_runs_seeded_candidates_and_majority_skips_selector(tmp_path: Path) -> None:
    config = max_reasoning_config()
    generator = MaxTextGenerator(config, tmp_path, ["first answer", "first answer", "third answer"])
    global_before = torch.random.get_rng_state().clone()

    result = generator.generate_result(prompt="synthetic question")

    assert result == GenerationResult(
        answer="first answer",
        reasoning_mode="max",
        reasoning_tokens=1,
        reasoning_trace=None,
        candidate_count=3,
        reasoning_compute_tokens=6,
        answer_compute_tokens=3,
    )
    assert len([call for call in generator.calls if call["phase"] == "reasoning"]) == 3
    assert not any(call["phase"] == "reasoning_selector" for call in generator.calls)
    seeds = [call["generator_seed"] for call in generator.calls if call["phase"] == "reasoning"]
    assert len(set(seeds)) == 3
    assert torch.equal(global_before, torch.random.get_rng_state())


def test_candidate_seed_tracks_private_system_prompt_without_exposing_it(tmp_path: Path) -> None:
    config = max_reasoning_config()
    generator = MaxTextGenerator(config, tmp_path, ["first answer"])
    before = generator._candidate_generator("synthetic question", 0).initial_seed()
    config["inference"]["model_system_prompt"] = "different private instruction"

    after = generator._candidate_generator("synthetic question", 0).initial_seed()

    assert after != before


def test_max_selector_sees_final_answers_only_and_selects_candidate(tmp_path: Path) -> None:
    config = max_reasoning_config()
    generator = MaxTextGenerator(
        config,
        tmp_path,
        ["first answer", "second answer", "third answer"],
        selector="B",
    )

    result = generator.generate_result(prompt="synthetic question")

    selector_calls = [call for call in generator.calls if call["phase"] == "reasoning_selector"]
    assert len(selector_calls) == 1
    selector_prompt = selector_calls[0]["prompt"]
    assert result.answer == "second answer"
    assert result.candidate_count == 3
    assert result.reasoning_compute_tokens == 6
    assert result.answer_compute_tokens == 3
    assert result.selector_compute_tokens == 1
    assert result.selector_used is True
    assert "first answer" in selector_prompt
    assert "second answer" in selector_prompt
    assert "third answer" in selector_prompt
    assert "private scratchpad" not in selector_prompt
    assert selector_calls[0]["generator_seed"] is None


def test_max_invalid_selector_falls_back_to_earliest_candidate(tmp_path: Path) -> None:
    config = max_reasoning_config()
    generator = MaxTextGenerator(
        config,
        tmp_path,
        ["first answer", "second answer", "third answer"],
        selector="B because it looks better",
    )

    result = generator.generate_result(prompt="synthetic question")

    assert result.answer == "first answer"
    assert result.selector_used is True
    assert result.selector_compute_tokens == 1


def test_max_rejects_active_noise_intervention_before_generating_candidates(tmp_path: Path) -> None:
    config = max_reasoning_config()
    generator = MaxTextGenerator(config, tmp_path, ["first answer", "second answer", "third answer"])
    generator.activation_experiment.has_active_stochastic_interventions = lambda: True  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="active noise activation intervention"):
        generator.generate_result(prompt="synthetic question")

    assert generator.calls == []


def test_max_persists_only_selected_reasoning_and_final_trace(tmp_path: Path) -> None:
    config = max_reasoning_config()
    trace_file = tmp_path / "final-token-trace.jsonl"
    config["inference"]["token_trace_file"] = str(trace_file)
    config["reasoning"].update(expose_reasoning_trace=True, save_reasoning_trace=True)
    generator = MaxTextGenerator(
        config,
        tmp_path,
        ["first answer", "second answer", "third answer"],
        selector="B",
    )

    result = generator.generate_result(prompt="synthetic question")

    assert result.answer == "second answer"
    assert result.reasoning_trace == "private scratchpad 1"
    saved = json.loads((tmp_path / "reasoning_trace.jsonl").read_text(encoding="utf-8"))
    assert saved["reasoning_trace"] == "private scratchpad 1"
    assert "private scratchpad 0" not in json.dumps(saved)
    assert "private scratchpad 2" not in json.dumps(saved)
    final_trace = json.loads(trace_file.read_text(encoding="utf-8"))
    assert final_trace["token_ids"] == [42]


def test_max_memory_observes_only_the_selected_final_answer(tmp_path: Path) -> None:
    config = max_reasoning_config()
    generator = MaxTextGenerator(
        config,
        tmp_path,
        ["first answer", "second answer", "third answer"],
        selector="B",
    )
    memory = FakeMemory()
    embed_calls: list[tuple[str, str]] = []
    generator.memory = memory

    def embed(text: str, phase: str) -> list[float]:
        embed_calls.append((text, phase))
        return [1.0]

    generator._embed_text = embed  # type: ignore[method-assign]

    result = generator.generate_result(prompt="synthetic question")

    assert result.answer == "second answer"
    assert len(memory.retrieve_calls) == 1
    assert len(memory.observe_calls) == 1
    assert memory.observe_calls[0][1] == "second answer"
    assert all(answer not in memory.observe_calls[0] for answer in ("first answer", "third answer"))
    assert [phase for _, phase in embed_calls] == ["memory_query", "memory_encoding"]


def test_max_activation_flush_discards_losers_and_selector(tmp_path: Path) -> None:
    config = max_reasoning_config()
    generator = MaxTextGenerator(
        config,
        tmp_path,
        ["first answer", "second answer", "third answer"],
        selector="B",
    )
    activation = RecordingActivationExperiment()
    generator.activation_experiment = activation
    decode = generator._decode
    decode_from_ids = generator._decode_from_ids

    def recording_decode(*args: Any, **kwargs: Any) -> _DecodedPhase:
        phase = decode(*args, **kwargs)
        activation.records.append({"phase": str(kwargs["phase"]), "text": phase.text})
        return phase

    def recording_decode_from_ids(*args: Any, **kwargs: Any) -> _DecodedPhase:
        phase = decode_from_ids(*args, **kwargs)
        activation.records.append({"phase": str(kwargs["phase"]), "text": phase.text})
        return phase

    generator._decode = recording_decode  # type: ignore[method-assign]
    generator._decode_from_ids = recording_decode_from_ids  # type: ignore[method-assign]

    result = generator.generate_result(prompt="synthetic question")

    persisted = json.dumps(activation.records)
    assert result.answer == "second answer"
    assert activation.flush_count == 1
    assert "private scratchpad 1" in persisted
    assert "second answer" in persisted
    assert "private scratchpad 0" not in persisted
    assert "private scratchpad 2" not in persisted
    assert "first answer" not in persisted
    assert "third answer" not in persisted
    assert "reasoning_selector" not in persisted


def test_high_reasoning_keeps_single_legacy_candidate_when_tta_is_enabled(tmp_path: Path) -> None:
    config = max_reasoning_config()
    config["inference"]["reasoning_mode"] = "high"
    generator = MaxTextGenerator(config, tmp_path, ["first answer"])

    assert generator.generate(prompt="synthetic question") == "first answer"
    assert len([call for call in generator.calls if call["phase"] == "reasoning"]) == 1
    assert not any(call["phase"] == "reasoning_selector" for call in generator.calls)
