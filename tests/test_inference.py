from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

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
    def flush(self) -> None:
        return None


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
    ) -> _DecodedPhase:
        self.calls.append(
            {
                "kind": "prompt",
                "prompt": rendered_prompt,
                "phase": phase,
                "reserve": reserve_new_tokens,
                "allow_boundary": allow_reasoning_boundary,
                "trace": allow_token_trace,
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
    ) -> _DecodedPhase:
        self.calls.append(
            {
                "kind": "ids",
                "ids": input_ids.tolist()[0],
                "phase": phase,
                "max_new_tokens": max_new_tokens,
                "trace": allow_token_trace,
            }
        )
        return _DecodedPhase(
            text="final answer",
            input_ids=input_ids,
            token_ids=(41,),
            trace=[{"phase": "ar", "token_ids": [41]}] if allow_token_trace else None,
        )


def reasoning_config() -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["reasoning"]["max_reasoning_tokens"] = 8
    config["inference"].update(reasoning_mode="medium", max_new_tokens=3)
    return config


def test_reasoning_control_is_adjacent_to_the_assistant_cue() -> None:
    config = reasoning_config()

    prompt = build_inference_prompt(config, "synthetic question")

    assert prompt.endswith("<reasoning:medium>\n<assistant>\n")
    assert not prompt.startswith("<reasoning:medium>")
    assert reasoning_token_budget(config, "low") == 2
    assert reasoning_token_budget(config, "medium") == 4
    assert reasoning_token_budget(config, "high") == 8
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
    assert len(generator.calls) == 1
    assert generator.calls[0]["prompt"].endswith("<assistant>\n")
    assert "<reasoning:low>" not in generator.calls[0]["prompt"]
