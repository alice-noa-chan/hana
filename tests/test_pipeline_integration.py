from __future__ import annotations

import json
from pathlib import Path

import yaml

from llm_pipeline.config import load_config, redacted_config_for_artifact
from llm_pipeline.main import main
from llm_pipeline.tokenizer import load_tokenizer
from scripts.prepare_synthetic_smoke import write_smoke_fixture

ROOT = Path(__file__).resolve().parents[1]


def test_complete_cpu_pipeline(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").mutable_copy()
    config.pop("__config_path__", None)
    config["run"]["output_dir"] = str(tmp_path / "checkpoints")
    config["run"]["experiment_name"] = "integration"
    fixture = write_smoke_fixture(tmp_path / "synthetic.jsonl", line_count=96)
    config["data"].update(
        train_file=str(fixture),
        valid_file=str(fixture),
        test_file=str(fixture),
        processed_dir=str(tmp_path / "processed"),
        token_cache_dir=str(tmp_path / "token_cache"),
        sources_file=None,
        sources=[],
    )
    config["tokenizer"].update(
        save_dir=str(tmp_path / "tokenizer"),
        model_path=str(tmp_path / "tokenizer/tokenizer.model"),
    )
    config["export"].update(
        export_non_it_dir=str(tmp_path / "exports/non_it"),
        export_it_dir=str(tmp_path / "exports/it"),
        export_quantized_dir=str(tmp_path / "exports/quantized"),
    )
    config["experiments"]["output_dir"] = str(tmp_path / "experiments")
    config["inference"]["token_trace_file"] = str(tmp_path / "experiments/token_trace.jsonl")
    config["cognitive_architecture"]["memory"]["path"] = str(tmp_path / "cognitive_state/memory.json")
    config_path = tmp_path / "integration.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    main(config_path=str(config_path), run_mode="auto", auto_continue=True)

    root = tmp_path / "checkpoints/integration"
    checkpoint = root / "pretrain/best"
    logs = root / "logs"
    exported = tmp_path / "exports/non_it"
    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "checkpoint_manifest.json").is_file()
    assert (logs / "data_stats.json").is_file()
    assert (logs / "eval_summary.md").is_file()
    assert (logs / "inference.json").is_file()
    assert (exported / "export_manifest.json").is_file()
    export_manifest = json.loads((exported / "export_manifest.json").read_text(encoding="utf-8"))
    assert export_manifest["source_checkpoint"] == "best"
    assert str(tmp_path) not in json.dumps(export_manifest)
    tokenizer = load_tokenizer(config)
    boundary_token = config["tokenizer"]["special_tokens"]["reasoning_off"]
    boundary_ids = tokenizer.encode(f"\n{boundary_token}\n", add_special_tokens=False)
    assert boundary_ids.count(tokenizer.piece_to_id(boundary_token)) == 1
    assert len(boundary_ids) > 1
    assert (tmp_path / "experiments/pretrain/activations.jsonl").stat().st_size > 0
    assert (tmp_path / "experiments/pretrain/gradients.jsonl").stat().st_size > 0
    assert (tmp_path / "experiments/inference/activations.jsonl").stat().st_size > 0
    assert (tmp_path / "experiments/token_trace.jsonl").stat().st_size > 0
    assert (tmp_path / "cognitive_state/memory.json").stat().st_size > 0

    results = [json.loads(line) for line in (logs / "eval_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(results) == 2
    assert all(math_value < float("inf") for math_value in (row["perplexity"] for row in results))
    inference = json.loads((logs / "inference.json").read_text(encoding="utf-8"))
    assert inference["status"] == "ok"
    assert "<assistant>" not in inference["output"]

    # A second auto run must recognize every fresh artifact and perform no work.
    before = (logs / "run.log").stat().st_size
    main(config_path=str(config_path), run_mode="auto", auto_continue=True)
    after = (logs / "run.log").stat().st_size
    assert after > before
    assert "No pending runnable mode remains" in (logs / "run.log").read_text(encoding="utf-8")

    # Checkpoint configs stay free of local dataset paths.
    redacted = redacted_config_for_artifact(load_config(config_path))
    assert redacted["data"]["train_file"] == "<redacted-local-data>"
    assert redacted["dpo"]["train_file"] == "<redacted-local-data>"
    assert redacted["cognitive_architecture"]["memory"]["path"] == "<redacted-local-state>"
