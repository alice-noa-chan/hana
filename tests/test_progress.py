from __future__ import annotations

import json
from pathlib import Path

from llm_pipeline.artifacts import tokenizer_training_fingerprint
from llm_pipeline.config import load_config
from llm_pipeline.data import source_manifest_fingerprint
from llm_pipeline.progress import mode_is_complete

ROOT = Path(__file__).resolve().parents[1]


def test_tokenizer_progress_invalidates_changed_special_tokens(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").mutable_copy()
    config["tokenizer"]["save_dir"] = str(tmp_path)
    for name in ("tokenizer.model", "tokenizer.json", "special_tokens_map.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    metadata = {
        "target_vocab_size": config["tokenizer"]["vocab_size"],
        "data_sources_fingerprint": source_manifest_fingerprint(config, "tokenizer"),
        "training_fingerprint": tokenizer_training_fingerprint(config),
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert mode_is_complete("train_tokenizer", config)

    config["tokenizer"]["special_tokens"]["mask"] = "<different-mask>"
    assert not mode_is_complete("train_tokenizer", config)
