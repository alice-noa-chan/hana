"""Finish a configured SentencePiece tokenizer from an existing corpus file."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_pipeline.config import load_config  # noqa: E402
from llm_pipeline.data import save_json  # noqa: E402
from llm_pipeline.tokenizer import (  # noqa: E402
    SentencePieceTokenizer,
    publish_tokenizer_bundle,
    sentencepiece_trainer_kwargs,
    validate_tokenizer_candidate,
    verify_tokenizer_corpus_manifest,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    tok_cfg = config["tokenizer"]
    save_dir = Path(tok_cfg["save_dir"])
    corpus = save_dir / "tokenizer_corpus.txt"
    if not corpus.is_file() or corpus.stat().st_size == 0:
        raise FileNotFoundError(f"Tokenizer corpus is missing or empty: {corpus}")
    corpus_manifest = verify_tokenizer_corpus_manifest(config.mutable_copy(), corpus)

    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before finalizing the tokenizer.") from exc

    specials = tok_cfg["special_tokens"]
    with tempfile.TemporaryDirectory(prefix="tokenizer-finalize-", dir=save_dir) as temporary:
        prefix = Path(temporary) / "tokenizer"
        spm.SentencePieceTrainer.train(
            **sentencepiece_trainer_kwargs(
                config,
                input_path=corpus,
                model_prefix=prefix,
                num_threads=max(1, os.cpu_count() or 1),
            )
        )
        candidate = SentencePieceTokenizer(prefix.with_suffix(".model"), specials)
        if candidate.vocab_size != int(tok_cfg["vocab_size"]):
            raise RuntimeError(
                f"Tokenizer produced vocab_size={candidate.vocab_size}, expected {tok_cfg['vocab_size']}."
            )
        validation = validate_tokenizer_candidate(candidate, config, corpus)
        tokenizer = publish_tokenizer_bundle(
            candidate,
            config,
            validation,
            save_dir,
            vocab_path=prefix.with_suffix(".vocab"),
        )
    corpus_manifest.update(
        input_sentence_size=int(tok_cfg["input_sentence_size"]),
        vocab_size=tokenizer.vocab_size,
        finalized_from_existing_corpus=True,
    )
    save_json(save_dir / "corpus_manifest.json", corpus_manifest)
    print(f"Tokenizer ready: vocab={tokenizer.vocab_size:,}, corpus_bytes={corpus.stat().st_size:,}")
    print(
        "Validation passed: "
        f"probes={validation['probe_count']}, "
        f"corpus_samples={validation['corpus_samples_checked']}, "
        f"unknown_tokens={validation['unk_count']}, "
        f"model_sha256={validation['model_sha256']}"
    )


if __name__ == "__main__":
    main()
