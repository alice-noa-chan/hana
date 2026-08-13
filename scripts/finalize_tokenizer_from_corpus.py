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
from llm_pipeline.tokenizer import SentencePieceTokenizer, verify_tokenizer_corpus_manifest  # noqa: E402


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
    user_defined = [
        specials["user"],
        specials["assistant"],
        specials["system"],
        specials["reasoning_off"],
        specials["reasoning_low"],
        specials["reasoning_medium"],
        specials["reasoning_high"],
        specials["mask"],
    ]
    with tempfile.TemporaryDirectory(prefix="tokenizer-finalize-", dir=save_dir) as temporary:
        prefix = Path(temporary) / "tokenizer"
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(prefix),
            model_type=tok_cfg["model_type"],
            vocab_size=int(tok_cfg["vocab_size"]),
            character_coverage=float(tok_cfg["character_coverage"]),
            input_sentence_size=int(tok_cfg["input_sentence_size"]),
            shuffle_input_sentence=bool(tok_cfg["shuffle_input_sentence"]),
            byte_fallback=bool(tok_cfg["byte_fallback"]),
            hard_vocab_limit=False,
            num_threads=max(1, os.cpu_count() or 1),
            unk_id=0,
            bos_id=1,
            eos_id=2,
            pad_id=3,
            pad_piece=specials["pad"],
            unk_piece=specials["unk"],
            bos_piece=specials["bos"],
            eos_piece=specials["eos"],
            user_defined_symbols=",".join(user_defined),
        )
        candidate = SentencePieceTokenizer(prefix.with_suffix(".model"), specials)
        if candidate.vocab_size != int(tok_cfg["vocab_size"]):
            raise RuntimeError(
                f"Tokenizer produced vocab_size={candidate.vocab_size}, expected {tok_cfg['vocab_size']}."
            )
        prefix.with_suffix(".model").replace(save_dir / "tokenizer.model")
        prefix.with_suffix(".vocab").replace(save_dir / "tokenizer.vocab")

    tokenizer = SentencePieceTokenizer(save_dir / "tokenizer.model", specials)
    tokenizer.save_metadata(save_dir, config)
    corpus_manifest.update(
        input_sentence_size=int(tok_cfg["input_sentence_size"]),
        vocab_size=tokenizer.vocab_size,
        finalized_from_existing_corpus=True,
    )
    save_json(save_dir / "corpus_manifest.json", corpus_manifest)
    with corpus.open("r", encoding="utf-8") as handle:
        probe = next((line.strip() for line in handle if line.strip()), "")
    if not probe:
        raise RuntimeError("Tokenizer corpus has no non-empty round-trip probe row.")
    encoded = tokenizer.encode(probe)
    decoded = tokenizer.decode(encoded)
    if not decoded:
        raise RuntimeError("Tokenizer round-trip probe decoded to an empty string.")
    print(f"Tokenizer ready: vocab={tokenizer.vocab_size:,}, corpus_bytes={corpus.stat().st_size:,}")
    print(f"Round-trip: {decoded}")


if __name__ == "__main__":
    main()
