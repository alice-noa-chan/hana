"""Generate deterministic ASCII-only input for repository smoke tests.

The generated text is test scaffolding. It is not a training corpus, a persona
asset, or a sample from any private data pack. The output lives under the
ignored ``.smoke`` directory and is never committed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / ".smoke" / "input" / "synthetic.jsonl"
WORDS = (
    "alpha",
    "bravo",
    "cobalt",
    "delta",
    "ember",
    "forest",
    "globe",
    "harbor",
    "island",
    "jigsaw",
    "kernel",
    "lantern",
    "matrix",
    "nectar",
    "orbit",
    "puzzle",
    "quartz",
    "river",
    "signal",
    "timber",
    "unison",
    "vector",
    "window",
    "xenon",
    "yellow",
    "zenith",
)


def synthetic_lines(line_count: int = 512) -> list[str]:
    """Return deterministic, meaningless lines with enough variation for BPE."""

    if line_count < 32:
        raise ValueError("Synthetic smoke input requires at least 32 lines.")
    lines = []
    for index in range(line_count):
        selected = [WORDS[(index * 5 + offset * 7) % len(WORDS)] for offset in range(8)]
        lines.append(
            f"Synthetic record {index:04d} checks tokenization training evaluation and export "
            + " ".join(selected)
            + "."
        )
    return lines


def write_smoke_fixture(path: Path = DEFAULT_OUTPUT, line_count: int = 512) -> Path:
    """Atomically write the ignored synthetic smoke input and return its path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    records = (json.dumps({"text": line}, ensure_ascii=True) for line in synthetic_lines(line_count))
    try:
        temporary.write_text("\n".join(records) + "\n", encoding="utf-8", newline="\n")
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main() -> None:
    """Write a requested synthetic fixture from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lines", type=int, default=512)
    args = parser.parse_args()
    output = write_smoke_fixture(args.output, args.lines)
    print(f"Synthetic smoke input ready: {output} ({args.lines} lines)")


if __name__ == "__main__":
    main()
