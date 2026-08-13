"""Reject dataset artifacts and private profile assets from the public tree."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_ROOTS = (
    PurePosixPath(".smoke"),
    PurePosixPath("train_data"),
    PurePosixPath("private_data"),
    PurePosixPath("private_profiles"),
    PurePosixPath("profile_assets"),
    PurePosixPath("personas"),
    PurePosixPath("data"),
    PurePosixPath("datasets"),
    PurePosixPath("corpora"),
    PurePosixPath("tests/fixtures"),
    PurePosixPath("checkpoints"),
    PurePosixPath("logs"),
    PurePosixPath("exports"),
    PurePosixPath("runs"),
    PurePosixPath("cognitive_state"),
    PurePosixPath("gpu_transfer"),
    PurePosixPath("build"),
    PurePosixPath("dist"),
)
FORBIDDEN_DIRECTORY_NAMES = {
    ".smoke",
    "checkpoints",
    "cognitive_state",
    "corpora",
    "corpus",
    "data",
    "dataset",
    "datasets",
    "exports",
    "gpu_transfer",
    "logs",
    "personas",
    "private_data",
    "private_profiles",
    "profile_assets",
    "profiles",
    "prompts",
    "runs",
}
FORBIDDEN_SUFFIXES = (
    ".json",
    ".jsonl",
    ".ndjson",
    ".json.gz",
    ".jsonl.gz",
    ".csv",
    ".csv.gz",
    ".tsv",
    ".tsv.gz",
    ".parquet",
    ".arrow",
    ".feather",
    ".pdf",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".bin",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".h5",
    ".hdf5",
    ".safetensors",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".gguf",
    ".ggml",
    ".model",
    ".vocab",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".7z",
    ".rar",
    ".zst",
    ".log",
    ".sha256",
    ".md5",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".wav",
    ".mp3",
    ".mp4",
)
FORBIDDEN_BASENAMES = {
    "benchmark_denylist.txt",
    "corpus_manifest.json",
    "data_audit.json",
    "sources.lock.json",
    "tokenizer_corpus.txt",
}
ALLOWED_TEXT_FILES = {"requirements.txt", "requirements-dev.txt"}
ALLOWED_MEDIA_FILES = {"docs/assets/hana-character.png"}
ALLOWED_SCRIPTS = {
    "scripts/check_public_tree.py",
    "scripts/finalize_tokenizer_from_corpus.py",
    "scripts/prepare_synthetic_smoke.py",
}
COMMENT_SUFFIXES = {".py", ".yaml", ".yml", ".toml"}
DOCUMENT_SUFFIXES = {".md"}
NON_ENGLISH_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
LITERAL_SHA256 = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
PERSONAL_ABSOLUTE_PATH = re.compile(
    r"(?i)(?:[a-z]:" + r"\\users\\" + r"[^\\\s]+|/ho" + r"me/[^/\s]+|/us" + r"ers/[^/\s]+)"
)
PUBLIC_TEXT_SUFFIXES = {".md", ".py", ".toml", ".txt", ".yaml", ".yml"}


def tracked_paths(root: Path = PROJECT_ROOT) -> list[str]:
    """Return tracked plus non-ignored untracked paths from Git."""

    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
    )
    return [value.decode("utf-8") for value in output.split(b"\0") if value]


def _under(path: PurePosixPath, root: PurePosixPath) -> bool:
    return path == root or root in path.parents


def _path_violations(root: Path, paths: Iterable[str], *, require_current_file: bool = True) -> list[str]:
    violations = []
    for value in paths:
        relative = PurePosixPath(value)
        local_path = root.joinpath(*relative.parts)
        if any(_under(relative, forbidden) for forbidden in FORBIDDEN_ROOTS):
            violations.append(f"private or dataset directory is tracked: {relative}")
        if any(part.casefold() in FORBIDDEN_DIRECTORY_NAMES for part in relative.parts[:-1]):
            violations.append(f"private or dataset directory component is tracked: {relative}")
        lower_name = relative.name.casefold()
        if relative.as_posix() not in ALLOWED_MEDIA_FILES and (
            lower_name in FORBIDDEN_BASENAMES or any(lower_name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)
        ):
            violations.append(f"dataset or model artifact is tracked: {relative}")
        if lower_name.startswith(".env") or ".local." in lower_name:
            violations.append(f"local configuration or secret file is tracked: {relative}")
        if relative.suffix.casefold() == ".txt" and relative.name not in ALLOWED_TEXT_FILES:
            violations.append(f"unapproved text payload is tracked: {relative}")
        if relative.parts and relative.parts[0] == "scripts" and relative.as_posix() not in ALLOWED_SCRIPTS:
            violations.append(f"unreviewed public script is tracked: {relative}")
        if require_current_file and not local_path.is_file():
            continue
    return violations


def historical_paths(root: Path = PROJECT_ROOT) -> list[str]:
    """Return every path recorded in the history reachable from ``HEAD``."""

    output = subprocess.check_output(
        ["git", "log", "--format=", "--name-only", "--diff-filter=AM", "HEAD"],
        cwd=root,
    )
    return sorted({line.strip() for line in output.decode("utf-8").splitlines() if line.strip()})


def _language_violations(root: Path, paths: Iterable[str]) -> list[str]:
    violations = []
    for value in paths:
        relative = PurePosixPath(value)
        local_path = root.joinpath(*relative.parts)
        if not local_path.is_file() or relative.suffix.casefold() not in COMMENT_SUFFIXES | DOCUMENT_SUFFIXES:
            continue
        try:
            lines = local_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            stripped = line.lstrip()
            is_document = relative.suffix.casefold() in DOCUMENT_SUFFIXES
            is_comment = stripped.startswith("#")
            if (is_document or is_comment) and NON_ENGLISH_SCRIPT.search(line):
                kind = "documentation" if is_document else "comment"
                violations.append(f"non-English {kind}: {relative}:{line_number}")
    return violations


def _content_violations(root: Path, paths: Iterable[str]) -> list[str]:
    """Reject obvious private evidence embedded in otherwise allowed text files."""

    violations = []
    for value in paths:
        relative = PurePosixPath(value)
        local_path = root.joinpath(*relative.parts)
        if not local_path.is_file() or relative.suffix.casefold() not in PUBLIC_TEXT_SUFFIXES:
            continue
        try:
            content = local_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if PERSONAL_ABSOLUTE_PATH.search(content):
            violations.append(f"personal absolute path is embedded in public text: {relative}")
        if LITERAL_SHA256.search(content):
            violations.append(f"literal SHA-256 evidence is embedded in public text: {relative}")
        if relative.parts and relative.parts[0] not in {"tests"} and NON_ENGLISH_SCRIPT.search(content):
            violations.append(f"non-English literal is embedded outside synthetic tests: {relative}")
        if relative.suffix.casefold() in {".yaml", ".yml"}:
            for match in re.finditer(r"(?m)^\s*source_url:\s*[\"']?([^\s\"']+)", content):
                url = match.group(1)
                if not (url.startswith("https://example.invalid/") or url.startswith("internal://project/")):
                    violations.append(f"non-placeholder source_url is embedded in public YAML: {relative}")
    return violations


def _config_violations(root: Path) -> list[str]:
    paths = [root / "config.yaml"]
    configs_dir = root / "configs"
    if configs_dir.is_dir():
        paths.extend(path for path in configs_dir.glob("*.yaml") if path.name != "sources.example.yaml")
    if not paths[0].is_file():
        return ["public config.yaml is missing"]
    violations = []
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            violations.append(f"public configuration must contain a mapping: {relative}")
            continue
        data = payload.get("data", {})
        if data.get("sources"):
            violations.append(f"public configuration must not contain data.sources entries: {relative}")
        inference = payload.get("inference", {})
        if inference.get("model_system_prompt") or inference.get("model_system_prompt_files"):
            violations.append(f"public configuration must not contain private model prompts: {relative}")
        if inference.get("user_system_prompt") or inference.get("user_system_prompt_file"):
            violations.append(f"public configuration must not contain private user prompts: {relative}")
        if payload.get("dpo", {}).get("prompt_sources"):
            violations.append(f"public configuration must not name private DPO sources: {relative}")
        if payload.get("reasoning", {}).get("scratchpad_instruction_file"):
            violations.append(f"public configuration must not contain a private reasoning prompt path: {relative}")
        knowledge_pilot = payload.get("eval", {}).get("knowledge_pilot", {})
        if knowledge_pilot.get("enabled") or knowledge_pilot.get("file") or knowledge_pilot.get("prompt_file"):
            violations.append(
                f"public configuration must keep the private knowledge pilot disabled and data-free: {relative}"
            )
    return violations


def find_publication_violations(root: Path = PROJECT_ROOT, paths: Iterable[str] | None = None) -> list[str]:
    """Return every code-only publication-boundary violation."""

    selected = list(paths) if paths is not None else tracked_paths(root)
    return [
        *_path_violations(root, selected),
        *_language_violations(root, selected),
        *_content_violations(root, selected),
        *_config_violations(root),
    ]


def find_history_violations(root: Path = PROJECT_ROOT) -> list[str]:
    """Return forbidden paths from every commit reachable from ``HEAD``."""

    return [
        f"reachable Git history contains {violation}"
        for violation in _path_violations(root, historical_paths(root), require_current_file=False)
    ]


def main() -> int:
    """Print violations and return a stable process status."""

    violations = [*find_publication_violations(), *find_history_violations()]
    if violations:
        print("Public-tree validation failed:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    print("Public-tree validation passed: code, CJK-free public prose, and synthetic test scaffolding only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
