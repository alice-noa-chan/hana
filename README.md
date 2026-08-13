# Hana

<p align="center">
  <img src="docs/assets/hana-character.png" alt="Hana, a pink-haired student holding a book and a pen beside a small robot companion" width="560">
</p>

<p align="center">
  <strong>A fail-closed, reusable PyTorch pipeline for decoder-only language-model research.</strong>
</p>

Hana is a language-model project built around a fictional human persona. In a private Hana profile, she is represented as a nine-year-old Korean-Japanese girl with a Korean father and a Japanese mother. The illustration expresses her curiosity, school life, love of words, and warm conversational style.

> [!IMPORTANT]
> This is a code-only public repository. It ships pipeline code, schemas, placeholder configuration, English documentation, and generated synthetic test scaffolding. It does not distribute or identify any production training corpus, evaluation set, local source manifest, generated split, tokenizer corpus, benchmark item, persona dataset, or private runtime prompt.

Every operator must assemble, review, license, and audit a private local data pack. A model produced from private data must publish its own appropriate model card and data statement outside this repository.

## Character and software boundary

Hana's conversational identity is a fictional person, not a claim about the software itself.

- Private persona conversations may keep Hana consistently inside her fictional human identity.
- Public technical documentation states clearly that Hana is software.
- The character image does not prove intelligence, safety, accuracy, or consciousness.
- No persona training examples or persona system prompts are included in Git.
- A local operator may attach private prompts through ignored configuration files.

The small robot in the illustration is Hana's companion. It is not Hana's identity.

## What the public repository contains

The checked-in architecture is a decoder-only Transformer with exactly 303,350,784 parameters when built with the published production dimensions and 32,000-token vocabulary.

The code provides:

1. immutable schema-v2 configuration
2. private local data-pack loading
3. source provenance and policy validation
4. SHA-256 source locks
5. privacy, safety, and benchmark-component filtering
6. SentencePiece tokenizer training with provenance
7. pretraining, supervised fine-tuning, and DPO
8. checkpoint save, resume, and freshness checks
9. evaluation and inference entry points
10. full-precision and dynamic INT8 export
11. CPU and distributed smoke-test paths
12. experiment-contract validation for architecture, portability, and systems studies

The public `config.yaml` intentionally contains no source entries, no private language list, no persona prompt paths, and no DPO prompt-source names.

## What the public repository does not contain

The following material must remain outside Git:

- training, validation, and test rows
- source-provider names selected by a private run
- source URLs or download recipes for a private run
- local corpus sizes, row counts, token counts, and mixture ratios
- tokenizer corpora and tokenizer model files
- source locks, audit reports, and benchmark denylists
- persona examples, templates, cards, style data, and generated splits
- system, developer, style, safety, and refusal prompts for a private profile
- checkpoints, exports, traces, logs, and memory stores
- benchmark questions, passages, choices, answers, explanations, or derivatives

The `.gitignore` file protects common local roots and artifact formats. The repository verification command also rejects listed private paths and artifact formats if they are force-added.

## Requirements

- Python 3.10, 3.11, or 3.12
- PyTorch 2.3 or newer, but earlier than 3.0
- enough local storage for the operator's private data and generated artifacts
- a CUDA GPU for practical full-size training

CPU execution is intended for tests and small smoke runs. It is not a practical full-training target for the published architecture.

## Installation

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Bash:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

No public command downloads a production dataset.

## Prepare a private local data pack

Start with the placeholder schema in [configs/sources.example.yaml](configs/sources.example.yaml). Copy it into the ignored local data directory:

```powershell
New-Item -ItemType Directory -Force train_data | Out-Null
Copy-Item configs/sources.example.yaml train_data/sources.local.yaml
```

Then replace every placeholder locally. Do not commit the completed file.

A pack declares:

- a private pack name and version
- language tags used by that private pack
- one unique logical name per source
- local file paths or patterns
- training or evaluation purpose
- applicable stages
- explicit split behavior
- schema and field names
- domain metadata
- deterministic sampling rate when needed
- provenance, license, privacy, child-safety, and reviewer evidence

Language tags are metadata. The loader does not prove that a row is written in the declared language. The operator must validate content language and mixture quality privately.

### Translation data

Parallel data is not tied to one language pair. A private source declares:

- `source_lang_field`
- `target_lang_field`
- `source_lang`
- `target_lang`
- a natural `prompt_template`

The public core does not infer a Korean-Japanese schema or inject a Hana prompt.

### Private profile prompts

Keep private identity and behavior prompts under an ignored directory such as `private_profiles/`. Point to them from the ignored `config.local.yaml`:

```yaml
schema_version: 2

inference:
  model_system_prompt_files:
    - "./private_profiles/system.md"
    - "./private_profiles/safety.md"
```

This keeps the character artwork and software architecture public while leaving the actual behavior corpus and runtime profile private.

## Create local safety evidence

Production data commands require evidence generated from the operator's private files.

```powershell
hana data lock --config config.yaml
hana data audit --config config.yaml
hana doctor --config config.yaml --allow-cpu --verify-hashes
```

The production configuration also requires a non-empty private benchmark denylist at:

```text
train_data/benchmark_denylist.txt
```

Each line is a lowercase SHA-256 hash of one canonical evaluation component. Canonicalization collapses whitespace, applies Unicode-aware case folding, encodes UTF-8, and then hashes the result.

This filter performs exact normalized matching. It does not detect paraphrases, substrings, or semantic similarity. The operator must add every relevant component and derivative separately and must still perform human contamination review.

Do not invent placeholder hashes to make the audit pass.

## Run the pipeline

Train one stage:

```powershell
hana run --config config.yaml --mode pretrain
```

Continue through all runnable stages:

```powershell
hana run --config config.yaml --mode auto --continue
```

Repeat one completed stage:

```powershell
hana run --config config.yaml --mode pretrain --force
```

`--force` repeats work. It never bypasses source policy, locking, audit, integrity, or environment checks.

The configured stage sequence is:

1. `train_tokenizer`
2. `analyze_data`
3. `pretrain`
4. `sft`
5. `build_rejects`
6. `dpo`
7. `eval`
8. `inference`
9. `export`
10. `quantize`

DPO is disabled in the public configuration. A private override must enable it and explicitly list reviewed local SFT source names.

## Command reference

| Goal | Command |
|---|---|
| Validate the public repository | `hana verify` |
| Lock private sources | `hana data lock --config config.yaml` |
| Audit private sources | `hana data audit --config config.yaml` |
| Check environment and evidence | `hana doctor --config config.yaml --allow-cpu` |
| Run one stage | `hana run --config config.yaml --mode pretrain` |
| Continue the pipeline | `hana run --config config.yaml --mode auto --continue` |
| Validate an experiment contract | `hana experiment validate --registry configs/experiments/registry.example.yaml` |
| Build a transfer manifest | `hana transfer manifest --config config.yaml --output gpu_transfer` |
| Build a verified transfer archive | `hana transfer bundle --config config.yaml --output gpu_transfer --name private-run.zip` |

## Declared evaluation sources are excluded from training

An evaluation source must declare:

```yaml
purpose: "evaluation"
stages: ["eval"]
split: "test"
tokenizer: false
```

The pipeline excludes evaluation sources from tokenizer training, pretraining, SFT, and DPO even when ordinary source-policy enforcement is disabled. It also rejects a path or byte-identical file assigned to both training and evaluation.

An evaluation-source declaration is quarantine metadata. The generic evaluation stage does not automatically execute a third-party benchmark. The operator must use a separate licensed evaluator and must never route its content into a training source.

## Reusable data-pack boundary

The model and trainer are language-neutral. Identity and data are separate:

- `profile` provides a display identity and memory speaker label.
- `data.pack` describes a private dataset collection.
- `data.sources` describes private inputs and their policy evidence.
- translation fields describe arbitrary language pairs explicitly.
- deterministic `sample_rate` provides basic source downsampling.

Training a fresh tokenizer for another private pack is supported. Expanding the vocabulary of an existing checkpoint is not currently supported because checkpoint tensor loading is strict. Continued language expansion therefore needs either a stable broad tokenizer or a separately tested token-ID and embedding migration.

## Research claims require contracts

Experimental mechanisms are not described as faster, better, or novel merely because they are implemented.

The schema-v1 experiment contract requires:

- one baseline
- exact changed configuration keys
- shared seeds
- a shared compute or token budget
- one primary metric
- required metrics
- regression guardrails
- conservative decisions

Supported study kinds are:

- `architecture_ablation`
- `data_portability`
- `systems_speed`

Validate a registry:

```powershell
hana experiment validate --registry configs/experiments/registry.example.yaml
```

See [docs/EXPERIMENT_CONTRACT.md](docs/EXPERIMENT_CONTRACT.md) for the complete format.

## Synthetic smoke testing

The repository does not store a smoke dataset. Generate meaningless ASCII-only input under the ignored `.smoke` directory:

```powershell
python scripts/prepare_synthetic_smoke.py
python -m llm_pipeline run --config configs/smoke.yaml --mode train_tokenizer --force
```

The generated input exists only to exercise tokenizer, training, evaluation, inference, and export code paths. It is not production training data and must never be presented as a production corpus.

## Verification

Run the complete gate before committing or pushing:

```powershell
hana verify
```

The command performs:

1. public-tree dataset and private-profile checks
2. checks that public documentation and developer comments contain no CJK text
3. Python bytecode compilation
4. Ruff lint checks
5. Ruff formatting checks
6. the complete pytest suite
7. focused branch-coverage gates

GitHub Actions repeats the gate and runs a small two-process CPU/Gloo test using generated synthetic input.

## Repository map

```text
llm_pipeline/                 # Model, data interfaces, governance, training, and export code.
configs/                      # Placeholder, smoke, and experiment-contract configurations.
docs/                         # English architecture, research, and roadmap documentation.
scripts/                      # Public-tree checks and generic tokenizer/smoke utilities.
tests/                        # Unit, integration, regression, and publication-boundary tests.
config.yaml                   # Data-free public architecture configuration.
```

Private data roots, profiles, prompts, tokenizers, checkpoints, and run evidence are intentionally absent.

## Honest limitations

- The code cannot decide whether an operator has a valid license.
- Exact hash filtering cannot detect every contaminated derivative.
- Automatic privacy and safety filters can miss harmful content.
- A fictional human persona does not make the software human or conscious.
- The full-size configuration still requires private hardware validation and capacity planning.
- Experimental structures need fair ablations before promotion.
- Dynamic INT8 export currently relies on PyTorch's dynamic quantization API.

## Further reading

- [Capability and scale roadmap](docs/ROADMAP.md)
- [Experiment contracts](docs/EXPERIMENT_CONTRACT.md)
- [Internal-dynamics research lab](docs/RESEARCH_LAB.md)
- [Cognitive architecture](docs/COGNITIVE_ARCHITECTURE.md)

No Markdown file grants permission to train. Training authorization depends on the operator's licenses and provenance review. Reproducibility evidence includes the private source lock, benchmark denylist, complete audit, and tokenizer-corpus fingerprint.
