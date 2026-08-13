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

The checked-in architecture is a decoder-only Transformer with exactly 303,353,856 parameters when built with the published production dimensions and 32,000-token vocabulary. The count includes QK normalization in every attention layer.

Two paper-inspired architecture experiments are implemented but disabled. `model.attention_output_gate` adds one query-dependent sigmoid scalar per attention head after attention and before the output projection. At the published dimensions it adds 24,960 parameters. Its weights start at zero and `attention_output_gate_bias: 2.0` starts every gate at about 0.881, avoiding a random half-scale perturbation at initialization. `model.sliding_window.layer_pattern` can repeat an exact schedule such as `[full, sliding, sliding, sliding]` while keeping the same model weights and full KV-cache shape. The current sliding implementation uses an exact mask but retains every cached key and value and may still use dense SDPA work. It is therefore a quality and locality experiment, not a claimed speed or memory optimization. Both candidates require a controlled architecture-ablation contract before promotion.

QK normalization and the output gate add checkpoint tensors, so their values must match the checkpoint architecture. The layer pattern adds no tensors and can reuse weights for evaluation, but changing its masks still changes model behavior and invalidates cached results.

The code provides:

1. immutable schema-v2 configuration
2. private local data-pack loading
3. source provenance and policy validation
4. SHA-256 source locks
5. privacy, safety, and benchmark-component filtering
6. SentencePiece tokenizer training with provenance
7. pretraining, supervised fine-tuning, and DPO
8. checkpoint save, resume, and freshness checks
9. five-level hidden reasoning with a bounded maximum-effort search mode
10. a contamination-aware private multiple-choice pilot
11. evaluation and inference entry points
12. full-precision and dynamic INT8 export
13. CPU and distributed smoke-test paths
14. experiment-contract validation for architecture, portability, and systems studies

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

The automated publication check catches structural warning signs such as private directories, artifact formats, local configuration files, personal absolute paths, literal evidence hashes, and non-placeholder source URLs in public YAML. It is not a universal classifier for every dataset title, provider name, download link, or training recipe that someone could write in prose or a code comment. Before publishing a change, a human reviewer must still inspect every changed document, comment, and configuration example for real source identities, links, recipes, and run-specific statistics.

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
| Quarantine a private evaluation pilot | `hana data quarantine-eval --config config.yaml` |
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

An evaluation-source declaration is quarantine metadata. It does not automatically execute a third-party benchmark. The fixed private knowledge pilot described below is the only built-in multiple-choice adapter, and it reads a separate ignored file only when explicitly enabled. Every other official suite still needs an isolated, licensed evaluator.

## Hidden reasoning

Reasoning is a real two-phase generation path, not only a label placed in front of a prompt. The final answer continues from the same token context after a private reasoning boundary. The normal `generate()` method still returns only a string containing the final answer.

| Mode | Behavior |
|---|---|
| `off` | One direct answer pass and no scratchpad. |
| `low` | One private scratchpad with 25% of `max_reasoning_tokens`. |
| `medium` | One private scratchpad with 50% of the limit. |
| `high` | One private scratchpad with 75% of the limit. |
| `max` | The full scratchpad limit plus configurable multi-candidate test-time search. |

These values are upper bounds, not requested output lengths or quality grades. Generation may stop early. A larger budget is useful only when measured final-answer quality improves. The position-limit guard reserves room for the final answer before a scratchpad is generated.

### Maximum effort

`max` is a Hana experiment. It is not a decoding method claimed by any cited paper. With the public defaults it generates three independently sampled reasoning-and-answer candidates using `temperature: 1.0` and `top_p: 0.95`. The candidate count is configurable from 2 through 26, but time and token use grow roughly in proportion to that count.

A strict majority of normalized final answers wins. When no strict majority exists, the same local model runs a deterministic private selector. That selector sees the original request and bounded copies of the candidate final answers. It never sees candidate scratchpads. An invalid selector response, or a selector prompt that cannot fit the context, falls back to the earliest candidate.

Each candidate receives a local random-number stream derived from the reasoning protocol version, `run.seed`, the canonical prompt, and the candidate index. Candidate sampling does not reseed or consume the global PyTorch random-number stream. Reproducibility means a repeat with the same model, tokenizer, code, device, and sampling backend. It is not a promise of bit-identical output across different hardware or PyTorch backends.

An active `noise` activation intervention uses a separate stochastic operation. The seeded `max` path therefore refuses to run while that intervention is active. Disable the noise intervention or use a single-candidate mode. Deterministic activation interventions remain available.

The inference artifact separates generated-token accounting into `reasoning_compute_tokens`, `answer_compute_tokens`, and `selector_compute_tokens`. The first value includes every token sampled during every candidate reasoning phase, including a naturally generated reasoning boundary or end token. The second includes final-answer tokens from all candidates. The third includes tokens decoded by the private selector. Their sum is the generated-token count for the search. It does not include prompt tokens or a boundary inserted by code when the model did not generate one.

Only the selected final answer may reach the ordinary return value or cognitive memory. Losing answers, losing scratchpads, and selector inputs remain ephemeral. They are absent from the default token trace, inference JSON, and memory store. If trace exposure or persistence is explicitly enabled, only the selected scratchpad is eligible for that private local output.

By default, the scratchpad is absent from the returned answer, token trace, cognitive memory, and inference JSON. Set `expose_reasoning_trace: true` only for a deliberate local inspection. Set `save_reasoning_trace: true` only when raw private traces may be written under the ignored run-log directory. A saved trace may contain sensitive or incorrect text and must never be published automatically.

A private SFT message may teach the protocol with optional fields:

```json
{
  "role": "assistant",
  "reasoning_mode": "max",
  "reasoning": "<private intermediate target>",
  "content": "<private final target>"
}
```

Instruction-style sources may map different local column names through `reasoning_field` and `reasoning_mode_field`. The loader escapes control-token injection in both the scratchpad and final answer. Ordinary answer-only messages continue to use the normal assistant format.

Keep all reasoning targets private. Never use benchmark questions, official explanations, answer keys, or traces generated from held-out benchmark items as reasoning SFT data.

The default reasoning instruction is English and neutral. For natural Korean, Japanese, or another language, put a language-appropriate instruction in an ignored local text file and set `reasoning.scratchpad_instruction_file` in `config.local.yaml`. The path is removed from checkpoint-safe configuration artifacts.

Evaluation reports should compare `off`, `high`, and `max` on the same frozen prompts and seeds. Useful aggregate measurements include final-answer score, parse rate, majority and selector rates, selected-versus-first accuracy, total generated tokens, wall time, throughput, peak memory, safety failures, and visible trace leakage. Raw prompts, candidates, and scratchpads remain private.

Hidden reasoning is an application-output boundary, not a mathematical guarantee that a model can never paraphrase part of its scratchpad in a final answer. Safety evaluation must test that behavior directly.

## Private ten-item knowledge pilot

The initial measurable target is ten correct answers on one frozen, private ten-item pilot. This is a narrow integration and development target. A result of 10/10 on ten questions is not evidence of broad benchmark capability because the sample is too small.

The public repository contains no questions or answers. Prepare one ignored local JSONL file with exactly these fields on every row:

```json
{
  "id": "<private stable id>",
  "question": "<private question>",
  "choices": {
    "A": "<private choice>",
    "B": "<private choice>",
    "C": "<private choice>",
    "D": "<private choice>"
  },
  "answer": "<private correct label>"
}
```

Enable it only in the ignored `config.local.yaml`:

```yaml
schema_version: 2

eval:
  knowledge_pilot:
    enabled: true
    file: "./train_data/eval/private-pilot.local.jsonl"
    prompt_file: "./train_data/eval/prompt.local.txt"
    item_count: 10
    required_correct: 10
    choice_labels: ["A", "B", "C", "D"]
    reasoning_mode: "max"
    max_new_tokens: 8
    require_denylist_coverage: true
```

`prompt_file` is optional. When it is present, it must use the placeholders `{labels}`, `{question}`, and `{choices}`. This lets a Korean pilot use a natural Korean instruction and a Japanese pilot use a natural Japanese instruction without placing non-English private prompts in the public repository.

Run the safe sequence:

```powershell
hana data quarantine-eval --config config.yaml
hana data lock --config config.yaml
hana data audit --config config.yaml
hana run --config config.yaml --mode eval --force
```

The quarantine command adds canonical hashes for each question and full rendered prompt to the ignored benchmark denylist. Evaluation refuses to load the model when required hashes are missing. Changing the pilot file, its prompt template, the denylist, the checkpoint, or reasoning settings invalidates the cached result.

Direct pilot decoding uses temperature zero, top-p one, no top-k sampling, and a repetition penalty of one. When the pilot selects `max`, the inner candidates instead use the stable, locally seeded maximum-effort sampling policy described above; the selector remains deterministic. Cognitive memory, activation experiments, token traces, and reasoning-trace persistence stay disabled. The knowledge pilot contributes only item count, correct count, accuracy, parse rate, and pass/fail status to the normal evaluation result. It contributes no question, choice, answer key, model output, candidate, selector input, or scratchpad.

Official benchmark material may be evaluated only when its license permits that use. It must remain outside tokenizer, pretraining, SFT, DPO, replay, memory, and retrieval inputs. Improving this pilot must come from reviewed non-benchmark knowledge and reasoning data, not from memorizing the ten held-out items.

## Reusable data-pack boundary

The model and trainer are language-neutral. Identity and data are separate:

- `profile` provides a display identity and memory speaker label.
- `data.pack` describes a private dataset collection.
- `data.sources` describes private inputs and their policy evidence.
- translation fields describe arbitrary language pairs explicitly.
- deterministic `sample_rate` provides basic source downsampling.

Training a fresh tokenizer for another private pack is supported. Expanding the vocabulary of an existing checkpoint is not currently supported because checkpoint tensor loading is strict. Continued language expansion therefore needs either a stable broad tokenizer or a separately tested token-ID and embedding migration.

### Tokenizer numeric integrity

`split_digits: true` is a tokenizer-training decision. It asks SentencePiece to keep each decimal digit as its own piece, so an accidental vocabulary chunk such as `2026` cannot be the only learned representation for that number. It is not an inference-time arithmetic solver and does not guarantee mathematically correct answers.

Tokenizer training and corpus finalization use the same trainer settings and the same validation gate. Before a candidate tokenizer replaces the current local tokenizer, it must satisfy all of these checks:

1. code-owned probes cover integers, leading zeroes, signs, decimals, exponents, percentages, dates, times, currency, radix prefixes, long identifiers, and numbers inside Korean and Japanese text
2. every probe decodes to its explicitly expected canonical form after configured Unicode normalization
3. no ordinary, non-byte vocabulary piece contains more than one Unicode decimal digit
4. no probe produces the unknown-token ID while byte fallback is enabled
5. a bounded deterministic sample from the private tokenizer corpus produces no unknown-token ID and a non-empty round trip

Logs and validation artifacts contain only aggregate counts, model and probe-suite digests, and pass/fail status. They never print a private corpus row. The tokenizer model SHA-256 is included in cache and pipeline freshness decisions, so replacing a model while preserving its size and timestamp cannot reuse stale token shards.

Changing digit splitting, normalization, vocabulary, special tokens, or tokenizer corpus requires a freshly trained tokenizer. Existing checkpoints cannot safely reuse changed token IDs or embedding rows. A different grouping policy, such as one-to-three-digit pieces, is a separate tokenizer experiment and must use fresh model training under a `data_portability` contract.

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

## Research influences

The supplied reports are used as design references, not as proof that Hana inherits their results.

- [Kimi K3](https://arxiv.org/abs/2607.24653) provides user-facing reasoning-effort levels and reports a maximum-effort reasoning setup using temperature 1.0 and top-p 0.95. It also studies hybrid sequence mixing and attention across depth. The reasoning settings inform Hana's `max` interface; the larger KDA, MLA, and depth-routing structures remain future experiments.
- [Solar Open 2](https://arxiv.org/abs/2607.20062) combines gated full attention with linear-attention layers, uses a byte-level tokenizer with digit splitting, and emphasizes verifier-first construction for agent tasks. Hana's small gate and layer schedule are bounded studies, not a reproduction of its recurrent attention.
- [K-EXAONE 2.0](https://arxiv.org/abs/2608.04505) retains QK normalization and a repeating global/sliding attention layout. It also filters redundant self-reflection and disproportionately long reasoning trajectories. These choices inform Hana's normalized baseline, optional layer schedule, and treatment of reasoning budgets as compute limits rather than quality ranks.
- [Motif 3](https://arxiv.org/abs/2608.09119) applies query-dependent attention output gating and uses bounded numeric grouping. Its attention and one-to-three-digit tokenizer rules remain separate research arms; Hana currently keeps ordinary GQA and single-digit splitting.
- [A.X K2](https://github.com/SKT-AI/A.X-K2/blob/main/A_X_K2_Tech_Report.pdf) combines QK normalization with head-specific gated attention and trains one model to support thinking and non-thinking control modes. Hana uses the first as a baseline and the gate as an opt-in ablation, while evaluating safety separately at every effort level.

## Further reading

- [Capability and scale roadmap](docs/ROADMAP.md)
- [Experiment contracts](docs/EXPERIMENT_CONTRACT.md)
- [Internal-dynamics research lab](docs/RESEARCH_LAB.md)
- [Cognitive architecture](docs/COGNITIVE_ARCHITECTURE.md)

No Markdown file grants permission to train. Training authorization depends on the operator's licenses and provenance review. Reproducibility evidence includes the private source lock, benchmark denylist, complete audit, and tokenizer-corpus fingerprint.
