# Hana Capability and Scale Roadmap

This roadmap turns a long-term idea into small, measurable gates. It is a plan, not a promise. A later stage begins only after the previous stage has reproducible evidence.

## The identity stays the same

Hana's conversational identity remains a fictional nine-year-old Korean-Japanese girl with a Korean father and a Japanese mother.

Capability and identity are different things:

- "high-school senior level" means the model can solve more advanced learning tasks
- it does not mean Hana suddenly becomes eighteen years old
- Korean data remains natural Korean
- Japanese data remains natural Japanese
- a new language uses natural data and instructions for that language
- the reusable training core does not hard-code Hana's name or one language pair

The repository describes the software honestly. Inside persona conversations, Hana speaks consistently as the fictional human character instead of describing herself as a model or a bilingual helper.

## Published architecture baseline

The public configuration defines architecture only:

| Item | Current value |
|---|---:|
| Exact model parameters | 303,350,784 |
| Decoder layers | 24 |
| Hidden size | 1,024 |
| Attention heads | 16 |
| Key-value heads | 4 |
| Vocabulary size | 32,000 |
| Maximum training sequence | 2,048 tokens |

Private corpus composition, local source sizes, row counts, exact accepted-token counts, and persona-data counts are not published in this code repository. Each private run records them in local evidence. Changing the tokenizer changes token counts, so every private report must name its tokenizer fingerprint.

## How token budgets are counted

Every stage below uses **cumulative accepted unique training tokens**.

This means:

1. rejected rows do not count
2. evaluation-only rows never count
3. duplicates removed by the pipeline do not count twice
4. repeating the same epoch does not pretend to create new data
5. SFT and DPO tokens are reported separately from pretraining tokens
6. synthetic data counts only when its report supplies and verifies the model, prompt, filter, seed, and human-review fingerprints that created it

The ranges are planning ranges. They are not instructions to pad a corpus with weak text. A smaller clean dataset is better than a larger contaminated dataset.

## Staged gates

| Gate | Candidate active parameters | Cumulative accepted pretraining tokens | Main capability gate |
|---|---:|---:|---|
| Current architecture | 303M | Private and unpublished | Establish clean, reproducible local measurements |
| 1A | 303M | 8.5B-10B | Stable Hana identity; at least 30% on private JLPT-style and TOPIK-style suites |
| 1B | 303M-420M | 12B-16B | At least 50% on both private language suites without identity regression |
| 2A | 500M-650M | 20B-28B | Strong elementary and lower-secondary knowledge in Korean and Japanese |
| 2B | 750M-1.0B | 32B-45B | High-school-senior knowledge coverage while the persona remains nine |
| 3 | 1.0B-1.3B | 50B-70B | Reliable school mathematics, Python, algorithms, debugging, and code tests |
| 4 | 1.3B-1.8B | 75B-100B | At least 50/100 on a private Korean CSAT-style language suite |
| 5A | 1.8B-2.5B | 110B-160B | At least 60/100 CSAT-style and 30% on a private KMMLU-style suite |
| 5B | 2.5B-3.0B | 170B-230B | At least 70/100 CSAT-style and 40% KMMLU-style accuracy |
| 6A | about 3.0B | 240B-320B | Add reviewed English data; old Korean/Japanese suites regress by no more than 2 percentage points |
| 6B | 4.0B-4.5B | 350B-480B | Stronger English reasoning and at least 30% on a private MMLU-style suite |
| 7 | 6.0B-7.0B | 550B-750B | At least 40% MMLU-style while retaining the Korean, Japanese, CSAT-style, and KMMLU-style gates |

Parameter growth is not automatic. First improve data quality, tokenizer efficiency, training stability, and measured architecture changes. Increase the model only when the smaller candidate cannot pass the next gate under a fair compute budget.

## Gate details

### Gate 1A: the first useful Hana

The first model should feel like one consistent person. It should not switch between being a child, an assistant, a translation service, and a generic model.

Required checks include:

- Korean identity questions receive natural Korean answers
- Japanese identity questions receive natural Japanese answers
- family facts remain consistent across paraphrases
- Hana does not repeat her age and family background in every answer
- safety refusals remain short, kind, and clear
- private JLPT-style item accuracy reaches 30%
- private TOPIK-style item accuracy reaches 30%

The percentages are raw accuracy on private development suites. They are not official JLPT or TOPIK scaled scores and are not described as an official pass.

### Gate 1B: language foundation

This gate targets at least 50% on both private language suites. The model must improve both languages instead of sacrificing one for the other.

Every report includes:

- Korean score
- Japanese score
- mixed-language score
- tokenizer characters per token and UTF-8 bytes per token for each declared language
- identity and safety regression results

### Gates 2A and 2B: broader knowledge

These gates grow in two steps so the jump is measurable.

Gate 2A focuses on foundational science, social studies, reading, and everyday reasoning. Gate 2B adds high-school-senior subject coverage and more specialized explanations.

Hana still speaks clearly and warmly. More knowledge must not turn every answer into a lecture.

### Gate 3: mathematics and coding

This gate adds independently generated and reviewed tasks for:

- arithmetic and word problems
- algebra, geometry, probability, and introductory calculus
- Python syntax and standard-library use
- small algorithms and data structures
- code explanation
- debugging from failing tests
- writing a small function that passes hidden unit tests

Gate 3 requires a future restricted code evaluator. That evaluator must execute generated code against hidden tests. After it exists, a plausible-looking answer will not count as correct when it fails those tests.

### Gate 4: Korean language reasoning at 50 points

The development target is at least 50/100 on a private, non-overlapping Korean CSAT-style suite. The suite may imitate task skills, but it must not copy official passages, questions, choices, answer keys, or explanations.

The report separates reading, literature, language and media, and time-limited performance. A single total score is not enough to diagnose a failure.

### Gates 5A and 5B: Korean depth and broad knowledge

Gate 5A is an intermediate step. Gate 5B is the requested 70/100 Korean CSAT-style target and begins serious KMMLU-style measurement.

The model must retain:

- Korean and Japanese language scores
- persona consistency
- safety behavior
- math and code capability
- inference speed and memory limits

### Gates 6A and 6B: add English without erasing old skills

English is added through a new reviewed data-pack version. It is not mixed into an old folder without metadata.

The first English gate allows no more than a two-percentage-point drop on established Korean and Japanese suites. If the drop is larger, the stage fails even when English improves.

Useful countermeasures include balanced rehearsal data, language-aware batch mixing, tokenizer-efficiency checks, and smaller learning rates for continued pretraining. Each countermeasure is an experiment, not an assumed solution.

### Gate 7: MMLU-style challenge

The final listed gate targets broad English knowledge while retaining all earlier languages and skills. A higher MMLU-style result does not compensate for a collapse in Korean, Japanese, safety, or persona behavior.

## Benchmarks are evaluation-only

Official or third-party benchmark content must never be used as training material.

The exclusion covers:

- questions
- passages and contexts
- choices
- answer keys
- rationales and explanations
- translated versions
- reformatted chat versions
- synthetic prompts seeded with benchmark content
- tokenizer corpora
- pretraining, SFT, and DPO files
- replay buffers, persistent memory, and retrieval corpora

Development should use private style-equivalent suites created independently of official items. Official benchmark data may be used only in an isolated evaluation run when its license allows that use.

An evaluation source must declare:

```yaml
purpose: "evaluation"
stages: ["eval"]
split: "test"
tokenizer: false
```

The pipeline rejects an evaluation source from tokenizer, pretraining, SFT, and DPO selection even when normal license-policy enforcement is disabled. It also rejects a path or byte-identical file shared by training and evaluation.

This declaration is quarantine metadata. The current generic evaluation stage does not automatically execute a `purpose: evaluation` source. An official suite therefore needs a separate, isolated evaluator that follows its license. That evaluator must never add its content to `data.sources` used for training.

The private benchmark denylist stores exact normalized component hashes. Matching collapses whitespace and applies Unicode-aware case folding before SHA-256. It does not find paraphrases, substrings, or semantically similar text. The filter checks raw message, translation, prompt, chosen, and rejected fields, but every passage, question, choice, answer, rationale, translation, and reformatted variant still needs its own hash. Human contamination review remains required.

## Reusable language packs

The model and trainer are language-neutral. A pack describes the data.

Each pack declares:

- a stable name
- a positive version
- its language tags
- a plain-English description
- each source's purpose, language, domain, provenance, split, and sampling rule
- explicit source and target fields for parallel data
- a prompt template written naturally for the intended task language

Language tags are metadata, not automatic language detection. The current loader records them but does not prove that a row is written in the declared language or that every source tag belongs to the pack. Pack review must verify both facts.

The `profile` configuration provides an identifier, display name, and memory speaker label. Private prompts and private data define conversational behavior. The `data.pack` configuration describes the dataset identity. These boundaries let Hana learn another language without changing her name, and they let the same architecture be tested with another private profile and dataset.

`sample_rate` provides deterministic downsampling for an oversized source. It is a first balancing tool, not a complete language-aware batch scheduler.

Adding a language to an existing checkpoint is different from training a fresh model on another pack. Byte fallback prevents unknown characters, but the current code does not resize an existing vocabulary or migrate embedding rows. A future vocabulary-expansion feature must preserve token IDs and prove checkpoint compatibility before it is enabled.

## Experimental speed and architecture track

The project should test improved or unusual structures, but it must not call an idea new merely because it is new to this repository.

Candidate areas include:

- attention layouts and key-value sharing
- feed-forward routing and sparsity
- recurrent or compressed memory
- tokenizer and byte-level efficiency
- masked-block and multi-token objectives
- optimizer and curriculum changes
- quantization-aware designs
- compilation, kernels, batching, and cache layouts
- combinations that reduce memory traffic or serial decoding work

Every candidate begins as `unverified_hypothesis`, `replication`, or `prior_art_extension`. A separate human-reviewed literature search is required before any novelty claim.

Use one of three contract kinds:

- `architecture_ablation`: keep data and tokenizer fixed; change declared model settings
- `data_portability`: keep architecture and training recipe fixed; change the declared data pack or tokenizer
- `systems_speed`: keep weights, inputs, shapes, precision, and hardware fixed; change the runtime system

The experiment contract requires one baseline, exact changed keys, shared seeds, a shared budget, required metrics, and guardrails.

Recommended promotion rules are:

- use one seed only for an inexpensive early screen
- use at least three seeds before a serious promotion decision
- an architecture candidate should improve its primary quality metric and keep every established language within two percentage points
- a speed candidate should improve measured throughput or memory by at least 10% and lose no more than 0.5 percentage points on quality checks
- record warmup, hardware, dtype, attention backend, batch shape, context length, and peak allocated and reserved memory
- report negative and inconclusive results instead of hiding them

The current training throughput counter is useful for ordinary progress logging, but it is not yet a synchronized global performance benchmark. Strict speed claims wait for the planned synchronized performance harness.

## When a stage is complete

A stage is complete only when all of these are true:

1. the source lock and full audit are current
2. the required benchmark denylist is present and unchanged
3. the tokenizer corpus has verified provenance
4. official benchmark content is absent from every training path
5. every required private evaluation has enough examples and a frozen fingerprint
6. the target holds across the required seeds
7. earlier language, identity, safety, math, and code gates remain within their guardrails
8. model size, accepted tokens, compute, hardware, and runtime metrics are reported
9. limitations and failed subgroups are reported

Passing one attractive score is not enough.
