# Experiment Contracts

This document explains the first experiment-contract layer in Hana.

The layer is small on purpose. It validates a research plan before model runs are compared. It does not start training, change a model, run a benchmark, or claim that an idea is new.

## Why an experiment contract is useful

A model feature can appear helpful for accidental reasons. For example:

- the candidate may use more training tokens than the baseline
- the candidate may use a different random seed
- the candidate may use a different dataset
- the candidate may run on faster hardware
- a result may be selected only because it looked good
- an extra configuration change may be forgotten

An experiment contract makes these choices visible before the result is known.

Each study declares:

1. one question, called the hypothesis
2. one baseline arm
3. one or more candidate arms
4. the settings each candidate is allowed to change
5. the same random seeds for both arms
6. one shared budget
7. one primary metric
8. any guardrail metrics
9. the evidence needed before a decision is complete

The contract helps prevent an unfair comparison. It cannot prove that a result is correct, important, or original.

## What the current layer can do

The module is `llm_pipeline.experiment_contract`.

It can:

- read a UTF-8 YAML registry
- require registry schema version 1
- reject unknown fields so spelling mistakes do not pass silently
- validate study, arm, seed, budget, and metric rules
- compare two Python configuration mappings
- verify that declared changed keys exactly match the real configuration difference
- check that baseline and candidate results exist for every required seed
- return one conservative decision

The four possible decisions are:

- `promote`
- `reject`
- `inconclusive`
- `incomplete`

The `hana experiment validate` command validates a registry and can optionally verify declared configuration differences. Training orchestration and result-row decisions are not connected to the command yet; callers use the public functions for those steps.

## The three study kinds

### `architecture_ablation`

Use this kind when testing a model or training feature.

Examples:

- workspace off versus workspace on
- GQA head layout A versus layout B
- predictive loss off versus predictive loss on

The dataset, tokenizer, seed set, and budget should stay the same. The candidate should change only the declared feature settings.

### `data_portability`

Use this kind when testing the same model and pipeline with different data.

Examples:

- Korean data versus Japanese data
- one language pack versus another language pack
- general text versus code data

The model structure and training recipe should stay the same. Data and tokenizer settings may change when the contract declares those changes.

Perplexity from different tokenizers is not directly comparable. A portability study should prefer metrics such as held-out task accuracy, language consistency, bytes per token, and characters per token.

### `systems_speed`

Use this kind when testing an execution change.

Examples:

- one attention backend versus another backend
- compile off versus compile on
- one packing implementation versus another implementation

The model weights, input shapes, data, precision, and hardware should stay fixed unless the purpose of the study explicitly requires one of them to change.

## Claim status

Every study has one claim status. Schema version 1 accepts only these values:

- `unverified_hypothesis`: an idea that has not been established yet
- `replication`: an attempt to reproduce an existing result
- `prior_art_extension`: a measured extension of an existing method
- `negative_result`: a completed study that did not support its hypothesis

There is no `novel` or `first` status.

An experiment result alone cannot show that nobody else has tried an idea. A novelty claim needs a separate human-reviewed search of papers, code, patents, and other prior work.

## Registry example

The repository includes a complete example at `configs/experiments/registry.example.yaml`.

A shortened study looks like this:

```yaml
schema_version: 1

studies:
  - id: "workspace_v1"
    kind: "architecture_ablation"
    hypothesis: "A small causal workspace may improve validation loss at the same token budget."
    claim_status: "unverified_hypothesis"
    baseline_arm: "dense_control"

    controls:
      seeds: [42, 43, 44]
      budget:
        basis: "train_tokens"
        value: 100000000

    metrics:
      primary:
        name: "valid_loss"
        direction: "min"
        minimum_improvement: 0.02
        maximum_regression: 0.01
      required:
        - "valid_loss"
        - "token_accuracy"
      guardrails:
        token_accuracy:
          direction: "max"
          maximum_regression: 0.01

    arms:
      - id: "dense_control"
        config: "workspace_control.yaml"
        changed_keys: []
      - id: "workspace_on"
        config: "workspace_on.yaml"
        changed_keys:
          - "cognitive_architecture.enabled"
          - "cognitive_architecture.workspace.enabled"
```

## Field reference

### Registry fields

`schema_version`

: Must be the integer `1`.

`studies`

: A non-empty list of studies. Every study ID must be unique.

### Study fields

`id`

: A stable, lowercase name. It must start with a letter. It may also contain digits, `_`, and `-`.

`kind`

: One of the three study kinds described above.

`hypothesis`

: A non-empty sentence that says what may change and under which fair comparison.

`claim_status`

: One of the four cautious evidence labels described above.

`baseline_arm`

: The ID of exactly one arm in the same study.

`controls`

: The shared seed list and shared budget.

`metrics`

: The primary metric, required metric names, and guardrail rules.

`arms`

: At least two arms: one referenced baseline and at least one candidate.

### Control fields

`seeds`

: A non-empty list of unique, non-negative integers. The baseline and candidate need one result for every seed.

`budget.basis`

: One of:

  - `train_tokens`
  - `optimizer_steps`
  - `wall_clock_seconds`
  - `evaluation_examples`

`budget.value`

: A positive number. Token, step, and example budgets must be whole numbers.

The current module validates the declared budget. It does not yet read a training log and prove that the run obeyed that budget. That connection belongs to a later pipeline integration.

### Metric fields

`primary.name`

: The metric used to decide whether a candidate may be promoted.

`primary.direction`

: Use `min` when smaller is better, such as validation loss. Use `max` when larger is better, such as accuracy or throughput.

`primary.minimum_improvement`

: The minimum absolute improvement required for `promote`. It must be greater than zero. An unchanged result therefore cannot be promoted.

`primary.maximum_regression`

: The largest absolute primary-metric regression that is tolerated before `reject`.

`required`

: Every listed metric must be present for every required baseline and candidate run. The list must include the primary metric and every guardrail metric.

`guardrails`

: Optional safety checks for other metrics. Each guardrail declares whether smaller or larger is better and how much absolute regression is allowed.

All thresholds use the metric's own units. For example, an accuracy change from `0.72` to `0.71` is an absolute regression of `0.01`.

### Arm fields

`id`

: A unique, stable arm name inside the study.

`config`

: A non-empty config reference. Schema version 1 records the reference but does not open it automatically.

`changed_keys`

: Dot-separated configuration paths that this arm changes compared with the baseline.

The baseline must use an empty list. Every candidate must declare at least one changed key.

## Loading a registry

Validate only the registry structure from PowerShell:

```powershell
hana experiment validate --registry configs/experiments/registry.example.yaml
```

This checks the study IDs, baseline, arms, budgets, seeds, metrics, thresholds, and claim status. It does not start training.

After the arm config files exist, also verify that their real differences exactly match each arm's `changed_keys` list:

```powershell
hana experiment validate `
  --registry configs/experiments/registry.example.yaml `
  --check-configs `
  --ignore-key run.experiment_name `
  --ignore-key run.output_dir `
  --ignore-key logging.log_dir `
  --ignore-key experiments.output_dir
```

Every ignored key is explicit in the command. This makes operational exceptions visible during review.

The same validator is available as a Python API:

```python
from llm_pipeline.experiment_contract import load_registry

registry = load_registry("configs/experiments/registry.example.yaml")
study = registry.study("workspace_v1")
candidate = study.arm("workspace_on")
```

The returned dataclasses are frozen. This prevents later code from silently changing the contract in memory.

## Verifying exact configuration changes

First load the fully resolved baseline and candidate configurations as Python mappings. Then verify the candidate:

```python
from llm_pipeline.experiment_contract import verify_arm_config_diff

actual_paths = verify_arm_config_diff(
    study,
    "workspace_on",
    baseline_config,
    candidate_config,
)
```

The function raises an error in either case:

- a setting changed but was not declared
- a setting was declared but did not actually change

This catches accidental changes such as a different learning rate, context length, or data source.

Lists are compared as one value. A changed source list is reported as `data.sources`, not as many unstable list-index paths.

### Ignoring operational paths

Two runs may need different output folders. A caller may ignore those paths explicitly:

```python
actual_paths = verify_arm_config_diff(
    study,
    "workspace_on",
    baseline_config,
    candidate_config,
    ignored_keys=[
        "run.experiment_name",
        "run.output_dir",
        "logging.log_dir",
        "experiments.output_dir",
    ],
)
```

Ignored paths are not hidden automatically. The caller must list them so the exception is visible and reviewable.

## Evaluating result completeness

Result rows have a small common shape:

```python
results = [
    {
        "arm_id": "dense_control",
        "seed": 42,
        "metrics": {
            "valid_loss": 1.00,
            "token_accuracy": 0.70,
        },
    },
    {
        "arm_id": "workspace_on",
        "seed": 42,
        "metrics": {
            "valid_loss": 0.96,
            "token_accuracy": 0.71,
        },
    },
]
```

Evaluate one candidate against the baseline:

```python
from llm_pipeline.experiment_contract import evaluate_candidate_results

decision = evaluate_candidate_results(study, "workspace_on", results)
```

The function averages each required metric across the declared seeds.

It returns:

- `incomplete` when a required run or metric is missing, duplicated, non-numeric, or non-finite
- `reject` when the primary metric or a guardrail regresses too far
- `promote` when the primary improvement reaches its threshold and every guardrail passes
- `inconclusive` when the evidence is complete but the improvement is too small to promote and the regression is too small to reject

An unknown candidate ID and an attempt to compare the baseline with itself are contract errors rather than incomplete evidence.

## Benchmark isolation

Benchmark questions and answers do not belong in the training source manifest.

An experiment registry should record only the evaluation name, rules, fingerprints, and result metrics needed for comparison. Evaluation files should remain in a separate evaluation-only location. Existing data-governance checks and the benchmark denylist provide additional training-data safeguards.

The contract does not weaken those protections. It also does not automatically inspect benchmark files in schema version 1.

## What this first version does not do

Schema version 1 does not:

- start experiment runs
- load arm config files automatically
- enforce the budget from runtime logs
- measure speed or memory
- verify that two runs used the same hardware
- calculate confidence intervals
- require three or more seeds
- inspect benchmark contamination
- search for prior work
- claim novelty

These limits are intentional. The first version provides a reliable, testable vocabulary before automation is added.

## Recommended next steps

1. Write a resolved runtime artifact with actual dtype, backend, batch size, world size, and parameter count.
2. Add synchronized training and inference timing.
3. Check that measured tokens or steps match the declared budget.
4. Store a contract digest in each run manifest.
5. Aggregate several seeds and report raw values, means, and uncertainty.
6. Require the baseline results before a candidate can be summarized.
7. Add a portability matrix for new languages and data-only model variants.

Until those steps are complete, use this module as a strict pre-run plan validator and a conservative post-run completeness check.
