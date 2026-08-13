# Hana Internal-Dynamics Research Lab

Hana is more than a basic decoder-only training script. The same model can run as:

- an autoregressive next-token predictor
- a masked-block restoration model
- an observable dynamical system with bounded interventions

The project records how these modes differ. This document explains the available measurements, interventions, and experiments in direct language.

An implemented mechanism is not automatically faster, better, or novel. Every comparison that may influence the main model must first use the [experiment contract](EXPERIMENT_CONTRACT.md). The contract fixes the baseline, exact changed settings, seeds, shared budget, primary metric, required metrics, and regression guardrails before results are known.

## Why the research layer was added

Earlier long-context experiment flags were mostly safe switches with research names. The code could not inspect or modify internal activations in a controlled way. Multi-token prediction was only an auxiliary future-token head and was not a diffusion model. A training hyperparameter change usually required restarting from another configuration. Token uncertainty, layer activation size, and gradient-to-update ratios could not be compared on one experiment timeline.

The project had experiment names, but it did not yet have a practical experiment surface. The current implementation adds that surface.

## Measurements and interventions

| Surface | What it does | Output or API |
|---|---|---|
| Activations | Records mean, standard deviation, RMS, absolute maximum, zero ratio, and optional raw samples for selected module patterns. | `activations.jsonl` |
| Gradients | Records parameter norm, gradient norm, gradient mean, gradient standard deviation, and estimated update-to-weight ratio. | `gradients.jsonl` |
| Hidden states | Returns the post-embedding state and selected layer outputs when explicitly requested. | `return_hidden_states=True` |
| Internal interventions | Supports zeroing, scaling, clamping, noise, vector addition, and projection removal over selected step and token ranges. | `experiments.interventions` |
| Runtime changes | Applies a shape-safe allowlist of changes at optimizer-step boundaries and records before-and-after events. | `metrics.jsonl` |
| Token dynamics | Records autoregressive entropy and top-five candidates, plus masked-block token identifiers, pieces, confidence, and entropy for each denoising step. | `token_trace_file` |
| Objective labels | Marks hook records as `ar_train`, `diffusion_train`, `ar_validation`, or an inference phase. | `phase` in activation records |

Raw activation samples default to zero and are limited to 256 values. Dumping complete activation tensors can consume disk and GPU memory very quickly. Request full hidden states only when an experiment truly needs them.

## Meaning of autoregressive and masked-block training

The combined loss is:

```text
L_total = L_next_token + lambda * L_masked_same_token
```

The two paths work as follows:

1. The autoregressive path predicts token `t + 1` from information available at token `t`.
2. The masked-block path selects one block in each sample.
3. It replaces some tokens in that block with `<mask>`.
4. It predicts the original token at the same position.
5. The clean prefix keeps causal attention.
6. Tokens inside the noisy block can see both directions inside that block.
7. A later block may see an earlier block, but an earlier block may not see a later block.
8. Packed documents keep their boundaries isolated.
9. Block starts are aligned with document starts where required.

Generation uses the same structure. It first creates `ar_warmup_tokens` causally. It then appends a masked block and commits higher-confidence positions across `denoise_steps`. The next block starts only after the current block is complete. Document order stays causal while work inside one block is partially parallel.

This implementation combines the block-autoregressive idea from [Block Diffusion](https://arxiv.org/abs/2503.09573) with same-position denoising ideas from [generalized masked diffusion](https://arxiv.org/abs/2406.04329) and [masked diffusion language models](https://arxiv.org/abs/2406.07524).

## Why the project does not copy Golden Gate or J-space directly

[Anthropic's feature-steering work](https://www.anthropic.com/research/evaluating-feature-steering) adds constants along residual-stream directions found with dictionary learning. The study found useful operating points and unwanted off-target effects.

Hana does not assume that one feature dictionary is ground truth. The code supports additive, multiplicative, projection, ablation, and noise interventions so a contracted study can compare them on the same evaluation surface. It does not run or interpret that comparison automatically.

[J-space research](https://www.anthropic.com/research/global-workspace) studies reportability, controllability, and causal mediation for small internal patterns found with Jacobian analysis. Hana begins with a more basic question: does an important direction keep its causal effect when the objective or generation algorithm changes?

The project measures dynamics and counterfactual effects before giving an internal space a strong name.

## Candidate studies supported by current primitives

The following ideas are starting points, not established improvements. Register each comparison as an `architecture_ablation`, `data_portability`, or `systems_speed` study. Candidates begin as `unverified_hypothesis`, `replication`, or `prior_art_extension`; a completed unsupported hypothesis may become `negative_result`. The registry deliberately has no `novel` or `first` status. Any novelty discussion belongs in separately reviewed research prose.

The runtime can collect the relevant traces and apply the interventions. Plotting, direction discovery, sweep orchestration, statistical comparison, and probe training still require external analysis code unless a section says otherwise.

### 1. Objective shock and hysteresis

Change `hybrid_diffusion.loss_weight` from `0` to `0.8` and back to `0` during one run. Compare autoregressive and masked-block RMS, gradient-update ratio, and validation loss. Failure to return to the original path may indicate history dependence in optimizer state or representation space.

### 2. Token commitment front

Plot entropy and confidence by position for every denoising step in `token_trace_file`. Compare punctuation, grammar tokens, content tokens, and mixed Korean-Japanese tokens. Measure how committing one position changes later probabilities.

### 3. Intervention transport

Find an activation direction with a large effect during an autoregressive phase. Add or project out the same direction at the same layer during masked-block processing. A preserved effect suggests a shared circuit candidate. A reversed effect suggests different computational roles.

### 4. Steering hysteresis

Apply a scale or vector-add intervention only during a selected step range. Remove it and measure activation and loss recovery time. This tests whether a temporary training intervention leaves a lasting parameter effect.

### 5. Block-size phase diagram

Run a grid over `block_size`, `mask_probability`, and `denoise_steps`. Compare perplexity, masked-block accuracy, tokens per second, and entropy reduction. Find where autoregressive stability trades against block parallelism.

The ordinary training throughput counter is not a synchronized global performance benchmark. A strict speed claim also needs a controlled harness that fixes hardware, precision, input shapes, warmup, compilation state, and synchronization.

### 6. Silent-state falsification

Train a small probe on layer hidden states to separate the next spoken token, the token currently being restored, and an intermediate concept that will not be spoken. Project out the discovered direction and measure the performance change. A correlation without an ablation effect is not enough to call the direction a causal workspace candidate.

The current code provides hidden-state collection and intervention. A complete probe-training utility remains future work.

## Example configuration

```yaml
hybrid_diffusion:
  enabled: true
  loss_weight: 0.25
  mask_probability: 0.3
  block_size: 16
  denoise_steps: 4
  ar_warmup_tokens: 4

experiments:
  enabled: true
  output_dir: "./experiments/run_a"
  activation_monitor:
    enabled: true
    modules: ["layers.2", "layers.5"]
    every_n_calls: 1
    max_records: 10000
    sample_values: 16
    output_file: "activations.jsonl"
  gradient_monitor:
    enabled: true
    parameters: ["layers.*.attn.*", "layers.*.ffn.*"]
    every_n_steps: 10
    max_records: 10000
    output_file: "gradients.jsonl"
  interventions:
    - module: "layers.2"
      kind: "scale"
      value: 0.7
      start_step: 500
      end_step: 700
      token_positions: [-1]
  runtime_patches:
    - at_step: 1000
      changes:
        train.learning_rate: 0.0001
        hybrid_diffusion.loss_weight: 0.8

inference:
  generation_strategy: "hybrid"
  token_trace_file: "./experiments/run_a/token_trace.jsonl"
```

## Runtime-patch limits

The runtime patch allowlist includes:

- `train.learning_rate`
- `train.min_learning_rate`
- `train.label_smoothing`
- `train.max_grad_norm`
- the three configured dropout values
- `model.logit_softcap`
- masked-block loss weight
- masked-block mask probability

Values that change tensor shape or optimizer-state layout are rejected. Examples include layer count, hidden size, attention-head count, and vocabulary size.

## Verification

First validate the comparison plan:

```powershell
hana experiment validate --registry configs/experiments/registry.example.yaml
```

Add `--check-configs` after replacing the example config paths with real baseline and candidate files. That check fails if the actual configuration differences do not exactly match the declared `changed_keys`.

Then run the repository gate:

```powershell
hana verify
```

Focused tests check:

- no future-token leakage in block attention
- correct same-position label alignment
- preservation of tuple outputs in hooks
- runtime-patch boundaries
- complete resolution of masks by the end of generation
- stable activation and gradient logging

The smoke configuration performs a small synthetic integration run with joint autoregressive and masked-block training, activation and gradient records, a step-one runtime patch, hybrid inference, and a token trace.
