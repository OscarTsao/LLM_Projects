# Research: Thresholding Strategies for Dual Agents

## Questions

- Does per-class thresholding outperform a single global threshold for criteria matching?
- What ranges and step sizes stabilize HPO for thresholds?
- How should null-span detection be thresholded across evidence heads (non-CRF vs CRF)?

## Findings

- Per-class thresholds typically improve macro-F1 under class imbalance compared to a single
  global threshold. Recommended search range: 0.30–0.90.
- Coarser steps (e.g., 0.05) reduce search variance; uniform continuous works but may prolong
  convergence. Start with Uniform and switch to discretized grid if instability observed.
- For non-CRF span scorers, combining a `null_threshold` (no-evidence) with a `min_span_score`
  filter avoids low-confidence spans without suppressing high-precision cases.

## Decisions

- Default strategy: `per_class` thresholds for criteria matching; allow `global` as an option.
- Include `{null_threshold, min_span_score}` for evidence heads that score spans directly.
- Record thresholds in TrialConfig and EvaluationReport for reproducibility.

## Alternatives Considered

- Post-hoc calibration (Platt, temperature scaling): useful but adds complexity; can be a
  follow-up feature. Start with direct threshold tuning.
- Nested per-epoch threshold tuning inside a single trial: increases variance and runtime;
  prefer treating thresholds as part of the trial’s search space.

