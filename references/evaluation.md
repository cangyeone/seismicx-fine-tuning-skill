# Evaluation and acceptance

## Classification

Report at minimum:

- class counts by split and group counts by split;
- accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix;
- results by station/network, magnitude, distance, SNR, and time period when those
  metadata exist;
- calibration or threshold curves when outputs drive decisions.

For PNW earthquake/explosion classification, do not use accuracy alone because
the inspected class ratio is about 10.5:1. Keep event-group splits fixed across
head-only, partial, and full fine-tuning comparisons.

## Phase picking

For each of Pg, Sg, Pn, and Sn, report:

- labeled-example count;
- precision, recall, and F1 at a declared probability threshold;
- timing residual median, MAE, standard deviation, and 90th/95th percentiles;
- recall within declared tolerances such as 0.1 s, 0.5 s, and 1.0 s;
- false picks per window and, for continuous evaluation, false picks per hour.

The supplied training scripts record peak-position MAE and recall at one sample
tolerance for quick validation. Treat these as model-selection diagnostics, not
as a complete deployment evaluation.

## Acceptance sequence

1. Verify schema and waveform reads.
2. Overfit a tiny training subset to catch label/order bugs.
3. Compare frozen-head and broader fine-tuning on validation data.
4. Lock preprocessing and thresholds.
5. Evaluate once on the held-out event-group test set.
6. For operational use, run continuous streams spanning quiet periods, event
   sequences, multiple stations, gaps, clipping, and regional noise changes.

Flag a run as incomplete if it lacks a held-out test set, class/phase mapping,
checkpoint provenance, or continuous false-alarm evaluation for deployment.
