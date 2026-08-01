---
name: seismicx-fine-tuning
description: Prepare and adapt user-defined seismic waveform datasets, then fine-tune, validate, or compare SeismicXM and PNSN models for phase picking/detection (Pg, Sg, Pn, Sn) and seismic event classification. Use for arbitrary classification ontologies and storage layouts, custom data adapters, built-in HDF5/NPY/NPZ and SeismicX bucketed HDF5 inputs, selected ModelScope cangyeone/Seismic-AI-Data subsets, pretrained-checkpoint adaptation, frozen-head/full-model training, and reproducible seismic ML evaluation.
---

# SeismicX Fine-Tuning

Build reproducible single-station seismic fine-tuning runs around the upstream
`cangyeone/seismicxm` and `cangyeone/pnsn_training_demo` implementations.

## Choose the route

- Use **SeismicXM classification** for earthquake/explosion, regional event type,
  or other single-window class labels.
- Use **PNSN picking** for a compact 100 Hz Pg/Sg/Pn/Sn picker, especially with
  5120-sample windows or limited compute.
- Use **SeismicXM picking** for 10240-sample transfer learning, a shared backbone,
  or later multi-task extension.
- Do not use PNSN for event classification.

Read only the relevant detail before acting:

- User-defined data, adapters, or ModelScope subsets: [data-contract.md](references/data-contract.md)
- SeismicXM training: [seismicxm.md](references/seismicxm.md)
- PNSN training: [pnsn.md](references/pnsn.md)
- Metrics and acceptance: [evaluation.md](references/evaluation.md)
- Checkpoints, revisions, licenses: [model-sources.md](references/model-sources.md)

## Run the workflow

### 1. Confirm the experiment contract

Record the task, label meaning and order, waveform units, component order,
sampling rate, window length, grouping key, train/validation/test policy, target
metric, compute device, and starting checkpoint. Ask for missing information only
when it cannot be inferred safely.

Treat event/source identity as the default grouping key. Never split multiple
traces or augmented windows from the same event across data partitions.

### 2. Inspect before downloading

Prefer user-local data. For Seismic-AI-Data, download explicit paths only:

```bash
python <skill>/scripts/download_modelscope.py \
  --paths DATASET_SUBSET/metadata.csv \
  --local-dir data/seismic-ai-data
```

Before adding `--allow-large`, inspect the selected files, confirm available disk
space, and confirm the user's intent. Never fetch the full multi-terabyte
collection by default.

### 3. Set up pinned model code and weights

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r <skill>/requirements.txt
python <skill>/scripts/setup_workspace.py --root upstream
python <skill>/scripts/download_models.py --list
python <skill>/scripts/download_models.py --model pnsn-v3 --output-dir checkpoints
python <skill>/scripts/download_models.py --model seismicxm-middle --output-dir checkpoints
```

Use a user-supplied checkpoint when requested. Verify its architecture and
checksum before training. Do not commit SeismicXM checkpoints, downloaded data,
or experiment outputs unless the user explicitly requests it and repository
limits/licensing permit it.

### 4. Adapt, build, and validate the data interface

Treat the bundled manifest loader as a fast path, not as a restriction on user
data. First inspect representative metadata and waveform records. If the user's
layout differs, create a small, documented adapter in the experiment workspace
that maps their native schema to the model input and label contract. Do not
rewrite or reorganize the source dataset unless the user requests it.

Normalize metadata and create group-safe partitions:

```bash
python <skill>/scripts/prepare_manifest.py \
  --input data/metadata.csv \
  --output work/manifest.csv \
  --label-column event_label \
  --group-column event_id

python <skill>/scripts/validate_manifest.py \
  --metadata work/manifest.csv \
  --task classification \
  --waveform-h5 data/waveforms.hdf5 \
  --check-waveforms 8
```

Stop on split leakage, absent labels, unreadable waveform keys, wrong shapes, or
non-finite samples. Resample explicitly to 100 Hz before training; do not silently
reinterpret sample indices from another rate. Preserve the actual component order
in the run record.

### 5. Dry-run before a real job

Run the selected training command with `--dry-run`, small sample caps, and the
actual checkpoint. Verify output tensor shapes, class count/order, trainable
parameter count, and device. A successful import alone is not sufficient.

### 6. Fine-tune

Start with a frozen task head baseline, then unfreeze the last shared block or
the full model only if validation results justify the added compute and overfit
risk. Keep the split, seed, preprocessing, class mapping, and evaluation budget
fixed across comparisons.

For imbalanced classification, retain macro-F1 and per-class recall even when
using weighted loss. For phase picking, tune thresholds on validation data only.

### 7. Hand off complete artifacts

Return the exact command, upstream revision, source-checkpoint checksum, manifest
hash, class mapping, seed, trainable modules, best/last checkpoints, `run.json`,
validation metrics, held-out test metrics, and known limitations. Distinguish a
smoke test from a completed experiment.

## Safety and reproducibility rules

- Never load an untrusted pickle checkpoint. Use trusted project checkpoints and
  `weights_only=True` where supported.
- Never use Python `eval` to parse metadata or class mappings.
- Fit normalization statistics, thresholds, and sampling policies on training
  data only.
- Keep the held-out test set untouched until model selection is complete.
- Treat P/S-to-Pg/Sg fallback as an explicit regional assumption; disable it for
  teleseismic or phase-specific labels with `--no-regional-ps-fallback`.
- Do not claim operational detection performance from window-level accuracy.
  Evaluate continuous data, false alarms per time, station diversity, and timing
  residuals before deployment.
