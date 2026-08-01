# SeismicXM fine-tuning

## Contents

- Model contract
- Checkpoint selection
- Classification recipe
- Phase-picking recipe
- Unfreezing policy

## Model contract

The pinned `seismicxm.middle.SeismicXM` consumes `(batch, 3, 10240)` by default.
Its normal forward pass returns:

```text
phase, polarity, event_type, reconstructed_waveform, hidden
```

`phase` uses five channels (`background`, `Pg`, `Sg`, `Pn`, `Sn`). The middle
model has a 1024-feature shared representation and an eight-logit event head.
The training script replaces that event head when the requested class count or
ordering differs.

Use `--variant middle`, `tinny`, or `rnn` only with the matching checkpoint (the
upstream lightweight filename is intentionally spelled `tinny`).
Require the window length to be divisible by 64.

## Checkpoint selection

- `seismicxm-middle`: general transfer starting point.
- `seismicxm-classification`: upstream classification-tuned checkpoint; replace
  the head for a new label ontology.
- `transfer-pnw-balanced-200`: useful PNW-domain initialization from the public
  Google Drive model zoo.
- `seismicxm-tiny`: low-resource smoke tests or experiments; pass
  `--variant tinny`.

Keep an existing eight-class head only when the exact class order is known. Pass
that order explicitly with `--classes` and `--keep-classification-head`.

## Classification recipe

Dry-run a PNW binary classifier:

```bash
python <skill>/scripts/train_seismicxm.py \
  --task classification \
  --seismicxm-repo upstream/seismicxm \
  --checkpoint checkpoints/seismicxm.middle.pt \
  --metadata work/pnw.binary.csv \
  --waveform-h5 data/seismic-ai-data/PNW/PNW.hdf5 \
  --output-dir outputs/pnw-head \
  --classes earthquake explosion \
  --trainable head \
  --class-balance weighted \
  --max-train-samples 64 \
  --max-val-samples 32 \
  --dry-run
```

Remove the sample caps and `--dry-run` for training. Compare `head`,
`head-last-block`, and `all` using the same split and seed. Reduce learning rate
when unfreezing more of the backbone (for example, from `1e-3` for a new head to
`1e-4` or lower for broader fine-tuning).

Classification sample caps create deterministic class-stratified debug subsets;
they are for smoke tests, not for reporting population-level metrics.

## Phase-picking recipe

```bash
python <skill>/scripts/train_seismicxm.py \
  --task picking \
  --seismicxm-repo upstream/seismicxm \
  --checkpoint checkpoints/seismicxm.middle.pt \
  --metadata work/phases.csv \
  --waveform-h5 data/phases.h5 \
  --output-dir outputs/seismicxm-picker \
  --trainable head \
  --window-length 10240 \
  --dry-run
```

For generic P/S metadata, explicitly decide whether P/S may be treated as Pg/Sg.
Disable fallback for phase-specific or teleseismic work.

## Unfreezing policy

1. Train only `decoder_event_type` or `decoder_phase` as a baseline.
2. Unfreeze the last Transformer layer with `head-last-block` when validation
   underfits or the target domain differs materially.
3. Fine-tune `all` only with adequate data, a lower learning rate, early stopping,
   and a fixed held-out comparison.

The scripts save raw `best.pt` and `last.pt` state dictionaries plus `run.json`.
