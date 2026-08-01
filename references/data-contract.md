# Data contract

## Contents

- Supported storage
- Manifest columns
- Phase labels
- Splitting and preprocessing
- PNW recipe

## Supported storage

Use a CSV manifest plus one of these waveform layouts:

1. A shared HDF5 file passed with `--waveform-h5`. Address each sample with
   SeismicX `trace_name` syntax such as `bucket4$0,:3,:15001`, or use `hdf5_key`.
2. A per-row `waveform_path` pointing to `.h5`, `.hdf5`, `.npy`, or `.npz`.
   Relative paths resolve from the manifest directory. For NPZ, optionally set
   `waveform_key`; otherwise use `waveform` or the first array.

Store waveforms as `(channels, samples)` or `(samples, channels)`. The loader
detects common layouts and emits `(3, window_length)`. Set missing-component
handling to `zero`, `replicate`, or `error`; document the choice.

## Manifest columns

Common columns:

| Purpose | Canonical column | Accepted alternatives |
|---|---|---|
| Waveform address | `trace_name` | `trace_id`, `hdf5_key`, `waveform_path` |
| Event grouping | `source_group` | `event_id`, `source_id`, `trace_name` |
| Partition | `split` | values: `train`, `val`, `test` |
| Classification | `label` | map another column with `prepare_manifest.py` |
| Sample rate | `sampling_rate` | `trace_sampling_rate_hz`, `sample_rate` |

When `split` is absent, scripts hash the event/group ID with the seed. Do not
fall back to row-random splits when an event identifier exists.

## Phase labels

Provide arrival sample indices, not seconds. Preferred phase-specific columns:

```text
pg_sample,sg_sample,pn_sample,sn_sample
```

The aliases `Pg_sample`, `trace_Pg_arrival_sample`, and corresponding Sg/Pn/Sn
names are accepted. Generic `p_sample`/`s_sample` and PNW
`trace_P_arrival_sample`/`trace_S_arrival_sample` can map to Pg/Sg. This fallback
is suitable only when regional P/S labels genuinely mean crustal Pg/Sg.

Targets have five channels in this order:

```text
background, Pg, Sg, Pn, Sn
```

The scripts form Gaussian arrival targets with configurable sigma (20 samples,
0.20 s at 100 Hz, by default).

## Splitting and preprocessing

Before training:

1. Split by event/source group, then create windows or augmentations.
2. Resample all waveforms and arrival indices together to 100 Hz.
3. Fix component order (for example Z/N/E or E/N/Z) and record it. The supplied
   checkpoints expect three consistent components; metadata alone cannot infer
   physical orientation reliably.
4. Demean and normalize each component. The shared loader supports `std-max`,
   `max`, and `none`; use the same policy for train, validation, test, and inference.
5. Reject or explicitly handle gaps, clipped traces, missing components, NaNs,
   and inconsistent units.

## PNW recipe

ModelScope paths:

```text
PNW/PNW.csv   53,466,340 bytes
PNW/PNW.hdf5 67,343,580,072 bytes
```

The inspected metadata contains 183,909 traces at 100 Hz. `source_type` contains
167,966 `earthquake` and 15,943 `explosion` rows, so a binary run should use
weighted loss or a group-aware balanced sampler and report macro-F1. The richer
`source_type_pnsn_label` field contains eight codes (`eq`, `px`, `lf`, `su`,
`ex`, `qb`, `sn`, `uk`) with extreme tail-class imbalance; do not train it
without first defining the scientific meaning and evaluation policy for each code.

Prepare the binary manifest:

```bash
python <skill>/scripts/prepare_manifest.py \
  --input data/seismic-ai-data/PNW/PNW.csv \
  --output work/pnw.binary.csv \
  --label-column source_type \
  --labels earthquake explosion \
  --group-column event_id
```
