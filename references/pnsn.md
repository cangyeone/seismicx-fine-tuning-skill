# PNSN fine-tuning

## Contents

- Model contract
- Generic manifest route
- CREDIT-X1 upstream route
- Module selection

## Model contract

The compact PNSN BRNN picker consumes three-component 100 Hz waveforms and uses
5120 samples by default. Its five output channels are:

```text
background, Pg, Sg, Pn, Sn
```

The architecture exposes `encoder`, `rnns`, and `decoder`. Window length must be
divisible by 128. The bundled `assets/models/pnsn.v3.pt` base checkpoint is
1,844,958 bytes and is copied with `download_models.py --model pnsn-v3`.

## Generic manifest route

Use the skill's loader for SeismicX bucketed HDF5 or per-row NPY/NPZ data:

```bash
python <skill>/scripts/train_pnsn.py \
  --pnsn-repo upstream/pnsn \
  --checkpoint checkpoints/pnsn.v3.pt \
  --metadata work/phases.csv \
  --waveform-h5 data/phases.h5 \
  --output-dir outputs/pnsn-decoder \
  --trainable decoder \
  --dry-run
```

Remove `--dry-run` after verifying the tensor shape and parameter counts.

## CREDIT-X1 upstream route

When the user has the original CREDIT-X1local HDF5 and split-key NPZ, prefer the
upstream data builder because it preserves that project's record construction:

```bash
python upstream/pnsn/scripts/freeze_finetune_pnsn.py \
  --h5 /path/to/credit-x1.h5 \
  --keys /path/to/creditkeys.npz \
  --base-ckpt checkpoints/pnsn.v3.pt \
  --out-ckpt outputs/pnsn.freeze-decoder.pt \
  --freeze decoder \
  --steps 500 \
  --batch-size 16
```

Note that upstream `--freeze decoder` trains the encoder and recurrent context;
the skill's `--trainable decoder` does the opposite. State the chosen semantics
in the run record.

## Module selection

- `--trainable decoder`: fastest regional phase-head adaptation.
- `--trainable rnns,decoder`: adapt temporal context plus output mapping.
- `--trainable all`: full fine-tuning; use lower learning rates and more data.
- `--trainable encoder`: specialized waveform-front-end adaptation.

Start with the decoder baseline, then expand only if validation timing residuals
and recall show meaningful gains.
