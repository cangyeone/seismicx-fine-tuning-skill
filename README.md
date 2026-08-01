# SeismicX Fine-Tuning Skill

Use a coding agent to inspect seismic waveform data and fine-tune **SeismicXM** or **PNSN** models for seismic event classification and phase picking.

Start a complete task with one prompt:

> Use `$seismicx-fine-tuning` to fine-tune a SeismicXM classifier with my three-component waveforms and CSV labels. Check the class mapping and split leakage, complete a dry run before training, and report macro-F1, per-class metrics, and the confusion matrix.

This skill is built around:

- [cangyeone/seismicxm](https://github.com/cangyeone/seismicxm)
- [cangyeone/pnsn_training_demo](https://github.com/cangyeone/pnsn_training_demo)
- Optional public data: [cangyeone/Seismic-AI-Data](https://www.modelscope.cn/datasets/cangyeone/Seismic-AI-Data)

The workflow uses the native SeismicXM and PNSN implementations directly. It does not use SeisBench.

## Capabilities

| Task | Model | Typical use |
|---|---|---|
| Seismic event classification | SeismicXM | Regional earthquakes, blasts, noise, low-frequency events, or any user-defined event classes |
| Phase picking/detection | PNSN | A compact picker for `Pg`, `Sg`, `Pn`, and `Sn` |
| Phase picking/detection | SeismicXM | A shared SeismicXM backbone for `Pg`, `Sg`, `Pn`, and `Sn` |
| Transfer learning | SeismicXM / PNSN | Start from the supplied pretrained models or a user-provided checkpoint |
| Data adaptation | — | User-defined layouts through an adapter, with direct support for common CSV/HDF5/NPY/NPZ layouts |

Users provide their native data organization and label ontology. The agent inspects representative records and either uses a bundled loader or creates a documented adapter without requiring the source dataset to be reorganized.

## 1. Install the skill in a coding agent

### Option A: ask the agent to install it

Send this prompt to Codex or another coding agent that supports `SKILL.md`:

```text
Install https://github.com/cangyeone/seismicx-fine-tuning-skill in my user-level skills directory under the name seismicx-fine-tuning. After installation, read SKILL.md and tell me how to invoke $seismicx-fine-tuning for a training task.
```

After installation, start a new task or restart the agent so that it rescans the available skills.

### Option B: install for the current Codex user

A user-level installation makes the skill available across projects:

```bash
mkdir -p ~/.codex/skills
git clone https://github.com/cangyeone/seismicx-fine-tuning-skill.git \
  ~/.codex/skills/seismicx-fine-tuning
```

If you use a custom `CODEX_HOME`, clone the repository to:

```text
$CODEX_HOME/skills/seismicx-fine-tuning
```

### Option C: install in one project

A project-level installation is useful when a team wants to share the skill with the project:

```bash
mkdir -p .agents/skills
git clone https://github.com/cangyeone/seismicx-fine-tuning-skill.git \
  .agents/skills/seismicx-fine-tuning
```

Confirm that `SKILL.md` is directly inside the installed skill directory:

```text
.agents/skills/seismicx-fine-tuning/SKILL.md
```

### Other coding agents

For agents that support the [Agent Skills](https://agentskills.io/) format, the cross-client user directory can be used:

```bash
mkdir -p ~/.agents/skills
git clone https://github.com/cangyeone/seismicx-fine-tuning-skill.git \
  ~/.agents/skills/seismicx-fine-tuning
```

You may also clone the repository into the client-specific skills directory configured by your agent. Keep the whole repository together: the workflow needs `SKILL.md`, `scripts/`, `references/`, and `assets/`. Skill discovery paths vary by client, so consult the client's skill settings when necessary.

If an agent cannot discover skills automatically, point it at the repository explicitly:

```text
Clone https://github.com/cangyeone/seismicx-fine-tuning-skill, read its SKILL.md, and follow that skill to fine-tune a SeismicXM classifier with my dataset.
```

### Update the skill

For a user-level Codex installation:

```bash
git -C ~/.codex/skills/seismicx-fine-tuning pull --ff-only
```

For a project-level installation:

```bash
git -C .agents/skills/seismicx-fine-tuning pull --ff-only
```

## 2. Invoke the skill

Explicit invocation is recommended:

```text
Use $seismicx-fine-tuning to ...
```

Natural-language activation also works when the agent has loaded the skill. Requests involving SeismicXM, PNSN, seismic classification, Pg/Sg/Pn/Sn picking, or seismic waveform transfer learning should match the skill description.

A useful request should provide as much of the following information as possible:

- whether the task is `classification` or `picking`;
- paths to the dataset and representative metadata or waveform records;
- a description of the native data layout, when available;
- the classification label column or Pg/Sg/Pn/Sn arrival-sample columns;
- an event grouping column such as `event_id`;
- the preferred model, SeismicXM or PNSN;
- the path to an existing checkpoint, if any;
- device, runtime, epoch, or compute-budget limits.

When information is missing, the agent should inspect the data and infer only safe defaults. The user should confirm details that cannot be inferred reliably, especially class semantics, component order, waveform units, and phase meaning.

## 3. One-prompt examples

### Classify arbitrary event types in a custom dataset

```text
Use $seismicx-fine-tuning to inspect the native format under /data/my_dataset and fine-tune a SeismicXM classifier. The labels describe event type and records from the same event must stay in the same split. Reuse a bundled loader if compatible; otherwise create and test a non-destructive data adapter. Lock the class order, check split leakage and waveform shapes, run a small dry run, then train and report macro-F1, per-class precision/recall/F1, the confusion matrix, and reproducibility metadata.
```

The classes may use any user-defined ontology, for example `regional_earthquake`, `quarry_blast`, `noise`, and `low_frequency_event`.

### Fine-tune a PNSN Pg/Sg/Pn/Sn picker

```text
Use $seismicx-fine-tuning with /data/phases.h5 and /data/phases.csv to fine-tune the PNSN Pg/Sg/Pn/Sn picker. The data contain 100 Hz three-component waveforms and the arrival columns are pg_sample, sg_sample, pn_sample, and sn_sample. Validate phase coverage and event-level splits, run a dry run, and start with a decoder-only baseline.
```

### Fine-tune a SeismicXM phase picker

```text
Use $seismicx-fine-tuning to adapt SeismicXM to my 100 Hz three-component regional dataset. Keep the phase output order background, Pg, Sg, Pn, Sn. Run a small dry run first, then compare head and head-last-block fine-tuning with the same split and seed.
```

### Continue from an existing checkpoint

```text
Use $seismicx-fine-tuning to adapt /models/my_seismicxm.pt to /data/new_region.csv and its waveform files. Verify the checkpoint architecture, class order, and input length first. Do not overwrite the source checkpoint, and write all results to /outputs/new_region_run.
```

### Use an optional public dataset

```text
Use $seismicx-fine-tuning to download only the required subset of Seismic-AI-Data and fine-tune a SeismicXM classifier. Report the download size and check available disk space before downloading large waveform files. Complete metadata validation and a dry run before training.
```

For example, the public PNW subset can be used to demonstrate an earthquake/explosion classification run with the `source_type` labels.

## 4. What the agent should do

The expected workflow is:

1. Confirm the task, class or phase definitions, sample rate, component order, window length, and checkpoint.
2. Inspect local data before considering any public-data download.
3. Prepare pinned SeismicXM/PNSN upstream code and only the required checkpoint.
4. Inspect the native data layout; use a bundled loader when compatible or create a documented adapter in the experiment workspace.
5. Build group-safe partitions and check missing classes, label coverage, leakage, waveform access, tensor shapes, and NaN/Inf values.
6. Run `--dry-run` with the real checkpoint and a small sample cap to verify input/output shapes and trainable parameters.
7. Start with a frozen task-head or decoder baseline, and unfreeze more of the network only when validation results justify it.
8. Save model weights, metrics, arguments, source revisions, and hashes, and distinguish a smoke test from a completed experiment.

The agent should not download waveform files that are tens of gigabytes without checking disk space and user intent. It should not randomly distribute traces, windows, or augmentations from the same event across train, validation, and test partitions.

## 5. User-defined data and built-in formats

The dataset does not need to be converted into a prescribed on-disk layout. The agent should first inspect the user's native format and preserve the source data. If it differs from the bundled loader, the agent can create either:

- a manifest adapter that points to the original waveform records; or
- a dataset wrapper that returns model-ready `waveform` and `target` tensors.

Every adapter should document waveform addressing, tensor layout, sample rate, units, component order, label mapping, event grouping, filtering, and resampling. It should be tested on representative records before training.

The formats below are built-in fast paths.

### Classification manifest

Minimal example:

```csv
waveform_path,label,source_group,split,sampling_rate
waveforms/a001.npy,regional_earthquake,event_001,train,100
waveforms/a002.npy,quarry_blast,event_002,val,100
waveforms/a003.npy,noise,event_003,test,100
```

An existing label column such as `event_type` may be retained and mapped to the canonical `label` column with `prepare_manifest.py`.

### Phase-picking manifest

```csv
waveform_path,source_group,split,sampling_rate,pg_sample,sg_sample,pn_sample,sn_sample
waveforms/e001.npy,event_001,train,100,1030,2280,,
waveforms/e002.npy,event_002,val,100,,,1850,4320
```

Arrival values must be **sample indices**, not seconds. The target-channel order is fixed:

```text
background, Pg, Sg, Pn, Sn
```

### Built-in waveform storage

Supported layouts include:

- one shared HDF5 file addressed by `trace_name` or `hdf5_key`;
- SeismicX bucketed HDF5 addresses such as `bucket4$0,:3,:15001`;
- one `.h5`, `.hdf5`, `.npy`, or `.npz` file per manifest row;
- waveform arrays in `(channels, samples)` or `(samples, channels)` order.

The supplied pretrained models expect 100 Hz three-component inputs by default. The scripts do not silently reinterpret non-100 Hz data as 100 Hz. Resample the waveform and every arrival index together, and record the actual component order.

See [`references/data-contract.md`](references/data-contract.md) for the complete data contract.

## 6. Manual workflow

The primary use case is to let an agent organize the workflow, but every bundled script can also be run manually.

### 6.1 Use an existing Conda environment

Activate an environment that already contains PyTorch and the required libraries:

```bash
conda activate YOUR_ENV
python -c "import torch, numpy, h5py, einops; print(torch.__version__)"
```

Install only missing dependencies:

```bash
python -m pip install -r /path/to/seismicx-fine-tuning/requirements.txt
```

The examples below use a variable for the installed skill directory:

```bash
SEISMICX_SKILL=/path/to/seismicx-fine-tuning
```

### 6.2 Prepare upstream code and checkpoints

Prepare only the model code required by the task. For a SeismicXM-only experiment:

```bash
python "$SEISMICX_SKILL/scripts/setup_workspace.py" \
  --root upstream \
  --components seismicxm
```

List available checkpoints:

```bash
python "$SEISMICX_SKILL/scripts/download_models.py" --list
```

Download the SeismicXM classification checkpoint:

```bash
python "$SEISMICX_SKILL/scripts/download_models.py" \
  --model seismicxm-classification \
  --output-dir checkpoints
```

The compact PNSN checkpoint is bundled with the skill and can be copied into the experiment directory:

```bash
python "$SEISMICX_SKILL/scripts/download_models.py" \
  --model pnsn-v3 \
  --output-dir checkpoints
```

A compatible user-provided checkpoint may be passed directly to a training command. Do not load untrusted pickle checkpoints.

### 6.3 Prepare and validate classification data with the built-in manifest path

Assume that the source label column is `event_type` and the event identifier is `event_id`:

```bash
python "$SEISMICX_SKILL/scripts/prepare_manifest.py" \
  --input data/my_metadata.csv \
  --output work/classification.csv \
  --label-column event_type \
  --group-column event_id
```

Validate the manifest and a small number of real waveforms:

```bash
python "$SEISMICX_SKILL/scripts/validate_manifest.py" \
  --metadata work/classification.csv \
  --task classification \
  --waveform-h5 data/my_waveforms.h5 \
  --check-waveforms 8
```

### 6.4 Fine-tune a SeismicXM classifier with arbitrary classes

Run a dry run first:

```bash
python "$SEISMICX_SKILL/scripts/train_seismicxm.py" \
  --task classification \
  --seismicxm-repo upstream/seismicxm \
  --variant middle \
  --checkpoint checkpoints/seismicxm.middle.classification.pt \
  --metadata work/classification.csv \
  --waveform-h5 data/my_waveforms.h5 \
  --output-dir outputs/classifier-head \
  --classes regional_earthquake quarry_blast noise \
  --trainable head \
  --class-balance weighted \
  --max-train-samples 64 \
  --max-val-samples 32 \
  --dry-run
```

Replace the values after `--classes` with the dataset's actual class names and preferred order. If `--classes` is omitted, the script collects values from `label` and sorts them lexicographically.

After confirming the shapes, class order, and parameter counts, remove `--dry-run`, `--max-train-samples`, and `--max-val-samples` for the full training run:

```bash
python "$SEISMICX_SKILL/scripts/train_seismicxm.py" \
  --task classification \
  --seismicxm-repo upstream/seismicxm \
  --variant middle \
  --checkpoint checkpoints/seismicxm.middle.classification.pt \
  --metadata work/classification.csv \
  --waveform-h5 data/my_waveforms.h5 \
  --output-dir outputs/classifier-head \
  --classes regional_earthquake quarry_blast noise \
  --trainable head \
  --class-balance weighted \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-3
```

Establish the `head` baseline first, then compare `head-last-block` or `all` if validation results justify broader fine-tuning. Lower the learning rate when unfreezing more of the backbone, and keep the split, class order, and seed fixed across comparisons.

### 6.5 Fine-tune the PNSN phase picker

```bash
python "$SEISMICX_SKILL/scripts/setup_workspace.py" \
  --root upstream \
  --components pnsn

python "$SEISMICX_SKILL/scripts/train_pnsn.py" \
  --pnsn-repo upstream/pnsn \
  --checkpoint checkpoints/pnsn.v3.pt \
  --metadata work/phases.csv \
  --waveform-h5 data/phases.h5 \
  --output-dir outputs/pnsn-decoder \
  --trainable decoder \
  --window-length 5120 \
  --max-train-samples 64 \
  --max-val-samples 32 \
  --dry-run
```

Remove the sample caps and `--dry-run` after validation. PNSN training can select `encoder`, `rnns`, `decoder`, a comma-separated combination, or `all`.

### 6.6 Fine-tune the SeismicXM phase picker

```bash
python "$SEISMICX_SKILL/scripts/train_seismicxm.py" \
  --task picking \
  --seismicxm-repo upstream/seismicxm \
  --variant middle \
  --checkpoint checkpoints/seismicxm.middle.pt \
  --metadata work/phases.csv \
  --waveform-h5 data/phases.h5 \
  --output-dir outputs/seismicxm-picker \
  --trainable head \
  --window-length 10240 \
  --max-train-samples 64 \
  --max-val-samples 32 \
  --dry-run
```

### 6.7 Optionally download a public ModelScope subset

Download explicit paths only. The following is a classification-data example:

```bash
python "$SEISMICX_SKILL/scripts/download_modelscope.py" \
  --paths PNW/PNW.csv \
  --local-dir data/seismic-ai-data
```

The PNW HDF5 waveform file is approximately 62.7 GiB and requires the explicit `--allow-large` flag. Do not download the entire Seismic-AI-Data collection by default; it contains multiple large datasets.

## 7. Training outputs

A completed training directory contains at least:

```text
outputs/experiment/
├── best.pt
├── last.pt
└── run.json
```

- `best.pt`: model parameters at the best validation loss;
- `last.pt`: model parameters from the final epoch;
- `run.json`: input arguments, upstream revision, checkpoint and manifest hashes, parameter counts, and validation history.

Classification validation includes accuracy, macro precision/recall/F1, per-class precision/recall/F1, and the confusion matrix. Phase-picking validation includes per-phase position error and recall within the configured tolerance.

Research results should also be evaluated on an untouched test set. Before operational continuous monitoring, measure false alarms, detection probability, timing residuals, and stability across stations, time periods, and regions. Window-level accuracy alone is not an operational evaluation.

## 8. Frequently asked questions

### Does this skill use SeisBench?

No. It calls the SeismicXM and PNSN models directly and uses the manifest, data-loading, training, and validation scripts bundled in this repository.

### What if the data are not sampled at 100 Hz?

Resample the waveforms explicitly and transform all arrival-sample indices with the same factor. Do not change only the sample-rate value in the CSV.

### What if GPU memory is insufficient?

Reduce `--batch-size`, try the SeismicXM `tinny` variant, or use the compact PNSN picker. Always keep a small dry run before the full experiment.

### Why are event-group splits required?

If multiple station traces, windows, or augmentations from the same event appear in both training and validation/test data, leakage can substantially inflate measured performance. Prefer `event_id`, `source_id`, or `source_group` as the grouping key.

### Can generic P/S labels be treated as Pg/Sg?

Only when the scientific meaning of P/S in a regional dataset genuinely corresponds to Pg/Sg. For teleseismic or phase-specific studies, use explicit Pg/Sg/Pn/Sn labels and pass `--no-regional-ps-fallback`.

## 9. Repository layout

```text
seismicx-fine-tuning-skill/
├── SKILL.md                    # Agent execution instructions
├── agents/openai.yaml          # Agent UI metadata
├── scripts/                    # Download, validation, and training entry points
├── references/                 # Data, model, and evaluation contracts
├── assets/models/pnsn.v3.pt    # Bundled compact PNSN checkpoint
└── requirements.txt
```

See [`references/model-sources.md`](references/model-sources.md) for source revisions, checksums, and licensing notes. Review the original model and dataset licenses and citation requirements before redistributing or using selected public assets.
