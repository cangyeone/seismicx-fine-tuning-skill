# Model sources, revisions, and licenses

## Upstream code

| Component | Source | Pinned revision | License/status |
|---|---|---|---|
| SeismicXM | https://github.com/cangyeone/seismicxm | `2d8077c62b6600e94d71a512b704b6fd6902f91d` | GPL-3.0 in upstream repository |
| PNSN training demo | https://github.com/cangyeone/pnsn_training_demo | `32d317c37aa8129c938a241083ceec7a77108386` | Check upstream before redistribution; this skill imports it in-place |

`setup_workspace.py` uses these revisions so the training scripts see the model
interfaces validated with this skill. Review newer upstream revisions before
changing the pins.

## Checkpoints

| Key | File | Bytes | SHA-256/source |
|---|---|---:|---|
| `pnsn-v3` | `pnsn.v3.pt` | 1,844,958 | `9f626e5fff4e9390c88e43c2f6230802496163b5d6eefee05e1b6ac7ccebf9e8` |
| `seismicxm-middle` | `seismicxm.middle.pt` | 207,709,060 | `3f051c950ef26110f922c30a3c7d11ed5a73b84bb98e99c2b7943a2a0908b174` |
| `seismicxm-classification` | `seismicxm.middle.classification.pt` | 207,709,060 | `d91b746a727932c701f09e7e18714f4828eac9aa5e0b285cc585eaa6dd90fddf` |
| `seismicxm-tiny` | `seismicxm.tinny.pt` | 34,377,834 | `5139a755a1db98c7483d0e95bedbeb5e2185bb5a8ebd770ab3d3fca6b669c9f9` |

The bundled PNSN base checkpoint comes from
`https://huggingface.co/cangyeone/seismic-snr-filtering-bias`, where project-
controlled checkpoints are published under CC BY 4.0. Preserve attribution,
the source URL, and the adjacent license file when redistributing it.

SeismicXM checkpoints are fetched from the upstream GitHub LFS objects. The PNW
transfer checkpoints are fetched from the Google Drive model zoo linked by the
SeismicXM repository:

`https://drive.google.com/drive/folders/12cKctQFGZg4kqRQqMhdq1VGcDOYqUafG?usp=sharing`

Always compute and record the actual SHA-256 after a Google Drive download.

## Dataset

Seismic-AI-Data is at
`https://www.modelscope.cn/datasets/cangyeone/Seismic-AI-Data`. The repository
metadata declares MIT, but individual upstream datasets can retain their own
terms. Review the selected subset's provenance before redistribution or
commercial use. Never commit raw waveforms to the skill repository by default.
