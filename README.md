# SeismicX 地震模型微调 Skill

让 Coding Agent 使用 **SeismicXM** 或 **PNSN** 完成地震波形数据检查、震相拾取微调和地震事件分类微调。

一句话即可发起任务：

> 使用 `$seismicx-fine-tuning`，用我的三分量波形和 CSV 标签微调 SeismicXM 地震分类模型；先检查类别映射与数据泄漏，完成 dry-run 后再训练，并汇报 macro-F1、各类别指标和混淆矩阵。

本 Skill 基于以下项目：

- [cangyeone/seismicxm](https://github.com/cangyeone/seismicxm)
- [cangyeone/pnsn_training_demo](https://github.com/cangyeone/pnsn_training_demo)
- 可选公开数据：[cangyeone/Seismic-AI-Data](https://www.modelscope.cn/datasets/cangyeone/Seismic-AI-Data)

整个训练工作流直接使用 SeismicXM 和 PNSN 的模型与代码，不使用 SeisBench。

## 能做什么

| 任务 | 模型 | 典型用途 |
|---|---|---|
| 地震事件分类 | SeismicXM | 区域地震、爆破、噪声、低频事件或使用者自定义的任意事件类别 |
| 震相拾取/检测 | PNSN | 使用轻量模型拾取 `Pg`、`Sg`、`Pn`、`Sn` |
| 震相拾取/检测 | SeismicXM | 使用 SeismicXM 共享骨干网络拾取 `Pg`、`Sg`、`Pn`、`Sn` |
| 迁移学习 | SeismicXM / PNSN | 从仓库提供的预训练模型或使用者自己的 checkpoint 开始微调 |
| 数据适配 | — | 通过适配器支持使用者自定义格式，并直接支持常见 CSV/HDF5/NPY/NPZ 组织方式 |

使用者可以直接提供原始数据组织方式和标签体系。Agent 会检查代表性样本，兼容时使用内置读取器，不兼容时创建有记录、非破坏性的数据适配器，无需重新整理原始数据集。

## 1. 安装到 Coding Agent

### 方法一：让 Agent 自己安装

把下面这句话直接发给 Codex 或其他支持 `SKILL.md` 的 Coding Agent：

```text
请把 https://gitee.com/cangyeone/seismicx-fine-tuning-skill 安装到当前用户的 Skills 目录，技能名保持为 seismicx-fine-tuning。安装后读取 SKILL.md，并告诉我如何使用 $seismicx-fine-tuning 发起训练任务。
```

安装完成后，建议新建一个任务或重启 Agent，使其重新扫描 Skills。

### 方法二：安装到 Codex 用户目录

用户级安装适合在多个项目中复用：

```bash
mkdir -p ~/.codex/skills
git clone https://gitee.com/cangyeone/seismicx-fine-tuning-skill.git \
  ~/.codex/skills/seismicx-fine-tuning
```

如果设置了自定义 `CODEX_HOME`，请把仓库克隆到：

```text
$CODEX_HOME/skills/seismicx-fine-tuning
```

### 方法三：安装到当前项目

项目级安装便于让团队在同一个代码仓库中共享 Skill：

```bash
mkdir -p .agents/skills
git clone https://gitee.com/cangyeone/seismicx-fine-tuning-skill.git \
  .agents/skills/seismicx-fine-tuning
```

请确认最终目录中直接存在 `SKILL.md`：

```text
.agents/skills/seismicx-fine-tuning/SKILL.md
```

### 其他 Coding Agent

如果 Agent 支持 [Agent Skills](https://agentskills.io/) 或能够加载 `SKILL.md`，可以使用跨客户端的用户级目录：

```bash
mkdir -p ~/.agents/skills
git clone https://gitee.com/cangyeone/seismicx-fine-tuning-skill.git \
  ~/.agents/skills/seismicx-fine-tuning
```

也可以将本仓库完整复制到该 Agent 自己配置的 Skills 目录。不要只复制 `SKILL.md`，因为训练流程还会使用 `scripts/`、`references/` 和 `assets/`。不同客户端扫描的目录可能不同，请以该 Agent 的技能设置为准。

如果 Agent 没有自动发现 Skills 的机制，也可以在提示词中直接指定本仓库：

```text
先克隆 https://gitee.com/cangyeone/seismicx-fine-tuning-skill，读取其中的 SKILL.md，并严格按照该 Skill 使用我的数据完成 SeismicXM 分类微调。
```

### 更新 Skill

用户级安装：

```bash
git -C ~/.codex/skills/seismicx-fine-tuning pull --ff-only
```

项目级安装：

```bash
git -C .agents/skills/seismicx-fine-tuning pull --ff-only
```

## 2. 在 Agent 中调用

推荐显式写出技能名：

```text
使用 $seismicx-fine-tuning，……
```

也可以使用自然语言描述任务。只要 Agent 已加载该 Skill，涉及 SeismicXM、PNSN、地震分类、Pg/Sg/Pn/Sn 拾取或地震波形迁移学习的请求都可以触发它。

一次完整请求最好说明：

- 任务是 `classification` 还是 `picking`；
- 数据集路径，以及具有代表性的元数据或波形记录；
- 已知的原始数据组织方式；
- 分类标签列，或 Pg/Sg/Pn/Sn 到时样点列；
- 同一事件的分组列，例如 `event_id`；
- 使用 SeismicXM 还是 PNSN；
- 是否有自己的预训练 checkpoint；
- 训练设备、预算或 epoch 数等限制。

信息不完整时，Agent 会先检查数据并从现有字段推断安全的默认值；涉及类别含义、分量顺序或震相语义等不能可靠推断的信息，应由使用者确认。

## 3. 一句话任务示例

### 使用自有数据做任意类别的地震分类

```text
使用 $seismicx-fine-tuning，检查 /data/my_dataset 下的原始数据格式并微调 SeismicXM 分类模型；标签表示事件类型，同一事件的记录必须保留在同一数据分区。兼容时复用内置读取器，否则创建并测试非破坏性数据适配器；固定类别顺序，检查数据泄漏和波形形状，完成小样本 dry-run 后再正式训练，输出 macro-F1、各类别 precision/recall/F1、混淆矩阵及可复现的训练记录。
```

类别可以由使用者自行定义，例如 `regional_earthquake`、`quarry_blast`、`noise`、`low_frequency_event`。

### 使用 PNSN 微调 Pg/Sg/Pn/Sn 拾取模型

```text
使用 $seismicx-fine-tuning，用 /data/phases.h5 和 /data/phases.csv 微调 PNSN Pg/Sg/Pn/Sn 拾取模型；数据为 100 Hz 三分量波形，到时列为 pg_sample、sg_sample、pn_sample、sn_sample，先验证标签覆盖和事件级划分，再从 decoder-only 基线开始训练。
```

### 使用 SeismicXM 微调震相拾取模型

```text
使用 $seismicx-fine-tuning，用我的 100 Hz 三分量区域地震数据微调 SeismicXM 震相拾取模型，输出顺序固定为 background、Pg、Sg、Pn、Sn；先做小样本 dry-run，再比较 head 与 head-last-block 两种微调策略。
```

### 使用自己的已有模型继续微调

```text
使用 $seismicx-fine-tuning，从 /models/my_seismicxm.pt 开始，用 /data/new_region.csv 和对应波形适配新区域；先核对 checkpoint 架构、类别顺序和输入长度，不覆盖原模型，所有结果写到 /outputs/new_region_run。
```

### 使用公开数据做一次可复现实验

```text
使用 $seismicx-fine-tuning，从 Seismic-AI-Data 中只下载完成任务所需的子集，用 SeismicXM 做分类微调；下载大文件前先告诉我大小并确认磁盘空间，先完成元数据检查和 dry-run。
```

例如，可以使用公开的 PNW 子集及其 `source_type` 标签演示 earthquake/explosion 分类。

## 4. Agent 会执行什么

正常情况下，Agent 会按以下顺序工作：

1. 检查任务、类别或震相定义、采样率、分量顺序、窗口长度和 checkpoint。
2. 检查本地数据，避免无必要地下载大型公开数据。
3. 准备固定版本的 SeismicXM/PNSN 上游代码和所需模型权重。
4. 检查原始数据组织方式；兼容时使用内置读取器，否则在实验目录创建有记录的数据适配器。
5. 按事件或震源安全划分数据，并检查类别缺失、标签覆盖、数据泄漏、波形访问、张量形状以及 NaN/Inf。
6. 使用真实 checkpoint 和小样本执行 `--dry-run`，核对输入输出形状及可训练参数。
7. 从冻结任务头或 decoder 的基线开始，再根据验证结果决定是否扩大解冻范围。
8. 保存模型、指标、参数、数据和 checkpoint 哈希，并区分 smoke test 与正式实验。

Agent 不应在未确认磁盘空间时下载数十 GB 的波形文件，也不应把训练集、验证集或同一事件产生的多个窗口随机打散到不同分区。

## 5. 自定义数据与内置格式

数据集不需要转换成规定的磁盘结构。Agent 应先检查使用者的原始格式并保持源数据不变；如果与内置读取器不同，可以创建以下任一适配层：

- 生成指向原始波形记录的 manifest；
- 创建返回模型所需 `waveform` 和 `target` 张量的数据集包装器。

适配器应记录波形寻址方式、张量排列、采样率、单位、分量顺序、标签映射、事件分组、过滤规则和重采样过程，并在正式训练前使用代表性记录完成测试。

以下格式是仓库直接支持的快捷路径。

### 分类 manifest

最小示例：

```csv
waveform_path,label,source_group,split,sampling_rate
waveforms/a001.npy,regional_earthquake,event_001,train,100
waveforms/a002.npy,quarry_blast,event_002,val,100
waveforms/a003.npy,noise,event_003,test,100
```

也可以保留原始标签列，例如 `event_type`，然后让 Agent 或 `prepare_manifest.py` 将它映射到规范列 `label`。

### 震相拾取 manifest

```csv
waveform_path,source_group,split,sampling_rate,pg_sample,sg_sample,pn_sample,sn_sample
waveforms/e001.npy,event_001,train,100,1030,2280,,
waveforms/e002.npy,event_002,val,100,,,1850,4320
```

震相到时应使用**样点索引**，不是秒。目标通道顺序固定为：

```text
background, Pg, Sg, Pn, Sn
```

### 内置波形存储

支持：

- 一个共享 HDF5，通过 `trace_name` 或 `hdf5_key` 定位波形；
- SeismicX bucketed HDF5，例如 `bucket4$0,:3,:15001`；
- 每行一个 `.h5`、`.hdf5`、`.npy` 或 `.npz` 文件；
- `(channels, samples)` 或 `(samples, channels)` 排列。

预训练模型默认使用 100 Hz、三分量输入。当前脚本不会把非 100 Hz 数据悄悄当作 100 Hz 使用；应先对波形和到时样点进行一致的显式重采样，并记录实际分量顺序。

更完整的数据约定见 [`references/data-contract.md`](references/data-contract.md)。

## 6. 手动运行

Skill 的主要用途是让 Agent 组织工作流。研究者也可以手动执行其中的脚本。

### 6.1 使用已有 Conda 环境

先进入已安装 PyTorch 等依赖的环境：

```bash
conda activate YOUR_ENV
python -c "import torch, numpy, h5py, einops; print(torch.__version__)"
```

缺少依赖时再安装：

```bash
python -m pip install -r /path/to/seismicx-fine-tuning/requirements.txt
```

下面用变量表示 Skill 的实际路径：

```bash
SEISMICX_SKILL=/path/to/seismicx-fine-tuning
```

### 6.2 准备上游代码和模型

只准备需要的模型代码。例如只做 SeismicXM 分类：

```bash
python "$SEISMICX_SKILL/scripts/setup_workspace.py" \
  --root upstream \
  --components seismicxm
```

查看可用 checkpoint：

```bash
python "$SEISMICX_SKILL/scripts/download_models.py" --list
```

下载 SeismicXM 分类 checkpoint：

```bash
python "$SEISMICX_SKILL/scripts/download_models.py" \
  --model seismicxm-classification \
  --output-dir checkpoints
```

PNSN 模型较小，已经包含在仓库中，可复制到实验目录：

```bash
python "$SEISMICX_SKILL/scripts/download_models.py" \
  --model pnsn-v3 \
  --output-dir checkpoints
```

也可以直接把自己的 checkpoint 路径传给训练脚本。不要使用来源不可信的 pickle 权重。

### 6.3 使用内置 manifest 路径准备并检查分类数据

假设原始标签列是 `event_type`，事件标识列是 `event_id`：

```bash
python "$SEISMICX_SKILL/scripts/prepare_manifest.py" \
  --input data/my_metadata.csv \
  --output work/classification.csv \
  --label-column event_type \
  --group-column event_id
```

验证 manifest 和部分真实波形：

```bash
python "$SEISMICX_SKILL/scripts/validate_manifest.py" \
  --metadata work/classification.csv \
  --task classification \
  --waveform-h5 data/my_waveforms.h5 \
  --check-waveforms 8
```

### 6.4 微调任意类别的 SeismicXM 分类模型

先执行 dry-run：

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

把 `--classes` 后面的值替换为自己的类别，并按需要固定类别顺序。如果省略 `--classes`，脚本会从 `label` 列收集类别并按字符串排序。

确认 dry-run 的形状、类别和参数量无误后，移除 `--dry-run`、`--max-train-samples` 和 `--max-val-samples`，再开始正式训练：

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

建议先比较 `head`，再根据验证集结果尝试 `head-last-block` 或 `all`。扩大解冻范围时应降低学习率，并保持相同的数据划分、类别顺序和随机种子。

### 6.5 微调 PNSN 震相拾取模型

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

确认后移除样本上限和 `--dry-run`。PNSN 可训练模块包括 `encoder`、`rnns` 和 `decoder`，也可以使用 `--trainable all`。

### 6.6 微调 SeismicXM 震相拾取模型

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

### 6.7 可选：从 ModelScope 下载公开子集

只下载明确需要的路径。下面是一个公开分类数据示例：

```bash
python "$SEISMICX_SKILL/scripts/download_modelscope.py" \
  --paths PNW/PNW.csv \
  --local-dir data/seismic-ai-data
```

PNW 的 HDF5 波形约 62.7 GiB，下载时需要显式添加 `--allow-large`。不要默认下载整个 Seismic-AI-Data；它是一个包含多个数据集的大型集合。

## 7. 输出结果

训练目录至少包含：

```text
outputs/experiment/
├── best.pt
├── last.pt
└── run.json
```

- `best.pt`：验证损失最优的模型参数；
- `last.pt`：最后一个 epoch 的模型参数；
- `run.json`：输入参数、上游代码版本、checkpoint 与 manifest 哈希、可训练参数量及每个 epoch 的验证指标。

分类验证指标包含 accuracy、macro precision/recall/F1、各类别 precision/recall/F1 和混淆矩阵。震相拾取验证包含各震相的位置误差及指定容差内的召回率。

正式科研结果还应在独立测试集上评估。面向连续地震监测时，还需要报告连续数据中的误报率、检出率、到时残差以及跨台站、时间和区域的稳定性，不能只用窗口级准确率代替运行性能。

## 8. 常见问题

### 是否使用 SeisBench？

不使用。本 Skill 直接调用 SeismicXM/PNSN 模型，并使用仓库自带的数据读取、manifest、训练和验证脚本。

### 数据不是 100 Hz 怎么办？

先对波形做显式重采样，并同步换算所有震相到时样点。不要只修改 CSV 中的采样率字段。

### 显存不足怎么办？

先减小 `--batch-size`，选择 `--variant tinny` 或使用更轻量的 PNSN。执行正式训练前始终保留一次小样本 dry-run。

### 为什么必须按事件分组划分？

同一事件的多个台站记录、裁窗或增强样本如果同时进入训练集和验证/测试集，会造成数据泄漏并高估性能。默认优先使用 `event_id`、`source_id` 或 `source_group` 划分。

### P/S 标签能否直接当作 Pg/Sg？

只有在区域数据中 P/S 的科学含义确实对应 Pg/Sg 时才可以。对于远震或已经区分震相的研究，应使用明确的 Pg/Sg/Pn/Sn 标签，并加 `--no-regional-ps-fallback` 禁用自动回退。

## 9. 仓库结构

```text
seismicx-fine-tuning-skill/
├── SKILL.md                    # Agent 执行规范
├── agents/openai.yaml          # Agent UI 元数据
├── scripts/                    # 下载、数据验证和训练入口
├── references/                 # 数据、模型与评估约定
├── assets/models/pnsn.v3.pt    # 随仓库提供的小型 PNSN 权重
└── requirements.txt
```

模型来源、固定版本、校验值和许可证说明见 [`references/model-sources.md`](references/model-sources.md)。使用公开数据或模型前，请同时检查原始项目和数据子集的许可证与引用要求。
