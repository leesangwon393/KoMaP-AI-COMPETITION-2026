# Final model1 / model2 / model3 code

## Directory rules

```text
src/                  inference/ensemble, shared model factory and metrics
  model1/             model.py, train.py, infer.py and required model-specific code
  model2/
  model3/
resources/            settings.json, model configs, environment, checkpoint hashes, licenses
Thirdparty/weights/   public ImageNet initialization weights (not in Git)
run/checkpoints/      released model1/model2/model3 best.pt (not in Git)
run/inference/        newly generated probabilities, masks, metrics
run/retraining/       fresh training outputs
script/               reserved for reusable shell workflows (no redundant wrappers)
```

The existing root scripts, experiments/, docs/ and releases/ are historical repository content; this change does not relocate or overwrite those files. Required model-specific Python packages retain their internal layout to avoid import collisions. New project implementation lives under src. Do not edit external code in Thirdparty. Do not commit data, checkpoints or outputs.

## Final models

| Model | Architecture | Training epochs | Selected epoch | D4 Validation mIoU | Weight |
|---|---|---:|---:|---:|---:|
| model1 | ConvNeXt-Tiny U-Net/scSE + raw-detail fusion | 300 | 172 | 80.3762% | 0.30 |
| model2 | ResNet101 U-Net/scSE + semantic-gated detail | 300 | 184 | 80.3861% | 0.35 |
| model3 | Full U2-Net + ECA | 150 | 135 | 80.2360% | 0.35 |

Frozen Validation20 image-macro mIoU: **81.01828805454941%**. This is not an official Test score. Validation was used for checkpoint and ensemble selection. Absent in both prediction and truth: class IoU=1. Test ground truth was not used.

## Setup and separately supplied files

Linux + NVIDIA CUDA environment:

```sh
conda env create -f resources/environment.yaml
conda activate komap-review
```

No new environment is created automatically by these scripts. Reuse a matching existing environment where available.

The separately supplied `A004092_weights.zip` uses submission paths. Copy:
- `02_program/models/model1/best.pt` -> `run/checkpoints/model1/best.pt` (likewise model2/model3).
- `02_program/Thirdparty/weights/*.pth` -> `Thirdparty/weights/*.pth`.

Inference checks checkpoint SHA256 against resources/checkpoints.json and strict-loads the state. Public ConvNeXt-Tiny and ResNet101 weights are also required by model construction and verified by the model code. No weights are committed or downloaded automatically.

Edit resources/settings.json: data_root is relative to repository root (absolute path also accepted), device=cuda:0, split=test or valid. The default resources/data should reference authorized data, ideally by symlink. Expected folders: train/images, train/masks, valid/images, valid/masks, test/images. Do not duplicate shared dataset originals.

## Inference

```sh
python src/infer.py
```

Runs three models sequentially and then fixed soft voting. Input448, central224, stride112; D4 rotations/reflections with inverse transforms; average overlapping probabilities; weighted sum30/35/35; argmax; RGB masks at original resolution. Outputs: run/inference/ensemble/{split}/masks, with probabilities and metrics alongside. Validation20 and Test10 image counts are checked.

Individual model: `python src/model1/infer.py` (likewise model2/model3). Combine existing probabilities: `python src/ensemble.py`.

## Fresh training

```sh
python src/model1/train.py
python src/model2/train.py
python src/model3/train.py
```

Released seed42 recipes, AdamW/cosine, 300/300/150epochs. Models1/2 use public ImageNet encoder initialization; model3 is scratch. Output run/retraining/modelN must not already exist. Released checkpoints are preserved. Fresh retraining is optional and is not needed for inference; inference reproducibility does not imply bitwise-identical fresh training.

Model1: full448 supervision, boundary sampling, Cutout, gradient centralization, all-class Lovasz. Model2: full448 supervision, distinct-image batches, squared Dice. Model3: central224 supervision, ECA in outer RSU residual paths. Configs are unchanged from the released checkpoints; historical internal identifiers are retained only for compatibility. External names are model1/model2/model3.

## Source provenance and validation

Code is from the final submission package, with filesystem paths adapted to AGENTS.md roles. The copied model-specific implementation is kept in src; resources/licenses retains source attribution and U2-Net license. No dataset, masks, probability arrays, report or checkpoints are included.

Packaging validation: Python syntax, JSON parsing, entry-point path checks, and unchanged config/model source comparisons. Full GPU inference is not rerun for a path-only repository publication.
