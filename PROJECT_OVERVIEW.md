# Project Readthrough Notes

## What this repository does
This project trains a transfer-learning model for converter loss estimation where:
- The **source domain** has abundant component-level labels (`PIron`, `PCond`, `PCopp`, `PSw`) and total loss (`Ploss`).
- The **target domain** has limited data with only total loss (`Ploss`).

The model predicts component losses and computes total loss as their sum, so the physics-inspired additive constraint is always enforced.

## End-to-end pipeline
1. Load source/target CSV files.
2. Split both domains into train/validation sets.
3. Fit a `StandardScaler` using only source-train inputs.
4. Build dataloaders for source (with component + total labels) and target (total-only labels).
5. Train in two stages:
   - **Stage 1 pretraining** on source data with component and total objectives.
   - **Stage 2 fine-tuning** with mixed source+target batches, supervising target only on total loss.
6. Save checkpoints, metrics, training history, and evaluation plots.

## Core modules
- `src/config.py`: central hyperparameters, column definitions, split ratios, and training weights.
- `src/data_utils.py`: CSV loading, domain splits, scaling policy, `Dataset`/`DataLoader` construction, scaler persistence.
- `src/model.py`: MLP backbone and component head with `Softplus` non-negativity; total prediction is computed by summation.
- `src/train.py`: CLI, seed setup, source pretraining loop, source+target fine-tuning loop, early stopping, artifact writing.
- `src/evaluate.py`: regression metrics, batch prediction collection, JSON metrics persistence, and plotting utilities.

## Notable design choices
- **Constraint by construction**: `total = sum(components)` is implemented in the forward pass.
- **Non-negative components**: `Softplus` ensures each component prediction is >= 0.
- **Source-first scaling**: avoids leakage by fitting the scaler on source-train only.
- **Weak supervision for target**: target data contributes only through total-loss labels.
- **Optional backbone freezing**: `--freeze-backbone-in-finetune` supports head-only adaptation.

## How to run
Install dependencies:

```bash
pip install -r requirements.txt
```

Train with defaults:

```bash
python -m src.train
```

Override key hyperparameters:

```bash
python -m src.train --epochs-pretrain 300 --epochs-finetune 400 --lr 1e-3 --batch-size 16
```
