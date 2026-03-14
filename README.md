# Transformer Efficiency Transfer Learning Project

This PyCharm-ready project implements **component prediction + sum constraint** for converter loss estimation.

## Problem setting

- **Source domain**: abundant simulation/calculation data with component-level loss labels
  - Inputs: `Vin, Vo, D1, D2, DT, Fs, Po`
  - Component labels: `PIron, PCond, PCopp, PSw`
  - Total loss: `Ploss`
- **Target domain**: limited experimental data with only total loss label
  - Inputs: `Vin, Vo, D1, D2, DT, Fs, Po`
  - Total label: `Ploss`

The model predicts each source-domain loss component and enforces:

```math
\hat P_{loss}=\hat P_{Iron}+\hat P_{Cond}+\hat P_{Copp}+\hat P_{Sw}
```

During target-domain training, only the total loss is supervised.

## Structure

- `src/config.py`: hyperparameters and column definitions
- `src/data_utils.py`: loading, scaling, dataset builders
- `src/model.py`: multi-head MLP with non-negative component outputs
- `src/train.py`: source pretraining + target weakly supervised fine-tuning
- `src/experiments.py`: baseline and ablation experiments for method comparison
- `src/evaluate.py`: metrics and plotting helpers
- `data/`: CSV files
- `outputs/`: checkpoints, metrics, figures

## Run

Create a virtual environment and install:

```bash
pip install -r requirements.txt
```

Train:

```bash
python -m src.train
```

Optional arguments:

```bash
python -m src.train --epochs-pretrain 300 --epochs-finetune 400 --lr 1e-3 --batch-size 256
```

Run baseline + ablations:

```bash
python -m src.experiments --output outputs/experiments/metrics.json
```

Included comparisons:
- `full_method`: source pretraining + source/target fine-tuning with component supervision.
- `baseline_target_only_total_model`: direct total-loss MLP trained only on target data.
- `ablation_no_component_supervision`: transfer pipeline with source component loss weight set to 0.
- `ablation_no_source_transfer`: constrained model trained only on target total labels.

## Notes

1. The scaler is fit on the **source-domain training split only**.
2. Source-domain validation is used to select the best pretrained model.
3. Target-domain fine-tuning uses only total-loss supervision.
4. Component predictions in the target domain are latent but physically constrained to be non-negative and additive.
