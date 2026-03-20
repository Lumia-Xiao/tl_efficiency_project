# Power Converter Efficiency Transfer Learning Project

This PyCharm-ready project implements **component prediction + sum constraint** for converter loss estimation.

## Problem setting

- **Source domain**: abundant simulation/calculation data with component-level loss labels
  - Inputs: `Vlv, Vhv, D, fsw, deadtime_s, Pout`
  - Component labels: `PCond, PSw, Pcore, PCopp`
  - Total loss: `PTotal`
- **Target domain**: limited experimental data with only total loss label
  - Inputs: `Vlv, Vhv, D, fsw, deadtime_s, Pout`
  - Total label: `PTotal`

The model predicts each source-domain loss component and enforces:

```math
\hat P_{Total}=\hat P_{Cond}+\hat P_{Sw}+\hat P_{Core}+\hat P_{Copp}
```

During target-domain training, only the total loss is supervised, while the latent 4-component decomposition is additionally regularized by a source-domain relation prior in compositional (`clr/log-ratio`) space.

## Structure

- `src/config.py`: hyperparameters and column definitions
- `src/data_utils.py`: loading, scaling, dataset builders
- `src/model.py`: multi-head MLP with non-negative component outputs
- `src/train.py`: source pretraining + target weakly supervised fine-tuning
- `src/experiments.py`: baseline and ablation experiments for method comparison
- `src/evaluate.py`: metrics and plotting helpers
- `data/`: CSV files
- `outputs/`: checkpoints, metrics, figures
- `src/gui_app.py`: Tkinter GUI that takes Vlv/Vhv/D/fsw/deadtime_s/Pout and predicts PCond/PSw/Pcore/PCopp/PTotal for simulation and experiment checkpoints, with artifact path selectors when defaults are missing.
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
python -m src.train --epochs-pretrain 300 --epochs-finetune 100 --lr 1e-3 --batch-size 32
```

Run baseline + ablations:

```bash
python -m src.experiments --output outputs/experiments/metrics.json
```

Retest the comparison with a random 20-sample target-domain subset:

```bash
python -m src.experiments --target-subset-size 20 --seed 42 --output outputs/experiments/metrics_target20.json
```

Launch GUI for single-sample prediction (both simulation and experiment domains):

```bash
python -m src.gui_app
```

Optional: pass artifact paths explicitly (useful if `outputs/` is elsewhere):

```bash
python -m src.gui_app --scaler-path path/to/scaler.joblib --simulation-ckpt path/to/best_pretrained.pt --experiment-ckpt path/to/best_finetuned.pt
```

The GUI now starts even if artifacts are missing; use the built-in file pickers and **Load Artifacts** button.

If your IDE runs files directly (for example PyCharm), this also works:

```bash
python src/experiments.py --output outputs/experiments/metrics.json
```

Included comparisons:
- `proposed`: full method with source component supervision, source relation learning, and target relation-prior consistency.
- `baseline_target_only_total_model`: direct total-loss MLP trained only on target data.
- `ablation_no_source_transfer`: constrained component-sum model trained on target total labels only, without any source-domain learning.
- `source_only_no_target_adaptation`: source-pretrained model evaluated on target without target-domain fine-tuning.
- `ablation_no_component_supervision`: transfer pipeline with source component loss weight set to 0.
- `ablation_no_source_relation`: keeps source component labels but removes source-domain relation-consistency learning.
- `ablation_no_relation_learning`: removes both source relation learning and target relation prior, leaving only component/value supervision plus sum constraint.

## Notes

1. The scaler is fit on the **source-domain training split only**.
2. Source-domain validation is used to select the best pretrained model.
3. Source-domain training now also learns the 4-way component relation with a `clr`-space consistency loss, so the model captures co-variation and trade-offs rather than treating components as independent targets.
4. Target-domain fine-tuning still uses total-loss supervision only, but it also matches the source-domain component relation prior through `clr` mean/covariance and average share consistency.
5. Component predictions in the target domain are latent but physically constrained to be non-negative, additive, and relation-consistent.
