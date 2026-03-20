# Analytics, Logging, and Dashboard

## Run Logging Layout

Each training/deployment run writes to:

- `runs/<run_id>/manifest.json`
- `runs/<run_id>/learning/epoch_metrics.csv`
- `runs/<run_id>/learning/class_metrics.csv`
- `runs/<run_id>/learning/confusion_matrix.csv`
- `runs/<run_id>/deployment/inference_metrics.csv`
- `runs/<run_id>/deployment/calibration_bins.csv`
- `runs/<run_id>/deployment/system_metrics.csv`
- `runs/<run_id>/model_specific/<model_family>/*.csv`

All benchmark binaries in `benchmarks/` now emit the common schema plus their own
`model_specific/<family>/...` CSVs.

## Plotting

Install dependencies:

```bash
pip install -r analytics/requirements.txt
```

Generate all common + model-specific plots for one run:

```bash
python analytics/plots/common/plot_bundle.py \
  --run_dir runs/<run_id> \
  --out_dir reports/<run_id> \
  --format png
```

Generate via model-specific entrypoint (example MLP):

```bash
python analytics/plots/mlp/plot_mlp.py \
  --run_dir runs/<run_id> \
  --out_dir reports/<run_id>/mlp \
  --format png
```

## Dashboard

Run:

```bash
streamlit run analytics/dashboard/app.py
```

React dashboard (includes live run status, training/evaluation charts, model-specific CSV visuals, and report image gallery):

```bash
cd ui
npm install
npm run dev
```

Open `http://localhost:5173`.

Pages:

- Overview
- Training
- Deployment
- Model-Specific
- Compare Runs
- Data/Drift

## Model-Specific CSV Conventions

Expected paths for optional model-specific logs:

- `model_specific/mlp/neuron_stats.csv`
- `model_specific/cnn/filter_stats.csv`, `featuremap_stats.csv`
- `model_specific/rnn/timestep_stats.csv`
- `model_specific/lstm/gate_stats.csv`
- `model_specific/transformer/attention_entropy.csv`, `attn_weights_sampled.csv`
- `model_specific/vit/patch_attention.csv`, `cls_embedding_stats.csv`
- `model_specific/trees/tree_growth.csv`, `split_importance.csv`
- `model_specific/forests/oob_curve.csv`, `tree_agreement.csv`
- `model_specific/clustering/cluster_metrics.csv`, `centroid_shift.csv`
- `model_specific/hebbian/hebb_update.csv`, `energy_curve.csv`
- `model_specific/actor_critic/rl_train.csv`
- `model_specific/reinforcement/rl_algorithms.csv`, `muzero_search.csv`
- `model_specific/diffusion/timestep_loss.csv`, `sample_stats.csv`
- `model_specific/gnn/message_norms.csv`, `embedding_2d.csv`
- `model_specific/forward_forward/goodness_stats.csv`
- `model_specific/continuous/continuous_phase_metrics.csv`, `continuous_efficiency.csv`, `continuous_summary.csv`
- `model_specific/hybrid/continuous_phase_metrics.csv`, `continuous_efficiency.csv`, `hybrid_expert_metrics.csv` (`fusion_strength`, `expert_accuracy`), `continuous_summary.csv`
- `model_specific/<model_family>/bio_epoch_dynamics.csv`, `bio_layer_dynamics.csv` (biology addon telemetry)
