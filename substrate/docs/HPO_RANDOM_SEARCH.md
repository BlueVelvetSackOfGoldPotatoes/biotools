# Hyperparameter Random Search Plan

This repo now supports environment-driven random search per architecture, orchestrated by:

- `scripts/random_search.py`

The script samples architecture-specific hyperparameters, runs the corresponding benchmark binary, reads `runs/<run_id>/learning/epoch_metrics.csv`, and optimizes for best observed test accuracy (or fallback metric when accuracy is unavailable).

## Model Coverage

The following benchmark families are covered:

- `mlp`: optimizer/loss/lr/batch/epochs + hidden widths (`MLP_*`)
- `cnn`: variant + optimizer/loss/lr/batch/epochs (`CNN_*`)
- `rnn`: variant + optimizer/loss/lr/batch/epochs + hidden sizes (`RNN_*`)
- `lstm`: variant + optimizer/loss/lr/batch/epochs + hidden sizes (`LSTM_*`)
- `transformer`: optimizer/loss/lr/batch/epochs + depth/heads/width/dropout (`TRANSFORMER_*`)
- `vit`: optimizer/loss/lr/batch/epochs + patch/embed/heads/depth/dropout (`VIT_*`)
- `hebbian`: Hebbian rule + Hebbian/probe lrs + probe optimizer/loss + feature width (`HEBBIAN_*`)
- `forward_forward`: lr/threshold/epochs/batch + hidden widths (`FF_*`)
- `gnn`: lr/epochs/layers/hidden + train/eval subset sizes (`GNN_*`)
- `diffusion`: lr/steps/batch/timesteps/hidden (`DIFFUSION_*`)
- `actor_critic`: shared steps/batch/hidden/gamma + A2C/PPO rates/clip (`ACTOR_CRITIC_*`, `A2C_*`, `PPO_*`)
- `reinforcement`: algorithm-specific search over tabular, DQN/DDQN, MuZero-lite knobs (`RL_*`)
- `markov`: Markov order/bins/smoothing/data schedule knobs (`MARKOV_*`)
- `tictactoe`: model-agnostic deep-Q search over architecture + bit-bridge + opponent curriculum knobs (`TICTACTOE_*`)
- `trees`: depth and split thresholds (`TREE_*`)
- `forests`: algorithm variant + trees/depth/split/sample ratio (`FORESTS_*`)
- `clustering`: algorithm/k/iterations/batch/sample sizes (`CLUSTERING_*`)
- `continuous`: online-cycle and hybrid composition knobs (`ONLINE_*`)

## Execution

Run all model families:

```bash
./scripts/random_search.py --models all --trials 8
```

Run selected families:

```bash
./scripts/random_search.py --models mlp,cnn,transformer --trials 12
```

Force global env overrides:

```bash
./scripts/random_search.py --models vit --trials 6 \
  --extra-env BIO_ACTIVE=1,EVAL_BATCH_SIZE=128
```

## Outputs

Each search invocation writes:

- `reports/hpo_<timestamp>/summary.csv`
- `reports/hpo_<timestamp>/best_configs.json`

Each benchmark trial still emits full run artifacts into `runs/<run_id>/...` for downstream analytics/dashboard use.
