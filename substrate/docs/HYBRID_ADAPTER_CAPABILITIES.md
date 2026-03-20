# Hybrid Adapter Capabilities

`models/hybrid/src/model_registry.*` exposes adapters for model families that implement the shared differentiable `TrainableModel` contract used by hybrid composition and the bit-bridge runtimes.

## Adapter-supported families

- `mlp`
- `cnn`
- `rnn`
- `gru`
- `lstm`
- `transformer`
- `vit`
- `hebbian`
- `actor_critic`

## Non-adapter families (intentional)

These families currently use separate training/runtime contracts and are not available through the hybrid adapter path:

- `markov`
- `reinforcement`
- `gnn`
- `forward_forward`
- `diffusion`
- `trees`
- `forests`
- `clustering`
- `continuous`

If a model name from this list is requested through `create_model_adapter`, the runtime now returns an explicit error listing both supported adapters and non-adapter families.
