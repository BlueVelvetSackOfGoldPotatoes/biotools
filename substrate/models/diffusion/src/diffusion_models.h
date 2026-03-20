#pragma once

#include "../../../core/nn/module.h"
#include "../../../core/losses/losses.h"
#include <random>
#include <vector>
#include <cmath>

// ============================================================
// DenoisingUNet
// MLP-based U-Net for denoising diffusion on flattened MNIST images.
//
// Architecture (MLP-based, not convolutional):
//   Encoder:
//     enc1: (input_size + time_embed_size) -> hidden_size   (downsampled representation)
//     enc2: hidden_size -> hidden_size/2                     (bottleneck)
//   Decoder:
//     dec2: hidden_size/2 -> hidden_size                     (+ skip from enc1 output)
//     dec1: hidden_size*2 -> hidden_size                     (skip concatenation doubles width)
//     final_layer: hidden_size -> input_size                  (predict noise)
//
// Skip connections: enc1 output is concatenated with dec2 output before dec1.
// Time conditioning: timestep embedding is concatenated with input before enc1.
// ============================================================
class DenoisingUNet {
    Linear enc1, enc2, dec2, dec1, final_layer;
    Linear time_embed;
    ReLU relu;
    size_t hidden_size;
    size_t input_size;
    size_t time_embed_size;

    // Cached activations for backward pass
    Tensor cached_input_with_t;   // (batch x (input_size + time_embed_size))
    Tensor cached_enc1_out;       // (batch x hidden_size) - before ReLU
    Tensor cached_enc1_relu;      // (batch x hidden_size) - after ReLU
    Tensor cached_enc2_out;       // (batch x hidden_size/2) - before ReLU
    Tensor cached_enc2_relu;      // (batch x hidden_size/2) - after ReLU
    Tensor cached_dec2_out;       // (batch x hidden_size) - before ReLU
    Tensor cached_dec2_relu;      // (batch x hidden_size) - after ReLU
    Tensor cached_skip_cat;       // (batch x hidden_size*2) - after skip concat
    Tensor cached_dec1_out;       // (batch x hidden_size) - before ReLU
    Tensor cached_dec1_relu;      // (batch x hidden_size) - after ReLU

    // ReLU layers for caching during backward (one per activation)
    ReLU relu_enc1, relu_enc2, relu_dec2, relu_dec1;

public:
    DenoisingUNet(size_t input_size, size_t hidden_size, size_t time_embed_size,
                  std::mt19937& rng);

    // Forward pass.
    // x: batch x input_size (noisy images)
    // t_embed: batch x time_embed_size (timestep embeddings)
    // Returns: batch x input_size (predicted noise)
    Tensor forward(const Tensor& x, const Tensor& t_embed);

    // Backward pass given gradient on output.
    // Returns gradient w.r.t. input x (gradient w.r.t. t_embed is discarded).
    Tensor backward(const Tensor& grad);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();
    void zero_grad();
};

// ============================================================
// DDPM (Denoising Diffusion Probabilistic Model)
// Implements the forward diffusion (noise addition) and reverse
// denoising process for generating MNIST-like images.
//
// Noise schedule: linear beta schedule from beta_start to beta_end.
// Training: sample random timestep t, add noise to x_0, predict noise.
// Sampling: iterative denoising from pure noise x_T to x_0.
// ============================================================
class DDPM {
    DenoisingUNet model;
    int num_timesteps;
    std::vector<double> betas;       // noise schedule: beta_t
    std::vector<double> alphas;      // 1 - beta_t
    std::vector<double> alpha_bars;  // cumulative product of alphas
    std::mt19937& rng;
    size_t input_size_;
    static constexpr size_t time_embed_size_ = 64;

    // Adam optimizer state
    std::vector<Tensor> adam_m_, adam_v_;
    int adam_t_;

public:
    DDPM(size_t input_size, size_t hidden_size, int num_timesteps,
         std::mt19937& rng);

    // Training step: add noise at random timestep, predict noise, compute MSE.
    // Returns the loss value. Performs one gradient descent step with given lr.
    double train_step(const Tensor& x_0, double lr);

    // Generate num_samples new images by iterative denoising from pure noise.
    // Returns: num_samples x input_size
    Tensor sample(int num_samples);

    // Compute sinusoidal timestep embedding.
    // timesteps: vector of integer timesteps (one per sample)
    // embed_dim: dimensionality of the embedding
    // Returns: len(timesteps) x embed_dim
    Tensor get_timestep_embedding(const std::vector<int>& timesteps,
                                  size_t embed_dim) const;

    // Compute the denoising loss on a batch of clean images (for evaluation).
    // Same as train_step but without parameter update.
    double denoising_loss(const Tensor& x_0);

    std::vector<Tensor*> parameters();
    std::vector<Tensor*> gradients();

    // Accessors for visualization
    const std::vector<double>& get_alpha_bars() const { return alpha_bars; }
    const std::vector<double>& get_betas() const { return betas; }
    int get_num_timesteps() const { return num_timesteps; }

    // Sample with trajectory: returns intermediate x_t states at each denoising step
    // Result[i] is x_t at step i (from pure noise to final output)
    std::vector<Tensor> sample_trajectory(int num_samples);
};
