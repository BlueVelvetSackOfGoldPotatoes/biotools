#include "diffusion_models.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <stdexcept>

// ============================================================
// DenoisingUNet
// ============================================================

DenoisingUNet::DenoisingUNet(size_t input_size, size_t hidden_size,
                             size_t time_embed_size, std::mt19937& rng)
    : enc1(input_size + time_embed_size, hidden_size, "unet_enc1", rng),
      enc2(hidden_size, hidden_size / 2, "unet_enc2", rng),
      dec2(hidden_size / 2, hidden_size, "unet_dec2", rng),
      dec1(hidden_size * 2, hidden_size, "unet_dec1", rng),  // *2 for skip concat
      final_layer(hidden_size, input_size, "unet_final", rng),
      time_embed(time_embed_size, time_embed_size, "unet_time_embed", rng),
      hidden_size(hidden_size),
      input_size(input_size),
      time_embed_size(time_embed_size) {
    if (input_size == 0 || hidden_size < 2 || time_embed_size == 0) {
        throw std::runtime_error("DenoisingUNet: input_size > 0, hidden_size >= 2, and time_embed_size > 0 are required");
    }
}

Tensor DenoisingUNet::forward(const Tensor& x, const Tensor& t_embed) {
    size_t batch = x.rows;
    if (x.cols != input_size) {
        throw std::runtime_error("DenoisingUNet::forward input width mismatch");
    }
    if (t_embed.rows != batch || t_embed.cols != time_embed_size) {
        throw std::runtime_error("DenoisingUNet::forward timestep embedding shape mismatch");
    }

    // Process time embedding through a linear layer + ReLU
    Tensor t_processed = time_embed.forward(t_embed);
    // (We don't apply ReLU to time embedding to keep it expressive)

    // Concatenate input with time embedding: batch x (input_size + time_embed_size)
    cached_input_with_t = Tensor(batch, input_size + time_embed_size);
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < input_size; ++j)
            cached_input_with_t(i, j) = x(i, j);
        for (size_t j = 0; j < time_embed_size; ++j)
            cached_input_with_t(i, input_size + j) = t_processed(i, j);
    }

    // Encoder path
    cached_enc1_out = enc1.forward(cached_input_with_t);    // batch x hidden_size
    cached_enc1_relu = relu_enc1.forward(cached_enc1_out);  // batch x hidden_size

    cached_enc2_out = enc2.forward(cached_enc1_relu);       // batch x hidden_size/2
    cached_enc2_relu = relu_enc2.forward(cached_enc2_out);  // batch x hidden_size/2

    // Decoder path
    cached_dec2_out = dec2.forward(cached_enc2_relu);       // batch x hidden_size
    cached_dec2_relu = relu_dec2.forward(cached_dec2_out);  // batch x hidden_size

    // Skip connection: concatenate enc1_relu and dec2_relu
    // enc1_relu: batch x hidden_size, dec2_relu: batch x hidden_size
    // result: batch x (hidden_size * 2)
    cached_skip_cat = Tensor(batch, hidden_size * 2);
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < hidden_size; ++j)
            cached_skip_cat(i, j) = cached_enc1_relu(i, j);
        for (size_t j = 0; j < hidden_size; ++j)
            cached_skip_cat(i, hidden_size + j) = cached_dec2_relu(i, j);
    }

    cached_dec1_out = dec1.forward(cached_skip_cat);        // batch x hidden_size
    cached_dec1_relu = relu_dec1.forward(cached_dec1_out);  // batch x hidden_size

    // Final projection to input_size (predicted noise)
    Tensor output = final_layer.forward(cached_dec1_relu);  // batch x input_size
    return output;
}

Tensor DenoisingUNet::backward(const Tensor& grad) {
    size_t batch = grad.rows;

    // Backward through final_layer
    Tensor grad_dec1_relu = final_layer.backward(grad);     // batch x hidden_size

    // Backward through relu_dec1
    Tensor grad_dec1_out = relu_dec1.backward(grad_dec1_relu); // batch x hidden_size

    // Backward through dec1 -> gives gradient w.r.t. cached_skip_cat
    Tensor grad_skip_cat = dec1.backward(grad_dec1_out);    // batch x (hidden_size*2)

    // Split gradient for skip connection
    // First hidden_size columns -> gradient to enc1_relu (skip path)
    // Last hidden_size columns -> gradient to dec2_relu
    Tensor grad_enc1_relu_skip(batch, hidden_size, 0.0);
    Tensor grad_dec2_relu(batch, hidden_size, 0.0);
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < hidden_size; ++j)
            grad_enc1_relu_skip(i, j) = grad_skip_cat(i, j);
        for (size_t j = 0; j < hidden_size; ++j)
            grad_dec2_relu(i, j) = grad_skip_cat(i, hidden_size + j);
    }

    // Backward through relu_dec2
    Tensor grad_dec2_out = relu_dec2.backward(grad_dec2_relu); // batch x hidden_size

    // Backward through dec2
    Tensor grad_enc2_relu = dec2.backward(grad_dec2_out);   // batch x hidden_size/2

    // Backward through relu_enc2
    Tensor grad_enc2_out = relu_enc2.backward(grad_enc2_relu); // batch x hidden_size/2

    // Backward through enc2
    Tensor grad_enc1_relu_main = enc2.backward(grad_enc2_out); // batch x hidden_size

    // Accumulate gradients from skip connection and encoder path at enc1_relu
    Tensor grad_enc1_relu_total = grad_enc1_relu_main + grad_enc1_relu_skip;

    // Backward through relu_enc1
    Tensor grad_enc1_out = relu_enc1.backward(grad_enc1_relu_total); // batch x hidden_size

    // Backward through enc1 -> gives gradient w.r.t. cached_input_with_t
    Tensor grad_input_with_t = enc1.backward(grad_enc1_out); // batch x (input_size + time_embed_size)

    // Backward through time_embed for the time portion
    Tensor grad_t_processed(batch, time_embed_size);
    for (size_t i = 0; i < batch; ++i)
        for (size_t j = 0; j < time_embed_size; ++j)
            grad_t_processed(i, j) = grad_input_with_t(i, input_size + j);
    time_embed.backward(grad_t_processed);

    // Extract gradient w.r.t. input x (first input_size columns)
    Tensor grad_x(batch, input_size);
    for (size_t i = 0; i < batch; ++i)
        for (size_t j = 0; j < input_size; ++j)
            grad_x(i, j) = grad_input_with_t(i, j);

    return grad_x;
}

std::vector<Tensor*> DenoisingUNet::parameters() {
    std::vector<Tensor*> params;
    for (auto* p : enc1.parameters()) params.push_back(p);
    for (auto* p : enc2.parameters()) params.push_back(p);
    for (auto* p : dec2.parameters()) params.push_back(p);
    for (auto* p : dec1.parameters()) params.push_back(p);
    for (auto* p : final_layer.parameters()) params.push_back(p);
    for (auto* p : time_embed.parameters()) params.push_back(p);
    return params;
}

std::vector<Tensor*> DenoisingUNet::gradients() {
    std::vector<Tensor*> grads;
    for (auto* g : enc1.gradients()) grads.push_back(g);
    for (auto* g : enc2.gradients()) grads.push_back(g);
    for (auto* g : dec2.gradients()) grads.push_back(g);
    for (auto* g : dec1.gradients()) grads.push_back(g);
    for (auto* g : final_layer.gradients()) grads.push_back(g);
    for (auto* g : time_embed.gradients()) grads.push_back(g);
    return grads;
}

void DenoisingUNet::zero_grad() {
    auto grads = gradients();
    for (auto* g : grads)
        g->fill_zeros();
}

// ============================================================
// DDPM
// ============================================================

DDPM::DDPM(size_t input_size, size_t hidden_size, int num_timesteps,
           std::mt19937& rng)
    : model(input_size, hidden_size, time_embed_size_, rng),
      num_timesteps(num_timesteps), rng(rng),
      input_size_(input_size), adam_t_(0) {
    if (num_timesteps <= 0) {
        throw std::runtime_error("DDPM: num_timesteps must be > 0");
    }
    // Cosine noise schedule (Nichol & Dhariwal, 2021)
    // Provides more uniform noise levels than linear schedule
    double s = 0.008;
    double f_0 = std::cos(s / (1.0 + s) * M_PI / 2.0);
    f_0 = f_0 * f_0;

    betas.resize(num_timesteps);
    alphas.resize(num_timesteps);
    alpha_bars.resize(num_timesteps);

    // Compute alpha_bars directly from cosine formula
    for (int t = 0; t < num_timesteps; ++t) {
        double arg = ((double)(t + 1) / num_timesteps + s) / (1.0 + s) * M_PI / 2.0;
        double f_t = std::cos(arg);
        alpha_bars[t] = (f_t * f_t) / f_0;
    }

    // Derive betas from alpha_bars
    betas[0] = std::min(1.0 - alpha_bars[0], 0.999);
    for (int t = 1; t < num_timesteps; ++t)
        betas[t] = std::min(1.0 - alpha_bars[t] / alpha_bars[t - 1], 0.999);

    for (int t = 0; t < num_timesteps; ++t)
        alphas[t] = 1.0 - betas[t];
}

Tensor DDPM::get_timestep_embedding(const std::vector<int>& timesteps,
                                    size_t embed_dim) const {
    // Sinusoidal positional embedding (similar to transformer positional encoding)
    size_t batch = timesteps.size();
    Tensor embedding(batch, embed_dim, 0.0);
    size_t half_dim = embed_dim / 2;

    for (size_t i = 0; i < batch; ++i) {
        double t = (double)timesteps[i];
        for (size_t d = 0; d < half_dim; ++d) {
            double freq = std::exp(-(double)d * std::log(10000.0) / (double)half_dim);
            embedding(i, d) = std::sin(t * freq);
            embedding(i, half_dim + d) = std::cos(t * freq);
        }
        // If embed_dim is odd, leave the last position as 0
    }
    return embedding;
}

double DDPM::train_step(const Tensor& x_0, double lr) {
    size_t batch = x_0.rows;
    if (x_0.cols != input_size_) {
        throw std::runtime_error("DDPM::train_step input width mismatch");
    }

    // 1. Stratified timestep sampling: each timestep appears equally often
    std::vector<int> timesteps(batch);
    for (size_t i = 0; i < batch; ++i)
        timesteps[i] = (int)(i % num_timesteps);

    // 2. Sample noise epsilon ~ N(0, I)
    Tensor epsilon(batch, input_size_);
    epsilon.fill_random_normal(0.0, 1.0, rng);

    // 3. Compute noisy images: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    Tensor x_t(batch, input_size_);
    for (size_t i = 0; i < batch; ++i) {
        double sqrt_ab = std::sqrt(alpha_bars[timesteps[i]]);
        double sqrt_one_minus_ab = std::sqrt(1.0 - alpha_bars[timesteps[i]]);
        for (size_t j = 0; j < input_size_; ++j)
            x_t(i, j) = sqrt_ab * x_0(i, j) + sqrt_one_minus_ab * epsilon(i, j);
    }

    // 4. Get timestep embeddings
    Tensor t_embed = get_timestep_embedding(timesteps, time_embed_size_);

    // 5. Forward pass: predict clean image x_0 (not epsilon)
    //    x_0 prediction has much better gradient signal than epsilon prediction
    //    because the target (clean image) has non-zero mean, avoiding the
    //    "predict zero" equilibrium trap of epsilon prediction.
    model.zero_grad();
    Tensor x0_pred = model.forward(x_t, t_embed);

    // 6. Compute per-element MSE loss: ||x0_pred - x_0||^2
    double loss = 0.0;
    size_t total = batch * input_size_;
    for (size_t i = 0; i < total; ++i) {
        double diff = x0_pred.data[i] - x_0.data[i];
        loss += diff * diff;
    }
    loss /= (double)total;

    // 7. Compute gradient of MSE w.r.t. x0_pred
    //    Loss = sum((x0_pred - x_0)^2) / (batch * input_size_)
    //    d(Loss)/d(x0_pred_i) = 2 * (x0_pred_i - x_0_i) / (batch * input_size_)
    Tensor grad(batch, input_size_);
    for (size_t i = 0; i < total; ++i)
        grad.data[i] = 2.0 * (x0_pred.data[i] - x_0.data[i]) / (double)total;

    // 8. Backward pass
    model.backward(grad);

    // 9. Gradient clipping (max norm per parameter = 1.0)
    auto params = model.parameters();
    auto grads = model.gradients();
    for (size_t p = 0; p < grads.size(); ++p) {
        double norm_sq = 0;
        for (size_t k = 0; k < grads[p]->data.size(); ++k)
            norm_sq += grads[p]->data[k] * grads[p]->data[k];
        double norm = std::sqrt(norm_sq);
        if (norm > 1.0) {
            for (size_t k = 0; k < grads[p]->data.size(); ++k)
                grads[p]->data[k] /= norm;
        }
    }

    // 10. Adam parameter update
    adam_t_++;
    if (adam_m_.empty()) {
        for (auto* p : params) {
            adam_m_.push_back(Tensor::zeros(p->rows, p->cols));
            adam_v_.push_back(Tensor::zeros(p->rows, p->cols));
        }
    }
    for (size_t p = 0; p < params.size(); ++p) {
        for (size_t k = 0; k < params[p]->data.size(); ++k) {
            double g = grads[p]->data[k];
            adam_m_[p].data[k] = 0.9 * adam_m_[p].data[k] + 0.1 * g;
            adam_v_[p].data[k] = 0.999 * adam_v_[p].data[k] + 0.001 * g * g;
            double mh = adam_m_[p].data[k] / (1.0 - std::pow(0.9, adam_t_));
            double vh = adam_v_[p].data[k] / (1.0 - std::pow(0.999, adam_t_));
            params[p]->data[k] -= lr * mh / (std::sqrt(vh) + 1e-8);
        }
    }

    return loss;
}

double DDPM::denoising_loss(const Tensor& x_0) {
    size_t batch = x_0.rows;
    if (x_0.cols != input_size_) {
        throw std::runtime_error("DDPM::denoising_loss input width mismatch");
    }

    // Stratified timestep sampling for stable eval
    std::vector<int> timesteps(batch);
    for (size_t i = 0; i < batch; ++i)
        timesteps[i] = (int)(i % num_timesteps);

    // Sample noise
    Tensor epsilon(batch, input_size_);
    epsilon.fill_random_normal(0.0, 1.0, rng);

    // Compute noisy images
    Tensor x_t(batch, input_size_);
    for (size_t i = 0; i < batch; ++i) {
        double sqrt_ab = std::sqrt(alpha_bars[timesteps[i]]);
        double sqrt_one_minus_ab = std::sqrt(1.0 - alpha_bars[timesteps[i]]);
        for (size_t j = 0; j < input_size_; ++j)
            x_t(i, j) = sqrt_ab * x_0(i, j) + sqrt_one_minus_ab * epsilon(i, j);
    }

    // Get timestep embeddings
    Tensor t_embed = get_timestep_embedding(timesteps, time_embed_size_);

    // Forward pass: predict x_0
    Tensor x0_pred = model.forward(x_t, t_embed);

    // Compute x_0 prediction MSE
    double loss = 0.0;
    size_t total = batch * input_size_;
    for (size_t i = 0; i < total; ++i) {
        double diff = x0_pred.data[i] - x_0.data[i];
        loss += diff * diff;
    }
    loss /= (double)total;

    return loss;
}

std::vector<Tensor*> DDPM::parameters() {
    return model.parameters();
}

std::vector<Tensor*> DDPM::gradients() {
    return model.gradients();
}

Tensor DDPM::sample(int num_samples) {
    // Start from pure noise
    Tensor x_t(num_samples, input_size_);
    x_t.fill_random_normal(0.0, 1.0, rng);

    // Iterative denoising: t = T-1, T-2, ..., 0
    for (int t = num_timesteps - 1; t >= 0; --t) {
        std::vector<int> timesteps(num_samples, t);
        Tensor t_embed = get_timestep_embedding(timesteps, time_embed_size_);

        // Predict clean image x_0
        Tensor x0_pred = model.forward(x_t, t_embed);

        if (t == 0) {
            // Final step: directly use predicted clean image
            x_t = x0_pred;
        } else {
            // Use posterior mean formula:
            // mu = sqrt(ab_{t-1}) * beta_t / (1 - ab_t) * x0_pred
            //    + sqrt(alpha_t) * (1 - ab_{t-1}) / (1 - ab_t) * x_t
            // sigma = sqrt(beta_t * (1 - ab_{t-1}) / (1 - ab_t))
            double ab_t = alpha_bars[t];
            double ab_t_prev = alpha_bars[t - 1];
            double beta_t = betas[t];
            double alpha_t = alphas[t];

            double coeff_x0 = std::sqrt(ab_t_prev) * beta_t / (1.0 - ab_t);
            double coeff_xt = std::sqrt(alpha_t) * (1.0 - ab_t_prev) / (1.0 - ab_t);
            double sigma_t = std::sqrt(beta_t * (1.0 - ab_t_prev) / (1.0 - ab_t));

            Tensor z(num_samples, input_size_);
            z.fill_random_normal(0.0, 1.0, rng);

            for (size_t i = 0; i < (size_t)num_samples; ++i) {
                for (size_t j = 0; j < input_size_; ++j) {
                    x_t(i, j) = coeff_x0 * x0_pred(i, j) + coeff_xt * x_t(i, j)
                                + sigma_t * z(i, j);
                }
            }
        }
    }

    // Clamp output to valid range [0, 1]
    for (size_t i = 0; i < x_t.data.size(); ++i)
        x_t.data[i] = std::clamp(x_t.data[i], 0.0, 1.0);

    return x_t;
}

std::vector<Tensor> DDPM::sample_trajectory(int num_samples) {
    std::vector<Tensor> trajectory;

    // Start from pure noise
    Tensor x_t(num_samples, input_size_);
    x_t.fill_random_normal(0.0, 1.0, rng);
    trajectory.push_back(x_t);  // t=T (pure noise)

    // Iterative denoising: t = T-1, T-2, ..., 0
    for (int t = num_timesteps - 1; t >= 0; --t) {
        std::vector<int> timesteps(num_samples, t);
        Tensor t_embed = get_timestep_embedding(timesteps, time_embed_size_);
        Tensor x0_pred = model.forward(x_t, t_embed);

        if (t == 0) {
            x_t = x0_pred;
        } else {
            double ab_t = alpha_bars[t];
            double ab_t_prev = alpha_bars[t - 1];
            double beta_t = betas[t];
            double alpha_t = alphas[t];

            double coeff_x0 = std::sqrt(ab_t_prev) * beta_t / (1.0 - ab_t);
            double coeff_xt = std::sqrt(alpha_t) * (1.0 - ab_t_prev) / (1.0 - ab_t);
            double sigma_t = std::sqrt(beta_t * (1.0 - ab_t_prev) / (1.0 - ab_t));

            Tensor z(num_samples, input_size_);
            z.fill_random_normal(0.0, 1.0, rng);

            for (size_t i = 0; i < (size_t)num_samples; ++i) {
                for (size_t j = 0; j < input_size_; ++j) {
                    x_t(i, j) = coeff_x0 * x0_pred(i, j) + coeff_xt * x_t(i, j)
                                + sigma_t * z(i, j);
                }
            }
        }
        trajectory.push_back(x_t);  // after denoising step t
    }

    // Clamp final output
    auto& final_state = trajectory.back();
    for (size_t i = 0; i < final_state.data.size(); ++i)
        final_state.data[i] = std::clamp(final_state.data[i], 0.0, 1.0);

    return trajectory;
}
