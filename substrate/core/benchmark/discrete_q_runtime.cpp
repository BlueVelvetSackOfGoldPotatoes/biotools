#include "core/benchmark/discrete_q_runtime.h"

#include "core/benchmark/bit_codec.h"
#include "core/losses/losses.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace benchmark {

namespace {

struct PolicyStats {
    int action = 0;
    double confidence = 0.0;
    double entropy = 0.0;
    double top2_margin = 0.0;
};

PolicyStats policy_stats_from_legal_q(const Tensor& q, const std::vector<int>& legal_actions) {
    if (q.rows != 1) {
        throw std::runtime_error("policy_stats_from_legal_q expects a single-row tensor");
    }
    if (legal_actions.empty()) {
        throw std::runtime_error("policy_stats_from_legal_q requires non-empty legal actions");
    }

    std::vector<double> logits;
    logits.reserve(legal_actions.size());
    for (const int action : legal_actions) {
        logits.push_back(q(0, static_cast<std::size_t>(action)));
    }

    const double max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probs(logits.size(), 0.0);
    double denom = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        denom += probs[i];
    }
    denom = std::max(denom, 1e-12);
    for (double& p : probs) p /= denom;

    std::size_t best_idx = 0;
    double best_p = probs[0];
    double second_p = 0.0;
    double entropy = 0.0;

    for (std::size_t i = 0; i < probs.size(); ++i) {
        const double p = probs[i];
        entropy -= p * std::log(p + 1e-12);
        if (p > best_p) {
            second_p = best_p;
            best_p = p;
            best_idx = i;
        } else if (p > second_p) {
            second_p = p;
        }
    }

    PolicyStats out;
    out.action = legal_actions[best_idx];
    out.confidence = best_p;
    out.entropy = entropy;
    out.top2_margin = best_p - second_p;
    return out;
}

void sync_action_owner_map(TrainableModel& model, const DiscreteEnv& env) {
    auto* aware = dynamic_cast<ActionOwnerAwareModel*>(&model);
    if (!aware) return;
    const std::vector<int> owner_map = env.action_owner_groups();
    if (owner_map.empty()) return;
    aware->set_action_owner_map(owner_map, std::max(1, env.num_action_owner_groups()));
}

} // namespace

DiscreteQRuntime::DiscreteQRuntime(DiscreteQConfig cfg, int seed) : cfg_(cfg), rng_(seed), global_step_(0) {}

int DiscreteQRuntime::select_action_epsilon_greedy(const Tensor& q,
                                                   const std::vector<int>& legal_actions,
                                                   double epsilon,
                                                   std::mt19937& rng) {
    if (legal_actions.empty()) return 0;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    if (u01(rng) < epsilon) {
        std::uniform_int_distribution<std::size_t> dist(0, legal_actions.size() - 1);
        return legal_actions[dist(rng)];
    }
    return select_action_greedy(q, legal_actions);
}

int DiscreteQRuntime::select_action_greedy(const Tensor& q, const std::vector<int>& legal_actions) {
    if (legal_actions.empty()) return 0;
    int best_action = legal_actions.front();
    double best_q = q(0, static_cast<std::size_t>(best_action));
    for (std::size_t i = 1; i < legal_actions.size(); ++i) {
        const int action = legal_actions[i];
        const double v = q(0, static_cast<std::size_t>(action));
        if (v > best_q) {
            best_q = v;
            best_action = action;
        }
    }
    return best_action;
}

double DiscreteQRuntime::max_legal_q(const Tensor& q, const std::vector<int>& legal_actions) {
    if (legal_actions.empty()) return 0.0;
    double best_q = q(0, static_cast<std::size_t>(legal_actions.front()));
    for (std::size_t i = 1; i < legal_actions.size(); ++i) {
        best_q = std::max(best_q, q(0, static_cast<std::size_t>(legal_actions[i])));
    }
    return best_q;
}

DiscreteEvalMetrics DiscreteQRuntime::evaluate(DiscreteEnv& env,
                                               TrainableModel& model,
                                               int episodes,
                                               bool collect_inference) {
    DiscreteEvalMetrics out;
    out.episodes = std::max(0, episodes);

    int sample_id = 0;
    int teacher_count = 0;
    int teacher_matches = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    model.eval();

    for (int episode = 0; episode < episodes; ++episode) {
        std::vector<uint8_t> obs = env.reset(rng_);
        int episode_outcome = 0;
        int episode_plies = 0;

        for (int step = 0; step < cfg_.max_steps_per_episode; ++step) {
            const std::vector<int> legal = env.legal_actions();
            if (legal.empty()) break;

            const int teacher = env.teacher_action();
            const auto start = std::chrono::high_resolution_clock::now();
            const Tensor state = bitcodec::bits_to_tensor_row(obs);
            sync_action_owner_map(model, env);
            const Tensor q = model.forward(state);
            const PolicyStats policy = policy_stats_from_legal_q(q, legal);
            const auto end = std::chrono::high_resolution_clock::now();
            const double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();

            const int pred = policy.action;
            const int immediate_win = env.tactical_immediate_win_action();
            const int immediate_block = env.tactical_immediate_block_action();
            if (immediate_win >= 0) {
                out.win_opportunities++;
                if (pred != immediate_win) out.missed_wins++;
            } else if (immediate_block >= 0) {
                out.block_opportunities++;
                if (pred != immediate_block) out.missed_blocks++;
            }

            const int truth = (teacher >= 0 && teacher < env.num_actions()) ? teacher : pred;

            out.preds.push_back(pred);
            out.truths.push_back(truth);
            out.moves++;
            out.avg_latency_ms += latency_ms;
            if (teacher >= 0 && teacher < env.num_actions()) {
                teacher_count++;
                if (pred == teacher) teacher_matches++;
            }

            if (collect_inference) {
                out.inference.push_back(
                    DiscreteInferenceSample{
                        sample_id++,
                        truth,
                        pred,
                        policy.confidence,
                        policy.entropy,
                        policy.top2_margin,
                        latency_ms,
                        pred == truth ? 1 : 0
                    }
                );
            }

            const DiscreteStepResult step_result = env.step(pred, rng_);
            if (step_result.invalid_action) out.invalid_actions++;
            episode_plies += step_result.plies;
            obs = step_result.observation_bits;

            if (step_result.done) {
                episode_outcome = step_result.outcome;
                break;
            }
        }

        out.total_plies += episode_plies;
        if (episode_outcome > 0) out.wins++;
        else if (episode_outcome < 0) out.losses++;
        else out.draws++;
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    out.eval_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    const double episodes_d = static_cast<double>(std::max(1, episodes));
    const double moves_d = static_cast<double>(std::max(1, out.moves));
    out.win_rate = out.wins / episodes_d;
    out.draw_rate = out.draws / episodes_d;
    out.loss_rate = out.losses / episodes_d;
    out.non_loss_rate = (out.wins + out.draws) / episodes_d;
    out.invalid_rate = out.invalid_actions / moves_d;
    out.avg_plies = out.total_plies / episodes_d;
    out.avg_latency_ms = out.avg_latency_ms / moves_d;
    if (teacher_count > 0) {
        out.action_match_rate = static_cast<double>(teacher_matches) / static_cast<double>(teacher_count);
    }
    if (out.win_opportunities > 0) {
        out.win_capture_rate =
            static_cast<double>(out.win_opportunities - out.missed_wins) / static_cast<double>(out.win_opportunities);
        out.missed_win_rate = static_cast<double>(out.missed_wins) / static_cast<double>(out.win_opportunities);
    }
    if (out.block_opportunities > 0) {
        out.block_capture_rate = static_cast<double>(out.block_opportunities - out.missed_blocks) /
                                 static_cast<double>(out.block_opportunities);
        out.missed_block_rate = static_cast<double>(out.missed_blocks) / static_cast<double>(out.block_opportunities);
    }

    return out;
}

DiscreteEvalMetrics DiscreteQRuntime::train(DiscreteEnv& env_train,
                                            DiscreteEnv& env_eval,
                                            TrainableModel& model,
                                            Adam& optimizer,
                                            const BatchCallback& on_batch,
                                            const EvalCallback& on_eval,
                                            const TrainHooks& hooks) {
    MSELoss q_loss;
    const auto train_start = std::chrono::steady_clock::now();

    model.train();
    double epsilon = cfg_.epsilon_start;
    double reward_ema = 0.0;
    double td_error_ema = 0.0;
    double q_loss_ema = 0.0;
    double action_match_ema = std::numeric_limits<double>::quiet_NaN();
    double missed_win_rate_ema = std::numeric_limits<double>::quiet_NaN();
    double missed_block_rate_ema = std::numeric_limits<double>::quiet_NaN();

    DiscreteEvalMetrics last_eval;
    bool have_last_eval = false;

    for (int episode = 1; episode <= cfg_.episodes; ++episode) {
        if (hooks.on_episode_begin) hooks.on_episode_begin(episode);
        std::vector<uint8_t> obs = env_train.reset(rng_);
        double episode_reward = 0.0;
        int match_count = 0;
        int teacher_count = 0;
        int episode_win_opportunities = 0;
        int episode_missed_wins = 0;
        int episode_block_opportunities = 0;
        int episode_missed_blocks = 0;

        for (int step = 0; step < cfg_.max_steps_per_episode; ++step) {
            const std::vector<int> legal = env_train.legal_actions();
            if (legal.empty()) break;

            const std::vector<uint8_t> obs_before = obs;
            const Tensor state = bitcodec::bits_to_tensor_row(obs);
            sync_action_owner_map(model, env_train);
            if (hooks.before_forward) hooks.before_forward();
            const Tensor q_select = model.forward(state);
            const int action = select_action_epsilon_greedy(q_select, legal, epsilon, rng_);
            const int immediate_win = env_train.tactical_immediate_win_action();
            const int immediate_block = env_train.tactical_immediate_block_action();
            if (immediate_win >= 0) {
                episode_win_opportunities++;
                if (action != immediate_win) episode_missed_wins++;
            } else if (immediate_block >= 0) {
                episode_block_opportunities++;
                if (action != immediate_block) episode_missed_blocks++;
            }

            const int teacher = env_train.teacher_action();
            if (teacher >= 0 && teacher < env_train.num_actions()) {
                teacher_count++;
                if (action == teacher) match_count++;
            }

            const DiscreteStepResult step_result = env_train.step(action, rng_);
            episode_reward += step_result.reward;

            double next_max_q = 0.0;
            if (!step_result.done) {
                const Tensor next_state = bitcodec::bits_to_tensor_row(step_result.observation_bits);
                sync_action_owner_map(model, env_train);
                if (hooks.before_forward) hooks.before_forward();
                const Tensor q_next = model.forward(next_state);
                const std::vector<int> next_legal = env_train.legal_actions();
                next_max_q = max_legal_q(q_next, next_legal);
            }

            model.zero_grad();
            sync_action_owner_map(model, env_train);
            if (hooks.before_forward) hooks.before_forward();
            const Tensor q_pred = model.forward(state);
            Tensor q_target = q_pred;
            const double target = step_result.reward + (step_result.done ? 0.0 : cfg_.gamma * next_max_q);
            q_target(0, static_cast<std::size_t>(action)) = target;

            const double loss = q_loss.forward(q_pred, q_target);
            Tensor grad = q_loss.backward(q_pred, q_target);
            model.backward(grad);
            if (hooks.after_backward) hooks.after_backward(loss, step_result.reward);

            auto params = model.parameters();
            auto grads = model.gradients();
            optimizer.step(params, grads, 1.0, cfg_.grad_clip_norm);
            Adam::zero_grad(grads);
            if (hooks.after_optimizer_step) hooks.after_optimizer_step();

            const double td_abs = std::abs(target - q_pred(0, static_cast<std::size_t>(action)));
            td_error_ema = 0.99 * td_error_ema + 0.01 * td_abs;
            q_loss_ema = 0.99 * q_loss_ema + 0.01 * loss;

            obs = step_result.observation_bits;
            const auto now = std::chrono::steady_clock::now();
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(now - train_start).count();
            if (hooks.on_transition) {
                hooks.on_transition(
                    DiscreteTransitionTrace{
                        episode,
                        step + 1,
                        global_step_ + 1,
                        action,
                        teacher,
                        immediate_win,
                        immediate_block,
                        step_result.reward,
                        step_result.done,
                        step_result.outcome,
                        epsilon,
                        elapsed_ms,
                        obs_before,
                        step_result.observation_bits
                    }
                );
            }
            global_step_++;

            if (step_result.done) break;
        }

        reward_ema = 0.99 * reward_ema + 0.01 * episode_reward;
        if (teacher_count > 0) {
            const double episode_match = static_cast<double>(match_count) / static_cast<double>(teacher_count);
            if (!std::isfinite(action_match_ema)) action_match_ema = episode_match;
            else action_match_ema = 0.99 * action_match_ema + 0.01 * episode_match;
        }
        if (episode_win_opportunities > 0) {
            const double episode_missed_win_rate =
                static_cast<double>(episode_missed_wins) / static_cast<double>(episode_win_opportunities);
            if (!std::isfinite(missed_win_rate_ema)) missed_win_rate_ema = episode_missed_win_rate;
            else missed_win_rate_ema = 0.99 * missed_win_rate_ema + 0.01 * episode_missed_win_rate;
        }
        if (episode_block_opportunities > 0) {
            const double episode_missed_block_rate =
                static_cast<double>(episode_missed_blocks) / static_cast<double>(episode_block_opportunities);
            if (!std::isfinite(missed_block_rate_ema)) missed_block_rate_ema = episode_missed_block_rate;
            else missed_block_rate_ema = 0.99 * missed_block_rate_ema + 0.01 * episode_missed_block_rate;
        }

        epsilon = std::max(cfg_.epsilon_min, epsilon * cfg_.epsilon_decay);

        if (on_batch && (static_cast<std::size_t>(episode) % cfg_.batch_log_every == 0 || episode == cfg_.episodes)) {
            on_batch(
                DiscreteBatchMetrics{
                    episode,
                    global_step_,
                    epsilon,
                    reward_ema,
                    td_error_ema,
                    q_loss_ema,
                    action_match_ema,
                    missed_win_rate_ema,
                    missed_block_rate_ema
                }
            );
        }

        if (episode % cfg_.eval_every == 0 || episode == cfg_.episodes) {
            const bool collect_inference = episode == cfg_.episodes;
            DiscreteEvalMetrics eval = evaluate(env_eval, model, cfg_.eval_episodes, collect_inference);
            eval.episode = episode;
            eval.final_pass = collect_inference;
            have_last_eval = true;
            last_eval = eval;
            if (on_eval) on_eval(eval);
            model.train();
        }

        if (hooks.on_episode_end) hooks.on_episode_end(episode);
    }

    if (!have_last_eval || last_eval.episode != cfg_.episodes || !last_eval.final_pass) {
        DiscreteEvalMetrics final_eval = evaluate(env_eval, model, cfg_.eval_episodes, true);
        final_eval.episode = cfg_.episodes;
        final_eval.final_pass = true;
        if (on_eval) on_eval(final_eval);
        return final_eval;
    }

    return last_eval;
}

} // namespace benchmark
