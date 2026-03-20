CXX ?= g++

# Build/profile configuration
BUILD_TYPE ?= release
OPENMP ?= 1
NATIVE ?= 1
USE_CUDA ?= auto
CUDA_HOME ?= /usr/local/cuda

CPPFLAGS = -Icore -I.
CXXFLAGS = -std=c++17 -Wall -Wextra -MMD -MP
LDFLAGS =
LDLIBS =

ifeq ($(BUILD_TYPE),debug)
	CXXFLAGS += -O0 -g3
else
	CXXFLAGS += -O3 -DNDEBUG
endif

ifeq ($(NATIVE),1)
	CXXFLAGS += -march=native -mtune=native
endif

ifeq ($(OPENMP),1)
	CXXFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif

CUDA_INC_DIR ?= $(CUDA_HOME)/include
CUDA_LIB_DIR ?= $(CUDA_HOME)/targets/x86_64-linux/lib
CUDA_AVAILABLE := $(if $(and $(wildcard $(CUDA_INC_DIR)/cublas_v2.h),$(wildcard $(CUDA_LIB_DIR)/libcublas.so),$(wildcard $(CUDA_LIB_DIR)/libcudart.so)),1,0)

ifeq ($(USE_CUDA),1)
	ifeq ($(CUDA_AVAILABLE),1)
		CPPFLAGS += -DUSE_CUDA -I$(CUDA_INC_DIR)
		LDFLAGS += -L$(CUDA_LIB_DIR) -Wl,-rpath,$(CUDA_LIB_DIR)
		LDLIBS += -lcublas -lcudart -lnvrtc -lcuda
	else
		$(error USE_CUDA=1 but CUDA libraries were not found under $(CUDA_HOME))
	endif
else ifeq ($(USE_CUDA),auto)
	ifeq ($(CUDA_AVAILABLE),1)
		CPPFLAGS += -DUSE_CUDA -I$(CUDA_INC_DIR)
		LDFLAGS += -L$(CUDA_LIB_DIR) -Wl,-rpath,$(CUDA_LIB_DIR)
		LDLIBS += -lcublas -lcudart -lnvrtc -lcuda
	endif
endif

BUILD_DIR = build
BIN_DIR = bin

# Core source files
GAME_ENVS_SRC = core/benchmark/game_envs_connect_four.cpp \
                core/benchmark/game_envs_battleship.cpp \
                core/benchmark/game_envs_golite.cpp \
                core/benchmark/game_envs_chesslite.cpp \
                core/benchmark/game_envs_cartpole_bits.cpp

BIO_RUNTIME_SRC = core/bio/bio_runtime_core.cpp \
                  core/bio/bio_runtime_runtime.cpp

CELL_MODELS_SRC = models/cells/src/cell_models_membrane.cpp \
                  models/cells/src/cell_models_sim.cpp

CELL_MODELS_OBJ = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(CELL_MODELS_SRC))

CELLENGINE_SRC = models/cells/cellengine/cellengine.cpp \
                 models/cells/cellengine/cellengine_primitives.cpp \
                 models/cells/cellengine/cellengine_cuda.cpp \
                 models/cells/cellengine/cellengine_env_body.cpp \
                 models/cells/cellengine/cellengine_population.cpp \
                 models/cells/cellengine/cellengine_trace_io.cpp \
                 models/cells/cellengine/cellengine_evolution_odd.cpp \
                 models/cells/cellengine/cellengine_rl_replay.cpp

CELLENGINE_OBJ = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(CELLENGINE_SRC))

CORE_SRC = core/tensor/tensor.cpp \
           core/tensor/cuda_backend.cpp \
           core/nn/module.cpp \
           core/optim/optimizer.cpp \
           core/losses/losses.cpp \
           core/metrics/metrics.cpp \
           core/data/dataloader.cpp \
           core/io/run_logger.cpp \
           core/benchmark/bit_codec.cpp \
           core/benchmark/bit_bridge_model.cpp \
           core/benchmark/piecewise_bit_bridge_model.cpp \
           core/benchmark/discrete_q_runtime.cpp \
           core/benchmark/tictactoe_env.cpp \
           $(GAME_ENVS_SRC) \
           $(BIO_RUNTIME_SRC) \
           core/online/continuous_runtime.cpp

CORE_OBJ = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(CORE_SRC))

# Automatically track headers for all compile-able translation units
ALL_CPP = $(CORE_SRC) $(wildcard benchmarks/*.cpp) $(wildcard models/*/src/*.cpp) $(CELLENGINE_SRC) tests/test_core_smoke.cpp tests/test_cells_deep.cpp tests/test_leaf_cells.cpp tests/test_cellengine.cpp
ALL_OBJ = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(ALL_CPP))
DEP_FILES = $(ALL_OBJ:.o=.d)

# Model benchmarks
BENCHMARKS = benchmark_mlp benchmark_cnn benchmark_rnn benchmark_lstm \
             benchmark_transformer benchmark_vit benchmark_trees benchmark_forests \
             benchmark_clustering benchmark_hebbian benchmark_actor_critic \
             benchmark_diffusion benchmark_gnn benchmark_forward_forward \
             benchmark_reinforcement benchmark_markov benchmark_tictactoe \
             benchmark_continuous benchmark_games benchmark_cells benchmark_cell_circuit \
             benchmark_cell_circuit_compare benchmark_leaf_cells benchmark_cell_tasks \
             benchmark_cell_developmental benchmark_cellengine
BENCHMARK_BIN_TARGETS = $(addprefix $(BIN_DIR)/, $(BENCHMARKS))

.PHONY: all clean dirs benchmarks run_all test_smoke test_cells_deep test_leaf_cells test_cells_ogre_smoke test_cells_suite cells_prod_check $(BENCHMARKS)

all: dirs $(CORE_OBJ) benchmarks

dirs:
	@mkdir -p $(sort $(dir $(ALL_OBJ))) $(BIN_DIR) output output/final

core: $(CORE_OBJ)

# Compile core
$(BUILD_DIR)/%.o: %.cpp | dirs
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

# --- Benchmarks ---
$(BIN_DIR)/benchmark_mlp: $(CORE_OBJ) $(BUILD_DIR)/benchmarks/benchmark_mlp.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/benchmarks/benchmark_mlp.o $(LDLIBS)

$(BIN_DIR)/benchmark_cnn: $(CORE_OBJ) $(BUILD_DIR)/models/cnn/src/cnn_models.o $(BUILD_DIR)/benchmarks/benchmark_cnn.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/cnn/src/cnn_models.o $(BUILD_DIR)/benchmarks/benchmark_cnn.o $(LDLIBS)

$(BIN_DIR)/benchmark_rnn: $(CORE_OBJ) $(BUILD_DIR)/models/rnn/src/rnn_models.o $(BUILD_DIR)/benchmarks/benchmark_rnn.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/rnn/src/rnn_models.o $(BUILD_DIR)/benchmarks/benchmark_rnn.o $(LDLIBS)

$(BIN_DIR)/benchmark_lstm: $(CORE_OBJ) $(BUILD_DIR)/models/lstm/src/lstm_models.o $(BUILD_DIR)/benchmarks/benchmark_lstm.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/lstm/src/lstm_models.o $(BUILD_DIR)/benchmarks/benchmark_lstm.o $(LDLIBS)

$(BIN_DIR)/benchmark_transformer: $(CORE_OBJ) $(BUILD_DIR)/models/transformer/src/transformer_models.o $(BUILD_DIR)/benchmarks/benchmark_transformer.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/transformer/src/transformer_models.o $(BUILD_DIR)/benchmarks/benchmark_transformer.o $(LDLIBS)

$(BIN_DIR)/benchmark_vit: $(CORE_OBJ) $(BUILD_DIR)/models/vit/src/vit_models.o $(BUILD_DIR)/benchmarks/benchmark_vit.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/vit/src/vit_models.o $(BUILD_DIR)/benchmarks/benchmark_vit.o $(LDLIBS)

$(BIN_DIR)/benchmark_trees: $(CORE_OBJ) $(BUILD_DIR)/models/trees/src/tree_models.o $(BUILD_DIR)/benchmarks/benchmark_trees.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/trees/src/tree_models.o $(BUILD_DIR)/benchmarks/benchmark_trees.o $(LDLIBS)

$(BIN_DIR)/benchmark_forests: $(CORE_OBJ) $(BUILD_DIR)/models/trees/src/tree_models.o $(BUILD_DIR)/models/forests/src/forest_models.o $(BUILD_DIR)/benchmarks/benchmark_forests.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/trees/src/tree_models.o $(BUILD_DIR)/models/forests/src/forest_models.o $(BUILD_DIR)/benchmarks/benchmark_forests.o $(LDLIBS)

$(BIN_DIR)/benchmark_clustering: $(CORE_OBJ) $(BUILD_DIR)/models/clustering/src/clustering_models.o $(BUILD_DIR)/benchmarks/benchmark_clustering.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/clustering/src/clustering_models.o $(BUILD_DIR)/benchmarks/benchmark_clustering.o $(LDLIBS)

$(BIN_DIR)/benchmark_hebbian: $(CORE_OBJ) $(BUILD_DIR)/models/hebbian/src/hebbian_models.o $(BUILD_DIR)/benchmarks/benchmark_hebbian.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/hebbian/src/hebbian_models.o $(BUILD_DIR)/benchmarks/benchmark_hebbian.o $(LDLIBS)

$(BIN_DIR)/benchmark_actor_critic: $(CORE_OBJ) $(BUILD_DIR)/models/actor_critic/src/ac_models.o $(BUILD_DIR)/benchmarks/benchmark_actor_critic.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/actor_critic/src/ac_models.o $(BUILD_DIR)/benchmarks/benchmark_actor_critic.o $(LDLIBS)

$(BIN_DIR)/benchmark_diffusion: $(CORE_OBJ) $(BUILD_DIR)/models/diffusion/src/diffusion_models.o $(BUILD_DIR)/benchmarks/benchmark_diffusion.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/diffusion/src/diffusion_models.o $(BUILD_DIR)/benchmarks/benchmark_diffusion.o $(LDLIBS)

$(BIN_DIR)/benchmark_gnn: $(CORE_OBJ) $(BUILD_DIR)/models/gnn/src/gnn_models.o $(BUILD_DIR)/benchmarks/benchmark_gnn.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/gnn/src/gnn_models.o $(BUILD_DIR)/benchmarks/benchmark_gnn.o $(LDLIBS)

$(BIN_DIR)/benchmark_forward_forward: $(CORE_OBJ) $(BUILD_DIR)/models/forward_forward/src/ff_models.o $(BUILD_DIR)/benchmarks/benchmark_forward_forward.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/forward_forward/src/ff_models.o $(BUILD_DIR)/benchmarks/benchmark_forward_forward.o $(LDLIBS)

$(BIN_DIR)/benchmark_reinforcement: $(CORE_OBJ) $(BUILD_DIR)/models/reinforcement/src/rl_models.o $(BUILD_DIR)/benchmarks/benchmark_reinforcement.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/reinforcement/src/rl_models.o $(BUILD_DIR)/benchmarks/benchmark_reinforcement.o $(LDLIBS)

$(BIN_DIR)/benchmark_markov: $(CORE_OBJ) $(BUILD_DIR)/models/markov/src/markov_models.o $(BUILD_DIR)/benchmarks/benchmark_markov.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(BUILD_DIR)/models/markov/src/markov_models.o $(BUILD_DIR)/benchmarks/benchmark_markov.o $(LDLIBS)

$(BIN_DIR)/benchmark_tictactoe: $(CORE_OBJ) \
	$(BUILD_DIR)/models/hybrid/src/hybrid_models.o \
	$(BUILD_DIR)/models/hybrid/src/model_registry.o \
	$(BUILD_DIR)/models/actor_critic/src/ac_models.o \
	$(BUILD_DIR)/models/cnn/src/cnn_models.o \
	$(BUILD_DIR)/models/hebbian/src/hebbian_models.o \
	$(BUILD_DIR)/models/rnn/src/rnn_models.o \
	$(BUILD_DIR)/models/lstm/src/lstm_models.o \
	$(BUILD_DIR)/models/transformer/src/transformer_models.o \
	$(BUILD_DIR)/models/vit/src/vit_models.o \
	$(BUILD_DIR)/benchmarks/benchmark_tictactoe.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) \
	$(BUILD_DIR)/models/hybrid/src/hybrid_models.o \
	$(BUILD_DIR)/models/hybrid/src/model_registry.o \
	$(BUILD_DIR)/models/actor_critic/src/ac_models.o \
	$(BUILD_DIR)/models/cnn/src/cnn_models.o \
	$(BUILD_DIR)/models/hebbian/src/hebbian_models.o \
	$(BUILD_DIR)/models/rnn/src/rnn_models.o \
	$(BUILD_DIR)/models/lstm/src/lstm_models.o \
	$(BUILD_DIR)/models/transformer/src/transformer_models.o \
	$(BUILD_DIR)/models/vit/src/vit_models.o \
	$(BUILD_DIR)/benchmarks/benchmark_tictactoe.o $(LDLIBS)

$(BIN_DIR)/benchmark_continuous: $(CORE_OBJ) \
	$(BUILD_DIR)/models/hybrid/src/hybrid_models.o \
	$(BUILD_DIR)/models/hybrid/src/model_registry.o \
	$(BUILD_DIR)/models/actor_critic/src/ac_models.o \
	$(BUILD_DIR)/models/cnn/src/cnn_models.o \
	$(BUILD_DIR)/models/hebbian/src/hebbian_models.o \
	$(BUILD_DIR)/models/rnn/src/rnn_models.o \
	$(BUILD_DIR)/models/lstm/src/lstm_models.o \
	$(BUILD_DIR)/models/transformer/src/transformer_models.o \
	$(BUILD_DIR)/models/vit/src/vit_models.o \
	$(BUILD_DIR)/benchmarks/benchmark_continuous.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) \
	$(BUILD_DIR)/models/hybrid/src/hybrid_models.o \
	$(BUILD_DIR)/models/hybrid/src/model_registry.o \
	$(BUILD_DIR)/models/actor_critic/src/ac_models.o \
	$(BUILD_DIR)/models/cnn/src/cnn_models.o \
	$(BUILD_DIR)/models/hebbian/src/hebbian_models.o \
	$(BUILD_DIR)/models/rnn/src/rnn_models.o \
	$(BUILD_DIR)/models/lstm/src/lstm_models.o \
	$(BUILD_DIR)/models/transformer/src/transformer_models.o \
	$(BUILD_DIR)/models/vit/src/vit_models.o \
	$(BUILD_DIR)/benchmarks/benchmark_continuous.o $(LDLIBS)

$(BIN_DIR)/benchmark_games: $(CORE_OBJ) \
	$(BUILD_DIR)/models/hybrid/src/hybrid_models.o \
	$(BUILD_DIR)/models/hybrid/src/model_registry.o \
	$(BUILD_DIR)/models/actor_critic/src/ac_models.o \
	$(BUILD_DIR)/models/cnn/src/cnn_models.o \
	$(BUILD_DIR)/models/hebbian/src/hebbian_models.o \
	$(BUILD_DIR)/models/rnn/src/rnn_models.o \
	$(BUILD_DIR)/models/lstm/src/lstm_models.o \
	$(BUILD_DIR)/models/transformer/src/transformer_models.o \
	$(BUILD_DIR)/models/vit/src/vit_models.o \
	$(BUILD_DIR)/benchmarks/benchmark_games.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) \
	$(BUILD_DIR)/models/hybrid/src/hybrid_models.o \
	$(BUILD_DIR)/models/hybrid/src/model_registry.o \
	$(BUILD_DIR)/models/actor_critic/src/ac_models.o \
	$(BUILD_DIR)/models/cnn/src/cnn_models.o \
	$(BUILD_DIR)/models/hebbian/src/hebbian_models.o \
	$(BUILD_DIR)/models/rnn/src/rnn_models.o \
	$(BUILD_DIR)/models/lstm/src/lstm_models.o \
	$(BUILD_DIR)/models/transformer/src/transformer_models.o \
	$(BUILD_DIR)/models/vit/src/vit_models.o \
	$(BUILD_DIR)/benchmarks/benchmark_games.o $(LDLIBS)

$(BIN_DIR)/benchmark_cells: $(CORE_OBJ) $(CELL_MODELS_OBJ) $(BUILD_DIR)/benchmarks/benchmark_cells.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) $(CELL_MODELS_OBJ) $(BUILD_DIR)/benchmarks/benchmark_cells.o $(LDLIBS)

$(BIN_DIR)/benchmark_cell_circuit: $(CORE_OBJ) \
	$(CELL_MODELS_OBJ) \
	$(BUILD_DIR)/models/cells/src/cell_module.o \
	$(BUILD_DIR)/models/cells/src/cell_circuit.o \
	$(BUILD_DIR)/benchmarks/benchmark_cell_circuit.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) \
	$(CELL_MODELS_OBJ) \
	$(BUILD_DIR)/models/cells/src/cell_module.o \
	$(BUILD_DIR)/models/cells/src/cell_circuit.o \
	$(BUILD_DIR)/benchmarks/benchmark_cell_circuit.o $(LDLIBS)

$(BIN_DIR)/benchmark_cell_circuit_compare: $(CORE_OBJ) \
		$(CELL_MODELS_OBJ) \
		$(BUILD_DIR)/models/cells/src/cell_module.o \
		$(BUILD_DIR)/models/cells/src/cell_circuit.o \
		$(BUILD_DIR)/benchmarks/benchmark_cell_circuit_compare.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) \
		$(CELL_MODELS_OBJ) \
		$(BUILD_DIR)/models/cells/src/cell_module.o \
		$(BUILD_DIR)/models/cells/src/cell_circuit.o \
		$(BUILD_DIR)/benchmarks/benchmark_cell_circuit_compare.o $(LDLIBS)

$(BIN_DIR)/benchmark_leaf_cells: $(CORE_OBJ) \
		$(BUILD_DIR)/models/cells/src/cell_tissue2d.o \
		$(BUILD_DIR)/benchmarks/benchmark_leaf_cells.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) \
		$(BUILD_DIR)/models/cells/src/cell_tissue2d.o \
		$(BUILD_DIR)/benchmarks/benchmark_leaf_cells.o $(LDLIBS)

$(BIN_DIR)/benchmark_cell_tasks: $(CORE_OBJ) \
		$(CELL_MODELS_OBJ) \
		$(BUILD_DIR)/models/cells/src/cell_tasks.o \
		$(BUILD_DIR)/benchmarks/benchmark_cell_tasks.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) \
		$(CELL_MODELS_OBJ) \
		$(BUILD_DIR)/models/cells/src/cell_tasks.o \
		$(BUILD_DIR)/benchmarks/benchmark_cell_tasks.o $(LDLIBS)

$(BIN_DIR)/benchmark_cell_developmental: $(CORE_OBJ) \
		$(BUILD_DIR)/models/cells/src/cell_tissue2d.o \
		$(BUILD_DIR)/models/cells/src/cell_developmental.o \
		$(BUILD_DIR)/models/cells/src/cell_module.o \
		$(BUILD_DIR)/models/cells/src/cell_circuit.o \
		$(BUILD_DIR)/benchmarks/benchmark_cell_developmental.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) \
		$(BUILD_DIR)/models/cells/src/cell_tissue2d.o \
		$(BUILD_DIR)/models/cells/src/cell_developmental.o \
		$(BUILD_DIR)/models/cells/src/cell_module.o \
		$(BUILD_DIR)/models/cells/src/cell_circuit.o \
		$(BUILD_DIR)/benchmarks/benchmark_cell_developmental.o $(LDLIBS)

$(BIN_DIR)/benchmark_cellengine: $(CORE_OBJ) \
		$(CELLENGINE_OBJ) \
		$(BUILD_DIR)/models/reinforcement/src/rl_models.o \
		$(BUILD_DIR)/benchmarks/benchmark_cellengine.o | dirs
	$(CXX) $(LDFLAGS) -o $@ $(CORE_OBJ) \
		$(CELLENGINE_OBJ) \
		$(BUILD_DIR)/models/reinforcement/src/rl_models.o \
		$(BUILD_DIR)/benchmarks/benchmark_cellengine.o $(LDLIBS)

$(BENCHMARKS): %: $(BIN_DIR)/%

benchmarks: $(BENCHMARK_BIN_TARGETS)

# Run all benchmarks
run_all: benchmarks
	@echo "=== Running all benchmarks ==="
	@for b in $(BENCHMARKS); do \
		echo "\n--- $$b ---"; \
		./$(BIN_DIR)/$$b; \
	done

# Quick, deterministic core sanity check.
test_smoke: $(CORE_OBJ) $(BUILD_DIR)/models/reinforcement/src/rl_models.o $(BUILD_DIR)/models/markov/src/markov_models.o $(CELL_MODELS_OBJ) $(BUILD_DIR)/tests/test_core_smoke.o | dirs
	$(CXX) $(LDFLAGS) -o $(BIN_DIR)/test_core_smoke $(CORE_OBJ) $(BUILD_DIR)/models/reinforcement/src/rl_models.o $(BUILD_DIR)/models/markov/src/markov_models.o $(CELL_MODELS_OBJ) $(BUILD_DIR)/tests/test_core_smoke.o $(LDLIBS)
	@./$(BIN_DIR)/test_core_smoke

test_cells_deep: $(CELL_MODELS_OBJ) \
		$(BUILD_DIR)/models/cells/src/cell_tasks.o \
		$(BUILD_DIR)/models/cells/src/cell_module.o \
		$(BUILD_DIR)/models/cells/src/cell_circuit.o \
		$(BUILD_DIR)/tests/test_cells_deep.o | dirs
	$(CXX) $(LDFLAGS) -o $(BIN_DIR)/test_cells_deep \
	$(CELL_MODELS_OBJ) \
	$(BUILD_DIR)/models/cells/src/cell_tasks.o \
	$(BUILD_DIR)/models/cells/src/cell_module.o \
	$(BUILD_DIR)/models/cells/src/cell_circuit.o \
	$(BUILD_DIR)/tests/test_cells_deep.o $(LDLIBS)
	@./$(BIN_DIR)/test_cells_deep

test_leaf_cells: $(CORE_OBJ) \
		$(BUILD_DIR)/models/cells/src/cell_tissue2d.o \
		$(BUILD_DIR)/models/cells/src/cell_developmental.o \
		$(BUILD_DIR)/tests/test_leaf_cells.o | dirs
	$(CXX) $(LDFLAGS) -o $(BIN_DIR)/test_leaf_cells \
	$(CORE_OBJ) \
	$(BUILD_DIR)/models/cells/src/cell_tissue2d.o \
	$(BUILD_DIR)/models/cells/src/cell_developmental.o \
	$(BUILD_DIR)/tests/test_leaf_cells.o $(LDLIBS)
	@./$(BIN_DIR)/test_leaf_cells

test_cellengine: $(CORE_OBJ) \
		$(CELLENGINE_OBJ) \
		$(BUILD_DIR)/models/reinforcement/src/rl_models.o \
		$(BUILD_DIR)/tests/test_cellengine.o | dirs
	$(CXX) $(LDFLAGS) -o $(BIN_DIR)/test_cellengine \
	$(CORE_OBJ) \
	$(CELLENGINE_OBJ) \
	$(BUILD_DIR)/models/reinforcement/src/rl_models.o \
	$(BUILD_DIR)/tests/test_cellengine.o $(LDLIBS)
	@./$(BIN_DIR)/test_cellengine

test_cells_ogre_smoke:
	@./scripts/cells_ogre_smoke.sh

test_cells_suite: test_smoke test_cells_deep test_leaf_cells test_cellengine test_cells_ogre_smoke

cells_prod_check:
	@./scripts/cells_production_check.sh

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

-include $(DEP_FILES)
