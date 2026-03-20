#include "models/cells/cellengine/cellengine_internal.h"

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#endif

namespace cells {
namespace cellengine {
namespace detail {

#include "cellengine_impl_020_cuda.inc"

bool try_cuda_cell_tick(CellPopulation& pop,
                        const BodyTemplate& body,
                        const DecodedGenome& d,
                        const CellEngineConfig& cfg,
                        const double dt,
                        const bool teaching_active,
                        const ODDIntervention* intervention) {
#ifdef USE_CUDA
    auto& runtime = cuda_cell_tick_runtime();
    if (!runtime.should_try(pop.count, static_cast<int>(body.neighbor_flat.size()), body.grid_cell_count())) {
        return false;
    }
    return runtime.tick(pop, body, d, cfg, dt, teaching_active, intervention);
#else
    (void)pop;
    (void)body;
    (void)d;
    (void)cfg;
    (void)dt;
    (void)teaching_active;
    (void)intervention;
    return false;
#endif
}

} // namespace detail
} // namespace cellengine
} // namespace cells
