#include "models/cells/cellengine/cellengine_internal.h"

#define CELLENGINE_USE_EXTERNAL_CUDA_DISPATCH 1

namespace cells {
namespace cellengine {
namespace detail {

#include "cellengine_impl_040_population.inc"

} // namespace detail
} // namespace cellengine
} // namespace cells
