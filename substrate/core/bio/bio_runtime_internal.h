#pragma once

#include "core/bio/bio_runtime.h"

#include <string>

namespace bio {

extern thread_local BioRuntime* g_active_runtime;
std::string precise(double v);

} // namespace bio
