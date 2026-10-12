// Shared by the MSM kernels; see src/msm.rs.
#pragma once
#include <cstdint>
#include "icicle/curves/curve_config.h"

using namespace curve_config;

// Counts the sort leaves on the device. Work items are uint4 (bucket, start, len, unused), the
// tasks of the large buckets first.
#define INFO_PAIRS 0
#define INFO_LARGE_TASKS 1
#define INFO_WORK 2
#define INFO_LARGE_BUCKETS 3
