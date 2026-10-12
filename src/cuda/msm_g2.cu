// MSM kernels over BN254 G2.
#include "msm.cuh"
typedef g2_projective_t P;
typedef g2_affine_t A;
typedef g2_point_field_t F;
#define XYZZ 1
#include "msm_curve.cuh"
