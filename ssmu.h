#ifndef __UCI_EECS_SSMU_HEADER_20260106_BIG__
#define __UCI_EECS_SSMU_HEADER_20260106_BIG__

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

// =============================================================
// Optional debug banner
// =============================================================
#ifndef SSMU_HEADER_DEBUG
#define SSMU_HEADER_DEBUG 0
#endif

#if SSMU_HEADER_DEBUG
#warning ">>> ssmu.h INCLUDED (FULL SIZE MODE) <<<"
#endif

// =============================================================
// Model sizes (FULL checkpoint sizes ONLY)
// DIM=2560, N=5120, VEC_FACTOR=16 => VEC_D=160
// =============================================================
#ifndef DIM
#define DIM 2560
#endif

#ifndef N
#define N 5120
#endif

#ifndef VEC_FACTOR
#define VEC_FACTOR 16
#endif

#ifndef K
#define K 4
#endif

// =============================================================
// Derived sizes
// =============================================================
#ifndef VEC_D
#define VEC_D (DIM / VEC_FACTOR)
#endif

#ifndef HUGE_LEN
#define HUGE_LEN (N * VEC_D)
#endif

// =============================================================
// Types
// =============================================================
typedef ap_fixed<16,4> DTYPE;
typedef hls::vector<DTYPE, VEC_FACTOR> DTYPE_VEC;

// =============================================================
// constexpr mirrors
// =============================================================
static constexpr int DIM_C        = DIM;
static constexpr int N_C          = N;
static constexpr int K_C          = K;
static constexpr int VEC_FACTOR_C = VEC_FACTOR;
static constexpr int VEC_D_C      = VEC_D;
static constexpr int HUGE_LEN_C   = HUGE_LEN;

// =============================================================
// Compile-time safety checks
// =============================================================
static_assert(VEC_FACTOR_C > 0, "VEC_FACTOR must be > 0");
static_assert(DIM_C > 0,        "DIM must be > 0");
static_assert((DIM_C % VEC_FACTOR_C) == 0, "DIM must be divisible by VEC_FACTOR");

// keep these (your code assumes tiles of 8/16)
static_assert((VEC_D_C % 8)  == 0, "VEC_D must be multiple of 8");
static_assert((VEC_D_C % 16) == 0, "VEC_D must be multiple of 16");

static_assert(HUGE_LEN_C > 0, "HUGE_LEN must be > 0");

// =============================================================
// TOP prototype (matches your current SSMU.cpp)
// =============================================================
void SSMU(
    hls::stream<DTYPE>& kernel_in,
    hls::stream<DTYPE_VEC>& A_in,
    const DTYPE_VEC W_B[N][VEC_D],
    const DTYPE_VEC W_C[N][VEC_D],
    const DTYPE_VEC W_delta[VEC_D][VEC_D],
    hls::stream<DTYPE_VEC>& X_in,
    hls::stream<DTYPE_VEC>& H0_in,
    DTYPE_VEC* C_ddr,      // length = HUGE_LEN
    DTYPE_VEC* H1_ddr,     // length = HUGE_LEN
    hls::stream<DTYPE_VEC>& out
);

#endif // __UCI_EECS_SSMU_HEADER_20260106_BIG__
