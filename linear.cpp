// SSMU.cpp (BRAM-SLASHED + NO exp/log: PWL + poly approx, keep your UNROLL/PIPELINE unchanged)
// - Only changes vs your pasted version:
//   1) Replace SiLU/softplus/exp with fixed-point PWL / 2nd-order poly (NO hls::expf/logf)
//   2) Keep all your existing UNROLL/PIPELINE pragmas exactly as-is (no LUT tuning via pragmas here)
//   3) IMPORTANT FIX: rename dX_in -> X_ssm_in to match actual wiring (you pass X_ssm_upd_stream).
//      This prevents semantic mismatch with TB/golden. Computation uses X_ssm as the "dX term".
//
// NOTE: TB golden MUST use the SAME approximations AND same "dX term" choice (X_ssm), otherwise FAIL.

#include "ssmu.h"
#include <ap_fixed.h>
#include <ap_int.h>

#ifndef __SYNTHESIS__
#include <cstdio>
#endif

#ifndef __SYNTHESIS__
  #define DUT_PRINTF(...) do { std::printf(__VA_ARGS__); std::fflush(stdout); } while(0)
#else
  #define DUT_PRINTF(...) do {} while(0)
#endif

#ifndef USE_FIXED_ACC
#define USE_FIXED_ACC 1
#endif

#if USE_FIXED_ACC
typedef ap_fixed<32, 10> ACC_T;
#else
typedef float ACC_T;
#endif

// ------------------------------------------------------------
// Vector accessors
// ------------------------------------------------------------
static inline DTYPE vget(const DTYPE_VEC &v, int idx) {
#pragma HLS INLINE
    return v[(unsigned)idx];
}
static inline void vset(DTYPE_VEC &v, int idx, DTYPE val) {
#pragma HLS INLINE
    v[(unsigned)idx] = val;
}

// ============================================================
// NO exp/log approximations (fixed-point)
// ============================================================
// Keep widths modest to avoid big shifters/comparators.
typedef ap_fixed<18, 6>  ACT_T;   // activations / inputs
typedef ap_fixed<20, 8>  EXP_T;   // exp approx output

template<typename T>
static inline T clamp_fx(T x, T lo, T hi) {
#pragma HLS INLINE
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// sigmoid(x) ≈ clamp(0.5 + x/4, 0, 1)
static inline ACT_T sigmoid_pwl_fx(ACT_T x) {
#pragma HLS INLINE
    const ACT_T half = (ACT_T)0.5;
    const ACT_T qtr  = (ACT_T)0.25;
    ACT_T y = half + qtr * x;
    return clamp_fx<ACT_T>(y, (ACT_T)0.0, (ACT_T)1.0);
}

// SiLU(x)=x*sigmoid(x)
static inline DTYPE silu_elem(DTYPE a) {
#pragma HLS INLINE
    ACT_T x = (ACT_T)a;
    ACT_T s = sigmoid_pwl_fx(x);
    ACT_T y = x * s;
    return (DTYPE)y;
}

// softplus PWL:
//   x > 8:  ≈ x
//   x < -8: ≈ 0
//   else:   ≈ 0.5*x + 1
static inline DTYPE softplus_pwl_fx(ACC_T xin) {
#pragma HLS INLINE
    ACT_T x = (ACT_T)xin;
    const ACT_T TH  = (ACT_T)8.0;
    const ACT_T NTH = (ACT_T)(-8.0);

    if (x > TH)  return (DTYPE)x;
    if (x < NTH) return (DTYPE)0;

    const ACT_T half = (ACT_T)0.5;
    const ACT_T one  = (ACT_T)1.0;
    ACT_T y = half * x + one;
    return (DTYPE)y;
}

// exp(t) approx: clamp + 2nd order poly in [-3,3]
//   t > 3  -> exp(3)
//   t < -3 -> exp(-3)
//   else   -> 1 + t + 0.5 t^2
static inline EXP_T exp2_poly_fx(ACT_T t) {
#pragma HLS INLINE
    const ACT_T TH  = (ACT_T)3.0;
    const ACT_T NTH = (ACT_T)(-3.0);

    const EXP_T EXP3  = (EXP_T)20.0855369; // exp(3)
    const EXP_T EXPN3 = (EXP_T)0.0497871;  // exp(-3)

    if (t > TH)  return EXP3;
    if (t < NTH) return EXPN3;

    ap_fixed<24, 8> tt = (ap_fixed<24,8>)t * (ap_fixed<24,8>)t;
    ap_fixed<24, 8> y  = (ap_fixed<24,8>)1.0 + (ap_fixed<24,8>)t + (ap_fixed<24,8>)0.5 * tt;

    if (y < 0) y = 0;
    return (EXP_T)y;
}

// ============================================================
// dup for VEC_D tokens (safe small)
// ============================================================
static void dup_vecD_stream_local(hls::stream<DTYPE_VEC>& in,
                                  hls::stream<DTYPE_VEC>& out1,
                                  hls::stream<DTYPE_VEC>& out2) {
#pragma HLS INLINE off
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
    }
}

// ============================================================
// Part 1: conv1d + SiLU
// ============================================================
static void conv1d_silu_stream_local(hls::stream<DTYPE_VEC>& X_in,
                                     hls::stream<DTYPE>& kernel_in,
                                     hls::stream<DTYPE_VEC>& X_gate_out,
                                     hls::stream<DTYPE_VEC>& X_ssm_out) {
#pragma HLS INLINE off

    static DTYPE line_buffer[K-1][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_s2p impl=bram

    DTYPE kernel_buffer[K];
#pragma HLS ARRAY_PARTITION variable=kernel_buffer complete

    for (int i = 0; i < K; ++i) {
#pragma HLS PIPELINE II=1
        kernel_buffer[i] = kernel_in.read();
    }

    DTYPE_VEC X_buffer[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buffer type=ram_s2p impl=bram

    // read X, output gate, buffer for conv
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC xv = X_in.read();
        X_buffer[i] = xv;

        DTYPE_VEC gate_out;
        for (int k = 0; k < VEC_FACTOR; ++k) {
#pragma HLS UNROLL
            vset(gate_out, k, silu_elem(vget(xv, k)));
        }
        X_gate_out.write(gate_out);
    }

    // clear line buffer
    for (int i = 0; i < K-1; ++i)
        for (int k = 0; k < VEC_FACTOR; ++k)
            line_buffer[i][k] = 0;

    // conv1d
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=2
        DTYPE_VEC in_vec = X_buffer[i];

        DTYPE window[K][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=window complete dim=2

        for (int j = 0; j < K-1; ++j)
            for (int k = 0; k < VEC_FACTOR; ++k)
                window[j][k] = line_buffer[j][k];

        for (int k = 0; k < VEC_FACTOR; ++k)
            window[K-1][k] = vget(in_vec, k);

        for (int j = K-2; j > 0; --j)
            for (int k = 0; k < VEC_FACTOR; ++k)
                line_buffer[j][k] = line_buffer[j-1][k];

        for (int k = 0; k < VEC_FACTOR; ++k)
            line_buffer[0][k] = vget(in_vec, k);

        DTYPE_VEC conv_out;
        for (int lane = 0; lane < VEC_FACTOR; ++lane) {
#pragma HLS UNROLL
            // Use ACC_T (fixed) instead of float
            ACC_T sum = 0;
            for (int kk = 0; kk < K; ++kk) {
                sum += (ACC_T)kernel_buffer[kk] * (ACC_T)window[kk][lane];
            }
            vset(conv_out, lane, (DTYPE)sum);
        }

        DTYPE_VEC ssm_out;
        for (int k = 0; k < VEC_FACTOR; ++k) {
#pragma HLS UNROLL
            vset(ssm_out, k, silu_elem(vget(conv_out, k)));
        }
        X_ssm_out.write(ssm_out);
    }
}

// ============================================================
// Part 2: projections (delta) then (B,C) interleaved per i
// - delta produces VEC_D tokens
// - B_out_N and C_out_N produce N tokens each
// ============================================================
static void projection_streams_local(hls::stream<DTYPE_VEC>& X_ssm_in,
                                     const DTYPE_VEC W_B[N][VEC_D],
                                     const DTYPE_VEC W_C[N][VEC_D],
                                     const DTYPE_VEC W_delta[VEC_D][VEC_D],
                                     hls::stream<DTYPE_VEC>& B_out_N,
                                     hls::stream<DTYPE_VEC>& C_out_N,
                                     hls::stream<DTYPE_VEC>& delta_out) {
#pragma HLS INLINE off

    const int J_TILE = 8;

    DTYPE_VEC X_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=bram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_ssm_in.read();
    }

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] delta: start (produce %d)\n", VEC_D);
#endif
    // ---- delta ----
project:
    for (int i = 0; i < VEC_D; ++i) {
        ACC_T acc[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            acc[l] = (ACC_T)0;
        }

        for (int jt = 0; jt < VEC_D; jt += J_TILE) {
#pragma HLS PIPELINE II=8
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                X_tile[jj] = X_buf[jt + jj];
            }
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                DTYPE_VEC w = W_delta[i][jt + jj];
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    acc[l] += (ACC_T)vget(X_tile[jj], l) * (ACC_T)vget(w, l);
                }
            }
        }

        DTYPE_VEC delta_vec;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(delta_vec, l, softplus_pwl_fx(acc[l]));
        }
        delta_out.write(delta_vec);

#ifndef __SYNTHESIS__
        if ((i & 15) == 0) DUT_PRINTF("[DUT] delta: i=%d/%d\n", i, VEC_D);
#endif
    }

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] B+C: start (produce %d each)\n", N);
#endif
    // ---- B and C (per i) ----
    for (int i = 0; i < N; ++i) {
        ACC_T accB[VEC_FACTOR];
        ACC_T accC[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accB complete dim=1
#pragma HLS ARRAY_PARTITION variable=accC complete dim=1
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            accB[l] = (ACC_T)0;
            accC[l] = (ACC_T)0;
        }

        for (int jt = 0; jt < VEC_D; jt += J_TILE) {
#pragma HLS PIPELINE II=8
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                X_tile[jj] = X_buf[jt + jj];
            }

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                DTYPE_VEC wB = W_B[i][jt + jj];
                DTYPE_VEC wC = W_C[i][jt + jj];
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    ACC_T x = (ACC_T)vget(X_tile[jj], l);
                    accB[l] += x * (ACC_T)vget(wB, l);
                    accC[l] += x * (ACC_T)vget(wC, l);
                }
            }
        }

        DTYPE_VEC outB, outC;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(outB, l, (DTYPE)accB[l]);
            vset(outC, l, (DTYPE)accC[l]);
        }
        B_out_N.write(outB);
        C_out_N.write(outC);

#ifndef __SYNTHESIS__
        if ((i & 511) == 0) DUT_PRINTF("[DUT] B+C: i=%d/%d\n", i, N);
#endif
    }
}

// ============================================================
// FUSED stage: update_H + write DDR + accumulate + final output
// ============================================================
static void fused_update_write_accum_output(
        hls::stream<DTYPE_VEC>& X_gate_in,
        hls::stream<DTYPE_VEC>& X_ssm_in,    // VEC_D tokens (this is what you actually wire in)
        hls::stream<DTYPE_VEC>& delta_in,    // VEC_D tokens
        hls::stream<DTYPE_VEC>& A_in,        // N tokens (per-i) in this fused design
        hls::stream<DTYPE_VEC>& B_in,        // N tokens
        hls::stream<DTYPE_VEC>& C_in,        // N tokens
        hls::stream<DTYPE_VEC>& H0_in,       // HUGE_LEN tokens
        DTYPE_VEC* C_ddr,
        DTYPE_VEC* H1_ddr,
        hls::stream<DTYPE_VEC>& out) {
#pragma HLS INLINE off

    // Buffer X_gate[VEC_D]
    DTYPE_VEC X_gate[VEC_D];
#pragma HLS BIND_STORAGE variable=X_gate type=ram_s2p impl=bram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_gate[j] = X_gate_in.read();
    }

    // Buffer X_ssm[VEC_D]  (used as the "dX term" in this design)
    DTYPE_VEC X_ssm_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_ssm_buf type=ram_s2p impl=bram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_ssm_buf[j] = X_ssm_in.read();
    }

    // Buffer delta[VEC_D]
    DTYPE_VEC delta_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=delta_buf type=ram_s2p impl=bram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        delta_buf[j] = delta_in.read();
    }

    // Accumulator acc[VEC_D] init = 0
    DTYPE_VEC acc[VEC_D];
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=bram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC z;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(z, l, (DTYPE)0);
        }
        acc[j] = z;
    }

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] FUSED: start update/write/acc (N=%d, VEC_D=%d, HUGE_LEN=%d)\n", N, VEC_D, (int)HUGE_LEN);
#endif

    // Main loops: i-major, j-minor
    for (int i = 0; i < N; ++i) {
        DTYPE_VEC A_vec = A_in.read();
        DTYPE_VEC B_vec = B_in.read();
        DTYPE_VEC C_vec = C_in.read();

        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
            int idx = i * VEC_D + j;

            DTYPE_VEC H0v  = H0_in.read();
            DTYPE_VEC dlt  = delta_buf[j];
            DTYPE_VEC xssm = X_ssm_buf[j];

            // Compute H1 = H0*exp(A*delta) + (B*delta)*X_ssm  (NO expf: poly approx)
            DTYPE_VEC H1v;
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACT_T a  = (ACT_T)vget(A_vec, l);
                ACT_T dl = (ACT_T)vget(dlt,  l);
                EXP_T ddA_fx = exp2_poly_fx(a * dl);

                ACC_T H0  = (ACC_T)vget(H0v, l);
                ACC_T Bx  = (ACC_T)vget(B_vec, l);
                ACC_T Xs  = (ACC_T)vget(xssm, l);

                ACC_T H1  = H0 * (ACC_T)ddA_fx + (Bx * (ACC_T)dl) * Xs;
                vset(H1v, l, (DTYPE)H1);
            }

            // Write DDR in-place (no buffering)
            C_ddr[idx]  = C_vec;
            H1_ddr[idx] = H1v;

            // acc[j] += H1 * C
            DTYPE_VEC aold = acc[j];
            DTYPE_VEC anew;
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACC_T base = (ACC_T)vget(aold, l);
                ACC_T addt = (ACC_T)vget(H1v,  l) * (ACC_T)vget(C_vec, l);
                vset(anew, l, (DTYPE)(base + addt));
            }
            acc[j] = anew;
        }
#ifndef __SYNTHESIS__
        if ((i & 511) == 0) DUT_PRINTF("[DUT] FUSED: i=%d/%d\n", i, N);
#endif
    }

    // out = X_gate + acc
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC a = acc[j];
        DTYPE_VEC x = X_gate[j];
        DTYPE_VEC outv;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(outv, l, (DTYPE)((ACC_T)vget(x, l) + (ACC_T)vget(a, l)));
        }
        out.write(outv);
    }

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] FUSED: done\n");
#endif
}

// ============================================================
// TOP: SSMU
// ============================================================
void SSMU(hls::stream<DTYPE>& kernel_in,
          hls::stream<DTYPE_VEC>& A_in,
          const DTYPE_VEC W_B[N][VEC_D],
          const DTYPE_VEC W_C[N][VEC_D],
          const DTYPE_VEC W_delta[VEC_D][VEC_D],
          hls::stream<DTYPE_VEC>& X_in,
          hls::stream<DTYPE_VEC>& H0_in,
          DTYPE_VEC* C_ddr,
          DTYPE_VEC* H1_ddr,
          hls::stream<DTYPE_VEC>& out) {

#pragma HLS INTERFACE ap_fifo port=kernel_in
#pragma HLS INTERFACE ap_fifo port=A_in
#pragma HLS INTERFACE ap_fifo port=X_in
#pragma HLS INTERFACE ap_fifo port=H0_in
#pragma HLS INTERFACE ap_fifo port=out

#pragma HLS INTERFACE m_axi port=C_ddr  offset=slave bundle=gmem0 depth=HUGE_LEN
#pragma HLS INTERFACE m_axi port=H1_ddr offset=slave bundle=gmem1 depth=HUGE_LEN
#pragma HLS INTERFACE s_axilite port=C_ddr  bundle=control
#pragma HLS INTERFACE s_axilite port=H1_ddr bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

// External ports: moderate depths (no waste)
#pragma HLS STREAM variable=kernel_in depth=64
#pragma HLS STREAM variable=A_in      depth=32
#pragma HLS STREAM variable=X_in      depth=32
#pragma HLS STREAM variable=H0_in     depth=64
#pragma HLS STREAM variable=out       depth=32

#pragma HLS DATAFLOW

    hls::stream<DTYPE_VEC> X_gate_stream("X_gate_stream");
    hls::stream<DTYPE_VEC> X_ssm_stream ("X_ssm_stream");
    hls::stream<DTYPE_VEC> X_ssm_proj_stream("X_ssm_proj_stream");
    hls::stream<DTYPE_VEC> X_ssm_upd_stream ("X_ssm_upd_stream");

    hls::stream<DTYPE_VEC> B_stream_N("B_stream_N");
    hls::stream<DTYPE_VEC> C_stream_N("C_stream_N");
    hls::stream<DTYPE_VEC> delta_stream("delta_stream");

// Internal streams: “just enough” depths
#pragma HLS STREAM variable=X_gate_stream      depth=32
#pragma HLS STREAM variable=X_ssm_stream       depth=32
#pragma HLS STREAM variable=X_ssm_proj_stream  depth=32
#pragma HLS STREAM variable=X_ssm_upd_stream   depth=32

#pragma HLS STREAM variable=delta_stream       depth=256   // VEC_D tokens -> 256 safe
#pragma HLS STREAM variable=B_stream_N         depth=32    // producer/consumer both per-i
#pragma HLS STREAM variable=C_stream_N         depth=32

    conv1d_silu_stream_local(X_in, kernel_in, X_gate_stream, X_ssm_stream);
    dup_vecD_stream_local(X_ssm_stream, X_ssm_proj_stream, X_ssm_upd_stream);

    // projection: delta then B/C
    projection_streams_local(X_ssm_proj_stream, W_B, W_C, W_delta,
                             B_stream_N, C_stream_N, delta_stream);

    // FUSED: update_H + write DDR + accumulate + output
    // NOTE: second argument is X_ssm_upd_stream (used as X_ssm term in fused stage)
    fused_update_write_accum_output(
        X_gate_stream,
        X_ssm_upd_stream,
        delta_stream,
        A_in,
        B_stream_N,
        C_stream_N,
        H0_in,
        C_ddr,
        H1_ddr,
        out
    );
}
