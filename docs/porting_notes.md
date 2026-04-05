# Porting Notes

## Matmul Benchmark - Naive C++ (no OpenMP)

Date: 2026-04

Shape                   Avg ms    GFLOPS
-------------------------------------------------------
QKV proj (T=1)              2.11 ms    1.68
QKV proj (T=8)             10.34 ms    2.74
QKV proj (T=64)            78.09 ms    2.90
out proj (T=64)            25.96 ms    2.91
FFN expand (T=64)         113.74 ms    2.66
FFN contract (T=64)       112.28 ms    2.69

## Key Decisions

**Row-major memory layout** — matched NumPy's default C-order so weight
files load without transposition. Element (i,j) of a [T, D] matrix
lives at index i*D + j.

**-1e10f for causal mask instead of -INFINITY** — true -inf can produce
NaN in exp(-inf) * 0 on some hardware. Large negative is numerically
safer and produces the same softmax result.

**Stack allocation for per-head buffers** — GPT-2 small has d_head=64,
max T=1024, so per-head scratch is 64KB — within stack limits. For
GPT-2 medium (d_head=64, larger T) or longer sequences, switch to
heap allocation.

**Config prints moved to stderr** — stdout must be clean JSON for the
Flask subprocess call. All diagnostic output goes to stderr.