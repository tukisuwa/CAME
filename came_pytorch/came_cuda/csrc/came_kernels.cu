#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cub/block/block_reduce.cuh>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace {

constexpr int THREADS = 256;
constexpr float QMAX = 127.0f;
constexpr float UQMAX = 255.0f;
constexpr int TILE = 16;
constexpr int VEC4_THREADS = 64; // 16 rows * (16 cols / 4)

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float(float x) { return x; }
__device__ __forceinline__ half from_float_half(float x) { return __float2half_rn(x); }
__device__ __forceinline__ __nv_bfloat16 from_float_bf16(float x) { return __float2bfloat16(x); }

template <typename T>
__device__ __forceinline__ T from_float_t(float x);

template <>
__device__ __forceinline__ float from_float_t<float>(float x) {
    return from_float(x);
}

template <>
__device__ __forceinline__ half from_float_t<half>(float x) {
    return from_float_half(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float_t<__nv_bfloat16>(float x) {
    return from_float_bf16(x);
}

__device__ __forceinline__ float dequant_u8(const uint8_t q, const float absmax);
__device__ __forceinline__ uint8_t quant_u8(const float x, const float inv_absmax);
__device__ __forceinline__ void atomic_max_positive_f32(float* addr, float val) {
    atomicMax(reinterpret_cast<int*>(addr), __float_as_int(val));
}

template <typename T>
__global__ void kUpdateSqRowQuantized(
    const T* __restrict__ g,
    const uint8_t* __restrict__ exp_avg_sq_row_q,
    const float* __restrict__ exp_avg_sq_row_absmax,
    float* __restrict__ updated_row,
    float* __restrict__ next_absmax,
    const float beta2,
    const float eps0,
    const int rows,
    const int cols,
    const int block_size
) {
    const int row = (int)blockIdx.x;
    if (row >= rows)
        return;

    float local = 0.0f;
    for (int col = (int)threadIdx.x; col < cols; col += THREADS) {
        const int idx = row * cols + col;
        const float gv = to_float(g[idx]);
        local += (gv * gv) + eps0;
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float row_sum = BlockReduceT(tmp).Sum(local);

    if (threadIdx.x == 0) {
        const float mean = row_sum / (float)cols;
        const int qblock = row / block_size;
        const float old = dequant_u8(exp_avg_sq_row_q[row], exp_avg_sq_row_absmax[qblock]);
        const float updated = (old * beta2) + ((1.0f - beta2) * mean);
        updated_row[row] = updated;
        atomic_max_positive_f32(next_absmax + qblock, updated);
    }
}

template <typename T>
__global__ void kUpdateSqRowQuantizedBatched(
    const T* __restrict__ g,
    const uint8_t* __restrict__ exp_avg_sq_row_q,
    const float* __restrict__ exp_avg_sq_row_absmax,
    float* __restrict__ updated_row,
    float* __restrict__ next_absmax,
    const float beta2,
    const float eps0,
    const int64_t per_item_numel,
    const int rows,
    const int cols,
    const int row_q_blocks,
    const int block_size
) {
    const int linear_row = (int)blockIdx.x;
    const int item = linear_row / rows;
    const int row = linear_row - item * rows;
    if (row >= rows)
        return;

    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t row_offset = (int64_t)item * rows;
    const int64_t row_absmax_offset = (int64_t)item * row_q_blocks;

    float local = 0.0f;
    for (int col = (int)threadIdx.x; col < cols; col += THREADS) {
        const int64_t idx = item_offset + (int64_t)row * cols + col;
        const float gv = to_float(g[idx]);
        local += (gv * gv) + eps0;
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float row_sum = BlockReduceT(tmp).Sum(local);

    if (threadIdx.x == 0) {
        const float mean = row_sum / (float)cols;
        const int qblock = row / block_size;
        const float old = dequant_u8(
            exp_avg_sq_row_q[row_offset + row],
            exp_avg_sq_row_absmax[row_absmax_offset + qblock]
        );
        const float updated = (old * beta2) + ((1.0f - beta2) * mean);
        updated_row[row_offset + row] = updated;
        atomic_max_positive_f32(next_absmax + row_absmax_offset + qblock, updated);
    }
}

template <typename T>
__global__ void kUpdateSqRowQuantizedBatchedAccum(
    const T* __restrict__ g,
    const uint8_t* __restrict__ exp_avg_sq_row_q,
    const float* __restrict__ exp_avg_sq_row_absmax,
    float* __restrict__ updated_row,
    float* __restrict__ next_absmax,
    float* __restrict__ sum_row_state,
    const float beta2,
    const float eps0,
    const int64_t per_item_numel,
    const int rows,
    const int cols,
    const int row_q_blocks,
    const int block_size
) {
    const int linear_row = (int)blockIdx.x;
    const int item = linear_row / rows;
    const int row = linear_row - item * rows;
    if (row >= rows)
        return;

    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t row_offset = (int64_t)item * rows;
    const int64_t row_absmax_offset = (int64_t)item * row_q_blocks;

    float local = 0.0f;
    for (int col = (int)threadIdx.x; col < cols; col += THREADS) {
        const int64_t idx = item_offset + (int64_t)row * cols + col;
        const float gv = to_float(g[idx]);
        local += (gv * gv) + eps0;
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float row_sum = BlockReduceT(tmp).Sum(local);

    if (threadIdx.x == 0) {
        const float mean = row_sum / (float)cols;
        const int qblock = row / block_size;
        const float old = dequant_u8(
            exp_avg_sq_row_q[row_offset + row],
            exp_avg_sq_row_absmax[row_absmax_offset + qblock]
        );
        const float updated = (old * beta2) + ((1.0f - beta2) * mean);
        updated_row[row_offset + row] = updated;
        atomic_max_positive_f32(next_absmax + row_absmax_offset + qblock, updated);
        atomicAdd(sum_row_state + item, updated);
    }
}

template <typename T>
__global__ void kUpdateSqColQuantizedFinalize(
    const T* __restrict__ g,
    uint8_t* __restrict__ exp_avg_sq_col_q,
    float* __restrict__ exp_avg_sq_col_absmax,
    float* __restrict__ c_factor,
    const float beta2,
    const float eps0,
    const int rows,
    const int cols,
    const int block_size
) {
    const int block = (int)blockIdx.x;
    const int start = block * block_size;
    const int col = start + (int)threadIdx.x;
    const bool valid = col < cols;

    float updated = 0.0f;
    if (valid) {
        float local = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const int idx = row * cols + col;
            const float gv = to_float(g[idx]);
            local += (gv * gv) + eps0;
        }
        const float mean = local / (float)rows;
        const float old = dequant_u8(exp_avg_sq_col_q[col], exp_avg_sq_col_absmax[block]);
        updated = (old * beta2) + ((1.0f - beta2) * mean);
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_absmax = BlockReduceT(tmp).Reduce(updated, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        exp_avg_sq_col_absmax[block] = shared_absmax;
    }
    __syncthreads();

    if (valid) {
        const float inv_absmax = 1.0f / shared_absmax;
        exp_avg_sq_col_q[col] = quant_u8(updated, inv_absmax);
        c_factor[col] = rsqrtf(updated);
    }
}

template <typename T>
__global__ void kUpdateSqColQuantizedFinalizeBatched(
    const T* __restrict__ g,
    uint8_t* __restrict__ exp_avg_sq_col_q,
    float* __restrict__ exp_avg_sq_col_absmax,
    float* __restrict__ c_factor,
    const float beta2,
    const float eps0,
    const int64_t per_item_numel,
    const int rows,
    const int cols,
    const int col_q_blocks,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / col_q_blocks;
    const int item_block = linear_block - item * col_q_blocks;
    const int start = item_block * block_size;
    const int col = start + (int)threadIdx.x;
    const bool valid = col < cols;
    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t col_offset = (int64_t)item * cols;

    float updated = 0.0f;
    if (valid) {
        float local = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const int64_t idx = item_offset + (int64_t)row * cols + col;
            const float gv = to_float(g[idx]);
            local += (gv * gv) + eps0;
        }
        const float mean = local / (float)rows;
        const float old = dequant_u8(exp_avg_sq_col_q[col_offset + col], exp_avg_sq_col_absmax[linear_block]);
        updated = (old * beta2) + ((1.0f - beta2) * mean);
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_absmax = BlockReduceT(tmp).Reduce(updated, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        exp_avg_sq_col_absmax[linear_block] = shared_absmax;
    }
    __syncthreads();

    if (valid) {
        const float inv_absmax = 1.0f / shared_absmax;
        exp_avg_sq_col_q[col_offset + col] = quant_u8(updated, inv_absmax);
        c_factor[col_offset + col] = rsqrtf(updated);
    }
}

__global__ void kFinalizeQuantizedRFactor(
    const float* __restrict__ updated,
    uint8_t* __restrict__ q,
    float* __restrict__ absmax,
    const float* __restrict__ next_absmax,
    float* __restrict__ factor_out,
    const float* __restrict__ sum_vec,
    const int n,
    const int block_size
) {
    const int block = (int)blockIdx.x;
    const int start = block * block_size;
    if (start >= n)
        return;

    __shared__ float shared_absmax;
    __shared__ float shared_mean;
    if (threadIdx.x == 0) {
        const float block_absmax = next_absmax[block];
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        absmax[block] = shared_absmax;
        float mean = (*sum_vec) / (float)n;
        shared_mean = (mean > 0.0f) ? mean : 1.0f;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int idx = start + offset;
        if (idx < n) {
            const float v = updated[idx];
            q[idx] = quant_u8(v, inv_absmax);
            factor_out[idx] = rsqrtf(v / shared_mean);
        }
    }
}

__global__ void kFinalizeQuantizedRFactorBatched(
    const float* __restrict__ updated,
    uint8_t* __restrict__ q,
    float* __restrict__ absmax,
    const float* __restrict__ next_absmax,
    float* __restrict__ factor_out,
    const float* __restrict__ sum_vec,
    const int n,
    const int blocks_per_item,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / blocks_per_item;
    const int item_block = linear_block - item * blocks_per_item;
    const int start = item_block * block_size;
    if (start >= n)
        return;

    const int64_t item_offset = (int64_t)item * n;

    __shared__ float shared_absmax;
    __shared__ float shared_mean;
    if (threadIdx.x == 0) {
        const float block_absmax = next_absmax[linear_block];
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        absmax[linear_block] = shared_absmax;
        float mean = sum_vec[item] / (float)n;
        shared_mean = (mean > 0.0f) ? mean : 1.0f;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int idx = start + offset;
        if (idx < n) {
            const float v = updated[item_offset + idx];
            q[item_offset + idx] = quant_u8(v, inv_absmax);
            factor_out[item_offset + idx] = rsqrtf(v / shared_mean);
        }
    }
}

template <typename T>
__global__ void kUpdateSqRowQuantizedMultiTensorSameShape(
    const T* __restrict__ g,
    const int64_t* __restrict__ exp_avg_sq_row_q_ptrs,
    const int64_t* __restrict__ exp_avg_sq_row_absmax_ptrs,
    float* __restrict__ updated_row,
    float* __restrict__ next_absmax,
    const float beta2,
    const float eps0,
    const int64_t per_item_numel,
    const int rows,
    const int cols,
    const int row_q_blocks,
    const int block_size
) {
    const int linear_row = (int)blockIdx.x;
    const int item = linear_row / rows;
    const int row = linear_row - item * rows;
    if (row >= rows)
        return;

    const auto* q_in = reinterpret_cast<const uint8_t*>(exp_avg_sq_row_q_ptrs[item]);
    const auto* absmax_in = reinterpret_cast<const float*>(exp_avg_sq_row_absmax_ptrs[item]);
    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t row_offset = (int64_t)item * rows;
    const int64_t row_absmax_offset = (int64_t)item * row_q_blocks;

    float local = 0.0f;
    for (int col = (int)threadIdx.x; col < cols; col += THREADS) {
        const int64_t idx = item_offset + (int64_t)row * cols + col;
        const float gv = to_float(g[idx]);
        local += (gv * gv) + eps0;
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float row_sum = BlockReduceT(tmp).Sum(local);

    if (threadIdx.x == 0) {
        const float mean = row_sum / (float)cols;
        const int qblock = row / block_size;
        const float old = dequant_u8(q_in[row], absmax_in[qblock]);
        const float updated = (old * beta2) + ((1.0f - beta2) * mean);
        updated_row[row_offset + row] = updated;
        atomic_max_positive_f32(next_absmax + row_absmax_offset + qblock, updated);
    }
}

template <typename T>
__global__ void kUpdateSqColQuantizedFinalizeMultiTensorSameShape(
    const T* __restrict__ g,
    const int64_t* __restrict__ exp_avg_sq_col_q_ptrs,
    const int64_t* __restrict__ exp_avg_sq_col_absmax_ptrs,
    float* __restrict__ c_factor,
    const float beta2,
    const float eps0,
    const int64_t per_item_numel,
    const int rows,
    const int cols,
    const int col_q_blocks,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / col_q_blocks;
    const int item_block = linear_block - item * col_q_blocks;
    const int start = item_block * block_size;
    const int col = start + (int)threadIdx.x;
    const bool valid = col < cols;
    auto* q_out = reinterpret_cast<uint8_t*>(exp_avg_sq_col_q_ptrs[item]);
    auto* absmax_out = reinterpret_cast<float*>(exp_avg_sq_col_absmax_ptrs[item]);
    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t col_offset = (int64_t)item * cols;

    float updated = 0.0f;
    if (valid) {
        float local = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const int64_t idx = item_offset + (int64_t)row * cols + col;
            const float gv = to_float(g[idx]);
            local += (gv * gv) + eps0;
        }
        const float mean = local / (float)rows;
        const float old = dequant_u8(q_out[col], absmax_out[item_block]);
        updated = (old * beta2) + ((1.0f - beta2) * mean);
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_absmax = BlockReduceT(tmp).Reduce(updated, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        absmax_out[item_block] = shared_absmax;
    }
    __syncthreads();

    if (valid) {
        const float inv_absmax = 1.0f / shared_absmax;
        q_out[col] = quant_u8(updated, inv_absmax);
        c_factor[col_offset + col] = rsqrtf(fmaxf(updated, 1e-30f));
    }
}

__global__ void kFinalizeQuantizedRFactorMultiTensorSameShape(
    const float* __restrict__ updated,
    const int64_t* __restrict__ q_ptrs,
    const int64_t* __restrict__ absmax_ptrs,
    const float* __restrict__ next_absmax,
    float* __restrict__ factor_out,
    const float* __restrict__ sum_vec,
    const int n,
    const int blocks_per_item,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / blocks_per_item;
    const int item_block = linear_block - item * blocks_per_item;
    const int start = item_block * block_size;
    if (start >= n)
        return;

    auto* q = reinterpret_cast<uint8_t*>(q_ptrs[item]);
    auto* absmax = reinterpret_cast<float*>(absmax_ptrs[item]);
    const int64_t item_offset = (int64_t)item * n;

    __shared__ float shared_absmax;
    __shared__ float shared_mean;
    if (threadIdx.x == 0) {
        const float block_absmax = next_absmax[linear_block];
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        absmax[item_block] = shared_absmax;
        const float mean = sum_vec[item] / (float)n;
        shared_mean = (mean > 0.0f) ? mean : 1.0f;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int idx = start + offset;
        if (idx < n) {
            const float v = updated[item_offset + idx];
            q[idx] = quant_u8(v, inv_absmax);
            factor_out[item_offset + idx] = rsqrtf(v / shared_mean);
        }
    }
}

__global__ void kUpdateSqRowFp32(
    const float* __restrict__ g32,
    float* __restrict__ exp_avg_sq_row,
    const float beta2,
    const float eps0,
    const int rows,
    const int cols
) {
    const int row = (int)blockIdx.x;
    if (row >= rows)
        return;

    float local = 0.0f;
    for (int col = (int)threadIdx.x; col < cols; col += THREADS) {
        const float g = g32[row * cols + col];
        local += (g * g) + eps0;
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float row_sum = BlockReduceT(tmp).Sum(local);

    if (threadIdx.x == 0) {
        const float mean = row_sum / (float)cols;
        exp_avg_sq_row[row] = (exp_avg_sq_row[row] * beta2) + ((1.0f - beta2) * mean);
    }
}

__global__ void kUpdateSqColFinalizeFp32(
    const float* __restrict__ g32,
    float* __restrict__ exp_avg_sq_col,
    float* __restrict__ c_factor,
    const float beta2,
    const float eps0,
    const int rows,
    const int cols
) {
    const int col = (int)blockIdx.x * THREADS + (int)threadIdx.x;
    const bool valid = col < cols;

    float updated = 0.0f;
    if (valid) {
        float local = 0.0f;
        for (int row = 0; row < rows; ++row) {
            const float g = g32[row * cols + col];
            local += (g * g) + eps0;
        }
        const float mean = local / (float)rows;
        updated = (exp_avg_sq_col[col] * beta2) + ((1.0f - beta2) * mean);
    }

    if (valid) {
        exp_avg_sq_col[col] = updated;
        c_factor[col] = rsqrtf(fmaxf(updated, 1e-30f));
    }
}

__global__ void kFinalizeRFactorFp32(
    const float* __restrict__ updated_row,
    float* __restrict__ factor_out,
    const float* __restrict__ sum_row_state,
    const int rows
) {
    const int idx = (int)blockIdx.x * THREADS + (int)threadIdx.x;
    if (idx >= rows)
        return;

    const float mean = fmaxf(sum_row_state[0] / (float)rows, 1e-30f);
    factor_out[idx] = rsqrtf(fmaxf(updated_row[idx] / mean, 1e-30f));
}

template <typename T>
__global__ void kSumUpdateSq(
    const T* __restrict__ g,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    float* __restrict__ partial_sum,
    const int n,
    const int rows,
    const int cols
) {
    float local = 0.0f;
    for (int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x); idx < n; idx += gridDim.x * blockDim.x) {
        const int row = idx / cols;
        const int col = idx - row * cols;
        const float gv = to_float(g[idx]);
        const float u = gv * r_factor[row] * c_factor[col];
        local += u * u;
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_sum = BlockReduceT(tmp).Sum(local);
    if (threadIdx.x == 0)
        partial_sum[blockIdx.x] = block_sum;
}

template <typename T>
__global__ void kSumUpdateSqBatched(
    const T* __restrict__ g,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    float* __restrict__ partial_sum,
    const int64_t per_item_numel,
    const int rows,
    const int cols,
    const int blocks_per_item
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / blocks_per_item;
    const int item_block = linear_block - item * blocks_per_item;
    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t row_offset = (int64_t)item * rows;
    const int64_t col_offset = (int64_t)item * cols;

    float local = 0.0f;
    const int64_t stride = (int64_t)blocks_per_item * blockDim.x;
    for (int64_t item_idx = (int64_t)item_block * blockDim.x + threadIdx.x; item_idx < per_item_numel; item_idx += stride) {
        const int row = (int)(item_idx / cols);
        const int col = (int)(item_idx - (int64_t)row * cols);
        const float gv = to_float(g[item_offset + item_idx]);
        const float u = gv * r_factor[row_offset + row] * c_factor[col_offset + col];
        local += u * u;
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_sum = BlockReduceT(tmp).Sum(local);
    if (threadIdx.x == 0) {
        partial_sum[linear_block] = block_sum;
    }
}

__global__ void kSumUpdateSqVec2Half(
    const half* __restrict__ g,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    float* __restrict__ partial_sum,
    const int n,
    const int rows,
    const int cols
) {
    float local = 0.0f;
    const int stride = (int)(gridDim.x * blockDim.x) * 2;
    for (int base = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 2; base < n; base += stride) {
        const int row = base / cols;
        const int col = base - row * cols; // even if cols is even

        const half2 gg = reinterpret_cast<const half2*>(g + base)[0];
        const float2 gf = __half22float2(gg);

        const float rf = r_factor[row];
        const float u0 = gf.x * rf * c_factor[col + 0];
        const float u1 = gf.y * rf * c_factor[col + 1];
        local += (u0 * u0) + (u1 * u1);
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_sum = BlockReduceT(tmp).Sum(local);
    if (threadIdx.x == 0)
        partial_sum[blockIdx.x] = block_sum;
}

__global__ void kSumUpdateSqVec2Bf16(
    const __nv_bfloat16* __restrict__ g,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    float* __restrict__ partial_sum,
    const int n,
    const int rows,
    const int cols
) {
    float local = 0.0f;
    const int stride = (int)(gridDim.x * blockDim.x) * 2;
    for (int base = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 2; base < n; base += stride) {
        const int row = base / cols;
        const int col = base - row * cols; // even if cols is even

        const __nv_bfloat162 gg = reinterpret_cast<const __nv_bfloat162*>(g + base)[0];
        const float2 gf = __bfloat1622float2(gg);

        const float rf = r_factor[row];
        const float u0 = gf.x * rf * c_factor[col + 0];
        const float u1 = gf.y * rf * c_factor[col + 1];
        local += (u0 * u0) + (u1 * u1);
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_sum = BlockReduceT(tmp).Sum(local);
    if (threadIdx.x == 0)
        partial_sum[blockIdx.x] = block_sum;
}

__global__ void kReduceVectorPartial(
    const float* __restrict__ vec,
    float* __restrict__ partial_sum,
    const int n
) {
    float local = 0.0f;
    const int start = (int)blockIdx.x * THREADS;
    for (int idx = start + (int)threadIdx.x; idx < n && idx < (start + THREADS); idx += THREADS) {
        local += vec[idx];
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_sum = BlockReduceT(tmp).Sum(local);
    if (threadIdx.x == 0) {
        partial_sum[blockIdx.x] = block_sum;
    }
}

__global__ void kReduceVectorPartialBatched(
    const float* __restrict__ vec,
    float* __restrict__ partial_sum,
    const int n,
    const int partials_per_item
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / partials_per_item;
    const int item_block = linear_block - item * partials_per_item;
    const int start = item_block * THREADS;
    const int64_t item_offset = (int64_t)item * n;

    float local = 0.0f;
    for (int idx = start + (int)threadIdx.x; idx < n && idx < (start + THREADS); idx += THREADS) {
        local += vec[item_offset + idx];
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_sum = BlockReduceT(tmp).Sum(local);
    if (threadIdx.x == 0) {
        partial_sum[linear_block] = block_sum;
    }
}

__global__ void kReducePartialSum(
    const float* __restrict__ partial_sum,
    float* __restrict__ sum_out,
    const int n
) {
    float local = 0.0f;
    for (int idx = (int)threadIdx.x; idx < n; idx += THREADS) {
        local += partial_sum[idx];
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float total = BlockReduceT(tmp).Sum(local);
    if (threadIdx.x == 0) {
        sum_out[0] = total;
    }
}

__global__ void kReducePartialSumBatched(
    const float* __restrict__ partial_sum,
    float* __restrict__ sum_out,
    const int partials_per_item
) {
    const int item = (int)blockIdx.x;
    const int64_t item_offset = (int64_t)item * partials_per_item;
    float local = 0.0f;
    for (int idx = (int)threadIdx.x; idx < partials_per_item; idx += THREADS) {
        local += partial_sum[item_offset + idx];
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float total = BlockReduceT(tmp).Sum(local);
    if (threadIdx.x == 0) {
        sum_out[item] = total;
    }
}

__global__ void kReducePartialSumAccumulate(
    const float* __restrict__ partial_sum,
    float* __restrict__ dst,
    const int n
) {
    float local = 0.0f;
    for (int idx = (int)threadIdx.x; idx < n; idx += THREADS) {
        local += partial_sum[idx];
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float total = BlockReduceT(tmp).Sum(local);
    if (threadIdx.x == 0) {
        dst[0] += total;
    }
}

__global__ void kStoreScaledScalar(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const float scale
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        dst[0] = src[0] / scale;
    }
}

__device__ __forceinline__ float dequant_i8(const int8_t q, const float absmax) {
    return ((float)q) * (absmax / QMAX);
}

__device__ __forceinline__ int8_t quant_i8(const float x, const float inv_absmax) {
    float q = nearbyintf(x * inv_absmax * QMAX);
    q = fminf(QMAX, fmaxf(-QMAX, q));
    return (int8_t)q;
}

__device__ __forceinline__ float dequant_u8(const uint8_t q, const float absmax) {
    return ((float)q) * (absmax / UQMAX);
}

__device__ __forceinline__ uint8_t quant_u8(const float x, const float inv_absmax) {
    float q = nearbyintf(x * inv_absmax * UQMAX);
    q = fminf(UQMAX, fmaxf(0.0f, q));
    return (uint8_t)q;
}

template <bool Signed>
__global__ void kBlockwiseQuantF32(
    const float* __restrict__ src,
    typename std::conditional<Signed, int8_t, uint8_t>::type* __restrict__ q_out,
    float* __restrict__ absmax_out,
    const int64_t numel,
    const int block_size
) {
    const int block = (int)blockIdx.x;
    const int64_t start = (int64_t)block * block_size;
    if (start >= numel)
        return;

    float local_max = 0.0f;
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int64_t idx = start + offset;
        if (idx < numel) {
            const float v = src[idx];
            if constexpr (Signed) {
                local_max = fmaxf(local_max, fabsf(v));
            } else {
                local_max = fmaxf(local_max, v);
            }
        }
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_max = BlockReduceT(tmp).Reduce(local_max, cub::Max());

    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        absmax_out[block] = shared_absmax;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int64_t idx = start + offset;
        if (idx < numel) {
            const float v = src[idx];
            if constexpr (Signed) {
                q_out[idx] = quant_i8(v, inv_absmax);
            } else {
                q_out[idx] = quant_u8(v, inv_absmax);
            }
        }
    }
}

template <bool Signed>
__global__ void kBlockwiseDequantF32(
    float* __restrict__ out,
    const typename std::conditional<Signed, int8_t, uint8_t>::type* __restrict__ q_in,
    const float* __restrict__ absmax,
    const int64_t numel,
    const int block_size
) {
    const int block = (int)blockIdx.x;
    const int64_t start = (int64_t)block * block_size;
    if (start >= numel)
        return;

    const float block_absmax = absmax[block];
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int64_t idx = start + offset;
        if (idx < numel) {
            if constexpr (Signed) {
                out[idx] = dequant_i8(q_in[idx], block_absmax);
            } else {
                out[idx] = dequant_u8(q_in[idx], block_absmax);
            }
        }
    }
}

template <bool Signed>
__global__ void kBlockwiseQuantF32Batched(
    const float* __restrict__ src,
    typename std::conditional<Signed, int8_t, uint8_t>::type* __restrict__ q_out,
    float* __restrict__ absmax_out,
    const int64_t per_item_numel,
    const int blocks_per_item,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / blocks_per_item;
    const int item_block = linear_block - item * blocks_per_item;
    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t start = item_offset + (int64_t)item_block * block_size;
    const int64_t item_end = item_offset + per_item_numel;
    if (start >= item_end)
        return;

    float local_max = 0.0f;
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int64_t idx = start + offset;
        if (idx < item_end) {
            const float v = src[idx];
            if constexpr (Signed) {
                local_max = fmaxf(local_max, fabsf(v));
            } else {
                local_max = fmaxf(local_max, v);
            }
        }
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_max = BlockReduceT(tmp).Reduce(local_max, cub::Max());

    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        absmax_out[linear_block] = shared_absmax;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int64_t idx = start + offset;
        if (idx < item_end) {
            const float v = src[idx];
            if constexpr (Signed) {
                q_out[idx] = quant_i8(v, inv_absmax);
            } else {
                q_out[idx] = quant_u8(v, inv_absmax);
            }
        }
    }
}

template <bool Signed>
__global__ void kBlockwiseDequantF32Batched(
    float* __restrict__ out,
    const typename std::conditional<Signed, int8_t, uint8_t>::type* __restrict__ q_in,
    const float* __restrict__ absmax,
    const int64_t per_item_numel,
    const int blocks_per_item,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / blocks_per_item;
    const int item_block = linear_block - item * blocks_per_item;
    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t start = item_offset + (int64_t)item_block * block_size;
    const int64_t item_end = item_offset + per_item_numel;
    if (start >= item_end)
        return;

    const float block_absmax = absmax[linear_block];
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int64_t idx = start + offset;
        if (idx < item_end) {
            if constexpr (Signed) {
                out[idx] = dequant_i8(q_in[idx], block_absmax);
            } else {
                out[idx] = dequant_u8(q_in[idx], block_absmax);
            }
        }
    }
}

__global__ void kCameNonfactoredSqUpdateAndQuant(
    const float* __restrict__ g32,
    uint8_t* __restrict__ exp_avg_sq_q,
    float* __restrict__ exp_avg_sq_absmax,
    float* __restrict__ update_out,
    float* __restrict__ partial_sum,
    const float beta2,
    const float eps0,
    const int64_t numel,
    const int block_size
) {
    const int block = (int)blockIdx.x;
    const int64_t start = (int64_t)block * block_size;
    if (start >= numel)
        return;

    const float old_absmax = exp_avg_sq_absmax[block];
    const int thread = (int)threadIdx.x;
    const int64_t idx = start + thread;
    const bool valid = thread < block_size && idx < numel;

    float new_sq = 0.0f;
    float u = 0.0f;
    if (valid) {
        const float g = g32[idx];
        const float old_sq = dequant_u8(exp_avg_sq_q[idx], old_absmax);
        new_sq = (old_sq * beta2) + ((1.0f - beta2) * ((g * g) + eps0));
        u = rsqrtf(fmaxf(new_sq, 1e-30f)) * g;
        update_out[idx] = u;
    }

    const float local_max = valid ? new_sq : 0.0f;
    const float local_sum = valid ? (u * u) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __syncthreads();
    __shared__ typename BlockReduceT::TempStorage sum_tmp;
    const float block_sum = BlockReduceT(sum_tmp).Sum(local_sum);

    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_sq_absmax[block] = shared_absmax;
        partial_sum[block] = block_sum;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    if (valid) {
        exp_avg_sq_q[idx] = quant_u8(new_sq, inv_absmax);
    }
}

template <typename UpdateT>
__global__ void kCameNonfactoredSqUpdateAndQuantTypedUpdate(
    const float* __restrict__ g32,
    uint8_t* __restrict__ exp_avg_sq_q,
    float* __restrict__ exp_avg_sq_absmax,
    UpdateT* __restrict__ update_out,
    float* __restrict__ partial_sum,
    const float beta2,
    const float eps0,
    const int64_t numel,
    const int block_size
) {
    const int block = (int)blockIdx.x;
    const int64_t start = (int64_t)block * block_size;
    if (start >= numel)
        return;

    const float old_absmax = exp_avg_sq_absmax[block];
    const int thread = (int)threadIdx.x;
    const int64_t idx = start + thread;
    const bool valid = thread < block_size && idx < numel;

    float new_sq = 0.0f;
    float u = 0.0f;
    if (valid) {
        const float g = g32[idx];
        const float old_sq = dequant_u8(exp_avg_sq_q[idx], old_absmax);
        new_sq = (old_sq * beta2) + ((1.0f - beta2) * ((g * g) + eps0));
        u = rsqrtf(fmaxf(new_sq, 1e-30f)) * g;
        update_out[idx] = from_float_t<UpdateT>(u);
    }

    const float local_max = valid ? new_sq : 0.0f;
    const float local_sum = valid ? (u * u) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __syncthreads();
    __shared__ typename BlockReduceT::TempStorage sum_tmp;
    const float block_sum = BlockReduceT(sum_tmp).Sum(local_sum);

    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_sq_absmax[block] = shared_absmax;
        partial_sum[block] = block_sum;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    if (valid) {
        exp_avg_sq_q[idx] = quant_u8(new_sq, inv_absmax);
    }
}

__global__ void kUpdateMeanRowQuantized(
    const float* __restrict__ src,
    const uint8_t* __restrict__ q_in,
    const float* __restrict__ absmax_in,
    float* __restrict__ updated_row,
    float* __restrict__ next_absmax,
    const float beta,
    const int rows,
    const int cols,
    const int block_size
) {
    const int row = (int)blockIdx.x;
    if (row >= rows)
        return;

    float local = 0.0f;
    for (int col = (int)threadIdx.x; col < cols; col += THREADS) {
        local += src[row * cols + col];
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float row_sum = BlockReduceT(tmp).Sum(local);

    if (threadIdx.x == 0) {
        const float mean = row_sum / (float)cols;
        const int qblock = row / block_size;
        const float old = dequant_u8(q_in[row], absmax_in[qblock]);
        const float updated = (old * beta) + ((1.0f - beta) * mean);
        updated_row[row] = updated;
        atomic_max_positive_f32(next_absmax + qblock, updated);
    }
}

template <typename SrcT>
__global__ void kUpdateMeanRowQuantizedBatched(
    const SrcT* __restrict__ src,
    const uint8_t* __restrict__ q_in,
    const float* __restrict__ absmax_in,
    float* __restrict__ updated_row,
    float* __restrict__ next_absmax,
    const float beta,
    const int64_t per_item_numel,
    const int rows,
    const int cols,
    const int row_q_blocks,
    const int block_size
) {
    const int linear_row = (int)blockIdx.x;
    const int item = linear_row / rows;
    const int row = linear_row - item * rows;
    if (row >= rows)
        return;

    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t row_offset = (int64_t)item * rows;
    const int64_t row_absmax_offset = (int64_t)item * row_q_blocks;

    float local = 0.0f;
    for (int col = (int)threadIdx.x; col < cols; col += THREADS) {
        local += to_float(src[item_offset + (int64_t)row * cols + col]);
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float row_sum = BlockReduceT(tmp).Sum(local);

    if (threadIdx.x == 0) {
        const float mean = row_sum / (float)cols;
        const int qblock = row / block_size;
        const float old = dequant_u8(q_in[row_offset + row], absmax_in[row_absmax_offset + qblock]);
        const float updated = (old * beta) + ((1.0f - beta) * mean);
        updated_row[row_offset + row] = updated;
        atomic_max_positive_f32(next_absmax + row_absmax_offset + qblock, updated);
    }
}

template <typename SrcT>
__global__ void kUpdateMeanRowQuantizedBatchedAccum(
    const SrcT* __restrict__ src,
    const uint8_t* __restrict__ q_in,
    const float* __restrict__ absmax_in,
    float* __restrict__ updated_row,
    float* __restrict__ next_absmax,
    float* __restrict__ sum_row_state,
    const float beta,
    const int64_t per_item_numel,
    const int rows,
    const int cols,
    const int row_q_blocks,
    const int block_size
) {
    const int linear_row = (int)blockIdx.x;
    const int item = linear_row / rows;
    const int row = linear_row - item * rows;
    if (row >= rows)
        return;

    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t row_offset = (int64_t)item * rows;
    const int64_t row_absmax_offset = (int64_t)item * row_q_blocks;

    float local = 0.0f;
    for (int col = (int)threadIdx.x; col < cols; col += THREADS) {
        local += to_float(src[item_offset + (int64_t)row * cols + col]);
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float row_sum = BlockReduceT(tmp).Sum(local);

    if (threadIdx.x == 0) {
        const float mean = row_sum / (float)cols;
        const int qblock = row / block_size;
        const float old = dequant_u8(q_in[row_offset + row], absmax_in[row_absmax_offset + qblock]);
        const float updated = (old * beta) + ((1.0f - beta) * mean);
        updated_row[row_offset + row] = updated;
        atomic_max_positive_f32(next_absmax + row_absmax_offset + qblock, updated);
        atomicAdd(sum_row_state + item, updated);
    }
}

__global__ void kUpdateMeanColQuantizedFinalize(
    const float* __restrict__ src,
    uint8_t* __restrict__ q_out,
    float* __restrict__ absmax_out,
    float* __restrict__ c_factor,
    const float beta,
    const int rows,
    const int cols,
    const int block_size
) {
    const int block = (int)blockIdx.x;
    const int start = block * block_size;
    const int col = start + (int)threadIdx.x;
    const bool valid = col < cols;

    float updated = 0.0f;
    if (valid) {
        float local = 0.0f;
        for (int row = 0; row < rows; ++row) {
            local += src[row * cols + col];
        }
        const float mean = local / (float)rows;
        const float old = dequant_u8(q_out[col], absmax_out[block]);
        updated = (old * beta) + ((1.0f - beta) * mean);
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_absmax = BlockReduceT(tmp).Reduce(updated, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        absmax_out[block] = shared_absmax;
    }
    __syncthreads();

    if (valid) {
        const float inv_absmax = 1.0f / shared_absmax;
        q_out[col] = quant_u8(updated, inv_absmax);
        c_factor[col] = rsqrtf(fmaxf(updated, 1e-30f));
    }
}

template <typename SrcT>
__global__ void kUpdateMeanColQuantizedFinalizeBatched(
    const SrcT* __restrict__ src,
    uint8_t* __restrict__ q_out,
    float* __restrict__ absmax_out,
    float* __restrict__ c_factor,
    const float beta,
    const int64_t per_item_numel,
    const int rows,
    const int cols,
    const int col_q_blocks,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / col_q_blocks;
    const int item_block = linear_block - item * col_q_blocks;
    const int start = item_block * block_size;
    const int col = start + (int)threadIdx.x;
    const bool valid = col < cols;
    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t col_offset = (int64_t)item * cols;

    float updated = 0.0f;
    if (valid) {
        float local = 0.0f;
        for (int row = 0; row < rows; ++row) {
            local += to_float(src[item_offset + (int64_t)row * cols + col]);
        }
        const float mean = local / (float)rows;
        const float old = dequant_u8(q_out[col_offset + col], absmax_out[linear_block]);
        updated = (old * beta) + ((1.0f - beta) * mean);
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_absmax = BlockReduceT(tmp).Reduce(updated, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        absmax_out[linear_block] = shared_absmax;
    }
    __syncthreads();

    if (valid) {
        const float inv_absmax = 1.0f / shared_absmax;
        q_out[col_offset + col] = quant_u8(updated, inv_absmax);
        c_factor[col_offset + col] = rsqrtf(fmaxf(updated, 1e-30f));
    }
}

template <typename SrcT>
__global__ void kUpdateMeanRowQuantizedMultiTensorSameShape(
    const SrcT* __restrict__ src,
    const int64_t* __restrict__ q_in_ptrs,
    const int64_t* __restrict__ absmax_in_ptrs,
    float* __restrict__ updated_row,
    float* __restrict__ next_absmax,
    const float beta,
    const int64_t per_item_numel,
    const int rows,
    const int cols,
    const int row_q_blocks,
    const int block_size
) {
    const int linear_row = (int)blockIdx.x;
    const int item = linear_row / rows;
    const int row = linear_row - item * rows;
    if (row >= rows)
        return;

    const auto* q_in = reinterpret_cast<const uint8_t*>(q_in_ptrs[item]);
    const auto* absmax_in = reinterpret_cast<const float*>(absmax_in_ptrs[item]);
    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t row_offset = (int64_t)item * rows;
    const int64_t row_absmax_offset = (int64_t)item * row_q_blocks;

    float local = 0.0f;
    for (int col = (int)threadIdx.x; col < cols; col += THREADS) {
        local += to_float(src[item_offset + (int64_t)row * cols + col]);
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float row_sum = BlockReduceT(tmp).Sum(local);

    if (threadIdx.x == 0) {
        const float mean = row_sum / (float)cols;
        const int qblock = row / block_size;
        const float old = dequant_u8(q_in[row], absmax_in[qblock]);
        const float updated = (old * beta) + ((1.0f - beta) * mean);
        updated_row[row_offset + row] = updated;
        atomic_max_positive_f32(next_absmax + row_absmax_offset + qblock, updated);
    }
}

template <typename SrcT>
__global__ void kUpdateMeanColQuantizedFinalizeMultiTensorSameShape(
    const SrcT* __restrict__ src,
    const int64_t* __restrict__ q_out_ptrs,
    const int64_t* __restrict__ absmax_out_ptrs,
    float* __restrict__ c_factor,
    const float beta,
    const int64_t per_item_numel,
    const int rows,
    const int cols,
    const int col_q_blocks,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / col_q_blocks;
    const int item_block = linear_block - item * col_q_blocks;
    const int start = item_block * block_size;
    const int col = start + (int)threadIdx.x;
    const bool valid = col < cols;
    auto* q_out = reinterpret_cast<uint8_t*>(q_out_ptrs[item]);
    auto* absmax_out = reinterpret_cast<float*>(absmax_out_ptrs[item]);
    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t col_offset = (int64_t)item * cols;

    float updated = 0.0f;
    if (valid) {
        float local = 0.0f;
        for (int row = 0; row < rows; ++row) {
            local += to_float(src[item_offset + (int64_t)row * cols + col]);
        }
        const float mean = local / (float)rows;
        const float old = dequant_u8(q_out[col], absmax_out[item_block]);
        updated = (old * beta) + ((1.0f - beta) * mean);
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_absmax = BlockReduceT(tmp).Reduce(updated, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        absmax_out[item_block] = shared_absmax;
    }
    __syncthreads();

    if (valid) {
        const float inv_absmax = 1.0f / shared_absmax;
        q_out[col] = quant_u8(updated, inv_absmax);
        c_factor[col_offset + col] = rsqrtf(fmaxf(updated, 1e-30f));
    }
}

__global__ void kUpdateMeanRowFp32(
    const float* __restrict__ src,
    float* __restrict__ exp_avg_res_row,
    const float beta3,
    const int rows,
    const int cols
) {
    const int row = (int)blockIdx.x;
    if (row >= rows)
        return;

    float local = 0.0f;
    for (int col = (int)threadIdx.x; col < cols; col += THREADS) {
        local += src[row * cols + col];
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float row_sum = BlockReduceT(tmp).Sum(local);

    if (threadIdx.x == 0) {
        const float mean = row_sum / (float)cols;
        exp_avg_res_row[row] = (exp_avg_res_row[row] * beta3) + ((1.0f - beta3) * mean);
    }
}

__global__ void kUpdateMeanColFinalizeFp32(
    const float* __restrict__ src,
    float* __restrict__ exp_avg_res_col,
    float* __restrict__ c_factor,
    const float beta3,
    const int rows,
    const int cols
) {
    const int col = (int)blockIdx.x * THREADS + (int)threadIdx.x;
    const bool valid = col < cols;

    float updated = 0.0f;
    if (valid) {
        float local = 0.0f;
        for (int row = 0; row < rows; ++row) {
            local += src[row * cols + col];
        }
        const float mean = local / (float)rows;
        updated = (exp_avg_res_col[col] * beta3) + ((1.0f - beta3) * mean);
    }

    if (valid) {
        exp_avg_res_col[col] = updated;
        c_factor[col] = rsqrtf(fmaxf(updated, 1e-30f));
    }
}

template <typename T>
__global__ void kCameNonfactoredExpAvgParamUpdate(
    T* __restrict__ p,
    const float* __restrict__ update_in,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_absmax,
    const float* __restrict__ sum_update,
    const float beta1,
    const float lr,
    const float clip_threshold,
    const float weight_decay,
    const int64_t numel,
    const int block_size
) {
    const int block = (int)blockIdx.x;
    const int64_t start = (int64_t)block * block_size;
    if (start >= numel)
        return;

    const float old_absmax = exp_avg_absmax[block];
    const float rms = sqrtf(sum_update[0] / (float)numel);
    const float clip = fmaxf(1.0f, rms / clip_threshold);
    const int thread = (int)threadIdx.x;
    const int64_t idx = start + thread;
    const bool valid = thread < block_size && idx < numel;

    float new_avg = 0.0f;
    if (valid) {
        const float old_avg = dequant_i8(exp_avg_q[idx], old_absmax);
        const float u = update_in[idx] / clip;
        new_avg = (old_avg * beta1) + ((1.0f - beta1) * u);
    }

    const float local_max = valid ? fabsf(new_avg) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_absmax[block] = shared_absmax;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    if (valid) {
        float p_val = to_float(p[idx]);
        if (weight_decay != 0.0f) {
            p_val -= weight_decay * lr * p_val;
        }
        p_val -= lr * new_avg;
        p[idx] = from_float_t<T>(p_val);
        exp_avg_q[idx] = quant_i8(new_avg, inv_absmax);
    }
}

template <typename T, typename UpdateT>
__global__ void kCameNonfactoredExpAvgParamUpdateTypedUpdate(
    T* __restrict__ p,
    const UpdateT* __restrict__ update_in,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_absmax,
    const float* __restrict__ sum_update,
    const float beta1,
    const float lr,
    const float clip_threshold,
    const float weight_decay,
    const int64_t numel,
    const int block_size
) {
    const int block = (int)blockIdx.x;
    const int64_t start = (int64_t)block * block_size;
    if (start >= numel)
        return;

    const float old_absmax = exp_avg_absmax[block];
    const float rms = sqrtf(sum_update[0] / (float)numel);
    const float clip = fmaxf(1.0f, rms / clip_threshold);
    const int thread = (int)threadIdx.x;
    const int64_t idx = start + thread;
    const bool valid = thread < block_size && idx < numel;

    float new_avg = 0.0f;
    if (valid) {
        const float old_avg = dequant_i8(exp_avg_q[idx], old_absmax);
        const float u = to_float(update_in[idx]) / clip;
        new_avg = (old_avg * beta1) + ((1.0f - beta1) * u);
    }

    const float local_max = valid ? fabsf(new_avg) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_absmax[block] = shared_absmax;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    if (valid) {
        float p_val = to_float(p[idx]);
        if (weight_decay != 0.0f) {
            p_val -= weight_decay * lr * p_val;
        }
        p_val -= lr * new_avg;
        p[idx] = from_float_t<T>(p_val);
        exp_avg_q[idx] = quant_i8(new_avg, inv_absmax);
    }
}

__global__ void kCameNonfactoredSqUpdateAndQuantBatched(
    const float* __restrict__ g32,
    uint8_t* __restrict__ exp_avg_sq_q,
    float* __restrict__ exp_avg_sq_absmax,
    float* __restrict__ update_out,
    float* __restrict__ partial_sum,
    const float beta2,
    const float eps0,
    const int64_t per_item_numel,
    const int blocks_per_item,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / blocks_per_item;
    const int item_block = linear_block - item * blocks_per_item;
    const int64_t start = (int64_t)item_block * block_size;
    if (start >= per_item_numel)
        return;

    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t absmax_offset = (int64_t)item * blocks_per_item;
    const float old_absmax = exp_avg_sq_absmax[absmax_offset + item_block];
    const int thread = (int)threadIdx.x;
    const int64_t item_idx = start + thread;
    const bool valid = thread < block_size && item_idx < per_item_numel;
    const int64_t idx = item_offset + item_idx;

    float new_sq = 0.0f;
    float u = 0.0f;
    if (valid) {
        const float g = g32[idx];
        const float old_sq = dequant_u8(exp_avg_sq_q[idx], old_absmax);
        new_sq = (old_sq * beta2) + ((1.0f - beta2) * ((g * g) + eps0));
        u = rsqrtf(fmaxf(new_sq, 1e-30f)) * g;
        update_out[idx] = u;
    }

    const float local_max = valid ? new_sq : 0.0f;
    const float local_sum = valid ? (u * u) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __syncthreads();
    __shared__ typename BlockReduceT::TempStorage sum_tmp;
    const float block_sum = BlockReduceT(sum_tmp).Sum(local_sum);

    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_sq_absmax[absmax_offset + item_block] = shared_absmax;
        partial_sum[linear_block] = block_sum;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    if (valid) {
        exp_avg_sq_q[idx] = quant_u8(new_sq, inv_absmax);
    }
}

template <typename T>
__global__ void kCameNonfactoredExpAvgParamUpdateBatched(
    T* __restrict__ p,
    const float* __restrict__ update_in,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_absmax,
    const float* __restrict__ sum_update,
    const float beta1,
    const float lr,
    const float clip_threshold,
    const float weight_decay,
    const int64_t per_item_numel,
    const int blocks_per_item,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / blocks_per_item;
    const int item_block = linear_block - item * blocks_per_item;
    const int64_t start = (int64_t)item_block * block_size;
    if (start >= per_item_numel)
        return;

    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t absmax_offset = (int64_t)item * blocks_per_item;
    const float old_absmax = exp_avg_absmax[absmax_offset + item_block];
    const float rms = sqrtf(sum_update[item] / (float)per_item_numel);
    const float clip = fmaxf(1.0f, rms / clip_threshold);
    const int thread = (int)threadIdx.x;
    const int64_t item_idx = start + thread;
    const bool valid = thread < block_size && item_idx < per_item_numel;
    const int64_t idx = item_offset + item_idx;

    float new_avg = 0.0f;
    if (valid) {
        const float old_avg = dequant_i8(exp_avg_q[idx], old_absmax);
        const float u = update_in[idx] / clip;
        new_avg = (old_avg * beta1) + ((1.0f - beta1) * u);
    }

    const float local_max = valid ? fabsf(new_avg) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_absmax[absmax_offset + item_block] = shared_absmax;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    if (valid) {
        float p_val = to_float(p[idx]);
        if (weight_decay != 0.0f) {
            p_val -= weight_decay * lr * p_val;
        }
        p_val -= lr * new_avg;
        p[idx] = from_float_t<T>(p_val);
        exp_avg_q[idx] = quant_i8(new_avg, inv_absmax);
    }
}

__global__ void kZeroScalarsMultiTensor(
    const int64_t* __restrict__ scalar_ptrs
) {
    const int item = (int)blockIdx.x;
    float* scalar = reinterpret_cast<float*>(scalar_ptrs[item]);
    if (threadIdx.x == 0) {
        scalar[0] = 0.0f;
    }
}

__global__ void kCameNonfactoredSqUpdateAndQuantMultiTensor(
    const int64_t* __restrict__ g32_ptrs,
    const int64_t* __restrict__ exp_avg_sq_q_ptrs,
    const int64_t* __restrict__ exp_avg_sq_absmax_ptrs,
    const int64_t* __restrict__ update_ptrs,
    const int64_t* __restrict__ sum_update_ptrs,
    const float beta2,
    const float eps0,
    const int64_t per_item_numel,
    const int blocks_per_item,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / blocks_per_item;
    const int item_block = linear_block - item * blocks_per_item;
    const int64_t start = (int64_t)item_block * block_size;
    if (start >= per_item_numel)
        return;

    const float* g32 = reinterpret_cast<const float*>(g32_ptrs[item]);
    uint8_t* exp_avg_sq_q = reinterpret_cast<uint8_t*>(exp_avg_sq_q_ptrs[item]);
    float* exp_avg_sq_absmax = reinterpret_cast<float*>(exp_avg_sq_absmax_ptrs[item]);
    float* update_out = reinterpret_cast<float*>(update_ptrs[item]);
    float* sum_update = reinterpret_cast<float*>(sum_update_ptrs[item]);
    const float old_absmax = exp_avg_sq_absmax[item_block];
    const int thread = (int)threadIdx.x;
    const int64_t item_idx = start + thread;
    const bool valid = thread < block_size && item_idx < per_item_numel;

    float new_sq = 0.0f;
    float u = 0.0f;
    if (valid) {
        const float g = g32[item_idx];
        const float old_sq = dequant_u8(exp_avg_sq_q[item_idx], old_absmax);
        new_sq = (old_sq * beta2) + ((1.0f - beta2) * ((g * g) + eps0));
        u = rsqrtf(fmaxf(new_sq, 1e-30f)) * g;
        update_out[item_idx] = u;
    }

    const float local_max = valid ? new_sq : 0.0f;
    const float local_sum = valid ? (u * u) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __syncthreads();
    __shared__ typename BlockReduceT::TempStorage sum_tmp;
    const float block_sum = BlockReduceT(sum_tmp).Sum(local_sum);

    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_sq_absmax[item_block] = shared_absmax;
        atomicAdd(sum_update, block_sum);
    }
    __syncthreads();

    if (valid) {
        exp_avg_sq_q[item_idx] = quant_u8(new_sq, 1.0f / shared_absmax);
    }
}


template <typename UpdateT>
__global__ void kCameNonfactoredSqUpdateAndQuantMultiTensorTypedUpdate(
    const int64_t* __restrict__ g32_ptrs,
    const int64_t* __restrict__ exp_avg_sq_q_ptrs,
    const int64_t* __restrict__ exp_avg_sq_absmax_ptrs,
    const int64_t* __restrict__ update_ptrs,
    const int64_t* __restrict__ sum_update_ptrs,
    const float beta2,
    const float eps0,
    const int64_t per_item_numel,
    const int blocks_per_item,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / blocks_per_item;
    const int item_block = linear_block - item * blocks_per_item;
    const int64_t start = (int64_t)item_block * block_size;
    if (start >= per_item_numel)
        return;

    const float* g32 = reinterpret_cast<const float*>(g32_ptrs[item]);
    uint8_t* exp_avg_sq_q = reinterpret_cast<uint8_t*>(exp_avg_sq_q_ptrs[item]);
    float* exp_avg_sq_absmax = reinterpret_cast<float*>(exp_avg_sq_absmax_ptrs[item]);
    UpdateT* update_out = reinterpret_cast<UpdateT*>(update_ptrs[item]);
    float* sum_update = reinterpret_cast<float*>(sum_update_ptrs[item]);
    const float old_absmax = exp_avg_sq_absmax[item_block];
    const int thread = (int)threadIdx.x;
    const int64_t item_idx = start + thread;
    const bool valid = thread < block_size && item_idx < per_item_numel;

    float new_sq = 0.0f;
    float u = 0.0f;
    if (valid) {
        const float g = g32[item_idx];
        const float old_sq = dequant_u8(exp_avg_sq_q[item_idx], old_absmax);
        new_sq = (old_sq * beta2) + ((1.0f - beta2) * ((g * g) + eps0));
        u = rsqrtf(fmaxf(new_sq, 1e-30f)) * g;
        update_out[item_idx] = from_float_t<UpdateT>(u);
    }

    const float local_max = valid ? new_sq : 0.0f;
    const float local_sum = valid ? (u * u) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __syncthreads();
    __shared__ typename BlockReduceT::TempStorage sum_tmp;
    const float block_sum = BlockReduceT(sum_tmp).Sum(local_sum);

    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_sq_absmax[item_block] = shared_absmax;
        atomicAdd(sum_update, block_sum);
    }
    __syncthreads();

    if (valid) {
        exp_avg_sq_q[item_idx] = quant_u8(new_sq, 1.0f / shared_absmax);
    }
}


template <typename T>
__global__ void kCameNonfactoredExpAvgParamUpdateMultiTensor(
    const int64_t* __restrict__ p_ptrs,
    const int64_t* __restrict__ update_ptrs,
    const int64_t* __restrict__ exp_avg_q_ptrs,
    const int64_t* __restrict__ exp_avg_absmax_ptrs,
    const int64_t* __restrict__ sum_update_ptrs,
    const float beta1,
    const float lr,
    const float clip_threshold,
    const float weight_decay,
    const int64_t per_item_numel,
    const int blocks_per_item,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / blocks_per_item;
    const int item_block = linear_block - item * blocks_per_item;
    const int64_t start = (int64_t)item_block * block_size;
    if (start >= per_item_numel)
        return;

    T* p = reinterpret_cast<T*>(p_ptrs[item]);
    const float* update_in = reinterpret_cast<const float*>(update_ptrs[item]);
    int8_t* exp_avg_q = reinterpret_cast<int8_t*>(exp_avg_q_ptrs[item]);
    float* exp_avg_absmax = reinterpret_cast<float*>(exp_avg_absmax_ptrs[item]);
    const float* sum_update = reinterpret_cast<const float*>(sum_update_ptrs[item]);
    const float old_absmax = exp_avg_absmax[item_block];
    const float rms = sqrtf(sum_update[0] / (float)per_item_numel);
    const float clip = fmaxf(1.0f, rms / clip_threshold);
    const int thread = (int)threadIdx.x;
    const int64_t item_idx = start + thread;
    const bool valid = thread < block_size && item_idx < per_item_numel;

    float new_avg = 0.0f;
    if (valid) {
        const float old_avg = dequant_i8(exp_avg_q[item_idx], old_absmax);
        const float u = update_in[item_idx] / clip;
        new_avg = (old_avg * beta1) + ((1.0f - beta1) * u);
    }

    const float local_max = valid ? fabsf(new_avg) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_absmax[item_block] = shared_absmax;
    }
    __syncthreads();

    if (valid) {
        float p_val = to_float(p[item_idx]);
        if (weight_decay != 0.0f) {
            p_val -= weight_decay * lr * p_val;
        }
        p_val -= lr * new_avg;
        p[item_idx] = from_float_t<T>(p_val);
        exp_avg_q[item_idx] = quant_i8(new_avg, 1.0f / shared_absmax);
    }
}

template <typename T, typename UpdateT>
__global__ void kCameNonfactoredExpAvgParamUpdateMultiTensorTypedUpdate(
    const int64_t* __restrict__ p_ptrs,
    const int64_t* __restrict__ update_ptrs,
    const int64_t* __restrict__ exp_avg_q_ptrs,
    const int64_t* __restrict__ exp_avg_absmax_ptrs,
    const int64_t* __restrict__ sum_update_ptrs,
    const float beta1,
    const float lr,
    const float clip_threshold,
    const float weight_decay,
    const int64_t per_item_numel,
    const int blocks_per_item,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / blocks_per_item;
    const int item_block = linear_block - item * blocks_per_item;
    const int64_t start = (int64_t)item_block * block_size;
    if (start >= per_item_numel)
        return;

    T* p = reinterpret_cast<T*>(p_ptrs[item]);
    const UpdateT* update_in = reinterpret_cast<const UpdateT*>(update_ptrs[item]);
    int8_t* exp_avg_q = reinterpret_cast<int8_t*>(exp_avg_q_ptrs[item]);
    float* exp_avg_absmax = reinterpret_cast<float*>(exp_avg_absmax_ptrs[item]);
    const float* sum_update = reinterpret_cast<const float*>(sum_update_ptrs[item]);
    const float old_absmax = exp_avg_absmax[item_block];
    const float rms = sqrtf(sum_update[0] / (float)per_item_numel);
    const float clip = fmaxf(1.0f, rms / clip_threshold);
    const int thread = (int)threadIdx.x;
    const int64_t item_idx = start + thread;
    const bool valid = thread < block_size && item_idx < per_item_numel;

    float new_avg = 0.0f;
    if (valid) {
        const float old_avg = dequant_i8(exp_avg_q[item_idx], old_absmax);
        const float u = to_float(update_in[item_idx]) / clip;
        new_avg = (old_avg * beta1) + ((1.0f - beta1) * u);
    }

    const float local_max = valid ? fabsf(new_avg) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_absmax[item_block] = shared_absmax;
    }
    __syncthreads();

    if (valid) {
        float p_val = to_float(p[item_idx]);
        if (weight_decay != 0.0f) {
            p_val -= weight_decay * lr * p_val;
        }
        p_val -= lr * new_avg;
        p[item_idx] = from_float_t<T>(p_val);
        exp_avg_q[item_idx] = quant_i8(new_avg, 1.0f / shared_absmax);
    }
}

__global__ void kCameNonfactoredSqUpdateAndQuantMultiTensorVarlen(
    const int64_t* __restrict__ g32_ptrs,
    const int64_t* __restrict__ exp_avg_sq_q_ptrs,
    const int64_t* __restrict__ exp_avg_sq_absmax_ptrs,
    const int64_t* __restrict__ update_ptrs,
    const int64_t* __restrict__ sum_update_ptrs,
    const int64_t* __restrict__ item_numels,
    const float beta2,
    const float eps0,
    const int block_size
) {
    const int item = (int)blockIdx.y;
    const int item_block = (int)blockIdx.x;
    const int64_t per_item_numel = item_numels[item];
    const int64_t start = (int64_t)item_block * block_size;
    if (start >= per_item_numel)
        return;

    const float* g32 = reinterpret_cast<const float*>(g32_ptrs[item]);
    uint8_t* exp_avg_sq_q = reinterpret_cast<uint8_t*>(exp_avg_sq_q_ptrs[item]);
    float* exp_avg_sq_absmax = reinterpret_cast<float*>(exp_avg_sq_absmax_ptrs[item]);
    float* update_out = reinterpret_cast<float*>(update_ptrs[item]);
    float* sum_update = reinterpret_cast<float*>(sum_update_ptrs[item]);
    const float old_absmax = exp_avg_sq_absmax[item_block];
    const int thread = (int)threadIdx.x;
    const int64_t item_idx = start + thread;
    const bool valid = thread < block_size && item_idx < per_item_numel;

    float new_sq = 0.0f;
    float u = 0.0f;
    if (valid) {
        const float g = g32[item_idx];
        const float old_sq = dequant_u8(exp_avg_sq_q[item_idx], old_absmax);
        new_sq = (old_sq * beta2) + ((1.0f - beta2) * ((g * g) + eps0));
        u = rsqrtf(fmaxf(new_sq, 1e-30f)) * g;
        update_out[item_idx] = u;
    }

    const float local_max = valid ? new_sq : 0.0f;
    const float local_sum = valid ? (u * u) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __syncthreads();
    __shared__ typename BlockReduceT::TempStorage sum_tmp;
    const float block_sum = BlockReduceT(sum_tmp).Sum(local_sum);

    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_sq_absmax[item_block] = shared_absmax;
        atomicAdd(sum_update, block_sum);
    }
    __syncthreads();

    if (valid) {
        exp_avg_sq_q[item_idx] = quant_u8(new_sq, 1.0f / shared_absmax);
    }
}

template <typename T>
__global__ void kCameNonfactoredExpAvgParamUpdateMultiTensorVarlen(
    const int64_t* __restrict__ p_ptrs,
    const int64_t* __restrict__ update_ptrs,
    const int64_t* __restrict__ exp_avg_q_ptrs,
    const int64_t* __restrict__ exp_avg_absmax_ptrs,
    const int64_t* __restrict__ sum_update_ptrs,
    const int64_t* __restrict__ item_numels,
    const float beta1,
    const float lr,
    const float clip_threshold,
    const float weight_decay,
    const int block_size
) {
    const int item = (int)blockIdx.y;
    const int item_block = (int)blockIdx.x;
    const int64_t per_item_numel = item_numels[item];
    const int64_t start = (int64_t)item_block * block_size;
    if (start >= per_item_numel)
        return;

    T* p = reinterpret_cast<T*>(p_ptrs[item]);
    const float* update_in = reinterpret_cast<const float*>(update_ptrs[item]);
    int8_t* exp_avg_q = reinterpret_cast<int8_t*>(exp_avg_q_ptrs[item]);
    float* exp_avg_absmax = reinterpret_cast<float*>(exp_avg_absmax_ptrs[item]);
    const float* sum_update = reinterpret_cast<const float*>(sum_update_ptrs[item]);
    const float old_absmax = exp_avg_absmax[item_block];
    const float rms = sqrtf(sum_update[0] / (float)per_item_numel);
    const float clip = fmaxf(1.0f, rms / clip_threshold);
    const int thread = (int)threadIdx.x;
    const int64_t item_idx = start + thread;
    const bool valid = thread < block_size && item_idx < per_item_numel;

    float new_avg = 0.0f;
    if (valid) {
        const float old_avg = dequant_i8(exp_avg_q[item_idx], old_absmax);
        const float u = update_in[item_idx] / clip;
        new_avg = (old_avg * beta1) + ((1.0f - beta1) * u);
    }

    const float local_max = valid ? fabsf(new_avg) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_absmax[item_block] = shared_absmax;
    }
    __syncthreads();

    if (valid) {
        float p_val = to_float(p[item_idx]);
        if (weight_decay != 0.0f) {
            p_val -= weight_decay * lr * p_val;
        }
        p_val -= lr * new_avg;
        p[item_idx] = from_float_t<T>(p_val);
        exp_avg_q[item_idx] = quant_i8(new_avg, 1.0f / shared_absmax);
    }
}

__global__ void kCameNonfactoredSqUpdateAndQuantMultiTensorCompactVarlen(
    const int64_t* __restrict__ g32_ptrs,
    const int64_t* __restrict__ exp_avg_sq_q_ptrs,
    const int64_t* __restrict__ exp_avg_sq_absmax_ptrs,
    const int64_t* __restrict__ update_ptrs,
    const int64_t* __restrict__ sum_update_ptrs,
    const int64_t* __restrict__ item_numels,
    const int64_t* __restrict__ block_item_ids,
    const int64_t* __restrict__ block_item_starts,
    const float beta2,
    const float eps0,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = (int)block_item_ids[linear_block];
    const int64_t start = block_item_starts[linear_block];
    const int64_t per_item_numel = item_numels[item];
    if (start >= per_item_numel)
        return;

    const int item_block = (int)(start / block_size);
    const float* g32 = reinterpret_cast<const float*>(g32_ptrs[item]);
    uint8_t* exp_avg_sq_q = reinterpret_cast<uint8_t*>(exp_avg_sq_q_ptrs[item]);
    float* exp_avg_sq_absmax = reinterpret_cast<float*>(exp_avg_sq_absmax_ptrs[item]);
    float* update_out = reinterpret_cast<float*>(update_ptrs[item]);
    float* sum_update = reinterpret_cast<float*>(sum_update_ptrs[item]);
    const float old_absmax = exp_avg_sq_absmax[item_block];
    const int thread = (int)threadIdx.x;
    const int64_t item_idx = start + thread;
    const bool valid = thread < block_size && item_idx < per_item_numel;

    float new_sq = 0.0f;
    float u = 0.0f;
    if (valid) {
        const float g = g32[item_idx];
        const float old_sq = dequant_u8(exp_avg_sq_q[item_idx], old_absmax);
        new_sq = (old_sq * beta2) + ((1.0f - beta2) * ((g * g) + eps0));
        u = rsqrtf(fmaxf(new_sq, 1e-30f)) * g;
        update_out[item_idx] = u;
    }

    const float local_max = valid ? new_sq : 0.0f;
    const float local_sum = valid ? (u * u) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __syncthreads();
    __shared__ typename BlockReduceT::TempStorage sum_tmp;
    const float block_sum = BlockReduceT(sum_tmp).Sum(local_sum);

    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_sq_absmax[item_block] = shared_absmax;
        atomicAdd(sum_update, block_sum);
    }
    __syncthreads();

    if (valid) {
        exp_avg_sq_q[item_idx] = quant_u8(new_sq, 1.0f / shared_absmax);
    }
}

template <typename T>
__global__ void kCameNonfactoredExpAvgParamUpdateMultiTensorCompactVarlen(
    const int64_t* __restrict__ p_ptrs,
    const int64_t* __restrict__ update_ptrs,
    const int64_t* __restrict__ exp_avg_q_ptrs,
    const int64_t* __restrict__ exp_avg_absmax_ptrs,
    const int64_t* __restrict__ sum_update_ptrs,
    const int64_t* __restrict__ item_numels,
    const int64_t* __restrict__ block_item_ids,
    const int64_t* __restrict__ block_item_starts,
    const float beta1,
    const float lr,
    const float clip_threshold,
    const float weight_decay,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = (int)block_item_ids[linear_block];
    const int64_t start = block_item_starts[linear_block];
    const int64_t per_item_numel = item_numels[item];
    if (start >= per_item_numel)
        return;

    const int item_block = (int)(start / block_size);
    T* p = reinterpret_cast<T*>(p_ptrs[item]);
    const float* update_in = reinterpret_cast<const float*>(update_ptrs[item]);
    int8_t* exp_avg_q = reinterpret_cast<int8_t*>(exp_avg_q_ptrs[item]);
    float* exp_avg_absmax = reinterpret_cast<float*>(exp_avg_absmax_ptrs[item]);
    const float* sum_update = reinterpret_cast<const float*>(sum_update_ptrs[item]);
    const float old_absmax = exp_avg_absmax[item_block];
    const float rms = sqrtf(sum_update[0] / (float)per_item_numel);
    const float clip = fmaxf(1.0f, rms / clip_threshold);
    const int thread = (int)threadIdx.x;
    const int64_t item_idx = start + thread;
    const bool valid = thread < block_size && item_idx < per_item_numel;

    float new_avg = 0.0f;
    if (valid) {
        const float old_avg = dequant_i8(exp_avg_q[item_idx], old_absmax);
        const float u = update_in[item_idx] / clip;
        new_avg = (old_avg * beta1) + ((1.0f - beta1) * u);
    }

    const float local_max = valid ? fabsf(new_avg) : 0.0f;

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage max_tmp;
    const float block_max = BlockReduceT(max_tmp).Reduce(local_max, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_absmax[item_block] = shared_absmax;
    }
    __syncthreads();

    if (valid) {
        float p_val = to_float(p[item_idx]);
        if (weight_decay != 0.0f) {
            p_val -= weight_decay * lr * p_val;
        }
        p_val -= lr * new_avg;
        p[item_idx] = from_float_t<T>(p_val);
        exp_avg_q[item_idx] = quant_i8(new_avg, 1.0f / shared_absmax);
    }
}

__global__ void kCameFactoredExpAvgResPrepare(
    const float* __restrict__ g32,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_absmax,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    float* __restrict__ exp_avg_fp32,
    float* __restrict__ res32,
    const float* __restrict__ sum_update,
    const float beta1,
    const float eps1,
    const float clip_threshold,
    const int rows,
    const int cols,
    const int block_size
) {
    const int block = (int)blockIdx.x;
    const int64_t start = (int64_t)block * block_size;
    const int64_t numel = (int64_t)rows * cols;
    if (start >= numel)
        return;

    const float old_absmax = exp_avg_absmax[block];
    const float rms = sqrtf(sum_update[0] / (float)numel);
    const float clip = fmaxf(1.0f, rms / clip_threshold);

    float local_max = 0.0f;
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int64_t idx = start + offset;
        if (idx < numel) {
            const int row = (int)(idx / cols);
            const int col = (int)(idx - (int64_t)row * cols);
            const float u = (g32[idx] * r_factor[row] * c_factor[col]) / clip;
            const float old_avg = dequant_i8(exp_avg_q[idx], old_absmax);
            const float new_avg = (old_avg * beta1) + ((1.0f - beta1) * u);
            exp_avg_fp32[idx] = new_avg;
            const float diff = u - new_avg;
            res32[idx] = (diff * diff) + eps1;
            local_max = fmaxf(local_max, fabsf(new_avg));
        }
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_max = BlockReduceT(tmp).Reduce(local_max, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_absmax[block] = shared_absmax;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int64_t idx = start + offset;
        if (idx < numel) {
            exp_avg_q[idx] = quant_i8(exp_avg_fp32[idx], inv_absmax);
        }
    }
}

__global__ void kCameFactoredExpAvgResPrepareFp32(
    const float* __restrict__ g32,
    float* __restrict__ exp_avg,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    float* __restrict__ res32,
    const float* __restrict__ sum_update,
    const float beta1,
    const float eps1,
    const float clip_threshold,
    const int rows,
    const int cols,
    const int block_size
) {
    const int block = (int)blockIdx.x;
    const int64_t start = (int64_t)block * block_size;
    const int64_t numel = (int64_t)rows * cols;
    if (start >= numel)
        return;

    const float rms = sqrtf(sum_update[0] / (float)numel);
    const float clip = fmaxf(1.0f, rms / clip_threshold);

    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int64_t idx = start + offset;
        if (idx < numel) {
            const int row = (int)(idx / cols);
            const int col = (int)(idx - (int64_t)row * cols);
            const float u = (g32[idx] * r_factor[row] * c_factor[col]) / clip;
            const float new_avg = (exp_avg[idx] * beta1) + ((1.0f - beta1) * u);
            exp_avg[idx] = new_avg;
            const float diff = u - new_avg;
            res32[idx] = (diff * diff) + eps1;
        }
    }
}

template <typename ExpAvgT, typename ResT>
__global__ void kCameFactoredExpAvgResPrepareBatched(
    const float* __restrict__ g32,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_absmax,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    ExpAvgT* __restrict__ exp_avg_scratch,
    ResT* __restrict__ res32,
    const float* __restrict__ sum_update,
    const float beta1,
    const float eps1,
    const float clip_threshold,
    const int64_t per_item_numel,
    const int rows,
    const int cols,
    const int blocks_per_item,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / blocks_per_item;
    const int item_block = linear_block - item * blocks_per_item;
    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t start = item_offset + (int64_t)item_block * block_size;
    const int64_t item_end = item_offset + per_item_numel;
    if (start >= item_end)
        return;

    const float old_absmax = exp_avg_absmax[linear_block];
    const float rms = sqrtf(sum_update[0] / (float)per_item_numel);
    const float clip = fmaxf(1.0f, rms / clip_threshold);
    const int64_t row_offset = (int64_t)item * rows;
    const int64_t col_offset = (int64_t)item * cols;

    float local_max = 0.0f;
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int64_t idx = start + offset;
        if (idx < item_end) {
            const int64_t item_idx = idx - item_offset;
            const int row = (int)(item_idx / cols);
            const int col = (int)(item_idx - (int64_t)row * cols);
            const float u = (g32[idx] * r_factor[row_offset + row] * c_factor[col_offset + col]) / clip;
            const float old_avg = dequant_i8(exp_avg_q[idx], old_absmax);
            const float new_avg = (old_avg * beta1) + ((1.0f - beta1) * u);
            exp_avg_scratch[idx] = from_float_t<ExpAvgT>(new_avg);
            const float diff = u - new_avg;
            res32[idx] = from_float_t<ResT>((diff * diff) + eps1);
            local_max = fmaxf(local_max, fabsf(new_avg));
        }
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_max = BlockReduceT(tmp).Reduce(local_max, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_absmax[linear_block] = shared_absmax;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int64_t idx = start + offset;
        if (idx < item_end) {
            exp_avg_q[idx] = quant_i8(to_float(exp_avg_scratch[idx]), inv_absmax);
        }
    }
}

template <typename ExpAvgT, typename ResT>
__global__ void kCameFactoredExpAvgResPrepareMultiTensorSameShape(
    const float* __restrict__ g32,
    const int64_t* __restrict__ exp_avg_q_ptrs,
    const int64_t* __restrict__ exp_avg_absmax_ptrs,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    ExpAvgT* __restrict__ exp_avg_scratch,
    ResT* __restrict__ res32,
    const float* __restrict__ sum_update,
    const float beta1,
    const float eps1,
    const float clip_threshold,
    const int64_t per_item_numel,
    const int rows,
    const int cols,
    const int blocks_per_item,
    const int block_size
) {
    const int linear_block = (int)blockIdx.x;
    const int item = linear_block / blocks_per_item;
    const int item_block = linear_block - item * blocks_per_item;
    const int64_t item_offset = (int64_t)item * per_item_numel;
    const int64_t start = item_offset + (int64_t)item_block * block_size;
    const int64_t item_end = item_offset + per_item_numel;
    if (start >= item_end)
        return;

    auto* exp_avg_q = reinterpret_cast<int8_t*>(exp_avg_q_ptrs[item]);
    auto* exp_avg_absmax = reinterpret_cast<float*>(exp_avg_absmax_ptrs[item]);
    const float old_absmax = exp_avg_absmax[item_block];
    const float rms = sqrtf(sum_update[0] / (float)per_item_numel);
    const float clip = fmaxf(1.0f, rms / clip_threshold);
    const int64_t row_offset = (int64_t)item * rows;
    const int64_t col_offset = (int64_t)item * cols;

    float local_max = 0.0f;
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int64_t idx = start + offset;
        if (idx < item_end) {
            const int64_t item_idx = idx - item_offset;
            const int row = (int)(item_idx / cols);
            const int col = (int)(item_idx - (int64_t)row * cols);
            const float u = (g32[idx] * r_factor[row_offset + row] * c_factor[col_offset + col]) / clip;
            const float old_avg = dequant_i8(exp_avg_q[item_idx], old_absmax);
            const float new_avg = (old_avg * beta1) + ((1.0f - beta1) * u);
            exp_avg_scratch[idx] = from_float_t<ExpAvgT>(new_avg);
            const float diff = u - new_avg;
            res32[idx] = from_float_t<ResT>((diff * diff) + eps1);
            local_max = fmaxf(local_max, fabsf(new_avg));
        }
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_max = BlockReduceT(tmp).Reduce(local_max, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_max > 0.0f) ? block_max : 1.0f;
        exp_avg_absmax[item_block] = shared_absmax;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    for (int offset = (int)threadIdx.x; offset < block_size; offset += THREADS) {
        const int64_t idx = start + offset;
        if (idx < item_end) {
            const int64_t item_idx = idx - item_offset;
            exp_avg_q[item_idx] = quant_i8(to_float(exp_avg_scratch[idx]), inv_absmax);
        }
    }
}

template <typename T>
__global__ void kCameFactoredParamUpdate(
    T* __restrict__ p,
    const float* __restrict__ exp_avg_fp32,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    const float lr,
    const float weight_decay,
    const int rows,
    const int cols
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t numel = (int64_t)rows * cols;
    if (idx >= numel)
        return;

    const int row = (int)(idx / cols);
    const int col = (int)(idx - (int64_t)row * cols);
    float p_val = to_float(p[idx]);
    if (weight_decay != 0.0f) {
        p_val -= weight_decay * lr * p_val;
    }
    p_val -= lr * (exp_avg_fp32[idx] * r_factor[row] * c_factor[col]);
    p[idx] = from_float_t<T>(p_val);
}

template <typename T, typename ExpAvgT>
__global__ void kCameFactoredParamUpdateBatched(
    T* __restrict__ p,
    const ExpAvgT* __restrict__ exp_avg_scratch,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    const float lr,
    const float weight_decay,
    const int batch,
    const int64_t per_item_numel,
    const int rows,
    const int cols
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_numel = (int64_t)batch * per_item_numel;
    if (idx >= total_numel)
        return;

    const int item = (int)(idx / per_item_numel);
    const int64_t item_idx = idx - (int64_t)item * per_item_numel;
    const int row = (int)(item_idx / cols);
    const int col = (int)(item_idx - (int64_t)row * cols);
    float p_val = to_float(p[idx]);
    if (weight_decay != 0.0f) {
        p_val -= weight_decay * lr * p_val;
    }
    p_val -= lr * (
        to_float(exp_avg_scratch[idx])
        * r_factor[(int64_t)item * rows + row]
        * c_factor[(int64_t)item * cols + col]
    );
    p[idx] = from_float_t<T>(p_val);
}

template <typename T, typename ExpAvgT>
__global__ void kCameFactoredParamUpdateMultiTensorSameShape(
    const int64_t* __restrict__ p_ptrs,
    const ExpAvgT* __restrict__ exp_avg_scratch,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    const float lr,
    const float weight_decay,
    const int batch,
    const int64_t per_item_numel,
    const int rows,
    const int cols
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_numel = (int64_t)batch * per_item_numel;
    if (idx >= total_numel)
        return;

    const int item = (int)(idx / per_item_numel);
    const int64_t item_idx = idx - (int64_t)item * per_item_numel;
    const int row = (int)(item_idx / cols);
    const int col = (int)(item_idx - (int64_t)row * cols);
    auto* p = reinterpret_cast<T*>(p_ptrs[item]);
    float p_val = to_float(p[item_idx]);
    if (weight_decay != 0.0f) {
        p_val -= weight_decay * lr * p_val;
    }
    p_val -= lr * (
        to_float(exp_avg_scratch[idx])
        * r_factor[(int64_t)item * rows + row]
        * c_factor[(int64_t)item * cols + col]
    );
    p[item_idx] = from_float_t<T>(p_val);
}


template <typename T>
__global__ void kUpdateExpAvgQuantAndResSumsTiled(
    T* __restrict__ p,
    const T* __restrict__ g,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_absmax,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    const float* __restrict__ sum_update,
    float* __restrict__ res_row_partial,
    float* __restrict__ res_col_partial,
    const float beta1,
    const float eps1,
    const float clip_threshold,
    const float weight_decay,
    const float lr,
    const int rows,
    const int cols
) {
    const int tile_x = (int)blockIdx.x;
    const int tile_y = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;

    const int local_r = tid >> 4;
    const int local_c = tid & 15;

    const int row = tile_y * TILE + local_r;
    const int col = tile_x * TILE + local_c;
    const bool valid = (row < rows) && (col < cols);

    const int idx = row * cols + col;
    const int n = rows * cols;
    const int tile_rows = (rows + TILE - 1) / TILE;
    const int tile_cols = (cols + TILE - 1) / TILE;
    const int tile_id = tile_y * tile_cols + tile_x;

    const float rms = sqrtf((*sum_update) / (float)n);
    const float clip = fmaxf(1.0f, rms / clip_threshold);

    const float gv = valid ? to_float(g[idx]) : 0.0f;
    const float update = valid ? ((gv * r_factor[row] * c_factor[col]) / clip) : 0.0f;

    const float old_absmax = exp_avg_absmax[tile_id];
    const float m_old = (valid && old_absmax > 0.0f) ? dequant_i8(exp_avg_q[idx], old_absmax) : 0.0f;
    const float m_new = (m_old * beta1) + ((1.0f - beta1) * update);

    // Compute per-element res and store into shared tile buffer. We then do
    // row/col reductions without shared atomics (one thread per row/col).
    __shared__ float tile_res[THREADS];
    float res_val = 0.0f;
    if (valid) {
        const float diff = update - m_new;
        res_val = (diff * diff) + eps1;
    }
    tile_res[tid] = res_val;

    // blockwise absmax for quantization
    float absval = valid ? fabsf(m_new) : 0.0f;
    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_absmax = BlockReduceT(tmp).Reduce(absval, cub::Max());
    __shared__ float shared_absmax;
    if (tid == 0) {
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        exp_avg_absmax[tile_id] = shared_absmax;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    if (valid) {
        exp_avg_q[idx] = quant_i8(m_new, inv_absmax);
    }

    __syncthreads();
    if (tid < TILE) {
        float rsum = 0.0f;
        float csum = 0.0f;
#pragma unroll
        for (int j = 0; j < TILE; j++) {
            rsum += tile_res[tid * TILE + j];
            csum += tile_res[j * TILE + tid];
        }

        const int rr = tile_y * TILE + tid;
        if (rr < rows) {
            res_row_partial[rr * tile_cols + tile_x] = rsum;
        }
        const int cc = tile_x * TILE + tid;
        if (cc < cols) {
            res_col_partial[cc * tile_rows + tile_y] = csum;
        }
    }

    // Apply weight decay here on p to avoid a separate pass.
    // The final param update (with res_approx) is done in a later kernel.
    if (valid && weight_decay != 0.0f) {
        float pv = to_float(p[idx]);
        pv *= (1.0f - lr * weight_decay);
        p[idx] = from_float_t<T>(pv);
    }
}

__global__ void kUpdateResRowQuantized(
    const float* __restrict__ res_row_sum,
    const uint8_t* __restrict__ exp_avg_res_row_q,
    const float* __restrict__ exp_avg_res_row_absmax,
    float* __restrict__ updated_row,
    float* __restrict__ next_absmax,
    const float beta3,
    const int rows,
    const int cols,
    const int block_size
) {
    const int row = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (row >= rows)
        return;

    const float mean = res_row_sum[row] / (float)cols;
    const int qblock = row / block_size;
    const float old = dequant_u8(exp_avg_res_row_q[row], exp_avg_res_row_absmax[qblock]);
    const float updated = (old * beta3) + ((1.0f - beta3) * mean);
    updated_row[row] = updated;
    atomic_max_positive_f32(next_absmax + qblock, updated);
}

__global__ void kReduceRowTilePartials(
    const float* __restrict__ partial,
    float* __restrict__ out,
    const int rows,
    const int tile_cols
) {
    const int row = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (row >= rows)
        return;

    float total = 0.0f;
    for (int tile_x = 0; tile_x < tile_cols; ++tile_x) {
        total += partial[row * tile_cols + tile_x];
    }
    out[row] = total;
}

__global__ void kReduceColTilePartials(
    const float* __restrict__ partial,
    float* __restrict__ out,
    const int cols,
    const int tile_rows
) {
    const int col = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (col >= cols)
        return;

    float total = 0.0f;
    for (int tile_y = 0; tile_y < tile_rows; ++tile_y) {
        total += partial[col * tile_rows + tile_y];
    }
    out[col] = total;
}

__global__ void kUpdateResColQuantizedFinalize(
    const float* __restrict__ res_col_sum,
    uint8_t* __restrict__ exp_avg_res_col_q,
    float* __restrict__ exp_avg_res_col_absmax,
    float* __restrict__ c_res_factor,
    const float beta3,
    const int rows,
    const int cols,
    const int block_size
) {
    const int block = (int)blockIdx.x;
    const int start = block * block_size;
    const int col = start + (int)threadIdx.x;
    const bool valid = col < cols;

    float updated = 0.0f;
    if (valid) {
        const float mean = res_col_sum[col] / (float)rows;
        const float old = dequant_u8(exp_avg_res_col_q[col], exp_avg_res_col_absmax[block]);
        updated = (old * beta3) + ((1.0f - beta3) * mean);
    }

    using BlockReduceT = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduceT::TempStorage tmp;
    const float block_absmax = BlockReduceT(tmp).Reduce(updated, cub::Max());
    __shared__ float shared_absmax;
    if (threadIdx.x == 0) {
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        exp_avg_res_col_absmax[block] = shared_absmax;
    }
    __syncthreads();

    if (valid) {
        const float inv_absmax = 1.0f / shared_absmax;
        exp_avg_res_col_q[col] = quant_u8(updated, inv_absmax);
        c_res_factor[col] = rsqrtf(updated);
    }
}

__global__ void kUpdateExpAvgQuantAndResSumsTiledVec4Half(
    half* __restrict__ p,
    const half* __restrict__ g,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_absmax,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    const float* __restrict__ sum_update,
    float* __restrict__ res_row_partial,
    float* __restrict__ res_col_partial,
    const float beta1,
    const float eps1,
    const float clip_threshold,
    const float weight_decay,
    const float lr,
    const int rows,
    const int cols
) {
    const int tile_x = (int)blockIdx.x;
    const int tile_y = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    if (tid >= VEC4_THREADS)
        return;

    const int local_r = tid >> 2;  // 0..15
    const int local_g = tid & 3;   // 0..3 (group of 4 cols)
    const int row = tile_y * TILE + local_r;
    const int col0 = tile_x * TILE + local_g * 4;

    const int n = rows * cols;
    const int tile_rows = (rows + TILE - 1) / TILE;
    const int tile_cols = (cols + TILE - 1) / TILE;
    const int tile_id = tile_y * tile_cols + tile_x;

    const float rms = sqrtf((*sum_update) / (float)n);
    const float clip = fmaxf(1.0f, rms / clip_threshold);

    const bool row_valid = row < rows;
    const bool v0 = row_valid && (col0 + 0) < cols;
    const bool v1 = row_valid && (col0 + 1) < cols;
    const bool v2 = row_valid && (col0 + 2) < cols;
    const bool v3 = row_valid && (col0 + 3) < cols;

    const int base = row * cols + col0;

    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    if (v0 && v1) {
        const half2 gg01 = reinterpret_cast<const half2*>(g + base)[0];
        const float2 gf01 = __half22float2(gg01);
        g0 = gf01.x;
        g1 = gf01.y;
    } else {
        if (v0)
            g0 = __half2float(g[base + 0]);
        if (v1)
            g1 = __half2float(g[base + 1]);
    }
    if (v2 && v3) {
        const half2 gg23 = reinterpret_cast<const half2*>(g + base + 2)[0];
        const float2 gf23 = __half22float2(gg23);
        g2 = gf23.x;
        g3 = gf23.y;
    } else {
        if (v2)
            g2 = __half2float(g[base + 2]);
        if (v3)
            g3 = __half2float(g[base + 3]);
    }

    int8_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    if (v0 && v1 && v2 && v3) {
        const int qpack = reinterpret_cast<const int*>(exp_avg_q + base)[0];
        q0 = (int8_t)(qpack & 0xFF);
        q1 = (int8_t)((qpack >> 8) & 0xFF);
        q2 = (int8_t)((qpack >> 16) & 0xFF);
        q3 = (int8_t)((qpack >> 24) & 0xFF);
    } else {
        if (v0)
            q0 = exp_avg_q[base + 0];
        if (v1)
            q1 = exp_avg_q[base + 1];
        if (v2)
            q2 = exp_avg_q[base + 2];
        if (v3)
            q3 = exp_avg_q[base + 3];
    }

    const float old_absmax = exp_avg_absmax[tile_id];
    const float m0_old = (v0 && old_absmax > 0.0f) ? dequant_i8(q0, old_absmax) : 0.0f;
    const float m1_old = (v1 && old_absmax > 0.0f) ? dequant_i8(q1, old_absmax) : 0.0f;
    const float m2_old = (v2 && old_absmax > 0.0f) ? dequant_i8(q2, old_absmax) : 0.0f;
    const float m3_old = (v3 && old_absmax > 0.0f) ? dequant_i8(q3, old_absmax) : 0.0f;

    const float rf = row_valid ? r_factor[row] : 0.0f;
    const float cf0 = v0 ? c_factor[col0 + 0] : 0.0f;
    const float cf1 = v1 ? c_factor[col0 + 1] : 0.0f;
    const float cf2 = v2 ? c_factor[col0 + 2] : 0.0f;
    const float cf3 = v3 ? c_factor[col0 + 3] : 0.0f;

    const float u0 = v0 ? (g0 * rf * cf0) / clip : 0.0f;
    const float u1 = v1 ? (g1 * rf * cf1) / clip : 0.0f;
    const float u2 = v2 ? (g2 * rf * cf2) / clip : 0.0f;
    const float u3 = v3 ? (g3 * rf * cf3) / clip : 0.0f;

    const float m0_new = (m0_old * beta1) + ((1.0f - beta1) * u0);
    const float m1_new = (m1_old * beta1) + ((1.0f - beta1) * u1);
    const float m2_new = (m2_old * beta1) + ((1.0f - beta1) * u2);
    const float m3_new = (m3_old * beta1) + ((1.0f - beta1) * u3);

    __shared__ float tile_res[TILE * TILE];
    const int base_pos = local_r * TILE + local_g * 4;
    tile_res[base_pos + 0] = v0 ? ((u0 - m0_new) * (u0 - m0_new) + eps1) : 0.0f;
    tile_res[base_pos + 1] = v1 ? ((u1 - m1_new) * (u1 - m1_new) + eps1) : 0.0f;
    tile_res[base_pos + 2] = v2 ? ((u2 - m2_new) * (u2 - m2_new) + eps1) : 0.0f;
    tile_res[base_pos + 3] = v3 ? ((u3 - m3_new) * (u3 - m3_new) + eps1) : 0.0f;

    const float absval = fmaxf(fmaxf(fabsf(m0_new), fabsf(m1_new)), fmaxf(fabsf(m2_new), fabsf(m3_new)));
    using BlockReduceT = cub::BlockReduce<float, VEC4_THREADS>;
    __shared__ typename BlockReduceT::TempStorage reduce_tmp;
    const float block_absmax = BlockReduceT(reduce_tmp).Reduce(absval, cub::Max());

    __shared__ float shared_absmax;
    if (tid == 0) {
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        exp_avg_absmax[tile_id] = shared_absmax;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    const int8_t nq0 = v0 ? quant_i8(m0_new, inv_absmax) : 0;
    const int8_t nq1 = v1 ? quant_i8(m1_new, inv_absmax) : 0;
    const int8_t nq2 = v2 ? quant_i8(m2_new, inv_absmax) : 0;
    const int8_t nq3 = v3 ? quant_i8(m3_new, inv_absmax) : 0;

    if (v0 && v1 && v2 && v3) {
        const unsigned int packed = ((unsigned int)(uint8_t)nq0) | (((unsigned int)(uint8_t)nq1) << 8) |
                                    (((unsigned int)(uint8_t)nq2) << 16) | (((unsigned int)(uint8_t)nq3) << 24);
        reinterpret_cast<unsigned int*>(exp_avg_q + base)[0] = packed;
    } else {
        if (v0)
            exp_avg_q[base + 0] = nq0;
        if (v1)
            exp_avg_q[base + 1] = nq1;
        if (v2)
            exp_avg_q[base + 2] = nq2;
        if (v3)
            exp_avg_q[base + 3] = nq3;
    }

    // Reduce row/col sums (threads 0..15 do it)
    __syncthreads();
    if (tid < TILE) {
        float rsum = 0.0f;
        float csum = 0.0f;
#pragma unroll
        for (int j = 0; j < TILE; j++) {
            rsum += tile_res[tid * TILE + j];
            csum += tile_res[j * TILE + tid];
        }
        const int rr = tile_y * TILE + tid;
        if (rr < rows)
            res_row_partial[rr * tile_cols + tile_x] = rsum;
        const int cc = tile_x * TILE + tid;
        if (cc < cols)
            res_col_partial[cc * tile_rows + tile_y] = csum;
    }

    // Apply weight decay to p (vectorized where possible)
    if (weight_decay != 0.0f && row_valid) {
        const float decay = 1.0f - lr * weight_decay;
        if (v0 && v1) {
            const half2 pp01 = reinterpret_cast<const half2*>(p + base)[0];
            const float2 pf01 = __half22float2(pp01);
            const half2 out01 = __floats2half2_rn(pf01.x * decay, pf01.y * decay);
            reinterpret_cast<half2*>(p + base)[0] = out01;
        } else {
            if (v0)
                p[base + 0] = __float2half_rn(__half2float(p[base + 0]) * decay);
            if (v1)
                p[base + 1] = __float2half_rn(__half2float(p[base + 1]) * decay);
        }
        if (v2 && v3) {
            const half2 pp23 = reinterpret_cast<const half2*>(p + base + 2)[0];
            const float2 pf23 = __half22float2(pp23);
            const half2 out23 = __floats2half2_rn(pf23.x * decay, pf23.y * decay);
            reinterpret_cast<half2*>(p + base + 2)[0] = out23;
        } else {
            if (v2)
                p[base + 2] = __float2half_rn(__half2float(p[base + 2]) * decay);
            if (v3)
                p[base + 3] = __float2half_rn(__half2float(p[base + 3]) * decay);
        }
    }
}

__global__ void kUpdateExpAvgQuantAndResSumsTiledVec4Bf16(
    __nv_bfloat16* __restrict__ p,
    const __nv_bfloat16* __restrict__ g,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_absmax,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    const float* __restrict__ sum_update,
    float* __restrict__ res_row_partial,
    float* __restrict__ res_col_partial,
    const float beta1,
    const float eps1,
    const float clip_threshold,
    const float weight_decay,
    const float lr,
    const int rows,
    const int cols
) {
    const int tile_x = (int)blockIdx.x;
    const int tile_y = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    if (tid >= VEC4_THREADS)
        return;

    const int local_r = tid >> 2;
    const int local_g = tid & 3;
    const int row = tile_y * TILE + local_r;
    const int col0 = tile_x * TILE + local_g * 4;

    const int n = rows * cols;
    const int tile_rows = (rows + TILE - 1) / TILE;
    const int tile_cols = (cols + TILE - 1) / TILE;
    const int tile_id = tile_y * tile_cols + tile_x;

    const float rms = sqrtf((*sum_update) / (float)n);
    const float clip = fmaxf(1.0f, rms / clip_threshold);

    const bool row_valid = row < rows;
    const bool v0 = row_valid && (col0 + 0) < cols;
    const bool v1 = row_valid && (col0 + 1) < cols;
    const bool v2 = row_valid && (col0 + 2) < cols;
    const bool v3 = row_valid && (col0 + 3) < cols;

    const int base = row * cols + col0;

    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    if (v0 && v1) {
        const __nv_bfloat162 gg01 = reinterpret_cast<const __nv_bfloat162*>(g + base)[0];
        const float2 gf01 = __bfloat1622float2(gg01);
        g0 = gf01.x;
        g1 = gf01.y;
    } else {
        if (v0)
            g0 = __bfloat162float(g[base + 0]);
        if (v1)
            g1 = __bfloat162float(g[base + 1]);
    }
    if (v2 && v3) {
        const __nv_bfloat162 gg23 = reinterpret_cast<const __nv_bfloat162*>(g + base + 2)[0];
        const float2 gf23 = __bfloat1622float2(gg23);
        g2 = gf23.x;
        g3 = gf23.y;
    } else {
        if (v2)
            g2 = __bfloat162float(g[base + 2]);
        if (v3)
            g3 = __bfloat162float(g[base + 3]);
    }

    int8_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    if (v0 && v1 && v2 && v3) {
        const int qpack = reinterpret_cast<const int*>(exp_avg_q + base)[0];
        q0 = (int8_t)(qpack & 0xFF);
        q1 = (int8_t)((qpack >> 8) & 0xFF);
        q2 = (int8_t)((qpack >> 16) & 0xFF);
        q3 = (int8_t)((qpack >> 24) & 0xFF);
    } else {
        if (v0)
            q0 = exp_avg_q[base + 0];
        if (v1)
            q1 = exp_avg_q[base + 1];
        if (v2)
            q2 = exp_avg_q[base + 2];
        if (v3)
            q3 = exp_avg_q[base + 3];
    }

    const float old_absmax = exp_avg_absmax[tile_id];
    const float m0_old = (v0 && old_absmax > 0.0f) ? dequant_i8(q0, old_absmax) : 0.0f;
    const float m1_old = (v1 && old_absmax > 0.0f) ? dequant_i8(q1, old_absmax) : 0.0f;
    const float m2_old = (v2 && old_absmax > 0.0f) ? dequant_i8(q2, old_absmax) : 0.0f;
    const float m3_old = (v3 && old_absmax > 0.0f) ? dequant_i8(q3, old_absmax) : 0.0f;

    const float rf = row_valid ? r_factor[row] : 0.0f;
    const float cf0 = v0 ? c_factor[col0 + 0] : 0.0f;
    const float cf1 = v1 ? c_factor[col0 + 1] : 0.0f;
    const float cf2 = v2 ? c_factor[col0 + 2] : 0.0f;
    const float cf3 = v3 ? c_factor[col0 + 3] : 0.0f;

    const float u0 = v0 ? (g0 * rf * cf0) / clip : 0.0f;
    const float u1 = v1 ? (g1 * rf * cf1) / clip : 0.0f;
    const float u2 = v2 ? (g2 * rf * cf2) / clip : 0.0f;
    const float u3 = v3 ? (g3 * rf * cf3) / clip : 0.0f;

    const float m0_new = (m0_old * beta1) + ((1.0f - beta1) * u0);
    const float m1_new = (m1_old * beta1) + ((1.0f - beta1) * u1);
    const float m2_new = (m2_old * beta1) + ((1.0f - beta1) * u2);
    const float m3_new = (m3_old * beta1) + ((1.0f - beta1) * u3);

    __shared__ float tile_res[TILE * TILE];
    const int base_pos = local_r * TILE + local_g * 4;
    tile_res[base_pos + 0] = v0 ? ((u0 - m0_new) * (u0 - m0_new) + eps1) : 0.0f;
    tile_res[base_pos + 1] = v1 ? ((u1 - m1_new) * (u1 - m1_new) + eps1) : 0.0f;
    tile_res[base_pos + 2] = v2 ? ((u2 - m2_new) * (u2 - m2_new) + eps1) : 0.0f;
    tile_res[base_pos + 3] = v3 ? ((u3 - m3_new) * (u3 - m3_new) + eps1) : 0.0f;

    const float absval = fmaxf(fmaxf(fabsf(m0_new), fabsf(m1_new)), fmaxf(fabsf(m2_new), fabsf(m3_new)));
    using BlockReduceT = cub::BlockReduce<float, VEC4_THREADS>;
    __shared__ typename BlockReduceT::TempStorage reduce_tmp;
    const float block_absmax = BlockReduceT(reduce_tmp).Reduce(absval, cub::Max());

    __shared__ float shared_absmax;
    if (tid == 0) {
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        exp_avg_absmax[tile_id] = shared_absmax;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    const int8_t nq0 = v0 ? quant_i8(m0_new, inv_absmax) : 0;
    const int8_t nq1 = v1 ? quant_i8(m1_new, inv_absmax) : 0;
    const int8_t nq2 = v2 ? quant_i8(m2_new, inv_absmax) : 0;
    const int8_t nq3 = v3 ? quant_i8(m3_new, inv_absmax) : 0;

    if (v0 && v1 && v2 && v3) {
        const unsigned int packed = ((unsigned int)(uint8_t)nq0) | (((unsigned int)(uint8_t)nq1) << 8) |
                                    (((unsigned int)(uint8_t)nq2) << 16) | (((unsigned int)(uint8_t)nq3) << 24);
        reinterpret_cast<unsigned int*>(exp_avg_q + base)[0] = packed;
    } else {
        if (v0)
            exp_avg_q[base + 0] = nq0;
        if (v1)
            exp_avg_q[base + 1] = nq1;
        if (v2)
            exp_avg_q[base + 2] = nq2;
        if (v3)
            exp_avg_q[base + 3] = nq3;
    }

    __syncthreads();
    if (tid < TILE) {
        float rsum = 0.0f;
        float csum = 0.0f;
#pragma unroll
        for (int j = 0; j < TILE; j++) {
            rsum += tile_res[tid * TILE + j];
            csum += tile_res[j * TILE + tid];
        }
        const int rr = tile_y * TILE + tid;
        if (rr < rows)
            res_row_partial[rr * tile_cols + tile_x] = rsum;
        const int cc = tile_x * TILE + tid;
        if (cc < cols)
            res_col_partial[cc * tile_rows + tile_y] = csum;
    }

    if (weight_decay != 0.0f && row_valid) {
        const float decay = 1.0f - lr * weight_decay;
        if (v0 && v1) {
            const __nv_bfloat162 pp01 = reinterpret_cast<const __nv_bfloat162*>(p + base)[0];
            const float2 pf01 = __bfloat1622float2(pp01);
            const __nv_bfloat162 out01 = __floats2bfloat162_rn(pf01.x * decay, pf01.y * decay);
            reinterpret_cast<__nv_bfloat162*>(p + base)[0] = out01;
        } else {
            if (v0)
                p[base + 0] = __float2bfloat16(__bfloat162float(p[base + 0]) * decay);
            if (v1)
                p[base + 1] = __float2bfloat16(__bfloat162float(p[base + 1]) * decay);
        }
        if (v2 && v3) {
            const __nv_bfloat162 pp23 = reinterpret_cast<const __nv_bfloat162*>(p + base + 2)[0];
            const float2 pf23 = __bfloat1622float2(pp23);
            const __nv_bfloat162 out23 = __floats2bfloat162_rn(pf23.x * decay, pf23.y * decay);
            reinterpret_cast<__nv_bfloat162*>(p + base + 2)[0] = out23;
        } else {
            if (v2)
                p[base + 2] = __float2bfloat16(__bfloat162float(p[base + 2]) * decay);
            if (v3)
                p[base + 3] = __float2bfloat16(__bfloat162float(p[base + 3]) * decay);
        }
    }
}

__global__ void kUpdateExpAvgQuantAndResSumsTiledVec4Float(
    float* __restrict__ p,
    const float* __restrict__ g,
    int8_t* __restrict__ exp_avg_q,
    float* __restrict__ exp_avg_absmax,
    const float* __restrict__ r_factor,
    const float* __restrict__ c_factor,
    const float* __restrict__ sum_update,
    float* __restrict__ res_row_partial,
    float* __restrict__ res_col_partial,
    const float beta1,
    const float eps1,
    const float clip_threshold,
    const float weight_decay,
    const float lr,
    const int rows,
    const int cols
) {
    const int tile_x = (int)blockIdx.x;
    const int tile_y = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    if (tid >= VEC4_THREADS)
        return;

    const int local_r = tid >> 2;
    const int local_g = tid & 3;
    const int row = tile_y * TILE + local_r;
    const int col0 = tile_x * TILE + local_g * 4;

    const int n = rows * cols;
    const int tile_rows = (rows + TILE - 1) / TILE;
    const int tile_cols = (cols + TILE - 1) / TILE;
    const int tile_id = tile_y * tile_cols + tile_x;

    const float rms = sqrtf((*sum_update) / (float)n);
    const float clip = fmaxf(1.0f, rms / clip_threshold);

    const bool row_valid = row < rows;
    const bool v0 = row_valid && (col0 + 0) < cols;
    const bool v1 = row_valid && (col0 + 1) < cols;
    const bool v2 = row_valid && (col0 + 2) < cols;
    const bool v3 = row_valid && (col0 + 3) < cols;

    const int base = row * cols + col0;
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    if (v0 && v1 && v2 && v3) {
        const float4 gg = reinterpret_cast<const float4*>(g + base)[0];
        g0 = gg.x;
        g1 = gg.y;
        g2 = gg.z;
        g3 = gg.w;
    } else {
        if (v0) g0 = g[base + 0];
        if (v1) g1 = g[base + 1];
        if (v2) g2 = g[base + 2];
        if (v3) g3 = g[base + 3];
    }

    int8_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    if (v0 && v1 && v2 && v3) {
        const int qpack = reinterpret_cast<const int*>(exp_avg_q + base)[0];
        q0 = (int8_t)(qpack & 0xFF);
        q1 = (int8_t)((qpack >> 8) & 0xFF);
        q2 = (int8_t)((qpack >> 16) & 0xFF);
        q3 = (int8_t)((qpack >> 24) & 0xFF);
    } else {
        if (v0) q0 = exp_avg_q[base + 0];
        if (v1) q1 = exp_avg_q[base + 1];
        if (v2) q2 = exp_avg_q[base + 2];
        if (v3) q3 = exp_avg_q[base + 3];
    }

    const float old_absmax = exp_avg_absmax[tile_id];
    const float m0_old = (v0 && old_absmax > 0.0f) ? dequant_i8(q0, old_absmax) : 0.0f;
    const float m1_old = (v1 && old_absmax > 0.0f) ? dequant_i8(q1, old_absmax) : 0.0f;
    const float m2_old = (v2 && old_absmax > 0.0f) ? dequant_i8(q2, old_absmax) : 0.0f;
    const float m3_old = (v3 && old_absmax > 0.0f) ? dequant_i8(q3, old_absmax) : 0.0f;

    const float rf = row_valid ? r_factor[row] : 0.0f;
    const float cf0 = v0 ? c_factor[col0 + 0] : 0.0f;
    const float cf1 = v1 ? c_factor[col0 + 1] : 0.0f;
    const float cf2 = v2 ? c_factor[col0 + 2] : 0.0f;
    const float cf3 = v3 ? c_factor[col0 + 3] : 0.0f;

    const float u0 = v0 ? (g0 * rf * cf0) / clip : 0.0f;
    const float u1 = v1 ? (g1 * rf * cf1) / clip : 0.0f;
    const float u2 = v2 ? (g2 * rf * cf2) / clip : 0.0f;
    const float u3 = v3 ? (g3 * rf * cf3) / clip : 0.0f;

    const float m0_new = (m0_old * beta1) + ((1.0f - beta1) * u0);
    const float m1_new = (m1_old * beta1) + ((1.0f - beta1) * u1);
    const float m2_new = (m2_old * beta1) + ((1.0f - beta1) * u2);
    const float m3_new = (m3_old * beta1) + ((1.0f - beta1) * u3);

    __shared__ float tile_res[TILE * TILE];
    const int base_pos = local_r * TILE + local_g * 4;
    tile_res[base_pos + 0] = v0 ? ((u0 - m0_new) * (u0 - m0_new) + eps1) : 0.0f;
    tile_res[base_pos + 1] = v1 ? ((u1 - m1_new) * (u1 - m1_new) + eps1) : 0.0f;
    tile_res[base_pos + 2] = v2 ? ((u2 - m2_new) * (u2 - m2_new) + eps1) : 0.0f;
    tile_res[base_pos + 3] = v3 ? ((u3 - m3_new) * (u3 - m3_new) + eps1) : 0.0f;

    const float absval = fmaxf(fmaxf(fabsf(m0_new), fabsf(m1_new)), fmaxf(fabsf(m2_new), fabsf(m3_new)));
    using BlockReduceT = cub::BlockReduce<float, VEC4_THREADS>;
    __shared__ typename BlockReduceT::TempStorage reduce_tmp;
    const float block_absmax = BlockReduceT(reduce_tmp).Reduce(absval, cub::Max());

    __shared__ float shared_absmax;
    if (tid == 0) {
        shared_absmax = (block_absmax > 0.0f) ? block_absmax : 1.0f;
        exp_avg_absmax[tile_id] = shared_absmax;
    }
    __syncthreads();

    const float inv_absmax = 1.0f / shared_absmax;
    const int8_t nq0 = v0 ? quant_i8(m0_new, inv_absmax) : 0;
    const int8_t nq1 = v1 ? quant_i8(m1_new, inv_absmax) : 0;
    const int8_t nq2 = v2 ? quant_i8(m2_new, inv_absmax) : 0;
    const int8_t nq3 = v3 ? quant_i8(m3_new, inv_absmax) : 0;

    if (v0 && v1 && v2 && v3) {
        const unsigned int packed = ((unsigned int)(uint8_t)nq0) | (((unsigned int)(uint8_t)nq1) << 8) |
                                    (((unsigned int)(uint8_t)nq2) << 16) | (((unsigned int)(uint8_t)nq3) << 24);
        reinterpret_cast<unsigned int*>(exp_avg_q + base)[0] = packed;
    } else {
        if (v0) exp_avg_q[base + 0] = nq0;
        if (v1) exp_avg_q[base + 1] = nq1;
        if (v2) exp_avg_q[base + 2] = nq2;
        if (v3) exp_avg_q[base + 3] = nq3;
    }

    __syncthreads();
    if (tid < TILE) {
        float rsum = 0.0f;
        float csum = 0.0f;
#pragma unroll
        for (int j = 0; j < TILE; j++) {
            rsum += tile_res[tid * TILE + j];
            csum += tile_res[j * TILE + tid];
        }
        const int rr = tile_y * TILE + tid;
        if (rr < rows) res_row_partial[rr * tile_cols + tile_x] = rsum;
        const int cc = tile_x * TILE + tid;
        if (cc < cols) res_col_partial[cc * tile_rows + tile_y] = csum;
    }

    if (weight_decay != 0.0f && row_valid) {
        const float decay = 1.0f - lr * weight_decay;
        if (v0 && v1 && v2 && v3) {
            float4 pp = reinterpret_cast<float4*>(p + base)[0];
            pp.x *= decay;
            pp.y *= decay;
            pp.z *= decay;
            pp.w *= decay;
            reinterpret_cast<float4*>(p + base)[0] = pp;
        } else {
            if (v0) p[base + 0] *= decay;
            if (v1) p[base + 1] *= decay;
            if (v2) p[base + 2] *= decay;
            if (v3) p[base + 3] *= decay;
        }
    }
}

template <typename T>
__global__ void kParamUpdateTiled(
    T* __restrict__ p,
    const int8_t* __restrict__ exp_avg_q,
    const float* __restrict__ exp_avg_absmax,
    const float* __restrict__ r_res_factor,
    const float* __restrict__ c_res_factor,
    const float lr,
    const int rows,
    const int cols
) {
    const int tile_x = (int)blockIdx.x;
    const int tile_y = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;

    const int local_r = tid >> 4;
    const int local_c = tid & 15;

    const int row = tile_y * TILE + local_r;
    const int col = tile_x * TILE + local_c;
    if (row >= rows || col >= cols)
        return;

    const int idx = row * cols + col;
    const int tile_cols = (cols + TILE - 1) / TILE;
    const int tile_id = tile_y * tile_cols + tile_x;

    const float absmax = exp_avg_absmax[tile_id];
    const float m = (absmax > 0.0f) ? dequant_i8(exp_avg_q[idx], absmax) : 0.0f;
    const float update2 = m * r_res_factor[row] * c_res_factor[col];

    float pv = to_float(p[idx]);
    pv -= lr * update2;
    p[idx] = from_float_t<T>(pv);
}

__global__ void kParamUpdateTiledVec4Half(
    half* __restrict__ p,
    const int8_t* __restrict__ exp_avg_q,
    const float* __restrict__ exp_avg_absmax,
    const float* __restrict__ r_res_factor,
    const float* __restrict__ c_res_factor,
    const float lr,
    const int rows,
    const int cols
) {
    const int tile_x = (int)blockIdx.x;
    const int tile_y = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    if (tid >= VEC4_THREADS)
        return;

    const int local_r = tid >> 2;
    const int local_g = tid & 3;
    const int row = tile_y * TILE + local_r;
    const int col0 = tile_x * TILE + local_g * 4;
    if (row >= rows)
        return;

    const int base = row * cols + col0;
    const int tile_cols = (cols + TILE - 1) / TILE;
    const int tile_id = tile_y * tile_cols + tile_x;

    const bool v0 = (col0 + 0) < cols;
    const bool v1 = (col0 + 1) < cols;
    const bool v2 = (col0 + 2) < cols;
    const bool v3 = (col0 + 3) < cols;

    int8_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    if (v0 && v1 && v2 && v3) {
        const int qpack = reinterpret_cast<const int*>(exp_avg_q + base)[0];
        q0 = (int8_t)(qpack & 0xFF);
        q1 = (int8_t)((qpack >> 8) & 0xFF);
        q2 = (int8_t)((qpack >> 16) & 0xFF);
        q3 = (int8_t)((qpack >> 24) & 0xFF);
    } else {
        if (v0)
            q0 = exp_avg_q[base + 0];
        if (v1)
            q1 = exp_avg_q[base + 1];
        if (v2)
            q2 = exp_avg_q[base + 2];
        if (v3)
            q3 = exp_avg_q[base + 3];
    }

    const float absmax = exp_avg_absmax[tile_id];
    const float rf = r_res_factor[row];
    const float u0 = v0 ? (dequant_i8(q0, absmax) * rf * c_res_factor[col0 + 0]) : 0.0f;
    const float u1 = v1 ? (dequant_i8(q1, absmax) * rf * c_res_factor[col0 + 1]) : 0.0f;
    const float u2 = v2 ? (dequant_i8(q2, absmax) * rf * c_res_factor[col0 + 2]) : 0.0f;
    const float u3 = v3 ? (dequant_i8(q3, absmax) * rf * c_res_factor[col0 + 3]) : 0.0f;

    if (v0 && v1) {
        const half2 pp01 = reinterpret_cast<const half2*>(p + base)[0];
        const float2 pf01 = __half22float2(pp01);
        const half2 out01 = __floats2half2_rn(pf01.x - lr * u0, pf01.y - lr * u1);
        reinterpret_cast<half2*>(p + base)[0] = out01;
    } else {
        if (v0)
            p[base + 0] = __float2half_rn(__half2float(p[base + 0]) - lr * u0);
        if (v1)
            p[base + 1] = __float2half_rn(__half2float(p[base + 1]) - lr * u1);
    }
    if (v2 && v3) {
        const half2 pp23 = reinterpret_cast<const half2*>(p + base + 2)[0];
        const float2 pf23 = __half22float2(pp23);
        const half2 out23 = __floats2half2_rn(pf23.x - lr * u2, pf23.y - lr * u3);
        reinterpret_cast<half2*>(p + base + 2)[0] = out23;
    } else {
        if (v2)
            p[base + 2] = __float2half_rn(__half2float(p[base + 2]) - lr * u2);
        if (v3)
            p[base + 3] = __float2half_rn(__half2float(p[base + 3]) - lr * u3);
    }
}

__global__ void kParamUpdateTiledVec4Float(
    float* __restrict__ p,
    const int8_t* __restrict__ exp_avg_q,
    const float* __restrict__ exp_avg_absmax,
    const float* __restrict__ r_res_factor,
    const float* __restrict__ c_res_factor,
    const float lr,
    const int rows,
    const int cols
) {
    const int tile_x = (int)blockIdx.x;
    const int tile_y = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    if (tid >= VEC4_THREADS)
        return;

    const int local_r = tid >> 2;
    const int local_g = tid & 3;
    const int row = tile_y * TILE + local_r;
    const int col0 = tile_x * TILE + local_g * 4;
    if (row >= rows)
        return;

    const int base = row * cols + col0;
    const int tile_cols = (cols + TILE - 1) / TILE;
    const int tile_id = tile_y * tile_cols + tile_x;

    const bool v0 = (col0 + 0) < cols;
    const bool v1 = (col0 + 1) < cols;
    const bool v2 = (col0 + 2) < cols;
    const bool v3 = (col0 + 3) < cols;

    int8_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    if (v0 && v1 && v2 && v3) {
        const int qpack = reinterpret_cast<const int*>(exp_avg_q + base)[0];
        q0 = (int8_t)(qpack & 0xFF);
        q1 = (int8_t)((qpack >> 8) & 0xFF);
        q2 = (int8_t)((qpack >> 16) & 0xFF);
        q3 = (int8_t)((qpack >> 24) & 0xFF);
    } else {
        if (v0) q0 = exp_avg_q[base + 0];
        if (v1) q1 = exp_avg_q[base + 1];
        if (v2) q2 = exp_avg_q[base + 2];
        if (v3) q3 = exp_avg_q[base + 3];
    }

    const float absmax = exp_avg_absmax[tile_id];
    const float rf = r_res_factor[row];
    const float u0 = v0 ? (dequant_i8(q0, absmax) * rf * c_res_factor[col0 + 0]) : 0.0f;
    const float u1 = v1 ? (dequant_i8(q1, absmax) * rf * c_res_factor[col0 + 1]) : 0.0f;
    const float u2 = v2 ? (dequant_i8(q2, absmax) * rf * c_res_factor[col0 + 2]) : 0.0f;
    const float u3 = v3 ? (dequant_i8(q3, absmax) * rf * c_res_factor[col0 + 3]) : 0.0f;

    if (v0 && v1 && v2 && v3) {
        float4 pp = reinterpret_cast<float4*>(p + base)[0];
        pp.x -= lr * u0;
        pp.y -= lr * u1;
        pp.z -= lr * u2;
        pp.w -= lr * u3;
        reinterpret_cast<float4*>(p + base)[0] = pp;
    } else {
        if (v0) p[base + 0] -= lr * u0;
        if (v1) p[base + 1] -= lr * u1;
        if (v2) p[base + 2] -= lr * u2;
        if (v3) p[base + 3] -= lr * u3;
    }
}

__global__ void kParamUpdateTiledVec4Bf16(
    __nv_bfloat16* __restrict__ p,
    const int8_t* __restrict__ exp_avg_q,
    const float* __restrict__ exp_avg_absmax,
    const float* __restrict__ r_res_factor,
    const float* __restrict__ c_res_factor,
    const float lr,
    const int rows,
    const int cols
) {
    const int tile_x = (int)blockIdx.x;
    const int tile_y = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    if (tid >= VEC4_THREADS)
        return;

    const int local_r = tid >> 2;
    const int local_g = tid & 3;
    const int row = tile_y * TILE + local_r;
    const int col0 = tile_x * TILE + local_g * 4;
    if (row >= rows)
        return;

    const int base = row * cols + col0;
    const int tile_cols = (cols + TILE - 1) / TILE;
    const int tile_id = tile_y * tile_cols + tile_x;

    const bool v0 = (col0 + 0) < cols;
    const bool v1 = (col0 + 1) < cols;
    const bool v2 = (col0 + 2) < cols;
    const bool v3 = (col0 + 3) < cols;

    int8_t q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    if (v0 && v1 && v2 && v3) {
        const int qpack = reinterpret_cast<const int*>(exp_avg_q + base)[0];
        q0 = (int8_t)(qpack & 0xFF);
        q1 = (int8_t)((qpack >> 8) & 0xFF);
        q2 = (int8_t)((qpack >> 16) & 0xFF);
        q3 = (int8_t)((qpack >> 24) & 0xFF);
    } else {
        if (v0)
            q0 = exp_avg_q[base + 0];
        if (v1)
            q1 = exp_avg_q[base + 1];
        if (v2)
            q2 = exp_avg_q[base + 2];
        if (v3)
            q3 = exp_avg_q[base + 3];
    }

    const float absmax = exp_avg_absmax[tile_id];
    const float rf = r_res_factor[row];
    const float u0 = v0 ? (dequant_i8(q0, absmax) * rf * c_res_factor[col0 + 0]) : 0.0f;
    const float u1 = v1 ? (dequant_i8(q1, absmax) * rf * c_res_factor[col0 + 1]) : 0.0f;
    const float u2 = v2 ? (dequant_i8(q2, absmax) * rf * c_res_factor[col0 + 2]) : 0.0f;
    const float u3 = v3 ? (dequant_i8(q3, absmax) * rf * c_res_factor[col0 + 3]) : 0.0f;

    if (v0 && v1) {
        const __nv_bfloat162 pp01 = reinterpret_cast<const __nv_bfloat162*>(p + base)[0];
        const float2 pf01 = __bfloat1622float2(pp01);
        const __nv_bfloat162 out01 = __floats2bfloat162_rn(pf01.x - lr * u0, pf01.y - lr * u1);
        reinterpret_cast<__nv_bfloat162*>(p + base)[0] = out01;
    } else {
        if (v0)
            p[base + 0] = __float2bfloat16(__bfloat162float(p[base + 0]) - lr * u0);
        if (v1)
            p[base + 1] = __float2bfloat16(__bfloat162float(p[base + 1]) - lr * u1);
    }
    if (v2 && v3) {
        const __nv_bfloat162 pp23 = reinterpret_cast<const __nv_bfloat162*>(p + base + 2)[0];
        const float2 pf23 = __bfloat1622float2(pp23);
        const __nv_bfloat162 out23 = __floats2bfloat162_rn(pf23.x - lr * u2, pf23.y - lr * u3);
        reinterpret_cast<__nv_bfloat162*>(p + base + 2)[0] = out23;
    } else {
        if (v2)
            p[base + 2] = __float2bfloat16(__bfloat162float(p[base + 2]) - lr * u2);
        if (v3)
            p[base + 3] = __float2bfloat16(__bfloat162float(p[base + 3]) - lr * u3);
    }
}

template <typename T>
void launch(
    torch::Tensor p,
    torch::Tensor g,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_absmax,
    torch::Tensor exp_avg_sq_row_q,
    torch::Tensor exp_avg_sq_row_absmax,
    torch::Tensor exp_avg_sq_col_q,
    torch::Tensor exp_avg_sq_col_absmax,
    torch::Tensor exp_avg_res_row_q,
    torch::Tensor exp_avg_res_row_absmax,
    torch::Tensor exp_avg_res_col_q,
    torch::Tensor exp_avg_res_col_absmax,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    torch::Tensor r_res_factor,
    torch::Tensor c_res_factor,
    torch::Tensor res_row_sum,
    torch::Tensor res_col_sum,
    torch::Tensor res_row_partial,
    torch::Tensor res_col_partial,
    torch::Tensor sum_sq_row,
    torch::Tensor sum_update,
    torch::Tensor sum_update_partial,
    torch::Tensor sum_res_row,
    float beta1,
    float beta2,
    float beta3,
    float eps0,
    float eps1,
    float lr,
    float clip_threshold,
    float weight_decay,
    int block_size
) {
    const c10::cuda::CUDAGuard device_guard(p.device());
    const int vec_threads = 256;
    const int rows = (int)p.size(0);
    const int cols = (int)p.size(1);
    const int n = rows * cols;
    const int blocks = (n + THREADS - 1) / THREADS;
    const int row_blocks = (rows + vec_threads - 1) / vec_threads;
    const int col_blocks = (cols + vec_threads - 1) / vec_threads;
    const int row_q_blocks = (rows + block_size - 1) / block_size;
    const int col_q_blocks = (cols + block_size - 1) / block_size;
    const int tile_rows = (rows + TILE - 1) / TILE;
    const int tile_cols = (cols + TILE - 1) / TILE;
    dim3 grid_tiles(tile_cols, tile_rows, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    cudaMemsetAsync(sum_sq_row.data_ptr(), 0, sizeof(float), stream.stream());
    cudaMemsetAsync(sum_update.data_ptr(), 0, sizeof(float), stream.stream());
    cudaMemsetAsync(sum_res_row.data_ptr(), 0, sizeof(float), stream.stream());
    cudaMemsetAsync(res_row_sum.data_ptr(), 0, row_q_blocks * sizeof(float), stream.stream());

    kUpdateSqRowQuantized<T><<<rows, THREADS, 0, stream.stream()>>>(
        (const T*)g.data_ptr(),
        (const uint8_t*)exp_avg_sq_row_q.data_ptr(),
        (const float*)exp_avg_sq_row_absmax.data_ptr(),
        (float*)r_factor.data_ptr(),
        (float*)res_row_sum.data_ptr(),
        beta2,
        eps0,
        rows,
        cols,
        block_size
    );
    kReduceVectorPartial<<<row_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)r_factor.data_ptr(),
        (float*)sum_update_partial.data_ptr(),
        rows
    );
    kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
        (const float*)sum_update_partial.data_ptr(),
        (float*)sum_sq_row.data_ptr(),
        row_blocks
    );
    kUpdateSqColQuantizedFinalize<T><<<col_q_blocks, THREADS, 0, stream.stream()>>>(
        (const T*)g.data_ptr(),
        (uint8_t*)exp_avg_sq_col_q.data_ptr(),
        (float*)exp_avg_sq_col_absmax.data_ptr(),
        (float*)c_factor.data_ptr(),
        beta2,
        eps0,
        rows,
        cols,
        block_size
    );

    // Quantize updated state and turn the same workspace into final factors.
    kFinalizeQuantizedRFactor<<<row_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)r_factor.data_ptr(),
        (uint8_t*)exp_avg_sq_row_q.data_ptr(),
        (float*)exp_avg_sq_row_absmax.data_ptr(),
        (const float*)res_row_sum.data_ptr(),
        (float*)r_factor.data_ptr(),
        (const float*)sum_sq_row.data_ptr(),
        rows,
        block_size
    );
    const bool can_vec2 = ((cols & 1) == 0) && ((((uintptr_t)g.data_ptr()) & 3) == 0);
    if constexpr (std::is_same_v<T, half>) {
        if (can_vec2) {
            const int blocks2 = ((n / 2) + THREADS - 1) / THREADS;
            kSumUpdateSqVec2Half<<<blocks2, THREADS, 0, stream.stream()>>>(
                (const half*)g.data_ptr(),
                (const float*)r_factor.data_ptr(),
                (const float*)c_factor.data_ptr(),
                (float*)sum_update_partial.data_ptr(),
                n,
                rows,
                cols
            );
            kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
                (const float*)sum_update_partial.data_ptr(),
                (float*)sum_update.data_ptr(),
                blocks2
            );
        } else {
            kSumUpdateSq<T><<<blocks, THREADS, 0, stream.stream()>>>(
                (const T*)g.data_ptr(),
                (const float*)r_factor.data_ptr(),
                (const float*)c_factor.data_ptr(),
                (float*)sum_update_partial.data_ptr(),
                n,
                rows,
                cols
            );
            kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
                (const float*)sum_update_partial.data_ptr(),
                (float*)sum_update.data_ptr(),
                blocks
            );
        }
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        if (can_vec2) {
            const int blocks2 = ((n / 2) + THREADS - 1) / THREADS;
            kSumUpdateSqVec2Bf16<<<blocks2, THREADS, 0, stream.stream()>>>(
                (const __nv_bfloat16*)g.data_ptr(),
                (const float*)r_factor.data_ptr(),
                (const float*)c_factor.data_ptr(),
                (float*)sum_update_partial.data_ptr(),
                n,
                rows,
                cols
            );
            kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
                (const float*)sum_update_partial.data_ptr(),
                (float*)sum_update.data_ptr(),
                blocks2
            );
        } else {
            kSumUpdateSq<T><<<blocks, THREADS, 0, stream.stream()>>>(
                (const T*)g.data_ptr(),
                (const float*)r_factor.data_ptr(),
                (const float*)c_factor.data_ptr(),
                (float*)sum_update_partial.data_ptr(),
                n,
                rows,
                cols
            );
            kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
                (const float*)sum_update_partial.data_ptr(),
                (float*)sum_update.data_ptr(),
                blocks
            );
        }
    } else {
        kSumUpdateSq<T><<<blocks, THREADS, 0, stream.stream()>>>(
            (const T*)g.data_ptr(),
            (const float*)r_factor.data_ptr(),
            (const float*)c_factor.data_ptr(),
            (float*)sum_update_partial.data_ptr(),
            n,
            rows,
            cols
        );
        kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
            (const float*)sum_update_partial.data_ptr(),
            (float*)sum_update.data_ptr(),
            blocks
        );
    }

    const bool can_vec4_half_bf16 = (cols % 4 == 0) && ((((uintptr_t)p.data_ptr()) & 3) == 0) && ((((uintptr_t)g.data_ptr()) & 3) == 0) &&
                                    ((((uintptr_t)exp_avg_q.data_ptr()) & 3) == 0);
    const bool can_vec4_fp32 = (cols % 4 == 0) && ((((uintptr_t)p.data_ptr()) & 15) == 0) && ((((uintptr_t)g.data_ptr()) & 15) == 0) &&
                               ((((uintptr_t)exp_avg_q.data_ptr()) & 3) == 0);
    if constexpr (std::is_same_v<T, half>) {
        if (can_vec4_half_bf16) {
            kUpdateExpAvgQuantAndResSumsTiledVec4Half<<<grid_tiles, VEC4_THREADS, 0, stream.stream()>>>(
                (half*)p.data_ptr(),
                (const half*)g.data_ptr(),
                (int8_t*)exp_avg_q.data_ptr(),
                (float*)exp_avg_absmax.data_ptr(),
                (const float*)r_factor.data_ptr(),
                (const float*)c_factor.data_ptr(),
                (const float*)sum_update.data_ptr(),
                (float*)res_row_partial.data_ptr(),
                (float*)res_col_partial.data_ptr(),
                beta1,
                eps1,
                clip_threshold,
                weight_decay,
                lr,
                rows,
                cols
            );
        } else {
            kUpdateExpAvgQuantAndResSumsTiled<T><<<grid_tiles, THREADS, 0, stream.stream()>>>(
                (T*)p.data_ptr(),
                (const T*)g.data_ptr(),
                (int8_t*)exp_avg_q.data_ptr(),
                (float*)exp_avg_absmax.data_ptr(),
                (const float*)r_factor.data_ptr(),
                (const float*)c_factor.data_ptr(),
                (const float*)sum_update.data_ptr(),
                (float*)res_row_partial.data_ptr(),
                (float*)res_col_partial.data_ptr(),
                beta1,
                eps1,
                clip_threshold,
                weight_decay,
                lr,
                rows,
                cols
            );
        }
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        if (can_vec4_half_bf16) {
            kUpdateExpAvgQuantAndResSumsTiledVec4Bf16<<<grid_tiles, VEC4_THREADS, 0, stream.stream()>>>(
                (__nv_bfloat16*)p.data_ptr(),
                (const __nv_bfloat16*)g.data_ptr(),
                (int8_t*)exp_avg_q.data_ptr(),
                (float*)exp_avg_absmax.data_ptr(),
                (const float*)r_factor.data_ptr(),
                (const float*)c_factor.data_ptr(),
                (const float*)sum_update.data_ptr(),
                (float*)res_row_partial.data_ptr(),
                (float*)res_col_partial.data_ptr(),
                beta1,
                eps1,
                clip_threshold,
                weight_decay,
                lr,
                rows,
                cols
            );
        } else {
            kUpdateExpAvgQuantAndResSumsTiled<T><<<grid_tiles, THREADS, 0, stream.stream()>>>(
                (T*)p.data_ptr(),
                (const T*)g.data_ptr(),
                (int8_t*)exp_avg_q.data_ptr(),
                (float*)exp_avg_absmax.data_ptr(),
                (const float*)r_factor.data_ptr(),
                (const float*)c_factor.data_ptr(),
                (const float*)sum_update.data_ptr(),
                (float*)res_row_partial.data_ptr(),
                (float*)res_col_partial.data_ptr(),
                beta1,
                eps1,
                clip_threshold,
                weight_decay,
                lr,
                rows,
                cols
            );
        }
    } else {
        if (can_vec4_fp32) {
            kUpdateExpAvgQuantAndResSumsTiledVec4Float<<<grid_tiles, VEC4_THREADS, 0, stream.stream()>>>(
                (float*)p.data_ptr(),
                (const float*)g.data_ptr(),
                (int8_t*)exp_avg_q.data_ptr(),
                (float*)exp_avg_absmax.data_ptr(),
                (const float*)r_factor.data_ptr(),
                (const float*)c_factor.data_ptr(),
                (const float*)sum_update.data_ptr(),
                (float*)res_row_partial.data_ptr(),
                (float*)res_col_partial.data_ptr(),
                beta1,
                eps1,
                clip_threshold,
                weight_decay,
                lr,
                rows,
                cols
            );
        } else {
            kUpdateExpAvgQuantAndResSumsTiled<T><<<grid_tiles, THREADS, 0, stream.stream()>>>(
                (T*)p.data_ptr(),
                (const T*)g.data_ptr(),
                (int8_t*)exp_avg_q.data_ptr(),
                (float*)exp_avg_absmax.data_ptr(),
                (const float*)r_factor.data_ptr(),
                (const float*)c_factor.data_ptr(),
                (const float*)sum_update.data_ptr(),
                (float*)res_row_partial.data_ptr(),
                (float*)res_col_partial.data_ptr(),
                beta1,
                eps1,
                clip_threshold,
                weight_decay,
                lr,
                rows,
                cols
            );
        }
    }

    kReduceRowTilePartials<<<row_blocks, vec_threads, 0, stream.stream()>>>(
        (const float*)res_row_partial.data_ptr(),
        (float*)res_row_sum.data_ptr(),
        rows,
        tile_cols
    );
    kReduceColTilePartials<<<col_blocks, vec_threads, 0, stream.stream()>>>(
        (const float*)res_col_partial.data_ptr(),
        (float*)res_col_sum.data_ptr(),
        cols,
        tile_rows
    );

    cudaMemsetAsync(r_factor.data_ptr(), 0, row_q_blocks * sizeof(float), stream.stream());

    kUpdateResRowQuantized<<<row_blocks, vec_threads, 0, stream.stream()>>>(
        (const float*)res_row_sum.data_ptr(),
        (const uint8_t*)exp_avg_res_row_q.data_ptr(),
        (const float*)exp_avg_res_row_absmax.data_ptr(),
        (float*)r_res_factor.data_ptr(),
        (float*)r_factor.data_ptr(),
        beta3,
        rows,
        cols,
        block_size
    );
    kReduceVectorPartial<<<row_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)r_res_factor.data_ptr(),
        (float*)sum_update_partial.data_ptr(),
        rows
    );
    kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
        (const float*)sum_update_partial.data_ptr(),
        (float*)sum_res_row.data_ptr(),
        row_blocks
    );
    kUpdateResColQuantizedFinalize<<<col_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)res_col_sum.data_ptr(),
        (uint8_t*)exp_avg_res_col_q.data_ptr(),
        (float*)exp_avg_res_col_absmax.data_ptr(),
        (float*)c_res_factor.data_ptr(),
        beta3,
        rows,
        cols,
        block_size
    );

    // Quantize updated residual state and turn the same workspace into final factors.
    kFinalizeQuantizedRFactor<<<row_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)r_res_factor.data_ptr(),
        (uint8_t*)exp_avg_res_row_q.data_ptr(),
        (float*)exp_avg_res_row_absmax.data_ptr(),
        (const float*)r_factor.data_ptr(),
        (float*)r_res_factor.data_ptr(),
        (const float*)sum_res_row.data_ptr(),
        rows,
        block_size
    );
    if constexpr (std::is_same_v<T, half>) {
        if (can_vec4_half_bf16) {
            kParamUpdateTiledVec4Half<<<grid_tiles, VEC4_THREADS, 0, stream.stream()>>>(
                (half*)p.data_ptr(),
                (const int8_t*)exp_avg_q.data_ptr(),
                (const float*)exp_avg_absmax.data_ptr(),
                (const float*)r_res_factor.data_ptr(),
                (const float*)c_res_factor.data_ptr(),
                lr,
                rows,
                cols
            );
        } else {
            kParamUpdateTiled<T><<<grid_tiles, THREADS, 0, stream.stream()>>>(
                (T*)p.data_ptr(),
                (const int8_t*)exp_avg_q.data_ptr(),
                (const float*)exp_avg_absmax.data_ptr(),
                (const float*)r_res_factor.data_ptr(),
                (const float*)c_res_factor.data_ptr(),
                lr,
                rows,
                cols
            );
        }
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        if (can_vec4_half_bf16) {
            kParamUpdateTiledVec4Bf16<<<grid_tiles, VEC4_THREADS, 0, stream.stream()>>>(
                (__nv_bfloat16*)p.data_ptr(),
                (const int8_t*)exp_avg_q.data_ptr(),
                (const float*)exp_avg_absmax.data_ptr(),
                (const float*)r_res_factor.data_ptr(),
                (const float*)c_res_factor.data_ptr(),
                lr,
                rows,
                cols
            );
        } else {
            kParamUpdateTiled<T><<<grid_tiles, THREADS, 0, stream.stream()>>>(
                (T*)p.data_ptr(),
                (const int8_t*)exp_avg_q.data_ptr(),
                (const float*)exp_avg_absmax.data_ptr(),
                (const float*)r_res_factor.data_ptr(),
                (const float*)c_res_factor.data_ptr(),
                lr,
                rows,
                cols
            );
        }
    } else {
        if (can_vec4_fp32) {
            kParamUpdateTiledVec4Float<<<grid_tiles, VEC4_THREADS, 0, stream.stream()>>>(
                (float*)p.data_ptr(),
                (const int8_t*)exp_avg_q.data_ptr(),
                (const float*)exp_avg_absmax.data_ptr(),
                (const float*)r_res_factor.data_ptr(),
                (const float*)c_res_factor.data_ptr(),
                lr,
                rows,
                cols
            );
        } else {
            kParamUpdateTiled<T><<<grid_tiles, THREADS, 0, stream.stream()>>>(
                (T*)p.data_ptr(),
                (const int8_t*)exp_avg_q.data_ptr(),
                (const float*)exp_avg_absmax.data_ptr(),
                (const float*)r_res_factor.data_ptr(),
                (const float*)c_res_factor.data_ptr(),
                lr,
                rows,
                cols
            );
        }
    }
}

} // namespace

void blockwise_quant_cuda(
    torch::Tensor src,
    torch::Tensor q_out,
    torch::Tensor absmax_out,
    int64_t block_size,
    bool signed_quant
) {
    const c10::cuda::CUDAGuard device_guard(src.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t numel = src.numel();
    const int blocks = (int)((numel + block_size - 1) / block_size);
    if (signed_quant) {
        kBlockwiseQuantF32<true><<<blocks, THREADS, 0, stream.stream()>>>(
            (const float*)src.data_ptr(),
            (int8_t*)q_out.data_ptr(),
            (float*)absmax_out.data_ptr(),
            numel,
            (int)block_size
        );
    } else {
        kBlockwiseQuantF32<false><<<blocks, THREADS, 0, stream.stream()>>>(
            (const float*)src.data_ptr(),
            (uint8_t*)q_out.data_ptr(),
            (float*)absmax_out.data_ptr(),
            numel,
            (int)block_size
        );
    }
}

void blockwise_dequant_cuda(
    torch::Tensor out,
    torch::Tensor q_in,
    torch::Tensor absmax,
    int64_t block_size,
    bool signed_quant
) {
    const c10::cuda::CUDAGuard device_guard(out.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t numel = out.numel();
    const int blocks = (int)((numel + block_size - 1) / block_size);
    if (signed_quant) {
        kBlockwiseDequantF32<true><<<blocks, THREADS, 0, stream.stream()>>>(
            (float*)out.data_ptr(),
            (const int8_t*)q_in.data_ptr(),
            (const float*)absmax.data_ptr(),
            numel,
            (int)block_size
        );
    } else {
        kBlockwiseDequantF32<false><<<blocks, THREADS, 0, stream.stream()>>>(
            (float*)out.data_ptr(),
            (const uint8_t*)q_in.data_ptr(),
            (const float*)absmax.data_ptr(),
            numel,
            (int)block_size
        );
    }
}

void blockwise_quant_batched_cuda(
    torch::Tensor src,
    torch::Tensor q_out,
    torch::Tensor absmax_out,
    int64_t block_size,
    bool signed_quant
) {
    const c10::cuda::CUDAGuard device_guard(src.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int batch = (int)src.size(0);
    const int64_t per_item_numel = src.numel() / batch;
    const int blocks_per_item = (int)((per_item_numel + block_size - 1) / block_size);
    const int total_blocks = batch * blocks_per_item;
    if (signed_quant) {
        kBlockwiseQuantF32Batched<true><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (const float*)src.data_ptr(),
            (int8_t*)q_out.data_ptr(),
            (float*)absmax_out.data_ptr(),
            per_item_numel,
            blocks_per_item,
            (int)block_size
        );
    } else {
        kBlockwiseQuantF32Batched<false><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (const float*)src.data_ptr(),
            (uint8_t*)q_out.data_ptr(),
            (float*)absmax_out.data_ptr(),
            per_item_numel,
            blocks_per_item,
            (int)block_size
        );
    }
}

void blockwise_dequant_batched_cuda(
    torch::Tensor out,
    torch::Tensor q_in,
    torch::Tensor absmax,
    int64_t block_size,
    bool signed_quant
) {
    const c10::cuda::CUDAGuard device_guard(out.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int batch = (int)out.size(0);
    const int64_t per_item_numel = out.numel() / batch;
    const int blocks_per_item = (int)((per_item_numel + block_size - 1) / block_size);
    const int total_blocks = batch * blocks_per_item;
    if (signed_quant) {
        kBlockwiseDequantF32Batched<true><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (float*)out.data_ptr(),
            (const int8_t*)q_in.data_ptr(),
            (const float*)absmax.data_ptr(),
            per_item_numel,
            blocks_per_item,
            (int)block_size
        );
    } else {
        kBlockwiseDequantF32Batched<false><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (float*)out.data_ptr(),
            (const uint8_t*)q_in.data_ptr(),
            (const float*)absmax.data_ptr(),
            per_item_numel,
            blocks_per_item,
            (int)block_size
        );
    }
}

void came_full_nonfactored_step_cuda(
    torch::Tensor p,
    torch::Tensor g32,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_absmax,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor exp_avg_sq_absmax,
    torch::Tensor update,
    torch::Tensor sum_update_partial,
    torch::Tensor sum_update,
    double beta1,
    double beta2,
    double eps0,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(p.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t numel = p.numel();
    const int blocks = (int)((numel + block_size - 1) / block_size);
    kCameNonfactoredSqUpdateAndQuant<<<blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (uint8_t*)exp_avg_sq_q.data_ptr(),
        (float*)exp_avg_sq_absmax.data_ptr(),
        (float*)update.data_ptr(),
        (float*)sum_update_partial.data_ptr(),
        (float)beta2,
        (float)eps0,
        numel,
        (int)block_size
    );
    kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
        (const float*)sum_update_partial.data_ptr(),
        (float*)sum_update.data_ptr(),
        blocks
    );

    const auto dtype = p.scalar_type();
    if (dtype == torch::kFloat32) {
        kCameNonfactoredExpAvgParamUpdate<float><<<blocks, THREADS, 0, stream.stream()>>>(
            (float*)p.data_ptr(),
            (const float*)update.data_ptr(),
            (int8_t*)exp_avg_q.data_ptr(),
            (float*)exp_avg_absmax.data_ptr(),
            (const float*)sum_update.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            numel,
            (int)block_size
        );
    } else if (dtype == torch::kFloat16) {
        kCameNonfactoredExpAvgParamUpdate<half><<<blocks, THREADS, 0, stream.stream()>>>(
            (half*)p.data_ptr(),
            (const float*)update.data_ptr(),
            (int8_t*)exp_avg_q.data_ptr(),
            (float*)exp_avg_absmax.data_ptr(),
            (const float*)sum_update.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            numel,
            (int)block_size
        );
    } else {
        kCameNonfactoredExpAvgParamUpdate<__nv_bfloat16><<<blocks, THREADS, 0, stream.stream()>>>(
            (__nv_bfloat16*)p.data_ptr(),
            (const float*)update.data_ptr(),
            (int8_t*)exp_avg_q.data_ptr(),
            (float*)exp_avg_absmax.data_ptr(),
            (const float*)sum_update.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            numel,
            (int)block_size
        );
    }
}

void came_full_nonfactored_step_fp16_update_cuda(
    torch::Tensor p,
    torch::Tensor g32,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_absmax,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor exp_avg_sq_absmax,
    torch::Tensor update,
    torch::Tensor sum_update_partial,
    torch::Tensor sum_update,
    double beta1,
    double beta2,
    double eps0,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(p.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t numel = p.numel();
    const int blocks = (int)((numel + block_size - 1) / block_size);
    kCameNonfactoredSqUpdateAndQuantTypedUpdate<half><<<blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (uint8_t*)exp_avg_sq_q.data_ptr(),
        (float*)exp_avg_sq_absmax.data_ptr(),
        (half*)update.data_ptr(),
        (float*)sum_update_partial.data_ptr(),
        (float)beta2,
        (float)eps0,
        numel,
        (int)block_size
    );
    kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
        (const float*)sum_update_partial.data_ptr(),
        (float*)sum_update.data_ptr(),
        blocks
    );

    const auto dtype = p.scalar_type();
    if (dtype == torch::kFloat32) {
        kCameNonfactoredExpAvgParamUpdateTypedUpdate<float, half><<<blocks, THREADS, 0, stream.stream()>>>(
            (float*)p.data_ptr(),
            (const half*)update.data_ptr(),
            (int8_t*)exp_avg_q.data_ptr(),
            (float*)exp_avg_absmax.data_ptr(),
            (const float*)sum_update.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            numel,
            (int)block_size
        );
    } else if (dtype == torch::kFloat16) {
        kCameNonfactoredExpAvgParamUpdateTypedUpdate<half, half><<<blocks, THREADS, 0, stream.stream()>>>(
            (half*)p.data_ptr(),
            (const half*)update.data_ptr(),
            (int8_t*)exp_avg_q.data_ptr(),
            (float*)exp_avg_absmax.data_ptr(),
            (const float*)sum_update.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            numel,
            (int)block_size
        );
    } else {
        kCameNonfactoredExpAvgParamUpdateTypedUpdate<__nv_bfloat16, half><<<blocks, THREADS, 0, stream.stream()>>>(
            (__nv_bfloat16*)p.data_ptr(),
            (const half*)update.data_ptr(),
            (int8_t*)exp_avg_q.data_ptr(),
            (float*)exp_avg_absmax.data_ptr(),
            (const float*)sum_update.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            numel,
            (int)block_size
        );
    }
}

void came_full_nonfactored_step_batched_cuda(
    torch::Tensor p,
    torch::Tensor g32,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_absmax,
    torch::Tensor exp_avg_sq_q,
    torch::Tensor exp_avg_sq_absmax,
    torch::Tensor update,
    torch::Tensor sum_update_partial,
    torch::Tensor sum_update,
    double beta1,
    double beta2,
    double eps0,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(p.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int batch = (int)p.size(0);
    const int64_t per_item_numel = p.numel() / batch;
    const int blocks_per_item = (int)((per_item_numel + block_size - 1) / block_size);
    const int total_blocks = batch * blocks_per_item;

    kCameNonfactoredSqUpdateAndQuantBatched<<<total_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (uint8_t*)exp_avg_sq_q.data_ptr(),
        (float*)exp_avg_sq_absmax.data_ptr(),
        (float*)update.data_ptr(),
        (float*)sum_update_partial.data_ptr(),
        (float)beta2,
        (float)eps0,
        per_item_numel,
        blocks_per_item,
        (int)block_size
    );
    kReducePartialSumBatched<<<batch, THREADS, 0, stream.stream()>>>(
        (const float*)sum_update_partial.data_ptr(),
        (float*)sum_update.data_ptr(),
        blocks_per_item
    );

    const auto dtype = p.scalar_type();
    if (dtype == torch::kFloat32) {
        kCameNonfactoredExpAvgParamUpdateBatched<float><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (float*)p.data_ptr(),
            (const float*)update.data_ptr(),
            (int8_t*)exp_avg_q.data_ptr(),
            (float*)exp_avg_absmax.data_ptr(),
            (const float*)sum_update.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            per_item_numel,
            blocks_per_item,
            (int)block_size
        );
    } else if (dtype == torch::kFloat16) {
        kCameNonfactoredExpAvgParamUpdateBatched<half><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (half*)p.data_ptr(),
            (const float*)update.data_ptr(),
            (int8_t*)exp_avg_q.data_ptr(),
            (float*)exp_avg_absmax.data_ptr(),
            (const float*)sum_update.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            per_item_numel,
            blocks_per_item,
            (int)block_size
        );
    } else {
        kCameNonfactoredExpAvgParamUpdateBatched<__nv_bfloat16><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (__nv_bfloat16*)p.data_ptr(),
            (const float*)update.data_ptr(),
            (int8_t*)exp_avg_q.data_ptr(),
            (float*)exp_avg_absmax.data_ptr(),
            (const float*)sum_update.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            per_item_numel,
            blocks_per_item,
            (int)block_size
        );
    }
}

void came_full_nonfactored_step_multitensor_cuda(
    torch::Tensor p_ptrs,
    torch::Tensor g32_ptrs,
    torch::Tensor exp_avg_q_ptrs,
    torch::Tensor exp_avg_absmax_ptrs,
    torch::Tensor exp_avg_sq_q_ptrs,
    torch::Tensor exp_avg_sq_absmax_ptrs,
    torch::Tensor update_ptrs,
    torch::Tensor sum_update_ptrs,
    torch::ScalarType param_dtype,
    int64_t per_item_numel,
    double beta1,
    double beta2,
    double eps0,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(p_ptrs.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int batch = (int)p_ptrs.numel();
    const int blocks_per_item = (int)((per_item_numel + block_size - 1) / block_size);
    const int total_blocks = batch * blocks_per_item;

    kZeroScalarsMultiTensor<<<batch, 1, 0, stream.stream()>>>(
        (const int64_t*)sum_update_ptrs.data_ptr()
    );
    kCameNonfactoredSqUpdateAndQuantMultiTensor<<<total_blocks, THREADS, 0, stream.stream()>>>(
        (const int64_t*)g32_ptrs.data_ptr(),
        (const int64_t*)exp_avg_sq_q_ptrs.data_ptr(),
        (const int64_t*)exp_avg_sq_absmax_ptrs.data_ptr(),
        (const int64_t*)update_ptrs.data_ptr(),
        (const int64_t*)sum_update_ptrs.data_ptr(),
        (float)beta2,
        (float)eps0,
        per_item_numel,
        blocks_per_item,
        (int)block_size
    );

    if (param_dtype == torch::kFloat32) {
        kCameNonfactoredExpAvgParamUpdateMultiTensor<float><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const int64_t*)update_ptrs.data_ptr(),
            (const int64_t*)exp_avg_q_ptrs.data_ptr(),
            (const int64_t*)exp_avg_absmax_ptrs.data_ptr(),
            (const int64_t*)sum_update_ptrs.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            per_item_numel,
            blocks_per_item,
            (int)block_size
        );
    } else if (param_dtype == torch::kFloat16) {
        kCameNonfactoredExpAvgParamUpdateMultiTensor<half><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const int64_t*)update_ptrs.data_ptr(),
            (const int64_t*)exp_avg_q_ptrs.data_ptr(),
            (const int64_t*)exp_avg_absmax_ptrs.data_ptr(),
            (const int64_t*)sum_update_ptrs.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            per_item_numel,
            blocks_per_item,
            (int)block_size
        );
    } else {
        kCameNonfactoredExpAvgParamUpdateMultiTensor<__nv_bfloat16><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const int64_t*)update_ptrs.data_ptr(),
            (const int64_t*)exp_avg_q_ptrs.data_ptr(),
            (const int64_t*)exp_avg_absmax_ptrs.data_ptr(),
            (const int64_t*)sum_update_ptrs.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            per_item_numel,
            blocks_per_item,
            (int)block_size
        );
    }
}

void came_full_nonfactored_step_multitensor_fp16_update_cuda(
    torch::Tensor p_ptrs,
    torch::Tensor g32_ptrs,
    torch::Tensor exp_avg_q_ptrs,
    torch::Tensor exp_avg_absmax_ptrs,
    torch::Tensor exp_avg_sq_q_ptrs,
    torch::Tensor exp_avg_sq_absmax_ptrs,
    torch::Tensor update_ptrs,
    torch::Tensor sum_update_ptrs,
    torch::ScalarType param_dtype,
    int64_t per_item_numel,
    double beta1,
    double beta2,
    double eps0,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(p_ptrs.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int batch = (int)p_ptrs.numel();
    const int blocks_per_item = (int)((per_item_numel + block_size - 1) / block_size);
    const int total_blocks = batch * blocks_per_item;

    kZeroScalarsMultiTensor<<<batch, 1, 0, stream.stream()>>>(
        (const int64_t*)sum_update_ptrs.data_ptr()
    );
    kCameNonfactoredSqUpdateAndQuantMultiTensorTypedUpdate<half><<<total_blocks, THREADS, 0, stream.stream()>>>(
        (const int64_t*)g32_ptrs.data_ptr(),
        (const int64_t*)exp_avg_sq_q_ptrs.data_ptr(),
        (const int64_t*)exp_avg_sq_absmax_ptrs.data_ptr(),
        (const int64_t*)update_ptrs.data_ptr(),
        (const int64_t*)sum_update_ptrs.data_ptr(),
        (float)beta2,
        (float)eps0,
        per_item_numel,
        blocks_per_item,
        (int)block_size
    );

    if (param_dtype == torch::kFloat32) {
        kCameNonfactoredExpAvgParamUpdateMultiTensorTypedUpdate<float, half><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const int64_t*)update_ptrs.data_ptr(),
            (const int64_t*)exp_avg_q_ptrs.data_ptr(),
            (const int64_t*)exp_avg_absmax_ptrs.data_ptr(),
            (const int64_t*)sum_update_ptrs.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            per_item_numel,
            blocks_per_item,
            (int)block_size
        );
    } else if (param_dtype == torch::kFloat16) {
        kCameNonfactoredExpAvgParamUpdateMultiTensorTypedUpdate<half, half><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const int64_t*)update_ptrs.data_ptr(),
            (const int64_t*)exp_avg_q_ptrs.data_ptr(),
            (const int64_t*)exp_avg_absmax_ptrs.data_ptr(),
            (const int64_t*)sum_update_ptrs.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            per_item_numel,
            blocks_per_item,
            (int)block_size
        );
    } else {
        kCameNonfactoredExpAvgParamUpdateMultiTensorTypedUpdate<__nv_bfloat16, half><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const int64_t*)update_ptrs.data_ptr(),
            (const int64_t*)exp_avg_q_ptrs.data_ptr(),
            (const int64_t*)exp_avg_absmax_ptrs.data_ptr(),
            (const int64_t*)sum_update_ptrs.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            per_item_numel,
            blocks_per_item,
            (int)block_size
        );
    }
}

void came_full_nonfactored_step_multitensor_varlen_cuda(
    torch::Tensor p_ptrs,
    torch::Tensor g32_ptrs,
    torch::Tensor exp_avg_q_ptrs,
    torch::Tensor exp_avg_absmax_ptrs,
    torch::Tensor exp_avg_sq_q_ptrs,
    torch::Tensor exp_avg_sq_absmax_ptrs,
    torch::Tensor update_ptrs,
    torch::Tensor sum_update_ptrs,
    torch::Tensor item_numels,
    torch::ScalarType param_dtype,
    int64_t max_item_numel,
    double beta1,
    double beta2,
    double eps0,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(p_ptrs.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int batch = (int)p_ptrs.numel();
    const int max_blocks_per_item = (int)((max_item_numel + block_size - 1) / block_size);
    const dim3 grid((unsigned int)max_blocks_per_item, (unsigned int)batch, 1);

    kZeroScalarsMultiTensor<<<batch, 1, 0, stream.stream()>>>(
        (const int64_t*)sum_update_ptrs.data_ptr()
    );
    kCameNonfactoredSqUpdateAndQuantMultiTensorVarlen<<<grid, THREADS, 0, stream.stream()>>>(
        (const int64_t*)g32_ptrs.data_ptr(),
        (const int64_t*)exp_avg_sq_q_ptrs.data_ptr(),
        (const int64_t*)exp_avg_sq_absmax_ptrs.data_ptr(),
        (const int64_t*)update_ptrs.data_ptr(),
        (const int64_t*)sum_update_ptrs.data_ptr(),
        (const int64_t*)item_numels.data_ptr(),
        (float)beta2,
        (float)eps0,
        (int)block_size
    );

    if (param_dtype == torch::kFloat32) {
        kCameNonfactoredExpAvgParamUpdateMultiTensorVarlen<float><<<grid, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const int64_t*)update_ptrs.data_ptr(),
            (const int64_t*)exp_avg_q_ptrs.data_ptr(),
            (const int64_t*)exp_avg_absmax_ptrs.data_ptr(),
            (const int64_t*)sum_update_ptrs.data_ptr(),
            (const int64_t*)item_numels.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            (int)block_size
        );
    } else if (param_dtype == torch::kFloat16) {
        kCameNonfactoredExpAvgParamUpdateMultiTensorVarlen<half><<<grid, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const int64_t*)update_ptrs.data_ptr(),
            (const int64_t*)exp_avg_q_ptrs.data_ptr(),
            (const int64_t*)exp_avg_absmax_ptrs.data_ptr(),
            (const int64_t*)sum_update_ptrs.data_ptr(),
            (const int64_t*)item_numels.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            (int)block_size
        );
    } else {
        kCameNonfactoredExpAvgParamUpdateMultiTensorVarlen<__nv_bfloat16><<<grid, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const int64_t*)update_ptrs.data_ptr(),
            (const int64_t*)exp_avg_q_ptrs.data_ptr(),
            (const int64_t*)exp_avg_absmax_ptrs.data_ptr(),
            (const int64_t*)sum_update_ptrs.data_ptr(),
            (const int64_t*)item_numels.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            (int)block_size
        );
    }
}

void came_full_nonfactored_step_multitensor_compact_varlen_cuda(
    torch::Tensor p_ptrs,
    torch::Tensor g32_ptrs,
    torch::Tensor exp_avg_q_ptrs,
    torch::Tensor exp_avg_absmax_ptrs,
    torch::Tensor exp_avg_sq_q_ptrs,
    torch::Tensor exp_avg_sq_absmax_ptrs,
    torch::Tensor update_ptrs,
    torch::Tensor sum_update_ptrs,
    torch::Tensor item_numels,
    torch::Tensor block_item_ids,
    torch::Tensor block_item_starts,
    torch::ScalarType param_dtype,
    double beta1,
    double beta2,
    double eps0,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(p_ptrs.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int batch = (int)p_ptrs.numel();
    const int total_blocks = (int)block_item_ids.numel();

    kZeroScalarsMultiTensor<<<batch, 1, 0, stream.stream()>>>(
        (const int64_t*)sum_update_ptrs.data_ptr()
    );
    kCameNonfactoredSqUpdateAndQuantMultiTensorCompactVarlen<<<total_blocks, THREADS, 0, stream.stream()>>>(
        (const int64_t*)g32_ptrs.data_ptr(),
        (const int64_t*)exp_avg_sq_q_ptrs.data_ptr(),
        (const int64_t*)exp_avg_sq_absmax_ptrs.data_ptr(),
        (const int64_t*)update_ptrs.data_ptr(),
        (const int64_t*)sum_update_ptrs.data_ptr(),
        (const int64_t*)item_numels.data_ptr(),
        (const int64_t*)block_item_ids.data_ptr(),
        (const int64_t*)block_item_starts.data_ptr(),
        (float)beta2,
        (float)eps0,
        (int)block_size
    );

    if (param_dtype == torch::kFloat32) {
        kCameNonfactoredExpAvgParamUpdateMultiTensorCompactVarlen<float><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const int64_t*)update_ptrs.data_ptr(),
            (const int64_t*)exp_avg_q_ptrs.data_ptr(),
            (const int64_t*)exp_avg_absmax_ptrs.data_ptr(),
            (const int64_t*)sum_update_ptrs.data_ptr(),
            (const int64_t*)item_numels.data_ptr(),
            (const int64_t*)block_item_ids.data_ptr(),
            (const int64_t*)block_item_starts.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            (int)block_size
        );
    } else if (param_dtype == torch::kFloat16) {
        kCameNonfactoredExpAvgParamUpdateMultiTensorCompactVarlen<half><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const int64_t*)update_ptrs.data_ptr(),
            (const int64_t*)exp_avg_q_ptrs.data_ptr(),
            (const int64_t*)exp_avg_absmax_ptrs.data_ptr(),
            (const int64_t*)sum_update_ptrs.data_ptr(),
            (const int64_t*)item_numels.data_ptr(),
            (const int64_t*)block_item_ids.data_ptr(),
            (const int64_t*)block_item_starts.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            (int)block_size
        );
    } else {
        kCameNonfactoredExpAvgParamUpdateMultiTensorCompactVarlen<__nv_bfloat16><<<total_blocks, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const int64_t*)update_ptrs.data_ptr(),
            (const int64_t*)exp_avg_q_ptrs.data_ptr(),
            (const int64_t*)exp_avg_absmax_ptrs.data_ptr(),
            (const int64_t*)sum_update_ptrs.data_ptr(),
            (const int64_t*)item_numels.data_ptr(),
            (const int64_t*)block_item_ids.data_ptr(),
            (const int64_t*)block_item_starts.data_ptr(),
            (float)beta1,
            (float)lr,
            (float)clip_threshold,
            (float)weight_decay,
            (int)block_size
        );
    }
}

void came_full_factored_sq_step_cuda(
    torch::Tensor g32,
    torch::Tensor exp_avg_sq_row_q,
    torch::Tensor exp_avg_sq_row_absmax,
    torch::Tensor exp_avg_sq_col_q,
    torch::Tensor exp_avg_sq_col_absmax,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    torch::Tensor row_absmax_scratch,
    torch::Tensor reduce_partial,
    torch::Tensor sum_row_state,
    torch::Tensor sum_update,
    double beta2,
    double eps0,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(g32.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = (int)g32.size(0);
    const int cols = (int)g32.size(1);
    const int n = rows * cols;
    const int row_blocks = (rows + THREADS - 1) / THREADS;
    const int col_q_blocks = (cols + (int)block_size - 1) / (int)block_size;
    const int row_q_blocks = (rows + (int)block_size - 1) / (int)block_size;
    const int update_blocks = (n + THREADS - 1) / THREADS;

    cudaMemsetAsync(row_absmax_scratch.data_ptr(), 0, row_q_blocks * sizeof(float), stream.stream());
    kUpdateSqRowQuantized<float><<<rows, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (const uint8_t*)exp_avg_sq_row_q.data_ptr(),
        (const float*)exp_avg_sq_row_absmax.data_ptr(),
        (float*)r_factor.data_ptr(),
        (float*)row_absmax_scratch.data_ptr(),
        (float)beta2,
        (float)eps0,
        rows,
        cols,
        (int)block_size
    );
    kReduceVectorPartial<<<row_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)r_factor.data_ptr(),
        (float*)reduce_partial.data_ptr(),
        rows
    );
    kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
        (const float*)reduce_partial.data_ptr(),
        (float*)sum_row_state.data_ptr(),
        row_blocks
    );
    kUpdateSqColQuantizedFinalize<float><<<col_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (uint8_t*)exp_avg_sq_col_q.data_ptr(),
        (float*)exp_avg_sq_col_absmax.data_ptr(),
        (float*)c_factor.data_ptr(),
        (float)beta2,
        (float)eps0,
        rows,
        cols,
        (int)block_size
    );
    kFinalizeQuantizedRFactor<<<row_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)r_factor.data_ptr(),
        (uint8_t*)exp_avg_sq_row_q.data_ptr(),
        (float*)exp_avg_sq_row_absmax.data_ptr(),
        (const float*)row_absmax_scratch.data_ptr(),
        (float*)r_factor.data_ptr(),
        (const float*)sum_row_state.data_ptr(),
        rows,
        (int)block_size
    );
    kSumUpdateSq<float><<<update_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (const float*)r_factor.data_ptr(),
        (const float*)c_factor.data_ptr(),
        (float*)reduce_partial.data_ptr(),
        n,
        rows,
        cols
    );
    kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
        (const float*)reduce_partial.data_ptr(),
        (float*)sum_update.data_ptr(),
        update_blocks
    );
}

void came_full_factored_sq_step_batched_cuda(
    torch::Tensor g32,
    torch::Tensor exp_avg_sq_row_q,
    torch::Tensor exp_avg_sq_row_absmax,
    torch::Tensor exp_avg_sq_col_q,
    torch::Tensor exp_avg_sq_col_absmax,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    torch::Tensor row_absmax_scratch,
    torch::Tensor reduce_partial,
    torch::Tensor sum_row_state,
    torch::Tensor sum_update_slice,
    torch::Tensor sum_update_total,
    double beta2,
    double eps0,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(g32.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int batch = (int)g32.size(0);
    const int rows = (int)g32.size(1);
    const int cols = (int)g32.size(2);
    const int64_t per_item_numel = (int64_t)rows * cols;
    const int n = rows * cols;
    const int row_blocks = (rows + THREADS - 1) / THREADS;
    const int col_q_blocks = (cols + (int)block_size - 1) / (int)block_size;
    const int row_q_blocks = (rows + (int)block_size - 1) / (int)block_size;
    const int update_blocks = (n + THREADS - 1) / THREADS;

    cudaMemsetAsync(row_absmax_scratch.data_ptr(), 0, row_absmax_scratch.numel() * sizeof(float), stream.stream());
    kUpdateSqRowQuantizedBatched<float><<<batch * rows, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (const uint8_t*)exp_avg_sq_row_q.data_ptr(),
        (const float*)exp_avg_sq_row_absmax.data_ptr(),
        (float*)r_factor.data_ptr(),
        (float*)row_absmax_scratch.data_ptr(),
        (float)beta2,
        (float)eps0,
        per_item_numel,
        rows,
        cols,
        row_q_blocks,
        (int)block_size
    );
    kReduceVectorPartialBatched<<<batch * row_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)r_factor.data_ptr(),
        (float*)reduce_partial.data_ptr(),
        rows,
        row_blocks
    );
    kReducePartialSumBatched<<<batch, THREADS, 0, stream.stream()>>>(
        (const float*)reduce_partial.data_ptr(),
        (float*)sum_row_state.data_ptr(),
        row_blocks
    );
    kUpdateSqColQuantizedFinalizeBatched<float><<<batch * col_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (uint8_t*)exp_avg_sq_col_q.data_ptr(),
        (float*)exp_avg_sq_col_absmax.data_ptr(),
        (float*)c_factor.data_ptr(),
        (float)beta2,
        (float)eps0,
        per_item_numel,
        rows,
        cols,
        col_q_blocks,
        (int)block_size
    );
    kFinalizeQuantizedRFactorBatched<<<batch * row_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)r_factor.data_ptr(),
        (uint8_t*)exp_avg_sq_row_q.data_ptr(),
        (float*)exp_avg_sq_row_absmax.data_ptr(),
        (const float*)row_absmax_scratch.data_ptr(),
        (float*)r_factor.data_ptr(),
        (const float*)sum_row_state.data_ptr(),
        rows,
        row_q_blocks,
        (int)block_size
    );
    kSumUpdateSqBatched<float><<<batch * update_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (const float*)r_factor.data_ptr(),
        (const float*)c_factor.data_ptr(),
        (float*)reduce_partial.data_ptr(),
        per_item_numel,
        rows,
        cols,
        update_blocks
    );
    kReducePartialSumBatched<<<batch, THREADS, 0, stream.stream()>>>(
        (const float*)reduce_partial.data_ptr(),
        (float*)sum_update_slice.data_ptr(),
        update_blocks
    );
    kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
        (const float*)sum_update_slice.data_ptr(),
        (float*)sum_update_total.data_ptr(),
        batch
    );
}

void came_full_factored_res_step_cuda(
    torch::Tensor res32,
    torch::Tensor exp_avg_res_row_q,
    torch::Tensor exp_avg_res_row_absmax,
    torch::Tensor exp_avg_res_col_q,
    torch::Tensor exp_avg_res_col_absmax,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    torch::Tensor row_absmax_scratch,
    torch::Tensor reduce_partial,
    torch::Tensor sum_row_state,
    double beta3,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(res32.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = (int)res32.size(0);
    const int cols = (int)res32.size(1);
    const int row_blocks = (rows + THREADS - 1) / THREADS;
    const int col_q_blocks = (cols + (int)block_size - 1) / (int)block_size;
    const int row_q_blocks = (rows + (int)block_size - 1) / (int)block_size;

    cudaMemsetAsync(row_absmax_scratch.data_ptr(), 0, row_q_blocks * sizeof(float), stream.stream());
    kUpdateMeanRowQuantized<<<rows, THREADS, 0, stream.stream()>>>(
        (const float*)res32.data_ptr(),
        (const uint8_t*)exp_avg_res_row_q.data_ptr(),
        (const float*)exp_avg_res_row_absmax.data_ptr(),
        (float*)r_factor.data_ptr(),
        (float*)row_absmax_scratch.data_ptr(),
        (float)beta3,
        rows,
        cols,
        (int)block_size
    );
    kReduceVectorPartial<<<row_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)r_factor.data_ptr(),
        (float*)reduce_partial.data_ptr(),
        rows
    );
    kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
        (const float*)reduce_partial.data_ptr(),
        (float*)sum_row_state.data_ptr(),
        row_blocks
    );
    kUpdateMeanColQuantizedFinalize<<<col_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)res32.data_ptr(),
        (uint8_t*)exp_avg_res_col_q.data_ptr(),
        (float*)exp_avg_res_col_absmax.data_ptr(),
        (float*)c_factor.data_ptr(),
        (float)beta3,
        rows,
        cols,
        (int)block_size
    );
    kFinalizeQuantizedRFactor<<<row_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)r_factor.data_ptr(),
        (uint8_t*)exp_avg_res_row_q.data_ptr(),
        (float*)exp_avg_res_row_absmax.data_ptr(),
        (const float*)row_absmax_scratch.data_ptr(),
        (float*)r_factor.data_ptr(),
        (const float*)sum_row_state.data_ptr(),
        rows,
        (int)block_size
    );
}

void came_full_factored_res_step_batched_cuda(
    torch::Tensor res32,
    torch::Tensor exp_avg_res_row_q,
    torch::Tensor exp_avg_res_row_absmax,
    torch::Tensor exp_avg_res_col_q,
    torch::Tensor exp_avg_res_col_absmax,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    torch::Tensor row_absmax_scratch,
    torch::Tensor reduce_partial,
    torch::Tensor sum_row_state,
    double beta3,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(res32.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int batch = (int)res32.size(0);
    const int rows = (int)res32.size(1);
    const int cols = (int)res32.size(2);
    const int64_t per_item_numel = (int64_t)rows * cols;
    const int row_blocks = (rows + THREADS - 1) / THREADS;
    const int col_q_blocks = (cols + (int)block_size - 1) / (int)block_size;
    const int row_q_blocks = (rows + (int)block_size - 1) / (int)block_size;

    cudaMemsetAsync(row_absmax_scratch.data_ptr(), 0, row_absmax_scratch.numel() * sizeof(float), stream.stream());
    kUpdateMeanRowQuantizedBatched<<<batch * rows, THREADS, 0, stream.stream()>>>(
        (const float*)res32.data_ptr(),
        (const uint8_t*)exp_avg_res_row_q.data_ptr(),
        (const float*)exp_avg_res_row_absmax.data_ptr(),
        (float*)r_factor.data_ptr(),
        (float*)row_absmax_scratch.data_ptr(),
        (float)beta3,
        per_item_numel,
        rows,
        cols,
        row_q_blocks,
        (int)block_size
    );
    kReduceVectorPartialBatched<<<batch * row_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)r_factor.data_ptr(),
        (float*)reduce_partial.data_ptr(),
        rows,
        row_blocks
    );
    kReducePartialSumBatched<<<batch, THREADS, 0, stream.stream()>>>(
        (const float*)reduce_partial.data_ptr(),
        (float*)sum_row_state.data_ptr(),
        row_blocks
    );
    kUpdateMeanColQuantizedFinalizeBatched<<<batch * col_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)res32.data_ptr(),
        (uint8_t*)exp_avg_res_col_q.data_ptr(),
        (float*)exp_avg_res_col_absmax.data_ptr(),
        (float*)c_factor.data_ptr(),
        (float)beta3,
        per_item_numel,
        rows,
        cols,
        col_q_blocks,
        (int)block_size
    );
    kFinalizeQuantizedRFactorBatched<<<batch * row_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)r_factor.data_ptr(),
        (uint8_t*)exp_avg_res_row_q.data_ptr(),
        (float*)exp_avg_res_row_absmax.data_ptr(),
        (const float*)row_absmax_scratch.data_ptr(),
        (float*)r_factor.data_ptr(),
        (const float*)sum_row_state.data_ptr(),
        rows,
        row_q_blocks,
        (int)block_size
    );
}

void came_full_factored_expavg_res_prepare_cuda(
    torch::Tensor g32,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_absmax,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    torch::Tensor exp_avg_fp32,
    torch::Tensor res32,
    torch::Tensor sum_update,
    double beta1,
    double eps1,
    double clip_threshold,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(g32.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = (int)g32.size(0);
    const int cols = (int)g32.size(1);
    const int64_t numel = g32.numel();
    const int blocks = (int)((numel + block_size - 1) / block_size);
    kCameFactoredExpAvgResPrepare<<<blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (int8_t*)exp_avg_q.data_ptr(),
        (float*)exp_avg_absmax.data_ptr(),
        (const float*)r_factor.data_ptr(),
        (const float*)c_factor.data_ptr(),
        (float*)exp_avg_fp32.data_ptr(),
        (float*)res32.data_ptr(),
        (const float*)sum_update.data_ptr(),
        (float)beta1,
        (float)eps1,
        (float)clip_threshold,
        rows,
        cols,
        (int)block_size
    );
}

void came_full_factored_expavg_res_prepare_batched_cuda(
    torch::Tensor g32,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_absmax,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    torch::Tensor exp_avg_fp32,
    torch::Tensor res32,
    torch::Tensor sum_update,
    double beta1,
    double eps1,
    double clip_threshold,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(g32.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int batch = (int)g32.size(0);
    const int rows = (int)g32.size(1);
    const int cols = (int)g32.size(2);
    const int64_t per_item_numel = (int64_t)rows * cols;
    const int blocks_per_item = (int)((per_item_numel + block_size - 1) / block_size);
    const int total_blocks = batch * blocks_per_item;
    kCameFactoredExpAvgResPrepareBatched<<<total_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (int8_t*)exp_avg_q.data_ptr(),
        (float*)exp_avg_absmax.data_ptr(),
        (const float*)r_factor.data_ptr(),
        (const float*)c_factor.data_ptr(),
        (float*)exp_avg_fp32.data_ptr(),
        (float*)res32.data_ptr(),
        (const float*)sum_update.data_ptr(),
        (float)beta1,
        (float)eps1,
        (float)clip_threshold,
        per_item_numel,
        rows,
        cols,
        blocks_per_item,
        (int)block_size
    );
}

void came_full_factored_param_update_cuda(
    torch::Tensor p,
    torch::Tensor exp_avg_fp32,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    double lr,
    double weight_decay
) {
    const c10::cuda::CUDAGuard device_guard(p.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = (int)p.size(0);
    const int cols = (int)p.size(1);
    const int64_t numel = p.numel();
    const int blocks = (int)((numel + THREADS - 1) / THREADS);
    const auto dtype = p.scalar_type();
    if (dtype == torch::kFloat32) {
        kCameFactoredParamUpdate<float><<<blocks, THREADS, 0, stream.stream()>>>(
            (float*)p.data_ptr(),
            (const float*)exp_avg_fp32.data_ptr(),
            (const float*)r_factor.data_ptr(),
            (const float*)c_factor.data_ptr(),
            (float)lr,
            (float)weight_decay,
            rows,
            cols
        );
    } else if (dtype == torch::kFloat16) {
        kCameFactoredParamUpdate<half><<<blocks, THREADS, 0, stream.stream()>>>(
            (half*)p.data_ptr(),
            (const float*)exp_avg_fp32.data_ptr(),
            (const float*)r_factor.data_ptr(),
            (const float*)c_factor.data_ptr(),
            (float)lr,
            (float)weight_decay,
            rows,
            cols
        );
    } else {
        kCameFactoredParamUpdate<__nv_bfloat16><<<blocks, THREADS, 0, stream.stream()>>>(
            (__nv_bfloat16*)p.data_ptr(),
            (const float*)exp_avg_fp32.data_ptr(),
            (const float*)r_factor.data_ptr(),
            (const float*)c_factor.data_ptr(),
            (float)lr,
            (float)weight_decay,
            rows,
            cols
        );
    }
}

void came_fp_factored_step_cuda(
    torch::Tensor p,
    torch::Tensor g32,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq_row,
    torch::Tensor exp_avg_sq_col,
    torch::Tensor exp_avg_res_row,
    torch::Tensor exp_avg_res_col,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    torch::Tensor scratch,
    torch::Tensor reduce_partial,
    torch::Tensor sum_row_state,
    torch::Tensor sum_update,
    double beta1,
    double beta2,
    double beta3,
    double eps0,
    double eps1,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(p.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int rows = (int)g32.size(0);
    const int cols = (int)g32.size(1);
    const int64_t numel = g32.numel();
    const int row_blocks = (rows + THREADS - 1) / THREADS;
    const int col_blocks = (cols + THREADS - 1) / THREADS;
    const int update_blocks = (int)((numel + block_size - 1) / block_size);
    const int reduce_blocks = (int)((numel + THREADS - 1) / THREADS);

    kUpdateSqRowFp32<<<rows, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (float*)exp_avg_sq_row.data_ptr(),
        (float)beta2,
        (float)eps0,
        rows,
        cols
    );
    kReduceVectorPartial<<<row_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)exp_avg_sq_row.data_ptr(),
        (float*)reduce_partial.data_ptr(),
        rows
    );
    kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
        (const float*)reduce_partial.data_ptr(),
        (float*)sum_row_state.data_ptr(),
        row_blocks
    );
    kUpdateSqColFinalizeFp32<<<col_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (float*)exp_avg_sq_col.data_ptr(),
        (float*)c_factor.data_ptr(),
        (float)beta2,
        (float)eps0,
        rows,
        cols
    );
    kFinalizeRFactorFp32<<<row_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)exp_avg_sq_row.data_ptr(),
        (float*)r_factor.data_ptr(),
        (const float*)sum_row_state.data_ptr(),
        rows
    );
    kSumUpdateSq<float><<<reduce_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (const float*)r_factor.data_ptr(),
        (const float*)c_factor.data_ptr(),
        (float*)reduce_partial.data_ptr(),
        (int)numel,
        rows,
        cols
    );
    kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
        (const float*)reduce_partial.data_ptr(),
        (float*)sum_update.data_ptr(),
        reduce_blocks
    );
    kCameFactoredExpAvgResPrepareFp32<<<update_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (float*)exp_avg.data_ptr(),
        (const float*)r_factor.data_ptr(),
        (const float*)c_factor.data_ptr(),
        (float*)scratch.data_ptr(),
        (const float*)sum_update.data_ptr(),
        (float)beta1,
        (float)eps1,
        (float)clip_threshold,
        rows,
        cols,
        (int)block_size
    );
    kUpdateMeanRowFp32<<<rows, THREADS, 0, stream.stream()>>>(
        (const float*)scratch.data_ptr(),
        (float*)exp_avg_res_row.data_ptr(),
        (float)beta3,
        rows,
        cols
    );
    kReduceVectorPartial<<<row_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)exp_avg_res_row.data_ptr(),
        (float*)reduce_partial.data_ptr(),
        rows
    );
    kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
        (const float*)reduce_partial.data_ptr(),
        (float*)sum_row_state.data_ptr(),
        row_blocks
    );
    kUpdateMeanColFinalizeFp32<<<col_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)scratch.data_ptr(),
        (float*)exp_avg_res_col.data_ptr(),
        (float*)c_factor.data_ptr(),
        (float)beta3,
        rows,
        cols
    );
    kFinalizeRFactorFp32<<<row_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)exp_avg_res_row.data_ptr(),
        (float*)r_factor.data_ptr(),
        (const float*)sum_row_state.data_ptr(),
        rows
    );

    const auto dtype = p.scalar_type();
    const int param_blocks = (int)((numel + THREADS - 1) / THREADS);
    if (dtype == torch::kFloat32) {
        kCameFactoredParamUpdate<float><<<param_blocks, THREADS, 0, stream.stream()>>>(
            (float*)p.data_ptr(),
            (const float*)exp_avg.data_ptr(),
            (const float*)r_factor.data_ptr(),
            (const float*)c_factor.data_ptr(),
            (float)lr,
            (float)weight_decay,
            rows,
            cols
        );
    } else if (dtype == torch::kFloat16) {
        kCameFactoredParamUpdate<half><<<param_blocks, THREADS, 0, stream.stream()>>>(
            (half*)p.data_ptr(),
            (const float*)exp_avg.data_ptr(),
            (const float*)r_factor.data_ptr(),
            (const float*)c_factor.data_ptr(),
            (float)lr,
            (float)weight_decay,
            rows,
            cols
        );
    } else {
        kCameFactoredParamUpdate<__nv_bfloat16><<<param_blocks, THREADS, 0, stream.stream()>>>(
            (__nv_bfloat16*)p.data_ptr(),
            (const float*)exp_avg.data_ptr(),
            (const float*)r_factor.data_ptr(),
            (const float*)c_factor.data_ptr(),
            (float)lr,
            (float)weight_decay,
            rows,
            cols
        );
    }
}

void came_full_factored_param_update_batched_cuda(
    torch::Tensor p,
    torch::Tensor exp_avg_fp32,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    double lr,
    double weight_decay
) {
    const c10::cuda::CUDAGuard device_guard(p.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int batch = (int)p.size(0);
    const int rows = (int)p.size(1);
    const int cols = (int)p.size(2);
    const int64_t per_item_numel = (int64_t)rows * cols;
    const int64_t total_numel = (int64_t)batch * per_item_numel;
    const int blocks = (int)((total_numel + THREADS - 1) / THREADS);
    const auto dtype = p.scalar_type();
    if (dtype == torch::kFloat32) {
        kCameFactoredParamUpdateBatched<float><<<blocks, THREADS, 0, stream.stream()>>>(
            (float*)p.data_ptr(),
            (const float*)exp_avg_fp32.data_ptr(),
            (const float*)r_factor.data_ptr(),
            (const float*)c_factor.data_ptr(),
            (float)lr,
            (float)weight_decay,
            batch,
            per_item_numel,
            rows,
            cols
        );
    } else if (dtype == torch::kFloat16) {
        kCameFactoredParamUpdateBatched<half><<<blocks, THREADS, 0, stream.stream()>>>(
            (half*)p.data_ptr(),
            (const float*)exp_avg_fp32.data_ptr(),
            (const float*)r_factor.data_ptr(),
            (const float*)c_factor.data_ptr(),
            (float)lr,
            (float)weight_decay,
            batch,
            per_item_numel,
            rows,
            cols
        );
    } else {
        kCameFactoredParamUpdateBatched<__nv_bfloat16><<<blocks, THREADS, 0, stream.stream()>>>(
            (__nv_bfloat16*)p.data_ptr(),
            (const float*)exp_avg_fp32.data_ptr(),
            (const float*)r_factor.data_ptr(),
            (const float*)c_factor.data_ptr(),
            (float)lr,
            (float)weight_decay,
            batch,
            per_item_numel,
            rows,
            cols
        );
    }
}

void came_full_factored_step_multitensor_same_shape_cuda(
    torch::Tensor g32,
    torch::Tensor p_ptrs,
    torch::Tensor exp_avg_q_ptrs,
    torch::Tensor exp_avg_absmax_ptrs,
    torch::Tensor exp_avg_sq_row_q_ptrs,
    torch::Tensor exp_avg_sq_row_absmax_ptrs,
    torch::Tensor exp_avg_sq_col_q_ptrs,
    torch::Tensor exp_avg_sq_col_absmax_ptrs,
    torch::Tensor exp_avg_res_row_q_ptrs,
    torch::Tensor exp_avg_res_row_absmax_ptrs,
    torch::Tensor exp_avg_res_col_q_ptrs,
    torch::Tensor exp_avg_res_col_absmax_ptrs,
    torch::Tensor row_factor,
    torch::Tensor c_factor,
    torch::Tensor row_absmax_scratch,
    torch::Tensor reduce_partial,
    torch::Tensor sum_row_state,
    torch::Tensor sum_update_slice,
    torch::Tensor sum_update_total,
    torch::Tensor sum_update_equiv,
    torch::Tensor exp_avg_fp32,
    torch::Tensor res32,
    torch::ScalarType param_dtype,
    double beta1,
    double beta2,
    double beta3,
    double eps0,
    double eps1,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    const c10::cuda::CUDAGuard device_guard(g32.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    const int batch = (int)g32.size(0);
    const int rows = (int)g32.size(1);
    const int cols = (int)g32.size(2);
    const int64_t per_item_numel = (int64_t)rows * cols;
    const int row_blocks = (rows + THREADS - 1) / THREADS;
    const int col_q_blocks = (cols + (int)block_size - 1) / (int)block_size;
    const int row_q_blocks = (rows + (int)block_size - 1) / (int)block_size;
    const int update_blocks = (int)((per_item_numel + THREADS - 1) / THREADS);
    const int total_update_blocks = batch * update_blocks;

    cudaMemsetAsync(row_absmax_scratch.data_ptr(), 0, row_absmax_scratch.numel() * sizeof(float), stream.stream());
    kUpdateSqRowQuantizedMultiTensorSameShape<float><<<batch * rows, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (const int64_t*)exp_avg_sq_row_q_ptrs.data_ptr(),
        (const int64_t*)exp_avg_sq_row_absmax_ptrs.data_ptr(),
        (float*)row_factor.data_ptr(),
        (float*)row_absmax_scratch.data_ptr(),
        (float)beta2,
        (float)eps0,
        per_item_numel,
        rows,
        cols,
        row_q_blocks,
        (int)block_size
    );
    kReduceVectorPartialBatched<<<batch * row_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)row_factor.data_ptr(),
        (float*)reduce_partial.data_ptr(),
        rows,
        row_blocks
    );
    kReducePartialSumBatched<<<batch, THREADS, 0, stream.stream()>>>(
        (const float*)reduce_partial.data_ptr(),
        (float*)sum_row_state.data_ptr(),
        row_blocks
    );
    kUpdateSqColQuantizedFinalizeMultiTensorSameShape<float><<<batch * col_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (const int64_t*)exp_avg_sq_col_q_ptrs.data_ptr(),
        (const int64_t*)exp_avg_sq_col_absmax_ptrs.data_ptr(),
        (float*)c_factor.data_ptr(),
        (float)beta2,
        (float)eps0,
        per_item_numel,
        rows,
        cols,
        col_q_blocks,
        (int)block_size
    );
    kFinalizeQuantizedRFactorMultiTensorSameShape<<<batch * row_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)row_factor.data_ptr(),
        (const int64_t*)exp_avg_sq_row_q_ptrs.data_ptr(),
        (const int64_t*)exp_avg_sq_row_absmax_ptrs.data_ptr(),
        (const float*)row_absmax_scratch.data_ptr(),
        (float*)row_factor.data_ptr(),
        (const float*)sum_row_state.data_ptr(),
        rows,
        row_q_blocks,
        (int)block_size
    );
    kSumUpdateSqBatched<float><<<total_update_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (const float*)row_factor.data_ptr(),
        (const float*)c_factor.data_ptr(),
        (float*)reduce_partial.data_ptr(),
        per_item_numel,
        rows,
        cols,
        update_blocks
    );
    kReducePartialSumBatched<<<batch, THREADS, 0, stream.stream()>>>(
        (const float*)reduce_partial.data_ptr(),
        (float*)sum_update_slice.data_ptr(),
        update_blocks
    );
    kReducePartialSum<<<1, THREADS, 0, stream.stream()>>>(
        (const float*)sum_update_slice.data_ptr(),
        (float*)sum_update_total.data_ptr(),
        batch
    );
    kStoreScaledScalar<<<1, 1, 0, stream.stream()>>>(
        (const float*)sum_update_total.data_ptr(),
        (float*)sum_update_equiv.data_ptr(),
        (float)batch
    );

    kCameFactoredExpAvgResPrepareMultiTensorSameShape<float, float><<<batch * ((per_item_numel + block_size - 1) / block_size), THREADS, 0, stream.stream()>>>(
        (const float*)g32.data_ptr(),
        (const int64_t*)exp_avg_q_ptrs.data_ptr(),
        (const int64_t*)exp_avg_absmax_ptrs.data_ptr(),
        (const float*)row_factor.data_ptr(),
        (const float*)c_factor.data_ptr(),
        (float*)exp_avg_fp32.data_ptr(),
        (float*)res32.data_ptr(),
        (const float*)sum_update_equiv.data_ptr(),
        (float)beta1,
        (float)eps1,
        (float)clip_threshold,
        per_item_numel,
        rows,
        cols,
        (int)((per_item_numel + block_size - 1) / block_size),
        (int)block_size
    );

    cudaMemsetAsync(row_absmax_scratch.data_ptr(), 0, row_absmax_scratch.numel() * sizeof(float), stream.stream());
    kUpdateMeanRowQuantizedMultiTensorSameShape<float><<<batch * rows, THREADS, 0, stream.stream()>>>(
        (const float*)res32.data_ptr(),
        (const int64_t*)exp_avg_res_row_q_ptrs.data_ptr(),
        (const int64_t*)exp_avg_res_row_absmax_ptrs.data_ptr(),
        (float*)row_factor.data_ptr(),
        (float*)row_absmax_scratch.data_ptr(),
        (float)beta3,
        per_item_numel,
        rows,
        cols,
        row_q_blocks,
        (int)block_size
    );
    kReduceVectorPartialBatched<<<batch * row_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)row_factor.data_ptr(),
        (float*)reduce_partial.data_ptr(),
        rows,
        row_blocks
    );
    kReducePartialSumBatched<<<batch, THREADS, 0, stream.stream()>>>(
        (const float*)reduce_partial.data_ptr(),
        (float*)sum_row_state.data_ptr(),
        row_blocks
    );
    kUpdateMeanColQuantizedFinalizeMultiTensorSameShape<float><<<batch * col_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)res32.data_ptr(),
        (const int64_t*)exp_avg_res_col_q_ptrs.data_ptr(),
        (const int64_t*)exp_avg_res_col_absmax_ptrs.data_ptr(),
        (float*)c_factor.data_ptr(),
        (float)beta3,
        per_item_numel,
        rows,
        cols,
        col_q_blocks,
        (int)block_size
    );
    kFinalizeQuantizedRFactorMultiTensorSameShape<<<batch * row_q_blocks, THREADS, 0, stream.stream()>>>(
        (const float*)row_factor.data_ptr(),
        (const int64_t*)exp_avg_res_row_q_ptrs.data_ptr(),
        (const int64_t*)exp_avg_res_row_absmax_ptrs.data_ptr(),
        (const float*)row_absmax_scratch.data_ptr(),
        (float*)row_factor.data_ptr(),
        (const float*)sum_row_state.data_ptr(),
        rows,
        row_q_blocks,
        (int)block_size
    );

    const int param_blocks = (int)(((int64_t)batch * per_item_numel + THREADS - 1) / THREADS);
    if (param_dtype == torch::kFloat32) {
        kCameFactoredParamUpdateMultiTensorSameShape<float, float><<<param_blocks, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const float*)exp_avg_fp32.data_ptr(),
            (const float*)row_factor.data_ptr(),
            (const float*)c_factor.data_ptr(),
            (float)lr,
            (float)weight_decay,
            batch,
            per_item_numel,
            rows,
            cols
        );
    } else if (param_dtype == torch::kFloat16) {
        kCameFactoredParamUpdateMultiTensorSameShape<half, float><<<param_blocks, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const float*)exp_avg_fp32.data_ptr(),
            (const float*)row_factor.data_ptr(),
            (const float*)c_factor.data_ptr(),
            (float)lr,
            (float)weight_decay,
            batch,
            per_item_numel,
            rows,
            cols
        );
    } else {
        kCameFactoredParamUpdateMultiTensorSameShape<__nv_bfloat16, float><<<param_blocks, THREADS, 0, stream.stream()>>>(
            (const int64_t*)p_ptrs.data_ptr(),
            (const float*)exp_avg_fp32.data_ptr(),
            (const float*)row_factor.data_ptr(),
            (const float*)c_factor.data_ptr(),
            (float)lr,
            (float)weight_decay,
            batch,
            per_item_numel,
            rows,
            cols
        );
    }
}

void came_full_factored_nd_chunked_step_cuda(
    torch::Tensor p,
    torch::Tensor g32,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_absmax,
    torch::Tensor exp_avg_sq_row_q,
    torch::Tensor exp_avg_sq_row_absmax,
    torch::Tensor exp_avg_sq_col_q,
    torch::Tensor exp_avg_sq_col_absmax,
    torch::Tensor exp_avg_res_row_q,
    torch::Tensor exp_avg_res_row_absmax,
    torch::Tensor exp_avg_res_col_q,
    torch::Tensor exp_avg_res_col_absmax,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    torch::Tensor row_absmax_scratch,
    torch::Tensor reduce_partial,
    torch::Tensor sum_row_state,
    torch::Tensor sum_update_slice,
    torch::Tensor sum_update_chunk,
    torch::Tensor sum_update_total,
    torch::Tensor sum_update_equiv,
    torch::Tensor exp_avg_fp32,
    torch::Tensor res32,
    double beta1,
    double beta2,
    double beta3,
    double eps0,
    double eps1,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t chunk_size,
    int64_t block_size,
    bool direct_row_sum
) {
    const c10::cuda::CUDAGuard device_guard(p.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    const int batch = (int)p.size(0);
    const int rows = (int)p.size(1);
    const int cols = (int)p.size(2);
    const int64_t per_item_numel = (int64_t)rows * cols;
    const int blocks_per_item = (int)((per_item_numel + block_size - 1) / block_size);
    const int row_blocks = (rows + THREADS - 1) / THREADS;
    const int row_q_blocks = (rows + (int)block_size - 1) / (int)block_size;
    const int col_q_blocks = (cols + (int)block_size - 1) / (int)block_size;
    const int update_blocks = (int)((per_item_numel + THREADS - 1) / THREADS);

    auto* g32_ptr = (const float*)g32.data_ptr();
    auto* exp_avg_q_ptr = (int8_t*)exp_avg_q.data_ptr();
    auto* exp_avg_absmax_ptr = (float*)exp_avg_absmax.data_ptr();
    auto* exp_avg_sq_row_q_ptr = (uint8_t*)exp_avg_sq_row_q.data_ptr();
    auto* exp_avg_sq_row_absmax_ptr = (float*)exp_avg_sq_row_absmax.data_ptr();
    auto* exp_avg_sq_col_q_ptr = (uint8_t*)exp_avg_sq_col_q.data_ptr();
    auto* exp_avg_sq_col_absmax_ptr = (float*)exp_avg_sq_col_absmax.data_ptr();
    auto* exp_avg_res_row_q_ptr = (uint8_t*)exp_avg_res_row_q.data_ptr();
    auto* exp_avg_res_row_absmax_ptr = (float*)exp_avg_res_row_absmax.data_ptr();
    auto* exp_avg_res_col_q_ptr = (uint8_t*)exp_avg_res_col_q.data_ptr();
    auto* exp_avg_res_col_absmax_ptr = (float*)exp_avg_res_col_absmax.data_ptr();
    auto* r_factor_ptr = (float*)r_factor.data_ptr();
    auto* c_factor_ptr = (float*)c_factor.data_ptr();
    auto* row_absmax_scratch_ptr = (float*)row_absmax_scratch.data_ptr();
    auto* reduce_partial_ptr = (float*)reduce_partial.data_ptr();
    auto* sum_row_state_ptr = (float*)sum_row_state.data_ptr();
    auto* sum_update_slice_ptr = (float*)sum_update_slice.data_ptr();
    auto* sum_update_chunk_ptr = (float*)sum_update_chunk.data_ptr();
    auto* sum_update_total_ptr = (float*)sum_update_total.data_ptr();
    auto* sum_update_equiv_ptr = (float*)sum_update_equiv.data_ptr();
    auto* exp_avg_fp32_ptr = (float*)exp_avg_fp32.data_ptr();
    auto* exp_avg_fp16_ptr = (half*)exp_avg_fp32.data_ptr();
    auto* res32_ptr = (float*)res32.data_ptr();

    cudaMemsetAsync(sum_update_total_ptr, 0, sizeof(float), stream.stream());
    for (int start = 0; start < batch; start += (int)chunk_size) {
        const int active = std::min<int>(batch - start, (int)chunk_size);
        const int64_t matrix_offset = (int64_t)start * per_item_numel;
        const int64_t row_offset = (int64_t)start * rows;
        const int64_t col_offset = (int64_t)start * cols;
        const int64_t row_absmax_offset = (int64_t)start * row_q_blocks;
        const int64_t col_absmax_offset = (int64_t)start * col_q_blocks;

        cudaMemsetAsync(row_absmax_scratch_ptr, 0, active * row_q_blocks * sizeof(float), stream.stream());
        if (direct_row_sum) {
            cudaMemsetAsync(sum_row_state_ptr, 0, active * sizeof(float), stream.stream());
            kUpdateSqRowQuantizedBatchedAccum<float><<<active * rows, THREADS, 0, stream.stream()>>>(
                g32_ptr + matrix_offset,
                exp_avg_sq_row_q_ptr + row_offset,
                exp_avg_sq_row_absmax_ptr + row_absmax_offset,
                r_factor_ptr + row_offset,
                row_absmax_scratch_ptr,
                sum_row_state_ptr,
                (float)beta2,
                (float)eps0,
                per_item_numel,
                rows,
                cols,
                row_q_blocks,
                (int)block_size
            );
        } else {
            kUpdateSqRowQuantizedBatched<float><<<active * rows, THREADS, 0, stream.stream()>>>(
                g32_ptr + matrix_offset,
                exp_avg_sq_row_q_ptr + row_offset,
                exp_avg_sq_row_absmax_ptr + row_absmax_offset,
                r_factor_ptr + row_offset,
                row_absmax_scratch_ptr,
                (float)beta2,
                (float)eps0,
                per_item_numel,
                rows,
                cols,
                row_q_blocks,
                (int)block_size
            );
            kReduceVectorPartialBatched<<<active * row_blocks, THREADS, 0, stream.stream()>>>(
                r_factor_ptr + row_offset,
                reduce_partial_ptr,
                rows,
                row_blocks
            );
            kReducePartialSumBatched<<<active, THREADS, 0, stream.stream()>>>(
                reduce_partial_ptr,
                sum_row_state_ptr,
                row_blocks
            );
        }
        kUpdateSqColQuantizedFinalizeBatched<float><<<active * col_q_blocks, THREADS, 0, stream.stream()>>>(
            g32_ptr + matrix_offset,
            exp_avg_sq_col_q_ptr + col_offset,
            exp_avg_sq_col_absmax_ptr + col_absmax_offset,
            c_factor_ptr + col_offset,
            (float)beta2,
            (float)eps0,
            per_item_numel,
            rows,
            cols,
            col_q_blocks,
            (int)block_size
        );
        kFinalizeQuantizedRFactorBatched<<<active * row_q_blocks, THREADS, 0, stream.stream()>>>(
            r_factor_ptr + row_offset,
            exp_avg_sq_row_q_ptr + row_offset,
            exp_avg_sq_row_absmax_ptr + row_absmax_offset,
            row_absmax_scratch_ptr,
            r_factor_ptr + row_offset,
            sum_row_state_ptr,
            rows,
            row_q_blocks,
            (int)block_size
        );
        kSumUpdateSqBatched<float><<<active * update_blocks, THREADS, 0, stream.stream()>>>(
            g32_ptr + matrix_offset,
            r_factor_ptr + row_offset,
            c_factor_ptr + col_offset,
            reduce_partial_ptr,
            per_item_numel,
            rows,
            cols,
            update_blocks
        );
        kReducePartialSumBatched<<<active, THREADS, 0, stream.stream()>>>(
            reduce_partial_ptr,
            sum_update_slice_ptr,
            update_blocks
        );
        kReducePartialSumAccumulate<<<1, THREADS, 0, stream.stream()>>>(
            sum_update_slice_ptr,
            sum_update_total_ptr,
            active
        );
    }

    kStoreScaledScalar<<<1, 1, 0, stream.stream()>>>(
        sum_update_total_ptr,
        sum_update_equiv_ptr,
        (float)batch
    );

    for (int start = 0; start < batch; start += (int)chunk_size) {
        const int active = std::min<int>(batch - start, (int)chunk_size);
        const int64_t matrix_offset = (int64_t)start * per_item_numel;
        const int64_t row_offset = (int64_t)start * rows;
        const int64_t col_offset = (int64_t)start * cols;
        const int64_t absmax_offset = (int64_t)start * blocks_per_item;
        const int64_t row_absmax_offset = (int64_t)start * row_q_blocks;
        const int64_t col_absmax_offset = (int64_t)start * col_q_blocks;
        const int total_blocks = active * blocks_per_item;

        cudaMemsetAsync(row_absmax_scratch_ptr, 0, active * row_q_blocks * sizeof(float), stream.stream());
        if (direct_row_sum) {
            cudaMemsetAsync(sum_row_state_ptr, 0, active * sizeof(float), stream.stream());
        }
        if (res32.scalar_type() == torch::kFloat16) {
            auto* res16_ptr = (half*)res32.data_ptr();
            if (exp_avg_fp32.scalar_type() == torch::kFloat16) {
                kCameFactoredExpAvgResPrepareBatched<half, half><<<total_blocks, THREADS, 0, stream.stream()>>>(
                    g32_ptr + matrix_offset,
                    exp_avg_q_ptr + matrix_offset,
                    exp_avg_absmax_ptr + absmax_offset,
                    r_factor_ptr + row_offset,
                    c_factor_ptr + col_offset,
                    exp_avg_fp16_ptr,
                    res16_ptr,
                    sum_update_equiv_ptr,
                    (float)beta1,
                    (float)eps1,
                    (float)clip_threshold,
                    per_item_numel,
                    rows,
                    cols,
                    blocks_per_item,
                    (int)block_size
                );
            } else {
                kCameFactoredExpAvgResPrepareBatched<float, half><<<total_blocks, THREADS, 0, stream.stream()>>>(
                    g32_ptr + matrix_offset,
                    exp_avg_q_ptr + matrix_offset,
                    exp_avg_absmax_ptr + absmax_offset,
                    r_factor_ptr + row_offset,
                    c_factor_ptr + col_offset,
                    exp_avg_fp32_ptr,
                    res16_ptr,
                    sum_update_equiv_ptr,
                    (float)beta1,
                    (float)eps1,
                    (float)clip_threshold,
                    per_item_numel,
                    rows,
                    cols,
                    blocks_per_item,
                    (int)block_size
                );
            }
            if (direct_row_sum) {
                kUpdateMeanRowQuantizedBatchedAccum<half><<<active * rows, THREADS, 0, stream.stream()>>>(
                    res16_ptr,
                    exp_avg_res_row_q_ptr + row_offset,
                    exp_avg_res_row_absmax_ptr + row_absmax_offset,
                    r_factor_ptr + row_offset,
                    row_absmax_scratch_ptr,
                    sum_row_state_ptr,
                    (float)beta3,
                    per_item_numel,
                    rows,
                    cols,
                    row_q_blocks,
                    (int)block_size
                );
            } else {
                kUpdateMeanRowQuantizedBatched<half><<<active * rows, THREADS, 0, stream.stream()>>>(
                    res16_ptr,
                    exp_avg_res_row_q_ptr + row_offset,
                    exp_avg_res_row_absmax_ptr + row_absmax_offset,
                    r_factor_ptr + row_offset,
                    row_absmax_scratch_ptr,
                    (float)beta3,
                    per_item_numel,
                    rows,
                    cols,
                    row_q_blocks,
                    (int)block_size
                );
            }
        } else {
            if (exp_avg_fp32.scalar_type() == torch::kFloat16) {
                kCameFactoredExpAvgResPrepareBatched<half, float><<<total_blocks, THREADS, 0, stream.stream()>>>(
                    g32_ptr + matrix_offset,
                    exp_avg_q_ptr + matrix_offset,
                    exp_avg_absmax_ptr + absmax_offset,
                    r_factor_ptr + row_offset,
                    c_factor_ptr + col_offset,
                    exp_avg_fp16_ptr,
                    res32_ptr,
                    sum_update_equiv_ptr,
                    (float)beta1,
                    (float)eps1,
                    (float)clip_threshold,
                    per_item_numel,
                    rows,
                    cols,
                    blocks_per_item,
                    (int)block_size
                );
            } else {
                kCameFactoredExpAvgResPrepareBatched<float, float><<<total_blocks, THREADS, 0, stream.stream()>>>(
                    g32_ptr + matrix_offset,
                    exp_avg_q_ptr + matrix_offset,
                    exp_avg_absmax_ptr + absmax_offset,
                    r_factor_ptr + row_offset,
                    c_factor_ptr + col_offset,
                    exp_avg_fp32_ptr,
                    res32_ptr,
                    sum_update_equiv_ptr,
                    (float)beta1,
                    (float)eps1,
                    (float)clip_threshold,
                    per_item_numel,
                    rows,
                    cols,
                    blocks_per_item,
                    (int)block_size
                );
            }
            if (direct_row_sum) {
                kUpdateMeanRowQuantizedBatchedAccum<float><<<active * rows, THREADS, 0, stream.stream()>>>(
                    res32_ptr,
                    exp_avg_res_row_q_ptr + row_offset,
                    exp_avg_res_row_absmax_ptr + row_absmax_offset,
                    r_factor_ptr + row_offset,
                    row_absmax_scratch_ptr,
                    sum_row_state_ptr,
                    (float)beta3,
                    per_item_numel,
                    rows,
                    cols,
                    row_q_blocks,
                    (int)block_size
                );
            } else {
                kUpdateMeanRowQuantizedBatched<float><<<active * rows, THREADS, 0, stream.stream()>>>(
                    res32_ptr,
                    exp_avg_res_row_q_ptr + row_offset,
                    exp_avg_res_row_absmax_ptr + row_absmax_offset,
                    r_factor_ptr + row_offset,
                    row_absmax_scratch_ptr,
                    (float)beta3,
                    per_item_numel,
                    rows,
                    cols,
                    row_q_blocks,
                    (int)block_size
                );
            }
        }
        if (!direct_row_sum) {
            kReduceVectorPartialBatched<<<active * row_blocks, THREADS, 0, stream.stream()>>>(
                r_factor_ptr + row_offset,
                reduce_partial_ptr,
                rows,
                row_blocks
            );
            kReducePartialSumBatched<<<active, THREADS, 0, stream.stream()>>>(
                reduce_partial_ptr,
                sum_row_state_ptr,
                row_blocks
            );
        }
        if (res32.scalar_type() == torch::kFloat16) {
            auto* res16_ptr = (half*)res32.data_ptr();
            kUpdateMeanColQuantizedFinalizeBatched<half><<<active * col_q_blocks, THREADS, 0, stream.stream()>>>(
                res16_ptr,
                exp_avg_res_col_q_ptr + col_offset,
                exp_avg_res_col_absmax_ptr + col_absmax_offset,
                c_factor_ptr + col_offset,
                (float)beta3,
                per_item_numel,
                rows,
                cols,
                col_q_blocks,
                (int)block_size
            );
        } else {
            kUpdateMeanColQuantizedFinalizeBatched<float><<<active * col_q_blocks, THREADS, 0, stream.stream()>>>(
                res32_ptr,
                exp_avg_res_col_q_ptr + col_offset,
                exp_avg_res_col_absmax_ptr + col_absmax_offset,
                c_factor_ptr + col_offset,
                (float)beta3,
                per_item_numel,
                rows,
                cols,
                col_q_blocks,
                (int)block_size
            );
        }
        kFinalizeQuantizedRFactorBatched<<<active * row_q_blocks, THREADS, 0, stream.stream()>>>(
            r_factor_ptr + row_offset,
            exp_avg_res_row_q_ptr + row_offset,
            exp_avg_res_row_absmax_ptr + row_absmax_offset,
            row_absmax_scratch_ptr,
            r_factor_ptr + row_offset,
            sum_row_state_ptr,
            rows,
            row_q_blocks,
            (int)block_size
        );

        const int64_t total_numel = (int64_t)active * per_item_numel;
        const int blocks = (int)((total_numel + THREADS - 1) / THREADS);
        const auto dtype = p.scalar_type();
        if (dtype == torch::kFloat32) {
            if (exp_avg_fp32.scalar_type() == torch::kFloat16) {
                kCameFactoredParamUpdateBatched<float, half><<<blocks, THREADS, 0, stream.stream()>>>(
                    (float*)p.data_ptr() + matrix_offset,
                    exp_avg_fp16_ptr,
                    r_factor_ptr + row_offset,
                    c_factor_ptr + col_offset,
                    (float)lr,
                    (float)weight_decay,
                    active,
                    per_item_numel,
                    rows,
                    cols
                );
            } else {
                kCameFactoredParamUpdateBatched<float, float><<<blocks, THREADS, 0, stream.stream()>>>(
                    (float*)p.data_ptr() + matrix_offset,
                    exp_avg_fp32_ptr,
                    r_factor_ptr + row_offset,
                    c_factor_ptr + col_offset,
                    (float)lr,
                    (float)weight_decay,
                    active,
                    per_item_numel,
                    rows,
                    cols
                );
            }
        } else if (dtype == torch::kFloat16) {
            if (exp_avg_fp32.scalar_type() == torch::kFloat16) {
                kCameFactoredParamUpdateBatched<half, half><<<blocks, THREADS, 0, stream.stream()>>>(
                    (half*)p.data_ptr() + matrix_offset,
                    exp_avg_fp16_ptr,
                    r_factor_ptr + row_offset,
                    c_factor_ptr + col_offset,
                    (float)lr,
                    (float)weight_decay,
                    active,
                    per_item_numel,
                    rows,
                    cols
                );
            } else {
                kCameFactoredParamUpdateBatched<half, float><<<blocks, THREADS, 0, stream.stream()>>>(
                    (half*)p.data_ptr() + matrix_offset,
                    exp_avg_fp32_ptr,
                    r_factor_ptr + row_offset,
                    c_factor_ptr + col_offset,
                    (float)lr,
                    (float)weight_decay,
                    active,
                    per_item_numel,
                    rows,
                    cols
                );
            }
        } else {
            if (exp_avg_fp32.scalar_type() == torch::kFloat16) {
                kCameFactoredParamUpdateBatched<__nv_bfloat16, half><<<blocks, THREADS, 0, stream.stream()>>>(
                    (__nv_bfloat16*)p.data_ptr() + matrix_offset,
                    exp_avg_fp16_ptr,
                    r_factor_ptr + row_offset,
                    c_factor_ptr + col_offset,
                    (float)lr,
                    (float)weight_decay,
                    active,
                    per_item_numel,
                    rows,
                    cols
                );
            } else {
                kCameFactoredParamUpdateBatched<__nv_bfloat16, float><<<blocks, THREADS, 0, stream.stream()>>>(
                    (__nv_bfloat16*)p.data_ptr() + matrix_offset,
                    exp_avg_fp32_ptr,
                    r_factor_ptr + row_offset,
                    c_factor_ptr + col_offset,
                    (float)lr,
                    (float)weight_decay,
                    active,
                    per_item_numel,
                    rows,
                    cols
                );
            }
        }
    }
}

void came2d_step_cuda(
    torch::Tensor p,
    torch::Tensor g,
    torch::Tensor exp_avg_q,
    torch::Tensor exp_avg_absmax,
    torch::Tensor exp_avg_sq_row_q,
    torch::Tensor exp_avg_sq_row_absmax,
    torch::Tensor exp_avg_sq_col_q,
    torch::Tensor exp_avg_sq_col_absmax,
    torch::Tensor exp_avg_res_row_q,
    torch::Tensor exp_avg_res_row_absmax,
    torch::Tensor exp_avg_res_col_q,
    torch::Tensor exp_avg_res_col_absmax,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    torch::Tensor r_res_factor,
    torch::Tensor c_res_factor,
    torch::Tensor res_row_sum,
    torch::Tensor res_col_sum,
    torch::Tensor res_row_partial,
    torch::Tensor res_col_partial,
    torch::Tensor sum_sq_row,
    torch::Tensor sum_update,
    torch::Tensor sum_update_partial,
    torch::Tensor sum_res_row,
    double beta1,
    double beta2,
    double beta3,
    double eps0,
    double eps1,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    const auto dtype = p.scalar_type();
    TORCH_CHECK(dtype == torch::kFloat16 || dtype == torch::kBFloat16 || dtype == torch::kFloat32, "Unsupported dtype");

    const float fbeta1 = (float)beta1;
    const float fbeta2 = (float)beta2;
    const float fbeta3 = (float)beta3;
    const float feps0 = (float)eps0;
    const float feps1 = (float)eps1;
    const float flr = (float)lr;
    const float fclip = (float)clip_threshold;
    const float fwd = (float)weight_decay;

    if (dtype == torch::kFloat32) {
        launch<float>(
            p, g, exp_avg_q, exp_avg_absmax, exp_avg_sq_row_q, exp_avg_sq_row_absmax, exp_avg_sq_col_q,
            exp_avg_sq_col_absmax, exp_avg_res_row_q, exp_avg_res_row_absmax, exp_avg_res_col_q,
            exp_avg_res_col_absmax, r_factor, c_factor, r_res_factor, c_res_factor, res_row_sum, res_col_sum, res_row_partial, res_col_partial,
            sum_sq_row, sum_update, sum_update_partial, sum_res_row, fbeta1, fbeta2, fbeta3, feps0, feps1, flr, fclip, fwd,
            (int)block_size
        );
    } else if (dtype == torch::kFloat16) {
        launch<half>(
            p, g, exp_avg_q, exp_avg_absmax, exp_avg_sq_row_q, exp_avg_sq_row_absmax, exp_avg_sq_col_q,
            exp_avg_sq_col_absmax, exp_avg_res_row_q, exp_avg_res_row_absmax, exp_avg_res_col_q,
            exp_avg_res_col_absmax, r_factor, c_factor, r_res_factor, c_res_factor, res_row_sum, res_col_sum, res_row_partial, res_col_partial,
            sum_sq_row, sum_update, sum_update_partial, sum_res_row, fbeta1, fbeta2, fbeta3, feps0, feps1, flr, fclip, fwd,
            (int)block_size
        );
    } else {
        // torch::kBFloat16
        launch<__nv_bfloat16>(
            p, g, exp_avg_q, exp_avg_absmax, exp_avg_sq_row_q, exp_avg_sq_row_absmax, exp_avg_sq_col_q,
            exp_avg_sq_col_absmax, exp_avg_res_row_q, exp_avg_res_row_absmax, exp_avg_res_col_q,
            exp_avg_res_col_absmax, r_factor, c_factor, r_res_factor, c_res_factor, res_row_sum, res_col_sum, res_row_partial, res_col_partial,
            sum_sq_row, sum_update, sum_update_partial, sum_res_row, fbeta1, fbeta2, fbeta3, feps0, feps1, flr, fclip, fwd,
            (int)block_size
        );
    }
}
