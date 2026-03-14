#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <vector>

namespace py = pybind11;

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
);

void blockwise_quant_cuda(
    torch::Tensor src,
    torch::Tensor q_out,
    torch::Tensor absmax_out,
    int64_t block_size,
    bool signed_quant
);

void blockwise_dequant_cuda(
    torch::Tensor out,
    torch::Tensor q_in,
    torch::Tensor absmax,
    int64_t block_size,
    bool signed_quant
);

void blockwise_quant_batched_cuda(
    torch::Tensor src,
    torch::Tensor q_out,
    torch::Tensor absmax_out,
    int64_t block_size,
    bool signed_quant
);

void blockwise_dequant_batched_cuda(
    torch::Tensor out,
    torch::Tensor q_in,
    torch::Tensor absmax,
    int64_t block_size,
    bool signed_quant
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

void came_full_factored_param_update_cuda(
    torch::Tensor p,
    torch::Tensor exp_avg_fp32,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    double lr,
    double weight_decay
);

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
);

void came_full_factored_param_update_batched_cuda(
    torch::Tensor p,
    torch::Tensor exp_avg_fp32,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    double lr,
    double weight_decay
);

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
);

static std::vector<torch::Tensor> tensor_list_from_pylist(const py::list& tensors, const char* name) {
    std::vector<torch::Tensor> out;
    out.reserve((size_t)tensors.size());
    for (const auto& item : tensors) {
        try {
            out.push_back(item.cast<torch::Tensor>());
        } catch (const py::cast_error&) {
            TORCH_CHECK(false, name, " must contain only torch.Tensor values");
        }
    }
    return out;
}

static torch::Tensor make_device_ptr_tensor(
    const std::vector<torch::Tensor>& tensors,
    torch::Device device
) {
    auto ptr_tensor = torch::empty(
        {(int64_t)tensors.size()},
        torch::TensorOptions().device(device).dtype(torch::kInt64)
    );
    std::vector<int64_t> host_ptrs;
    host_ptrs.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        host_ptrs.push_back((int64_t)tensor.data_ptr());
    }
    auto stream = at::cuda::getCurrentCUDAStream(device.index());
    cudaMemcpyAsync(ptr_tensor.data_ptr(), host_ptrs.data(), host_ptrs.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream.stream());
    return ptr_tensor;
}

static void fill_device_ptr_tensor(
    const std::vector<torch::Tensor>& tensors,
    torch::Tensor ptr_tensor
) {
    TORCH_CHECK(ptr_tensor.is_cuda(), "ptr_tensor must be CUDA");
    TORCH_CHECK(ptr_tensor.scalar_type() == torch::kInt64, "ptr_tensor must be int64");
    TORCH_CHECK(ptr_tensor.dim() == 1, "ptr_tensor must be 1D");
    TORCH_CHECK(ptr_tensor.is_contiguous(), "ptr_tensor must be contiguous");
    TORCH_CHECK((int64_t)tensors.size() == ptr_tensor.numel(), "ptr_tensor length mismatch");

    std::vector<int64_t> host_ptrs;
    host_ptrs.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        host_ptrs.push_back((int64_t)tensor.data_ptr());
    }
    auto stream = at::cuda::getCurrentCUDAStream(ptr_tensor.device().index());
    cudaMemcpyAsync(ptr_tensor.data_ptr(), host_ptrs.data(), host_ptrs.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream.stream());
}

static void validate_multitensor_ptr_inputs(
    torch::Tensor sample_p,
    torch::Tensor p_ptrs,
    torch::Tensor g32_ptrs,
    torch::Tensor exp_avg_q_ptrs,
    torch::Tensor exp_avg_absmax_ptrs,
    torch::Tensor exp_avg_sq_q_ptrs,
    torch::Tensor exp_avg_sq_absmax_ptrs,
    torch::Tensor update_ptrs,
    torch::Tensor sum_update_ptrs,
    int64_t per_item_numel,
    int64_t block_size
) {
    TORCH_CHECK(sample_p.is_cuda(), "sample_p must be CUDA");
    TORCH_CHECK(sample_p.dim() == 1, "sample_p must be 1D");
    TORCH_CHECK(sample_p.is_contiguous(), "sample_p must be contiguous");
    TORCH_CHECK(per_item_numel > 0, "per_item_numel must be positive");
    TORCH_CHECK(sample_p.numel() == per_item_numel, "sample_p.numel() must match per_item_numel");
    TORCH_CHECK(
        sample_p.scalar_type() == torch::kFloat32 || sample_p.scalar_type() == torch::kFloat16 || sample_p.scalar_type() == torch::kBFloat16,
        "sample_p must be float32/float16/bfloat16"
    );
    const auto device = sample_p.device();
    const int64_t batch = p_ptrs.numel();
    const auto check_ptr_tensor = [&](torch::Tensor tensor, const char* name) {
        TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
        TORCH_CHECK(tensor.device() == device, name, " device mismatch");
        TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must be int64");
        TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(tensor.numel() == batch, name, " batch size mismatch");
    };
    check_ptr_tensor(p_ptrs, "p_ptrs");
    check_ptr_tensor(g32_ptrs, "g32_ptrs");
    check_ptr_tensor(exp_avg_q_ptrs, "exp_avg_q_ptrs");
    check_ptr_tensor(exp_avg_absmax_ptrs, "exp_avg_absmax_ptrs");
    check_ptr_tensor(exp_avg_sq_q_ptrs, "exp_avg_sq_q_ptrs");
    check_ptr_tensor(exp_avg_sq_absmax_ptrs, "exp_avg_sq_absmax_ptrs");
    check_ptr_tensor(update_ptrs, "update_ptrs");
    check_ptr_tensor(sum_update_ptrs, "sum_update_ptrs");
    TORCH_CHECK(batch > 0, "multitensor ptr tensors must not be empty");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
}

static void validate_multitensor_varlen_ptr_inputs(
    torch::Tensor sample_p,
    torch::Tensor p_ptrs,
    torch::Tensor g32_ptrs,
    torch::Tensor exp_avg_q_ptrs,
    torch::Tensor exp_avg_absmax_ptrs,
    torch::Tensor exp_avg_sq_q_ptrs,
    torch::Tensor exp_avg_sq_absmax_ptrs,
    torch::Tensor update_ptrs,
    torch::Tensor sum_update_ptrs,
    torch::Tensor item_numels,
    int64_t max_item_numel,
    int64_t block_size
) {
    TORCH_CHECK(sample_p.is_cuda(), "sample_p must be CUDA");
    TORCH_CHECK(sample_p.dim() == 1, "sample_p must be 1D");
    TORCH_CHECK(sample_p.is_contiguous(), "sample_p must be contiguous");
    TORCH_CHECK(
        sample_p.scalar_type() == torch::kFloat32 || sample_p.scalar_type() == torch::kFloat16 || sample_p.scalar_type() == torch::kBFloat16,
        "sample_p must be float32/float16/bfloat16"
    );
    const auto device = sample_p.device();
    const int64_t batch = p_ptrs.numel();
    const auto check_ptr_tensor = [&](torch::Tensor tensor, const char* name) {
        TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
        TORCH_CHECK(tensor.device() == device, name, " device mismatch");
        TORCH_CHECK(tensor.scalar_type() == torch::kInt64, name, " must be int64");
        TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
        TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(tensor.numel() == batch, name, " batch size mismatch");
    };
    check_ptr_tensor(p_ptrs, "p_ptrs");
    check_ptr_tensor(g32_ptrs, "g32_ptrs");
    check_ptr_tensor(exp_avg_q_ptrs, "exp_avg_q_ptrs");
    check_ptr_tensor(exp_avg_absmax_ptrs, "exp_avg_absmax_ptrs");
    check_ptr_tensor(exp_avg_sq_q_ptrs, "exp_avg_sq_q_ptrs");
    check_ptr_tensor(exp_avg_sq_absmax_ptrs, "exp_avg_sq_absmax_ptrs");
    check_ptr_tensor(update_ptrs, "update_ptrs");
    check_ptr_tensor(sum_update_ptrs, "sum_update_ptrs");
    TORCH_CHECK(batch > 0, "multitensor ptr tensors must not be empty");
    TORCH_CHECK(item_numels.is_cuda(), "item_numels must be CUDA");
    TORCH_CHECK(item_numels.device() == device, "item_numels device mismatch");
    TORCH_CHECK(item_numels.scalar_type() == torch::kInt64, "item_numels must be int64");
    TORCH_CHECK(item_numels.dim() == 1, "item_numels must be 1D");
    TORCH_CHECK(item_numels.is_contiguous(), "item_numels must be contiguous");
    TORCH_CHECK(item_numels.numel() == batch, "item_numels batch size mismatch");
    TORCH_CHECK(max_item_numel > 0, "max_item_numel must be positive");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
}

static void validate_multitensor_compact_varlen_ptr_inputs(
    torch::Tensor sample_p,
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
    int64_t block_size
) {
    validate_multitensor_varlen_ptr_inputs(
        sample_p,
        p_ptrs,
        g32_ptrs,
        exp_avg_q_ptrs,
        exp_avg_absmax_ptrs,
        exp_avg_sq_q_ptrs,
        exp_avg_sq_absmax_ptrs,
        update_ptrs,
        sum_update_ptrs,
        item_numels,
        sample_p.numel(),
        block_size
    );
    TORCH_CHECK(block_item_ids.is_cuda(), "block_item_ids must be CUDA");
    TORCH_CHECK(block_item_ids.device() == sample_p.device(), "block_item_ids device mismatch");
    TORCH_CHECK(block_item_ids.scalar_type() == torch::kInt64, "block_item_ids must be int64");
    TORCH_CHECK(block_item_ids.dim() == 1, "block_item_ids must be 1D");
    TORCH_CHECK(block_item_ids.is_contiguous(), "block_item_ids must be contiguous");
    TORCH_CHECK(block_item_ids.numel() > 0, "block_item_ids must not be empty");
    TORCH_CHECK(block_item_starts.is_cuda(), "block_item_starts must be CUDA");
    TORCH_CHECK(block_item_starts.device() == sample_p.device(), "block_item_starts device mismatch");
    TORCH_CHECK(block_item_starts.scalar_type() == torch::kInt64, "block_item_starts must be int64");
    TORCH_CHECK(block_item_starts.dim() == 1, "block_item_starts must be 1D");
    TORCH_CHECK(block_item_starts.is_contiguous(), "block_item_starts must be contiguous");
    TORCH_CHECK(block_item_starts.numel() == block_item_ids.numel(), "block_item_starts length mismatch");
}

static void came2d_step(
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
    TORCH_CHECK(p.is_cuda(), "p must be CUDA");
    TORCH_CHECK(g.is_cuda(), "g must be CUDA");
    TORCH_CHECK(p.dim() == 2, "p must be 2D");
    TORCH_CHECK(g.sizes() == p.sizes(), "g.shape must match p.shape");
    TORCH_CHECK(g.scalar_type() == p.scalar_type(), "g.dtype must match p.dtype");
    TORCH_CHECK(p.is_contiguous(), "p must be contiguous");
    TORCH_CHECK(g.is_contiguous(), "g must be contiguous");

    TORCH_CHECK(exp_avg_q.is_cuda(), "exp_avg_q must be CUDA");
    TORCH_CHECK(exp_avg_q.scalar_type() == torch::kInt8, "exp_avg_q must be int8");
    TORCH_CHECK(exp_avg_q.sizes() == p.sizes(), "exp_avg_q.shape must match p.shape");

    TORCH_CHECK(exp_avg_absmax.is_cuda(), "exp_avg_absmax must be CUDA");
    TORCH_CHECK(exp_avg_absmax.scalar_type() == torch::kFloat32, "exp_avg_absmax must be float32");
    TORCH_CHECK(exp_avg_sq_row_q.is_cuda() && exp_avg_sq_row_q.scalar_type() == torch::kUInt8, "exp_avg_sq_row_q must be CUDA uint8");
    TORCH_CHECK(exp_avg_sq_col_q.is_cuda() && exp_avg_sq_col_q.scalar_type() == torch::kUInt8, "exp_avg_sq_col_q must be CUDA uint8");
    TORCH_CHECK(exp_avg_res_row_q.is_cuda() && exp_avg_res_row_q.scalar_type() == torch::kUInt8, "exp_avg_res_row_q must be CUDA uint8");
    TORCH_CHECK(exp_avg_res_col_q.is_cuda() && exp_avg_res_col_q.scalar_type() == torch::kUInt8, "exp_avg_res_col_q must be CUDA uint8");
    TORCH_CHECK(exp_avg_sq_row_absmax.is_cuda() && exp_avg_sq_row_absmax.scalar_type() == torch::kFloat32, "exp_avg_sq_row_absmax must be CUDA float32");
    TORCH_CHECK(exp_avg_sq_col_absmax.is_cuda() && exp_avg_sq_col_absmax.scalar_type() == torch::kFloat32, "exp_avg_sq_col_absmax must be CUDA float32");
    TORCH_CHECK(exp_avg_res_row_absmax.is_cuda() && exp_avg_res_row_absmax.scalar_type() == torch::kFloat32, "exp_avg_res_row_absmax must be CUDA float32");
    TORCH_CHECK(exp_avg_res_col_absmax.is_cuda() && exp_avg_res_col_absmax.scalar_type() == torch::kFloat32, "exp_avg_res_col_absmax must be CUDA float32");

    TORCH_CHECK(r_factor.is_cuda() && r_factor.scalar_type() == torch::kFloat32, "r_factor must be CUDA float32");
    TORCH_CHECK(c_factor.is_cuda() && c_factor.scalar_type() == torch::kFloat32, "c_factor must be CUDA float32");
    TORCH_CHECK(r_res_factor.is_cuda() && r_res_factor.scalar_type() == torch::kFloat32, "r_res_factor must be CUDA float32");
    TORCH_CHECK(c_res_factor.is_cuda() && c_res_factor.scalar_type() == torch::kFloat32, "c_res_factor must be CUDA float32");

    TORCH_CHECK(res_row_sum.is_cuda() && res_row_sum.scalar_type() == torch::kFloat32, "res_row_sum must be CUDA float32");
    TORCH_CHECK(res_col_sum.is_cuda() && res_col_sum.scalar_type() == torch::kFloat32, "res_col_sum must be CUDA float32");
    TORCH_CHECK(res_row_partial.is_cuda() && res_row_partial.scalar_type() == torch::kFloat32, "res_row_partial must be CUDA float32");
    TORCH_CHECK(res_col_partial.is_cuda() && res_col_partial.scalar_type() == torch::kFloat32, "res_col_partial must be CUDA float32");
    TORCH_CHECK(sum_sq_row.is_cuda() && sum_sq_row.scalar_type() == torch::kFloat32 && sum_sq_row.numel() == 1, "sum_sq_row must be CUDA float32 scalar");
    TORCH_CHECK(sum_update.is_cuda() && sum_update.scalar_type() == torch::kFloat32 && sum_update.numel() == 1, "sum_update must be CUDA float32 scalar");
    TORCH_CHECK(sum_update_partial.is_cuda() && sum_update_partial.scalar_type() == torch::kFloat32, "sum_update_partial must be CUDA float32");
    TORCH_CHECK(sum_res_row.is_cuda() && sum_res_row.scalar_type() == torch::kFloat32 && sum_res_row.numel() == 1, "sum_res_row must be CUDA float32 scalar");

    TORCH_CHECK(block_size == 256, "Only block_size=256 is supported in this minimal implementation.");

    const auto rows = p.size(0);
    const auto cols = p.size(1);
    TORCH_CHECK(exp_avg_sq_row_q.numel() == rows, "exp_avg_sq_row_q must have shape (rows,)");
    TORCH_CHECK(exp_avg_res_row_q.numel() == rows, "exp_avg_res_row_q must have shape (rows,)");
    TORCH_CHECK(res_row_sum.numel() == rows, "res_row_sum must have shape (rows,)");
    TORCH_CHECK(r_factor.numel() == rows, "r_factor must have shape (rows,)");
    TORCH_CHECK(r_res_factor.numel() == rows, "r_res_factor must have shape (rows,)");

    TORCH_CHECK(exp_avg_sq_col_q.numel() == cols, "exp_avg_sq_col_q must have shape (cols,)");
    TORCH_CHECK(exp_avg_res_col_q.numel() == cols, "exp_avg_res_col_q must have shape (cols,)");
    TORCH_CHECK(res_col_sum.numel() == cols, "res_col_sum must have shape (cols,)");
    TORCH_CHECK(c_factor.numel() == cols, "c_factor must have shape (cols,)");
    TORCH_CHECK(c_res_factor.numel() == cols, "c_res_factor must have shape (cols,)");

    const int64_t rows_i = rows;
    const int64_t cols_i = cols;
    TORCH_CHECK(sum_update_partial.numel() >= ((rows_i * cols_i) + 255) / 256, "sum_update_partial is too small");
    const int64_t tile_rows = (rows_i + 15) / 16;
    const int64_t tile_cols = (cols_i + 15) / 16;
    const int64_t num_tiles = tile_rows * tile_cols;
    TORCH_CHECK(res_row_partial.numel() >= tile_cols * rows_i, "res_row_partial is too small");
    TORCH_CHECK(res_col_partial.numel() >= tile_rows * cols_i, "res_col_partial is too small");
    TORCH_CHECK(exp_avg_absmax.numel() >= num_tiles, "exp_avg_absmax is too small for (rows, cols) tiling");
    TORCH_CHECK(exp_avg_sq_row_absmax.numel() >= (rows_i + block_size - 1) / block_size, "exp_avg_sq_row_absmax is too small");
    TORCH_CHECK(exp_avg_res_row_absmax.numel() >= (rows_i + block_size - 1) / block_size, "exp_avg_res_row_absmax is too small");
    TORCH_CHECK(exp_avg_sq_col_absmax.numel() >= (cols_i + block_size - 1) / block_size, "exp_avg_sq_col_absmax is too small");
    TORCH_CHECK(exp_avg_res_col_absmax.numel() >= (cols_i + block_size - 1) / block_size, "exp_avg_res_col_absmax is too small");

    came2d_step_cuda(
        p,
        g,
        exp_avg_q,
        exp_avg_absmax,
        exp_avg_sq_row_q,
        exp_avg_sq_row_absmax,
        exp_avg_sq_col_q,
        exp_avg_sq_col_absmax,
        exp_avg_res_row_q,
        exp_avg_res_row_absmax,
        exp_avg_res_col_q,
        exp_avg_res_col_absmax,
        r_factor,
        c_factor,
        r_res_factor,
        c_res_factor,
        res_row_sum,
        res_col_sum,
        res_row_partial,
        res_col_partial,
        sum_sq_row,
        sum_update,
        sum_update_partial,
        sum_res_row,
        beta1,
        beta2,
        beta3,
        eps0,
        eps1,
        lr,
        clip_threshold,
        weight_decay,
        block_size
    );
}

static void blockwise_quant(
    torch::Tensor src,
    torch::Tensor q_out,
    torch::Tensor absmax_out,
    int64_t block_size,
    bool signed_quant
) {
    TORCH_CHECK(src.is_cuda() && q_out.is_cuda() && absmax_out.is_cuda(), "blockwise_quant tensors must be CUDA");
    TORCH_CHECK(src.scalar_type() == torch::kFloat32, "src must be float32");
    TORCH_CHECK(src.is_contiguous() && q_out.is_contiguous() && absmax_out.is_contiguous(), "blockwise_quant tensors must be contiguous");
    TORCH_CHECK(src.numel() == q_out.numel(), "src and q_out must have the same numel");
    TORCH_CHECK(
        absmax_out.numel() >= (src.numel() + block_size - 1) / block_size,
        "absmax_out is too small for blockwise_quant"
    );
    if (signed_quant) {
        TORCH_CHECK(q_out.scalar_type() == torch::kInt8, "signed blockwise_quant requires int8 q_out");
    } else {
        TORCH_CHECK(q_out.scalar_type() == torch::kUInt8, "unsigned blockwise_quant requires uint8 q_out");
    }
    blockwise_quant_cuda(src, q_out, absmax_out, block_size, signed_quant);
}

static void blockwise_dequant(
    torch::Tensor out,
    torch::Tensor q_in,
    torch::Tensor absmax,
    int64_t block_size,
    bool signed_quant
) {
    TORCH_CHECK(out.is_cuda() && q_in.is_cuda() && absmax.is_cuda(), "blockwise_dequant tensors must be CUDA");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(out.is_contiguous() && q_in.is_contiguous() && absmax.is_contiguous(), "blockwise_dequant tensors must be contiguous");
    TORCH_CHECK(out.numel() == q_in.numel(), "out and q_in must have the same numel");
    TORCH_CHECK(
        absmax.numel() >= (q_in.numel() + block_size - 1) / block_size,
        "absmax is too small for blockwise_dequant"
    );
    if (signed_quant) {
        TORCH_CHECK(q_in.scalar_type() == torch::kInt8, "signed blockwise_dequant requires int8 q_in");
    } else {
        TORCH_CHECK(q_in.scalar_type() == torch::kUInt8, "unsigned blockwise_dequant requires uint8 q_in");
    }
    blockwise_dequant_cuda(out, q_in, absmax, block_size, signed_quant);
}

static void blockwise_quant_batched(
    torch::Tensor src,
    torch::Tensor q_out,
    torch::Tensor absmax_out,
    int64_t block_size,
    bool signed_quant
) {
    TORCH_CHECK(src.is_cuda() && q_out.is_cuda() && absmax_out.is_cuda(), "blockwise_quant_batched tensors must be CUDA");
    TORCH_CHECK(src.scalar_type() == torch::kFloat32, "src must be float32");
    TORCH_CHECK(src.is_contiguous() && q_out.is_contiguous() && absmax_out.is_contiguous(), "blockwise_quant_batched tensors must be contiguous");
    TORCH_CHECK(src.dim() >= 1 && q_out.dim() >= 1 && absmax_out.dim() == 2, "blockwise_quant_batched expects batched tensors");
    TORCH_CHECK(src.size(0) == q_out.size(0), "src and q_out batch size must match");
    TORCH_CHECK(src.size(0) == absmax_out.size(0), "src and absmax_out batch size must match");
    TORCH_CHECK(src.numel() == q_out.numel(), "src and q_out must have the same numel");
    const int64_t per_item_numel = src.numel() / src.size(0);
    TORCH_CHECK(
        absmax_out.size(1) >= (per_item_numel + block_size - 1) / block_size,
        "absmax_out inner dimension is too small for blockwise_quant_batched"
    );
    if (signed_quant) {
        TORCH_CHECK(q_out.scalar_type() == torch::kInt8, "signed blockwise_quant_batched requires int8 q_out");
    } else {
        TORCH_CHECK(q_out.scalar_type() == torch::kUInt8, "unsigned blockwise_quant_batched requires uint8 q_out");
    }
    blockwise_quant_batched_cuda(src, q_out, absmax_out, block_size, signed_quant);
}

static void blockwise_dequant_batched(
    torch::Tensor out,
    torch::Tensor q_in,
    torch::Tensor absmax,
    int64_t block_size,
    bool signed_quant
) {
    TORCH_CHECK(out.is_cuda() && q_in.is_cuda() && absmax.is_cuda(), "blockwise_dequant_batched tensors must be CUDA");
    TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
    TORCH_CHECK(out.is_contiguous() && q_in.is_contiguous() && absmax.is_contiguous(), "blockwise_dequant_batched tensors must be contiguous");
    TORCH_CHECK(out.dim() >= 1 && q_in.dim() >= 1 && absmax.dim() == 2, "blockwise_dequant_batched expects batched tensors");
    TORCH_CHECK(out.size(0) == q_in.size(0), "out and q_in batch size must match");
    TORCH_CHECK(out.size(0) == absmax.size(0), "out and absmax batch size must match");
    TORCH_CHECK(out.numel() == q_in.numel(), "out and q_in must have the same numel");
    const int64_t per_item_numel = q_in.numel() / q_in.size(0);
    TORCH_CHECK(
        absmax.size(1) >= (per_item_numel + block_size - 1) / block_size,
        "absmax inner dimension is too small for blockwise_dequant_batched"
    );
    if (signed_quant) {
        TORCH_CHECK(q_in.scalar_type() == torch::kInt8, "signed blockwise_dequant_batched requires int8 q_in");
    } else {
        TORCH_CHECK(q_in.scalar_type() == torch::kUInt8, "unsigned blockwise_dequant_batched requires uint8 q_in");
    }
    blockwise_dequant_batched_cuda(out, q_in, absmax, block_size, signed_quant);
}

static void came_full_nonfactored_step(
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
    TORCH_CHECK(p.is_cuda() && g32.is_cuda(), "came_full_nonfactored_step tensors must be CUDA");
    TORCH_CHECK(p.is_contiguous() && g32.is_contiguous(), "p and g32 must be contiguous");
    TORCH_CHECK(g32.scalar_type() == torch::kFloat32, "g32 must be float32");
    TORCH_CHECK(
        p.scalar_type() == torch::kFloat32 || p.scalar_type() == torch::kFloat16 || p.scalar_type() == torch::kBFloat16,
        "p must be float32/float16/bfloat16"
    );
    TORCH_CHECK(p.numel() == g32.numel(), "p and g32 must have same numel");
    TORCH_CHECK(exp_avg_q.is_cuda() && exp_avg_q.scalar_type() == torch::kInt8, "exp_avg_q must be CUDA int8");
    TORCH_CHECK(exp_avg_sq_q.is_cuda() && exp_avg_sq_q.scalar_type() == torch::kUInt8, "exp_avg_sq_q must be CUDA uint8");
    TORCH_CHECK(exp_avg_absmax.is_cuda() && exp_avg_absmax.scalar_type() == torch::kFloat32, "exp_avg_absmax must be CUDA float32");
    TORCH_CHECK(exp_avg_sq_absmax.is_cuda() && exp_avg_sq_absmax.scalar_type() == torch::kFloat32, "exp_avg_sq_absmax must be CUDA float32");
    TORCH_CHECK(update.is_cuda() && update.scalar_type() == torch::kFloat32 && update.is_contiguous(), "update must be CUDA float32 contiguous");
    TORCH_CHECK(sum_update_partial.is_cuda() && sum_update_partial.scalar_type() == torch::kFloat32, "sum_update_partial must be CUDA float32");
    TORCH_CHECK(sum_update.is_cuda() && sum_update.scalar_type() == torch::kFloat32 && sum_update.numel() == 1, "sum_update must be CUDA float32 scalar");
    TORCH_CHECK(exp_avg_q.numel() == p.numel(), "exp_avg_q must match p.numel()");
    TORCH_CHECK(exp_avg_sq_q.numel() == p.numel(), "exp_avg_sq_q must match p.numel()");
    TORCH_CHECK(update.numel() == p.numel(), "update must match p.numel()");
    TORCH_CHECK(exp_avg_absmax.numel() >= (p.numel() + block_size - 1) / block_size, "exp_avg_absmax too small");
    TORCH_CHECK(exp_avg_sq_absmax.numel() >= (p.numel() + block_size - 1) / block_size, "exp_avg_sq_absmax too small");
    TORCH_CHECK(sum_update_partial.numel() >= (p.numel() + block_size - 1) / block_size, "sum_update_partial too small");

    came_full_nonfactored_step_cuda(
        p,
        g32,
        exp_avg_q,
        exp_avg_absmax,
        exp_avg_sq_q,
        exp_avg_sq_absmax,
        update,
        sum_update_partial,
        sum_update,
        beta1,
        beta2,
        eps0,
        lr,
        clip_threshold,
        weight_decay,
        block_size
    );
}

static void came_full_nonfactored_step_fp16_update(
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
    TORCH_CHECK(p.is_cuda() && g32.is_cuda(), "came_full_nonfactored_step_fp16_update tensors must be CUDA");
    TORCH_CHECK(p.is_contiguous() && g32.is_contiguous(), "p and g32 must be contiguous");
    TORCH_CHECK(g32.scalar_type() == torch::kFloat32, "g32 must be float32");
    TORCH_CHECK(
        p.scalar_type() == torch::kFloat32 || p.scalar_type() == torch::kFloat16 || p.scalar_type() == torch::kBFloat16,
        "p must be float32/float16/bfloat16"
    );
    TORCH_CHECK(p.numel() == g32.numel(), "p and g32 must have same numel");
    TORCH_CHECK(exp_avg_q.is_cuda() && exp_avg_q.scalar_type() == torch::kInt8, "exp_avg_q must be CUDA int8");
    TORCH_CHECK(exp_avg_sq_q.is_cuda() && exp_avg_sq_q.scalar_type() == torch::kUInt8, "exp_avg_sq_q must be CUDA uint8");
    TORCH_CHECK(exp_avg_absmax.is_cuda() && exp_avg_absmax.scalar_type() == torch::kFloat32, "exp_avg_absmax must be CUDA float32");
    TORCH_CHECK(exp_avg_sq_absmax.is_cuda() && exp_avg_sq_absmax.scalar_type() == torch::kFloat32, "exp_avg_sq_absmax must be CUDA float32");
    TORCH_CHECK(update.is_cuda() && update.scalar_type() == torch::kFloat16 && update.is_contiguous(), "update must be CUDA float16 contiguous");
    TORCH_CHECK(sum_update_partial.is_cuda() && sum_update_partial.scalar_type() == torch::kFloat32, "sum_update_partial must be CUDA float32");
    TORCH_CHECK(sum_update.is_cuda() && sum_update.scalar_type() == torch::kFloat32 && sum_update.numel() == 1, "sum_update must be CUDA float32 scalar");
    TORCH_CHECK(exp_avg_q.numel() == p.numel(), "exp_avg_q must match p.numel()");
    TORCH_CHECK(exp_avg_sq_q.numel() == p.numel(), "exp_avg_sq_q must match p.numel()");
    TORCH_CHECK(update.numel() == p.numel(), "update must match p.numel()");
    TORCH_CHECK(exp_avg_absmax.numel() >= (p.numel() + block_size - 1) / block_size, "exp_avg_absmax too small");
    TORCH_CHECK(exp_avg_sq_absmax.numel() >= (p.numel() + block_size - 1) / block_size, "exp_avg_sq_absmax too small");
    TORCH_CHECK(sum_update_partial.numel() >= (p.numel() + block_size - 1) / block_size, "sum_update_partial too small");

    came_full_nonfactored_step_fp16_update_cuda(
        p,
        g32,
        exp_avg_q,
        exp_avg_absmax,
        exp_avg_sq_q,
        exp_avg_sq_absmax,
        update,
        sum_update_partial,
        sum_update,
        beta1,
        beta2,
        eps0,
        lr,
        clip_threshold,
        weight_decay,
        block_size
    );
}

static void came_full_nonfactored_step_batched(
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
    TORCH_CHECK(p.is_cuda() && g32.is_cuda(), "came_full_nonfactored_step_batched tensors must be CUDA");
    TORCH_CHECK(p.is_contiguous() && g32.is_contiguous(), "p and g32 must be contiguous");
    TORCH_CHECK(p.dim() == 2 && g32.dim() == 2, "p and g32 must be 2D batched tensors");
    TORCH_CHECK(p.sizes() == g32.sizes(), "p and g32 must have matching batched shapes");
    TORCH_CHECK(g32.scalar_type() == torch::kFloat32, "g32 must be float32");
    TORCH_CHECK(
        p.scalar_type() == torch::kFloat32 || p.scalar_type() == torch::kFloat16 || p.scalar_type() == torch::kBFloat16,
        "p must be float32/float16/bfloat16"
    );
    TORCH_CHECK(exp_avg_q.is_cuda() && exp_avg_q.scalar_type() == torch::kInt8 && exp_avg_q.dim() == 2, "exp_avg_q must be CUDA int8 2D");
    TORCH_CHECK(exp_avg_sq_q.is_cuda() && exp_avg_sq_q.scalar_type() == torch::kUInt8 && exp_avg_sq_q.dim() == 2, "exp_avg_sq_q must be CUDA uint8 2D");
    TORCH_CHECK(exp_avg_absmax.is_cuda() && exp_avg_absmax.scalar_type() == torch::kFloat32 && exp_avg_absmax.dim() == 2, "exp_avg_absmax must be CUDA float32 2D");
    TORCH_CHECK(exp_avg_sq_absmax.is_cuda() && exp_avg_sq_absmax.scalar_type() == torch::kFloat32 && exp_avg_sq_absmax.dim() == 2, "exp_avg_sq_absmax must be CUDA float32 2D");
    TORCH_CHECK(update.is_cuda() && update.scalar_type() == torch::kFloat32 && update.is_contiguous() && update.dim() == 2, "update must be CUDA float32 contiguous 2D");
    TORCH_CHECK(sum_update_partial.is_cuda() && sum_update_partial.scalar_type() == torch::kFloat32 && sum_update_partial.dim() == 2, "sum_update_partial must be CUDA float32 2D");
    TORCH_CHECK(sum_update.is_cuda() && sum_update.scalar_type() == torch::kFloat32 && sum_update.dim() == 1, "sum_update must be CUDA float32 1D");
    TORCH_CHECK(exp_avg_q.sizes() == p.sizes(), "exp_avg_q must match p shape");
    TORCH_CHECK(exp_avg_sq_q.sizes() == p.sizes(), "exp_avg_sq_q must match p shape");
    TORCH_CHECK(update.sizes() == p.sizes(), "update must match p shape");
    TORCH_CHECK(exp_avg_q.size(0) == exp_avg_absmax.size(0), "exp_avg_absmax batch size mismatch");
    TORCH_CHECK(exp_avg_sq_q.size(0) == exp_avg_sq_absmax.size(0), "exp_avg_sq_absmax batch size mismatch");
    TORCH_CHECK(sum_update.size(0) == p.size(0), "sum_update batch size mismatch");
    const int64_t per_item_numel = p.numel() / p.size(0);
    TORCH_CHECK(exp_avg_absmax.size(1) >= (per_item_numel + block_size - 1) / block_size, "exp_avg_absmax too small");
    TORCH_CHECK(exp_avg_sq_absmax.size(1) >= (per_item_numel + block_size - 1) / block_size, "exp_avg_sq_absmax too small");
    TORCH_CHECK(sum_update_partial.size(0) == p.size(0), "sum_update_partial batch size mismatch");
    TORCH_CHECK(sum_update_partial.size(1) >= (per_item_numel + block_size - 1) / block_size, "sum_update_partial too small");

    came_full_nonfactored_step_batched_cuda(
        p,
        g32,
        exp_avg_q,
        exp_avg_absmax,
        exp_avg_sq_q,
        exp_avg_sq_absmax,
        update,
        sum_update_partial,
        sum_update,
        beta1,
        beta2,
        eps0,
        lr,
        clip_threshold,
        weight_decay,
        block_size
    );
}

static void came_full_nonfactored_step_multitensor(
    py::list p_list,
    py::list g32_list,
    py::list exp_avg_q_list,
    py::list exp_avg_absmax_list,
    py::list exp_avg_sq_q_list,
    py::list exp_avg_sq_absmax_list,
    py::list update_list,
    py::list sum_update_partial_list,
    py::list sum_update_list,
    double beta1,
    double beta2,
    double eps0,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    const auto p_tensors = tensor_list_from_pylist(p_list, "p_list");
    const auto g32_tensors = tensor_list_from_pylist(g32_list, "g32_list");
    const auto exp_avg_q_tensors = tensor_list_from_pylist(exp_avg_q_list, "exp_avg_q_list");
    const auto exp_avg_absmax_tensors = tensor_list_from_pylist(exp_avg_absmax_list, "exp_avg_absmax_list");
    const auto exp_avg_sq_q_tensors = tensor_list_from_pylist(exp_avg_sq_q_list, "exp_avg_sq_q_list");
    const auto exp_avg_sq_absmax_tensors = tensor_list_from_pylist(exp_avg_sq_absmax_list, "exp_avg_sq_absmax_list");
    const auto update_tensors = tensor_list_from_pylist(update_list, "update_list");
    const auto sum_update_partial_tensors = tensor_list_from_pylist(sum_update_partial_list, "sum_update_partial_list");
    const auto sum_update_tensors = tensor_list_from_pylist(sum_update_list, "sum_update_list");

    const int64_t batch = (int64_t)p_tensors.size();
    TORCH_CHECK(batch > 0, "came_full_nonfactored_step_multitensor requires at least one tensor");
    TORCH_CHECK((int64_t)g32_tensors.size() == batch, "g32_list length mismatch");
    TORCH_CHECK((int64_t)exp_avg_q_tensors.size() == batch, "exp_avg_q_list length mismatch");
    TORCH_CHECK((int64_t)exp_avg_absmax_tensors.size() == batch, "exp_avg_absmax_list length mismatch");
    TORCH_CHECK((int64_t)exp_avg_sq_q_tensors.size() == batch, "exp_avg_sq_q_list length mismatch");
    TORCH_CHECK((int64_t)exp_avg_sq_absmax_tensors.size() == batch, "exp_avg_sq_absmax_list length mismatch");
    TORCH_CHECK((int64_t)update_tensors.size() == batch, "update_list length mismatch");
    TORCH_CHECK((int64_t)sum_update_partial_tensors.size() == batch, "sum_update_partial_list length mismatch");
    TORCH_CHECK((int64_t)sum_update_tensors.size() == batch, "sum_update_list length mismatch");

    const auto& p0 = p_tensors[0];
    const auto device = p0.device();
    const auto dtype = p0.scalar_type();
    TORCH_CHECK(device.is_cuda(), "came_full_nonfactored_step_multitensor tensors must be CUDA");
    TORCH_CHECK(
        dtype == torch::kFloat32 || dtype == torch::kFloat16 || dtype == torch::kBFloat16,
        "p must be float32/float16/bfloat16"
    );
    TORCH_CHECK(p0.dim() == 1, "multitensor nonfactored path currently requires 1D tensors");
    TORCH_CHECK(p0.is_contiguous(), "p tensors must be contiguous");
    const int64_t per_item_numel = p0.numel();
    const int64_t q_blocks = (per_item_numel + block_size - 1) / block_size;

    for (int64_t idx = 0; idx < batch; ++idx) {
        const auto& p = p_tensors[(size_t)idx];
        const auto& g32 = g32_tensors[(size_t)idx];
        const auto& exp_avg_q = exp_avg_q_tensors[(size_t)idx];
        const auto& exp_avg_absmax = exp_avg_absmax_tensors[(size_t)idx];
        const auto& exp_avg_sq_q = exp_avg_sq_q_tensors[(size_t)idx];
        const auto& exp_avg_sq_absmax = exp_avg_sq_absmax_tensors[(size_t)idx];
        const auto& update = update_tensors[(size_t)idx];
        const auto& sum_update_partial = sum_update_partial_tensors[(size_t)idx];
        const auto& sum_update = sum_update_tensors[(size_t)idx];

        TORCH_CHECK(p.device() == device, "all p tensors must share the same CUDA device");
        TORCH_CHECK(g32.device() == device, "all g32 tensors must share the same CUDA device");
        TORCH_CHECK(exp_avg_q.device() == device, "all exp_avg_q tensors must share the same CUDA device");
        TORCH_CHECK(exp_avg_absmax.device() == device, "all exp_avg_absmax tensors must share the same CUDA device");
        TORCH_CHECK(exp_avg_sq_q.device() == device, "all exp_avg_sq_q tensors must share the same CUDA device");
        TORCH_CHECK(exp_avg_sq_absmax.device() == device, "all exp_avg_sq_absmax tensors must share the same CUDA device");
        TORCH_CHECK(update.device() == device, "all update tensors must share the same CUDA device");
        TORCH_CHECK(sum_update_partial.device() == device, "all sum_update_partial tensors must share the same CUDA device");
        TORCH_CHECK(sum_update.device() == device, "all sum_update tensors must share the same CUDA device");
        TORCH_CHECK(p.scalar_type() == dtype, "all p tensors must share the same dtype");
        TORCH_CHECK(g32.scalar_type() == torch::kFloat32, "g32 tensors must be float32");
        TORCH_CHECK(exp_avg_q.scalar_type() == torch::kInt8, "exp_avg_q tensors must be int8");
        TORCH_CHECK(exp_avg_sq_q.scalar_type() == torch::kUInt8, "exp_avg_sq_q tensors must be uint8");
        TORCH_CHECK(exp_avg_absmax.scalar_type() == torch::kFloat32, "exp_avg_absmax tensors must be float32");
        TORCH_CHECK(exp_avg_sq_absmax.scalar_type() == torch::kFloat32, "exp_avg_sq_absmax tensors must be float32");
        TORCH_CHECK(update.scalar_type() == torch::kFloat32, "update tensors must be float32");
        TORCH_CHECK(sum_update_partial.scalar_type() == torch::kFloat32, "sum_update_partial tensors must be float32");
        TORCH_CHECK(sum_update.scalar_type() == torch::kFloat32 && sum_update.numel() == 1, "sum_update tensors must be float32 scalars");
        TORCH_CHECK(p.dim() == 1 && p.numel() == per_item_numel, "all p tensors must be 1D with matching numel");
        TORCH_CHECK(g32.dim() == 1 && g32.numel() == per_item_numel, "all g32 tensors must be 1D with matching numel");
        TORCH_CHECK(exp_avg_q.dim() == 1 && exp_avg_q.numel() == per_item_numel, "exp_avg_q shape mismatch");
        TORCH_CHECK(exp_avg_sq_q.dim() == 1 && exp_avg_sq_q.numel() == per_item_numel, "exp_avg_sq_q shape mismatch");
        TORCH_CHECK(update.dim() == 1 && update.numel() == per_item_numel, "update shape mismatch");
        TORCH_CHECK(exp_avg_absmax.dim() == 1 && exp_avg_absmax.numel() >= q_blocks, "exp_avg_absmax too small");
        TORCH_CHECK(exp_avg_sq_absmax.dim() == 1 && exp_avg_sq_absmax.numel() >= q_blocks, "exp_avg_sq_absmax too small");
        TORCH_CHECK(sum_update_partial.dim() == 1 && sum_update_partial.numel() >= q_blocks, "sum_update_partial too small");
        TORCH_CHECK(
            p.is_contiguous() && g32.is_contiguous() && exp_avg_q.is_contiguous() && exp_avg_absmax.is_contiguous()
            && exp_avg_sq_q.is_contiguous() && exp_avg_sq_absmax.is_contiguous() && update.is_contiguous()
            && sum_update_partial.is_contiguous() && sum_update.is_contiguous(),
            "multitensor tensors must be contiguous"
        );
    }

    const c10::cuda::CUDAGuard device_guard(device);
    auto p_ptrs = make_device_ptr_tensor(p_tensors, device);
    auto g32_ptrs = make_device_ptr_tensor(g32_tensors, device);
    auto exp_avg_q_ptrs = make_device_ptr_tensor(exp_avg_q_tensors, device);
    auto exp_avg_absmax_ptrs = make_device_ptr_tensor(exp_avg_absmax_tensors, device);
    auto exp_avg_sq_q_ptrs = make_device_ptr_tensor(exp_avg_sq_q_tensors, device);
    auto exp_avg_sq_absmax_ptrs = make_device_ptr_tensor(exp_avg_sq_absmax_tensors, device);
    auto update_ptrs = make_device_ptr_tensor(update_tensors, device);
    auto sum_update_ptrs = make_device_ptr_tensor(sum_update_tensors, device);

    came_full_nonfactored_step_multitensor_cuda(
        p_ptrs,
        g32_ptrs,
        exp_avg_q_ptrs,
        exp_avg_absmax_ptrs,
        exp_avg_sq_q_ptrs,
        exp_avg_sq_absmax_ptrs,
        update_ptrs,
        sum_update_ptrs,
        dtype,
        per_item_numel,
        beta1,
        beta2,
        eps0,
        lr,
        clip_threshold,
        weight_decay,
        block_size
    );
}

static void came_full_nonfactored_step_multitensor_ptrs(
    torch::Tensor sample_p,
    torch::Tensor p_ptrs,
    torch::Tensor g32_ptrs,
    torch::Tensor exp_avg_q_ptrs,
    torch::Tensor exp_avg_absmax_ptrs,
    torch::Tensor exp_avg_sq_q_ptrs,
    torch::Tensor exp_avg_sq_absmax_ptrs,
    torch::Tensor update_ptrs,
    torch::Tensor sum_update_ptrs,
    int64_t per_item_numel,
    double beta1,
    double beta2,
    double eps0,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    validate_multitensor_ptr_inputs(
        sample_p,
        p_ptrs,
        g32_ptrs,
        exp_avg_q_ptrs,
        exp_avg_absmax_ptrs,
        exp_avg_sq_q_ptrs,
        exp_avg_sq_absmax_ptrs,
        update_ptrs,
        sum_update_ptrs,
        per_item_numel,
        block_size
    );
    came_full_nonfactored_step_multitensor_cuda(
        p_ptrs,
        g32_ptrs,
        exp_avg_q_ptrs,
        exp_avg_absmax_ptrs,
        exp_avg_sq_q_ptrs,
        exp_avg_sq_absmax_ptrs,
        update_ptrs,
        sum_update_ptrs,
        sample_p.scalar_type(),
        per_item_numel,
        beta1,
        beta2,
        eps0,
        lr,
        clip_threshold,
        weight_decay,
        block_size
    );
}

static void came_full_nonfactored_step_multitensor_fp16_update_ptrs(
    torch::Tensor sample_p,
    torch::Tensor sample_update,
    torch::Tensor p_ptrs,
    torch::Tensor g32_ptrs,
    torch::Tensor exp_avg_q_ptrs,
    torch::Tensor exp_avg_absmax_ptrs,
    torch::Tensor exp_avg_sq_q_ptrs,
    torch::Tensor exp_avg_sq_absmax_ptrs,
    torch::Tensor update_ptrs,
    torch::Tensor sum_update_ptrs,
    int64_t per_item_numel,
    double beta1,
    double beta2,
    double eps0,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    validate_multitensor_ptr_inputs(
        sample_p,
        p_ptrs,
        g32_ptrs,
        exp_avg_q_ptrs,
        exp_avg_absmax_ptrs,
        exp_avg_sq_q_ptrs,
        exp_avg_sq_absmax_ptrs,
        update_ptrs,
        sum_update_ptrs,
        per_item_numel,
        block_size
    );
    TORCH_CHECK(sample_update.is_cuda(), "sample_update must be CUDA");
    TORCH_CHECK(sample_update.device() == sample_p.device(), "sample_update device mismatch");
    TORCH_CHECK(sample_update.scalar_type() == torch::kFloat16, "sample_update must be float16");
    TORCH_CHECK(sample_update.dim() == 1, "sample_update must be 1D");
    TORCH_CHECK(sample_update.is_contiguous(), "sample_update must be contiguous");
    TORCH_CHECK(sample_update.numel() == per_item_numel, "sample_update.numel() must match per_item_numel");
    came_full_nonfactored_step_multitensor_fp16_update_cuda(
        p_ptrs,
        g32_ptrs,
        exp_avg_q_ptrs,
        exp_avg_absmax_ptrs,
        exp_avg_sq_q_ptrs,
        exp_avg_sq_absmax_ptrs,
        update_ptrs,
        sum_update_ptrs,
        sample_p.scalar_type(),
        per_item_numel,
        beta1,
        beta2,
        eps0,
        lr,
        clip_threshold,
        weight_decay,
        block_size
    );
}

static void came_full_nonfactored_step_multitensor_varlen_ptrs(
    torch::Tensor sample_p,
    torch::Tensor p_ptrs,
    torch::Tensor g32_ptrs,
    torch::Tensor exp_avg_q_ptrs,
    torch::Tensor exp_avg_absmax_ptrs,
    torch::Tensor exp_avg_sq_q_ptrs,
    torch::Tensor exp_avg_sq_absmax_ptrs,
    torch::Tensor update_ptrs,
    torch::Tensor sum_update_ptrs,
    torch::Tensor item_numels,
    int64_t max_item_numel,
    double beta1,
    double beta2,
    double eps0,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    validate_multitensor_varlen_ptr_inputs(
        sample_p,
        p_ptrs,
        g32_ptrs,
        exp_avg_q_ptrs,
        exp_avg_absmax_ptrs,
        exp_avg_sq_q_ptrs,
        exp_avg_sq_absmax_ptrs,
        update_ptrs,
        sum_update_ptrs,
        item_numels,
        max_item_numel,
        block_size
    );
    came_full_nonfactored_step_multitensor_varlen_cuda(
        p_ptrs,
        g32_ptrs,
        exp_avg_q_ptrs,
        exp_avg_absmax_ptrs,
        exp_avg_sq_q_ptrs,
        exp_avg_sq_absmax_ptrs,
        update_ptrs,
        sum_update_ptrs,
        item_numels,
        sample_p.scalar_type(),
        max_item_numel,
        beta1,
        beta2,
        eps0,
        lr,
        clip_threshold,
        weight_decay,
        block_size
    );
}

static void came_full_nonfactored_step_multitensor_compact_varlen_ptrs(
    torch::Tensor sample_p,
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
    double beta1,
    double beta2,
    double eps0,
    double lr,
    double clip_threshold,
    double weight_decay,
    int64_t block_size
) {
    validate_multitensor_compact_varlen_ptr_inputs(
        sample_p,
        p_ptrs,
        g32_ptrs,
        exp_avg_q_ptrs,
        exp_avg_absmax_ptrs,
        exp_avg_sq_q_ptrs,
        exp_avg_sq_absmax_ptrs,
        update_ptrs,
        sum_update_ptrs,
        item_numels,
        block_item_ids,
        block_item_starts,
        block_size
    );
    came_full_nonfactored_step_multitensor_compact_varlen_cuda(
        p_ptrs,
        g32_ptrs,
        exp_avg_q_ptrs,
        exp_avg_absmax_ptrs,
        exp_avg_sq_q_ptrs,
        exp_avg_sq_absmax_ptrs,
        update_ptrs,
        sum_update_ptrs,
        item_numels,
        block_item_ids,
        block_item_starts,
        sample_p.scalar_type(),
        beta1,
        beta2,
        eps0,
        lr,
        clip_threshold,
        weight_decay,
        block_size
    );
}

static void fill_device_ptr_tensor_binding(
    py::list tensors,
    torch::Tensor ptr_tensor
) {
    auto tensor_vec = tensor_list_from_pylist(tensors, "tensors");
    fill_device_ptr_tensor(tensor_vec, ptr_tensor);
}

static void came_full_factored_sq_step(
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
    TORCH_CHECK(g32.is_cuda() && g32.scalar_type() == torch::kFloat32 && g32.is_contiguous(), "g32 must be CUDA float32 contiguous");
    TORCH_CHECK(g32.dim() == 2, "g32 must be 2D");
    TORCH_CHECK(exp_avg_sq_row_q.is_cuda() && exp_avg_sq_row_q.scalar_type() == torch::kUInt8, "exp_avg_sq_row_q must be CUDA uint8");
    TORCH_CHECK(exp_avg_sq_col_q.is_cuda() && exp_avg_sq_col_q.scalar_type() == torch::kUInt8, "exp_avg_sq_col_q must be CUDA uint8");
    TORCH_CHECK(exp_avg_sq_row_absmax.is_cuda() && exp_avg_sq_row_absmax.scalar_type() == torch::kFloat32, "exp_avg_sq_row_absmax must be CUDA float32");
    TORCH_CHECK(exp_avg_sq_col_absmax.is_cuda() && exp_avg_sq_col_absmax.scalar_type() == torch::kFloat32, "exp_avg_sq_col_absmax must be CUDA float32");
    TORCH_CHECK(r_factor.is_cuda() && r_factor.scalar_type() == torch::kFloat32, "r_factor must be CUDA float32");
    TORCH_CHECK(c_factor.is_cuda() && c_factor.scalar_type() == torch::kFloat32, "c_factor must be CUDA float32");
    TORCH_CHECK(row_absmax_scratch.is_cuda() && row_absmax_scratch.scalar_type() == torch::kFloat32, "row_absmax_scratch must be CUDA float32");
    TORCH_CHECK(reduce_partial.is_cuda() && reduce_partial.scalar_type() == torch::kFloat32, "reduce_partial must be CUDA float32");
    TORCH_CHECK(sum_row_state.is_cuda() && sum_row_state.scalar_type() == torch::kFloat32 && sum_row_state.numel() == 1, "sum_row_state must be CUDA float32 scalar");
    TORCH_CHECK(sum_update.is_cuda() && sum_update.scalar_type() == torch::kFloat32 && sum_update.numel() == 1, "sum_update must be CUDA float32 scalar");
    TORCH_CHECK(exp_avg_sq_row_q.numel() == g32.size(0), "row state shape mismatch");
    TORCH_CHECK(exp_avg_sq_col_q.numel() == g32.size(1), "col state shape mismatch");
    came_full_factored_sq_step_cuda(
        g32,
        exp_avg_sq_row_q,
        exp_avg_sq_row_absmax,
        exp_avg_sq_col_q,
        exp_avg_sq_col_absmax,
        r_factor,
        c_factor,
        row_absmax_scratch,
        reduce_partial,
        sum_row_state,
        sum_update,
        beta2,
        eps0,
        block_size
    );
}

static void came_full_factored_sq_step_batched(
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
    TORCH_CHECK(g32.is_cuda() && g32.dim() == 3, "g32 must be CUDA 3D");
    TORCH_CHECK(exp_avg_sq_row_q.is_cuda() && exp_avg_sq_row_q.dim() == 2, "exp_avg_sq_row_q must be CUDA 2D");
    TORCH_CHECK(exp_avg_sq_col_q.is_cuda() && exp_avg_sq_col_q.dim() == 2, "exp_avg_sq_col_q must be CUDA 2D");
    TORCH_CHECK(exp_avg_sq_row_absmax.is_cuda() && exp_avg_sq_row_absmax.dim() == 2, "exp_avg_sq_row_absmax must be CUDA 2D");
    TORCH_CHECK(exp_avg_sq_col_absmax.is_cuda() && exp_avg_sq_col_absmax.dim() == 2, "exp_avg_sq_col_absmax must be CUDA 2D");
    TORCH_CHECK(r_factor.is_cuda() && r_factor.dim() == 2, "r_factor must be CUDA 2D");
    TORCH_CHECK(c_factor.is_cuda() && c_factor.dim() == 2, "c_factor must be CUDA 2D");
    const auto batch = g32.size(0);
    TORCH_CHECK(exp_avg_sq_row_q.size(0) == batch, "row batch size mismatch");
    TORCH_CHECK(exp_avg_sq_col_q.size(0) == batch, "col batch size mismatch");
    TORCH_CHECK(r_factor.size(0) == batch, "r_factor batch size mismatch");
    TORCH_CHECK(c_factor.size(0) == batch, "c_factor batch size mismatch");
    TORCH_CHECK(row_absmax_scratch.is_cuda() && row_absmax_scratch.dim() == 2 && row_absmax_scratch.size(0) == batch, "row_absmax_scratch batch mismatch");
    TORCH_CHECK(reduce_partial.is_cuda() && reduce_partial.dim() == 2 && reduce_partial.size(0) == batch, "reduce_partial batch mismatch");
    TORCH_CHECK(sum_row_state.is_cuda() && sum_row_state.dim() == 1 && sum_row_state.size(0) == batch, "sum_row_state batch mismatch");
    TORCH_CHECK(sum_update_slice.is_cuda() && sum_update_slice.dim() == 1 && sum_update_slice.size(0) == batch, "sum_update_slice batch mismatch");
    TORCH_CHECK(sum_update_total.is_cuda() && sum_update_total.numel() == 1, "sum_update_total must be CUDA scalar");
    came_full_factored_sq_step_batched_cuda(
        g32,
        exp_avg_sq_row_q,
        exp_avg_sq_row_absmax,
        exp_avg_sq_col_q,
        exp_avg_sq_col_absmax,
        r_factor,
        c_factor,
        row_absmax_scratch,
        reduce_partial,
        sum_row_state,
        sum_update_slice,
        sum_update_total,
        beta2,
        eps0,
        block_size
    );
}

static void came_full_factored_res_step(
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
    TORCH_CHECK(res32.is_cuda() && res32.scalar_type() == torch::kFloat32 && res32.is_contiguous(), "res32 must be CUDA float32 contiguous");
    TORCH_CHECK(res32.dim() == 2, "res32 must be 2D");
    TORCH_CHECK(exp_avg_res_row_q.is_cuda() && exp_avg_res_row_q.scalar_type() == torch::kUInt8, "exp_avg_res_row_q must be CUDA uint8");
    TORCH_CHECK(exp_avg_res_col_q.is_cuda() && exp_avg_res_col_q.scalar_type() == torch::kUInt8, "exp_avg_res_col_q must be CUDA uint8");
    TORCH_CHECK(exp_avg_res_row_absmax.is_cuda() && exp_avg_res_row_absmax.scalar_type() == torch::kFloat32, "exp_avg_res_row_absmax must be CUDA float32");
    TORCH_CHECK(exp_avg_res_col_absmax.is_cuda() && exp_avg_res_col_absmax.scalar_type() == torch::kFloat32, "exp_avg_res_col_absmax must be CUDA float32");
    TORCH_CHECK(r_factor.is_cuda() && r_factor.scalar_type() == torch::kFloat32, "r_factor must be CUDA float32");
    TORCH_CHECK(c_factor.is_cuda() && c_factor.scalar_type() == torch::kFloat32, "c_factor must be CUDA float32");
    TORCH_CHECK(row_absmax_scratch.is_cuda() && row_absmax_scratch.scalar_type() == torch::kFloat32, "row_absmax_scratch must be CUDA float32");
    TORCH_CHECK(reduce_partial.is_cuda() && reduce_partial.scalar_type() == torch::kFloat32, "reduce_partial must be CUDA float32");
    TORCH_CHECK(sum_row_state.is_cuda() && sum_row_state.scalar_type() == torch::kFloat32 && sum_row_state.numel() == 1, "sum_row_state must be CUDA float32 scalar");
    TORCH_CHECK(exp_avg_res_row_q.numel() == res32.size(0), "row state shape mismatch");
    TORCH_CHECK(exp_avg_res_col_q.numel() == res32.size(1), "col state shape mismatch");
    came_full_factored_res_step_cuda(
        res32,
        exp_avg_res_row_q,
        exp_avg_res_row_absmax,
        exp_avg_res_col_q,
        exp_avg_res_col_absmax,
        r_factor,
        c_factor,
        row_absmax_scratch,
        reduce_partial,
        sum_row_state,
        beta3,
        block_size
    );
}

static void came_full_factored_res_step_batched(
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
    TORCH_CHECK(res32.is_cuda() && res32.dim() == 3, "res32 must be CUDA 3D");
    const auto batch = res32.size(0);
    TORCH_CHECK(exp_avg_res_row_q.is_cuda() && exp_avg_res_row_q.dim() == 2 && exp_avg_res_row_q.size(0) == batch, "exp_avg_res_row_q batch mismatch");
    TORCH_CHECK(exp_avg_res_col_q.is_cuda() && exp_avg_res_col_q.dim() == 2 && exp_avg_res_col_q.size(0) == batch, "exp_avg_res_col_q batch mismatch");
    TORCH_CHECK(r_factor.is_cuda() && r_factor.dim() == 2 && r_factor.size(0) == batch, "r_factor batch mismatch");
    TORCH_CHECK(c_factor.is_cuda() && c_factor.dim() == 2 && c_factor.size(0) == batch, "c_factor batch mismatch");
    TORCH_CHECK(row_absmax_scratch.is_cuda() && row_absmax_scratch.dim() == 2 && row_absmax_scratch.size(0) == batch, "row_absmax_scratch batch mismatch");
    TORCH_CHECK(reduce_partial.is_cuda() && reduce_partial.dim() == 2 && reduce_partial.size(0) == batch, "reduce_partial batch mismatch");
    TORCH_CHECK(sum_row_state.is_cuda() && sum_row_state.dim() == 1 && sum_row_state.size(0) == batch, "sum_row_state batch mismatch");
    came_full_factored_res_step_batched_cuda(
        res32,
        exp_avg_res_row_q,
        exp_avg_res_row_absmax,
        exp_avg_res_col_q,
        exp_avg_res_col_absmax,
        r_factor,
        c_factor,
        row_absmax_scratch,
        reduce_partial,
        sum_row_state,
        beta3,
        block_size
    );
}

static void came_full_factored_expavg_res_prepare(
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
    TORCH_CHECK(g32.is_cuda() && g32.scalar_type() == torch::kFloat32 && g32.is_contiguous(), "g32 must be CUDA float32 contiguous");
    TORCH_CHECK(g32.dim() == 2, "g32 must be 2D");
    TORCH_CHECK(exp_avg_q.is_cuda() && exp_avg_q.scalar_type() == torch::kInt8 && exp_avg_q.is_contiguous(), "exp_avg_q must be CUDA int8 contiguous");
    TORCH_CHECK(exp_avg_absmax.is_cuda() && exp_avg_absmax.scalar_type() == torch::kFloat32 && exp_avg_absmax.is_contiguous(), "exp_avg_absmax must be CUDA float32 contiguous");
    TORCH_CHECK(r_factor.is_cuda() && r_factor.scalar_type() == torch::kFloat32, "r_factor must be CUDA float32");
    TORCH_CHECK(c_factor.is_cuda() && c_factor.scalar_type() == torch::kFloat32, "c_factor must be CUDA float32");
    TORCH_CHECK(exp_avg_fp32.is_cuda() && exp_avg_fp32.scalar_type() == torch::kFloat32 && exp_avg_fp32.is_contiguous(), "exp_avg_fp32 must be CUDA float32 contiguous");
    TORCH_CHECK(res32.is_cuda() && res32.scalar_type() == torch::kFloat32 && res32.is_contiguous(), "res32 must be CUDA float32 contiguous");
    TORCH_CHECK(sum_update.is_cuda() && sum_update.scalar_type() == torch::kFloat32 && sum_update.numel() == 1, "sum_update must be CUDA float32 scalar");
    TORCH_CHECK(exp_avg_q.numel() == g32.numel(), "exp_avg_q shape mismatch");
    TORCH_CHECK(exp_avg_fp32.sizes() == g32.sizes(), "exp_avg_fp32 shape mismatch");
    TORCH_CHECK(res32.sizes() == g32.sizes(), "res32 shape mismatch");
    TORCH_CHECK(r_factor.numel() == g32.size(0), "r_factor shape mismatch");
    TORCH_CHECK(c_factor.numel() == g32.size(1), "c_factor shape mismatch");
    TORCH_CHECK(exp_avg_absmax.numel() >= (g32.numel() + block_size - 1) / block_size, "exp_avg_absmax too small");
    came_full_factored_expavg_res_prepare_cuda(
        g32,
        exp_avg_q,
        exp_avg_absmax,
        r_factor,
        c_factor,
        exp_avg_fp32,
        res32,
        sum_update,
        beta1,
        eps1,
        clip_threshold,
        block_size
    );
}

static void came_full_factored_expavg_res_prepare_batched(
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
    TORCH_CHECK(g32.is_cuda() && g32.dim() == 3, "g32 must be CUDA 3D");
    TORCH_CHECK(exp_avg_q.is_cuda() && exp_avg_q.dim() == 3, "exp_avg_q must be CUDA 3D");
    TORCH_CHECK(exp_avg_absmax.is_cuda() && exp_avg_absmax.dim() == 2, "exp_avg_absmax must be CUDA 2D");
    TORCH_CHECK(exp_avg_fp32.is_cuda() && exp_avg_fp32.dim() == 3, "exp_avg_fp32 must be CUDA 3D");
    TORCH_CHECK(res32.is_cuda() && res32.dim() == 3, "res32 must be CUDA 3D");
    const auto batch = g32.size(0);
    TORCH_CHECK(exp_avg_q.size(0) == batch, "exp_avg_q batch mismatch");
    TORCH_CHECK(exp_avg_absmax.size(0) == batch, "exp_avg_absmax batch mismatch");
    TORCH_CHECK(r_factor.is_cuda() && r_factor.dim() == 2 && r_factor.size(0) == batch, "r_factor batch mismatch");
    TORCH_CHECK(c_factor.is_cuda() && c_factor.dim() == 2 && c_factor.size(0) == batch, "c_factor batch mismatch");
    TORCH_CHECK(exp_avg_fp32.size(0) == batch && res32.size(0) == batch, "exp_avg_fp32/res32 batch mismatch");
    came_full_factored_expavg_res_prepare_batched_cuda(
        g32,
        exp_avg_q,
        exp_avg_absmax,
        r_factor,
        c_factor,
        exp_avg_fp32,
        res32,
        sum_update,
        beta1,
        eps1,
        clip_threshold,
        block_size
    );
}

static void came_full_factored_param_update(
    torch::Tensor p,
    torch::Tensor exp_avg_fp32,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    double lr,
    double weight_decay
) {
    TORCH_CHECK(p.is_cuda() && exp_avg_fp32.is_cuda(), "p and exp_avg_fp32 must be CUDA");
    TORCH_CHECK(exp_avg_fp32.scalar_type() == torch::kFloat32 && exp_avg_fp32.is_contiguous(), "exp_avg_fp32 must be CUDA float32 contiguous");
    TORCH_CHECK(p.dim() == 2 && exp_avg_fp32.dim() == 2, "p and exp_avg_fp32 must be 2D");
    TORCH_CHECK(p.sizes() == exp_avg_fp32.sizes(), "p and exp_avg_fp32 shape mismatch");
    TORCH_CHECK(r_factor.is_cuda() && r_factor.scalar_type() == torch::kFloat32 && r_factor.numel() == p.size(0), "r_factor shape mismatch");
    TORCH_CHECK(c_factor.is_cuda() && c_factor.scalar_type() == torch::kFloat32 && c_factor.numel() == p.size(1), "c_factor shape mismatch");
    came_full_factored_param_update_cuda(p, exp_avg_fp32, r_factor, c_factor, lr, weight_decay);
}

static void came_fp_factored_step(
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
    TORCH_CHECK(p.is_cuda() && p.is_contiguous(), "p must be CUDA contiguous");
    TORCH_CHECK(
        p.scalar_type() == torch::kFloat32 || p.scalar_type() == torch::kFloat16 || p.scalar_type() == torch::kBFloat16,
        "p must be float32/float16/bfloat16"
    );
    TORCH_CHECK(g32.is_cuda() && g32.scalar_type() == torch::kFloat32 && g32.is_contiguous(), "g32 must be CUDA float32 contiguous");
    TORCH_CHECK(p.dim() == 2 && g32.dim() == 2, "p and g32 must be 2D");
    TORCH_CHECK(p.sizes() == g32.sizes(), "p and g32 shape mismatch");
    TORCH_CHECK(exp_avg.is_cuda() && exp_avg.scalar_type() == torch::kFloat32 && exp_avg.is_contiguous(), "exp_avg must be CUDA float32 contiguous");
    TORCH_CHECK(exp_avg.sizes() == g32.sizes(), "exp_avg shape mismatch");
    TORCH_CHECK(exp_avg_sq_row.is_cuda() && exp_avg_sq_row.scalar_type() == torch::kFloat32 && exp_avg_sq_row.is_contiguous(), "exp_avg_sq_row must be CUDA float32 contiguous");
    TORCH_CHECK(exp_avg_sq_col.is_cuda() && exp_avg_sq_col.scalar_type() == torch::kFloat32 && exp_avg_sq_col.is_contiguous(), "exp_avg_sq_col must be CUDA float32 contiguous");
    TORCH_CHECK(exp_avg_res_row.is_cuda() && exp_avg_res_row.scalar_type() == torch::kFloat32 && exp_avg_res_row.is_contiguous(), "exp_avg_res_row must be CUDA float32 contiguous");
    TORCH_CHECK(exp_avg_res_col.is_cuda() && exp_avg_res_col.scalar_type() == torch::kFloat32 && exp_avg_res_col.is_contiguous(), "exp_avg_res_col must be CUDA float32 contiguous");
    TORCH_CHECK(r_factor.is_cuda() && r_factor.scalar_type() == torch::kFloat32 && r_factor.is_contiguous(), "r_factor must be CUDA float32 contiguous");
    TORCH_CHECK(c_factor.is_cuda() && c_factor.scalar_type() == torch::kFloat32 && c_factor.is_contiguous(), "c_factor must be CUDA float32 contiguous");
    TORCH_CHECK(scratch.is_cuda() && scratch.scalar_type() == torch::kFloat32 && scratch.is_contiguous(), "scratch must be CUDA float32 contiguous");
    TORCH_CHECK(reduce_partial.is_cuda() && reduce_partial.scalar_type() == torch::kFloat32 && reduce_partial.is_contiguous(), "reduce_partial must be CUDA float32 contiguous");
    TORCH_CHECK(sum_row_state.is_cuda() && sum_row_state.scalar_type() == torch::kFloat32 && sum_row_state.numel() == 1, "sum_row_state must be CUDA float32 scalar");
    TORCH_CHECK(sum_update.is_cuda() && sum_update.scalar_type() == torch::kFloat32 && sum_update.numel() == 1, "sum_update must be CUDA float32 scalar");
    TORCH_CHECK(exp_avg_sq_row.numel() == g32.size(0), "exp_avg_sq_row shape mismatch");
    TORCH_CHECK(exp_avg_res_row.numel() == g32.size(0), "exp_avg_res_row shape mismatch");
    TORCH_CHECK(r_factor.numel() == g32.size(0), "r_factor shape mismatch");
    TORCH_CHECK(exp_avg_sq_col.numel() == g32.size(1), "exp_avg_sq_col shape mismatch");
    TORCH_CHECK(exp_avg_res_col.numel() == g32.size(1), "exp_avg_res_col shape mismatch");
    TORCH_CHECK(c_factor.numel() == g32.size(1), "c_factor shape mismatch");
    TORCH_CHECK(scratch.sizes() == g32.sizes(), "scratch shape mismatch");
    TORCH_CHECK(reduce_partial.numel() >= std::max((g32.size(0) + 255) / 256, (g32.numel() + 255) / 256), "reduce_partial too small");

    came_fp_factored_step_cuda(
        p,
        g32,
        exp_avg,
        exp_avg_sq_row,
        exp_avg_sq_col,
        exp_avg_res_row,
        exp_avg_res_col,
        r_factor,
        c_factor,
        scratch,
        reduce_partial,
        sum_row_state,
        sum_update,
        beta1,
        beta2,
        beta3,
        eps0,
        eps1,
        lr,
        clip_threshold,
        weight_decay,
        block_size
    );
}

static void came_full_factored_param_update_batched(
    torch::Tensor p,
    torch::Tensor exp_avg_fp32,
    torch::Tensor r_factor,
    torch::Tensor c_factor,
    double lr,
    double weight_decay
) {
    TORCH_CHECK(p.is_cuda() && p.dim() == 3, "p must be CUDA 3D");
    TORCH_CHECK(exp_avg_fp32.is_cuda() && exp_avg_fp32.dim() == 3, "exp_avg_fp32 must be CUDA 3D");
    const auto batch = p.size(0);
    TORCH_CHECK(exp_avg_fp32.size(0) == batch, "exp_avg_fp32 batch mismatch");
    TORCH_CHECK(r_factor.is_cuda() && r_factor.dim() == 2 && r_factor.size(0) == batch, "r_factor batch mismatch");
    TORCH_CHECK(c_factor.is_cuda() && c_factor.dim() == 2 && c_factor.size(0) == batch, "c_factor batch mismatch");
    came_full_factored_param_update_batched_cuda(
        p,
        exp_avg_fp32,
        r_factor,
        c_factor,
        lr,
        weight_decay
    );
}

static void came_full_factored_nd_chunked_step(
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
    bool direct_row_sum = false
) {
    TORCH_CHECK(p.is_cuda() && p.dim() == 3, "p must be CUDA 3D");
    TORCH_CHECK(g32.is_cuda() && g32.scalar_type() == torch::kFloat32 && g32.dim() == 3, "g32 must be CUDA float32 3D");
    TORCH_CHECK(p.sizes() == g32.sizes(), "p and g32 shape mismatch");
    TORCH_CHECK(exp_avg_q.is_cuda() && exp_avg_q.dim() == 3 && exp_avg_q.sizes() == p.sizes(), "exp_avg_q must be CUDA 3D and match p");
    TORCH_CHECK(exp_avg_absmax.is_cuda() && exp_avg_absmax.dim() == 2 && exp_avg_absmax.size(0) == p.size(0), "exp_avg_absmax batch mismatch");
    TORCH_CHECK(exp_avg_sq_row_q.is_cuda() && exp_avg_sq_row_q.dim() == 2 && exp_avg_sq_row_q.size(0) == p.size(0), "exp_avg_sq_row_q batch mismatch");
    TORCH_CHECK(exp_avg_sq_row_absmax.is_cuda() && exp_avg_sq_row_absmax.dim() == 2 && exp_avg_sq_row_absmax.size(0) == p.size(0), "exp_avg_sq_row_absmax batch mismatch");
    TORCH_CHECK(exp_avg_sq_col_q.is_cuda() && exp_avg_sq_col_q.dim() == 2 && exp_avg_sq_col_q.size(0) == p.size(0), "exp_avg_sq_col_q batch mismatch");
    TORCH_CHECK(exp_avg_sq_col_absmax.is_cuda() && exp_avg_sq_col_absmax.dim() == 2 && exp_avg_sq_col_absmax.size(0) == p.size(0), "exp_avg_sq_col_absmax batch mismatch");
    TORCH_CHECK(exp_avg_res_row_q.is_cuda() && exp_avg_res_row_q.dim() == 2 && exp_avg_res_row_q.size(0) == p.size(0), "exp_avg_res_row_q batch mismatch");
    TORCH_CHECK(exp_avg_res_row_absmax.is_cuda() && exp_avg_res_row_absmax.dim() == 2 && exp_avg_res_row_absmax.size(0) == p.size(0), "exp_avg_res_row_absmax batch mismatch");
    TORCH_CHECK(exp_avg_res_col_q.is_cuda() && exp_avg_res_col_q.dim() == 2 && exp_avg_res_col_q.size(0) == p.size(0), "exp_avg_res_col_q batch mismatch");
    TORCH_CHECK(exp_avg_res_col_absmax.is_cuda() && exp_avg_res_col_absmax.dim() == 2 && exp_avg_res_col_absmax.size(0) == p.size(0), "exp_avg_res_col_absmax batch mismatch");
    TORCH_CHECK(r_factor.is_cuda() && r_factor.dim() == 2 && r_factor.size(0) == p.size(0), "r_factor batch mismatch");
    TORCH_CHECK(c_factor.is_cuda() && c_factor.dim() == 2 && c_factor.size(0) == p.size(0), "c_factor batch mismatch");
    TORCH_CHECK(
        exp_avg_fp32.is_cuda()
        && exp_avg_fp32.dim() == 3
        && (exp_avg_fp32.scalar_type() == torch::kFloat32 || exp_avg_fp32.scalar_type() == torch::kFloat16),
        "exp_avg_fp32 must be CUDA float32/float16 3D"
    );
    TORCH_CHECK(
        res32.is_cuda()
        && res32.dim() == 3
        && (res32.scalar_type() == torch::kFloat32 || res32.scalar_type() == torch::kFloat16),
        "res32 must be CUDA float32/float16 3D"
    );
    TORCH_CHECK(row_absmax_scratch.is_cuda() && row_absmax_scratch.dim() == 2, "row_absmax_scratch must be CUDA 2D");
    TORCH_CHECK(reduce_partial.is_cuda() && reduce_partial.dim() == 2, "reduce_partial must be CUDA 2D");
    TORCH_CHECK(sum_row_state.is_cuda() && sum_row_state.dim() == 1, "sum_row_state must be CUDA 1D");
    TORCH_CHECK(sum_update_slice.is_cuda() && sum_update_slice.dim() == 1, "sum_update_slice must be CUDA 1D");
    TORCH_CHECK(sum_update_chunk.is_cuda() && sum_update_chunk.numel() == 1, "sum_update_chunk must be CUDA scalar");
    TORCH_CHECK(sum_update_total.is_cuda() && sum_update_total.numel() == 1, "sum_update_total must be CUDA scalar");
    TORCH_CHECK(sum_update_equiv.is_cuda() && sum_update_equiv.numel() == 1, "sum_update_equiv must be CUDA scalar");
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

    const auto batch = p.size(0);
    const auto max_chunk = std::min<int64_t>(chunk_size, batch);
    TORCH_CHECK(exp_avg_fp32.size(0) >= max_chunk, "exp_avg_fp32 scratch batch too small");
    TORCH_CHECK(res32.size(0) >= max_chunk, "res32 scratch batch too small");
    TORCH_CHECK(row_absmax_scratch.size(0) >= max_chunk, "row_absmax_scratch batch too small");
    TORCH_CHECK(reduce_partial.size(0) >= max_chunk, "reduce_partial batch too small");
    TORCH_CHECK(sum_row_state.size(0) >= max_chunk, "sum_row_state batch too small");
    TORCH_CHECK(sum_update_slice.size(0) >= max_chunk, "sum_update_slice batch too small");
    came_full_factored_nd_chunked_step_cuda(
        p,
        g32,
        exp_avg_q,
        exp_avg_absmax,
        exp_avg_sq_row_q,
        exp_avg_sq_row_absmax,
        exp_avg_sq_col_q,
        exp_avg_sq_col_absmax,
        exp_avg_res_row_q,
        exp_avg_res_row_absmax,
        exp_avg_res_col_q,
        exp_avg_res_col_absmax,
        r_factor,
        c_factor,
        row_absmax_scratch,
        reduce_partial,
        sum_row_state,
        sum_update_slice,
        sum_update_chunk,
        sum_update_total,
        sum_update_equiv,
        exp_avg_fp32,
        res32,
        beta1,
        beta2,
        beta3,
        eps0,
        eps1,
        lr,
        clip_threshold,
        weight_decay,
        chunk_size,
        block_size,
        direct_row_sum
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("came2d_step", &came2d_step, "CAME 2D optimizer step (CUDA, exp_avg int8)");
    m.def("blockwise_quant", &blockwise_quant, "Generic blockwise quantize (CUDA)");
    m.def("blockwise_dequant", &blockwise_dequant, "Generic blockwise dequantize (CUDA)");
    m.def("blockwise_quant_batched", &blockwise_quant_batched, "Generic blockwise batched quantize (CUDA)");
    m.def("blockwise_dequant_batched", &blockwise_dequant_batched, "Generic blockwise batched dequantize (CUDA)");
    m.def("fill_device_ptr_tensor", &fill_device_ptr_tensor_binding, "Fill a CUDA int64 pointer tensor from a tensor list");
    m.def("came_full_nonfactored_step", &came_full_nonfactored_step, "CAME full non-factored step (CUDA)");
    m.def("came_full_nonfactored_step_fp16_update", &came_full_nonfactored_step_fp16_update, "CAME full non-factored step fp16-update (CUDA)");
    m.def("came_full_nonfactored_step_batched", &came_full_nonfactored_step_batched, "CAME full non-factored step batched (CUDA)");
    m.def("came_full_nonfactored_step_multitensor", &came_full_nonfactored_step_multitensor, "CAME full non-factored step multitensor (CUDA)");
    m.def("came_full_nonfactored_step_multitensor_ptrs", &came_full_nonfactored_step_multitensor_ptrs, "CAME full non-factored step multitensor ptrs (CUDA)");
    m.def("came_full_nonfactored_step_multitensor_fp16_update_ptrs", &came_full_nonfactored_step_multitensor_fp16_update_ptrs, "CAME full non-factored step multitensor fp16-update ptrs (CUDA)");
    m.def("came_full_nonfactored_step_multitensor_varlen_ptrs", &came_full_nonfactored_step_multitensor_varlen_ptrs, "CAME full non-factored step multitensor variable-length ptrs (CUDA)");
    m.def("came_full_nonfactored_step_multitensor_compact_varlen_ptrs", &came_full_nonfactored_step_multitensor_compact_varlen_ptrs, "CAME full non-factored step multitensor compact variable-length ptrs (CUDA)");
    m.def("came_full_factored_sq_step", &came_full_factored_sq_step, "CAME full factored sq-state step (CUDA)");
    m.def("came_full_factored_sq_step_batched", &came_full_factored_sq_step_batched, "CAME full factored sq-state step batched (CUDA)");
    m.def("came_full_factored_res_step", &came_full_factored_res_step, "CAME full factored residual-state step (CUDA)");
    m.def("came_full_factored_res_step_batched", &came_full_factored_res_step_batched, "CAME full factored residual-state step batched (CUDA)");
    m.def("came_full_factored_expavg_res_prepare", &came_full_factored_expavg_res_prepare, "CAME full factored exp_avg/residual prep (CUDA)");
    m.def("came_full_factored_expavg_res_prepare_batched", &came_full_factored_expavg_res_prepare_batched, "CAME full factored exp_avg/residual prep batched (CUDA)");
    m.def("came_full_factored_param_update", &came_full_factored_param_update, "CAME full factored param update (CUDA)");
    m.def("came_fp_factored_step", &came_fp_factored_step, "CAME fp-state factored step (CUDA)");
    m.def("came_full_factored_param_update_batched", &came_full_factored_param_update_batched, "CAME full factored param update batched (CUDA)");
    m.def("came_full_factored_nd_chunked_step", &came_full_factored_nd_chunked_step, "CAME full factored ND chunked step (CUDA)");
}
