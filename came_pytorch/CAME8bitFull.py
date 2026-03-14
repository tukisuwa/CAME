from __future__ import annotations

import math
import time

import torch
import torch.optim

from came_pytorch.blockwise_quantization import (
    dequantize_blockwise,
    dequantize_blockwise_batched_into,
    dequantize_blockwise_into,
    init_batched_qstate,
    init_batched_blockwise_workspace,
    init_blockwise_workspace,
    init_qstate,
    quantize_blockwise_batched_into,
    quantize_blockwise,
    quantize_blockwise_into,
)
from came_pytorch.came_cuda import (
    blockwise_dequantize_batched_into as cuda_blockwise_dequantize_batched_into,
    blockwise_dequantize_into as cuda_blockwise_dequantize_into,
    came_full_factored_expavg_res_prepare as cuda_came_full_factored_expavg_res_prepare,
    came_full_factored_expavg_res_prepare_batched as cuda_came_full_factored_expavg_res_prepare_batched,
    came_full_factored_nd_chunked_step as cuda_came_full_factored_nd_chunked_step,
    came_full_factored_param_update as cuda_came_full_factored_param_update,
    came_full_factored_param_update_batched as cuda_came_full_factored_param_update_batched,
    came_full_factored_res_step as cuda_came_full_factored_res_step,
    came_full_factored_res_step_batched as cuda_came_full_factored_res_step_batched,
    came_full_factored_sq_step as cuda_came_full_factored_sq_step,
    came_full_factored_sq_step_batched as cuda_came_full_factored_sq_step_batched,
    came_full_nonfactored_step as cuda_came_full_nonfactored_step,
    came_full_nonfactored_step_fp16_update as cuda_came_full_nonfactored_step_fp16_update,
    blockwise_quantize_batched_into as cuda_blockwise_quantize_batched_into,
    blockwise_quantize_into as cuda_blockwise_quantize_into,
)


class CAME8bitFull(torch.optim.Optimizer):
    """
    Reference all-state 8-bit implementation of CAME.

    Unlike the CUDA fast path, this optimizer quantizes every optimizer state
    blockwise and works on CPU or CUDA for arbitrary parameter shapes. It is
    intended as a correctness-oriented "full 8-bit" implementation that can be
    optimized further later.
    """

    _ND_EXT_ALWAYS_PREFIX_MAX = 4
    _ND_EXT_LARGE_PREFIX_MAX = 8
    _ND_EXT_BATCHED_LARGE_MATRIX_MIN_NUMEL = 48 * 48
    _ND_EXT_CHUNKED_LARGE_MATRIX_MIN_NUMEL = 128 * 128
    _ND_EXT_STRATEGY_NONE = "none"
    _ND_EXT_STRATEGY_BATCHED = "batched"
    _ND_EXT_STRATEGY_CHUNKED = "chunked"

    def __init__(
        self,
        params,
        lr: float,
        eps: tuple[float, float] = (1e-30, 1e-16),
        clip_threshold: float = 1.0,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        block_size: int = 256,
        prefer_factored_cuda_ext_path: bool = True,
        cuda_factored_nd_chunk_size_override: int | None = None,
        cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel: int | None = None,
        cuda_factored_nd_chunked_direct_row_sum_max_matrix_numel: int | None = None,
        cuda_nonfactored_use_fp16_update: bool = False,
    ):
        if lr is None or lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if clip_threshold <= 0.0:
            raise ValueError(f"clip_threshold must be > 0, got {clip_threshold}")
        if len(eps) != 2:
            raise ValueError(f"eps must be a tuple of length 2, got eps={eps}")
        if len(betas) != 3:
            raise ValueError(f"betas must be a tuple of length 3, got betas={betas}")
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        defaults = dict(
            lr=float(lr),
            eps=(float(eps[0]), float(eps[1])),
            clip_threshold=float(clip_threshold),
            betas=(float(betas[0]), float(betas[1]), float(betas[2])),
            weight_decay=float(weight_decay),
            block_size=int(block_size),
        )
        super().__init__(params, defaults)
        self.prefer_factored_cuda_ext_path = bool(prefer_factored_cuda_ext_path)
        self.cuda_factored_nd_chunk_size_override = (
            None if cuda_factored_nd_chunk_size_override is None else int(cuda_factored_nd_chunk_size_override)
        )
        self.cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel = (
            None
            if cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel is None
            else int(cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel)
        )
        self.cuda_factored_nd_chunked_direct_row_sum_max_matrix_numel = (
            None
            if cuda_factored_nd_chunked_direct_row_sum_max_matrix_numel is None
            else int(cuda_factored_nd_chunked_direct_row_sum_max_matrix_numel)
        )
        self.cuda_nonfactored_use_fp16_update = bool(cuda_nonfactored_use_fp16_update)
        if self.cuda_factored_nd_chunk_size_override is not None and self.cuda_factored_nd_chunk_size_override <= 0:
            raise ValueError("cuda_factored_nd_chunk_size_override must be positive when set")
        if (
            self.cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel is None
            and self.cuda_factored_nd_chunked_direct_row_sum_max_matrix_numel is None
        ):
            self.cuda_factored_nd_chunked_direct_row_sum = False
        else:
            self.cuda_factored_nd_chunked_direct_row_sum = True
        # The 8-bit implementations do not consume per-step parameter RMS internally.
        # Skipping the CUDA-side state update avoids an extra reduction per tensor.
        self.track_cuda_rms_state = False

    def _rms(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor) -> torch.Tensor:
        row_mean = exp_avg_sq_row.mean(dim=-1, keepdim=True).clamp_min_(1e-30)
        r_factor = (exp_avg_sq_row / row_mean).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).clamp_min_(1e-30).rsqrt()
        return torch.mul(r_factor, c_factor)

    def _quantize_into(self, state: dict, q_key: str, absmax_key: str, tensor: torch.Tensor, *, signed: bool, block_size: int) -> None:
        q, absmax = quantize_blockwise(tensor, signed=signed, block_size=block_size)
        state[q_key].copy_(q)
        state[absmax_key].copy_(absmax)

    def _dequantize(self, state: dict, q_key: str, absmax_key: str, *, signed: bool, block_size: int) -> torch.Tensor:
        return dequantize_blockwise(state[q_key], state[absmax_key], signed=signed, block_size=block_size)

    def _get_grad_fp32(self, state: dict, grad: torch.Tensor) -> torch.Tensor:
        if grad.dtype == torch.float32:
            return grad
        grad_fp32 = state.get("grad_fp32")
        if grad_fp32 is None or grad_fp32.shape != grad.shape or grad_fp32.device != grad.device:
            grad_fp32 = torch.empty_like(grad, dtype=torch.float32)
            state["grad_fp32"] = grad_fp32
        grad_fp32.copy_(grad)
        return grad_fp32

    def _get_cuda_factored_nd_chunk_size(self, matrix_numel: int) -> int:
        if matrix_numel >= self._ND_EXT_CHUNKED_LARGE_MATRIX_MIN_NUMEL:
            if self.cuda_factored_nd_chunk_size_override is not None:
                return self.cuda_factored_nd_chunk_size_override
            return self._ND_EXT_LARGE_PREFIX_MAX
        return 0

    def _should_use_cuda_factored_nd_chunked_direct_row_sum(self, matrix_numel: int) -> bool:
        if not self.cuda_factored_nd_chunked_direct_row_sum:
            return False
        min_numel = self.cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel
        max_numel = self.cuda_factored_nd_chunked_direct_row_sum_max_matrix_numel
        if min_numel is not None and matrix_numel < min_numel:
            return False
        if max_numel is not None and matrix_numel > max_numel:
            return False
        return True

    def _get_cuda_factored_nd_ext_strategy(self, grad: torch.Tensor) -> str:
        if not self.prefer_factored_cuda_ext_path:
            return self._ND_EXT_STRATEGY_NONE
        if not grad.is_cuda or grad.dim() <= 2 or not grad.is_contiguous():
            return self._ND_EXT_STRATEGY_NONE

        prefix_batch = math.prod(grad.shape[:-2])
        matrix_numel = grad.shape[-2] * grad.shape[-1]
        if prefix_batch <= self._ND_EXT_ALWAYS_PREFIX_MAX:
            return self._ND_EXT_STRATEGY_BATCHED
        if (
            prefix_batch <= self._ND_EXT_LARGE_PREFIX_MAX
            and matrix_numel >= self._ND_EXT_BATCHED_LARGE_MATRIX_MIN_NUMEL
        ):
            return self._ND_EXT_STRATEGY_BATCHED
        if prefix_batch > self._ND_EXT_LARGE_PREFIX_MAX and self._get_cuda_factored_nd_chunk_size(matrix_numel) > 0:
            return self._ND_EXT_STRATEGY_CHUNKED
        return self._ND_EXT_STRATEGY_NONE

    def _uses_cuda_factored_nd_batched_layout(self, state: dict) -> bool:
        return state.get("_cuda_factored_nd_ext_path", False) or state.get("_cuda_factored_nd_chunked_ext_path", False)

    def _bind_factored_views(self, state: dict) -> None:
        if self._uses_cuda_factored_nd_batched_layout(state):
            batch = state["prefix_batch"]
            rows = state["matrix_rows"]
            cols = state["matrix_cols"]
            row_absmax_blocks = state["factored_row_absmax"].shape[-1]
            col_absmax_blocks = state["factored_col_absmax"].shape[-1]

            factored_row_q = state["factored_row_q"].view(2, batch, rows)
            factored_col_q = state["factored_col_q"].view(2, batch, cols)
            factored_row_absmax = state["factored_row_absmax"].view(2, batch, row_absmax_blocks)
            factored_col_absmax = state["factored_col_absmax"].view(2, batch, col_absmax_blocks)

            state["exp_avg_sq_row_q"] = factored_row_q[0]
            state["exp_avg_res_row_q"] = factored_row_q[1]
            state["exp_avg_sq_col_q"] = factored_col_q[0]
            state["exp_avg_res_col_q"] = factored_col_q[1]

            state["exp_avg_sq_row_absmax"] = factored_row_absmax[0]
            state["exp_avg_res_row_absmax"] = factored_row_absmax[1]
            state["exp_avg_sq_col_absmax"] = factored_col_absmax[0]
            state["exp_avg_res_col_absmax"] = factored_col_absmax[1]

            state["exp_avg_q_batched"] = state["exp_avg_q"].view(batch, rows, cols)
            state["row_factor_batched"] = state["row_factor_fp32"].view(batch, rows)
            state["col_factor_batched"] = state["col_factor_fp32"].view(batch, cols)
        else:
            state["exp_avg_sq_row_q"] = state["factored_row_q"][0]
            state["exp_avg_res_row_q"] = state["factored_row_q"][1]
            state["exp_avg_sq_col_q"] = state["factored_col_q"][0]
            state["exp_avg_res_col_q"] = state["factored_col_q"][1]

            state["exp_avg_sq_row_absmax"] = state["factored_row_absmax"][0]
            state["exp_avg_res_row_absmax"] = state["factored_row_absmax"][1]
            state["exp_avg_sq_col_absmax"] = state["factored_col_absmax"][0]
            state["exp_avg_res_col_absmax"] = state["factored_col_absmax"][1]

        if state.get("_cuda_factored_fastpath", False) and "factored_row_fp32" in state and "factored_col_fp32" in state:
            state["exp_avg_sq_row_fp32"] = state["factored_row_fp32"][0]
            state["exp_avg_res_row_fp32"] = state["factored_row_fp32"][1]
            state["exp_avg_sq_col_fp32"] = state["factored_col_fp32"][0]
            state["exp_avg_res_col_fp32"] = state["factored_col_fp32"][1]

    def _step_param_factored_cuda_nd_ext(
        self,
        p: torch.Tensor,
        state: dict,
        grad: torch.Tensor,
        *,
        beta1: float,
        beta2: float,
        beta3: float,
        eps0: float,
        eps1: float,
        lr: float,
        clip_threshold: float,
        weight_decay: float,
        block_size: int,
    ) -> None:
        grad_fp32 = self._get_grad_fp32(state, grad)
        batch = state["prefix_batch"]
        rows = state["matrix_rows"]
        cols = state["matrix_cols"]

        grad_mats = grad_fp32.view(batch, rows, cols)
        param_mats = p.data.view(batch, rows, cols)
        exp_avg_q = state["exp_avg_q"].view(batch, rows, cols)
        exp_avg_absmax = state["exp_avg_absmax"]
        exp_avg = state["exp_avg_fp32"]
        scratch = state["scratch_fp32"]
        row_factor = state["row_factor_fp32"]
        col_factor = state["col_factor_fp32"]

        state["sum_update"].zero_()
        cuda_came_full_factored_sq_step_batched(
            g32=grad_mats,
            exp_avg_sq_row_q=state["exp_avg_sq_row_q"],
            exp_avg_sq_row_absmax=state["exp_avg_sq_row_absmax"],
            exp_avg_sq_col_q=state["exp_avg_sq_col_q"],
            exp_avg_sq_col_absmax=state["exp_avg_sq_col_absmax"],
            r_factor=row_factor,
            c_factor=col_factor,
            row_absmax_scratch=state["row_absmax_scratch"],
            reduce_partial=state["reduce_partial"],
            sum_row_state=state["sum_row_state"],
            sum_update_slice=state["sum_update_slice"],
            sum_update_total=state["sum_update"],
            beta2=beta2,
            eps0=eps0,
            block_size=block_size,
        )

        state["sum_update_equiv"].copy_(state["sum_update"]).div_(float(batch))
        cuda_came_full_factored_expavg_res_prepare_batched(
            g32=grad_mats,
            exp_avg_q=exp_avg_q,
            exp_avg_absmax=exp_avg_absmax,
            r_factor=row_factor,
            c_factor=col_factor,
            exp_avg_fp32=exp_avg,
            res32=scratch,
            sum_update=state["sum_update_equiv"],
            beta1=beta1,
            eps1=eps1,
            clip_threshold=clip_threshold,
            block_size=block_size,
        )

        cuda_came_full_factored_res_step_batched(
            res32=scratch,
            exp_avg_res_row_q=state["exp_avg_res_row_q"],
            exp_avg_res_row_absmax=state["exp_avg_res_row_absmax"],
            exp_avg_res_col_q=state["exp_avg_res_col_q"],
            exp_avg_res_col_absmax=state["exp_avg_res_col_absmax"],
            r_factor=row_factor,
            c_factor=col_factor,
            row_absmax_scratch=state["row_absmax_scratch"],
            reduce_partial=state["reduce_partial"],
            sum_row_state=state["sum_row_state"],
            beta3=beta3,
            block_size=block_size,
        )
        cuda_came_full_factored_param_update_batched(
            p=param_mats,
            exp_avg_fp32=exp_avg,
            r_factor=row_factor,
            c_factor=col_factor,
            lr=lr,
            weight_decay=weight_decay,
        )

    def _step_param_factored_cuda_nd_chunked_ext(
        self,
        p: torch.Tensor,
        state: dict,
        grad: torch.Tensor,
        *,
        beta1: float,
        beta2: float,
        beta3: float,
        eps0: float,
        eps1: float,
        lr: float,
        clip_threshold: float,
        weight_decay: float,
        block_size: int,
    ) -> None:
        grad_fp32 = self._get_grad_fp32(state, grad)
        batch = state["prefix_batch"]
        chunk_size = state["nd_chunk_size"]
        rows = state["matrix_rows"]
        cols = state["matrix_cols"]

        grad_mats = grad_fp32.view(batch, rows, cols)
        param_mats = p.data.view(batch, rows, cols)
        with torch.autograd.profiler.record_function("came_cuda.fact_nd_chunk"):
            cuda_came_full_factored_nd_chunked_step(
                p=param_mats,
                g32=grad_mats,
                exp_avg_q=state["exp_avg_q_batched"],
                exp_avg_absmax=state["exp_avg_absmax"],
                exp_avg_sq_row_q=state["exp_avg_sq_row_q"],
                exp_avg_sq_row_absmax=state["exp_avg_sq_row_absmax"],
                exp_avg_sq_col_q=state["exp_avg_sq_col_q"],
                exp_avg_sq_col_absmax=state["exp_avg_sq_col_absmax"],
                exp_avg_res_row_q=state["exp_avg_res_row_q"],
                exp_avg_res_row_absmax=state["exp_avg_res_row_absmax"],
                exp_avg_res_col_q=state["exp_avg_res_col_q"],
                exp_avg_res_col_absmax=state["exp_avg_res_col_absmax"],
                r_factor=state["row_factor_batched"],
                c_factor=state["col_factor_batched"],
                row_absmax_scratch=state["row_absmax_scratch"],
                reduce_partial=state["reduce_partial"],
                sum_row_state=state["sum_row_state"],
                sum_update_slice=state["sum_update_slice"],
                sum_update_chunk=state["sum_update_chunk"],
                sum_update_total=state["sum_update"],
                sum_update_equiv=state["sum_update_equiv"],
                exp_avg_fp32=state["exp_avg_fp32"],
                res32=state["scratch_fp32"],
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                eps0=eps0,
                eps1=eps1,
                lr=lr,
                clip_threshold=clip_threshold,
                weight_decay=weight_decay,
                chunk_size=chunk_size,
                block_size=block_size,
                direct_row_sum=bool(state.get("_cuda_factored_nd_chunked_direct_row_sum", False)),
            )

    def _cuda_dequantize_into(
        self,
        out: torch.Tensor,
        q_in: torch.Tensor,
        absmax: torch.Tensor,
        *,
        signed: bool,
        block_size: int,
    ) -> None:
        cuda_blockwise_dequantize_into(
            out=out,
            q_in=q_in,
            absmax=absmax,
            signed=signed,
            block_size=block_size,
        )

    def _cuda_quantize_into(
        self,
        src: torch.Tensor,
        q_out: torch.Tensor,
        absmax_out: torch.Tensor,
        *,
        signed: bool,
        block_size: int,
    ) -> None:
        cuda_blockwise_quantize_into(
            src=src,
            q_out=q_out,
            absmax_out=absmax_out,
            signed=signed,
            block_size=block_size,
        )

    def _cuda_dequantize_batched_into(
        self,
        out: torch.Tensor,
        q_in: torch.Tensor,
        absmax: torch.Tensor,
        *,
        signed: bool,
        block_size: int,
    ) -> None:
        cuda_blockwise_dequantize_batched_into(
            out=out,
            q_in=q_in,
            absmax=absmax,
            signed=signed,
            block_size=block_size,
        )

    def _cuda_quantize_batched_into(
        self,
        src: torch.Tensor,
        q_out: torch.Tensor,
        absmax_out: torch.Tensor,
        *,
        signed: bool,
        block_size: int,
    ) -> None:
        cuda_blockwise_quantize_batched_into(
            src=src,
            q_out=q_out,
            absmax_out=absmax_out,
            signed=signed,
            block_size=block_size,
        )

    def _fallback_timing_accumulator(self) -> tuple[dict[str, float] | None, dict[str, int] | None]:
        times = getattr(self, "_benchmark_fallback_timing_times", None)
        counts = getattr(self, "_benchmark_fallback_timing_counts", None)
        if not isinstance(times, dict) or not isinstance(counts, dict):
            return None, None
        return times, counts

    def _fallback_timing_add(
        self,
        *,
        key: str,
        start_time: float,
        count: int = 1,
    ) -> None:
        times, counts = self._fallback_timing_accumulator()
        if times is None or counts is None:
            return
        time_key = f"{key}_time_ms"
        count_key = f"{key}_count"
        times[time_key] = times.get(time_key, 0.0) + ((time.perf_counter() - start_time) * 1000.0)
        counts[count_key] = counts.get(count_key, 0) + count

    def _step_param_nonfactored_cuda(
        self,
        p: torch.Tensor,
        state: dict,
        grad: torch.Tensor,
        *,
        beta1: float,
        beta2: float,
        eps0: float,
        lr: float,
        clip_threshold: float,
        weight_decay: float,
        block_size: int,
    ) -> None:
        grad_fp32 = self._get_grad_fp32(state, grad)
        update_key = "update_fp32"
        step_fn = cuda_came_full_nonfactored_step
        if self.cuda_nonfactored_use_fp16_update:
            update_key = "_cuda_nonfactored_update_fp16"
            step_fn = cuda_came_full_nonfactored_step_fp16_update
        with torch.autograd.profiler.record_function("came_cuda.nonfactored_full"):
            step_fn(
                p=p.data,
                g32=grad_fp32,
                exp_avg_q=state["exp_avg_q"],
                exp_avg_absmax=state["exp_avg_absmax"],
                exp_avg_sq_q=state["exp_avg_sq_q"],
                exp_avg_sq_absmax=state["exp_avg_sq_absmax"],
                update=state[update_key],
                sum_update_partial=state["sum_update_partial"],
                sum_update=state["sum_update"],
                beta1=beta1,
                beta2=beta2,
                eps0=eps0,
                lr=lr,
                clip_threshold=clip_threshold,
                weight_decay=weight_decay,
                block_size=block_size,
            )

    def _step_param_nonfactored_cuda_direct(
        self,
        p: torch.Tensor,
        state: dict,
        grad: torch.Tensor,
        *,
        beta1: float,
        beta2: float,
        beta3: float,
        eps0: float,
        eps1: float,
        lr: float,
        clip_threshold: float,
        weight_decay: float,
        block_size: int,
    ) -> bool:
        if len(state) == 0 or state.get("_mode") != "full8bit":
            state.clear()
            self._init_state(state, grad, block_size)

        if state.get("factored", False) or not state.get("_cuda_nonfactored_fastpath", False):
            return False

        state["step"] += 1
        if self.track_cuda_rms_state or not p.is_cuda:
            state["RMS"] = self._rms(p.data)

        self._step_param_nonfactored_cuda(
            p,
            state,
            grad,
            beta1=beta1,
            beta2=beta2,
            eps0=eps0,
            lr=lr,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            block_size=block_size,
        )
        return True

    def _step_param_factored_cuda(
        self,
        p: torch.Tensor,
        state: dict,
        grad: torch.Tensor,
        *,
        beta1: float,
        beta2: float,
        beta3: float,
        eps0: float,
        eps1: float,
        lr: float,
        clip_threshold: float,
        weight_decay: float,
        block_size: int,
    ) -> None:
        if state.get("_cuda_factored_nd_chunked_ext_path", False):
            phase_start = time.perf_counter()
            self._step_param_factored_cuda_nd_chunked_ext(
                p,
                state,
                grad,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                eps0=eps0,
                eps1=eps1,
                lr=lr,
                clip_threshold=clip_threshold,
                weight_decay=weight_decay,
                block_size=block_size,
            )
            self._fallback_timing_add(key="fallback_factored_cuda_nd_chunked_ext", start_time=phase_start)
            return

        if state.get("_cuda_factored_nd_ext_path", False):
            phase_start = time.perf_counter()
            self._step_param_factored_cuda_nd_ext(
                p,
                state,
                grad,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                eps0=eps0,
                eps1=eps1,
                lr=lr,
                clip_threshold=clip_threshold,
                weight_decay=weight_decay,
                block_size=block_size,
            )
            self._fallback_timing_add(key="fallback_factored_cuda_nd_ext", start_time=phase_start)
            return

        phase_start = time.perf_counter()
        grad_fp32 = self._get_grad_fp32(state, grad)
        exp_avg = state["exp_avg_fp32"]
        scratch = state["scratch_fp32"]
        row_mean = state["row_mean_fp32"]
        row_factor = state["row_factor_fp32"]
        col_factor = state["col_factor_fp32"]
        exp_avg_sq_row = state["factored_row_fp32"][0]
        exp_avg_res_row = state["factored_row_fp32"][1]
        exp_avg_sq_col = state["factored_col_fp32"][0]
        exp_avg_res_col = state["factored_col_fp32"][1]

        if state.get("_cuda_factored_ext_path", False):
            cuda_came_full_factored_sq_step(
                g32=grad_fp32,
                exp_avg_sq_row_q=state["exp_avg_sq_row_q"],
                exp_avg_sq_row_absmax=state["exp_avg_sq_row_absmax"],
                exp_avg_sq_col_q=state["exp_avg_sq_col_q"],
                exp_avg_sq_col_absmax=state["exp_avg_sq_col_absmax"],
                r_factor=row_factor,
                c_factor=col_factor,
                row_absmax_scratch=state["row_absmax_scratch"],
                reduce_partial=state["reduce_partial"],
                sum_row_state=state["sum_row_state"],
                sum_update=state["sum_update"],
                beta2=beta2,
                eps0=eps0,
                block_size=block_size,
            )
            cuda_came_full_factored_expavg_res_prepare(
                g32=grad_fp32,
                exp_avg_q=state["exp_avg_q"],
                exp_avg_absmax=state["exp_avg_absmax"],
                r_factor=row_factor,
                c_factor=col_factor,
                exp_avg_fp32=exp_avg,
                res32=scratch,
                sum_update=state["sum_update"],
                beta1=beta1,
                eps1=eps1,
                clip_threshold=clip_threshold,
                block_size=block_size,
            )
            self._fallback_timing_add(key="fallback_factored_cuda_ext", start_time=phase_start)
        else:
            self._cuda_dequantize_into(exp_avg, state["exp_avg_q"], state["exp_avg_absmax"], signed=True, block_size=block_size)
            self._cuda_dequantize_batched_into(
                state["factored_row_fp32"],
                state["factored_row_q"],
                state["factored_row_absmax"],
                signed=False,
                block_size=block_size,
            )
            self._cuda_dequantize_batched_into(
                state["factored_col_fp32"],
                state["factored_col_q"],
                state["factored_col_absmax"],
                signed=False,
                block_size=block_size,
            )

            scratch.copy_(grad_fp32).mul_(scratch).add_(eps0)
            torch.sum(scratch, dim=-1, out=row_factor)
            row_factor.div_(grad_fp32.shape[-1])
            exp_avg_sq_row.mul_(beta2).add_(row_factor, alpha=1.0 - beta2)

            torch.sum(scratch, dim=-2, out=col_factor)
            col_factor.div_(grad_fp32.shape[-2])
            exp_avg_sq_col.mul_(beta2).add_(col_factor, alpha=1.0 - beta2)

            torch.sum(exp_avg_sq_row, dim=-1, keepdim=True, out=row_mean)
            row_mean.div_(exp_avg_sq_row.shape[-1]).clamp_min_(1e-30)
            row_factor.copy_(exp_avg_sq_row).div_(row_mean).clamp_min_(1e-30).rsqrt_()
            col_factor.copy_(exp_avg_sq_col).clamp_min_(1e-30).rsqrt_()

            scratch.copy_(grad_fp32).mul_(col_factor.unsqueeze(-2)).mul_(row_factor.unsqueeze(-1))
            scratch.div_((self._rms(scratch) / clip_threshold).clamp_(min=1.0))

            exp_avg.mul_(beta1).add_(scratch, alpha=1.0 - beta1)
            scratch.sub_(exp_avg).square_().add_(eps1)

        if state.get("_cuda_factored_ext_path", False):
            cuda_came_full_factored_res_step(
                res32=scratch,
                exp_avg_res_row_q=state["exp_avg_res_row_q"],
                exp_avg_res_row_absmax=state["exp_avg_res_row_absmax"],
                exp_avg_res_col_q=state["exp_avg_res_col_q"],
                exp_avg_res_col_absmax=state["exp_avg_res_col_absmax"],
                r_factor=row_factor,
                c_factor=col_factor,
                row_absmax_scratch=state["row_absmax_scratch"],
                reduce_partial=state["reduce_partial"],
                sum_row_state=state["sum_row_state"],
                beta3=beta3,
                block_size=block_size,
            )
        else:
            torch.sum(scratch, dim=-1, out=row_factor)
            row_factor.div_(grad_fp32.shape[-1])
            exp_avg_res_row.mul_(beta3).add_(row_factor, alpha=1.0 - beta3)

            torch.sum(scratch, dim=-2, out=col_factor)
            col_factor.div_(grad_fp32.shape[-2])
            exp_avg_res_col.mul_(beta3).add_(col_factor, alpha=1.0 - beta3)

            torch.sum(exp_avg_res_row, dim=-1, keepdim=True, out=row_mean)
            row_mean.div_(exp_avg_res_row.shape[-1]).clamp_min_(1e-30)
            row_factor.copy_(exp_avg_res_row).div_(row_mean).clamp_min_(1e-30).rsqrt_()
            col_factor.copy_(exp_avg_res_col).clamp_min_(1e-30).rsqrt_()

        if not state.get("_cuda_factored_ext_path", False):
            scratch.copy_(exp_avg).mul_(col_factor.unsqueeze(-2)).mul_(row_factor.unsqueeze(-1))
            if weight_decay != 0.0:
                p.data.add_(p.data, alpha=-weight_decay * lr)
            p.data.add_(scratch, alpha=-lr)

            self._cuda_quantize_into(exp_avg, state["exp_avg_q"], state["exp_avg_absmax"], signed=True, block_size=block_size)
            self._cuda_quantize_batched_into(
                state["factored_row_fp32"],
                state["factored_row_q"],
                state["factored_row_absmax"],
                signed=False,
                block_size=block_size,
            )
            self._cuda_quantize_batched_into(
                state["factored_col_fp32"],
                state["factored_col_q"],
                state["factored_col_absmax"],
                signed=False,
                block_size=block_size,
            )
            self._fallback_timing_add(key="fallback_factored_cuda_generic", start_time=phase_start)
        else:
            cuda_came_full_factored_param_update(
                p=p.data,
                exp_avg_fp32=exp_avg,
                r_factor=row_factor,
                c_factor=col_factor,
                lr=lr,
                weight_decay=weight_decay,
            )

    def _init_state(self, state: dict, grad: torch.Tensor, block_size: int) -> None:
        state["step"] = 0
        state["_mode"] = "full8bit"
        state["RMS"] = 0

        prefix_batch = math.prod(grad.shape[:-2]) if grad.dim() > 2 else 1
        nd_ext_strategy = self._get_cuda_factored_nd_ext_strategy(grad)
        exp_avg_q, exp_avg_absmax = init_qstate(grad.shape, device=grad.device, signed=True, block_size=block_size)
        state["exp_avg_q"] = exp_avg_q
        state["exp_avg_absmax"] = exp_avg_absmax

        factored = len(grad.shape) >= 2
        state["factored"] = factored
        if factored:
            row_shape = grad.shape[:-1]
            col_shape = grad.shape[:-2] + grad.shape[-1:]
            rows, cols = grad.shape[-2], grad.shape[-1]
            row_q, row_absmax = init_batched_qstate(
                row_shape,
                batch_size=2,
                device=grad.device,
                signed=False,
                block_size=block_size,
            )
            col_q, col_absmax = init_batched_qstate(
                col_shape,
                batch_size=2,
                device=grad.device,
                signed=False,
                block_size=block_size,
            )
            state["factored_row_q"] = row_q
            state["factored_row_absmax"] = row_absmax
            state["factored_col_q"] = col_q
            state["factored_col_absmax"] = col_absmax
            state["_cuda_factored_fastpath"] = grad.is_cuda
            state["_cuda_factored_nd_ext_path"] = nd_ext_strategy == self._ND_EXT_STRATEGY_BATCHED
            state["_cuda_factored_nd_chunked_ext_path"] = nd_ext_strategy == self._ND_EXT_STRATEGY_CHUNKED
            state["_cuda_factored_ext_path"] = self.prefer_factored_cuda_ext_path and grad.is_cuda and (
                grad.dim() == 2 or state["_cuda_factored_nd_ext_path"] or state["_cuda_factored_nd_chunked_ext_path"]
            )
            if state["_cuda_factored_fastpath"]:
                if self._uses_cuda_factored_nd_batched_layout(state):
                    prefix_batch = math.prod(grad.shape[:-2])
                    rows = grad.shape[-2]
                    cols = grad.shape[-1]
                    exp_avg_q_batched, exp_avg_absmax_batched = init_batched_qstate(
                        (rows, cols),
                        batch_size=prefix_batch,
                        device=grad.device,
                        signed=True,
                        block_size=block_size,
                    )
                    state["prefix_batch"] = prefix_batch
                    state["matrix_rows"] = rows
                    state["matrix_cols"] = cols
                    state["_cuda_factored_nd_chunked_direct_row_sum"] = (
                        state["_cuda_factored_nd_chunked_ext_path"]
                        and self._should_use_cuda_factored_nd_chunked_direct_row_sum(rows * cols)
                    )
                    state["exp_avg_q"] = exp_avg_q_batched.view_as(grad)
                    state["exp_avg_absmax"] = exp_avg_absmax_batched

                    row_q, row_absmax = init_batched_qstate(
                        (rows,),
                        batch_size=2 * prefix_batch,
                        device=grad.device,
                        signed=False,
                        block_size=block_size,
                    )
                    col_q, col_absmax = init_batched_qstate(
                        (cols,),
                        batch_size=2 * prefix_batch,
                        device=grad.device,
                        signed=False,
                        block_size=block_size,
                    )
                    state["factored_row_q"] = row_q
                    state["factored_row_absmax"] = row_absmax
                    state["factored_col_q"] = col_q
                    state["factored_col_absmax"] = col_absmax

                    state["grad_fp32"] = torch.empty_like(grad, dtype=torch.float32)
                    state["row_factor_fp32"] = torch.empty((prefix_batch, rows), device=grad.device, dtype=torch.float32)
                    state["col_factor_fp32"] = torch.empty((prefix_batch, cols), device=grad.device, dtype=torch.float32)
                    if state["_cuda_factored_nd_chunked_ext_path"]:
                        nd_chunk_size = min(prefix_batch, self._get_cuda_factored_nd_chunk_size(rows * cols))
                        state["nd_chunk_size"] = nd_chunk_size
                        state["exp_avg_fp32"] = torch.empty((nd_chunk_size, rows, cols), device=grad.device, dtype=torch.float32)
                        state["scratch_fp32"] = torch.empty((nd_chunk_size, rows, cols), device=grad.device, dtype=torch.float32)
                    else:
                        state["exp_avg_fp32"] = torch.empty((prefix_batch, rows, cols), device=grad.device, dtype=torch.float32)
                        state["scratch_fp32"] = torch.empty((prefix_batch, rows, cols), device=grad.device, dtype=torch.float32)
                        state["factored_row_fp32"] = torch.empty((2, prefix_batch, rows), device=grad.device, dtype=torch.float32)
                        state["factored_col_fp32"] = torch.empty((2, prefix_batch, cols), device=grad.device, dtype=torch.float32)
                        state["row_mean_fp32"] = torch.empty((prefix_batch, 1), device=grad.device, dtype=torch.float32)
                else:
                    row_mean_shape = row_shape[:-1] + (1,)
                    state["grad_fp32"] = torch.empty_like(grad, dtype=torch.float32)
                    state["exp_avg_fp32"] = torch.empty_like(grad, dtype=torch.float32)
                    state["scratch_fp32"] = torch.empty_like(grad, dtype=torch.float32)
                    state["factored_row_fp32"] = torch.empty((2, *row_shape), device=grad.device, dtype=torch.float32)
                    state["factored_col_fp32"] = torch.empty((2, *col_shape), device=grad.device, dtype=torch.float32)
                    state["row_factor_fp32"] = torch.empty(row_shape, device=grad.device, dtype=torch.float32)
                    state["col_factor_fp32"] = torch.empty(col_shape, device=grad.device, dtype=torch.float32)
                    state["row_mean_fp32"] = torch.empty(row_mean_shape, device=grad.device, dtype=torch.float32)
                if not state.get("_cuda_factored_nd_chunked_ext_path", False):
                    state["param_workspace"] = init_blockwise_workspace(
                        grad.shape,
                        device=grad.device,
                        block_size=block_size,
                    )
                    state["row_workspace"] = init_batched_blockwise_workspace(
                        row_shape,
                        batch_size=2,
                        device=grad.device,
                        block_size=block_size,
                    )
                    state["col_workspace"] = init_batched_blockwise_workspace(
                        col_shape,
                        batch_size=2,
                        device=grad.device,
                        block_size=block_size,
                    )
                if state["_cuda_factored_ext_path"]:
                    reduce_partial_size = max((rows + 255) // 256, (rows * cols + 255) // 256)
                    row_q_blocks = (rows + block_size - 1) // block_size
                    if state.get("_cuda_factored_nd_ext_path", False):
                        state["row_absmax_scratch"] = torch.empty(
                            (prefix_batch, row_q_blocks),
                            device=grad.device,
                            dtype=torch.float32,
                        )
                        state["reduce_partial"] = torch.empty(
                            (prefix_batch, reduce_partial_size),
                            device=grad.device,
                            dtype=torch.float32,
                        )
                        state["sum_row_state"] = torch.zeros((prefix_batch,), device=grad.device, dtype=torch.float32)
                    elif state.get("_cuda_factored_nd_chunked_ext_path", False):
                        nd_chunk_size = state["nd_chunk_size"]
                        state["row_absmax_scratch"] = torch.empty(
                            (nd_chunk_size, row_q_blocks),
                            device=grad.device,
                            dtype=torch.float32,
                        )
                        state["reduce_partial"] = torch.empty(
                            (nd_chunk_size, reduce_partial_size),
                            device=grad.device,
                            dtype=torch.float32,
                        )
                        state["sum_row_state"] = torch.zeros((nd_chunk_size,), device=grad.device, dtype=torch.float32)
                        state["sum_update_slice"] = torch.zeros((nd_chunk_size,), device=grad.device, dtype=torch.float32)
                        state["sum_update_chunk"] = torch.zeros((1,), device=grad.device, dtype=torch.float32)
                        state["sum_update_equiv"] = torch.zeros((1,), device=grad.device, dtype=torch.float32)
                    else:
                        state["row_absmax_scratch"] = torch.empty(
                            (row_q_blocks,),
                            device=grad.device,
                            dtype=torch.float32,
                        )
                        state["reduce_partial"] = torch.empty(
                            (reduce_partial_size,),
                            device=grad.device,
                            dtype=torch.float32,
                        )
                        state["sum_row_state"] = torch.zeros((1,), device=grad.device, dtype=torch.float32)
                    state["sum_update"] = torch.zeros((1,), device=grad.device, dtype=torch.float32)
                    if state.get("_cuda_factored_nd_ext_path", False):
                        state["sum_update_slice"] = torch.zeros((prefix_batch,), device=grad.device, dtype=torch.float32)
                        state["sum_update_equiv"] = torch.zeros((1,), device=grad.device, dtype=torch.float32)
            self._bind_factored_views(state)
        else:
            state["exp_avg_sq_q"], state["exp_avg_sq_absmax"] = init_qstate(
                grad.shape, device=grad.device, signed=False, block_size=block_size
            )
            state["_cuda_nonfactored_fastpath"] = grad.is_cuda
            if state["_cuda_nonfactored_fastpath"]:
                if grad.dtype != torch.float32:
                    state["grad_fp32"] = torch.empty_like(grad, dtype=torch.float32)
                if self.cuda_nonfactored_use_fp16_update:
                    state["_cuda_nonfactored_update_fp16"] = torch.empty_like(grad, dtype=torch.float16)
                else:
                    state["update_fp32"] = torch.empty_like(grad, dtype=torch.float32)
                state["sum_update_partial"] = torch.empty(
                    ((grad.numel() + block_size - 1) // block_size,),
                    device=grad.device,
                    dtype=torch.float32,
                )
                state["sum_update"] = torch.zeros((1,), device=grad.device, dtype=torch.float32)

    def _step_param(
        self,
        p: torch.Tensor,
        state: dict,
        grad: torch.Tensor,
        *,
        beta1: float,
        beta2: float,
        beta3: float,
        eps0: float,
        eps1: float,
        lr: float,
        clip_threshold: float,
        weight_decay: float,
        block_size: int,
    ) -> None:
        step_param_start = time.perf_counter()
        if len(state) == 0 or state.get("_mode") != "full8bit":
            phase_start = time.perf_counter()
            state.clear()
            self._init_state(state, grad, block_size)
            self._fallback_timing_add(key="fallback_init_state", start_time=phase_start)

        state["step"] += 1
        if self.track_cuda_rms_state or not p.is_cuda:
            phase_start = time.perf_counter()
            state["RMS"] = self._rms(p.data)
            self._fallback_timing_add(key="fallback_track_rms", start_time=phase_start)

        if state["factored"] and state.get("_cuda_factored_fastpath", False):
            phase_start = time.perf_counter()
            self._step_param_factored_cuda(
                p,
                state,
                grad,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                eps0=eps0,
                eps1=eps1,
                lr=lr,
                clip_threshold=clip_threshold,
                weight_decay=weight_decay,
                block_size=block_size,
            )
            self._fallback_timing_add(key="fallback_step_factored_cuda", start_time=phase_start)
            self._fallback_timing_add(key="fallback_step_param_total", start_time=step_param_start)
            return

        if not state["factored"] and state.get("_cuda_nonfactored_fastpath", False):
            phase_start = time.perf_counter()
            self._step_param_nonfactored_cuda(
                p,
                state,
                grad,
                beta1=beta1,
                beta2=beta2,
                eps0=eps0,
                lr=lr,
                clip_threshold=clip_threshold,
                weight_decay=weight_decay,
                block_size=block_size,
            )
            self._fallback_timing_add(key="fallback_step_nonfactored_cuda", start_time=phase_start)
            self._fallback_timing_add(key="fallback_step_param_total", start_time=step_param_start)
            return

        if grad.dtype in {torch.float16, torch.bfloat16}:
            grad = grad.float()

        exp_avg = self._dequantize(state, "exp_avg_q", "exp_avg_absmax", signed=True, block_size=block_size)

        update = (grad**2) + eps0
        if state["factored"]:
            exp_avg_sq_row = self._dequantize(
                state, "exp_avg_sq_row_q", "exp_avg_sq_row_absmax", signed=False, block_size=block_size
            )
            exp_avg_sq_col = self._dequantize(
                state, "exp_avg_sq_col_q", "exp_avg_sq_col_absmax", signed=False, block_size=block_size
            )

            exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
            exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)

            update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            update.mul_(grad)
        else:
            exp_avg_sq = self._dequantize(
                state, "exp_avg_sq_q", "exp_avg_sq_absmax", signed=False, block_size=block_size
            )
            exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
            update = exp_avg_sq.clamp_min_(1e-30).rsqrt().mul_(grad)

        update.div_((self._rms(update) / clip_threshold).clamp_(min=1.0))

        exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

        res = (update - exp_avg) ** 2 + eps1
        if state["factored"]:
            exp_avg_res_row = self._dequantize(
                state, "exp_avg_res_row_q", "exp_avg_res_row_absmax", signed=False, block_size=block_size
            )
            exp_avg_res_col = self._dequantize(
                state, "exp_avg_res_col_q", "exp_avg_res_col_absmax", signed=False, block_size=block_size
            )

            exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0 - beta3)
            exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0 - beta3)

            res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
            param_update = res_approx.mul_(exp_avg)
        else:
            param_update = exp_avg.clone()

        if weight_decay != 0.0:
            p.data.add_(p.data, alpha=-weight_decay * lr)

        p.data.add_(param_update, alpha=-lr)

        self._quantize_into(state, "exp_avg_q", "exp_avg_absmax", exp_avg, signed=True, block_size=block_size)
        if state["factored"]:
            self._quantize_into(
                state, "exp_avg_sq_row_q", "exp_avg_sq_row_absmax", exp_avg_sq_row, signed=False, block_size=block_size
            )
            self._quantize_into(
                state, "exp_avg_sq_col_q", "exp_avg_sq_col_absmax", exp_avg_sq_col, signed=False, block_size=block_size
            )
            self._quantize_into(
                state, "exp_avg_res_row_q", "exp_avg_res_row_absmax", exp_avg_res_row, signed=False, block_size=block_size
            )
            self._quantize_into(
                state, "exp_avg_res_col_q", "exp_avg_res_col_absmax", exp_avg_res_col, signed=False, block_size=block_size
            )
        else:
            self._quantize_into(
                state, "exp_avg_sq_q", "exp_avg_sq_absmax", exp_avg_sq, signed=False, block_size=block_size
            )
        self._fallback_timing_add(
            key="fallback_step_factored_cpu" if state["factored"] else "fallback_step_nonfactored_cpu",
            start_time=step_param_start,
        )
        self._fallback_timing_add(key="fallback_step_param_total", start_time=step_param_start)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            eps0, eps1 = group["eps"]
            lr = group["lr"]
            clip_threshold = group["clip_threshold"]
            weight_decay = group["weight_decay"]
            block_size = group["block_size"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                self._step_param(
                    p,
                    self.state[p],
                    p.grad.data,
                    beta1=beta1,
                    beta2=beta2,
                    beta3=beta3,
                    eps0=eps0,
                    eps1=eps1,
                    lr=lr,
                    clip_threshold=clip_threshold,
                    weight_decay=weight_decay,
                    block_size=block_size,
                )

        return loss
