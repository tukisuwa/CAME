from __future__ import annotations

import torch

from came_pytorch.CAME8bitFull import CAME8bitFull
from came_pytorch.blockwise_quantization import init_batched_qstate, init_qstate
from came_pytorch.came_cuda import (
    came_full_factored_expavg_res_prepare as cuda_came_full_factored_expavg_res_prepare,
    came_full_factored_param_update as cuda_came_full_factored_param_update,
    came_full_factored_res_step as cuda_came_full_factored_res_step,
    came_full_factored_sq_step as cuda_came_full_factored_sq_step,
    came_full_factored_step_multitensor_same_shape_ptrs as cuda_came_full_factored_step_multitensor_same_shape_ptrs,
    came_full_nonfactored_step_fp16_update as cuda_came_full_nonfactored_step_fp16_update,
)


class CAME8bitMemory(CAME8bitFull):
    """
    Compact 8-bit CAME variant.

    This mode keeps per-parameter optimizer state quantized, but reuses a small
    set of per-device shared CUDA scratch buffers for the common 2D and 1D
    cases so it does not fall all the way back to the very slow generic path.
    """

    _CUDA_FACTORED_GROUP_MAX_MATRIX_NUMEL = 64 * 1024
    _CUDA_FACTORED_GROUP_CHUNK_SIZE = 32

    def __init__(
        self,
        params,
        lr: float,
        eps: tuple[float, float] = (1e-30, 1e-16),
        clip_threshold: float = 1.0,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        block_size: int = 256,
    ):
        super().__init__(
            params,
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
            block_size=block_size,
            prefer_factored_cuda_ext_path=False,
            cuda_nonfactored_use_fp16_update=False,
        )
        self._shared_cuda_buffers: dict[tuple[object, ...], torch.Tensor] = {}
        self._cuda_factored_multitensor_ptr_cache: dict[tuple[object, ...], dict[str, torch.Tensor]] = {}

    def _get_shared_buffer(
        self,
        *,
        kind: str,
        device: torch.device,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        zero: bool = False,
    ) -> torch.Tensor:
        key = (kind, device.type, device.index, shape, dtype)
        buf = self._shared_cuda_buffers.get(key)
        if buf is None:
            factory = torch.zeros if zero else torch.empty
            buf = factory(shape, device=device, dtype=dtype)
            self._shared_cuda_buffers[key] = buf
        elif zero:
            buf.zero_()
        return buf

    def _make_device_ptr_tensor(self, tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        return torch.tensor([int(tensor.data_ptr()) for tensor in tensors], device=device, dtype=torch.int64)

    def _get_shared_grad_fp32(self, grad: torch.Tensor) -> torch.Tensor:
        if grad.dtype == torch.float32:
            return grad
        grad_fp32 = self._get_shared_buffer(
            kind="grad_fp32",
            device=grad.device,
            shape=tuple(grad.shape),
            dtype=torch.float32,
        )
        grad_fp32.copy_(grad)
        return grad_fp32

    def _get_shared_factored_cuda_buffers(self, grad: torch.Tensor, block_size: int) -> dict[str, torch.Tensor]:
        rows, cols = grad.shape[-2], grad.shape[-1]
        reduce_partial_size = max((rows + 255) // 256, (rows * cols + 255) // 256)
        row_q_blocks = (rows + block_size - 1) // block_size
        device = grad.device
        shape = tuple(grad.shape)
        return {
            "exp_avg_fp32": self._get_shared_buffer(
                kind="fact_exp_avg_fp32",
                device=device,
                shape=shape,
                dtype=torch.float32,
            ),
            "scratch_fp32": self._get_shared_buffer(
                kind="fact_scratch_fp32",
                device=device,
                shape=shape,
                dtype=torch.float32,
            ),
            "row_factor_fp32": self._get_shared_buffer(
                kind="fact_row_factor_fp32",
                device=device,
                shape=(rows,),
                dtype=torch.float32,
            ),
            "col_factor_fp32": self._get_shared_buffer(
                kind="fact_col_factor_fp32",
                device=device,
                shape=(cols,),
                dtype=torch.float32,
            ),
            "row_absmax_scratch": self._get_shared_buffer(
                kind="fact_row_absmax_scratch",
                device=device,
                shape=(row_q_blocks,),
                dtype=torch.float32,
            ),
            "reduce_partial": self._get_shared_buffer(
                kind="fact_reduce_partial",
                device=device,
                shape=(reduce_partial_size,),
                dtype=torch.float32,
            ),
            "sum_row_state": self._get_shared_buffer(
                kind="fact_sum_row_state",
                device=device,
                shape=(1,),
                dtype=torch.float32,
                zero=True,
            ),
            "sum_update": self._get_shared_buffer(
                kind="fact_sum_update",
                device=device,
                shape=(1,),
                dtype=torch.float32,
                zero=True,
            ),
        }

    def _get_shared_nonfactored_cuda_buffers(self, grad: torch.Tensor, block_size: int) -> dict[str, torch.Tensor]:
        device = grad.device
        numel = grad.numel()
        return {
            "update_fp16": self._get_shared_buffer(
                kind="nonfact_update_fp16",
                device=device,
                shape=tuple(grad.shape),
                dtype=torch.float16,
            ),
            "sum_update_partial": self._get_shared_buffer(
                kind="nonfact_sum_update_partial",
                device=device,
                shape=((numel + block_size - 1) // block_size,),
                dtype=torch.float32,
            ),
            "sum_update": self._get_shared_buffer(
                kind="nonfact_sum_update",
                device=device,
                shape=(1,),
                dtype=torch.float32,
                zero=True,
            ),
        }

    def _get_shared_factored_group_buffers(
        self,
        *,
        device: torch.device,
        batch_size: int,
        rows: int,
        cols: int,
        block_size: int,
    ) -> dict[str, torch.Tensor]:
        per_item_numel = rows * cols
        q_blocks = (per_item_numel + block_size - 1) // block_size
        row_q_blocks = (rows + block_size - 1) // block_size
        reduce_partial_size = max((rows + 255) // 256, (per_item_numel + 255) // 256)
        return {
            "g32": self._get_shared_buffer(
                kind="fact_group_g32",
                device=device,
                shape=(batch_size, rows, cols),
                dtype=torch.float32,
            ),
            "row_factor": self._get_shared_buffer(
                kind="fact_group_row_factor",
                device=device,
                shape=(batch_size, rows),
                dtype=torch.float32,
            ),
            "col_factor": self._get_shared_buffer(
                kind="fact_group_col_factor",
                device=device,
                shape=(batch_size, cols),
                dtype=torch.float32,
            ),
            "row_absmax_scratch": self._get_shared_buffer(
                kind="fact_group_row_absmax_scratch",
                device=device,
                shape=(batch_size, row_q_blocks),
                dtype=torch.float32,
            ),
            "reduce_partial": self._get_shared_buffer(
                kind="fact_group_reduce_partial",
                device=device,
                shape=(batch_size, reduce_partial_size),
                dtype=torch.float32,
            ),
            "sum_row_state": self._get_shared_buffer(
                kind="fact_group_sum_row_state",
                device=device,
                shape=(batch_size,),
                dtype=torch.float32,
                zero=True,
            ),
            "sum_update_slice": self._get_shared_buffer(
                kind="fact_group_sum_update_slice",
                device=device,
                shape=(batch_size,),
                dtype=torch.float32,
                zero=True,
            ),
            "sum_update_total": self._get_shared_buffer(
                kind="fact_group_sum_update_total",
                device=device,
                shape=(1,),
                dtype=torch.float32,
                zero=True,
            ),
            "sum_update_equiv": self._get_shared_buffer(
                kind="fact_group_sum_update_equiv",
                device=device,
                shape=(1,),
                dtype=torch.float32,
                zero=True,
            ),
            "exp_avg_fp32": self._get_shared_buffer(
                kind="fact_group_exp_avg_fp32",
                device=device,
                shape=(batch_size, rows, cols),
                dtype=torch.float32,
            ),
            "res32": self._get_shared_buffer(
                kind="fact_group_res32",
                device=device,
                shape=(batch_size, rows, cols),
                dtype=torch.float32,
            ),
        }

    def _should_group_factored_cuda_params(
        self,
        p: torch.Tensor,
        state: dict,
        grad: torch.Tensor,
        block_size: int,
    ) -> bool:
        if len(state) == 0 or state.get("_mode") != "full8bit":
            state.clear()
            self._init_state(state, grad, block_size)
        return bool(
            state.get("factored", False)
            and state.get("_cuda_factored_fastpath", False)
            and grad.dim() == 2
            and grad.numel() <= self._CUDA_FACTORED_GROUP_MAX_MATRIX_NUMEL
            and p.is_contiguous()
            and grad.is_contiguous()
        )

    def _get_factored_multitensor_static_ptrs(
        self,
        chunk: list[tuple[torch.Tensor, dict, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        sample_p = chunk[0][0].data
        key = (
            sample_p.device,
            sample_p.dtype,
            tuple(sample_p.shape),
            tuple(
                (
                    int(p.data_ptr()),
                    int(state["exp_avg_q"].data_ptr()),
                    int(state["exp_avg_absmax"].data_ptr()),
                    int(state["exp_avg_sq_row_q"].data_ptr()),
                    int(state["exp_avg_sq_row_absmax"].data_ptr()),
                    int(state["exp_avg_sq_col_q"].data_ptr()),
                    int(state["exp_avg_sq_col_absmax"].data_ptr()),
                    int(state["exp_avg_res_row_q"].data_ptr()),
                    int(state["exp_avg_res_row_absmax"].data_ptr()),
                    int(state["exp_avg_res_col_q"].data_ptr()),
                    int(state["exp_avg_res_col_absmax"].data_ptr()),
                )
                for p, state, _ in chunk
            ),
        )
        cached = self._cuda_factored_multitensor_ptr_cache.get(key)
        if cached is not None:
            return cached

        cached = {
            "p_ptrs": self._make_device_ptr_tensor([p.data for p, _, _ in chunk], sample_p.device),
            "exp_avg_q_ptrs": self._make_device_ptr_tensor([state["exp_avg_q"] for _, state, _ in chunk], sample_p.device),
            "exp_avg_absmax_ptrs": self._make_device_ptr_tensor([state["exp_avg_absmax"] for _, state, _ in chunk], sample_p.device),
            "exp_avg_sq_row_q_ptrs": self._make_device_ptr_tensor([state["exp_avg_sq_row_q"] for _, state, _ in chunk], sample_p.device),
            "exp_avg_sq_row_absmax_ptrs": self._make_device_ptr_tensor([state["exp_avg_sq_row_absmax"] for _, state, _ in chunk], sample_p.device),
            "exp_avg_sq_col_q_ptrs": self._make_device_ptr_tensor([state["exp_avg_sq_col_q"] for _, state, _ in chunk], sample_p.device),
            "exp_avg_sq_col_absmax_ptrs": self._make_device_ptr_tensor([state["exp_avg_sq_col_absmax"] for _, state, _ in chunk], sample_p.device),
            "exp_avg_res_row_q_ptrs": self._make_device_ptr_tensor([state["exp_avg_res_row_q"] for _, state, _ in chunk], sample_p.device),
            "exp_avg_res_row_absmax_ptrs": self._make_device_ptr_tensor([state["exp_avg_res_row_absmax"] for _, state, _ in chunk], sample_p.device),
            "exp_avg_res_col_q_ptrs": self._make_device_ptr_tensor([state["exp_avg_res_col_q"] for _, state, _ in chunk], sample_p.device),
            "exp_avg_res_col_absmax_ptrs": self._make_device_ptr_tensor([state["exp_avg_res_col_absmax"] for _, state, _ in chunk], sample_p.device),
        }
        self._cuda_factored_multitensor_ptr_cache[key] = cached
        return cached

    def _step_param_factored_cuda_group_multitensor(
        self,
        items: list[tuple[torch.Tensor, dict, torch.Tensor]],
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
        rows, cols = items[0][0].shape
        for start in range(0, len(items), self._CUDA_FACTORED_GROUP_CHUNK_SIZE):
            chunk = items[start : start + self._CUDA_FACTORED_GROUP_CHUNK_SIZE]
            active = len(chunk)
            shared = self._get_shared_factored_group_buffers(
                device=chunk[0][0].device,
                batch_size=active,
                rows=rows,
                cols=cols,
                block_size=block_size,
            )
            for idx, (p, state, grad) in enumerate(chunk):
                state["step"] += 1
                if self.track_cuda_rms_state or not p.is_cuda:
                    state["RMS"] = self._rms(p.data)
                state["_cuda_factored_group_multitensor_fastpath"] = True
                shared["g32"][idx].copy_(grad if grad.dtype == torch.float32 else grad.float())

            static_ptrs = self._get_factored_multitensor_static_ptrs(chunk)
            cuda_came_full_factored_step_multitensor_same_shape_ptrs(
                sample_p=chunk[0][0].data,
                g32=shared["g32"][:active],
                p_ptrs=static_ptrs["p_ptrs"][:active],
                exp_avg_q_ptrs=static_ptrs["exp_avg_q_ptrs"][:active],
                exp_avg_absmax_ptrs=static_ptrs["exp_avg_absmax_ptrs"][:active],
                exp_avg_sq_row_q_ptrs=static_ptrs["exp_avg_sq_row_q_ptrs"][:active],
                exp_avg_sq_row_absmax_ptrs=static_ptrs["exp_avg_sq_row_absmax_ptrs"][:active],
                exp_avg_sq_col_q_ptrs=static_ptrs["exp_avg_sq_col_q_ptrs"][:active],
                exp_avg_sq_col_absmax_ptrs=static_ptrs["exp_avg_sq_col_absmax_ptrs"][:active],
                exp_avg_res_row_q_ptrs=static_ptrs["exp_avg_res_row_q_ptrs"][:active],
                exp_avg_res_row_absmax_ptrs=static_ptrs["exp_avg_res_row_absmax_ptrs"][:active],
                exp_avg_res_col_q_ptrs=static_ptrs["exp_avg_res_col_q_ptrs"][:active],
                exp_avg_res_col_absmax_ptrs=static_ptrs["exp_avg_res_col_absmax_ptrs"][:active],
                row_factor=shared["row_factor"][:active],
                c_factor=shared["col_factor"][:active],
                row_absmax_scratch=shared["row_absmax_scratch"][:active],
                reduce_partial=shared["reduce_partial"][:active],
                sum_row_state=shared["sum_row_state"][:active],
                sum_update_slice=shared["sum_update_slice"][:active],
                sum_update_total=shared["sum_update_total"],
                sum_update_equiv=shared["sum_update_equiv"],
                exp_avg_fp32=shared["exp_avg_fp32"][:active],
                res32=shared["res32"][:active],
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

    def _init_state(self, state: dict, grad: torch.Tensor, block_size: int) -> None:
        state["step"] = 0
        state["_mode"] = "full8bit"
        state["RMS"] = 0

        exp_avg_q, exp_avg_absmax = init_qstate(grad.shape, device=grad.device, signed=True, block_size=block_size)
        state["exp_avg_q"] = exp_avg_q
        state["exp_avg_absmax"] = exp_avg_absmax

        factored = len(grad.shape) >= 2
        state["factored"] = factored

        if factored:
            row_shape = grad.shape[:-1]
            col_shape = grad.shape[:-2] + grad.shape[-1:]
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
            state["exp_avg_sq_row_q"] = row_q[0]
            state["exp_avg_res_row_q"] = row_q[1]
            state["exp_avg_sq_col_q"] = col_q[0]
            state["exp_avg_res_col_q"] = col_q[1]
            state["exp_avg_sq_row_absmax"] = row_absmax[0]
            state["exp_avg_res_row_absmax"] = row_absmax[1]
            state["exp_avg_sq_col_absmax"] = col_absmax[0]
            state["exp_avg_res_col_absmax"] = col_absmax[1]
            state["_cuda_factored_fastpath"] = bool(
                grad.is_cuda and grad.dim() == 2 and grad.is_contiguous()
            )
            state["_cuda_factored_nd_ext_path"] = False
            state["_cuda_factored_nd_chunked_ext_path"] = False
            state["_cuda_factored_ext_path"] = state["_cuda_factored_fastpath"]
            return

        state["exp_avg_sq_q"], state["exp_avg_sq_absmax"] = init_qstate(
            grad.shape,
            device=grad.device,
            signed=False,
            block_size=block_size,
        )
        state["_cuda_nonfactored_fastpath"] = bool(
            grad.is_cuda and grad.dim() == 1 and grad.is_contiguous()
        )

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
        grad_fp32 = self._get_shared_grad_fp32(grad)
        shared = self._get_shared_factored_cuda_buffers(grad_fp32, block_size)

        cuda_came_full_factored_sq_step(
            g32=grad_fp32,
            exp_avg_sq_row_q=state["exp_avg_sq_row_q"],
            exp_avg_sq_row_absmax=state["exp_avg_sq_row_absmax"],
            exp_avg_sq_col_q=state["exp_avg_sq_col_q"],
            exp_avg_sq_col_absmax=state["exp_avg_sq_col_absmax"],
            r_factor=shared["row_factor_fp32"],
            c_factor=shared["col_factor_fp32"],
            row_absmax_scratch=shared["row_absmax_scratch"],
            reduce_partial=shared["reduce_partial"],
            sum_row_state=shared["sum_row_state"],
            sum_update=shared["sum_update"],
            beta2=beta2,
            eps0=eps0,
            block_size=block_size,
        )

        cuda_came_full_factored_expavg_res_prepare(
            g32=grad_fp32,
            exp_avg_q=state["exp_avg_q"],
            exp_avg_absmax=state["exp_avg_absmax"],
            r_factor=shared["row_factor_fp32"],
            c_factor=shared["col_factor_fp32"],
            exp_avg_fp32=shared["exp_avg_fp32"],
            res32=shared["scratch_fp32"],
            sum_update=shared["sum_update"],
            beta1=beta1,
            eps1=eps1,
            clip_threshold=clip_threshold,
            block_size=block_size,
        )

        cuda_came_full_factored_res_step(
            res32=shared["scratch_fp32"],
            exp_avg_res_row_q=state["exp_avg_res_row_q"],
            exp_avg_res_row_absmax=state["exp_avg_res_row_absmax"],
            exp_avg_res_col_q=state["exp_avg_res_col_q"],
            exp_avg_res_col_absmax=state["exp_avg_res_col_absmax"],
            r_factor=shared["row_factor_fp32"],
            c_factor=shared["col_factor_fp32"],
            row_absmax_scratch=shared["row_absmax_scratch"],
            reduce_partial=shared["reduce_partial"],
            sum_row_state=shared["sum_row_state"],
            beta3=beta3,
            block_size=block_size,
        )

        cuda_came_full_factored_param_update(
            p=p.data,
            exp_avg_fp32=shared["exp_avg_fp32"],
            r_factor=shared["row_factor_fp32"],
            c_factor=shared["col_factor_fp32"],
            lr=lr,
            weight_decay=weight_decay,
        )

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
        grad_fp32 = self._get_shared_grad_fp32(grad)
        shared = self._get_shared_nonfactored_cuda_buffers(grad_fp32, block_size)
        cuda_came_full_nonfactored_step_fp16_update(
            p=p.data,
            g32=grad_fp32,
            exp_avg_q=state["exp_avg_q"],
            exp_avg_absmax=state["exp_avg_absmax"],
            exp_avg_sq_q=state["exp_avg_sq_q"],
            exp_avg_sq_absmax=state["exp_avg_sq_absmax"],
            update=shared["update_fp16"],
            sum_update_partial=shared["sum_update_partial"],
            sum_update=shared["sum_update"],
            beta1=beta1,
            beta2=beta2,
            eps0=eps0,
            lr=lr,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            block_size=block_size,
        )

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

            factored_buckets: dict[tuple[object, ...], list[tuple[torch.Tensor, dict, torch.Tensor]]] = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                grad = p.grad.data
                state = self.state[p]
                if self._should_group_factored_cuda_params(p, state, grad, block_size):
                    key = (p.device, p.dtype, tuple(grad.shape), block_size)
                    factored_buckets.setdefault(key, []).append((p, state, grad))
                    continue

                self._step_param(
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

            for bucket in factored_buckets.values():
                if len(bucket) < 2:
                    p, state, grad = bucket[0]
                    self._step_param(
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
                    continue

                self._step_param_factored_cuda_group_multitensor(
                    bucket,
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
