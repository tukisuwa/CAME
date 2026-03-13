from __future__ import annotations

import torch

from came_pytorch.CAME8bitFull import CAME8bitFull
from came_pytorch.blockwise_quantization import init_batched_qstate, init_qstate


class CAME8bitMemory(CAME8bitFull):
    """
    Compact 8-bit CAME variant that avoids the speed-first persistent helper
    buffers used by the current CUDA-oriented 8-bit paths.
    """

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
            state["_cuda_factored_fastpath"] = False
            state["_cuda_factored_nd_ext_path"] = False
            state["_cuda_factored_nd_chunked_ext_path"] = False
            state["_cuda_factored_ext_path"] = False
            state["exp_avg_sq_row_q"] = state["factored_row_q"][0]
            state["exp_avg_res_row_q"] = state["factored_row_q"][1]
            state["exp_avg_sq_col_q"] = state["factored_col_q"][0]
            state["exp_avg_res_col_q"] = state["factored_col_q"][1]
            state["exp_avg_sq_row_absmax"] = state["factored_row_absmax"][0]
            state["exp_avg_res_row_absmax"] = state["factored_row_absmax"][1]
            state["exp_avg_sq_col_absmax"] = state["factored_col_absmax"][0]
            state["exp_avg_res_col_absmax"] = state["factored_col_absmax"][1]
            return

        state["exp_avg_sq_q"], state["exp_avg_sq_absmax"] = init_qstate(
            grad.shape,
            device=grad.device,
            signed=False,
            block_size=block_size,
        )
        state["_cuda_nonfactored_fastpath"] = False
