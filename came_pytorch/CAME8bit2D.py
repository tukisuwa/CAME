from __future__ import annotations

import torch
import torch.optim

from came_pytorch.came8bit2d_state import init_state, prepare_step_tensors, step_2d, validate_state_shape


class CAME8bit2D(torch.optim.Optimizer):
    """
    CAME optimizer (2D-only) with int8 blockwise exp_avg on CUDA.

    Notes:
        - Only supports CUDA tensors and 2D parameters.
        - Persistent optimizer state is blockwise-quantized.
        - The CUDA kernel still computes on temporary fp32 row/col stats.
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
        cuda_graph_compatible: bool = False,
    ):
        if lr is None or lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if clip_threshold <= 0.0:
            raise ValueError(f"clip_threshold must be > 0, got {clip_threshold}")
        if len(eps) != 2:
            raise ValueError(f"eps must be a tuple of length 2, got eps={eps}")
        if len(betas) != 3:
            raise ValueError(f"betas must be a tuple of length 3, got betas={betas}")
        if block_size != 256:
            raise ValueError("Only block_size=256 is supported in this minimal implementation.")

        defaults = dict(
            lr=float(lr),
            eps=(float(eps[0]), float(eps[1])),
            clip_threshold=float(clip_threshold),
            betas=(float(betas[0]), float(betas[1]), float(betas[2])),
            weight_decay=float(weight_decay),
            block_size=int(block_size),
        )
        super().__init__(params, defaults)
        self.cuda_graph_compatible = bool(cuda_graph_compatible)

    @torch.no_grad()
    def prime_for_cuda_graph(self) -> None:
        for group in self.param_groups:
            block_size = group["block_size"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dim() != 2:
                    raise RuntimeError("CUDA graph capture for CAME8bit2D requires 2D parameters.")
                if not p.is_cuda:
                    raise RuntimeError("CUDA graph capture for CAME8bit2D requires CUDA parameters.")
                if p.grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")
                if p.grad.dtype != p.dtype:
                    raise RuntimeError(f"Expected g.dtype == p.dtype, got {p.grad.dtype} vs {p.dtype}")

                p_data, _, _ = prepare_step_tensors(p, p.grad, require_contiguous=True)
                state = self.state[p]
                if len(state) == 0:
                    init_state(state, p_data, block_size=block_size)
                validate_state_shape(state, p_data)

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
                if p.dim() != 2:
                    raise NotImplementedError("CAME8bit2D currently supports 2D parameters only.")
                if not p.is_cuda:
                    raise NotImplementedError("CAME8bit2D requires CUDA parameters.")
                if p.grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                g = p.grad
                if g.dtype != p.dtype:
                    raise RuntimeError(f"Expected g.dtype == p.dtype, got {g.dtype} vs {p.dtype}")

                p_data, g_data, needs_param_copyback = prepare_step_tensors(
                    p, g, require_contiguous=self.cuda_graph_compatible
                )

                state = self.state[p]
                if len(state) == 0:
                    init_state(state, p_data, block_size=block_size)

                validate_state_shape(state, p_data)

                step_2d(
                    state=state,
                    p_data=p_data,
                    g_data=g_data,
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

                if needs_param_copyback:
                    p.data.copy_(p_data)

        return loss
