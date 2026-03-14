from __future__ import annotations

import torch

try:
    from came_pytorch.came_cuda import came_fp_factored_step as cuda_came_fp_factored_step
except Exception:
    cuda_came_fp_factored_step = None

try:
    from came_pytorch.came_cuda import came_full_factored_param_update as cuda_came_full_factored_param_update
except Exception:
    cuda_came_full_factored_param_update = None


class CAMECUDA(torch.optim.Optimizer):
    """
    CUDA-oriented CAME variant with floating-point optimizer state.

    This mode separates CUDA-side execution from 8-bit state compression:
      - contiguous CUDA 2D tensors use a factored fp-state fast path
      - contiguous CUDA 1D tensors use a non-factored fp-state fast path
      - all other tensors use the same fp-state update rule without requiring
        quantized state or the CUDA extension
    """

    def __init__(
        self,
        params,
        lr: float,
        eps: tuple[float, float] = (1e-30, 1e-16),
        clip_threshold: float = 1.0,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
    ):
        if lr is None or lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if clip_threshold <= 0.0:
            raise ValueError(f"clip_threshold must be > 0, got {clip_threshold}")
        if len(eps) != 2:
            raise ValueError(f"eps must be a tuple of length 2, got eps={eps}")
        if len(betas) != 3:
            raise ValueError(f"betas must be a tuple of length 3, got betas={betas}")
        if not all(0.0 <= beta <= 1.0 for beta in betas):
            raise ValueError(f"betas must be in [0, 1], got betas={betas}")

        defaults = dict(
            lr=float(lr),
            eps=(float(eps[0]), float(eps[1])),
            clip_threshold=float(clip_threshold),
            betas=(float(betas[0]), float(betas[1]), float(betas[2])),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _rms(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor) -> torch.Tensor:
        row_mean = exp_avg_sq_row.mean(dim=-1, keepdim=True).clamp_min_(1e-30)
        r_factor = (exp_avg_sq_row / row_mean).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).clamp_min_(1e-30).rsqrt()
        return torch.mul(r_factor, c_factor)

    def _get_grad_fp32(self, state: dict, grad: torch.Tensor) -> torch.Tensor:
        if grad.dtype == torch.float32:
            return grad
        grad_fp32 = state.get("grad_fp32")
        if grad_fp32 is None or grad_fp32.shape != grad.shape or grad_fp32.device != grad.device:
            grad_fp32 = torch.empty_like(grad, dtype=torch.float32)
            state["grad_fp32"] = grad_fp32
        grad_fp32.copy_(grad)
        return grad_fp32

    def _init_state(self, p: torch.Tensor, state: dict, grad: torch.Tensor) -> None:
        state.clear()
        state["step"] = 0
        state["_mode"] = "cudafp"
        state["factored"] = grad.dim() >= 2

        grad_shape = tuple(grad.shape)
        state["exp_avg"] = torch.zeros(grad_shape, device=grad.device, dtype=torch.float32)

        if state["factored"]:
            row_shape = grad_shape[:-1]
            col_shape = grad_shape[:-2] + grad_shape[-1:]
            state["exp_avg_sq_row"] = torch.zeros(row_shape, device=grad.device, dtype=torch.float32)
            state["exp_avg_sq_col"] = torch.zeros(col_shape, device=grad.device, dtype=torch.float32)
            state["exp_avg_res_row"] = torch.zeros(row_shape, device=grad.device, dtype=torch.float32)
            state["exp_avg_res_col"] = torch.zeros(col_shape, device=grad.device, dtype=torch.float32)
        else:
            state["exp_avg_sq"] = torch.zeros(grad_shape, device=grad.device, dtype=torch.float32)

        use_cuda_factored_fastpath = (
            p.is_cuda
            and grad.dim() == 2
            and p.is_contiguous()
            and grad.is_contiguous()
            and grad.dtype == p.dtype
        )
        use_cuda_nonfactored_fastpath = (
            p.is_cuda
            and grad.dim() == 1
            and p.is_contiguous()
            and grad.is_contiguous()
            and grad.dtype == p.dtype
        )

        state["_cuda_fp_factored_fastpath"] = bool(state["factored"] and use_cuda_factored_fastpath)
        state["_cuda_fp_nonfactored_fastpath"] = bool((not state["factored"]) and use_cuda_nonfactored_fastpath)

        if state["_cuda_fp_factored_fastpath"]:
            rows = grad.shape[-2]
            row_shape = grad_shape[:-1]
            col_shape = grad_shape[:-2] + grad_shape[-1:]
            row_mean_shape = row_shape[:-1] + (1,)
            state["scratch_fp32"] = torch.empty_like(grad, dtype=torch.float32)
            state["row_factor_fp32"] = torch.empty(row_shape, device=grad.device, dtype=torch.float32)
            state["col_factor_fp32"] = torch.empty(col_shape, device=grad.device, dtype=torch.float32)
            state["row_mean_fp32"] = torch.empty(row_mean_shape, device=grad.device, dtype=torch.float32)
            state["sum_update"] = torch.zeros((1,), device=grad.device, dtype=torch.float32)
            reduce_partial_size = max((rows + 255) // 256, (grad.numel() + 255) // 256)
            state["reduce_partial"] = torch.empty((reduce_partial_size,), device=grad.device, dtype=torch.float32)
            state["sum_row_state"] = torch.zeros((1,), device=grad.device, dtype=torch.float32)
        elif state["_cuda_fp_nonfactored_fastpath"]:
            state["update_fp32"] = torch.empty_like(grad, dtype=torch.float32)
            state["sum_update"] = torch.zeros((1,), device=grad.device, dtype=torch.float32)
            state["sum_update_partial"] = torch.empty(
                ((grad.numel() + 255) // 256,),
                device=grad.device,
                dtype=torch.float32,
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
    ) -> None:
        grad_fp32 = self._get_grad_fp32(state, grad)
        exp_avg = state["exp_avg"]
        exp_avg_sq_row = state["exp_avg_sq_row"]
        exp_avg_sq_col = state["exp_avg_sq_col"]
        exp_avg_res_row = state["exp_avg_res_row"]
        exp_avg_res_col = state["exp_avg_res_col"]
        scratch = state["scratch_fp32"]
        row_factor = state["row_factor_fp32"]
        col_factor = state["col_factor_fp32"]

        if cuda_came_fp_factored_step is not None:
            cuda_came_fp_factored_step(
                p=p.data,
                g32=grad_fp32,
                exp_avg=exp_avg,
                exp_avg_sq_row=exp_avg_sq_row,
                exp_avg_sq_col=exp_avg_sq_col,
                exp_avg_res_row=exp_avg_res_row,
                exp_avg_res_col=exp_avg_res_col,
                r_factor=row_factor,
                c_factor=col_factor,
                scratch=scratch,
                reduce_partial=state["reduce_partial"],
                sum_row_state=state["sum_row_state"],
                sum_update=state["sum_update"],
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                eps0=eps0,
                eps1=eps1,
                lr=lr,
                clip_threshold=clip_threshold,
                weight_decay=weight_decay,
                block_size=256,
            )
            return

        row_mean = state["row_mean_fp32"]

        scratch.copy_(grad_fp32).mul_(grad_fp32).add_(eps0)

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

        if cuda_came_full_factored_param_update is not None:
            cuda_came_full_factored_param_update(
                p=p.data,
                exp_avg_fp32=exp_avg,
                r_factor=row_factor,
                c_factor=col_factor,
                lr=lr,
                weight_decay=weight_decay,
            )
            return

        scratch.copy_(exp_avg).mul_(col_factor.unsqueeze(-2)).mul_(row_factor.unsqueeze(-1))
        if weight_decay != 0.0:
            p.data.add_(p.data, alpha=-weight_decay * lr)
        p.data.add_(scratch, alpha=-lr)

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
    ) -> None:
        grad_fp32 = self._get_grad_fp32(state, grad)
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        update = state["update_fp32"]

        update.copy_(grad_fp32).mul_(grad_fp32).add_(eps0)
        exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
        update.copy_(exp_avg_sq).clamp_min_(1e-30).rsqrt_().mul_(grad_fp32)
        update.div_((self._rms(update) / clip_threshold).clamp_(min=1.0))
        exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

        if weight_decay != 0.0:
            p.data.add_(p.data, alpha=-weight_decay * lr)
        p.data.add_(exp_avg, alpha=-lr)

    def _step_param_generic(
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
    ) -> None:
        if grad.dtype in {torch.float16, torch.bfloat16}:
            grad = grad.float()

        exp_avg = state["exp_avg"]
        update = (grad**2) + eps0
        if state["factored"]:
            exp_avg_sq_row = state["exp_avg_sq_row"]
            exp_avg_sq_col = state["exp_avg_sq_col"]

            exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
            exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)

            update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
            update.mul_(grad)
        else:
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
            update = exp_avg_sq.clamp_min_(1e-30).rsqrt().mul_(grad)

        update.div_((self._rms(update) / clip_threshold).clamp_(min=1.0))
        exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

        if state["factored"]:
            exp_avg_res_row = state["exp_avg_res_row"]
            exp_avg_res_col = state["exp_avg_res_col"]
            res = (update - exp_avg) ** 2 + eps1

            exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0 - beta3)
            exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0 - beta3)

            param_update = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col).mul_(exp_avg)
        else:
            param_update = exp_avg

        if weight_decay != 0.0:
            p.data.add_(p.data, alpha=-weight_decay * lr)
        p.data.add_(param_update, alpha=-lr)

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
    ) -> None:
        if len(state) == 0 or state.get("_mode") != "cudafp":
            self._init_state(p, state, grad)

        state["step"] += 1

        if state.get("_cuda_fp_factored_fastpath", False):
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
            )
            return

        if state.get("_cuda_fp_nonfactored_fastpath", False):
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
            )
            return

        self._step_param_generic(
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
                )

        return loss
