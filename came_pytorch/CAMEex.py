import math

import torch
import torch.optim


class CAMEex(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
        optimizer_dtype=torch.bfloat16,
        cpu_offload=False,
    ):
        if lr is not None and lr <= 0.0:
             raise ValueError(f"Invalid learning rate: {lr}, if set.")
        assert all([0. <= beta <= 1. for beta in betas])

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
            optimizer_dtype=optimizer_dtype,
            cpu_offload=cpu_offload,
        )
        super(CAMEex, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False


    def _get_options(self, param_shape):
        factored = len(param_shape) >= 2
        return factored

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5 + 1e-12)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        mean_row_sq = exp_avg_sq_row.mean(dim=-1, keepdim=True).clamp_(min=self.defaults.get('eps', (1e-30, 1e-16))[0] * 1e-2)
        
        r_factor = (
            (exp_avg_sq_row / mean_row_sq)
            .rsqrt()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.clamp_(min=self.defaults.get('eps', (1e-30, 1e-16))[0] * 1e-2).unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["lr"] is None:
                raise ValueError("Learning rate was not set for param group. "
                                 "Please set it in the optimizer constructor or manually for the group.")

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad_data = p.grad.data
                if grad_data.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                target_optimizer_dtype = group["optimizer_dtype"]
                is_cpu_offload = group["cpu_offload"]
                
                grad = grad_data.to(dtype=target_optimizer_dtype)

                state = self.state[p]
                grad_shape = grad.shape
                
                state_device = torch.device('cpu') if is_cpu_offload else p.device
                compute_device = p.device

                factored = self._get_options(grad_shape)

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad, dtype=target_optimizer_dtype, device=state_device)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1], dtype=target_optimizer_dtype, device=state_device)
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:], dtype=target_optimizer_dtype, device=state_device
                        )
                        state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1], dtype=target_optimizer_dtype, device=state_device)
                        state["exp_avg_res_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:], dtype=target_optimizer_dtype, device=state_device
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad, dtype=target_optimizer_dtype, device=state_device)

                exp_avg_comp = state["exp_avg"].to(compute_device)
                if factored:
                    exp_avg_sq_row_comp = state["exp_avg_sq_row"].to(compute_device)
                    exp_avg_sq_col_comp = state["exp_avg_sq_col"].to(compute_device)
                    exp_avg_res_row_comp = state["exp_avg_res_row"].to(compute_device)
                    exp_avg_res_col_comp = state["exp_avg_res_col"].to(compute_device)
                else:
                    exp_avg_sq_comp = state["exp_avg_sq"].to(compute_device)
                
                grad_comp = grad.to(compute_device)

                state["step"] += 1
                
                update_val = grad_comp.pow(2).add_(group["eps"][0])

                if factored:
                    exp_avg_sq_row_comp.mul_(group["betas"][1]).add_(
                        update_val.mean(dim=-1), alpha=1.0 - group["betas"][1]
                    )
                    exp_avg_sq_col_comp.mul_(group["betas"][1]).add_(
                        update_val.mean(dim=-2), alpha=1.0 - group["betas"][1]
                    )
                    current_update = self._approx_sq_grad(exp_avg_sq_row_comp, exp_avg_sq_col_comp)
                    current_update.mul_(grad_comp)
                else:
                    exp_avg_sq_comp.mul_(group["betas"][1]).add_(update_val, alpha=1.0 - group["betas"][1])
                    current_update = exp_avg_sq_comp.add(group["eps"][0]).rsqrt().mul_(grad_comp)


                rms_val = self._rms(current_update)
                current_update.div_(
                    (rms_val / group["clip_threshold"]).clamp_(min=1.0)
                )

                exp_avg_comp.mul_(group["betas"][0]).add_(current_update, alpha=1.0 - group["betas"][0])
                
                res = (current_update - exp_avg_comp).pow(2).add_(group["eps"][1])

                if factored:
                    exp_avg_res_row_comp.mul_(group["betas"][2]).add_(
                        res.mean(dim=-1), alpha=1.0 - group["betas"][2]
                    )
                    exp_avg_res_col_comp.mul_(group["betas"][2]).add_(
                        res.mean(dim=-2), alpha=1.0 - group["betas"][2]
                    )
                    res_approx = self._approx_sq_grad(exp_avg_res_row_comp, exp_avg_res_col_comp)
                    final_update = res_approx.mul_(exp_avg_comp)
                else:
                    final_update = exp_avg_comp.clone() 
                    
                if is_cpu_offload:
                    state["exp_avg"].copy_(exp_avg_comp)
                    if factored:
                        state["exp_avg_sq_row"].copy_(exp_avg_sq_row_comp)
                        state["exp_avg_sq_col"].copy_(exp_avg_sq_col_comp)
                        state["exp_avg_res_row"].copy_(exp_avg_res_row_comp)
                        state["exp_avg_res_col"].copy_(exp_avg_res_col_comp)
                    else:
                        state["exp_avg_sq"].copy_(exp_avg_sq_comp)

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])
                
                if final_update.dtype != p.data.dtype:
                    final_update = final_update.to(p.data.dtype)
                
                p.data.add_(final_update, alpha=-group["lr"])

        return loss