import math
import torch
import torch.optim
from typing import Optional, Tuple

import bitsandbytes.functional as F
from bitsandbytes.optim.optimizer import Optimizer2State, MockArgs
from bitsandbytes.functional import QuantState 
from collections import abc as container_abcs, defaultdict

if not hasattr(F, 'name2qmap'):
    F.name2qmap = {}


class CAME8bit(Optimizer2State):

    def __init__(
        self,
        params,
        lr=1e-4,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999
        , 0.9999),
        weight_decay=0.0,
        optim_bits=8,
        args=None,
        min_8bit_size=4096,
        percentile_clipping=100,
        block_wise=True,
        is_paged=False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (isinstance(betas, tuple) and len(betas) == 3 and all([0. <= beta <= 1. for beta in betas])):
            raise ValueError(f"Invalid betas: {betas}. Expected tuple of 3 floats between 0 and 1.")
        if not (isinstance(eps, tuple) and len(eps) == 2 and all([e >= 0 for e in eps])):
             raise ValueError(f"Invalid eps: {eps}. Expected tuple of 2 non-negative floats.")


        super_betas = betas[:2] 
        super_eps = eps[0]

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
        )

        super().__init__(
            optimizer_name="came8bit",
            params=params,
            lr=lr,
            betas=super_betas,
            eps=super_eps,
            weight_decay=weight_decay,
            optim_bits=optim_bits,
            args=args,
            min_8bit_size=min_8bit_size,
            percentile_clipping=percentile_clipping,
            block_wise=block_wise,
            is_paged=is_paged,
        )

        for group in self.param_groups:
            group['betas'] = betas
            group['eps'] = eps
            group['clip_threshold'] = clip_threshold

        if optim_bits == 8 and not F.name2qmap:
            self.fill_qmap() 

    def fill_qmap(self):
        F.name2qmap["dynamic"] = F.create_dynamic_map(signed=True)
        F.name2qmap["udynamic"] = F.create_dynamic_map(signed=False)


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
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        exp_avg_sq_row_f = exp_avg_sq_row.float()
        exp_avg_sq_col_f = exp_avg_sq_col.float()

        mean_row = exp_avg_sq_row_f.mean(dim=-1, keepdim=True)
        mean_row = torch.clamp(mean_row, min=torch.finfo(mean_row.dtype).tiny)

        r_factor = (
            (exp_avg_sq_row_f / mean_row)
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col_f.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)
        state = self.state[p]
        grad_shape = p.shape

        if config["optim_bits"] == 32 or p.numel() < config["min_8bit_size"]:
            dtype_uint = None 
            dtype_fp = torch.float32
            is_8bit = False
        elif config["optim_bits"] == 8:
            dtype_uint = torch.uint8
            dtype_fp = torch.float32 
            is_8bit = True
            if not F.name2qmap or "dynamic" not in F.name2qmap or F.name2qmap["dynamic"].device != p.device:
                self.fill_qmap() 
        else:
            raise ValueError(f"Unsupported optim_bits: {config['optim_bits']}")

        state["step"] = 0
        factored = self._get_options(grad_shape)
        state["blocksize"] = config.get("blocksize", 256) 

        def _init_single_state(name_prefix, shape, is_signed_val):
            if is_8bit:
                state[name_prefix] = self.get_state_buffer(torch.empty(shape, device=p.device), dtype=dtype_uint)
                qmap_type = "dynamic" if is_signed_val else "udynamic"
                state[f"qmap_{name_prefix}"] = F.name2qmap[qmap_type].to(p.device)
                if config["block_wise"]:
                    numel_state = state[name_prefix].numel()
                    n_blocks = (numel_state // state["blocksize"]) + bool(numel_state % state["blocksize"])
                    state[f"absmax_{name_prefix}"] = torch.zeros((n_blocks,), dtype=dtype_fp, device=p.device)
                else:
                    state[f"absmax_{name_prefix}"] = torch.zeros((1,), dtype=dtype_fp, device=p.device)
            else:
                state[name_prefix] = self.get_state_buffer(torch.empty(shape, device=p.device), dtype=dtype_fp)

        _init_single_state("exp_avg", grad_shape, is_signed_val=True)

        if factored:
            row_shape = grad_shape[:-1]
            col_shape = grad_shape[:-2] + grad_shape[-1:]
            _init_single_state("exp_avg_sq_row", row_shape, is_signed_val=False)
            _init_single_state("exp_avg_sq_col", col_shape, is_signed_val=False)
            _init_single_state("exp_avg_res_row", row_shape, is_signed_val=False)
            _init_single_state("exp_avg_res_col", col_shape, is_signed_val=False)
        else:
            _init_single_state("exp_avg_sq", grad_shape, is_signed_val=False)

        state["RMS_val"] = torch.tensor(0.0, device=p.device, dtype=dtype_fp)

        if config["percentile_clipping"] < 100:
            state["gnorm_vec"] = torch.zeros((100,), device=p.device, dtype=dtype_fp)

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        state = self.state[p]
        config = self.get_config(gindex, pindex, group)
        is_8bit = state["exp_avg"].dtype == torch.uint8
        block_wise = config["block_wise"]
        current_blocksize = state.get("blocksize", 256) 

        grad = p.grad.data
        if grad.dtype in {torch.float16, torch.bfloat16}:
            grad = grad.float()

        state["step"] += 1
        state["RMS_val"].fill_(self._rms(p.data))

        # --- Dequantize states ---
        if is_8bit:
            if block_wise:
                exp_avg_fp32 = F.dequantize_blockwise(
                    A=state["exp_avg"],
                    absmax=state["absmax_exp_avg"],
                    code=state["qmap_exp_avg"],
                    blocksize=current_blocksize
                )
            else:
                exp_avg_fp32 = F.dequantize_per_tensor_dynamic(
                    state["exp_avg"], state["absmax_exp_avg"], state["qmap_exp_avg"]
                )
        else:
            exp_avg_fp32 = state["exp_avg"]

        factored = self._get_options(grad.shape)
        update_val = (grad**2) + group["eps"][0]

        if factored:
            if is_8bit:
                if block_wise:
                    exp_avg_sq_row_fp32 = F.dequantize_blockwise(
                        A=state["exp_avg_sq_row"],
                        absmax=state["absmax_exp_avg_sq_row"],
                        code=state["qmap_exp_avg_sq_row"],
                        blocksize=current_blocksize
                    )
                    exp_avg_sq_col_fp32 = F.dequantize_blockwise(
                        A=state["exp_avg_sq_col"],
                        absmax=state["absmax_exp_avg_sq_col"],
                        code=state["qmap_exp_avg_sq_col"],
                        blocksize=current_blocksize
                    )
                else:
                    exp_avg_sq_row_fp32 = F.dequantize_per_tensor_dynamic(state["exp_avg_sq_row"], state["absmax_exp_avg_sq_row"], state["qmap_exp_avg_sq_row"])
                    exp_avg_sq_col_fp32 = F.dequantize_per_tensor_dynamic(state["exp_avg_sq_col"], state["absmax_exp_avg_sq_col"], state["qmap_exp_avg_sq_col"])
            else:
                exp_avg_sq_row_fp32 = state["exp_avg_sq_row"]
                exp_avg_sq_col_fp32 = state["exp_avg_sq_col"]

            exp_avg_sq_row_fp32.mul_(group["betas"][1]).add_(update_val.mean(dim=-1), alpha=1.0 - group["betas"][1])
            exp_avg_sq_col_fp32.mul_(group["betas"][1]).add_(update_val.mean(dim=-2), alpha=1.0 - group["betas"][1])

            update_calc = self._approx_sq_grad(exp_avg_sq_row_fp32, exp_avg_sq_col_fp32)
            update_calc.mul_(grad)
        else:
            if is_8bit:
                if block_wise:
                    exp_avg_sq_fp32 = F.dequantize_blockwise(
                        A=state["exp_avg_sq"],
                        absmax=state["absmax_exp_avg_sq"],
                        code=state["qmap_exp_avg_sq"],
                        blocksize=current_blocksize
                    )
                else:
                    exp_avg_sq_fp32 = F.dequantize_per_tensor_dynamic(state["exp_avg_sq"], state["absmax_exp_avg_sq"], state["qmap_exp_avg_sq"])
            else:
                exp_avg_sq_fp32 = state["exp_avg_sq"]

            exp_avg_sq_fp32.mul_(group["betas"][1]).add_(update_val, alpha=1.0 - group["betas"][1])
            update_calc = exp_avg_sq_fp32.rsqrt().mul_(grad)

        update_calc.div_((self._rms(update_calc) / group["clip_threshold"]).clamp_(min=1.0))
        exp_avg_fp32.mul_(group["betas"][0]).add_(update_calc, alpha=1.0 - group["betas"][0])

        res = (update_calc - exp_avg_fp32)**2 + group["eps"][1]
        final_update_val = None

        if factored:
            if is_8bit:
                if block_wise:
                    exp_avg_res_row_fp32 = F.dequantize_blockwise(
                        A=state["exp_avg_res_row"],
                        absmax=state["absmax_exp_avg_res_row"],
                        code=state["qmap_exp_avg_res_row"],
                        blocksize=current_blocksize
                    )
                    exp_avg_res_col_fp32 = F.dequantize_blockwise(
                        A=state["exp_avg_res_col"],
                        absmax=state["absmax_exp_avg_res_col"],
                        code=state["qmap_exp_avg_res_col"],
                        blocksize=current_blocksize
                    )
                else:
                    exp_avg_res_row_fp32 = F.dequantize_per_tensor_dynamic(state["exp_avg_res_row"], state["absmax_exp_avg_res_row"], state["qmap_exp_avg_res_row"])
                    exp_avg_res_col_fp32 = F.dequantize_per_tensor_dynamic(state["exp_avg_res_col"], state["absmax_exp_avg_res_col"], state["qmap_exp_avg_res_col"])
            else:
                exp_avg_res_row_fp32 = state["exp_avg_res_row"]
                exp_avg_res_col_fp32 = state["exp_avg_res_col"]

            exp_avg_res_row_fp32.mul_(group["betas"][2]).add_(res.mean(dim=-1), alpha=1.0 - group["betas"][2])
            exp_avg_res_col_fp32.mul_(group["betas"][2]).add_(res.mean(dim=-2), alpha=1.0 - group["betas"][2])
            res_approx = self._approx_sq_grad(exp_avg_res_row_fp32, exp_avg_res_col_fp32)
            final_update_val = res_approx.mul_(exp_avg_fp32)
        else:
            final_update_val = exp_avg_fp32.clone() 

        # --- Quantize states ---
        if is_8bit:
            def _quantize_single_state_update(name_prefix, fp32_tensor):
                q_map = state[f"qmap_{name_prefix}"]
                absmax_current = state[f"absmax_{name_prefix}"]
                if block_wise:
                    quantized_tensor, quant_state_obj = F.quantize_blockwise(
                        A=fp32_tensor, code=q_map, absmax=absmax_current, blocksize=current_blocksize
                    )
                    state[name_prefix] = quantized_tensor
                    state[f"absmax_{name_prefix}"] = quant_state_obj.absmax
                else:
                    new_absmax, quantized_tensor = F.quantize_per_tensor_dynamic(
                        fp32_tensor, q_map, absmax_current
                    )
                    state[name_prefix] = quantized_tensor
                    state[f"absmax_{name_prefix}"] = new_absmax

            _quantize_single_state_update("exp_avg", exp_avg_fp32)
            if factored:
                _quantize_single_state_update("exp_avg_sq_row", exp_avg_sq_row_fp32)
                _quantize_single_state_update("exp_avg_sq_col", exp_avg_sq_col_fp32)
                _quantize_single_state_update("exp_avg_res_row", exp_avg_res_row_fp32)
                _quantize_single_state_update("exp_avg_res_col", exp_avg_res_col_fp32)
            else:
                _quantize_single_state_update("exp_avg_sq", exp_avg_sq_fp32)
        else:
            state["exp_avg"] = exp_avg_fp32
            if factored:
                state["exp_avg_sq_row"] = exp_avg_sq_row_fp32
                state["exp_avg_sq_col"] = exp_avg_sq_col_fp32
                state["exp_avg_res_row"] = exp_avg_res_row_fp32
                state["exp_avg_res_col"] = exp_avg_res_col_fp32
            else:
                state["exp_avg_sq"] = exp_avg_sq_fp32

        if group["weight_decay"] != 0:
            p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

        final_update_val.mul_(group["lr"])
        p.data.add_(-final_update_val)