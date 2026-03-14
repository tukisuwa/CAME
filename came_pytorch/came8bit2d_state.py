from __future__ import annotations

import math

import torch

from came_pytorch.blockwise_quantization import init_qstate
from came_pytorch.came_cuda import came2d_step


def prepare_step_tensors(
    p: torch.Tensor,
    g: torch.Tensor,
    *,
    require_contiguous: bool,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    p_data = p.data
    g_data = g.data

    if require_contiguous:
        if not p_data.is_contiguous() or not g_data.is_contiguous():
            raise RuntimeError("CUDA graph capture requires contiguous parameter and gradient tensors.")
        return p_data, g_data, False

    needs_param_copyback = not p_data.is_contiguous()
    if needs_param_copyback:
        p_data = p_data.contiguous()
    if not g_data.is_contiguous():
        g_data = g_data.contiguous()
    return p_data, g_data, needs_param_copyback


def init_state(state: dict, p_data: torch.Tensor, *, block_size: int) -> None:
    rows, cols = p_data.shape
    device = p_data.device
    tile_rows = int(math.ceil(rows / 16))
    tile_cols = int(math.ceil(cols / 16))
    num_tiles = tile_rows * tile_cols

    state["_mode"] = "8bit2d"
    state["exp_avg_q"] = torch.zeros_like(p_data, dtype=torch.int8)
    state["exp_avg_absmax"] = torch.zeros((num_tiles,), device=device, dtype=torch.float32)

    state["exp_avg_sq_row_q"], state["exp_avg_sq_row_absmax"] = init_qstate(
        (rows,), device=device, signed=False, block_size=block_size
    )
    state["exp_avg_sq_col_q"], state["exp_avg_sq_col_absmax"] = init_qstate(
        (cols,), device=device, signed=False, block_size=block_size
    )
    state["exp_avg_res_row_q"], state["exp_avg_res_row_absmax"] = init_qstate(
        (rows,), device=device, signed=False, block_size=block_size
    )
    state["exp_avg_res_col_q"], state["exp_avg_res_col_absmax"] = init_qstate(
        (cols,), device=device, signed=False, block_size=block_size
    )

    state["r_factor"] = torch.empty((rows,), device=device, dtype=torch.float32)
    state["c_factor"] = torch.empty((cols,), device=device, dtype=torch.float32)
    state["r_res_factor"] = torch.empty((rows,), device=device, dtype=torch.float32)
    state["c_res_factor"] = torch.empty((cols,), device=device, dtype=torch.float32)

    state["res_row_sum"] = torch.empty((rows,), device=device, dtype=torch.float32)
    state["res_col_sum"] = torch.empty((cols,), device=device, dtype=torch.float32)
    state["res_row_partial"] = torch.empty((tile_cols * rows,), device=device, dtype=torch.float32)
    state["res_col_partial"] = torch.empty((tile_rows * cols,), device=device, dtype=torch.float32)
    state["sum_sq_row"] = torch.empty((1,), device=device, dtype=torch.float32)
    state["sum_update"] = torch.empty((1,), device=device, dtype=torch.float32)
    state["sum_update_partial"] = torch.empty(((rows * cols + 255) // 256,), device=device, dtype=torch.float32)
    state["sum_res_row"] = torch.empty((1,), device=device, dtype=torch.float32)

    state["rows"] = rows
    state["cols"] = cols
    state["tile_rows"] = tile_rows
    state["tile_cols"] = tile_cols
    state["num_tiles"] = num_tiles


def validate_state_shape(state: dict, p_data: torch.Tensor) -> None:
    rows, cols = p_data.shape
    tile_rows = int(math.ceil(rows / 16))
    tile_cols = int(math.ceil(cols / 16))
    num_tiles = tile_rows * tile_cols

    if state["rows"] != rows or state["cols"] != cols:
        raise RuntimeError("Parameter shape changed; this optimizer does not support resizing 8-bit state.")
    if state["tile_rows"] != tile_rows or state["tile_cols"] != tile_cols or state["num_tiles"] != num_tiles:
        raise RuntimeError("Tile layout changed; this optimizer does not support resizing 8-bit state.")


def step_2d(
    *,
    state: dict,
    p_data: torch.Tensor,
    g_data: torch.Tensor,
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
    came2d_step(
        p=p_data,
        g=g_data,
        exp_avg_q=state["exp_avg_q"],
        exp_avg_absmax=state["exp_avg_absmax"],
        exp_avg_sq_row_q=state["exp_avg_sq_row_q"],
        exp_avg_sq_row_absmax=state["exp_avg_sq_row_absmax"],
        exp_avg_sq_col_q=state["exp_avg_sq_col_q"],
        exp_avg_sq_col_absmax=state["exp_avg_sq_col_absmax"],
        exp_avg_res_row_q=state["exp_avg_res_row_q"],
        exp_avg_res_row_absmax=state["exp_avg_res_row_absmax"],
        exp_avg_res_col_q=state["exp_avg_res_col_q"],
        exp_avg_res_col_absmax=state["exp_avg_res_col_absmax"],
        r_factor=state["r_factor"],
        c_factor=state["c_factor"],
        r_res_factor=state["r_res_factor"],
        c_res_factor=state["c_res_factor"],
        res_row_sum=state["res_row_sum"],
        res_col_sum=state["res_col_sum"],
        res_row_partial=state["res_row_partial"],
        res_col_partial=state["res_col_partial"],
        sum_sq_row=state["sum_sq_row"],
        sum_update=state["sum_update"],
        sum_update_partial=state["sum_update_partial"],
        sum_res_row=state["sum_res_row"],
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
