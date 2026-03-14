from __future__ import annotations

import math

import torch

SIGNED_QMAX = 127.0
UNSIGNED_QMAX = 255.0


def num_blocks(numel: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    return max(1, math.ceil(numel / block_size))


def init_qstate(
    shape: torch.Size | tuple[int, ...],
    *,
    device: torch.device,
    signed: bool,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    qdtype = torch.int8 if signed else torch.uint8
    q = torch.zeros(shape, device=device, dtype=qdtype)
    absmax = torch.ones((num_blocks(q.numel(), block_size),), device=device, dtype=torch.float32)
    return q, absmax


def init_batched_qstate(
    shape: torch.Size | tuple[int, ...],
    *,
    batch_size: int,
    device: torch.device,
    signed: bool,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    qdtype = torch.int8 if signed else torch.uint8
    q = torch.zeros((batch_size, *shape), device=device, dtype=qdtype)
    per_item_blocks = num_blocks(math.prod(shape), block_size)
    absmax = torch.ones((batch_size, per_item_blocks), device=device, dtype=torch.float32)
    return q, absmax


def init_blockwise_workspace(
    shape: torch.Size | tuple[int, ...],
    *,
    device: torch.device,
    block_size: int,
) -> torch.Tensor:
    total = num_blocks(math.prod(shape), block_size) * block_size
    return torch.empty((total,), device=device, dtype=torch.float32)


def init_batched_blockwise_workspace(
    shape: torch.Size | tuple[int, ...],
    *,
    batch_size: int,
    device: torch.device,
    block_size: int,
) -> torch.Tensor:
    per_item = num_blocks(math.prod(shape), block_size) * block_size
    return torch.empty((batch_size * per_item,), device=device, dtype=torch.float32)


def quantize_blockwise(tensor: torch.Tensor, *, signed: bool, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    flat = tensor.detach().reshape(-1).to(torch.float32)
    blocks = num_blocks(flat.numel(), block_size)
    padded = torch.zeros((blocks * block_size,), device=flat.device, dtype=torch.float32)
    padded[: flat.numel()] = flat
    padded = padded.view(blocks, block_size)

    if signed:
        absmax = padded.abs().amax(dim=1)
        absmax = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
        q = torch.round((padded / absmax.unsqueeze(1)) * SIGNED_QMAX).clamp(-SIGNED_QMAX, SIGNED_QMAX).to(torch.int8)
    else:
        absmax = padded.amax(dim=1)
        absmax = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
        q = torch.round((padded / absmax.unsqueeze(1)) * UNSIGNED_QMAX).clamp(0, UNSIGNED_QMAX).to(torch.uint8)

    return q.reshape(-1)[: flat.numel()].view_as(tensor), absmax


def quantize_blockwise_into(
    tensor: torch.Tensor,
    q_out: torch.Tensor,
    absmax_out: torch.Tensor,
    workspace: torch.Tensor,
    *,
    signed: bool,
    block_size: int,
) -> None:
    flat = tensor.detach().reshape(-1).to(torch.float32)
    blocks = num_blocks(flat.numel(), block_size)
    work = workspace[: blocks * block_size].view(blocks, block_size)
    work.zero_()
    work.view(-1)[: flat.numel()].copy_(flat)

    if signed:
        absmax_out.copy_(work.abs().amax(dim=1))
        absmax_out.masked_fill_(absmax_out == 0, 1.0)
        work.div_(absmax_out.unsqueeze(1)).mul_(SIGNED_QMAX).round_().clamp_(-SIGNED_QMAX, SIGNED_QMAX)
    else:
        absmax_out.copy_(work.amax(dim=1))
        absmax_out.masked_fill_(absmax_out == 0, 1.0)
        work.div_(absmax_out.unsqueeze(1)).mul_(UNSIGNED_QMAX).round_().clamp_(0.0, UNSIGNED_QMAX)

    q_out.reshape(-1).copy_(work.view(-1)[: flat.numel()])


def dequantize_blockwise(
    q: torch.Tensor,
    absmax: torch.Tensor,
    *,
    signed: bool,
    block_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    flat_q = q.reshape(-1).to(torch.float32)
    scales = absmax.repeat_interleave(block_size)[: flat_q.numel()]
    qmax = SIGNED_QMAX if signed else UNSIGNED_QMAX
    restored = flat_q * (scales / qmax)
    return restored.view_as(q).to(dtype)


def dequantize_blockwise_into(
    out: torch.Tensor,
    q: torch.Tensor,
    absmax: torch.Tensor,
    workspace: torch.Tensor,
    *,
    signed: bool,
    block_size: int,
) -> None:
    flat_q = q.reshape(-1)
    blocks = num_blocks(flat_q.numel(), block_size)
    work = workspace[: blocks * block_size].view(blocks, block_size)
    work.zero_()
    work.view(-1)[: flat_q.numel()].copy_(flat_q)
    qmax = SIGNED_QMAX if signed else UNSIGNED_QMAX
    work.mul_(absmax.view(-1, 1) / qmax)
    out.reshape(-1).copy_(work.view(-1)[: flat_q.numel()])


def quantize_blockwise_batched_into(
    tensor: torch.Tensor,
    q_out: torch.Tensor,
    absmax_out: torch.Tensor,
    workspace: torch.Tensor,
    *,
    signed: bool,
    block_size: int,
) -> None:
    batch_size = tensor.shape[0]
    flat = tensor.detach().reshape(batch_size, -1).to(torch.float32)
    per_item_numel = flat.shape[1]
    blocks = num_blocks(per_item_numel, block_size)
    work = workspace[: batch_size * blocks * block_size].view(batch_size, blocks, block_size)
    work.zero_()
    work.view(batch_size, -1)[:, :per_item_numel].copy_(flat)

    if signed:
        absmax_out.copy_(work.abs().amax(dim=2))
        absmax_out.masked_fill_(absmax_out == 0, 1.0)
        work.div_(absmax_out.unsqueeze(-1)).mul_(SIGNED_QMAX).round_().clamp_(-SIGNED_QMAX, SIGNED_QMAX)
    else:
        absmax_out.copy_(work.amax(dim=2))
        absmax_out.masked_fill_(absmax_out == 0, 1.0)
        work.div_(absmax_out.unsqueeze(-1)).mul_(UNSIGNED_QMAX).round_().clamp_(0.0, UNSIGNED_QMAX)

    q_out.reshape(batch_size, -1).copy_(work.view(batch_size, -1)[:, :per_item_numel])


def dequantize_blockwise_batched_into(
    out: torch.Tensor,
    q: torch.Tensor,
    absmax: torch.Tensor,
    workspace: torch.Tensor,
    *,
    signed: bool,
    block_size: int,
) -> None:
    batch_size = q.shape[0]
    flat_q = q.reshape(batch_size, -1)
    per_item_numel = flat_q.shape[1]
    blocks = num_blocks(per_item_numel, block_size)
    work = workspace[: batch_size * blocks * block_size].view(batch_size, blocks, block_size)
    work.zero_()
    work.view(batch_size, -1)[:, :per_item_numel].copy_(flat_q)
    qmax = SIGNED_QMAX if signed else UNSIGNED_QMAX
    work.mul_(absmax.unsqueeze(-1) / qmax)
    out.reshape(batch_size, -1).copy_(work.view(batch_size, -1)[:, :per_item_numel])
