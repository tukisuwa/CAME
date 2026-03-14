from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

import torch


def _maybe_set_cuda_home_to_torch_version() -> None:
    cuda_ver = getattr(torch.version, "cuda", None)
    if not cuda_ver:
        return
    major_minor = str(cuda_ver).split(".")[:2]
    if len(major_minor) != 2:
        return

    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if not base.exists():
        return

    major = major_minor[0]
    exact = base / f"v{major_minor[0]}.{major_minor[1]}"
    chosen: Path | None = None
    if (exact / "bin" / "nvcc.exe").exists():
        chosen = exact
    else:
        candidates: list[tuple[int, Path]] = []
        for p in base.glob(f"v{major}.*"):
            nvcc = p / "bin" / "nvcc.exe"
            if not nvcc.exists():
                continue
            try:
                minor = int(p.name.split(".")[1])
                candidates.append((minor, p))
            except Exception:
                pass
        if candidates:
            chosen = sorted(candidates, key=lambda x: x[0])[-1][1]

    if chosen is None:
        return

    os.environ.setdefault("CUDA_HOME", str(chosen))
    os.environ.setdefault("CUDA_PATH", str(chosen))
    bin_dir = chosen / "bin"
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def _maybe_add_cuda_bin_to_dll_search_path() -> None:
    # On Windows, an extension compiled against a CUDA Toolkit may depend on CUDA DLLs.
    if os.name != "nt":
        return

    add_dir = getattr(os, "add_dll_directory", None)
    if add_dir is None:
        return

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    candidates: list[Path] = []
    if cuda_home:
        candidates.append(Path(cuda_home) / "bin")
    # Common default install locations
    candidates.extend(
        [
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"),
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"),
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"),
        ]
    )

    for p in candidates:
        try:
            if p.exists():
                add_dir(str(p))
        except Exception:
            pass


@lru_cache(maxsize=1)
def _get_ext():
    _maybe_add_cuda_bin_to_dll_search_path()
    _maybe_set_cuda_home_to_torch_version()

    # Prefer a prebuilt extension (installed via setup.py / pip).
    try:
        from came_pytorch import came_cuda_ext  # type: ignore

        return came_cuda_ext
    except Exception:
        pass

    # Optional JIT build fallback (requires ninja).
    if shutil.which("ninja") is None:
        raise RuntimeError(
            "came_cuda extension is not built and ninja is not installed.\n"
            "Install ninja (`pip install ninja`) or build the extension via:\n"
            "  pip install -e .\n"
            "from the CAME repo root."
        )

    from torch.utils.cpp_extension import load

    this_dir = Path(__file__).resolve().parent
    src_dir = this_dir / "csrc"

    sources = [
        str(src_dir / "came_bindings.cpp"),
        str(src_dir / "came_kernels.cu"),
    ]

    extra_cuda_cflags = [
        "--use_fast_math",
        "-lineinfo",
    ]

    extra_cflags = ["/std:c++17"] if os.name == "nt" else ["-std=c++17"]

    return load(
        name="came_cuda_ext_jit",
        sources=sources,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=extra_cflags,
        with_cuda=True,
        verbose=True,
    )


def came2d_step(
    *,
    p: torch.Tensor,
    g: torch.Tensor,
    exp_avg_q: torch.Tensor,
    exp_avg_absmax: torch.Tensor,
    exp_avg_sq_row_q: torch.Tensor,
    exp_avg_sq_row_absmax: torch.Tensor,
    exp_avg_sq_col_q: torch.Tensor,
    exp_avg_sq_col_absmax: torch.Tensor,
    exp_avg_res_row_q: torch.Tensor,
    exp_avg_res_row_absmax: torch.Tensor,
    exp_avg_res_col_q: torch.Tensor,
    exp_avg_res_col_absmax: torch.Tensor,
    r_factor: torch.Tensor,
    c_factor: torch.Tensor,
    r_res_factor: torch.Tensor,
    c_res_factor: torch.Tensor,
    res_row_sum: torch.Tensor,
    res_col_sum: torch.Tensor,
    res_row_partial: torch.Tensor,
    res_col_partial: torch.Tensor,
    sum_sq_row: torch.Tensor,
    sum_update: torch.Tensor,
    sum_update_partial: torch.Tensor,
    sum_res_row: torch.Tensor,
    beta1: float,
    beta2: float,
    beta3: float,
    eps0: float,
    eps1: float,
    lr: float,
    clip_threshold: float,
    weight_decay: float,
    block_size: int = 256,
) -> None:
    ext = _get_ext()
    ext.came2d_step(
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
        float(beta1),
        float(beta2),
        float(beta3),
        float(eps0),
        float(eps1),
        float(lr),
        float(clip_threshold),
        float(weight_decay),
        int(block_size),
    )


def blockwise_quantize_into(
    *,
    src: torch.Tensor,
    q_out: torch.Tensor,
    absmax_out: torch.Tensor,
    signed: bool,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.blockwise_quant(src, q_out, absmax_out, int(block_size), bool(signed))


def blockwise_dequantize_into(
    *,
    out: torch.Tensor,
    q_in: torch.Tensor,
    absmax: torch.Tensor,
    signed: bool,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.blockwise_dequant(out, q_in, absmax, int(block_size), bool(signed))


def blockwise_quantize_batched_into(
    *,
    src: torch.Tensor,
    q_out: torch.Tensor,
    absmax_out: torch.Tensor,
    signed: bool,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.blockwise_quant_batched(src, q_out, absmax_out, int(block_size), bool(signed))


def blockwise_dequantize_batched_into(
    *,
    out: torch.Tensor,
    q_in: torch.Tensor,
    absmax: torch.Tensor,
    signed: bool,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.blockwise_dequant_batched(out, q_in, absmax, int(block_size), bool(signed))


def came_full_nonfactored_step(
    *,
    p: torch.Tensor,
    g32: torch.Tensor,
    exp_avg_q: torch.Tensor,
    exp_avg_absmax: torch.Tensor,
    exp_avg_sq_q: torch.Tensor,
    exp_avg_sq_absmax: torch.Tensor,
    update: torch.Tensor,
    sum_update_partial: torch.Tensor,
    sum_update: torch.Tensor,
    beta1: float,
    beta2: float,
    eps0: float,
    lr: float,
    clip_threshold: float,
    weight_decay: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_nonfactored_step(
        p,
        g32,
        exp_avg_q,
        exp_avg_absmax,
        exp_avg_sq_q,
        exp_avg_sq_absmax,
        update,
        sum_update_partial,
        sum_update,
        float(beta1),
        float(beta2),
        float(eps0),
        float(lr),
        float(clip_threshold),
        float(weight_decay),
        int(block_size),
    )


def came_full_nonfactored_step_fp16_update(
    *,
    p: torch.Tensor,
    g32: torch.Tensor,
    exp_avg_q: torch.Tensor,
    exp_avg_absmax: torch.Tensor,
    exp_avg_sq_q: torch.Tensor,
    exp_avg_sq_absmax: torch.Tensor,
    update: torch.Tensor,
    sum_update_partial: torch.Tensor,
    sum_update: torch.Tensor,
    beta1: float,
    beta2: float,
    eps0: float,
    lr: float,
    clip_threshold: float,
    weight_decay: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_nonfactored_step_fp16_update(
        p,
        g32,
        exp_avg_q,
        exp_avg_absmax,
        exp_avg_sq_q,
        exp_avg_sq_absmax,
        update,
        sum_update_partial,
        sum_update,
        float(beta1),
        float(beta2),
        float(eps0),
        float(lr),
        float(clip_threshold),
        float(weight_decay),
        int(block_size),
    )


def came_full_nonfactored_step_batched(
    *,
    p: torch.Tensor,
    g32: torch.Tensor,
    exp_avg_q: torch.Tensor,
    exp_avg_absmax: torch.Tensor,
    exp_avg_sq_q: torch.Tensor,
    exp_avg_sq_absmax: torch.Tensor,
    update: torch.Tensor,
    sum_update_partial: torch.Tensor,
    sum_update: torch.Tensor,
    beta1: float,
    beta2: float,
    eps0: float,
    lr: float,
    clip_threshold: float,
    weight_decay: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_nonfactored_step_batched(
        p,
        g32,
        exp_avg_q,
        exp_avg_absmax,
        exp_avg_sq_q,
        exp_avg_sq_absmax,
        update,
        sum_update_partial,
        sum_update,
        float(beta1),
        float(beta2),
        float(eps0),
        float(lr),
        float(clip_threshold),
        float(weight_decay),
        int(block_size),
    )


def came_full_nonfactored_step_multitensor(
    *,
    p_list: list[torch.Tensor],
    g32_list: list[torch.Tensor],
    exp_avg_q_list: list[torch.Tensor],
    exp_avg_absmax_list: list[torch.Tensor],
    exp_avg_sq_q_list: list[torch.Tensor],
    exp_avg_sq_absmax_list: list[torch.Tensor],
    update_list: list[torch.Tensor],
    sum_update_partial_list: list[torch.Tensor],
    sum_update_list: list[torch.Tensor],
    beta1: float,
    beta2: float,
    eps0: float,
    lr: float,
    clip_threshold: float,
    weight_decay: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_nonfactored_step_multitensor(
        p_list,
        g32_list,
        exp_avg_q_list,
        exp_avg_absmax_list,
        exp_avg_sq_q_list,
        exp_avg_sq_absmax_list,
        update_list,
        sum_update_partial_list,
        sum_update_list,
        float(beta1),
        float(beta2),
        float(eps0),
        float(lr),
        float(clip_threshold),
        float(weight_decay),
        int(block_size),
    )


def came_full_nonfactored_step_multitensor_ptrs(
    *,
    sample_p: torch.Tensor,
    p_ptrs: torch.Tensor,
    g32_ptrs: torch.Tensor,
    exp_avg_q_ptrs: torch.Tensor,
    exp_avg_absmax_ptrs: torch.Tensor,
    exp_avg_sq_q_ptrs: torch.Tensor,
    exp_avg_sq_absmax_ptrs: torch.Tensor,
    update_ptrs: torch.Tensor,
    sum_update_ptrs: torch.Tensor,
    per_item_numel: int,
    beta1: float,
    beta2: float,
    eps0: float,
    lr: float,
    clip_threshold: float,
    weight_decay: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_nonfactored_step_multitensor_ptrs(
        sample_p,
        p_ptrs,
        g32_ptrs,
        exp_avg_q_ptrs,
        exp_avg_absmax_ptrs,
        exp_avg_sq_q_ptrs,
        exp_avg_sq_absmax_ptrs,
        update_ptrs,
        sum_update_ptrs,
        int(per_item_numel),
        float(beta1),
        float(beta2),
        float(eps0),
        float(lr),
        float(clip_threshold),
        float(weight_decay),
        int(block_size),
    )


def came_full_nonfactored_step_multitensor_fp16_update_ptrs(
    *,
    sample_p: torch.Tensor,
    sample_update: torch.Tensor,
    p_ptrs: torch.Tensor,
    g32_ptrs: torch.Tensor,
    exp_avg_q_ptrs: torch.Tensor,
    exp_avg_absmax_ptrs: torch.Tensor,
    exp_avg_sq_q_ptrs: torch.Tensor,
    exp_avg_sq_absmax_ptrs: torch.Tensor,
    update_ptrs: torch.Tensor,
    sum_update_ptrs: torch.Tensor,
    per_item_numel: int,
    beta1: float,
    beta2: float,
    eps0: float,
    lr: float,
    clip_threshold: float,
    weight_decay: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_nonfactored_step_multitensor_fp16_update_ptrs(
        sample_p,
        sample_update,
        p_ptrs,
        g32_ptrs,
        exp_avg_q_ptrs,
        exp_avg_absmax_ptrs,
        exp_avg_sq_q_ptrs,
        exp_avg_sq_absmax_ptrs,
        update_ptrs,
        sum_update_ptrs,
        int(per_item_numel),
        float(beta1),
        float(beta2),
        float(eps0),
        float(lr),
        float(clip_threshold),
        float(weight_decay),
        int(block_size),
    )


def came_full_nonfactored_step_multitensor_varlen_ptrs(
    *,
    sample_p: torch.Tensor,
    p_ptrs: torch.Tensor,
    g32_ptrs: torch.Tensor,
    exp_avg_q_ptrs: torch.Tensor,
    exp_avg_absmax_ptrs: torch.Tensor,
    exp_avg_sq_q_ptrs: torch.Tensor,
    exp_avg_sq_absmax_ptrs: torch.Tensor,
    update_ptrs: torch.Tensor,
    sum_update_ptrs: torch.Tensor,
    item_numels: torch.Tensor,
    max_item_numel: int,
    beta1: float,
    beta2: float,
    eps0: float,
    lr: float,
    clip_threshold: float,
    weight_decay: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_nonfactored_step_multitensor_varlen_ptrs(
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
        int(max_item_numel),
        float(beta1),
        float(beta2),
        float(eps0),
        float(lr),
        float(clip_threshold),
        float(weight_decay),
        int(block_size),
    )


def came_full_nonfactored_step_multitensor_compact_varlen_ptrs(
    *,
    sample_p: torch.Tensor,
    p_ptrs: torch.Tensor,
    g32_ptrs: torch.Tensor,
    exp_avg_q_ptrs: torch.Tensor,
    exp_avg_absmax_ptrs: torch.Tensor,
    exp_avg_sq_q_ptrs: torch.Tensor,
    exp_avg_sq_absmax_ptrs: torch.Tensor,
    update_ptrs: torch.Tensor,
    sum_update_ptrs: torch.Tensor,
    item_numels: torch.Tensor,
    block_item_ids: torch.Tensor,
    block_item_starts: torch.Tensor,
    beta1: float,
    beta2: float,
    eps0: float,
    lr: float,
    clip_threshold: float,
    weight_decay: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_nonfactored_step_multitensor_compact_varlen_ptrs(
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
        float(beta1),
        float(beta2),
        float(eps0),
        float(lr),
        float(clip_threshold),
        float(weight_decay),
        int(block_size),
    )


def fill_device_ptr_tensor(
    *,
    tensors: list[torch.Tensor],
    ptr_tensor: torch.Tensor,
) -> None:
    ext = _get_ext()
    ext.fill_device_ptr_tensor(tensors, ptr_tensor)


def came_full_factored_sq_step(
    *,
    g32: torch.Tensor,
    exp_avg_sq_row_q: torch.Tensor,
    exp_avg_sq_row_absmax: torch.Tensor,
    exp_avg_sq_col_q: torch.Tensor,
    exp_avg_sq_col_absmax: torch.Tensor,
    r_factor: torch.Tensor,
    c_factor: torch.Tensor,
    row_absmax_scratch: torch.Tensor,
    reduce_partial: torch.Tensor,
    sum_row_state: torch.Tensor,
    sum_update: torch.Tensor,
    beta2: float,
    eps0: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_factored_sq_step(
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
        float(beta2),
        float(eps0),
        int(block_size),
    )


def came_full_factored_sq_step_batched(
    *,
    g32: torch.Tensor,
    exp_avg_sq_row_q: torch.Tensor,
    exp_avg_sq_row_absmax: torch.Tensor,
    exp_avg_sq_col_q: torch.Tensor,
    exp_avg_sq_col_absmax: torch.Tensor,
    r_factor: torch.Tensor,
    c_factor: torch.Tensor,
    row_absmax_scratch: torch.Tensor,
    reduce_partial: torch.Tensor,
    sum_row_state: torch.Tensor,
    sum_update_slice: torch.Tensor,
    sum_update_total: torch.Tensor,
    beta2: float,
    eps0: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_factored_sq_step_batched(
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
        float(beta2),
        float(eps0),
        int(block_size),
    )


def came_full_factored_res_step(
    *,
    res32: torch.Tensor,
    exp_avg_res_row_q: torch.Tensor,
    exp_avg_res_row_absmax: torch.Tensor,
    exp_avg_res_col_q: torch.Tensor,
    exp_avg_res_col_absmax: torch.Tensor,
    r_factor: torch.Tensor,
    c_factor: torch.Tensor,
    row_absmax_scratch: torch.Tensor,
    reduce_partial: torch.Tensor,
    sum_row_state: torch.Tensor,
    beta3: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_factored_res_step(
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
        float(beta3),
        int(block_size),
    )


def came_full_factored_res_step_batched(
    *,
    res32: torch.Tensor,
    exp_avg_res_row_q: torch.Tensor,
    exp_avg_res_row_absmax: torch.Tensor,
    exp_avg_res_col_q: torch.Tensor,
    exp_avg_res_col_absmax: torch.Tensor,
    r_factor: torch.Tensor,
    c_factor: torch.Tensor,
    row_absmax_scratch: torch.Tensor,
    reduce_partial: torch.Tensor,
    sum_row_state: torch.Tensor,
    beta3: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_factored_res_step_batched(
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
        float(beta3),
        int(block_size),
    )


def came_full_factored_expavg_res_prepare(
    *,
    g32: torch.Tensor,
    exp_avg_q: torch.Tensor,
    exp_avg_absmax: torch.Tensor,
    r_factor: torch.Tensor,
    c_factor: torch.Tensor,
    exp_avg_fp32: torch.Tensor,
    res32: torch.Tensor,
    sum_update: torch.Tensor,
    beta1: float,
    eps1: float,
    clip_threshold: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_factored_expavg_res_prepare(
        g32,
        exp_avg_q,
        exp_avg_absmax,
        r_factor,
        c_factor,
        exp_avg_fp32,
        res32,
        sum_update,
        float(beta1),
        float(eps1),
        float(clip_threshold),
        int(block_size),
    )


def came_full_factored_expavg_res_prepare_batched(
    *,
    g32: torch.Tensor,
    exp_avg_q: torch.Tensor,
    exp_avg_absmax: torch.Tensor,
    r_factor: torch.Tensor,
    c_factor: torch.Tensor,
    exp_avg_fp32: torch.Tensor,
    res32: torch.Tensor,
    sum_update: torch.Tensor,
    beta1: float,
    eps1: float,
    clip_threshold: float,
    block_size: int,
) -> None:
    ext = _get_ext()
    ext.came_full_factored_expavg_res_prepare_batched(
        g32,
        exp_avg_q,
        exp_avg_absmax,
        r_factor,
        c_factor,
        exp_avg_fp32,
        res32,
        sum_update,
        float(beta1),
        float(eps1),
        float(clip_threshold),
        int(block_size),
    )


def came_full_factored_param_update(
    *,
    p: torch.Tensor,
    exp_avg_fp32: torch.Tensor,
    r_factor: torch.Tensor,
    c_factor: torch.Tensor,
    lr: float,
    weight_decay: float,
) -> None:
    ext = _get_ext()
    ext.came_full_factored_param_update(
        p,
        exp_avg_fp32,
        r_factor,
        c_factor,
        float(lr),
        float(weight_decay),
    )


def came_fp_factored_step(
    *,
    p: torch.Tensor,
    g32: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq_row: torch.Tensor,
    exp_avg_sq_col: torch.Tensor,
    exp_avg_res_row: torch.Tensor,
    exp_avg_res_col: torch.Tensor,
    r_factor: torch.Tensor,
    c_factor: torch.Tensor,
    scratch: torch.Tensor,
    reduce_partial: torch.Tensor,
    sum_row_state: torch.Tensor,
    sum_update: torch.Tensor,
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
    ext = _get_ext()
    ext.came_fp_factored_step(
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
        float(beta1),
        float(beta2),
        float(beta3),
        float(eps0),
        float(eps1),
        float(lr),
        float(clip_threshold),
        float(weight_decay),
        int(block_size),
    )


def came_full_factored_param_update_batched(
    *,
    p: torch.Tensor,
    exp_avg_fp32: torch.Tensor,
    r_factor: torch.Tensor,
    c_factor: torch.Tensor,
    lr: float,
    weight_decay: float,
) -> None:
    ext = _get_ext()
    ext.came_full_factored_param_update_batched(
        p,
        exp_avg_fp32,
        r_factor,
        c_factor,
        float(lr),
        float(weight_decay),
    )


def came_full_factored_step_multitensor_same_shape_ptrs(
    *,
    sample_p: torch.Tensor,
    g32: torch.Tensor,
    p_ptrs: torch.Tensor,
    exp_avg_q_ptrs: torch.Tensor,
    exp_avg_absmax_ptrs: torch.Tensor,
    exp_avg_sq_row_q_ptrs: torch.Tensor,
    exp_avg_sq_row_absmax_ptrs: torch.Tensor,
    exp_avg_sq_col_q_ptrs: torch.Tensor,
    exp_avg_sq_col_absmax_ptrs: torch.Tensor,
    exp_avg_res_row_q_ptrs: torch.Tensor,
    exp_avg_res_row_absmax_ptrs: torch.Tensor,
    exp_avg_res_col_q_ptrs: torch.Tensor,
    exp_avg_res_col_absmax_ptrs: torch.Tensor,
    row_factor: torch.Tensor,
    c_factor: torch.Tensor,
    row_absmax_scratch: torch.Tensor,
    reduce_partial: torch.Tensor,
    sum_row_state: torch.Tensor,
    sum_update_slice: torch.Tensor,
    sum_update_total: torch.Tensor,
    sum_update_equiv: torch.Tensor,
    exp_avg_fp32: torch.Tensor,
    res32: torch.Tensor,
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
    ext = _get_ext()
    ext.came_full_factored_step_multitensor_same_shape_ptrs(
        sample_p,
        g32,
        p_ptrs,
        exp_avg_q_ptrs,
        exp_avg_absmax_ptrs,
        exp_avg_sq_row_q_ptrs,
        exp_avg_sq_row_absmax_ptrs,
        exp_avg_sq_col_q_ptrs,
        exp_avg_sq_col_absmax_ptrs,
        exp_avg_res_row_q_ptrs,
        exp_avg_res_row_absmax_ptrs,
        exp_avg_res_col_q_ptrs,
        exp_avg_res_col_absmax_ptrs,
        row_factor,
        c_factor,
        row_absmax_scratch,
        reduce_partial,
        sum_row_state,
        sum_update_slice,
        sum_update_total,
        sum_update_equiv,
        exp_avg_fp32,
        res32,
        float(beta1),
        float(beta2),
        float(beta3),
        float(eps0),
        float(eps1),
        float(lr),
        float(clip_threshold),
        float(weight_decay),
        int(block_size),
    )


def came_full_factored_nd_chunked_step(
    *,
    p: torch.Tensor,
    g32: torch.Tensor,
    exp_avg_q: torch.Tensor,
    exp_avg_absmax: torch.Tensor,
    exp_avg_sq_row_q: torch.Tensor,
    exp_avg_sq_row_absmax: torch.Tensor,
    exp_avg_sq_col_q: torch.Tensor,
    exp_avg_sq_col_absmax: torch.Tensor,
    exp_avg_res_row_q: torch.Tensor,
    exp_avg_res_row_absmax: torch.Tensor,
    exp_avg_res_col_q: torch.Tensor,
    exp_avg_res_col_absmax: torch.Tensor,
    r_factor: torch.Tensor,
    c_factor: torch.Tensor,
    row_absmax_scratch: torch.Tensor,
    reduce_partial: torch.Tensor,
    sum_row_state: torch.Tensor,
    sum_update_slice: torch.Tensor,
    sum_update_chunk: torch.Tensor,
    sum_update_total: torch.Tensor,
    sum_update_equiv: torch.Tensor,
    exp_avg_fp32: torch.Tensor,
    res32: torch.Tensor,
    beta1: float,
    beta2: float,
    beta3: float,
    eps0: float,
    eps1: float,
    lr: float,
    clip_threshold: float,
    weight_decay: float,
    chunk_size: int,
    block_size: int,
    direct_row_sum: bool = False,
) -> None:
    ext = _get_ext()
    fused = getattr(ext, "came_full_factored_nd_chunked_step", None)
    if fused is not None:
        fused(
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
            float(beta1),
            float(beta2),
            float(beta3),
            float(eps0),
            float(eps1),
            float(lr),
            float(clip_threshold),
            float(weight_decay),
            int(chunk_size),
            int(block_size),
            bool(direct_row_sum),
        )
        return

    batch = int(g32.size(0))
    sum_update_total.zero_()
    for start in range(0, batch, chunk_size):
        end = min(start + chunk_size, batch)
        active = end - start
        sum_update_chunk.zero_()
        came_full_factored_sq_step_batched(
            g32=g32[start:end],
            exp_avg_sq_row_q=exp_avg_sq_row_q[start:end],
            exp_avg_sq_row_absmax=exp_avg_sq_row_absmax[start:end],
            exp_avg_sq_col_q=exp_avg_sq_col_q[start:end],
            exp_avg_sq_col_absmax=exp_avg_sq_col_absmax[start:end],
            r_factor=r_factor[start:end],
            c_factor=c_factor[start:end],
            row_absmax_scratch=row_absmax_scratch[:active],
            reduce_partial=reduce_partial[:active],
            sum_row_state=sum_row_state[:active],
            sum_update_slice=sum_update_slice[:active],
            sum_update_total=sum_update_chunk,
            beta2=beta2,
            eps0=eps0,
            block_size=block_size,
        )
        sum_update_total.add_(sum_update_chunk)

    sum_update_equiv.copy_(sum_update_total).div_(float(batch))
    for start in range(0, batch, chunk_size):
        end = min(start + chunk_size, batch)
        active = end - start
        came_full_factored_expavg_res_prepare_batched(
            g32=g32[start:end],
            exp_avg_q=exp_avg_q[start:end],
            exp_avg_absmax=exp_avg_absmax[start:end],
            r_factor=r_factor[start:end],
            c_factor=c_factor[start:end],
            exp_avg_fp32=exp_avg_fp32[:active],
            res32=res32[:active],
            sum_update=sum_update_equiv,
            beta1=beta1,
            eps1=eps1,
            clip_threshold=clip_threshold,
            block_size=block_size,
        )
        came_full_factored_res_step_batched(
            res32=res32[:active],
            exp_avg_res_row_q=exp_avg_res_row_q[start:end],
            exp_avg_res_row_absmax=exp_avg_res_row_absmax[start:end],
            exp_avg_res_col_q=exp_avg_res_col_q[start:end],
            exp_avg_res_col_absmax=exp_avg_res_col_absmax[start:end],
            r_factor=r_factor[start:end],
            c_factor=c_factor[start:end],
            row_absmax_scratch=row_absmax_scratch[:active],
            reduce_partial=reduce_partial[:active],
            sum_row_state=sum_row_state[:active],
            beta3=beta3,
            block_size=block_size,
        )
        came_full_factored_param_update_batched(
            p=p[start:end],
            exp_avg_fp32=exp_avg_fp32[:active],
            r_factor=r_factor[start:end],
            c_factor=c_factor[start:end],
            lr=lr,
            weight_decay=weight_decay,
        )
