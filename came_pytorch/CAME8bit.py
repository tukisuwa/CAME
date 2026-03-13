from collections import defaultdict
import time
import warnings

import torch

from came_pytorch.CAME8bitFull import CAME8bitFull
from came_pytorch.came8bit2d_state import init_state, prepare_step_tensors, step_2d, validate_state_shape
from came_pytorch.came_cuda import came_full_nonfactored_step_batched as cuda_came_full_nonfactored_step_batched
try:
    from came_pytorch.came_cuda import came_full_nonfactored_step_multitensor_ptrs as cuda_came_full_nonfactored_step_multitensor_ptrs
except Exception:
    cuda_came_full_nonfactored_step_multitensor_ptrs = None
try:
    from came_pytorch.came_cuda import came_full_nonfactored_step_multitensor_fp16_update_ptrs as cuda_came_full_nonfactored_step_multitensor_fp16_update_ptrs
except Exception:
    cuda_came_full_nonfactored_step_multitensor_fp16_update_ptrs = None
try:
    from came_pytorch.came_cuda import fill_device_ptr_tensor as cuda_fill_device_ptr_tensor
except Exception:
    cuda_fill_device_ptr_tensor = None


class CAME8bit(CAME8bitFull):
    """
    Single-entrypoint 8-bit optimizer:
      - For CUDA 2D parameters: use the CAME8bit2D CUDA path (speed-oriented).
      - For CUDA 1D parameters: use the generic non-factored CUDA fallback by default.
      - Otherwise: fall back to the full-state blockwise 8-bit reference path.

    This is intended as a practical "single optimizer" entrypoint so users can
    train models that contain non-2D params (LayerNorm, Conv, biases, etc.).

    The CUDA fast path is only available when `block_size == 256`,
    `p.is_cuda`, `p.dim() == 2`, and `p.grad.dtype == p.dtype`.
    """

    CUDA_1D_RUNTIME_MODE_DEFAULT = "default"
    CUDA_1D_RUNTIME_MODE_REPEATED_FP16_PAIRMIN = "repeated_fp16_pairmin"
    CUDA_ND_RUNTIME_MODE_DEFAULT = "default"
    CUDA_ND_RUNTIME_MODE_CHUNK32_LARGE_DIRECT_ROW_SUM = "chunk32_large_direct_row_sum"
    EXPLICIT_USER_FACING_KWARGS = frozenset(
        {
            "prefer_cuda_fast_path",
            "prefer_cuda_2d_fast_path",
            "prefer_factored_cuda_ext_path",
            "cuda_1d_runtime_mode",
            "cuda_nd_runtime_mode",
            "cuda_graph_compatible",
        }
    )
    EXPERIMENTAL_CUDA_TUNING_DEFAULTS = {
        "prefer_cuda_1d_batched_fast_path": False,
        "prefer_cuda_1d_multitensor_fast_path": False,
        "cuda_1d_batched_use_fp16_update": False,
        "cuda_1d_batched_chunk_size": 8,
        "cuda_1d_batched_min_bucket_size": 4,
        "cuda_1d_batched_min_numel": 1024,
        "cuda_1d_batched_max_numel": 65536,
        "cuda_factored_nd_chunk_size_override": None,
        "cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel": None,
        "cuda_factored_nd_chunked_direct_row_sum_max_matrix_numel": None,
        "cuda_nonfactored_use_fp16_update": False,
    }

    def __init__(
        self,
        params,
        lr: float,
        eps: tuple[float, float] = (1e-30, 1e-16),
        clip_threshold: float = 1.0,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
        block_size: int = 256,
        prefer_cuda_fast_path: bool = True,
        prefer_cuda_2d_fast_path: bool | None = None,
        prefer_cuda_1d_fast_path: bool | None = None,
        prefer_cuda_1d_batched_fast_path: bool = False,
        prefer_cuda_1d_multitensor_fast_path: bool = False,
        cuda_1d_batched_use_fp16_update: bool = False,
        cuda_1d_batched_chunk_size: int = 8,
        cuda_1d_batched_min_bucket_size: int = 4,
        cuda_1d_batched_min_numel: int = 1024,
        cuda_1d_batched_max_numel: int = 65536,
        cuda_1d_runtime_mode: str = CUDA_1D_RUNTIME_MODE_DEFAULT,
        prefer_factored_cuda_ext_path: bool = True,
        cuda_factored_nd_chunk_size_override: int | None = None,
        cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel: int | None = None,
        cuda_factored_nd_chunked_direct_row_sum_max_matrix_numel: int | None = None,
        cuda_nd_runtime_mode: str = CUDA_ND_RUNTIME_MODE_DEFAULT,
        cuda_nonfactored_use_fp16_update: bool = False,
        cuda_graph_compatible: bool = False,
    ):
        if cuda_1d_runtime_mode not in {
            self.CUDA_1D_RUNTIME_MODE_DEFAULT,
            self.CUDA_1D_RUNTIME_MODE_REPEATED_FP16_PAIRMIN,
        }:
            raise ValueError(
                "cuda_1d_runtime_mode must be one of "
                f"{self.CUDA_1D_RUNTIME_MODE_DEFAULT!r}, {self.CUDA_1D_RUNTIME_MODE_REPEATED_FP16_PAIRMIN!r}; "
                f"got {cuda_1d_runtime_mode!r}"
            )
        if cuda_nd_runtime_mode not in {
            self.CUDA_ND_RUNTIME_MODE_DEFAULT,
            self.CUDA_ND_RUNTIME_MODE_CHUNK32_LARGE_DIRECT_ROW_SUM,
        }:
            raise ValueError(
                "cuda_nd_runtime_mode must be one of "
                f"{self.CUDA_ND_RUNTIME_MODE_DEFAULT!r}, {self.CUDA_ND_RUNTIME_MODE_CHUNK32_LARGE_DIRECT_ROW_SUM!r}; "
                f"got {cuda_nd_runtime_mode!r}"
            )
        experimental_overrides = self._experimental_cuda_tuning_overrides(
            prefer_cuda_1d_batched_fast_path=prefer_cuda_1d_batched_fast_path,
            prefer_cuda_1d_multitensor_fast_path=prefer_cuda_1d_multitensor_fast_path,
            cuda_1d_batched_use_fp16_update=cuda_1d_batched_use_fp16_update,
            cuda_1d_batched_chunk_size=cuda_1d_batched_chunk_size,
            cuda_1d_batched_min_bucket_size=cuda_1d_batched_min_bucket_size,
            cuda_1d_batched_min_numel=cuda_1d_batched_min_numel,
            cuda_1d_batched_max_numel=cuda_1d_batched_max_numel,
            cuda_factored_nd_chunk_size_override=cuda_factored_nd_chunk_size_override,
            cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel=(
                cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel
            ),
            cuda_factored_nd_chunked_direct_row_sum_max_matrix_numel=(
                cuda_factored_nd_chunked_direct_row_sum_max_matrix_numel
            ),
            cuda_nonfactored_use_fp16_update=cuda_nonfactored_use_fp16_update,
        )
        if experimental_overrides:
            warnings.warn(
                "CAME8bit low-level CUDA tuning kwargs are experimental and intended for "
                "benchmark/diagnostic use: "
                + ", ".join(experimental_overrides)
                + ". Prefer cuda_1d_runtime_mode/cuda_nd_runtime_mode for tracked user-facing opt-ins.",
                RuntimeWarning,
                stacklevel=2,
            )

        effective_prefer_cuda_1d_fast_path = prefer_cuda_1d_fast_path
        effective_prefer_cuda_1d_batched_fast_path = prefer_cuda_1d_batched_fast_path
        effective_prefer_cuda_1d_multitensor_fast_path = prefer_cuda_1d_multitensor_fast_path
        effective_cuda_1d_batched_use_fp16_update = cuda_1d_batched_use_fp16_update
        effective_cuda_1d_batched_min_bucket_size = cuda_1d_batched_min_bucket_size
        effective_cuda_factored_nd_chunk_size_override = cuda_factored_nd_chunk_size_override
        effective_cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel = (
            cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel
        )
        effective_cuda_nonfactored_use_fp16_update = cuda_nonfactored_use_fp16_update

        if cuda_1d_runtime_mode == self.CUDA_1D_RUNTIME_MODE_REPEATED_FP16_PAIRMIN:
            effective_prefer_cuda_1d_fast_path = True
            effective_prefer_cuda_1d_batched_fast_path = True
            effective_prefer_cuda_1d_multitensor_fast_path = True
            effective_cuda_1d_batched_use_fp16_update = True
            effective_cuda_1d_batched_min_bucket_size = 2
            effective_cuda_nonfactored_use_fp16_update = True

        if cuda_nd_runtime_mode == self.CUDA_ND_RUNTIME_MODE_CHUNK32_LARGE_DIRECT_ROW_SUM:
            effective_cuda_factored_nd_chunk_size_override = 32
            effective_cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel = 256 * 256

        super().__init__(
            params,
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
            block_size=block_size,
            prefer_factored_cuda_ext_path=prefer_factored_cuda_ext_path,
            cuda_factored_nd_chunk_size_override=effective_cuda_factored_nd_chunk_size_override,
            cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel=(
                effective_cuda_factored_nd_chunked_direct_row_sum_min_matrix_numel
            ),
            cuda_factored_nd_chunked_direct_row_sum_max_matrix_numel=cuda_factored_nd_chunked_direct_row_sum_max_matrix_numel,
            cuda_nonfactored_use_fp16_update=effective_cuda_nonfactored_use_fp16_update,
        )
        self.cuda_1d_runtime_mode = cuda_1d_runtime_mode
        self.cuda_nd_runtime_mode = cuda_nd_runtime_mode
        self.prefer_cuda_fast_path = bool(prefer_cuda_fast_path)
        self.prefer_cuda_2d_fast_path = (
            self.prefer_cuda_fast_path if prefer_cuda_2d_fast_path is None else bool(prefer_cuda_2d_fast_path)
        )
        self.prefer_cuda_1d_batched_fast_path = bool(effective_prefer_cuda_1d_batched_fast_path)
        self.prefer_cuda_1d_multitensor_fast_path = bool(effective_prefer_cuda_1d_multitensor_fast_path)
        self.cuda_1d_batched_use_fp16_update = bool(effective_cuda_1d_batched_use_fp16_update)
        default_cuda_1d_fast_path = self.prefer_cuda_1d_batched_fast_path
        self.prefer_cuda_1d_fast_path = (
            default_cuda_1d_fast_path
            if effective_prefer_cuda_1d_fast_path is None
            else bool(effective_prefer_cuda_1d_fast_path)
        )
        self.cuda_1d_batched_chunk_size = int(cuda_1d_batched_chunk_size)
        self.cuda_1d_batched_min_bucket_size = int(effective_cuda_1d_batched_min_bucket_size)
        self.cuda_1d_batched_min_numel = int(cuda_1d_batched_min_numel)
        self.cuda_1d_batched_max_numel = int(cuda_1d_batched_max_numel)
        if self.cuda_1d_batched_chunk_size <= 0:
            raise ValueError("cuda_1d_batched_chunk_size must be positive")
        if self.cuda_1d_batched_min_bucket_size <= 0:
            raise ValueError("cuda_1d_batched_min_bucket_size must be positive")
        if self.cuda_1d_batched_min_numel <= 0 or self.cuda_1d_batched_max_numel < self.cuda_1d_batched_min_numel:
            raise ValueError("cuda_1d_batched_min_numel/max_numel must define a positive range")
        self._cuda_1d_batched_workspaces: dict[tuple[torch.device, torch.dtype, int, int, int], dict[str, torch.Tensor]] = {}
        self._cuda_1d_multitensor_ptr_cache: dict[tuple[object, ...], dict[str, torch.Tensor]] = {}
        self.cuda_graph_compatible = bool(cuda_graph_compatible)
        self.experimental_cuda_tuning_overrides = experimental_overrides

    @classmethod
    def _experimental_cuda_tuning_overrides(cls, **kwargs) -> tuple[str, ...]:
        overrides: list[str] = []
        for key, default in cls.EXPERIMENTAL_CUDA_TUNING_DEFAULTS.items():
            if kwargs[key] != default:
                overrides.append(key)
        return tuple(overrides)

    def _should_use_cuda_fast_path(self, p: torch.Tensor, grad: torch.Tensor, block_size: int) -> bool:
        return (
            self.prefer_cuda_2d_fast_path
            and block_size == 256
            and p.is_cuda
            and p.dim() == 2
            and grad.dtype == p.dtype
        )

    def _should_use_cuda_1d_fast_path(self, p: torch.Tensor, grad: torch.Tensor) -> bool:
        return (
            self.prefer_cuda_1d_fast_path
            and p.is_cuda
            and p.dim() == 1
            and p.is_contiguous()
            and grad.is_contiguous()
        )

    def _should_use_cuda_1d_batched_fast_path(
        self, p: torch.Tensor, grad: torch.Tensor, block_size: int
    ) -> bool:
        return (
            self.prefer_cuda_1d_batched_fast_path
            and self._should_use_cuda_1d_fast_path(p, grad)
            and self.cuda_1d_batched_min_numel <= p.numel() <= self.cuda_1d_batched_max_numel
            and block_size > 0
        )

    def _should_use_cuda_1d_full8bit_direct_path(self, p: torch.Tensor, grad: torch.Tensor) -> bool:
        return p.is_cuda and p.dim() == 1 and p.is_contiguous() and grad.is_contiguous()

    def _ensure_nonfactored_full8bit_state(self, state: dict, grad: torch.Tensor, block_size: int) -> bool:
        if len(state) == 0 or state.get("_mode") != "full8bit":
            state.clear()
            self._init_state(state, grad, block_size)
        return (not state.get("factored", False)) and state.get("_cuda_nonfactored_fastpath", False)

    def _ensure_1d_state(self, state: dict, grad: torch.Tensor, block_size: int) -> None:
        if len(state) == 0 or state.get("_mode") != "8bit1d":
            state.clear()
            self._init_state(state, grad, block_size)
            state["_mode"] = "8bit1d"

    def _mark_1d_path_state(self, state: dict, *, batched: bool, multitensor: bool) -> None:
        state["_cuda_nonfactored_fastpath"] = True
        state["_cuda_nonfactored_batched_fastpath"] = bool(batched)
        state["_cuda_nonfactored_multitensor_fastpath"] = bool(multitensor)

    def _ensure_batched_fp16_update_state(self, state: dict, grad: torch.Tensor) -> torch.Tensor:
        update_fp16 = state.get("_cuda_batched_update_fp16")
        if (
            update_fp16 is None
            or update_fp16.device != grad.device
            or update_fp16.shape != grad.shape
            or update_fp16.dtype != torch.float16
        ):
            update_fp16 = torch.empty_like(grad, dtype=torch.float16)
            state["_cuda_batched_update_fp16"] = update_fp16
        return update_fp16

    def _make_device_ptr_tensor(self, tensors: list[torch.Tensor], device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [int(tensor.data_ptr()) for tensor in tensors],
            device=device,
            dtype=torch.int64,
        )

    def _get_multitensor_static_ptrs(
        self,
        chunk: list[tuple[torch.Tensor, dict, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        sample_p = chunk[0][0].data
        key = (
            "fp32_update",
            sample_p.device,
            sample_p.dtype,
            sample_p.numel(),
            tuple(
                (
                    int(p.data_ptr()),
                    int(state["exp_avg_q"].data_ptr()),
                    int(state["exp_avg_absmax"].data_ptr()),
                    int(state["exp_avg_sq_q"].data_ptr()),
                    int(state["exp_avg_sq_absmax"].data_ptr()),
                    int(state["update_fp32"].data_ptr()),
                    int(state["sum_update"].data_ptr()),
                )
                for p, state, _ in chunk
            ),
        )
        cached = self._cuda_1d_multitensor_ptr_cache.get(key)
        if cached is not None:
            return cached

        cached = {
            "p_ptrs": self._make_device_ptr_tensor([p.data for p, _, _ in chunk], sample_p.device),
            "g32_ptrs": torch.empty((len(chunk),), device=sample_p.device, dtype=torch.int64),
            "exp_avg_q_ptrs": self._make_device_ptr_tensor([state["exp_avg_q"] for _, state, _ in chunk], sample_p.device),
            "exp_avg_absmax_ptrs": self._make_device_ptr_tensor(
                [state["exp_avg_absmax"] for _, state, _ in chunk], sample_p.device
            ),
            "exp_avg_sq_q_ptrs": self._make_device_ptr_tensor(
                [state["exp_avg_sq_q"] for _, state, _ in chunk], sample_p.device
            ),
            "exp_avg_sq_absmax_ptrs": self._make_device_ptr_tensor(
                [state["exp_avg_sq_absmax"] for _, state, _ in chunk], sample_p.device
            ),
            "update_ptrs": self._make_device_ptr_tensor([state["update_fp32"] for _, state, _ in chunk], sample_p.device),
            "sum_update_ptrs": self._make_device_ptr_tensor([state["sum_update"] for _, state, _ in chunk], sample_p.device),
        }
        self._cuda_1d_multitensor_ptr_cache[key] = cached
        return cached

    def _get_multitensor_fp16_update_static_ptrs(
        self,
        chunk: list[tuple[torch.Tensor, dict, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        sample_p = chunk[0][0].data
        key = (
            "fp16_update",
            sample_p.device,
            sample_p.dtype,
            sample_p.numel(),
            tuple(
                (
                    int(p.data_ptr()),
                    int(state["exp_avg_q"].data_ptr()),
                    int(state["exp_avg_absmax"].data_ptr()),
                    int(state["exp_avg_sq_q"].data_ptr()),
                    int(state["exp_avg_sq_absmax"].data_ptr()),
                    int(state["_cuda_batched_update_fp16"].data_ptr()),
                    int(state["sum_update"].data_ptr()),
                )
                for p, state, _ in chunk
            ),
        )
        cached = self._cuda_1d_multitensor_ptr_cache.get(key)
        if cached is not None:
            return cached

        cached = {
            "p_ptrs": self._make_device_ptr_tensor([p.data for p, _, _ in chunk], sample_p.device),
            "g32_ptrs": torch.empty((len(chunk),), device=sample_p.device, dtype=torch.int64),
            "exp_avg_q_ptrs": self._make_device_ptr_tensor([state["exp_avg_q"] for _, state, _ in chunk], sample_p.device),
            "exp_avg_absmax_ptrs": self._make_device_ptr_tensor(
                [state["exp_avg_absmax"] for _, state, _ in chunk], sample_p.device
            ),
            "exp_avg_sq_q_ptrs": self._make_device_ptr_tensor(
                [state["exp_avg_sq_q"] for _, state, _ in chunk], sample_p.device
            ),
            "exp_avg_sq_absmax_ptrs": self._make_device_ptr_tensor(
                [state["exp_avg_sq_absmax"] for _, state, _ in chunk], sample_p.device
            ),
            "update_ptrs": self._make_device_ptr_tensor(
                [state["_cuda_batched_update_fp16"] for _, state, _ in chunk], sample_p.device
            ),
            "sum_update_ptrs": self._make_device_ptr_tensor([state["sum_update"] for _, state, _ in chunk], sample_p.device),
        }
        self._cuda_1d_multitensor_ptr_cache[key] = cached
        return cached

    def _get_cuda_1d_batched_workspace(
        self,
        *,
        device: torch.device,
        param_dtype: torch.dtype,
        numel: int,
        block_size: int,
    ) -> dict[str, torch.Tensor]:
        key = (device, param_dtype, self.cuda_1d_batched_chunk_size, numel, block_size)
        workspace = self._cuda_1d_batched_workspaces.get(key)
        if workspace is not None:
            return workspace

        q_blocks = (numel + block_size - 1) // block_size
        chunk_size = self.cuda_1d_batched_chunk_size
        workspace = {
            "param": torch.empty((chunk_size, numel), device=device, dtype=param_dtype),
            "grad_fp32": torch.empty((chunk_size, numel), device=device, dtype=torch.float32),
            "exp_avg_q": torch.empty((chunk_size, numel), device=device, dtype=torch.int8),
            "exp_avg_absmax": torch.empty((chunk_size, q_blocks), device=device, dtype=torch.float32),
            "exp_avg_fp32": torch.empty((chunk_size, numel), device=device, dtype=torch.float32),
            "exp_avg_sq_q": torch.empty((chunk_size, numel), device=device, dtype=torch.uint8),
            "exp_avg_sq_absmax": torch.empty((chunk_size, q_blocks), device=device, dtype=torch.float32),
            "exp_avg_sq_fp32": torch.empty((chunk_size, numel), device=device, dtype=torch.float32),
            "update_fp32": torch.empty((chunk_size, numel), device=device, dtype=torch.float32),
            "sum_update_partial": torch.empty((chunk_size, q_blocks), device=device, dtype=torch.float32),
            "sum_update": torch.empty((chunk_size,), device=device, dtype=torch.float32),
        }
        self._cuda_1d_batched_workspaces[key] = workspace
        return workspace

    def _step_param_1d_cuda(
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
        self._ensure_1d_state(state, grad, block_size)

        state["step"] += 1
        if self.track_cuda_rms_state or not p.is_cuda:
            state["RMS"] = self._rms(p.data)
        self._mark_1d_path_state(state, batched=False, multitensor=False)
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

    def _step_param_1d_cuda_batched_multitensor(
        self,
        items: list[tuple[torch.Tensor, dict, torch.Tensor]],
        *,
        beta1: float,
        beta2: float,
        eps0: float,
        lr: float,
        clip_threshold: float,
        weight_decay: float,
        block_size: int,
    ) -> bool:
        if not self.prefer_cuda_1d_multitensor_fast_path:
            return False
        if self.cuda_1d_batched_use_fp16_update:
            if cuda_came_full_nonfactored_step_multitensor_fp16_update_ptrs is None:
                return False
        elif cuda_came_full_nonfactored_step_multitensor_ptrs is None:
            return False

        for start in range(0, len(items), self.cuda_1d_batched_chunk_size):
            chunk = items[start : start + self.cuda_1d_batched_chunk_size]
            g32_list: list[torch.Tensor] = []

            for p, state, grad in chunk:
                self._ensure_1d_state(state, grad, block_size)
                if self.cuda_1d_batched_use_fp16_update:
                    self._ensure_batched_fp16_update_state(state, grad)
                state["step"] += 1
                if self.track_cuda_rms_state or not p.is_cuda:
                    state["RMS"] = self._rms(p.data)
                self._mark_1d_path_state(state, batched=True, multitensor=True)
                g32_list.append(self._get_grad_fp32(state, grad))

            if self.cuda_1d_batched_use_fp16_update:
                with torch.autograd.profiler.record_function("came_cuda.nonfactored_multitensor_1d_fp16_update"):
                    static_ptrs = self._get_multitensor_fp16_update_static_ptrs(chunk)
                    if cuda_fill_device_ptr_tensor is not None:
                        cuda_fill_device_ptr_tensor(tensors=g32_list, ptr_tensor=static_ptrs["g32_ptrs"])
                    else:
                        static_ptrs["g32_ptrs"].copy_(self._make_device_ptr_tensor(g32_list, chunk[0][0].device))
                    cuda_came_full_nonfactored_step_multitensor_fp16_update_ptrs(
                        sample_p=chunk[0][0].data,
                        sample_update=chunk[0][1]["_cuda_batched_update_fp16"],
                        p_ptrs=static_ptrs["p_ptrs"],
                        g32_ptrs=static_ptrs["g32_ptrs"],
                        exp_avg_q_ptrs=static_ptrs["exp_avg_q_ptrs"],
                        exp_avg_absmax_ptrs=static_ptrs["exp_avg_absmax_ptrs"],
                        exp_avg_sq_q_ptrs=static_ptrs["exp_avg_sq_q_ptrs"],
                        exp_avg_sq_absmax_ptrs=static_ptrs["exp_avg_sq_absmax_ptrs"],
                        update_ptrs=static_ptrs["update_ptrs"],
                        sum_update_ptrs=static_ptrs["sum_update_ptrs"],
                        per_item_numel=chunk[0][0].numel(),
                        beta1=beta1,
                        beta2=beta2,
                        eps0=eps0,
                        lr=lr,
                        clip_threshold=clip_threshold,
                        weight_decay=weight_decay,
                        block_size=block_size,
                    )
                continue

            with torch.autograd.profiler.record_function("came_cuda.nonfactored_multitensor_1d"):
                static_ptrs = self._get_multitensor_static_ptrs(chunk)
                if cuda_fill_device_ptr_tensor is not None:
                    cuda_fill_device_ptr_tensor(tensors=g32_list, ptr_tensor=static_ptrs["g32_ptrs"])
                else:
                    static_ptrs["g32_ptrs"].copy_(self._make_device_ptr_tensor(g32_list, chunk[0][0].device))
                cuda_came_full_nonfactored_step_multitensor_ptrs(
                    sample_p=chunk[0][0].data,
                    p_ptrs=static_ptrs["p_ptrs"],
                    g32_ptrs=static_ptrs["g32_ptrs"],
                    exp_avg_q_ptrs=static_ptrs["exp_avg_q_ptrs"],
                    exp_avg_absmax_ptrs=static_ptrs["exp_avg_absmax_ptrs"],
                    exp_avg_sq_q_ptrs=static_ptrs["exp_avg_sq_q_ptrs"],
                    exp_avg_sq_absmax_ptrs=static_ptrs["exp_avg_sq_absmax_ptrs"],
                    update_ptrs=static_ptrs["update_ptrs"],
                    sum_update_ptrs=static_ptrs["sum_update_ptrs"],
                    per_item_numel=chunk[0][0].numel(),
                    beta1=beta1,
                    beta2=beta2,
                    eps0=eps0,
                    lr=lr,
                    clip_threshold=clip_threshold,
                    weight_decay=weight_decay,
                    block_size=block_size,
                )
        return True

    def _step_param_1d_cuda_batched(
        self,
        items: list[tuple[torch.Tensor, dict, torch.Tensor]],
        *,
        beta1: float,
        beta2: float,
        eps0: float,
        lr: float,
        clip_threshold: float,
        weight_decay: float,
        block_size: int,
    ) -> None:
        if self._step_param_1d_cuda_batched_multitensor(
            items,
            beta1=beta1,
            beta2=beta2,
            eps0=eps0,
            lr=lr,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            block_size=block_size,
        ):
            return

        numel = items[0][0].numel()
        workspace = self._get_cuda_1d_batched_workspace(
            device=items[0][0].device,
            param_dtype=items[0][0].dtype,
            numel=numel,
            block_size=block_size,
        )
        q_blocks = (numel + block_size - 1) // block_size

        for start in range(0, len(items), self.cuda_1d_batched_chunk_size):
            chunk = items[start : start + self.cuda_1d_batched_chunk_size]
            active = len(chunk)
            param_batch = workspace["param"][:active]
            grad_fp32_batch = workspace["grad_fp32"][:active]
            exp_avg_q_batch = workspace["exp_avg_q"][:active]
            exp_avg_absmax_batch = workspace["exp_avg_absmax"][:active, :q_blocks]
            exp_avg_fp32_batch = workspace["exp_avg_fp32"][:active]
            exp_avg_sq_q_batch = workspace["exp_avg_sq_q"][:active]
            exp_avg_sq_absmax_batch = workspace["exp_avg_sq_absmax"][:active, :q_blocks]
            exp_avg_sq_fp32_batch = workspace["exp_avg_sq_fp32"][:active]
            update_batch = workspace["update_fp32"][:active]
            sum_update_partial_batch = workspace["sum_update_partial"][:active, :q_blocks]
            sum_update_batch = workspace["sum_update"][:active]

            for idx, (p, state, grad) in enumerate(chunk):
                self._ensure_1d_state(state, grad, block_size)
                state["step"] += 1
                if self.track_cuda_rms_state or not p.is_cuda:
                    state["RMS"] = self._rms(p.data)
                self._mark_1d_path_state(state, batched=True, multitensor=False)
                param_batch[idx].copy_(p.data)
                grad_fp32_batch[idx].copy_(grad)
                exp_avg_q_batch[idx].copy_(state["exp_avg_q"])
                exp_avg_absmax_batch[idx].copy_(state["exp_avg_absmax"])
                exp_avg_sq_q_batch[idx].copy_(state["exp_avg_sq_q"])
                exp_avg_sq_absmax_batch[idx].copy_(state["exp_avg_sq_absmax"])

            try:
                with torch.autograd.profiler.record_function("came_cuda.nonfactored_batched_1d"):
                    cuda_came_full_nonfactored_step_batched(
                        p=param_batch,
                        g32=grad_fp32_batch,
                        exp_avg_q=exp_avg_q_batch,
                        exp_avg_absmax=exp_avg_absmax_batch,
                        exp_avg_sq_q=exp_avg_sq_q_batch,
                        exp_avg_sq_absmax=exp_avg_sq_absmax_batch,
                        update=update_batch,
                        sum_update_partial=sum_update_partial_batch,
                        sum_update=sum_update_batch,
                        beta1=beta1,
                        beta2=beta2,
                        eps0=eps0,
                        lr=lr,
                        clip_threshold=clip_threshold,
                        weight_decay=weight_decay,
                        block_size=block_size,
                    )
            except AttributeError:
                self._cuda_dequantize_batched_into(
                    out=exp_avg_fp32_batch,
                    q_in=exp_avg_q_batch,
                    absmax=exp_avg_absmax_batch,
                    signed=True,
                    block_size=block_size,
                )
                self._cuda_dequantize_batched_into(
                    out=exp_avg_sq_fp32_batch,
                    q_in=exp_avg_sq_q_batch,
                    absmax=exp_avg_sq_absmax_batch,
                    signed=False,
                    block_size=block_size,
                )

                exp_avg_sq_fp32_batch.mul_(beta2)
                exp_avg_sq_fp32_batch.addcmul_(grad_fp32_batch, grad_fp32_batch, value=1.0 - beta2)
                exp_avg_sq_fp32_batch.add_((1.0 - beta2) * eps0)

                update_batch.copy_(grad_fp32_batch)
                update_batch.mul_(torch.rsqrt(torch.clamp_min(exp_avg_sq_fp32_batch, 1e-30)))
                sum_update_batch.copy_(update_batch.square().sum(dim=1))

                clip_batch = torch.sqrt(sum_update_batch / float(numel))
                clip_batch.div_(clip_threshold)
                clip_batch.clamp_min_(1.0)
                update_batch.div_(clip_batch.unsqueeze(1))

                exp_avg_fp32_batch.mul_(beta1)
                exp_avg_fp32_batch.add_(update_batch, alpha=1.0 - beta1)

                if weight_decay != 0.0:
                    param_batch.mul_(1.0 - (weight_decay * lr))
                param_batch.add_(exp_avg_fp32_batch, alpha=-lr)

                self._cuda_quantize_batched_into(
                    src=exp_avg_fp32_batch,
                    q_out=exp_avg_q_batch,
                    absmax_out=exp_avg_absmax_batch,
                    signed=True,
                    block_size=block_size,
                )
                self._cuda_quantize_batched_into(
                    src=exp_avg_sq_fp32_batch,
                    q_out=exp_avg_sq_q_batch,
                    absmax_out=exp_avg_sq_absmax_batch,
                    signed=False,
                    block_size=block_size,
                )

            for idx, (p, state, _) in enumerate(chunk):
                p.data.copy_(param_batch[idx])
                state["exp_avg_q"].copy_(exp_avg_q_batch[idx])
                state["exp_avg_absmax"].copy_(exp_avg_absmax_batch[idx])
                state["exp_avg_sq_q"].copy_(exp_avg_sq_q_batch[idx])
                state["exp_avg_sq_absmax"].copy_(exp_avg_sq_absmax_batch[idx])
                state["sum_update"].copy_(sum_update_batch[idx : idx + 1])


    @torch.no_grad()
    def prime_for_cuda_graph(self) -> None:
        for group in self.param_groups:
            block_size = group["block_size"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")
                grad = p.grad.data
                if not self._should_use_cuda_fast_path(p, grad, block_size):
                    raise RuntimeError(
                        "CUDA graph capture for CAME8bit currently requires all active parameters "
                        "to use the CUDA 2D fast path."
                    )

                p_data, _, _ = prepare_step_tensors(p, p.grad, require_contiguous=True)
                state = self.state[p]
                if len(state) == 0 or state.get("_mode") != "8bit2d":
                    state.clear()
                    init_state(state, p_data, block_size=block_size)
                validate_state_shape(state, p_data)

    def _dispatch_timing_is_enabled(self) -> bool:
        return bool(getattr(self, "_benchmark_dispatch_timing", False))

    def _dispatch_timing_add(
        self,
        times: dict[str, float] | None,
        counts: dict[str, int] | None,
        *,
        key: str,
        start_time: float,
        count: int = 1,
    ) -> None:
        if times is None or counts is None:
            return
        times[f"{key}_time_ms"] += (time.perf_counter() - start_time) * 1000.0
        counts[f"{key}_count"] += count

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        dispatch_times: dict[str, float] | None = None
        dispatch_counts: dict[str, int] | None = None
        if self._dispatch_timing_is_enabled():
            dispatch_times = defaultdict(float)
            dispatch_counts = defaultdict(int)
            self._benchmark_fallback_timing_times = {}
            self._benchmark_fallback_timing_counts = {}

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            eps0, eps1 = group["eps"]
            lr = group["lr"]
            clip_threshold = group["clip_threshold"]
            weight_decay = group["weight_decay"]
            block_size = group["block_size"]
            one_d_buckets: dict[tuple[torch.device, torch.dtype, int], list[tuple[torch.Tensor, dict, torch.Tensor]]] = (
                defaultdict(list)
            )
            param_loop_start = time.perf_counter() if dispatch_times is not None else 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                grad = p.grad.data

                # 8-bit CUDA fast path: CUDA + 2D only, and g.dtype must match p.dtype.
                phase_start = time.perf_counter() if dispatch_times is not None else 0.0
                use_2d_fast_path = self._should_use_cuda_fast_path(p, grad, block_size)
                self._dispatch_timing_add(dispatch_times, dispatch_counts, key="check_2d_fast_path", start_time=phase_start)
                if use_2d_fast_path:
                    p_data, g_data, needs_param_copyback = prepare_step_tensors(
                        p, p.grad, require_contiguous=self.cuda_graph_compatible
                    )

                    state = self.state[p]
                    if len(state) == 0 or state.get("_mode") != "8bit2d":
                        state.clear()
                        init_state(state, p_data, block_size=block_size)

                    validate_state_shape(state, p_data)

                    phase_start = time.perf_counter() if dispatch_times is not None else 0.0
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
                    self._dispatch_timing_add(dispatch_times, dispatch_counts, key="step_2d_fast_path", start_time=phase_start)

                    if needs_param_copyback:
                        p.data.copy_(p_data)
                    continue

                phase_start = time.perf_counter() if dispatch_times is not None else 0.0
                use_1d_fast_path = self._should_use_cuda_1d_fast_path(p, grad)
                self._dispatch_timing_add(dispatch_times, dispatch_counts, key="check_1d_fast_path", start_time=phase_start)
                if use_1d_fast_path:
                    phase_start = time.perf_counter() if dispatch_times is not None else 0.0
                    use_1d_batched_fast_path = self._should_use_cuda_1d_batched_fast_path(p, grad, block_size)
                    self._dispatch_timing_add(dispatch_times, dispatch_counts, key="check_1d_batched_fast_path", start_time=phase_start)
                    if use_1d_batched_fast_path:
                        one_d_buckets[(p.device, p.dtype, p.numel())].append((p, self.state[p], grad))
                        if dispatch_counts is not None:
                            dispatch_counts["enqueue_1d_bucket_count"] += 1
                        continue
                    phase_start = time.perf_counter() if dispatch_times is not None else 0.0
                    self._step_param_1d_cuda(
                        p,
                        self.state[p],
                        grad,
                        beta1=beta1,
                        beta2=beta2,
                        eps0=eps0,
                        lr=lr,
                        clip_threshold=clip_threshold,
                        weight_decay=weight_decay,
                        block_size=block_size,
                    )
                    self._dispatch_timing_add(dispatch_times, dispatch_counts, key="step_1d_fast_path", start_time=phase_start)
                    continue

                phase_start = time.perf_counter() if dispatch_times is not None else 0.0
                use_1d_full8bit_direct_path = self._should_use_cuda_1d_full8bit_direct_path(p, grad)
                self._dispatch_timing_add(
                    dispatch_times,
                    dispatch_counts,
                    key="check_1d_full8bit_direct_path",
                    start_time=phase_start,
                )
                if use_1d_full8bit_direct_path:
                    phase_start = time.perf_counter() if dispatch_times is not None else 0.0
                    direct_used = self._step_param_nonfactored_cuda_direct(
                        p,
                        self.state[p],
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
                    self._dispatch_timing_add(
                        dispatch_times,
                        dispatch_counts,
                        key="step_1d_full8bit_direct_path" if direct_used else "step_1d_full8bit_direct_path_failed",
                        start_time=phase_start,
                    )
                    if direct_used:
                        continue

                # Fallback path: full-state blockwise 8-bit, supports any shape/device.
                phase_start = time.perf_counter() if dispatch_times is not None else 0.0
                self._step_param(
                    p,
                    self.state[p],
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
                self._dispatch_timing_add(dispatch_times, dispatch_counts, key="step_fallback_full8bit", start_time=phase_start)

            self._dispatch_timing_add(dispatch_times, dispatch_counts, key="param_loop", start_time=param_loop_start, count=0)
            bucket_flush_start = time.perf_counter() if dispatch_times is not None else 0.0
            for items in one_d_buckets.values():
                if len(items) < self.cuda_1d_batched_min_bucket_size:
                    for p, state, grad in items:
                        phase_start = time.perf_counter() if dispatch_times is not None else 0.0
                        self._step_param_1d_cuda(
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
                        self._dispatch_timing_add(dispatch_times, dispatch_counts, key="step_1d_bucket_single", start_time=phase_start)
                    continue

                phase_start = time.perf_counter() if dispatch_times is not None else 0.0
                self._step_param_1d_cuda_batched(
                    items,
                    beta1=beta1,
                    beta2=beta2,
                    eps0=eps0,
                    lr=lr,
                    clip_threshold=clip_threshold,
                    weight_decay=weight_decay,
                    block_size=block_size,
                )
                self._dispatch_timing_add(dispatch_times, dispatch_counts, key="step_1d_bucket_batched", start_time=phase_start)
            self._dispatch_timing_add(dispatch_times, dispatch_counts, key="bucket_flush", start_time=bucket_flush_start, count=0)

        if dispatch_times is not None and dispatch_counts is not None:
            snapshot: dict[str, float | int] = {}
            snapshot.update(dispatch_times)
            snapshot.update(dispatch_counts)
            fallback_times = getattr(self, "_benchmark_fallback_timing_times", None)
            fallback_counts = getattr(self, "_benchmark_fallback_timing_counts", None)
            if isinstance(fallback_times, dict):
                snapshot.update(fallback_times)
            if isinstance(fallback_counts, dict):
                snapshot.update(fallback_counts)
            self._benchmark_dispatch_timing_last = snapshot
            self._benchmark_fallback_timing_times = {}
            self._benchmark_fallback_timing_counts = {}

        return loss
