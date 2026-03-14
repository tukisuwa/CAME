# CAME Fork — English Guide

For basic project information, see [`README.md`](README.md).
This document covers mode selection, installation, usage, and caveats.

## Quick Install

```bash
pip install setuptools wheel  # if not already installed
git clone https://github.com/tukisuwa/CAME.git
cd CAME
pip install -e . --no-build-isolation -v
python -c "import came_pytorch.came_cuda_ext; print('ext ok')"  # verify CUDA extension (optional)
```

> The CUDA extension check is only needed for `CAMECUDA`, `CAME8bit`, and `CAME8bitMemory`.
> You can skip it if you only use the pure-PyTorch `CAME`.
> See [Installation](#installation) for details.

## Mode Overview

This fork provides four main modes.
Choose the one that fits your goal.

| Mode | Description | CUDA Extension |
|------|-------------|----------------|
| `CAME` | Pure PyTorch implementation. Simplest option | Not required |
| `CAMECUDA` | Accelerated with CUDA kernels. **Best for speed** | Required |
| `CAME8bit` | 8-bit quantization balancing speed and VRAM | Required |
| `CAME8bitMemory` | 8-bit state + shared scratch for **minimal VRAM** | Required |

### Which one should I use?

Measured on a local `sd-scripts` SDXL LoRA run (same 300-step setup):

| Mode | Time | VRAM |
|------|------|------|
| `CAME` | `12:01` | `6.0 GB` |
| `CAMECUDA` | `08:32` | `6.2 GB` |
| `CAME8bit` | `08:47` | `6.1 GB` |
| `CAME8bitMemory` | `09:48` | `5.8 GB` |

- Maximum speed → `CAMECUDA`
- Balance of speed and VRAM → `CAME8bit`
- Minimize VRAM usage → `CAME8bitMemory`
- No CUDA extension needed → `CAME`

Numbers are environment-dependent but reflect the intended mode split.

## Installation

This guide assumes you already cloned this repository locally.
From the repository root, install with:

```bash
pip install -e . --no-build-isolation -v
```

The `--no-build-isolation` flag lets the build reference the `torch` already in your environment.

> **Note**: `setuptools` and `wheel` must be present in your environment.
> If missing, run `pip install setuptools wheel` first.

### Building the CUDA Extension (required for all modes except CAME)

`CAMECUDA`, `CAME8bit`, and `CAME8bitMemory` require the CUDA extension.
If you only need the pure-PyTorch `CAME`, you can skip this section.

#### Prerequisites

- NVIDIA GPU
- CUDA-enabled PyTorch
- CUDA Toolkit (must match PyTorch's CUDA major version — e.g. if `torch.version.cuda` is `12.6`, use a CUDA 12.x Toolkit)
- C++ compiler (on Windows: Visual Studio 2022 Build Tools)

#### Check PyTorch CUDA support

```bash
python -c "import torch; print('torch:', torch.__version__); print('torch.version.cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

- `torch.version.cuda` is `None` → You have CPU-only PyTorch. Reinstall with CUDA support.
- `torch.cuda.is_available()` is `False` → Check your GPU driver and CUDA setup.

#### Build

The install command above (`pip install -e . --no-build-isolation -v`) automatically builds the CUDA extension when CUDA-enabled PyTorch is present.

To build manually:

```bat
python setup.py build_ext --inplace
```

On Windows, if the compiler is not detected, first open `x64 Native Tools Command Prompt for VS 2022` or run `vcvars64.bat`.

#### Verify the CUDA extension was built

```bash
python -c "import came_pytorch.came_cuda_ext; print('ext ok')"
```

If this raises an error or `ModuleNotFoundError`, the extension was not built.

> **Note**: A successful `import came_pytorch` does not guarantee `came_cuda_ext` was built.
> The CUDA extension is lazily loaded only when CUDA-specific code paths are used.

If the extension is not pre-built, a JIT build will be attempted when `ninja` is available.

#### Common mistakes

- Using CPU-only PyTorch and expecting the CUDA build to work
- Installing CUDA Toolkit but not Visual Studio Build Tools (Windows)
- Mismatched CUDA Toolkit and PyTorch major versions

## Usage

### Basic Modes

**CAME** (pure PyTorch, no CUDA extension required):

```python
from came_pytorch import CAME

optimizer = CAME(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.999, 0.9999),
    eps=(1e-30, 1e-16),
)
```

**CAMECUDA** (speed-first):

```python
from came_pytorch import CAMECUDA

optimizer = CAMECUDA(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
)
```

**CAME8bit** (balanced speed and VRAM):

```python
from came_pytorch import CAME8bit

optimizer = CAME8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
)
```

**CAME8bitMemory** (minimal VRAM):

```python
from came_pytorch import CAME8bitMemory

optimizer = CAME8bitMemory(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
)
```

### Advanced Options

The basic modes above are sufficient for most use cases.
The following variants are for specific needs.

**CAME8bitFull** — Directly use the underlying implementation of `CAME8bit`.
`CAME8bit` delegates to this class internally; using `CAME8bit` is normally sufficient:

```python
from came_pytorch import CAME8bitFull

optimizer = CAME8bitFull(model.parameters(), lr=2e-4, weight_decay=1e-2)
```

**CAME8bit2D** — CUDA fast path for 2-D parameters only.
Only works with 2-D CUDA tensors:

```python
from came_pytorch import CAME8bit2D

optimizer = CAME8bit2D(model.parameters(), lr=2e-4, weight_decay=1e-2)
```

**Disable CUDA fast path** — Force `CAME8bit` to use the reference implementation
(`CAME8bitFull`) instead of the CUDA fast path:

```python
optimizer = CAME8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
    prefer_cuda_fast_path=False,
)
```

## Caveats

- Sparse gradients are not supported.
- `CAME8bit2D` is for 2-D parameters on CUDA only.
- 8-bit state layout is fixed after first use. Parameter resizing is not supported.
- The public API of the 8-bit variants may change as this fork evolves.
- This fork's package metadata is intentionally marked as a preview release.
