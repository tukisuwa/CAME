# CAME Fork — English Guide

For basic project information, see [`README.md`](README.md).
This document covers installation, CUDA build instructions, usage, and caveats.

## Installation

This guide assumes you already cloned this repository locally.

From the repository root, install with:

```bash
pip install -e . --no-build-isolation -v
```

This is the recommended default for this fork. It keeps the repository on disk,
builds against the `torch` already present in your environment, and makes local
rebuilds easier.


## CUDA Build Guide for Beginners

This section is only relevant if you want to use `CAME8bit`, `CAME8bitFull`, or `CAME8bit2D`
with CUDA support.

### 1. Check that PyTorch supports CUDA

Run the following:

```bash
python -c "import torch; print('torch:', torch.__version__); print('torch.version.cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

How to read the output:

- If `torch.version.cuda` is `None`, you have a CPU-only PyTorch installation.
- If `torch.cuda.is_available()` is `False`, PyTorch cannot currently use a GPU.
- If both are OK, you can proceed to building the extension.

### 2. Understand the requirements

Building the CUDA extension generally requires:

- An NVIDIA GPU
- A CUDA-enabled PyTorch installation
- The CUDA Toolkit
- A C++ compiler

Windows prerequisites:

- Visual Studio 2022 Build Tools
- NVIDIA CUDA Toolkit
- CUDA-enabled PyTorch

The CUDA Toolkit must match **the CUDA major version of your PyTorch**.
For example, if `torch.version.cuda` is `12.6`, you need a CUDA 12.x Toolkit.

### 3. Install the package with CUDA build enabled

If CUDA-enabled PyTorch is already installed in your environment, install from the
repository root with:

```bash
pip install -e . --no-build-isolation -v
```

Why `--no-build-isolation`:

- The build needs to reference the existing PyTorch installation in your environment.
- Without this flag, `pip` builds in an isolated environment where `torch` is not
  available, and the CUDA extension may be silently skipped.

> **Note**: When using `--no-build-isolation`, the build dependencies `setuptools` and
> `wheel` must already be present in your environment. If they are missing, run
> `pip install setuptools wheel` first.

### 4. Manual build for local development

To explicitly build the extension from the repository root:

```bat
python setup.py build_ext --inplace
```

On Windows, if the compiler is not detected, first open one of the following:

- `x64 Native Tools Command Prompt for VS 2022`
- Or run `vcvars64.bat` from Visual Studio Build Tools before running the command

### 5. Verify that the extension is available

Run the following:

```bash
python -c "import importlib.util; print(importlib.util.find_spec('came_pytorch.came_cuda_ext'))"
```

If this prints `None`, the CUDA extension is not available in the current environment.

You can also verify by importing the extension directly:

```bash
python -c "import came_pytorch.came_cuda_ext; print('ext ok')"
```

Note that a check like:

```bash
python -c "import torch; import came_pytorch; from came_pytorch import CAME8bit; print('cuda available:', torch.cuda.is_available()); print('ok')"
```

only proves that PyTorch CUDA is available and that the package can be imported.
It does **not** prove that `came_cuda_ext` was built. The CUDA extension is lazily loaded when
CUDA-specific code paths are used.

### Common mistakes

- Installing CPU-only PyTorch and expecting the CUDA build to work
- Installing the CUDA Toolkit on Windows but not Visual Studio Build Tools
- Using a CUDA Toolkit from the wrong major version family
- Assuming `pip install .` alone guarantees the CUDA extension was compiled
- Assuming `import came_pytorch` or `torch.cuda.is_available()` proves that `came_cuda_ext` was built
- Trying to use `CAME8bit2D` with CPU tensors or non-2D tensors

If the extension is not pre-built, `came_pytorch.came_cuda` can also attempt a
JIT build when `ninja` is available.

## Usage

Pure-PyTorch CAME:

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

Single-entry 8-bit optimizer:

```python
from came_pytorch import CAME8bit

optimizer = CAME8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
)
```

Force the reference full 8-bit path:

```python
from came_pytorch import CAME8bit

optimizer = CAME8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
    prefer_cuda_fast_path=False,
)
```

Explicitly use the full-state 8-bit implementation:

```python
from came_pytorch import CAME8bitFull

optimizer = CAME8bitFull(model.parameters(), lr=2e-4, weight_decay=1e-2)
```

Use the CUDA-only 2-D fast path:

```python
from came_pytorch import CAME8bit2D

optimizer = CAME8bit2D(model.parameters(), lr=2e-4, weight_decay=1e-2)
```

## Caveats

- The pure-PyTorch `CAME` is the simplest entry point and requires no CUDA extension.
- Sparse gradients are not supported.
- `CAME8bit2D` is for 2-D parameters on CUDA only.
- 8-bit state layout is fixed after first use; parameter resizing is not supported.
- The public API of the 8-bit variants may change as this fork evolves.
- This fork's package metadata is intentionally marked as a preview release.
