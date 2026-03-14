from pathlib import Path

import setuptools


def _get_ext_modules():
    try:
        import os
        from pathlib import Path

        import torch
    except Exception:
        return [], {}

    if torch.version.cuda is None:
        return [], {}

    # Prefer a CUDA Toolkit that matches PyTorch's CUDA *major* version (minor mismatch is OK).
    # If an exact match isn't installed, pick the highest available minor for that major that has nvcc.
    cuda_ver = str(torch.version.cuda)  # e.g. "12.6"
    major_minor = cuda_ver.split(".")[:2]
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    chosen = None
    if len(major_minor) == 2 and base.exists():
        major = major_minor[0]
        exact = base / f"v{major_minor[0]}.{major_minor[1]}"
        if (exact / "bin" / "nvcc.exe").exists():
            chosen = exact
        else:
            candidates = []
            for p in base.glob(f"v{major}.*"):
                nvcc = p / "bin" / "nvcc.exe"
                if nvcc.exists():
                    try:
                        minor = int(p.name.split(".")[1])
                        candidates.append((minor, p))
                    except Exception:
                        pass
            if candidates:
                chosen = sorted(candidates, key=lambda x: x[0])[-1][1]

    if chosen is not None:
        os.environ["CUDA_HOME"] = str(chosen)
        os.environ["CUDA_PATH"] = str(chosen)
        bin_dir = chosen / "bin"
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    except Exception:
        return [], {}

    ext_modules = [
        CUDAExtension(
            name="came_pytorch.came_cuda_ext",
            sources=[
                "came_pytorch/came_cuda/csrc/came_bindings.cpp",
                "came_pytorch/came_cuda/csrc/came_kernels.cu",
            ],
            extra_compile_args={
                "cxx": ["/std:c++17"] if os.name == "nt" else ["-std=c++17"],
                "nvcc": ["--use_fast_math", "-lineinfo"],
            },
        )
    ]

    cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=False)}
    return ext_modules, cmdclass

ROOT = Path(__file__).resolve().parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8")

ext_modules, cmdclass = _get_ext_modules()

setuptools.setup(
    name="came-pytorch",
    license="MIT",
    version="0.1.3.post1",
    author="Yang Luo and fork contributors",
    maintainer="tukisuwa",
    description="Prototype fork of the CAME optimizer for PyTorch with experimental 8-bit CUDA paths",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tukisuwa/CAME",
    project_urls={
        "Source": "https://github.com/tukisuwa/CAME",
        "Upstream": "https://github.com/yangluo7/CAME",
        "Paper": "https://arxiv.org/abs/2307.02047",
    },
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        "came_pytorch": [
            "came_cuda/csrc/*.cpp",
            "came_cuda/csrc/*.cu",
        ]
    },
    keywords=[
        "artificial intelligence",
        "deep learning",
        "optimizers",
        "memory efficient",
        "cuda",
        "8-bit",
    ],
    install_requires=[
        "torch>=1.13"
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
)
