# CAME フォーク — 日本語ガイド

基本的なプロジェクト情報については [`README.md`](README.md) を参照してください。
このドキュメントでは、モードの選び方・インストール・使い方・注意事項を説明します。

## モード概要

このフォークには 4 つの主要モードがあります。
目的に応じて使い分けてください。

| モード | 特徴 | CUDA 拡張 |
|--------|------|-----------|
| `CAME` | 純 PyTorch 実装。最もシンプル | 不要 |
| `CAMECUDA` | CUDA カーネルで高速化。**速度重視ならこれ** | 必要 |
| `CAME8bit` | 8bit 量子化で速度と VRAM のバランスを取る | 必要 |
| `CAME8bitMemory` | 8bit 状態 + 共有スクラッチで **VRAM を最小化** | 必要 |

### どれを選ぶ？

`sd-scripts` SDXL LoRA（同一 300 ステップ設定）での実測例:

| モード | 時間 | VRAM |
|--------|------|------|
| `CAME` | `12:01` | `6.0 GB` |
| `CAMECUDA` | `08:32` | `6.2 GB` |
| `CAME8bit` | `08:47` | `6.1 GB` |
| `CAME8bitMemory` | `09:48` | `5.8 GB` |

- とにかく速くしたい → `CAMECUDA`
- 速度と VRAM のバランス → `CAME8bit`
- VRAM を最小限に抑えたい → `CAME8bitMemory`
- CUDA 拡張なしで使いたい → `CAME`

数値は環境に依存しますが、傾向の参考にしてください。

## インストール

リポジトリを clone 済みであることを前提とします。
リポジトリルートで次を実行してください:

```bash
pip install -e . --no-build-isolation -v
```

手元の環境の `torch` を参照してビルドするため、`--no-build-isolation` を付けています。

> **注意**: `setuptools` と `wheel` が環境に必要です。
> 不足している場合は先に `pip install setuptools wheel` を実行してください。

### CUDA 拡張のビルド（CAME 以外を使う場合）

`CAMECUDA`・`CAME8bit`・`CAME8bitMemory` を使うには CUDA 拡張のビルドが必要です。
純 PyTorch の `CAME` だけを使う場合、このセクションは飛ばして構いません。

#### 前提条件

- NVIDIA GPU
- CUDA 対応の PyTorch
- CUDA Toolkit（PyTorch の CUDA メジャーバージョンと一致させること。例: `torch.version.cuda` が `12.6` なら CUDA 12.x Toolkit）
- C++ コンパイラ（Windows では Visual Studio 2022 Build Tools）

#### PyTorch の CUDA 対応を確認する

```bash
python -c "import torch; print('torch:', torch.__version__); print('torch.version.cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

- `torch.version.cuda` が `None` → CPU 専用の PyTorch です。CUDA 対応版を再インストールしてください。
- `torch.cuda.is_available()` が `False` → GPU ドライバや CUDA の設定を確認してください。

#### ビルドの実行

上記のインストールコマンド（`pip install -e . --no-build-isolation -v`）を CUDA 対応 PyTorch がある環境で実行すれば、CUDA 拡張も自動的にビルドされます。

手動でビルドしたい場合:

```bat
python setup.py build_ext --inplace
```

Windows でコンパイラが検出されない場合は、先に `x64 Native Tools Command Prompt for VS 2022` を開くか、`vcvars64.bat` を実行してください。

#### CUDA 拡張がビルドされたか確認する

```bash
python -c "import came_pytorch.came_cuda_ext; print('ext ok')"
```

エラーや `ModuleNotFoundError` が出る場合、拡張はビルドされていません。

> **注意**: `import came_pytorch` が成功しても、`came_cuda_ext` がビルド済みとは限りません。
> CUDA 拡張は実際に使われるまで遅延ロードされるためです。

拡張がプリビルドされていない場合でも、`ninja` が利用可能であれば JIT ビルドが試みられます。

#### よくあるミス

- CPU 専用の PyTorch で CUDA ビルドを期待する
- CUDA Toolkit だけ入れて Visual Studio Build Tools を入れていない（Windows）
- PyTorch と CUDA Toolkit のメジャーバージョンが不一致

## 使い方

### 基本モード

**CAME**（純 PyTorch、CUDA 拡張不要）:

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

**CAMECUDA**（速度重視）:

```python
from came_pytorch import CAMECUDA

optimizer = CAMECUDA(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
)
```

**CAME8bit**（速度と VRAM のバランス）:

```python
from came_pytorch import CAME8bit

optimizer = CAME8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
)
```

**CAME8bitMemory**（VRAM 最小化）:

```python
from came_pytorch import CAME8bitMemory

optimizer = CAME8bitMemory(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
)
```

### 上級オプション

通常は上記の基本モードで十分です。
以下は特定の用途向けのバリアントです。

**CAME8bitFull** — `CAME8bit` の内部実装を直接使用する。
`CAME8bit` は内部的にこのクラスへ委譲しており、通常は `CAME8bit` 経由で使えば十分です:

```python
from came_pytorch import CAME8bitFull

optimizer = CAME8bitFull(model.parameters(), lr=2e-4, weight_decay=1e-2)
```

**CAME8bit2D** — 2D パラメータ専用の CUDA 高速パス。
2D CUDA テンソルのみ対応です:

```python
from came_pytorch import CAME8bit2D

optimizer = CAME8bit2D(model.parameters(), lr=2e-4, weight_decay=1e-2)
```

**CUDA 高速パスの無効化** — `CAME8bit` で CUDA 高速パスを使わず、
リファレンス実装（`CAME8bitFull`）を強制する:

```python
optimizer = CAME8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
    prefer_cuda_fast_path=False,
)
```

## 注意事項

- スパース勾配はサポートされていません。
- `CAME8bit2D` は CUDA 上の 2D パラメータ専用です。
- 8bit 状態のレイアウトは初回使用後に固定されます。パラメータのリサイズはサポートされていません。
- 8bit バリアントの公開 API は、本フォークの改良に伴い変更される可能性があります。
- 本フォークのパッケージメタデータは、意図的にプレビューリリースとしてマーキングされています。
