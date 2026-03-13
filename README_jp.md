# CAME フォーク — 日本語ガイド

基本的なプロジェクト情報については [`README.md`](README.md) を参照してください。
このドキュメントでは、インストール手順・CUDA ビルド・使い方・注意事項を説明します。

## インストール

このリポジトリの利用方法は 2 通りあります:

- CAME そのものを試すだけなら、通常のインストールで十分です。
- CUDA 8bit パスを使いたい場合は、追加のローカルビルド手順が必要です。

リポジトリルートからインストール:

```bash
pip install .
```

GitHub から直接インストール:

```bash
pip install git+https://github.com/tukisuwa/CAME.git@test2
```


## 初心者向け CUDA ビルドガイド

このセクションは、`CAME8bit`、`CAME8bitFull`、`CAME8bit2D` を
CUDA サポート付きで使いたい場合にのみ参照してください。

### 1. PyTorch が CUDA をサポートしているか確認する

以下を実行してください:

```bash
python -c "import torch; print('torch:', torch.__version__); print('torch.version.cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
```

結果の読み方:

- `torch.version.cuda` が `None` の場合、CPU 専用の PyTorch がインストールされています。
- `torch.cuda.is_available()` が `False` の場合、PyTorch は現在 GPU を使用できません。
- 両方とも正常であれば、拡張のビルドに進めます。

### 2. 必要なものを理解する

CUDA 拡張のビルドには一般的に以下が必要です:

- NVIDIA GPU
- CUDA 対応の PyTorch インストール
- CUDA Toolkit のインストール
- C++ コンパイラ

Windows での前提条件:

- Visual Studio 2022 Build Tools
- NVIDIA CUDA Toolkit
- CUDA 対応の PyTorch

CUDA Toolkit は **PyTorch の CUDA メジャーバージョン** と一致させる必要があります。
例: `torch.version.cuda` が `12.6` の場合、CUDA 12.x Toolkit が正しいファミリーです。

### 3. CUDA ビルドを有効にしてパッケージをインストールする

CUDA 対応の PyTorch が現在の環境にインストール済みの場合:

```bash
pip install -e . --no-build-isolation
```

`--no-build-isolation` を使う理由:

- ビルド時に、環境内の既存の PyTorch インストールを参照する必要があります。
- これがないと、`pip` は `torch` が利用できない隔離環境でビルドを行い、
  CUDA 拡張がスキップされる可能性があります。

GitHub から同様にインストールすることもできます:

```bash
pip install git+https://github.com/tukisuwa/CAME.git@test2 --no-build-isolation
```

> **注意**: `--no-build-isolation` を使用する場合、ビルドに必要な `setuptools` と
> `wheel` が環境に存在している必要があります。不足している場合は先に
> `pip install setuptools wheel` を実行してください。

### 4. ローカル開発用の手動ビルド

リポジトリルートから拡張を明示的にビルドしたい場合:

```bat
python setup.py build_ext --inplace
```

Windows でコンパイラが検出されない場合、先に以下のいずれかを開いてください:

- `x64 Native Tools Command Prompt for VS 2022`
- または Visual Studio Build Tools の `vcvars64.bat` を実行してからコマンドを実行

### 5. 拡張が使用可能か確認する

以下を実行してください:

```bash
python -c "import torch; import came_pytorch; from came_pytorch import CAME8bit; print('cuda available:', torch.cuda.is_available()); print('ok')"
```

インポートが成功すれば、パッケージ自体は正しくインストールされています。
CUDA 拡張は、CUDA 固有のコードパスが使用されるときに遅延ロードされます。

### よくあるミス

- CPU 専用の PyTorch をインストールして CUDA ビルドが動作すると期待する
- Windows で CUDA Toolkit はインストールしたが Visual Studio Build Tools をインストールしていない
- 間違った CUDA メジャーバージョンファミリーの Toolkit を使用する
- `pip install .` を実行しただけで CUDA 拡張が確実にコンパイルされたと思い込む
- CPU テンソルや非 2D テンソルで `CAME8bit2D` を使おうとする

拡張がプリビルドされていない場合、`came_pytorch.came_cuda` は `ninja` が利用可能な
ときに JIT ビルドを試みることもできます。

## 使い方

純 PyTorch CAME:

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

単一エントリの 8bit オプティマイザ:

```python
from came_pytorch import CAME8bit

optimizer = CAME8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
)
```

リファレンスの完全 8bit パスを強制する:

```python
from came_pytorch import CAME8bit

optimizer = CAME8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
    prefer_cuda_fast_path=False,
)
```

明示的に全状態 8bit 実装を使用する:

```python
from came_pytorch import CAME8bitFull

optimizer = CAME8bitFull(model.parameters(), lr=2e-4, weight_decay=1e-2)
```

CUDA 専用 2D 高速パスを使用する:

```python
from came_pytorch import CAME8bit2D

optimizer = CAME8bit2D(model.parameters(), lr=2e-4, weight_decay=1e-2)
```

## 注意事項

- 純 PyTorch の `CAME` が最も簡単なエントリポイントで、CUDA 拡張は不要です。
- スパース勾配はサポートされていません。
- `CAME8bit2D` は CUDA 上の 2D パラメータ専用です。
- 8bit 状態のレイアウトは初回使用後に固定されるため、パラメータのリサイズは
  サポートされていません。
- 8bit バリアントの公開 API は、本フォークの改良に伴い変更される可能性があります。
- 本フォークのパッケージメタデータは、意図的にプレビューリリースとして
  マーキングされています。
