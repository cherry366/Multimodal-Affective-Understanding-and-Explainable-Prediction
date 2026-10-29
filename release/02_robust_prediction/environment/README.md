# 模块二运行环境与依赖说明

本目录的 `requirements.txt` 是 MoDE-Former+ v2.1 的完整依赖清单：

```bash
python -m pip install -r environment/requirements.txt
```

## 一、实测环境

| 项目 | 版本 |
|---|---|
| 操作系统 | Windows 10 (10.0.19045) |
| Python | 3.10.13（conda env：`<你的Python环境>`） |
| GPU 运行时 | CUDA 11.8（`torch==2.0.0+cu118`），无 GPU 时自动回落到 CPU |

关键依赖版本（均已在本机逐条核对）：

```
torch==2.0.0+cu118        torchvision==0.15.1+cu118   torchaudio==2.0.1+cu118
transformers==4.44.2      tokenizers==0.19.1          huggingface-hub==0.36.2
scikit-learn==1.3.0       numpy==1.26.4               scipy==1.11.4
pandas==2.1.4             joblib==1.2.0               tqdm==4.65.0
matplotlib==3.8.0         seaborn==0.12.2             einops==0.8.2
openpyxl==3.0.10          nltk==3.8.1                 pillow==10.2.0
```

## 二、安装与自检

```bash
python -m pip install -r requirements.txt
# GPU 版 torch 需按 CUDA 版本单独装：
# pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# 自检（打印实际版本，与上表逐条对照）
python -c "import torch,sklearn,transformers,numpy,pandas;print(torch.__version__, sklearn.__version__, transformers.__version__, numpy.__version__, pandas.__version__)"
```

## 三、两点注意

1. **文本编码器离线缓存**：模块二 的文本分支使用 HuggingFace `bert-base-uncased`，
   离线缓存位于 `%USERPROFILE%\.cache\huggingface\hub\models--bert-base-uncased`。
   若目标机器无法联网，请把该缓存目录一并拷贝过去，或设置
   `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` 后从缓存加载。
2. **transformers 不要升到 5.x**（5.x 要求 `torch>=2.5`，与本项目固定的 `torch==2.0.0+cu118` 冲突）。
