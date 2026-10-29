# 模块一：多模态情感特征提取与时序对齐

> 完整的方法推导、参数、实测结果与踩坑记录见 **`reproduction.md`**（唯一活文档）。
> 本 README 只讲「怎么在自己的机器上把它跑起来」。

---

## 1. 目录结构（先看这里）

本模块是自包含的：代码在 `code/`，数据放 `data/`，输出写 `q1_output/`。

```
01_feature_alignment/
├── code/                     # config.py  pipeline.py  figures.py  vector_figs.py
│                             # visualize.py  visualize_aligned.py  single_video.py
│                             # load_q1_features.py
├── features/                 # 已发布的 100 条特征文件（.pkl.gz + summary.csv + 对齐记录）
├── environment/requirements.txt
├── docs/                     # reproduction.md（方法+实测）  README.md（本文件）
├── data/raw/                 # 〔需自备〕100 个 mp4 + label-100.xlsx
└── q1_output/                # 运行后自动生成
```

`config.py` 会把「`code/` 的上一级」识别为模块根目录，正常情况下**不需要改任何路径**。
需要把数据放到别处时，用环境变量指定：

```powershell
$env:Q1_ROOT = "D:\work\01_feature_alignment"
```

---

## 2. 安装 Python 依赖

```powershell
python -m pip install -r requirements.txt
```

**torch 要按你的 CUDA 版本单独装**（上面 requirements 里只是兜底）：

```powershell
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

> ⚠️ **transformers 必须锁 4.44.2**。装成 5.x 会报
> `Disabling PyTorch because PyTorch >= 2.5 is required`，导致 BERT 用不了。

首次运行会自动下载 3 个模型（共约 1 GB）：`bert-base-uncased`(420MB)、
`whisper base`(139MB)、`whisper small`(461MB)。国内建议设镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

---

## 3. OpenFace 2.2.0（视觉 35 维必需）

1. 下载 **OpenFace 2.2.0 win x64** 压缩包并解压，例如到 `E:\tools\OpenFace_2.2.0_win_x64`。
2. 让代码找到它——二选一：
   - 设环境变量：`$env:OPENFACE_DIR = "你的\OpenFace_2.2.0_win_x64"`
   - 或直接改 `config.py` 里 `OPENFACE_DIR` 的默认值。
3. 解压后**不要删任何文件**。

> ⚠️ 标准二进制包缺 CEN patch experts（`cen_patches_*.dat`，仅托管在 Dropbox/OneDrive，国内通常下不动）。
> 本代码已内置规避方案：用 `-mloc model/main_clnf_general.txt`（CLNF 模型，其权重随包自带），
> **不需要**再额外下载任何模型文件。

---

## 4. 运行

### 方式 A：命令行

```powershell
cd <模块根目录>\code
$env:HF_HUB_OFFLINE="1"        # 模型已下载后，禁止联网检查，避免卡住
$env:PYTHONIOENCODING="utf-8"

python -u pipeline.py all      # 全流程（首次 20~30 分钟）
python -u vector_figs.py        # 生成 9 张矢量图 PDF
```

### 方式 B：Spyder

```python
%cd <模块根目录>\code
import os; os.environ["HF_HUB_OFFLINE"] = "1"
%run pipeline.py all
%run vector_figs.py
```

分步跑（调试用）：

```python
%run pipeline.py audio      # 抽音频 + 语音 74 维
%run pipeline.py vision     # OpenFace 视觉 35 维
%run pipeline.py text       # BERT 768 维 + Whisper 词时间戳
%run pipeline.py build      # 对齐 + 预处理 + 打包
%run pipeline.py summary    # 六项校验 + 相关性/同步质量
```

---

## 5. 产物在哪

**已发布的全量特征文件**在 `features/`：`q1_features_timebin.pkl.gz`、
`q1_features_word.pkl.gz`、`summary.csv`、`align_records.json`，附读取示例 `load_q1_features.py`。
不重跑流程时，直接读这里即可。

重跑时全部写入 `q1_output/`：

| 产物 | 路径 | 说明 |
|---|---|---|
| 主特征文件 | `q1_features_timebin.pkl` | 三模态对齐到 50 时间箱（含 `*_norm` 标准化版） |
| 词级特征文件 | `q1_features_word.pkl` | 与 `aligned_50` 的 50 槽位口径兼容 |
| 汇总表 | `summary.csv` | 100 行 × 25 列（含 7 个质量列） |
| 对齐记录 | `align_records.json` | 逐样本词时间戳、有效帧数、对齐置信度 |
| 预览图 | `figs/*.png` | 4 张（快速看） |
| **矢量图** | `vector_figs/*.pdf` | **9 张矢量图（中文已嵌入），可直接插 Word/LaTeX** |

`vector_figs/图清单.md` 里写了每张图的建议图题和对应说明。

---

## 6. 时间参考（本机实测）

| 阶段 | 首次 | 有缓存 |
|---|---|---|
| audio 抽音频 + 74 维 | 5~10 分钟 | 秒级跳过 |
| vision OpenFace 35 维 | ~12 分钟 | 秒级跳过 |
| text BERT + Whisper | 3~5 分钟 | 秒级跳过 |
| build 对齐 + 预处理 | ~15 秒 | 13 秒 |
| summary 校验 + 质量分析 | ~5 秒 | 3.5 秒 |
| **合计** | **约 20~30 分钟** | **约 30 秒** |

---

## 7. 常见问题

| 现象 | 解决 |
|---|---|
| `FileNotFoundError: [WinError 2]`（ffmpeg） | 已内置 `imageio-ffmpeg` 自动定位；确认它装了 |
| `No module named 'cv2'` | `pip install opencv-python==4.10.0.84` |
| `BertModel requires the PyTorch library` | transformers 版本不对，锁回 `4.44.2` |
| `import librosa` 卡死（CPU 占满一核） | 已内置 numba 缓存重定向到工作区，无需处理 |
| 抽帧目录为空 / 视觉只有 1 帧 | 中文路径下 `cv2.imwrite` 静默失败，代码已改用 `imencode+tofile` |
| OpenFace 报缺 `cen_patches_*.dat` | 已内置 `-mloc model/main_clnf_general.txt` 规避 |
| Whisper 把英文听写成其它语言 | 代码已强制 `language="en"` |
| `summary.csv` 写不进去 | 你可能在 Excel 里开着它，关掉重跑 |
