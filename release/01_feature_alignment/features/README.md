# 模块一 自生成特征文件说明（100 条样本）

本目录是模块一"特征提取与时序对齐"的**全量特征文件**，由 `code/pipeline.py` 生成，
覆盖全部 100 条样本，可直接用于后续模块，无需重新跑特征提取。

## 一、文件清单

| 文件 | 大小 | 说明 |
|---|---|---|
| `q1_features_timebin.pkl.gz` | 13.50 MB | **时间箱版（主版本）**：语音/视觉按有效时长分箱，掩码可反查真实时间 |
| `q1_features_word.pkl.gz` | 11.90 MB | **词级版**：与标准化特征 `aligned_50.pkl` 的 50 槽位口径兼容 |
| `summary.csv` | 19 KB | 100 条样本的提取结果全量汇总表（见第三节） |
| `load_q1_features.py` | — | 读取示例（同时支持 `.pkl` 与 `.pkl.gz`） |
| `restore_pkl.py` | — | 需要恢复成原始 `.pkl` 文件名时使用（解压后与原文件逐字节一致） |

> **为什么是 `.pkl.gz` 而不是 `.pkl`**
> 两个原始 pkl 各约 33.53 MB，合计 **67.06 MB**，直接入库会让仓库体积翻倍；
> gzip 无损压缩后合计 **25.40 MB**。需要原文件名时直接跑 `python restore_pkl.py`，内容一字不差。

## 二、数据结构（Pickle 字典）

顶层只有 `test` 一个键（100 条样本统一放在 `test` 下），即 `d = pickle.load(...)["test"]`。
`d` 的字段如下（两版字段一致，词级版多一个 `length`）：

| 键 | 形状 / 类型 | 含义 |
|---|---|---|
| `id` | list，100 | 样本编号，形如 `-3g5yACwYnA$13`（`video_id$clip_id`） |
| `raw_text` | list，100 | 原始英文转写文本 |
| `text` | (100, 50, 768) float32 | 文本特征（BERT 表示，50 个槽位） |
| `audio` | (100, 50, 74) float32 | 语音特征（时间箱版） |
| `vision` | (100, 50, 35) float32 | 视觉特征（时间箱版） |
| `text_mask` / `audio_mask` / `vision_mask` | (100, 50) float32 | 1 = 有效槽位，0 = 填充位 |
| `duration` | (100,) float64 | 原始有效时长（秒） |
| `n_audio_frames` / `n_vision_frames` | (100,) int32 | 原始语音/视觉帧数（用可反查时长） |
| `align_mode` | list，100 | 对齐模式（如 `whisper_seq`） |
| `annotations` | list，100 | 文字极性标注（Negative/Neutral/Positive） |
| `classification_labels` | list，100 | 三分类标签（0/1/2） |
| `regression_labels` | list，100 | 连续情感强度 |
| `text_norm` / `audio_norm` / `vision_norm` | 与原始同形 float32 | **归一化版本**，可直接送训练 |
| `length`（仅词级版） | (100,) int32 | 词级有效长度 |

> 说明：归一化特征**不是单独的文件**，而是同一个 pkl 内的 `*_norm` 字段，无需额外生成。

## 三、读取方式

```python
import pickle, gzip
d = pickle.load(gzip.open("q1_features_timebin.pkl.gz", "rb"))["test"]
text, tmask = d["text"], d["text_mask"]      # (100,50,768) / (100,50)
print(len(d["id"]), d["audio"].shape)        # 100 (100,50,74)
```

或直接用本目录的示例脚本：

```bash
python load_q1_features.py q1_features_timebin.pkl.gz
python restore_pkl.py          # 需要 .pkl 原始文件名时
```

## 四、summary.csv 字段

100 行，1 行 1 条样本，共 25 列，覆盖样本编号、模态类型、原始有效时长、
特征维度、对齐粒度等关键信息：

`id, duration_s, n_words, text_align, asr_model, n_audio_frames, n_vision_frames,
vision_backend, n_vision_valid, vision_valid_ratio, timebin_audio_valid,
timebin_vision_valid, timebin_text_valid, fill_ratio_audio, word_valid_length, dim,
sync_score, peak_lag_frames, peak_lag_sec, sharpness, valid_ratio, frame_dt,
corr_text_audio, corr_text_vision, corr_audio_vision`

其中 `sync_score / peak_lag_frames / peak_lag_sec / sharpness` 为跨模态同步核验量，
`corr_text_audio / corr_text_vision / corr_audio_vision` 为三模态相关度，
`dim` 记录文本/语音/视觉的特征维度。
