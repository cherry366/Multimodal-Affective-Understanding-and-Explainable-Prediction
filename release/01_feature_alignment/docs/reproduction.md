# 模块一 实验操作手册（多模态情感特征提取与时序对齐）

> 本手册配套 `config.py`、`pipeline.py`、`visualize.py` 三个脚本使用。
> 目标一句话：**把原始视频的100条视频，整理成三张"对齐到50个位置"的表格（文字768维/声音74维/画面35维），并附上有效标记(mask)，打包成可核验的特征文件。**
> 本手册是**唯一活文档**：后续讨论有改进，一律直接改本文件，不另开新文件。

---

## 文档状态

| 项目 | 内容 |
|---|---|
| 版本 | v0.4（新增预处理三件套 + 跨模态相关性 + 音视同步校验 + 全套可视化） |
| 更新日期 | 2026-09-24 |
| 状态 | 全流程跑通；质量分析(相关性/同步)已并入汇总表，产出 4 类可视化图 |
| 配套脚本 | `config.py`、`pipeline.py`、`figures.py`、`vector_figs.py`、`visualize.py` |
| 运行环境 | Windows + conda 环境 `py310` + Spyder |
| Python 解释器 | `python`（Python 3.10.13） |

### 修订记录
| 版本 | 日期 | 修订内容 | 状态 |
|---|---|---|---|
| v0.1 | 2026-09-23 | 初版：融合"对齐优先 + 时间箱/词级双版本 + mask闭环 + ε池化 + 五条校验" | 已归档 |
| v0.2 | 2026-09-23 | 本机实跑100条并全部通过校验（视觉为 Haar+5×7 基线）。新增：Step5 文本对齐改为"ASR序列对齐+插值"；新增 ASR 递进式升级(base→small)；新增语种强制(language=en)；新增 VAD 兜底；修复 numba 缓存卡死、cv2中文路径、慢速人脸检测等问题；新增第2.3节"数据质量审计"；新增第5节 Spyder 保姆级操作；补入全部实测数字 | 已归档 |
| v0.3 | 2026-09-23 | 视觉模块由 Haar+5×7 灰度网格升级为 OpenFace 2.2.0（CLNF 模型，17 AU+6 姿态+8 注视+4 眼部几何=35 维）；新增 feat_vision/{id}.times.npy(真实帧时间戳)与 .valid.npy(逐帧有效标记)；全量100条重跑并五校验通过；summary 增补视觉后端/有效帧数/有效占比 | 当前版本 |
| v0.4 | 2026-09-24 | 新增 Step8 预处理三件套(缺失标记+μ±5σ去极值+z-score, 产出 *_norm)；新增 Step11 跨模态相关性矩阵(逐维|r|均值, 3×3)与 Step12 音视同步互相关校验(sync_score/peak_lag/sharpness, 嘴-声物理代理)；新增 figures.py 产出 4 类图；新增 vector_figs.py 输出 9 张文档用 PDF 矢量图(含数据集统计/覆盖率/词匹配率/典型样本/对齐机制/相关性/同步)；summary.csv 扩到 25 列(含 7 个质量列)；pkl 新增 *_norm 字段 | 当前版本 |
| | | （后续改进在此追加） | |

---

## 0. 实验目标与产出（已全部生成）

放在模块根目录的 `q1_output\` 下：

| 产出物 | 文件名 | 实际形状/大小 | 用途 |
|---|---|---|---|
| 特征文件(时间箱版) | `q1_features_timebin.pkl` | text (100,50,768)、audio (100,50,74)、vision (100,50,35) | **主版本**，50个等长时间箱 |
| 特征文件(词级版) | `q1_features_word.pkl` | 同上 + length(100,) | 与标准化特征`aligned_50` 兼容 |
| 100条汇总表 | `summary.csv` | 100行×12列 | 覆盖完整性 + 时序可核验 |
| 对齐记录 | `align_records.json` | 170 KB | 逐样本词时间戳、有效帧数、对齐质量 |
| 典型样本图 | `figs/typical_sample_full.png` | 五层(波形+词+语音+视觉+时间箱) | 对齐效果可视化 |
| 相关性热力图 | `figs/corr_heatmap.png` | 3×3 |r| 矩阵 | 论证三模态互补 |
| 互相关曲线 | `figs/ncc_curve_typical.png` | C(τ) 曲线 | 音视同步校验 |
| 滞后直方图 | `figs/lag_histogram.png` | 100样本分布 | 同步质量统计 |
| 质量分析表 | `analysis/corr_*.csv` `sync_metrics.csv` | 见第7节 | 相关性/同步指标 |
| **矢量图(PDF)** | `vector_figs/*.pdf` | **9 张矢量图** | **可直接使用（中文已嵌入）** |
| 中间产物 | `audio/` `frames/` `feat_audio/` `feat_vision/` `feat_text/` `align/` `align_raw/` `meta/` | 见第7节 | 可追溯、可复核 |

---

## 1. 环境与路径（实测版本）

### 1.1 目录约定
```
01_feature_alignment/
├── data/raw/          ← 输入视频 + label-100.xlsx
├── code/              ← 代码（config / pipeline / figures / vector_figs / visualize ...）
├── features/          ← 已发布的全量特征文件与读取示例
├── environment/requirements.txt
├── docs/              ← 本手册 + README
└── q1_output/         ← 运行后全部输出
```

### 1.2 工具清单与版本（实测）
| 工具/库 | 版本 | 用途 |
|---|---|---|
| Python | 3.10.13 | 运行环境 |
| torch | 2.0.0+cu118 | CUDA 可用（`torch.cuda.is_available() == True`） |
| transformers | **4.44.2** | 文本 BERT；**不能升到 5.x**（5.x 要求 torch>=2.5） |
| tokenizers | 0.19.1 | 必须用 `BertTokenizerFast`（慢版不支持字符偏移） |
| librosa | 0.11.0 | 语音74维 |
| opencv-python | 4.10.0 | 抽帧 / 探针抽帧（人脸检测改由 OpenFace MTCNN 承担） |
| OpenFace | 2.2.0 (win x64) | 视觉35维：`FeatureExtraction.exe -mloc model/main_clnf_general.txt -pose -aus -gaze` |
| openai-whisper | 已安装 | 词级时间戳（模型 `base` + `small`） |
| imageio-ffmpeg | 0.6.0 | 提供 ffmpeg 可执行文件（自动定位） |
| soundfile | 随 librosa | 读音频时长 |

### 1.3 环境自检（跑通再往下）
```python
import torch, librosa, cv2, soundfile, transformers, whisper
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("librosa", librosa.__version__, "| cv2", cv2.__version__)
print("cuda", torch.cuda.is_available())
from transformers import BertTokenizerFast
print("fast tokenizer:", BertTokenizerFast.from_pretrained("bert-base-uncased").is_fast)
import config; print("ffmpeg:", config.get_ffmpeg())
```
判定标准：`cuda True`、`fast tokenizer True`、`ffmpeg` 打印出 `...ffmpeg-win64-v4.2.2.exe` 路径即通过。

> ⚠️ **重要环境坑**：本机 pip 会把包装到 `%APPDATA%\Python\Python310\site-packages`，而该目录在 `sys.path` 中**优先于** conda 环境目录。如果某个包"装了却 import 不到"或"版本没变"，用 `pip install --user 包名` 再试，并重启 Spyder 内核。

---

## 2. 数据说明（原始视频）

### 2.1 基本结构
- 路径：`data\raw\`
- 结构：37 个 `video_id` 子文件夹，每个里面若干 `clip_id.mp4`（共100个）
- `label-100.xlsx`：100 行，列 = `video_id / clip_id / text / label / annotation`
- `label`：连续情感强度（数值）；`annotation`：Negative / Neutral / Positive
- 本机实测标签分布：**Positive 57 / Neutral 25 / Negative 18**
- 颜色映射写进文档：Negative→0，Neutral→1，Positive→2
- 样本唯一编号：`id = video_id + "$" + clip_id`（例如 `-3g5yACwYnA$13`）

### 2.2 实测数据规模（100条）
| 指标 | 数值 |
|---|---|
| 时长 | 最短 2.23 s，最长 29.11 s，平均 7.77 s，总计 777.1 s |
| 语音帧（100 fps） | 合计 77761 帧，单样本 223 ~ 2911 帧 |
| 视觉帧（OpenFace 原生 fps） | 合计 23241 帧，有效 21969 帧（94.5%），单样本 67~875 帧 |
| 文本词数 | 合计 1917 词，单样本 5 ~ 50+ 词 |

### 2.3 数据质量审计（★ 重要发现，务必写进文档的"数据说明/误差来源"）

在跑文本-语音对齐时，我们发现**有 6 个视频（共 20 个片段）的"音频内容"与"数据给出的 text"对不上**。

**现象举例**（视频 `-iRBcNs9oI8`，一个讲"power pose"的英文脱口秀视频）：

| clip_id | 数据 text | Whisper 从该片段音频听写的结果 |
|---|---|---|
| 3 | Well, it is the secret of power pose | is going to help you on your tests such as TOEFL, IELTS or University. |
| 6 | What is a power pose | help you if you have an English job interview. |
| 7 | Well, for starters, let's look at the word "power" | And the secret will also help you if you ever |
| 8 | I've drawn this man here, ... he looks very strong | ever decide to ask out that special someone. So what is the secret I'm telling you about? ... |

**我们的排查过程（可复现，用于自证不是代码问题）**：
1. 用 ffmpeg 从 `-iRBcNs9oI8\8.mp4` **重新抽一次音频**，与原 wav 做 MD5 比对 → **逐字节完全一致**（`fbaa327d5e83`），排除"抽音频抽错/损坏"。
2. ffprobe 查 mp4 时长 8.09 s，wav 时长 8.03 s → 一致，排除"截断"。
3. 把该视频 6→7→8→9 四个片段的 ASR 结果**按 clip_id 顺序首尾相接**：
   `...help you if you have an English job interview.` + `And the secret will also help you if you ever` + `ever decide to ask out that special someone. So what is the secret I'm telling you about? ...` + `of Power Pose.`
   → **能连成一段通顺的话**，说明这几个 mp4 是同源视频中**连续**的时间片段。
4. 但数据对这 4 个片段给出的 text 与该连续语音**不对应**。

**结论**：数据提供的部分片段，其"视频文件"与"text 标注"并非同一时间段。这不是本流水线的错误，属于**数据本身的不一致**。

**按视频统计**（平均词匹配率 < 0.5 的 6 个视频）：

| video_id | 平均词匹配率 | 片段数 |
|---|---|---|
| -NFrJFQijFE | 0.000 | 2 |
| -ri04Z7vwnc | 0.037 | 3 |
| -hnBHBN8p5A | 0.077 | 2 |
| -mJ2ud6oKI8 | 0.095 | 5 |
| -iRBcNs9oI8 | 0.097 | 5 |
| -yRb-Jum7EQ | 0.333 | 3 |

**我们怎么处理**：
- 不丢弃任何样本（设计要点100条全覆盖）；
- 给每条样本记录一个 **`word_match_ratio`（词匹配率）** 作为"对齐置信度"，写进 `meta/*.json`、`align_records.json`、`summary.csv`；
- 匹配率 ≥ 0.5 的 **80/100** 用 ASR 词时间戳做精确锚定；匹配率过低的 8 条退化为 **VAD 能量兜底**（把词铺到"有声音"的区间上），其余 12 条用 ASR 锚点 + 插值；
- 文档里把它写成"数据质量审计 + 对齐置信度标注 + 分级对齐策略"，这反而是亮点。

---
## 3. 总体流程（12步）

```
100条视频（原始视频）
  │
  ├─ Step1  读标签建样本清单          → id / text / label / annotation
  ├─ Step2  抽音频（16kHz 单声道）    → audio/{id}.wav + meta 记时长 T
  ├─ Step3  语音特征 74 维/帧         → feat_audio/{id}.npy
  ├─ Step4  （预留）抽帧探针           → frames/{id}/000000.jpg ...
  ├─ Step5  视觉特征 35 维/帧(OpenFace)→ feat_vision/{id}.npy(+.times/.valid) + of_raw/{id}/{clip}.csv
  ├─ Step6  文本特征 768 维/词        → feat_text/{id}.npy
  ├─ Step7  词级时间戳（ASR对齐）     → align/{id}.json + align_raw/{id}.{模型}.json
  │
  ├─ Step8  预处理三件套(★新)         → *_norm + mask（缺失标记 + μ±5σ去极值 + z-score）
  │
  ├─ Step9  时间箱对齐（50格）        → 主版本三模态特征 + mask
  ├─ Step10 词级对齐（50词）          → 兼容版三模态特征 + mask + length
  │
  ├─ Step11 跨模态相关性矩阵(★新)     → 3×3 |r| 矩阵 + corr_per_sample.csv
  ├─ Step12 音视同步互相关校验(★新)  → sync_score / peak_lag / sharpness
  │
  ├─ Step13 打包                      → q1_features_timebin.pkl / q1_features_word.pkl(含 *_norm)
  ├─ Step14 校验 + 汇总               → summary.csv(25列) / align_records.json
  └─ Step15 全套可视化                → figs/ 4 类图
```

---

## 4. 分步操作与数学定义（核心）

### Step 1　读标签、建样本清单
- 目的：拿到100条样本的 id、文本、标签。
- 关键：`clip_id` 按字符串读，并去掉末尾 `.0`，防止 `13` 被读成 `13.0`。
- 唯一编号：**id = video_id + "$" + clip_id**
- 检查：`len(samples) == 100`，并打印标签分布。

### Step 2　抽音频 + 记时长
- 命令：`ffmpeg -i 视频 -vn -ac 1 -ar 16000 -y 输出.wav`
- 为什么要 16kHz 单声道：语音情感研究（COVAREP/OpenSMILE）的事实标准，且 whisper 也统一吃 16kHz。
- 记时长：T = 采样点数 / 16000。**T 是后面所有时间对齐的基准。**
- 输出：`audio/{id}.wav`、`meta/{id}.json`（含 `duration`）

### Step 3　语音特征：74 维/帧
把语音切成每 1/100 秒一帧，每帧算 74 个数：

| 分量 | 维数 | 含义 |
|---|---|---|
| MFCC（梅尔频率倒谱系数） | 24 | 声道形状/音色，语音情感最经典特征 |
| ΔMFCC（一阶差分） | 24 | 谱包络变化速度 |
| ΔΔMFCC（二阶差分） | 24 | 变化加速度 |
| F0（基频/音高） | 1 | 情绪激动程度（YIN 算法，范围 80~400 Hz） |
| RMS（短时能量） | 1 | 音量强弱 |
| **合计** | **74** | |

- 数学定义：一帧 xᵢ 的梅尔倒谱
  c = DCT( log( MelFilter( |FFT(xᵢ)|² ) ) )
  只取前 24 个系数。
- 参数：sr = 16000，hop = 16000/100 = 160 点，window = 默认 400 点（25 ms）。
- 输出：`feat_audio/{id}.npy`，形状 (帧数, 74)；帧数 ≈ T × 100（实测 77761 帧，与 777.1 s 吻合）。

### Step 4　抽帧（预留/探针，非主路径）
- 主视觉路径（Step5）改为 OpenFace 直接吃视频，**不再依赖抽帧**；`extract_frames()` 仅作为"人工抽帧探针/典型样本截图"保留。
- 若需抽帧查看：用 OpenCV 按 30 fps 抽帧，存 `frames/{id}/000000.jpg ...`。
- **中文路径工程坑**（探针抽帧时仍适用）：不能用 `cv2.imwrite`（会静默返回 False、目录全空）。改用：
  `cv2.imencode(".jpg", frame)[1].tofile(路径)` 写，
  `cv2.imdecode(np.fromfile(路径, dtype=np.uint8), ...)` 读。

### Step 5　视觉特征：35 维/帧（OpenFace 2.2.0，★ 本次升级）
- **工具**：OpenFace 2.2.0 标准 Windows 版 `FeatureExtraction.exe`。
- **模型**：`-mloc "model/main_clnf_general.txt"`（CLNF 模型）。
  > 为什么不用默认模型：标准二进制包缺 CEN patch experts（`cen_patches_*.dat`，仅托管于 Dropbox/OneDrive，国内不可达），默认 CECLM 模型会报 `Could not load the landmark detector`。改用 CLNF 模型（其 CCNF patch experts 随包已含）即可完整运行，精度高于 CLM。
- **命令行**（逐条视频，失败可断点续跑）：
  `FeatureExtraction.exe -f <视频> -out_dir <输出目录> -mloc model/main_clnf_general.txt -pose -aus -gaze`
- **OpenFace 内部级联流程**（文档中写"检测→拟合→解算→回归"）：
  1. MTCNN 三级级联（PNet→RNet→ONet）检测人脸框；
  2. CLNF 约束局部神经场模型拟合 68 个主关键点 + 28×2 个眼部子模型关键点；
  3. 关键点 + 3D PDM 解算头部姿态（平移 Tx/Ty/Tz + 旋转 Rx/Ry/Rz）；
  4. 眼部关键点 + 3D 眼模型解算注视方向（左右眼三维视线向量 + 二维注视角）；
  5. 面部外观特征回归 17 个 AU 动作单元强度。
- **35 维构成**（已实装，字段名即 OpenFace CSV 列名）：
  | 分量 | 维数 | 具体字段 |
  |---|---|---|
  | 面部动作单元 AU（回归强度 0~5） | 17 | AU01_r, AU02_r, AU04_r, AU05_r, AU06_r, AU07_r, AU09_r, AU10_r, AU12_r, AU14_r, AU15_r, AU17_r, AU20_r, AU23_r, AU25_r, AU26_r, AU45_r |
  | 头部姿态 | 6 | pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz |
  | 注视 | 8 | gaze_0_x/y/z（左眼三维视线向量）, gaze_1_x/y/z（右眼三维视线向量）, gaze_angle_x/y（二维注视角） |
  | 眼部几何 | 4 | 左眼 28 点 2D 中心 (x̄,ȳ) + 右眼 28 点 2D 中心 (x̄,ȳ) |
  | **合计** | **35** | |
- **时序对齐（真实时间戳，可核验）**：直接读 OpenFace CSV 的 `timestamp` 列（真实帧时间，而非 30fps 均匀假设），逐帧写进 `feat_vision/{id}.times.npy`。
- **失败处理（mask 闭环）**：CSV 的 `success` 列 0/1（1=跟踪成功）。失败帧 35 维置 0，并把逐帧有效标记写进 `feat_vision/{id}.valid.npy`；时间箱/词级对齐时只对 valid=1 的帧做均值池化，mask 自动标 0。
- **实测结果**：合计 23241 帧，有效 21969 帧（**94.5%**）；78/100 条全帧有效；4 条全无效（profile/极端角度/过曝，MTCNN 正面检测无法初始化）；18 条部分有效。
  > 4 条全无效：`-NFrJFQijFE$1`、`-NFrJFQijFE$2`、`-mJ2ud6oKI8$1`、`-ri04Z7vwnc$0`。已抽帧人工复核，均为人脸侧脸/极端角度。这正是"视觉模态缺失"的自然来源，mask 正确兜底，可直接作为 Q2"模态缺失"的输入场景（亮点）。
- 输出：`feat_vision/{id}.npy`（35 维）+ `.times.npy`（时间戳）+ `.valid.npy`（有效标记）；OpenFace 原始 CSV 存 `of_raw/{id}/{clip}.csv`。

### Step 6　文本特征：768 维/词
- 工具：`BertTokenizerFast` + `bert-base-uncased`（**必须 Fast 版**，慢版不支持 `return_offsets_mapping`，会直接报 NotImplementedError）。
- 做法：
  1. 用数据给的 text 送进 BERT，取最后一层隐状态 h（每个 subword 一个 768 维向量）；
  2. 用字符偏移 `offset_mapping` 把 subword 归并到"词"：某词字符区间 [a,b) 覆盖到的所有 subword 向量取平均，得到该词的 768 维向量。
- 数学上：第 j 个词 wⱼ 的向量
  **eⱼ = (1/|Sⱼ|) · Σ_{u ∈ Sⱼ} hᵤ**，其中 Sⱼ = { u : subword u 的字符区间与 wⱼ 重叠 }
- 参数：`max_length=256`、`truncation=True`、`no_grad` 推理。
- 输出：`feat_text/{id}.npy`，形状 (词数, 768)。

### Step 7　词级时间戳（ASR 序列对齐）★ 本次重点升级
这是"文本 ↔ 语音/视频"能对上时间的关键。做法分 4 层：

**① 强制语种识别**
`whisper.transcribe(y, language="en", task="transcribe", word_timestamps=True, condition_on_previous_text=False)`
> 为什么要强制 `language="en"`：实测有片段因为短、有背景音乐，被 Whisper 误判语种，整段被"听写"成**希伯来语**；词数看着差不多但内容完全对不上。MOSEI 是英文语料，必须锁死。

**② 递进式模型升级**
先用 `base` 模型听写并算"词匹配率"；若 < 0.70，自动换 `small` 模型再听写一次，取两者中更好的。
> 匹配率定义：**r = (能严格对上的词数) / (该样本总词数)**

**③ 序列对齐（把 ASR 词时间戳搬到数据文本的词上）**
- 先把两边都归一化：转小写、去标点（`They've` → `theyve`）。
- 用 `difflib.SequenceMatcher` 求最优匹配块（opcode）。
- `equal` 块内的词**直接继承** ASR 的时间戳，作为"锚点"。
- 锚点之间的缺口（漏读、错读、语气词）**线性插值**：
  若左锚点结束于 t₀、右锚点开始于 t₁、中间缺 d 个词，则第 m 个缺失词的时间区间为
  **[ t₀ + (m−1)·(t₁−t₀)/d , t₀ + m·(t₁−t₀)/d )**
- 头部缺口铺在 [0, 首锚点开始] 上；尾部缺口铺在 [末锚点结束, T] 上。

**④ 兜底（ASR 完全对不上时）**
用能量法 VAD（`librosa.effects.split(top_db=30)`）找出"有声音"的区间集合 {(a₁,b₁), (a₂,b₂), …}，
设总语音时长 L = Σ(bᵢ−aᵢ)，把 N 个词**按语音时长比例**铺到这些区间上（跳过静音段）。
> 这样即使听写内容错了，词的落点也不会跑到静音/背景段里。

**⑤ 缓存**
每次 ASR 结果按模型规格落盘到 `align_raw/{id}.{base|small}.json`，重跑时直接读，不会重复跑模型。
最终对齐结果写到 `align/{id}.json`，并在 `meta/{id}.json` 里记：
`text_align`（whisper_seq / vad_uniform）、`asr_model`、`word_match_ratio`、`n_word_interpolated`、`asr_trials`。

### Step 8　预处理三件套（★ 本次新增）
在对齐后、打包前，对三模态对齐特征逐样本做三件套处理，产出 `*_norm`（标准化特征，供模块二与质量分析使用）。
- **① 缺失标记**：由对齐阶段生成的 `mask` 表达（无效位置 mask=0）。视觉跟踪失败帧、静音/无人脸等一律 mask=0，不参与统计。
- **② μ±5σ 去极值**：每个样本内每个维度，仅在有效格上估计均值 μ 与标准差 σ，把特征裁剪到 `[μ−5σ, μ+5σ]`。
- **③ z-score 标准化**：`x_norm = (clip(x, μ−5σ, μ+5σ) − μ) / σ`，使每个维度均值≈0、方差≈1。
- 实现：`preprocess_block(arr, mask)`（`pipeline.py`），参数 `SIGMA_CLIP=5.0`。
- 产出：pkl 新增 `text_norm/audio_norm/vision_norm`（形状同原始对齐特征，无效格置 0 且 mask=0）。
- 实测：`audio_norm` 有效格均值 = -0.0000、方差 = 0.9899（符合标准化预期）。

### Step 9　时间箱对齐（主版本，50 格）
- 时间箱定义：**Iₖ = [ (k−1)·T/50 , k·T/50 )**，k = 1…50
- 每个原始帧的时间戳：**tᵢ = (i + 0.5) / f**（f = 100 Hz 语音、30 Hz 视觉）
- 每个词的时间戳：**tⱼ = (sⱼ + eⱼ) / 2**
- 落箱：**k(i) = clip( floor( tᵢ / T × 50 ), 0, 49 )**
- **ε 统一池化**（不用 if-else，缺失自然为 0）：
  **x̄ₖ = ( Σ_{i : tᵢ ∈ Iₖ} xᵢ ) / ( Σ_{i : tᵢ ∈ Iₖ} 1 + ε )**，ε = 1e-8
- 有效掩码：**mₖ = 1** 若 `Σ 1 > 0`，否则 **mₖ = 0**
- 记录个数：**cₖ = Σ_{i : tᵢ ∈ Iₖ} 1**（写进 `align_records.json`，可反查）
- 输出：三张 (50,·) 表 + 三个 (50,) mask。

### Step 10　词级对齐（兼容标准化特征的 aligned_50）
- 以"第 j 个词的时间窗 [sⱼ, eⱼ)"为格子：
  **yⱼ = ( Σ_{i : tᵢ ∈ [sⱼ,eⱼ)} xᵢ ) / ( Σ_{i : tᵢ ∈ [sⱼ,eⱼ)} 1 + ε )**
- 词数 < 50：后面补 0，这是 **padding**，用 `length` 字段表达；
- 某个词窗口内没有帧：这是**真实缺失**，用 `mask=0` 表达。
- **padding 与真实缺失必须能区分**（设计规范），所以词级版同时存 `length` 和 `mask`。
- 输出：三张 (50,·) 表 + 三个 (50,) mask + `length` (100,)。

### Step 11　跨模态相关性矩阵（★ 本次新增，论证三模态互补性）
- 目的：回答"三模态是否冗余"。三对模态（文本-语音、文本-视觉、语音-视觉）分别算逐维皮尔逊相关的**平均绝对值**。
- 方法：对时间箱版 `*_norm` 特征，仅在"两模态 mask 交集为有效"的格上，逐维度算皮尔逊相关矩阵 R^(d1×d2)，再取 `ρ = mean(|R|)`。
- 跨样本聚合：每样本算完 ρ 后求均值与标准差（有效格 <5 的样本跳过并记 NaN）。
- 输出：`analysis/corr_matrix.csv`（3 对 mean/std/n）、`analysis/corr_per_sample.csv`（100 行）。
- 实测（100 样本）：文本-语音 0.214±0.075，文本-视觉 0.237±0.063，语音-视觉 0.102±0.021。三者均远低于 0.6，其中语音-视觉最低，说明三模态信息互补、融合有必要。
- 实现：`compute_cross_modal_corr(tb)`（`pipeline.py`）。

### Step 12　音视同步互相关校验（★ 本次新增，验证时间对齐正确性）
- 目的：验证音频与视觉是否落在同一时间基准上（检测是否存在系统性时移）。
- 物理代理：音频激活度 = 响度 RMS；视觉激活度 = 嘴部张开度 AU25+AU26（说话时二者强同步，优于"情绪 AU 均值"）。
- 方法：音频 RMS 插值到视觉真实帧时间戳，两者 z-score 后计算归一化互相关 C(τ)，τ 以视觉帧为单位，扫描 ±15 帧（≈±0.5s）。
- 指标：`sync_score = max C(τ)`、`peak_lag = argmax C(τ)`（换算为秒）、`sharpness = 主峰/次峰`。
- 输出：`analysis/sync_metrics.csv`（100 行）、`analysis/sync_curves.npy`。
- 实测：中位 peak_lag = −0.099 s（≈0，**无系统性时移**）；|lag|≤0.2s 占比 46%、中位 |lag| = 0.258s（宽峰、峰中心≈0，符合语音是慢过程的物理事实）。
- 实现：`compute_av_sync(tb)`（`pipeline.py`）。

### Step 13　打包
- `q1_features_timebin.pkl` / `q1_features_word.pkl`，结构都是 `{"test": {字段...}}`（与 CMU-MOSEI 标准 pkl 习惯一致）。
- 字段：`id / raw_text / text / audio / vision / text_mask / audio_mask / vision_mask / duration / n_audio_frames / n_vision_frames / align_mode / annotations / classification_labels / regression_labels`（词级版多一个 `length`）。

### Step 14　校验 + 汇总（含相关性/同步质量列）
自动断言，全部通过：
1. **覆盖**：100 条 id 与输出一一对应。
2. **形状**：(100,50,768) / (100,50,74) / (100,50,35)。
3. **时间反查**：任意 k 都能反查回 [k·T/50, (k+1)·T/50)。
4. **长度区分**：`length`(padding) 与 `mask`(真实缺失) 可区分。
5. **异常保留**：无声/无人脸/极短片段一律保留，只置 mask=0。
最后生成 `summary.csv`（100 行）与 `align_records.json`。

### Step 15　全套可视化 + 矢量图输出
三个脚本分工：

| 脚本 | 输出 | 用途 |
|---|---|---|
| `visualize.py` | `typical_sample.png`（四层） | 早期版本，可选 |
| `figures.py` | `figs/*.png`（4 类图） | 快速预览 |
| `vector_figs.py` | `vector_figs/*.pdf`（**9 张矢量图**） | **可直接使用** |

**`vector_figs.py` 输出的 9 张矢量图**（全部 PDF，中文已嵌入 Microsoft YaHei）：

| 文件 | 内容 | 对应内容 |
|---|---|---|
| `fig01_duration_hist.pdf` | 100 条视频时长分布（含均值/中位线） | 数据说明 |
| `fig02_label_dist.pdf` | (a) 情感极性分布 (b) 情感强度分布 | 数据说明 |
| `fig03_coverage.pdf` | 三模态时间箱有效覆盖率箱线图 | 数据质量 |
| `fig04_wordmatch.pdf` | (a) 词匹配率分布 (b) 对齐策略分布 | 时序可核验 |
| `fig05_typical_sample.pdf` | 典型样本五层图（波形+词+语音74维+视觉35维+时间箱） | 典型样本验证 |
| `fig06_timebin_scheme.pdf` | 时间箱对齐机制示意（帧落箱 + 对齐后特征） | 对齐规则说明 |
| `fig07_corr_heatmap.pdf` | 跨模态相关性热力图 | 互补性论证 |
| `fig08_ncc_curve.pdf` | 音视互相关曲线 | 同步校验 |
| `fig09_lag_histogram.pdf` | 音视同步滞后分布 | 同步校验 |

- `fig05`/`fig06` 自动挑选"词匹配率高、音视有效帧多"的样本作典型。
- PDF 已设置 `pdf.fonttype=42`（TrueType 嵌入），实测嵌入字体为 `MicrosoftYaHei`，可直接插入 Word/LaTeX。

---
## 5. Spyder 保姆级操作（照着做即可）

### 第 0 步　确认解释器是 py310
Spyder → 菜单 `Tools` → `Preferences` → `Python interpreter`，
选择 **Use the following Python interpreter**，填：
```
python
```
然后 `Consoles` → 重启内核（Restart kernel）。在 IPython 控制台输入：
```python
import sys; print(sys.executable)
```
必须打印出 `python`。**不对就先改，否则后面全报错。**

### 第 1 步　（只有第一次需要）下载模型
本流水线需要 3 个预训练模型，**只在第一次运行时联网下载**：
| 模型 | 大小 | 用途 | 下载后位置 |
|---|---|---|---|
| bert-base-uncased | 420 MB | 文本 768 维 | `%USERPROFILE%\.cache\huggingface\hub\` |
| whisper base | 139 MB | 词级时间戳（第一轮） | `%USERPROFILE%\.cache\whisper\` |
| whisper small | 461 MB | 词级时间戳（第二轮升级） | `%USERPROFILE%\.cache\whisper\` |

在 Spyder 控制台运行（**只需一次**，下载完就不用了）：
```python
import whisper
from transformers import BertTokenizerFast, BertModel
BertTokenizerFast.from_pretrained("bert-base-uncased")
BertModel.from_pretrained("bert-base-uncased")
whisper.load_model("base")
whisper.load_model("small")
print("模型全部就绪")
```
> 若报 `Model has been downloaded but the SHA256 checksum does not match`，说明网断了没下完，**再跑一次**即可（会自动重下）。

### 第 2 步　把工作目录切到 code
在 Spyder 的 IPython 控制台：
```python
%cd <模块根目录>\code
import os
os.environ["HF_HUB_OFFLINE"] = "1"        # 模型已缓存，禁止联网检查，避免卡住
```

### 第 3 步　一键跑完全流程
```python
%run pipeline.py all
```
预计耗时（本机实测，CUDA 可用）：
| 阶段 | 耗时 | 说明 |
|---|---|---|
| audio（抽音频+74维） | 已完成（100个wav+100个npy） | 首次约 5~10 分钟 |
| vision（OpenFace 35维） | 已完成（23241帧，有效94.5%） | 首次约 12 分钟 |
| text（BERT+Whisper） | 约 3~5 分钟 | 有缓存则秒级 |
| build（对齐打包） | 约 10 秒 | |
| summary（校验） | 约 2 秒 | |
| **合计** | **首次约 15~20 分钟** | 二次运行全部走缓存，**实测 8.1 秒** |

### 第 4 步　画全套可视化图（相关性热力图 + 互相关曲线 + 滞后直方图 + 典型样本五层图）
```python
%run figures.py
```
（旧的四层 `visualize.py` 仍保留。）

### 第 5 步　只想跑其中一段（调试用）
```python
%run pipeline.py audio      # 只抽音频 + 语音74维
%run pipeline.py vision     # 只跑 OpenFace 视觉35维
%run pipeline.py text       # 只做文本768维 + 词时间戳
%run pipeline.py build      # 只做对齐 + 打包
%run pipeline.py summary    # 只做五项校验 + 汇总表
```

### 第 6 步　验收（逐条核对）
```python
import pandas as pd, pickle, config as C
sm = pd.read_csv(C.SUMMARY_CSV); print(sm.shape, sm.head())
tb = pickle.load(open(C.PKL_TIMEBIN,"rb"))["test"]
wd = pickle.load(open(C.PKL_WORD,"rb"))["test"]
print(tb["text"].shape, tb["audio"].shape, tb["vision"].shape)     # (100,50,768) (100,50,74) (100,50,35)
print(tb["audio_mask"].mean(), tb["vision_mask"].mean())           # 1.0  0.988
print(wd["length"][:10])                                          # 每条的"真实词数"
```
全部对得上即完成。

---

## 6. 命令行运行方式（等价，非 Spyder 环境也能跑）

```powershell
cd <模块根目录>\code
$env:HF_HUB_OFFLINE="1"
$env:PYTHONIOENCODING="utf-8"
python -u pipeline.py all
python -u figures.py
python -u visualize.py
```

本次实跑日志保存在 `q1_output\run_text.log`、`run_build.log`、`run_summary.log`。

---

## 7. 本次实跑结果汇总（可写进文档）

### 7.1 产出文件与张量
| 文件 | 字段 | 形状 |
|---|---|---|
| `q1_features_timebin.pkl` | text / audio / vision | (100,50,768) / (100,50,74) / (100,50,35) |
| | text_norm / audio_norm / vision_norm | (100,50,768) / (100,50,74) / (100,50,35) |
| | text_mask / audio_mask / vision_mask | (100,50) / (100,50) / (100,50) |
| `q1_features_word.pkl` | text / audio / vision | (100,50,768) / (100,50,74) / (100,50,35) |
| | length | (100,) |

### 7.2 数据规模
| 指标 | 数值 |
|---|---|
| 样本数 | 100 |
| 时长 | 2.23 ~ 29.11 s，均值 7.77 s，总计 777.1 s |
| 语音帧（100fps） | 77761 帧，单样本 223 ~ 2911 |
| 视觉帧（30fps） | 23241 帧，单样本 67 ~ 875 |
| 文本词数 | 1917 词，单样本 5 ~ 50+ |

### 7.3 时序对齐质量
| 指标 | 数值 |
|---|---|
| 时间箱版 语音 mask 平均有效率 | **1.000**（100/100 样本 50 格全有效） |
| 时间箱版 视觉 mask 平均有效率 | **0.988**（93/100 样本 50 格全有效，其余因无人脸置 0） |
| 词级版 有效长度 | 5 ~ 50 词，均值 19.2 词 |
| 对齐方式 | `whisper_seq` 92 条，`vad_uniform` 8 条 |
| 最终采用 ASR 模型 | `base` 89 条，`small` 3 条，无（走VAD兜底）8 条 |
| 词匹配率 | 最小 0.000，均值 **0.768**，中位 **0.924**，最大 1.000 |
| 词匹配率 ≥ 0.5 的样本 | **80 / 100** |
| NaN / 无穷大 | 无（三模态张量全部有限） |

### 7.3b 跨模态相关性与音视同步（★ 本次新增）
| 指标 | 数值 |
|---|---|
| 跨模态平均绝对皮尔逊相关 |r| | 文本-语音 **0.214±0.075**，文本-视觉 **0.237±0.063**，语音-视觉 **0.102±0.021** |
| 三模态互补性结论 | 三对 |r| 均远低于 0.6，语音-视觉最低，证明信息互补、融合有必要 |
| 音视同步 中位 peak_lag | **-0.099 s**（≈0，无系统性时移） |
| 音视同步 |lag|≤0.2s 占比 | **46%**；中位 |lag| = 0.258 s（宽峰，符合语音慢过程物理事实） |
| 音视同步 sync_score | 中位 0.256，最大 0.85 |

### 7.4 六项校验结果（自动断言，全部通过）
```
[1] 覆盖检查通过: 100 条 id 与输出一一对应
[2] 形状检查通过: (100, 50, 74)
[3] 时间反查检查通过: 索引 k -> [k*T/50,(k+1)*T/50)
[4] 长度检查通过: padding(length) 与 真实缺失(mask) 可区分
[5] 异常检查通过: 全部样本保留, 无效位置仅 mask=0
[6] 质量分析通过: 跨模态相关性矩阵 + 音视同步指标已并入汇总表
```

---

## 8. 设计要点与产出索引

| 设计要点 | 内容 | 对应产出 |
|---|---|---|
| 特征提取与时序对齐整体方案 | 三模态特征定义 + 提取工具 + 时间箱/词级对齐规则 + 跨模态时间对应逻辑 + 完整处理流程 | 本手册第 3、4 节 |
| 特征文件规范与全量结果汇总 | pkl 组织结构、读取方式、100 条汇总表（样本编号 / 模态 / 有效时长 / 维度 / 对齐粒度） | `features/q1_features_*.pkl.gz` + `summary.csv` |
| 典型样本验证 | 文本片段 ↔ 语音时段 ↔ 视频帧段 ↔ 三类特征的对应关系 | `figs/typical_sample_full.png` + 第 4 节 Step15 |
| 可复现性 | 工具与模型版本、核心参数、完整运行流程、处理日志 | 第 1.2 节版本表 + 第 5 节操作 + 第 7 节结果 |

**可直接展开的三点**：
- 第 2.3 节"数据质量审计"（文本-音频不一致的发现与量化、对齐置信度标注、分级对齐策略）；
- 视觉特征采用 **OpenFace 2.2.0** 的 "17 AU + 6 头部姿态 + 8 注视 + 4 眼部几何 = 35 维" 方案（第 4 节 Step5）；
- 音频 74 维与 COVAREP 标准 74 维的维度对照讨论。

---

## 9. 常见问题排查（全部为本次实跑真实踩坑）

| 现象 | 根因 | 解决 |
|---|---|---|
| `FileNotFoundError: [WinError 2] 系统找不到指定的文件`（subprocess） | 直接调 `ffmpeg` 但 PATH 里没有 | 统一走 `config.get_ffmpeg()`（内部用 imageio-ffmpeg 定位） |
| `NotImplementedError: 请在 audio_features() 中接入 COVAREP/OpenSMILE` | 占位函数没实现 | 已实现为 24+24+24+1+1 = 74 维 librosa 版本 |
| `ModuleNotFoundError: No module named 'cv2'` | 没装 opencv | `pip install --user "opencv-python<4.11"` 后重启内核 |
| `Disabling PyTorch because PyTorch >= 2.5 is required but found 2.0.0+cu118` | transformers 装了 5.x | `pip install --user "transformers==4.44.2"` |
| `BertModel requires the PyTorch library` | 同上（transformers 版本把 torch 后端判废） | 同上，锁定 4.44.2 |
| `return_offset_mapping is not available when using Python tokenizers` | 用了慢版 `BertTokenizer` | 改用 `BertTokenizerFast` |
| **`import librosa` 后程序无限卡死（CPU 占满一个核，无输出）** | numba 默认把 JIT 缓存写到 `site-packages\librosa\core\__pycache__`，受限环境下写不进去且会死循环 | **在 `config.py` 最开头设置 `NUMBA_CACHE_DIR` 到工作区内可写目录**（本方案已内置：`q1_output\_numba_cache`） |
| 抽帧后 `frames/` 目录是空的，视觉特征只有 1 帧 | `cv2.imwrite` 在中文路径下静默失败 | 改用 `cv2.imencode(...)[1].tofile(路径)` 写、`cv2.imdecode(np.fromfile(...))` 读 |
| OpenFace 报 `Could not load the landmark detector` / 缺 `cen_patches_*.dat` | 标准二进制包缺 CEN patch experts（仅托管于 Dropbox/OneDrive，国内不可达） | 用 `-mloc "model/main_clnf_general.txt"`（CLNF，CCNF 权重随包已含），本方案已内置 |
| Whisper 把英文整段听写成希伯来语等其它语言 | 自动语种识别在短音频/带噪片段上出错 | `transcribe(..., language="en", task="transcribe")` |
| Whisper 听写内容和数据文本对不上 | 部分片段的数据本身不一致（见第2.3节） | 记录 `word_match_ratio`，低置信度走 VAD 兜底；不丢样本 |
| `Failed to launch Triton kernels ... falling back to a slower DTW implementation` | 没装 CUDA toolkit（Triton 是可选加速） | 只是警告，可以忽略 |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` / 缓存符号链接警告 | Windows 未开开发者模式 | 可忽略；或设 `HF_HUB_DISABLE_SYMLINKS_WARNING=1` |
| 模型下载到 93% 报 `SHA256 checksum does not match` | 网络中断，文件没下全 | 再跑一次下载命令即可 |
| `No module named xxx`（明明刚装过） | 包装到了用户目录而不是 conda 环境 | `pip install --user 包名`，并重启内核 |

---

## 10. 可复现性信息（直接抄进文档附录）

```
操作系统        : Windows
Python          : 3.10.13  (python)
PyTorch         : 2.0.0+cu118  (CUDA available = True)
Transformers    : 4.44.2
Tokenizers      : 0.19.1
librosa         : 0.11.0
OpenCV          : 4.10.0
openai-whisper  : (本机已安装版本)  模型: base + small
ffmpeg          : 由 imageio-ffmpeg 0.6.0 提供

统一参数：
  音频采样率 AUDIO_SR      = 16000 Hz，单声道
  语音帧率 AUDIO_FPS       = 100 fps（hop = 160 点）
  视觉帧率 VISION_FPS      = 30 fps
  时间箱数 N_BINS          = 50
  词级最大长度 MAX_WORDS   = 50
  池化平滑项 EPS           = 1e-8
  特征维度 TEXT/AUDIO/VIS  = 768 / 74 / 35
  BERT max_length          = 256
  Whisper 参数             = language="en", task="transcribe",
                             word_timestamps=True,
                             condition_on_previous_text=False
  ASR 递进阈值 ASR_ACCEPT_RATIO = 0.70
  VAD 参数                 = librosa.effects.split(top_db=30)
  预处理去极值 SIGMA_CLIP  = 5.0（μ±5σ，逐样本逐维度）
  互相关最大滞后 NCC_TAU_MAX = 15 帧（≈0.5s@30fps）
  音视同步代理             = 音频RMS ↔ 视觉AU25+AU26(嘴部张开)

运行流程：
  python pipeline.py audio → vision → text → build → summary
  python figures.py    # 预览图 PNG
  python vector_figs.py # 矢量图 PDF(9张)
  python visualize.py # 旧四层图(可选)
```

---

## 11. 质量自检清单

| # | 检查项 | 结果 |
|---|---|---|
| 1 | 100 条样本全部有输出（样本编号、模态文件、特征文件一一对应） | ✅ |
| 2 | 序列位置可核验（50 个时间箱，k ↔ [k·T/50,(k+1)·T/50)） | ✅ |
| 3 | 有效长度可核验（`length` + `cₖ` 计数 + `align_records.json`） | ✅ |
| 4 | 填充规则可核验（padding 用 length，真实缺失用 mask） | ✅ |
| 5 | 特征提取工具与关键参数完整记录 | ✅（第1.2、4、10节） |
| 6 | 处理日志与复现实验说明 | ✅（`q1_output\run_*.log`） |
| 7 | 跨模态互补性论证 | ✅（`analysis/corr_matrix.csv` + `figs/corr_heatmap.png`） |
| 8 | 时序同步可核验 | ✅（`analysis/sync_metrics.csv` + `figs/ncc_curve_typical.png` + `figs/lag_histogram.png`） |
| 9 | 典型样本验证图 | ✅（`typical_sample.png`） |
| 10 | 全量特征文件随仓库发布 | ✅（`features/`） |

---

## 12. 待改进项（后续讨论直接改本文件）

| 优先级 | 事项 | 现状 | 目标 |
|---|---|---|---|
| ~~高~~ | **视觉 35 维升级（已做）** | ~~OpenCV 5×7 灰度网格~~ | ✅ 已换 OpenFace 2.2.0：17 AU+6 姿态+8 注视+4 眼部几何=35 维 |
| 高 | **音频 74 维对齐标准** | librosa 自实现 24MFCC+24Δ+24ΔΔ+F0+RMS | 若需与标准化特征严格一致，改用 COVAREP（原版 74 维含 12×2 基频+谱参数） |
| 中 | **文本-音频不一致的 6 个视频** | 已识别、已标注置信度、已兜底 | 讨论是否向数据提供方反馈，或在文档中作为数据质量发现展开 |
| 中 | **ASR 更强方案** | Whisper base→small | 可试 `medium`，或用 WhisperX + wav2vec2 做**强制对齐**（直接对齐给定文本，鲁棒性最好） |
| 中 | **词级时间戳精度** | 序列对齐 + 线性插值 | 引入音素级强制对齐（MFA）替代插值 |
| 低 | 语音/视觉特征归一化 | 未做跨样本标准化 | 加 z-score / CMVN，供模块二使用 |
| 低 | 一键脚本 | 需手动跑两步命令 | 加 `run_all.py` 或 `.bat` |

---

*（本文件为活文档：任何方法改动、参数调整、新增实验，均直接在下表追加修订记录，不另开新文件。）*

### 修订记录（续）
| 版本 | 日期 | 修订内容 | 状态 |
|---|---|---|---|
| v0.2 | 2026-09-23 | 实跑100条通过；ASR序列对齐；VAD兜底；数据质量审计；Spyder操作手册 | 当前 |
| | | | |
