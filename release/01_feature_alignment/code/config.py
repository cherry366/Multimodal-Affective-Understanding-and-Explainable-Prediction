# -*- coding: utf-8 -*-
"""模块一 统一配置（融合版：对齐优先 + 时间箱/词级双输出 + 掩码闭环）"""
import os

# --- 模块根目录自动检测 ---
# 优先用环境变量 Q1_ROOT; 否则取本文件所在目录(code)的上一级。
# 这样只要把 code 放在模块根目录下即可直接运行, 无需改路径。
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("Q1_ROOT") or os.path.dirname(_HERE)

# --- 必须在 import librosa/numba 之前设置 ---
# numba 默认把 JIT 缓存写到 site-packages 的 __pycache__，在受限环境下会卡死；
# 这里统一改写到工作区内可写目录。
_NUMBA_CACHE = os.path.join(ROOT, "q1_output", "_numba_cache")
os.makedirs(_NUMBA_CACHE, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", _NUMBA_CACHE)
os.environ.setdefault("MPLCONFIGDIR", _NUMBA_CACHE)
DATA = os.path.join(ROOT, "data")
ATT1 = os.path.join(DATA, "raw")
LABEL_XLSX = os.path.join(ATT1, "label-100.xlsx")

OUT = os.path.join(ROOT, "q1_output")
AUDIO_DIR   = os.path.join(OUT, "audio")
FRAMES_DIR  = os.path.join(OUT, "frames")
FEAT_AUDIO  = os.path.join(OUT, "feat_audio")
FEAT_VISION = os.path.join(OUT, "feat_vision")
FEAT_TEXT   = os.path.join(OUT, "feat_text")
ALIGN_DIR   = os.path.join(OUT, "align")
RAW_ALIGN_DIR = os.path.join(OUT, "align_raw")

# --- OpenFace 2.2.0 视觉特征 ---
# OpenFace 安装目录(可用环境变量 OPENFACE_DIR 覆盖, 协作者机器路径不同时改这里)
OPENFACE_DIR = os.environ.get("OPENFACE_DIR", r"E:\tools\OpenFace_2.2.0_win_x64")
OPENFACE_EXE = os.path.join(OPENFACE_DIR, "FeatureExtraction.exe")
# 关键说明: 标准二进制包缺 CEN patch experts(cen_patches_*.dat, 仅托管于 Dropbox/OneDrive, 国内不可达),
# 默认 CECLM 模型会因缺权而报错; 改用 CLNF 模型(其 CCNF patch experts 随包已含)即可完整运行。
OPENFACE_MLOC = "model/main_clnf_general.txt"
OF_RAW_DIR = os.path.join(OUT, "of_raw")
META_DIR    = os.path.join(OUT, "meta")
SUMMARY_CSV = os.path.join(OUT, "summary.csv")
PKL_TIMEBIN = os.path.join(OUT, "q1_features_timebin.pkl")
PKL_WORD    = os.path.join(OUT, "q1_features_word.pkl")
ANALYSIS_DIR = os.path.join(OUT, "analysis")   # 相关性/同步质量分析产物
FIGS_DIR     = os.path.join(OUT, "figs")       # 可视化图片

# --- 预处理三件套 & 质量校验参数 ---
SIGMA_CLIP = 5.0     # μ±5σ 去极值(逐样本逐维度, 仅有效格)
NCC_TAU_MAX = 15     # 音视互相关最大滞后(视觉帧, 15帧≈0.5s@30fps)
F0_IDX  = 72         # 74维语音: 24 MFCC + 24 Δ + 24 ΔΔ 之后为 F0
RMS_IDX = 73         # 74维语音: F0 之后为 RMS
AU_START, AU_END = 0, 17   # 35维视觉: 前 17 维为 AU 回归强度
AU_MOUTH_IDX = [14, 15]   # AU25(嘴唇分开)+AU26(下巴下降), 与说话声强同步


N_BINS     = 50      # 时间箱数量(时间箱版)
MAX_WORDS  = 50      # 词级版最大词数(与标准化特征aligned_50 一致)
EPS        = 1e-8    # 均值池化分母平滑项
AUDIO_SR   = 16000
VISION_FPS = 30
AUDIO_FPS  = 100
AUDIO_DIM  = 74
VISION_DIM = 35
TEXT_DIM   = 768
WHISPER_SIZES    = ["base", "small"]   # 递进式 ASR：先 base，匹配率低再上 small
ASR_ACCEPT_RATIO = 0.70                # 词匹配率达标即停止升级模型

CLASS_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
CLASS_NAMES = ["Negative", "Neutral", "Positive"]

def ensure_dirs():
    for d in [OUT, AUDIO_DIR, FRAMES_DIR, FEAT_AUDIO,
              FEAT_VISION, FEAT_TEXT, ALIGN_DIR, RAW_ALIGN_DIR, OF_RAW_DIR, META_DIR,
              ANALYSIS_DIR, FIGS_DIR]:
        os.makedirs(d, exist_ok=True)

def get_ffmpeg():
    """定位 ffmpeg：环境变量 -> imageio-ffmpeg -> PATH"""
    p = os.environ.get("FFMPEG_PATH")
    if p and os.path.exists(p):
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    import shutil
    p = shutil.which("ffmpeg")
    if p:
        return p
    raise FileNotFoundError(
        "未找到 ffmpeg。请任选其一：\n"
        "1) pip install imageio-ffmpeg\n"
        "2) 安装 ffmpeg 并加入 PATH\n"
        "3) 设置环境变量 FFMPEG_PATH")
