#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
包裹数据集半自动筛选脚本
Parcel Dataset Active Filter
============================================================

【这个脚本解决什么问题？】

当生产现场不断产生新图片时，并不是所有图片都值得重新人工标注。
这个脚本的目标是：

1. 自动找出“高重复/近重复图片”
2. 自动找出“严重模糊、严重过曝、严重欠曝等低价值图片”
3. 调用当前 YOLO / YOLO-OBB 模型进行预测
4. 根据当前模型的预测表现，寻找：
   - 漏检嫌疑样本
   - 低置信度样本
   - 不确定样本
   - 边缘目标
   - 小目标
   - 大目标
   - 遮挡 / 贴靠 / 重叠
   - 特殊角度
   - 极端长宽比
   - 多目标拥挤
   - 中度运动模糊
   - 中度强光 / 阴影
5. 将图片自动分流到：

   KEEP/high_value   ：高价值难例，优先人工复核和标注
   KEEP/valuable     ：有价值样本，可进入候选训练池
   DROP/duplicate    ：高重复 / 近重复图片
   DROP/low_value    ：严重不可用，或当前模型已经稳定掌握的简单样本

6. 输出 screening_report.csv，记录每张图为什么被保留或丢弃。

------------------------------------------------------------
【最重要的使用方式】

你不需要再通过命令行传参数。

只需要修改下面“用户配置区”里的路径和参数，然后直接运行本文件：

    INPUT_DIR
    MODEL_PATH
    OUTPUT_DIR

例如在 Spyder / PyCharm / VS Code 中直接点击“运行”即可。

------------------------------------------------------------
【非常重要】

本脚本默认采用 COPY_MODE = "copy"：

- 不删除原图
- 不移动原图
- 不改原始数据
- 只是把筛选结果复制到新的目录

因此可以放心先做测试。

首次使用强烈建议：

    MAX_FILES = 500

先筛 500 张，人工检查四个输出文件夹，确认阈值合适之后，
再把：

    MAX_FILES = 0

改成 0，表示处理全部图片。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from ultralytics import YOLO
except Exception as exc:
    raise SystemExit(
        "\n未能导入 ultralytics。\n"
        "请先在当前 Python 环境安装：\n\n"
        "    pip install ultralytics opencv-python numpy tqdm\n\n"
        f"原始错误：{exc}\n"
    )


# ============================================================
# 0. 用户配置区
# ============================================================
#
# 正常情况下，你主要只需要修改这一部分。
#
# 推荐第一次：
#   1. 修改 INPUT_DIR
#   2. 修改 MODEL_PATH
#   3. 修改 OUTPUT_DIR
#   4. MAX_FILES = 500
#   5. 运行并人工查看结果
#   6. 确认合理后 MAX_FILES = 0，处理全部图片
#
# ============================================================


# ------------------------------------------------------------
# 0.1 三个最重要的路径
# ------------------------------------------------------------

# 原始图片所在的总目录。
# 脚本会递归扫描它下面的所有子文件夹。
#
# Windows 路径前面建议加 r，例如：
# r"D:\parcel_project\raw_images"
INPUT_DIR = r"F:\image_process_data\images_filter_input"


# 当前正在生产环境中使用、或者当前训练完成的 YOLO / YOLO-OBB 模型。
#
# 对 YOLO11-OBB，一般就是：
# runs/obb/train/weights/best.pt
MODEL_PATH = r"F:\obb_test\models\best-083102s.pt"


# 筛选结果保存到哪里。
#
# 程序会自动建立：
#
# OUTPUT_DIR/
# ├─ KEEP/
# │  ├─ high_value/
# │  └─ valuable/
# ├─ DROP/
# │  ├─ duplicate/
# │  └─ low_value/
# └─ reports/
#    ├─ screening_report.csv
#    └─ summary.json
#
OUTPUT_DIR = r"F:\image_process_data\images_filter_output"


# ------------------------------------------------------------
# 0.2 运行规模与硬件设置
# ------------------------------------------------------------

# 第一次建议设成 300~1000 做小规模检查。
#
# 0   = 全部图片
# 500 = 只处理排序后的前 500 张
MAX_FILES = 0


# 推理设备。
#
# NVIDIA 第一张显卡：
DEVICE = "cpu"
#
# 如果没有 NVIDIA GPU：
# DEVICE = "cpu"
#
# 如果想让 Ultralytics 自动选择：
# DEVICE = None


# YOLO 推理输入尺寸。
#
# 如果你的 OBB 模型训练时 imgsz=1024，可改成 1024。
# 960 是精度和速度之间比较稳妥的选择。
IMGSZ = 960


# 推理 batch。
#
# 显存足够：16 / 32
# 显存不足：8 / 4
BATCH_SIZE = 4


# ------------------------------------------------------------
# 0.3 输出方式
# ------------------------------------------------------------

# "copy"：
#   真正复制图片。
#   最安全、最直观，但占用额外磁盘空间。
#
# "hardlink"：
#   Windows/NTFS 同一磁盘分区上可以节省空间。
#   如果硬链接失败，代码会自动退回 copy。
COPY_MODE = "copy"


# True：
#   只分析并生成 CSV，不实际复制图片。
#
# False：
#   正常把图片复制到 KEEP / DROP。
DRY_RUN = False


# ------------------------------------------------------------
# 0.4 当前模型“不确定性”相关阈值
# ------------------------------------------------------------
#
# 这一部分决定“什么样的预测值得重新学习”。
#
# 原理：
# 当前模型已经非常确定的常规样本，对下一轮训练的信息量通常较低。
# 当前模型低置信、不确定甚至完全漏检的图片，则往往更值得人工检查。
# ------------------------------------------------------------


# 模型推理时最低保留置信度。
#
# 注意：
# 这里故意设得很低。
# 因为我们不是为了最终展示检测结果，而是为了发现“模型不确定的样本”。
#
# 如果设成 0.5，那么低置信预测会直接被模型过滤掉，
# 我们反而看不到这些困难样本。
INFER_CONF = 0.05


# NMS / OBB NMS 的 IoU 阈值。
INFER_IOU = 0.70


# 把 0.15 ~ 0.55 定义为“不确定预测区域”。
#
# 例如：
# 0.95 -> 模型很确定
# 0.72 -> 相对确定
# 0.38 -> 很值得重新看
# 0.12 -> 可能是明显困难目标，也可能是假阳性
UNCERTAIN_LOW = 0.15
UNCERTAIN_HIGH = 0.55


# 低于这个置信度，视作“极低置信预测”。
VERY_LOW_CONF = 0.15


# 如果一张普通图片平均置信度高于这个值，
# 且没有边缘、小目标、遮挡等困难因素，
# 会被认为是“当前模型已经稳定掌握的简单样本”。
EASY_CONF = 0.9


# 简单样本不能全部删除，否则数据集会逐渐只剩难例，
# 从而破坏实际生产分布。
#
# 所以对非常简单的图片随机保留一部分。
#
# 0.10 = 保留 10%
# 0.20 = 保留 20%
EASY_KEEP_RATE = 0.10


# ------------------------------------------------------------
# 0.5 目标位置 / 尺寸 / 形状阈值
# ------------------------------------------------------------


# 距离图片边缘小于图像宽高的 3.5%，认为是“边缘目标”。
#
# 边缘目标通常容易：
# - 截断
# - 漏检
# - OBB 角点不稳定
EDGE_MARGIN = 0.035


# 目标 OBB / bbox 面积小于整幅图面积的 0.5%，认为是“小目标”。
#
# 如果你的相机非常高、包裹经常很小，可适当增大到 0.008~0.015。
SMALL_AREA_RATIO = 0.005


# 目标面积大于整幅图 35%，认为是“超大近景目标”。
HUGE_AREA_RATIO = 0.35


# OBB 旋转角度位于 15°~75° 时认为存在明显旋转。
#
# 对物流包裹来说，0° / 90°附近往往更常见，
# 中间角度可作为一个“特殊角度”信号。
ANGLED_MIN_DEG = 15.0
ANGLED_MAX_DEG = 75.0


# OBB 长边 / 短边大于 3.5，认为是“极端长宽比”。
#
# 例如长条形包裹、被遮挡后只露出部分区域等。
EXTREME_ASPECT_RATIO = 3.5


# ------------------------------------------------------------
# 0.6 图像质量阈值
# ------------------------------------------------------------
#
# 这里特别区分：
#
# 1. 严重到“人都很难判断”的图片
#       -> DROP/low_value
#
# 2. 中度模糊 / 中度强光 / 阴影，但人仍能标注
#       -> 反而属于困难样本，应加分
#
# 这是生产数据集很重要的一点：
# 不能把所有模糊、强光图片都删掉，否则模型永远学不会真实异常场景。
# ------------------------------------------------------------


# Laplacian variance 小于 20：
# 通常已经属于非常严重的模糊。
FATAL_BLUR = 20.0


# 20 ~ 80：
# 定义为中度模糊。
#
# 这类图片通常仍然可以人工判断，
# 但模型容易失败，所以作为困难样本加分。
MODERATE_BLUR = 80.0


# 灰度 <= 20 的像素比例超过 35%，认为严重欠曝。
FATAL_DARK_RATIO = 0.35


# 灰度 >= 235 的像素比例超过 35%，认为严重过曝。
FATAL_BRIGHT_RATIO = 0.35


# 暗区域超过 12%，开始视为“有明显阴影/欠曝特征”。
MODERATE_DARK_RATIO = 0.12


# 亮区域超过 10%，开始视为“有明显强光/高光特征”。
MODERATE_BRIGHT_RATIO = 0.10


# ------------------------------------------------------------
# 0.7 近重复 / 高重复检测参数
# ------------------------------------------------------------
#
# 重复检测采用：
#
#   pHash
#      +
#   HSV 颜色直方图
#
# 双重确认。
#
# 这样比单独使用文件 MD5 更适合：
# - 连续拍摄
# - 视频抽帧
# - 轻微亮度变化
# - 轻微压缩变化
# - 画面只发生极小位移
# ------------------------------------------------------------


# 64-bit pHash 汉明距离。
#
# 越小 -> 判断更严格
# 越大 -> 更容易把相似图片认为是重复
#
# 一般建议：
# 5~6  非常严格
# 7~9  常用
# 10+  需要谨慎
DUP_PHASH_HAMMING = 5


# HSV 直方图相关系数。
#
# 越接近 1，图像整体颜色分布越类似。
# 0.95 是比较保守的近重复确认值。
DUP_HIST_CORR = 0.95


# 一个近重复组最多保留多少张。
#
# 对连续视频帧：
# 1 -> 去重最激进
# 2 -> 默认推荐
# 3~5 -> 希望保留更多微小变化
MAX_KEEP_PER_DUP_GROUP = 2


# ------------------------------------------------------------
# 0.8 样本价值分级阈值
# ------------------------------------------------------------
#
# 每种困难属性都会给图片增加 challenge_score。
#
# 最终：
#
# score >= HIGH_VALUE_THRESHOLD
#     -> KEEP/high_value
#
# VALUE_THRESHOLD <= score < HIGH_VALUE_THRESHOLD
#     -> KEEP/valuable
#
# 对没有明显困难因素的高置信简单图，
#     -> 按 EASY_KEEP_RATE 抽样保留
# ------------------------------------------------------------


HIGH_VALUE_THRESHOLD = 4.0
VALUABLE_THRESHOLD = 1.5


# 随机保留简单样本时使用固定 seed，
# 保证重复运行结果尽可能稳定。
RANDOM_SEED = 42


# ============================================================
# 1. 支持的图片格式
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# 2. 数据结构
# ============================================================

@dataclass
class PreMetrics:
    """
    不调用 YOLO 时就能得到的图像级指标。

    主要用于：
    - 图像质量判断
    - 近重复检测
    - 决定是否值得继续做模型推理
    """

    path: str
    rel_path: str

    readable: bool = True

    width: int = 0
    height: int = 0

    # Laplacian variance：
    # 越低通常越模糊。
    blur_var: float = 0.0

    # 平均灰度，0~255。
    brightness: float = 0.0

    # 极暗像素比例。
    dark_ratio: float = 0.0

    # 极亮像素比例。
    bright_ratio: float = 0.0

    # 64-bit 感知哈希。
    phash: int = 0

    # 用于同一重复组中选择“质量最好代表图”的内部评分。
    quality_score: float = 0.0

    # 严重质量问题：
    # True 时通常直接 DROP/low_value。
    fatal_quality: bool = False

    # 中度模糊：
    # 不直接删除，反而可能是困难样本。
    moderate_blur: bool = False

    # 中度强光 / 阴影。
    moderate_lighting: bool = False

    # 近重复组编号。
    # -1 代表没有进入重复组。
    duplicate_group: int = -1

    # 在重复组内的质量排名。
    duplicate_rank: int = -1

    # 是否因为重复而被丢弃。
    is_duplicate_drop: bool = False


@dataclass
class ModelMetrics:
    """
    当前 YOLO / YOLO-OBB 对图片推理后得到的指标。

    这些指标用于回答：

        “这张图是不是当前模型的困难样本？”
    """

    # 检测到多少个目标。
    n_det: int = 0

    # 置信度统计。
    mean_conf: float = 0.0
    min_conf: float = 0.0
    max_conf: float = 0.0

    # 落入“不确定置信度区间”的目标比例。
    uncertain_ratio: float = 0.0

    # 极低置信目标比例。
    very_low_conf_ratio: float = 0.0

    # 靠近图片边缘的目标比例。
    edge_ratio: float = 0.0

    # 小目标比例。
    small_ratio: float = 0.0

    # 超大目标比例。
    huge_ratio: float = 0.0

    # 明显旋转目标比例。
    angled_ratio: float = 0.0

    # 极端长宽比目标比例。
    extreme_aspect_ratio: float = 0.0

    # 目标之间明显重叠 / 遮挡的目标对比例。
    overlap_pair_ratio: float = 0.0


@dataclass
class FinalRecord:
    """
    最终写入 CSV 的一行。

    每一张图片都会对应一条记录。
    """

    path: str
    rel_path: str

    category: str
    keep_drop: str
    reasons: str

    duplicate_group: int
    duplicate_rank: int

    width: int
    height: int

    blur_var: float
    brightness: float
    dark_ratio: float
    bright_ratio: float
    quality_score: float

    n_det: int
    mean_conf: float
    min_conf: float
    max_conf: float

    uncertain_ratio: float
    very_low_conf_ratio: float

    edge_ratio: float
    small_ratio: float
    huge_ratio: float
    angled_ratio: float
    extreme_aspect_ratio: float
    overlap_pair_ratio: float

    challenge_score: float
    output_path: str


# ============================================================
# 3. OpenCV 读图
# ============================================================

def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """
    使用 np.fromfile + cv2.imdecode 读取图片。

    为什么不用简单的：

        cv2.imread(...)

    因为在部分 Windows / OpenCV 环境下，
    cv2.imread 对中文路径兼容性可能不好。

    这个写法对：
    - 中文目录
    - 中文文件名
    通常更稳。
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)

        if data.size == 0:
            return None

        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    except Exception:
        return None


# ============================================================
# 4. 图像预处理与图像质量
# ============================================================

def resize_long_side(
    img: np.ndarray,
    long_side: int = 640,
) -> np.ndarray:
    """
    将图片最长边缩放到 long_side。

    图像质量分析不需要使用原始超高分辨率，
    缩小以后：
    - 速度更快
    - 内存占用更小
    - 对亮度/模糊/直方图统计影响很小
    """
    h, w = img.shape[:2]

    current_long_side = max(h, w)

    if current_long_side <= long_side:
        return img

    scale = long_side / float(current_long_side)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def compute_phash64(img_bgr: np.ndarray) -> int:
    """
    计算 64-bit perceptual hash（感知哈希）。

    与 MD5 不同：

    两张 JPEG 即使只是重新压缩一次，
    MD5 也会完全不同。

    但 pHash 比较的是视觉结构，
    所以对：
    - 轻微压缩
    - 亮度略微变化
    - 连续视频帧
    更适合判断“近重复”。

    计算流程：

    1. 转灰度
    2. 缩放到 32 x 32
    3. 做 DCT
    4. 取左上角 8 x 8 低频区域
    5. 与中位数比较
    6. 形成 64-bit 二进制特征
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    )

    dct = cv2.dct(np.float32(gray))

    low_frequency = dct[:8, :8].copy()

    flat = low_frequency.flatten()

    # 第一个系数是 DC 分量，
    # 主要反映整幅图平均亮度，所以不参与中位数判断。
    values = flat[1:]

    median_value = float(np.median(values))

    bits = np.zeros(64, dtype=np.uint8)

    bits[0] = 0
    bits[1:] = (values > median_value).astype(np.uint8)

    hash_value = 0

    for bit in bits:
        hash_value = (hash_value << 1) | int(bit)

    return int(hash_value)


def hsv_histogram(img_bgr: np.ndarray) -> np.ndarray:
    """
    提取 HSV 二维颜色直方图。

    为什么在 pHash 之外再使用 HSV？

    因为只靠 pHash 有时会把：
    - 结构相似
    - 但颜色/场景明显不同
    的图片误认为近重复。

    因此采用：

        pHash 负责“快速召回”
              +
        HSV histogram 负责“再次确认”
    """
    small = resize_long_side(img_bgr, 320)

    hsv = cv2.cvtColor(
        small,
        cv2.COLOR_BGR2HSV,
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 16],
        [0, 180, 0, 256],
    )

    cv2.normalize(
        hist,
        hist,
        alpha=1.0,
        norm_type=cv2.NORM_L1,
    )

    return hist.astype(np.float32)


def calc_quality(
    img_bgr: np.ndarray,
) -> Tuple[dict, np.ndarray]:
    """
    计算图像质量。

    这里不把所有“异常图片”都视为低价值。

    逻辑是：

    ----------------------------------------------------------
    严重异常：
        已经很难人工标注
        -> low_value

    中度异常：
        人仍然能标注，但当前模型可能容易失败
        -> 困难样本，加分
    ----------------------------------------------------------

    返回：
        quality metrics
        HSV histogram
    """
    img = resize_long_side(
        img_bgr,
        640,
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY,
    )

    # --------------------------------------------------------
    # 4.1 模糊度
    # --------------------------------------------------------
    #
    # Laplacian 对图像边缘敏感。
    # 清晰图像通常有更多高频边缘，
    # 因此 Laplacian variance 较大。
    #
    # 模糊图像边缘变少，
    # variance 会明显下降。
    #
    blur_var = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    # --------------------------------------------------------
    # 4.2 整体亮度
    # --------------------------------------------------------
    brightness = float(gray.mean())

    # --------------------------------------------------------
    # 4.3 极暗像素比例
    # --------------------------------------------------------
    dark_ratio = float(
        np.mean(gray <= 20)
    )

    # --------------------------------------------------------
    # 4.4 极亮像素比例
    # --------------------------------------------------------
    bright_ratio = float(
        np.mean(gray >= 235)
    )

    # --------------------------------------------------------
    # 4.5 是否严重到不可用
    # --------------------------------------------------------
    fatal_quality = (
        blur_var < FATAL_BLUR
        or dark_ratio >= FATAL_DARK_RATIO
        or bright_ratio >= FATAL_BRIGHT_RATIO
    )

    # --------------------------------------------------------
    # 4.6 中度模糊
    # --------------------------------------------------------
    moderate_blur_flag = (
        FATAL_BLUR
        <= blur_var
        < MODERATE_BLUR
    )

    # --------------------------------------------------------
    # 4.7 中度光照异常
    # --------------------------------------------------------
    moderate_lighting = (
        MODERATE_DARK_RATIO
        <= dark_ratio
        < FATAL_DARK_RATIO
    ) or (
        MODERATE_BRIGHT_RATIO
        <= bright_ratio
        < FATAL_BRIGHT_RATIO
    )

    # --------------------------------------------------------
    # 4.8 quality_score
    # --------------------------------------------------------
    #
    # 这个值不是“训练价值分数”。
    #
    # 它只用于：
    # 同一个近重复组里，如果要保留 1~2 张，
    # 尽量保留更清晰、更正常、分辨率更好的代表图。
    #
    blur_quality = float(
        np.clip(
            (blur_var - FATAL_BLUR)
            / max(
                1.0,
                180.0 - FATAL_BLUR,
            ),
            0.0,
            1.0,
        )
    )

    exposure_penalty = float(
        np.clip(
            max(
                dark_ratio
                / max(
                    1e-6,
                    FATAL_DARK_RATIO,
                ),
                bright_ratio
                / max(
                    1e-6,
                    FATAL_BRIGHT_RATIO,
                ),
            ),
            0.0,
            1.0,
        )
    )

    original_h, original_w = img_bgr.shape[:2]

    resolution_quality = float(
        np.clip(
            math.sqrt(
                original_h
                * original_w
            ) / 1500.0,
            0.0,
            1.0,
        )
    )

    quality_score = (
        0.50 * blur_quality
        + 0.30 * (1.0 - exposure_penalty)
        + 0.20 * resolution_quality
    )

    return (
        {
            "blur_var": blur_var,
            "brightness": brightness,
            "dark_ratio": dark_ratio,
            "bright_ratio": bright_ratio,
            "quality_score": quality_score,
            "fatal_quality": fatal_quality,
            "moderate_blur": moderate_blur_flag,
            "moderate_lighting": moderate_lighting,
        },
        hsv_histogram(img_bgr),
    )


# ============================================================
# 5. 近重复检测
# ============================================================

def hamming64(
    a: int,
    b: int,
) -> int:
    """
    计算两个 64-bit pHash 之间的汉明距离。

    例如：

    distance = 0
        完全相同的 pHash

    distance 很小
        图像视觉结构高度相似

    distance 很大
        图像差别较大
    """
    return int(
        (a ^ b).bit_count()
    )


class BKNode:
    """
    BK-tree 节点。

    BK-tree 适合做“离散距离下的近邻搜索”。

    在这里用来解决：

        有很多图片时，不希望每一张都和所有图片比较 pHash。

    否则 N 张图要做近似 N² 次比较，
    数据量大时会越来越慢。
    """

    def __init__(
        self,
        value: int,
        index: int,
    ):
        self.value = value
        self.index = index
        self.children: Dict[int, "BKNode"] = {}


class BKTree:
    """
    pHash BK-tree。
    """

    def __init__(self):
        self.root: Optional[BKNode] = None

    def add(
        self,
        value: int,
        index: int,
    ) -> None:
        """
        把一个 pHash 插入树中。
        """
        if self.root is None:
            self.root = BKNode(
                value,
                index,
            )
            return

        node = self.root

        while True:
            distance = hamming64(
                value,
                node.value,
            )

            child = node.children.get(
                distance
            )

            if child is None:
                node.children[distance] = BKNode(
                    value,
                    index,
                )
                return

            node = child

    def search(
        self,
        value: int,
        max_dist: int,
    ) -> List[int]:
        """
        搜索与 value 的汉明距离 <= max_dist 的候选图片。
        """
        if self.root is None:
            return []

        result_indices: List[int] = []

        stack = [self.root]

        while stack:
            node = stack.pop()

            distance = hamming64(
                value,
                node.value,
            )

            if distance <= max_dist:
                result_indices.append(
                    node.index
                )

            lower = distance - max_dist
            upper = distance + max_dist

            for (
                edge_distance,
                child,
            ) in node.children.items():

                if (
                    lower
                    <= edge_distance
                    <= upper
                ):
                    stack.append(child)

        return result_indices


class UnionFind:
    """
    并查集。

    如果：
        A 与 B 相似
        B 与 C 相似

    那么把 A、B、C 聚成同一个近重复组。
    """

    def __init__(
        self,
        n: int,
    ):
        self.parent = list(
            range(n)
        )

        self.rank = [0] * n

    def find(
        self,
        x: int,
    ) -> int:

        while (
            self.parent[x] != x
        ):
            self.parent[x] = self.parent[
                self.parent[x]
            ]

            x = self.parent[x]

        return x

    def union(
        self,
        a: int,
        b: int,
    ) -> None:

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if (
            self.rank[root_a]
            < self.rank[root_b]
        ):
            root_a, root_b = (
                root_b,
                root_a,
            )

        self.parent[root_b] = root_a

        if (
            self.rank[root_a]
            == self.rank[root_b]
        ):
            self.rank[root_a] += 1


def find_duplicate_groups(
    pre_metrics: List[PreMetrics],
    histograms: List[Optional[np.ndarray]],
) -> Dict[int, List[int]]:
    """
    搜索近重复组。

    两层判断：

    第 1 层：
        pHash 汉明距离 <= DUP_PHASH_HAMMING

    第 2 层：
        HSV histogram correlation >= DUP_HIST_CORR

    两者同时满足，才把两张图片连成同一个重复组。
    """
    n = len(pre_metrics)

    union_find = UnionFind(n)

    tree = BKTree()

    # 为避免极端情况下候选过多，
    # 每张图片最多精查最接近的 64 个候选。
    max_candidates_per_image = 64

    for i in tqdm(
        range(n),
        desc="近重复检测",
        unit="img",
    ):
        current = pre_metrics[i]

        if not current.readable:
            continue

        candidate_indices = tree.search(
            current.phash,
            DUP_PHASH_HAMMING,
        )

        if (
            len(candidate_indices)
            > max_candidates_per_image
        ):
            candidate_indices.sort(
                key=lambda j: hamming64(
                    current.phash,
                    pre_metrics[j].phash,
                )
            )

            candidate_indices = (
                candidate_indices[
                    :max_candidates_per_image
                ]
            )

        current_hist = histograms[i]

        if current_hist is not None:

            for j in candidate_indices:

                other_hist = histograms[j]

                if other_hist is None:
                    continue

                correlation = float(
                    cv2.compareHist(
                        current_hist,
                        other_hist,
                        cv2.HISTCMP_CORREL,
                    )
                )

                if (
                    correlation
                    >= DUP_HIST_CORR
                ):
                    union_find.union(
                        i,
                        j,
                    )

        tree.add(
            current.phash,
            i,
        )

    groups: Dict[
        int,
        List[int],
    ] = defaultdict(list)

    for i in range(n):

        if pre_metrics[i].readable:

            group_root = union_find.find(i)

            groups[group_root].append(i)

    # 只有 2 张及以上的组，
    # 才是真正意义上的“重复组”。
    return {
        root: indices
        for root, indices
        in groups.items()
        if len(indices) > 1
    }


def mark_duplicate_drops(
    pre_metrics: List[PreMetrics],
    groups: Dict[int, List[int]],
) -> None:
    """
    对每个近重复组进行排序。

    排序优先级：

    1. 没有严重质量问题
    2. quality_score 更高
    3. 原始分辨率更高

    然后只保留前 MAX_KEEP_PER_DUP_GROUP 张。

    例如：
        MAX_KEEP_PER_DUP_GROUP = 2

    一个连续帧组有 20 张：
        最好两张 -> 继续参加模型价值分析
        其余 18 张 -> DROP/duplicate
    """
    group_id = 0

    for indices in groups.values():

        group_id += 1

        ranked = sorted(
            indices,
            key=lambda i: (
                pre_metrics[i].fatal_quality,
                -pre_metrics[i].quality_score,
                -(
                    pre_metrics[i].width
                    * pre_metrics[i].height
                ),
            ),
        )

        for rank, i in enumerate(
            ranked,
            start=1,
        ):
            pre_metrics[i].duplicate_group = (
                group_id
            )

            pre_metrics[i].duplicate_rank = (
                rank
            )

            if (
                rank
                > MAX_KEEP_PER_DUP_GROUP
            ):
                pre_metrics[
                    i
                ].is_duplicate_drop = True


# ============================================================
# 6. YOLO / YOLO-OBB 结果解析
# ============================================================

def polygon_area(
    polygon: np.ndarray,
) -> float:
    """
    计算 OBB 四边形面积。
    """
    return float(
        abs(
            cv2.contourArea(
                np.asarray(
                    polygon,
                    dtype=np.float32,
                )
            )
        )
    )


def overlap_pair_ratio(
    polygons: List[np.ndarray],
) -> float:
    """
    估算多个包裹之间的遮挡 / 贴靠 / 重叠程度。

    对每一对检测框：

        overlap =
        intersection_area / min(area_A, area_B)

    如果 overlap >= 0.15，
    就认为这两个目标存在明显重叠关系。

    最终返回：

        明显重叠的目标对数量
        -------------------
        所有目标对数量

    注意：
    这只是“从模型预测框推断遮挡”的近似指标，
    不是严格的真实遮挡率。
    """
    n = len(polygons)

    if n < 2:
        return 0.0

    total_pairs = 0
    overlap_pairs = 0

    for i in range(n):

        area_i = polygon_area(
            polygons[i]
        )

        if area_i <= 1:
            continue

        for j in range(
            i + 1,
            n,
        ):

            area_j = polygon_area(
                polygons[j]
            )

            if area_j <= 1:
                continue

            total_pairs += 1

            try:
                intersection_area, _ = (
                    cv2.intersectConvexConvex(
                        np.asarray(
                            polygons[i],
                            np.float32,
                        ),
                        np.asarray(
                            polygons[j],
                            np.float32,
                        ),
                    )
                )

                intersection_area = float(
                    intersection_area
                )

            except Exception:
                intersection_area = 0.0

            overlap = (
                intersection_area
                / max(
                    1.0,
                    min(
                        area_i,
                        area_j,
                    ),
                )
            )

            if overlap >= 0.15:
                overlap_pairs += 1

    if total_pairs == 0:
        return 0.0

    return (
        overlap_pairs
        / total_pairs
    )


def normalize_angle_deg(
    angle_value: float,
) -> float:
    """
    将 OBB 角度转换成便于分析的 0~90°形式。

    Ultralytics OBB 的 xywhr 中，
    angle 在常见版本里通常使用弧度。

    这里同时兼容：
    - 弧度
    - 已经是角度的情况
    """
    angle = float(
        angle_value
    )

    if (
        abs(angle)
        <= math.pi + 1e-6
    ):
        angle = math.degrees(
            angle
        )

    angle = angle % 180.0

    if angle > 90.0:
        angle = (
            180.0 - angle
        )

    return angle


def parse_result(
    result,
    image_shape: Tuple[int, int],
) -> ModelMetrics:
    """
    把 Ultralytics 的单张预测结果转换成 ModelMetrics。

    同时兼容：

    1. YOLO-OBB
        result.obb

    2. 普通目标检测
        result.boxes

    如果使用 YOLO11-OBB，
    程序会优先解析 OBB。
    """
    image_h, image_w = (
        image_shape
    )

    image_area = max(
        1.0,
        float(
            image_h
            * image_w
        ),
    )

    confidences: np.ndarray

    polygons: List[
        np.ndarray
    ] = []

    widths_heights: List[
        Tuple[float, float]
    ] = []

    angles_deg: List[
        float
    ] = []

    # --------------------------------------------------------
    # 6.1 YOLO-OBB
    # --------------------------------------------------------
    if (
        getattr(
            result,
            "obb",
            None,
        )
        is not None
        and result.obb is not None
        and len(result.obb) > 0
    ):
        confidences = (
            result.obb.conf
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )

        # OBB 四个角点。
        try:
            xyxyxyxy = (
                result.obb.xyxyxyxy
                .detach()
                .cpu()
                .numpy()
            )

            polygons = [
                np.asarray(
                    polygon,
                    dtype=np.float32,
                ).reshape(-1, 2)
                for polygon
                in xyxyxyxy
            ]

        except Exception:
            polygons = []

        # 中心 x, y, width, height, rotation
        try:
            xywhr = (
                result.obb.xywhr
                .detach()
                .cpu()
                .numpy()
            )

            for row in xywhr:

                (
                    _cx,
                    _cy,
                    box_w,
                    box_h,
                    angle,
                ) = [
                    float(v)
                    for v
                    in row[:5]
                ]

                widths_heights.append(
                    (
                        box_w,
                        box_h,
                    )
                )

                angles_deg.append(
                    normalize_angle_deg(
                        angle
                    )
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # 6.2 普通 YOLO bbox
    # --------------------------------------------------------
    elif (
        getattr(
            result,
            "boxes",
            None,
        )
        is not None
        and result.boxes is not None
        and len(result.boxes) > 0
    ):
        confidences = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )

        xyxy = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        for (
            x1,
            y1,
            x2,
            y2,
        ) in xyxy:

            polygon = np.array(
                [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ],
                dtype=np.float32,
            )

            polygons.append(
                polygon
            )

            widths_heights.append(
                (
                    float(
                        x2 - x1
                    ),
                    float(
                        y2 - y1
                    ),
                )
            )

            # 普通 bbox 没有旋转角度。
            angles_deg.append(
                0.0
            )

    # --------------------------------------------------------
    # 6.3 没有任何检测
    # --------------------------------------------------------
    else:
        return ModelMetrics()

    number_of_detections = int(
        len(confidences)
    )

    if number_of_detections == 0:
        return ModelMetrics()

    mean_conf = float(
        np.mean(
            confidences
        )
    )

    min_conf = float(
        np.min(
            confidences
        )
    )

    max_conf = float(
        np.max(
            confidences
        )
    )

    uncertain_ratio = float(
        np.mean(
            (
                confidences
                >= UNCERTAIN_LOW
            )
            &
            (
                confidences
                <= UNCERTAIN_HIGH
            )
        )
    )

    very_low_conf_ratio = float(
        np.mean(
            confidences
            < VERY_LOW_CONF
        )
    )

    edge_count = 0
    small_count = 0
    huge_count = 0
    angled_count = 0
    extreme_aspect_count = 0

    for k in range(
        number_of_detections
    ):

        # ----------------------------------------------------
        # 根据 OBB 四边形计算：
        # - 边缘
        # - 小目标
        # - 大目标
        # ----------------------------------------------------
        if k < len(polygons):

            polygon = polygons[k]

            xs = polygon[:, 0]
            ys = polygon[:, 1]

            near_edge = (
                np.min(xs)
                <= EDGE_MARGIN
                * image_w
            ) or (
                np.max(xs)
                >= (
                    1.0
                    - EDGE_MARGIN
                )
                * image_w
            ) or (
                np.min(ys)
                <= EDGE_MARGIN
                * image_h
            ) or (
                np.max(ys)
                >= (
                    1.0
                    - EDGE_MARGIN
                )
                * image_h
            )

            edge_count += int(
                near_edge
            )

            area_ratio = (
                polygon_area(
                    polygon
                )
                / image_area
            )

            small_count += int(
                area_ratio
                <= SMALL_AREA_RATIO
            )

            huge_count += int(
                area_ratio
                >= HUGE_AREA_RATIO
            )

        # ----------------------------------------------------
        # 极端长宽比
        # ----------------------------------------------------
        if (
            k
            < len(
                widths_heights
            )
        ):
            box_w, box_h = (
                widths_heights[k]
            )

            if (
                box_w > 1
                and box_h > 1
            ):
                aspect_ratio = (
                    max(
                        box_w,
                        box_h,
                    )
                    / max(
                        1.0,
                        min(
                            box_w,
                            box_h,
                        ),
                    )
                )

                extreme_aspect_count += int(
                    aspect_ratio
                    >= EXTREME_ASPECT_RATIO
                )

        # ----------------------------------------------------
        # 特殊角度
        # ----------------------------------------------------
        if (
            k
            < len(
                angles_deg
            )
        ):
            angle = angles_deg[k]

            angled_count += int(
                ANGLED_MIN_DEG
                <= angle
                <= ANGLED_MAX_DEG
            )

    return ModelMetrics(
        n_det=number_of_detections,

        mean_conf=mean_conf,
        min_conf=min_conf,
        max_conf=max_conf,

        uncertain_ratio=uncertain_ratio,
        very_low_conf_ratio=very_low_conf_ratio,

        edge_ratio=(
            edge_count
            / number_of_detections
        ),

        small_ratio=(
            small_count
            / number_of_detections
        ),

        huge_ratio=(
            huge_count
            / number_of_detections
        ),

        angled_ratio=(
            angled_count
            / number_of_detections
        ),

        extreme_aspect_ratio=(
            extreme_aspect_count
            / number_of_detections
        ),

        overlap_pair_ratio=(
            overlap_pair_ratio(
                polygons
            )
        ),
    )


# ============================================================
# 7. 简单样本稳定抽样
# ============================================================

def deterministic_keep(
    rel_path: str,
    rate: float,
    seed: int,
) -> bool:
    """
    对简单样本按固定比例保留。

    不直接用 random.random()，
    而是用：

        文件相对路径 + seed

    生成稳定 hash。

    这样只要：
    - 文件名不变
    - seed 不变

    多次运行时，同一张简单图的保留结果通常不变。
    """
    if rate <= 0:
        return False

    if rate >= 1:
        return True

    payload = (
        f"{seed}|{rel_path}"
        .encode(
            "utf-8",
            errors="replace",
        )
    )

    digest = hashlib.sha1(
        payload
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    normalized = (
        value
        / float(
            (1 << 64) - 1
        )
    )

    return normalized < rate


# ============================================================
# 8. 样本价值评分
# ============================================================

def score_sample(
    pre: PreMetrics,
    model_metrics: ModelMetrics,
) -> Tuple[
    str,
    str,
    float,
    List[str],
]:
    """
    根据当前模型和图片属性进行最终分类。

    返回：

        category
        keep_drop
        challenge_score
        reasons

    ----------------------------------------------------------
    最核心思想：
    ----------------------------------------------------------

    “难”不等于“坏”。

    对主动学习来说：

    - 模型已经完全掌握的普通样本
        信息增益低

    - 当前模型犯错、低置信、漏检的样本
        信息增益高

    所以：
        中度困难 -> 加分
        严重不可用 -> 丢弃
    """
    reasons: List[str] = []

    # --------------------------------------------------------
    # 8.1 文件无法读取
    # --------------------------------------------------------
    if not pre.readable:

        return (
            "low_value",
            "DROP",
            -99.0,
            [
                "图片无法读取"
            ],
        )

    # --------------------------------------------------------
    # 8.2 严重质量问题
    # --------------------------------------------------------
    if pre.fatal_quality:

        if (
            pre.blur_var
            < FATAL_BLUR
        ):
            reasons.append(
                "严重模糊"
            )

        if (
            pre.dark_ratio
            >= FATAL_DARK_RATIO
        ):
            reasons.append(
                "严重欠曝"
            )

        if (
            pre.bright_ratio
            >= FATAL_BRIGHT_RATIO
        ):
            reasons.append(
                "严重过曝"
            )

        if not reasons:
            reasons.append(
                "严重图像质量问题"
            )

        return (
            "low_value",
            "DROP",
            -10.0,
            reasons,
        )

    # --------------------------------------------------------
    # 8.3 高重复图片
    # --------------------------------------------------------
    if pre.is_duplicate_drop:

        return (
            "duplicate",
            "DROP",
            -5.0,
            [
                "近重复/高重复样本"
            ],
        )

    # --------------------------------------------------------
    # 8.4 从 0 开始累计困难度
    # --------------------------------------------------------
    score = 0.0

    mm = model_metrics

    # --------------------------------------------------------
    # 8.5 当前模型完全没有检出
    # --------------------------------------------------------
    #
    # 非常重要：
    #
    # n_det == 0 不能自动认为“没有包裹”。
    #
    # 它可能是：
    # 1. 真空场景
    # 2. 当前模型漏检
    # 3. 新包裹外观
    # 4. 极端遮挡
    # 5. 极端光照
    #
    # 因此这类图片默认保留给人工看。
    #
    if mm.n_det == 0:

        score += 3.0

        reasons.append(
            "当前模型未检出：疑似漏检/困难负样本"
        )

    else:

        # ----------------------------------------------------
        # 极低置信
        # ----------------------------------------------------
        if (
            mm.very_low_conf_ratio
            > 0
        ):
            score += (
                2.0
                + 1.0
                * mm.very_low_conf_ratio
            )

            reasons.append(
                "存在极低置信预测"
            )

        # ----------------------------------------------------
        # 不确定预测
        # ----------------------------------------------------
        if (
            mm.uncertain_ratio
            > 0
        ):
            score += (
                1.5
                + 1.0
                * mm.uncertain_ratio
            )

            reasons.append(
                "存在不确定预测"
            )

        # ----------------------------------------------------
        # 边缘目标
        # ----------------------------------------------------
        if (
            mm.edge_ratio
            > 0
        ):
            score += (
                1.2
                + 0.8
                * mm.edge_ratio
            )

            reasons.append(
                "边缘/截断目标"
            )

        # ----------------------------------------------------
        # 小目标
        # ----------------------------------------------------
        if (
            mm.small_ratio
            > 0
        ):
            score += (
                1.2
                + 0.8
                * mm.small_ratio
            )

            reasons.append(
                "小目标"
            )

        # ----------------------------------------------------
        # 超大目标
        # ----------------------------------------------------
        if (
            mm.huge_ratio
            > 0
        ):
            score += (
                0.8
                + 0.5
                * mm.huge_ratio
            )

            reasons.append(
                "超大近景目标"
            )

        # ----------------------------------------------------
        # 遮挡 / 重叠 / 贴靠
        # ----------------------------------------------------
        if (
            mm.overlap_pair_ratio
            > 0
        ):
            score += (
                1.5
                + 1.0
                * mm.overlap_pair_ratio
            )

            reasons.append(
                "遮挡/贴靠/重叠"
            )

        # ----------------------------------------------------
        # 特殊角度
        # ----------------------------------------------------
        if (
            mm.angled_ratio
            > 0
        ):
            score += (
                0.8
                + 0.5
                * mm.angled_ratio
            )

            reasons.append(
                "特殊角度"
            )

        # ----------------------------------------------------
        # 极端长宽比
        # ----------------------------------------------------
        if (
            mm.extreme_aspect_ratio
            > 0
        ):
            score += (
                0.7
                + 0.5
                * mm.extreme_aspect_ratio
            )

            reasons.append(
                "极端长宽比"
            )

        # ----------------------------------------------------
        # 一张图中包裹很多
        # ----------------------------------------------------
        #
        # 对你这种流水线包裹检测，
        # 5 个及以上目标通常比单包裹更值得保留。
        #
        if mm.n_det >= 5:

            score += min(
                1.5,
                0.25
                * (
                    mm.n_det - 4
                ),
            )

            reasons.append(
                "多包裹拥挤场景"
            )

    # --------------------------------------------------------
    # 8.6 中度模糊
    # --------------------------------------------------------
    #
    # 注意：
    # 中度模糊不是垃圾图。
    #
    # 真实生产线中运动模糊是需要模型学会处理的。
    #
    if pre.moderate_blur:

        score += 1.4

        reasons.append(
            "中度运动模糊"
        )

    # --------------------------------------------------------
    # 8.7 中度强光 / 阴影
    # --------------------------------------------------------
    if pre.moderate_lighting:

        score += 1.2

        reasons.append(
            "强光/阴影/曝光异常"
        )

    # --------------------------------------------------------
    # 8.8 高价值难例
    # --------------------------------------------------------
    if (
        score
        >= HIGH_VALUE_THRESHOLD
    ):

        return (
            "high_value",
            "KEEP",
            score,
            reasons,
        )

    # --------------------------------------------------------
    # 8.9 一般有价值样本
    # --------------------------------------------------------
    if (
        score
        >= VALUABLE_THRESHOLD
    ):

        return (
            "valuable",
            "KEEP",
            score,
            reasons,
        )

    # --------------------------------------------------------
    # 8.10 判断是否为“模型已经掌握的简单样本”
    # --------------------------------------------------------
    easy_sample = (
        mm.n_det > 0

        and mm.mean_conf
        >= EASY_CONF

        and mm.uncertain_ratio
        == 0

        and mm.very_low_conf_ratio
        == 0

        and mm.edge_ratio
        == 0

        and mm.small_ratio
        == 0

        and mm.huge_ratio
        == 0

        and mm.overlap_pair_ratio
        == 0

        and not pre.moderate_blur

        and not pre.moderate_lighting
    )

    if easy_sample:

        # ----------------------------------------------------
        # 简单样本也保留一小部分，
        # 维持正常生产数据分布。
        # ----------------------------------------------------
        if deterministic_keep(
            pre.rel_path,
            EASY_KEEP_RATE,
            RANDOM_SEED,
        ):

            reasons.append(
                "常规简单样本抽样保留"
            )

            return (
                "valuable",
                "KEEP",
                score,
                reasons,
            )

        reasons.append(
            "当前模型已经稳定掌握的简单样本"
        )

        return (
            "low_value",
            "DROP",
            score,
            reasons,
        )

    # --------------------------------------------------------
    # 8.11 兜底逻辑
    # --------------------------------------------------------
    #
    # 如果一张图片：
    # - 没有达到高价值阈值
    # - 但又不能明确认为是简单样本
    #
    # 默认保留。
    #
    # 对数据集筛选来说：
    # “宁可多留一些待人工复核，也不要错误删除潜在难例”
    # 更安全。
    #
    if not reasons:
        reasons.append(
            "普通候选样本"
        )

    return (
        "valuable",
        "KEEP",
        score,
        reasons,
    )


# ============================================================
# 9. 文件复制
# ============================================================

def safe_transfer(
    source: Path,
    destination: Path,
) -> None:
    """
    把图片复制到筛选目录。

    COPY_MODE = "copy"
        shutil.copy2
        安全、直观。

    COPY_MODE = "hardlink"
        尝试建立硬链接。
        如果失败，自动退回 copy。

    本函数不会删除 source。
    """
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 如果目标文件已经存在，
    # 默认不重复复制。
    if destination.exists():
        return

    if COPY_MODE == "copy":

        shutil.copy2(
            source,
            destination,
        )

        return

    if COPY_MODE == "hardlink":

        try:
            os.link(
                source,
                destination,
            )

        except Exception:

            shutil.copy2(
                source,
                destination,
            )

        return

    raise ValueError(
        f"未知 COPY_MODE：{COPY_MODE}"
    )


def build_output_path(
    output_root: Path,
    keep_drop: str,
    category: str,
    relative_path: Path,
) -> Path:
    """
    根据分类生成目标路径。

    例如：

    原图：
        raw/camera1/day1/a.jpg

    如果是 high_value：

        output/
        KEEP/
        high_value/
        camera1/
        day1/
        a.jpg

    保留原来的相对目录结构，
    避免不同文件夹里同名图片互相覆盖。
    """
    if keep_drop == "KEEP":

        return (
            output_root
            / "KEEP"
            / category
            / relative_path
        )

    return (
        output_root
        / "DROP"
        / category
        / relative_path
    )


# ============================================================
# 10. 收集图片
# ============================================================

def collect_images(
    input_root: Path,
) -> List[Path]:
    """
    递归扫描 input_root 下面所有图片。
    """
    image_paths = [
        path
        for path
        in input_root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    image_paths.sort(
        key=lambda p: str(
            p
        ).lower()
    )

    return image_paths


def chunked(
    sequence: Sequence[Path],
    chunk_size: int,
) -> Iterable[Sequence[Path]]:
    """
    把图片列表分成一个个 batch。
    """
    for start in range(
        0,
        len(sequence),
        chunk_size,
    ):
        yield sequence[
            start:
            start + chunk_size
        ]


# ============================================================
# 11. 主程序
# ============================================================

def main() -> None:
    """
    完整流程：

    Step 1
        扫描所有原始图片

    Step 2
        图像质量分析
        + pHash
        + HSV histogram

    Step 3
        近重复聚类

    Step 4
        重复组中保留少量代表图

    Step 5
        对剩余候选图片运行当前 YOLO / YOLO-OBB

    Step 6
        根据模型不确定性 + 场景难度计算样本价值

    Step 7
        分流到 KEEP / DROP

    Step 8
        输出 CSV 和 JSON 报告
    """

    # --------------------------------------------------------
    # 11.1 将配置区字符串转换成 Path
    # --------------------------------------------------------
    input_root = Path(
        INPUT_DIR
    ).expanduser().resolve()

    model_path = Path(
        MODEL_PATH
    ).expanduser().resolve()

    output_root = Path(
        OUTPUT_DIR
    ).expanduser().resolve()

    # --------------------------------------------------------
    # 11.2 基本路径检查
    # --------------------------------------------------------
    if not input_root.exists():

        raise SystemExit(
            "\n输入目录不存在：\n"
            f"{input_root}\n\n"
            "请修改代码顶部 INPUT_DIR。\n"
        )

    if not model_path.exists():

        raise SystemExit(
            "\n模型文件不存在：\n"
            f"{model_path}\n\n"
            "请修改代码顶部 MODEL_PATH。\n"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir = (
        output_root
        / "reports"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # 11.3 收集全部图片
    # --------------------------------------------------------
    image_paths = collect_images(
        input_root
    )

    # 首次试跑时只处理一部分。
    if MAX_FILES > 0:

        image_paths = image_paths[
            :MAX_FILES
        ]

    if not image_paths:

        raise SystemExit(
            "\n输入目录中没有找到支持的图片。\n"
        )

    print(
        "=" * 72
    )

    print(
        "包裹数据集半自动筛选"
    )

    print(
        "=" * 72
    )

    print(
        f"输入目录：{input_root}"
    )

    print(
        f"当前模型：{model_path}"
    )

    print(
        f"输出目录：{output_root}"
    )

    print(
        f"本次图片数量：{len(image_paths)}"
    )

    print(
        f"MAX_FILES：{MAX_FILES}"
    )

    print(
        f"DEVICE：{DEVICE}"
    )

    print()

    # ========================================================
    # Step A
    # 图像质量 + pHash + HSV
    # ========================================================

    pre_metrics_list: List[
        PreMetrics
    ] = []

    histograms: List[
        Optional[np.ndarray]
    ] = []

    for path in tqdm(
        image_paths,
        desc="Step A 图像预分析",
        unit="img",
    ):
        relative_path = (
            path.relative_to(
                input_root
            )
        )

        image = imread_unicode(
            path
        )

        # ----------------------------------------------------
        # 图片损坏 / 无法读取
        # ----------------------------------------------------
        if image is None:

            pre_metrics_list.append(
                PreMetrics(
                    path=str(path),
                    rel_path=str(
                        relative_path
                    ),
                    readable=False,
                    fatal_quality=True,
                )
            )

            histograms.append(
                None
            )

            continue

        image_h, image_w = (
            image.shape[:2]
        )

        quality, histogram = (
            calc_quality(
                image
            )
        )

        pre_metrics_list.append(
            PreMetrics(
                path=str(path),
                rel_path=str(
                    relative_path
                ),

                readable=True,

                width=image_w,
                height=image_h,

                blur_var=quality[
                    "blur_var"
                ],

                brightness=quality[
                    "brightness"
                ],

                dark_ratio=quality[
                    "dark_ratio"
                ],

                bright_ratio=quality[
                    "bright_ratio"
                ],

                phash=compute_phash64(
                    image
                ),

                quality_score=quality[
                    "quality_score"
                ],

                fatal_quality=quality[
                    "fatal_quality"
                ],

                moderate_blur=quality[
                    "moderate_blur"
                ],

                moderate_lighting=quality[
                    "moderate_lighting"
                ],
            )
        )

        histograms.append(
            histogram
        )

    # ========================================================
    # Step B
    # 近重复检测
    # ========================================================

    duplicate_groups = (
        find_duplicate_groups(
            pre_metrics_list,
            histograms,
        )
    )

    mark_duplicate_drops(
        pre_metrics_list,
        duplicate_groups,
    )

    duplicate_drop_count = sum(
        item.is_duplicate_drop
        for item
        in pre_metrics_list
    )

    print()

    print(
        f"近重复组数：{len(duplicate_groups)}"
    )

    print(
        f"因近重复准备 DROP：{duplicate_drop_count}"
    )

    # ========================================================
    # Step C
    # 只对真正值得分析的图片做 YOLO 推理
    # ========================================================
    #
    # 已经明确：
    # - 严重坏图
    # - 重复冗余图
    #
    # 没必要再浪费 GPU 推理时间。
    # ========================================================

    inference_paths: List[
        Path
    ] = []

    pre_index_by_absolute_path: Dict[
        str,
        int,
    ] = {}

    for (
        index,
        item,
    ) in enumerate(
        pre_metrics_list
    ):

        absolute_key = str(
            Path(
                item.path
            ).resolve()
        )

        pre_index_by_absolute_path[
            absolute_key
        ] = index

        if (
            item.readable
            and not item.is_duplicate_drop
            and not item.fatal_quality
        ):
            inference_paths.append(
                Path(
                    item.path
                )
            )

    model_metrics_list: List[
        ModelMetrics
    ] = [
        ModelMetrics()
        for _ in pre_metrics_list
    ]

    print()

    print(
        "需要运行当前模型的图片："
        f"{len(inference_paths)} / "
        f"{len(pre_metrics_list)}"
    )

    # --------------------------------------------------------
    # 加载当前模型
    # --------------------------------------------------------
    model = YOLO(
        str(
            model_path
        )
    )

    predict_kwargs = {
        "imgsz": IMGSZ,
        "conf": INFER_CONF,
        "iou": INFER_IOU,
        "verbose": False,
        "batch": BATCH_SIZE,
    }

    if DEVICE is not None:

        predict_kwargs[
            "device"
        ] = DEVICE

    batches = list(
        chunked(
            inference_paths,
            max(
                1,
                BATCH_SIZE,
            ),
        )
    )

    for batch_paths in tqdm(
        batches,
        desc="Step C 当前模型推理",
        unit="batch",
    ):
        sources = [
            str(path)
            for path
            in batch_paths
        ]

        try:
            results = model.predict(
                source=sources,
                **predict_kwargs,
            )

        except TypeError:
            # 某些较旧的 Ultralytics 版本，
            # 对 list source + batch 参数处理可能不同。
            #
            # 如果失败，就去掉 batch 参数再试一次。
            fallback_kwargs = dict(
                predict_kwargs
            )

            fallback_kwargs.pop(
                "batch",
                None,
            )

            results = model.predict(
                source=sources,
                **fallback_kwargs,
            )

        for result in results:

            result_path = Path(
                str(
                    result.path
                )
            ).resolve()

            index = (
                pre_index_by_absolute_path
                .get(
                    str(
                        result_path
                    )
                )
            )

            # ------------------------------------------------
            # 极少数 Ultralytics 版本的 result.path
            # 可能和输入路径表示方式不同。
            #
            # 这里提供一个 basename 兜底匹配。
            # ------------------------------------------------
            if index is None:

                matches = [
                    i
                    for i, item
                    in enumerate(
                        pre_metrics_list
                    )
                    if (
                        Path(
                            item.path
                        ).name
                        == result_path.name
                    )
                ]

                if len(matches) == 1:
                    index = matches[0]

            if index is None:
                continue

            pre_item = (
                pre_metrics_list[
                    index
                ]
            )

            model_metrics_list[
                index
            ] = parse_result(
                result,
                image_shape=(
                    pre_item.height,
                    pre_item.width,
                ),
            )

    # ========================================================
    # Step D
    # 最终分类
    # ========================================================

    final_records: List[
        FinalRecord
    ] = []

    category_counts = defaultdict(
        int
    )

    for index, pre_item in enumerate(
        tqdm(
            pre_metrics_list,
            desc="Step D 分类与保存",
            unit="img",
        )
    ):
        model_item = (
            model_metrics_list[
                index
            ]
        )

        (
            category,
            keep_drop,
            challenge_score,
            reasons,
        ) = score_sample(
            pre_item,
            model_item,
        )

        relative_path = Path(
            pre_item.rel_path
        )

        destination = (
            build_output_path(
                output_root,
                keep_drop,
                category,
                relative_path,
            )
        )

        # ----------------------------------------------------
        # 只有 DRY_RUN=False 时才真正复制图片。
        # ----------------------------------------------------
        if not DRY_RUN:

            try:
                safe_transfer(
                    Path(
                        pre_item.path
                    ),
                    destination,
                )

            except Exception as exc:

                reasons.append(
                    "复制失败："
                    f"{type(exc).__name__}"
                )

        category_counts[
            f"{keep_drop}/{category}"
        ] += 1

        final_records.append(
            FinalRecord(
                path=pre_item.path,
                rel_path=pre_item.rel_path,

                category=category,
                keep_drop=keep_drop,

                reasons=(
                    " | ".join(
                        reasons
                    )
                ),

                duplicate_group=(
                    pre_item.duplicate_group
                ),

                duplicate_rank=(
                    pre_item.duplicate_rank
                ),

                width=pre_item.width,
                height=pre_item.height,

                blur_var=round(
                    pre_item.blur_var,
                    4,
                ),

                brightness=round(
                    pre_item.brightness,
                    4,
                ),

                dark_ratio=round(
                    pre_item.dark_ratio,
                    6,
                ),

                bright_ratio=round(
                    pre_item.bright_ratio,
                    6,
                ),

                quality_score=round(
                    pre_item.quality_score,
                    6,
                ),

                n_det=model_item.n_det,

                mean_conf=round(
                    model_item.mean_conf,
                    6,
                ),

                min_conf=round(
                    model_item.min_conf,
                    6,
                ),

                max_conf=round(
                    model_item.max_conf,
                    6,
                ),

                uncertain_ratio=round(
                    model_item.uncertain_ratio,
                    6,
                ),

                very_low_conf_ratio=round(
                    model_item.very_low_conf_ratio,
                    6,
                ),

                edge_ratio=round(
                    model_item.edge_ratio,
                    6,
                ),

                small_ratio=round(
                    model_item.small_ratio,
                    6,
                ),

                huge_ratio=round(
                    model_item.huge_ratio,
                    6,
                ),

                angled_ratio=round(
                    model_item.angled_ratio,
                    6,
                ),

                extreme_aspect_ratio=round(
                    model_item.extreme_aspect_ratio,
                    6,
                ),

                overlap_pair_ratio=round(
                    model_item.overlap_pair_ratio,
                    6,
                ),

                challenge_score=round(
                    challenge_score,
                    6,
                ),

                output_path=str(
                    destination
                ),
            )
        )

    # ========================================================
    # Step E
    # 输出 CSV
    # ========================================================

    csv_path = (
        report_dir
        / "screening_report.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:

        fieldnames = list(
            asdict(
                final_records[0]
            ).keys()
        )

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in final_records:

            writer.writerow(
                asdict(
                    record
                )
            )

    # ========================================================
    # Step F
    # 输出 summary.json
    # ========================================================

    summary = {
        "input": str(
            input_root
        ),

        "model": str(
            model_path
        ),

        "output": str(
            output_root
        ),

        "total_images": len(
            pre_metrics_list
        ),

        "duplicate_groups": len(
            duplicate_groups
        ),

        "counts": dict(
            sorted(
                category_counts.items()
            )
        ),

        "parameters": {
            "MAX_FILES": MAX_FILES,
            "DEVICE": DEVICE,
            "IMGSZ": IMGSZ,
            "BATCH_SIZE": BATCH_SIZE,

            "INFER_CONF": INFER_CONF,
            "INFER_IOU": INFER_IOU,

            "UNCERTAIN_LOW": UNCERTAIN_LOW,
            "UNCERTAIN_HIGH": UNCERTAIN_HIGH,
            "VERY_LOW_CONF": VERY_LOW_CONF,
            "EASY_CONF": EASY_CONF,
            "EASY_KEEP_RATE": EASY_KEEP_RATE,

            "EDGE_MARGIN": EDGE_MARGIN,
            "SMALL_AREA_RATIO": SMALL_AREA_RATIO,
            "HUGE_AREA_RATIO": HUGE_AREA_RATIO,

            "DUP_PHASH_HAMMING": DUP_PHASH_HAMMING,
            "DUP_HIST_CORR": DUP_HIST_CORR,
            "MAX_KEEP_PER_DUP_GROUP": (
                MAX_KEEP_PER_DUP_GROUP
            ),

            "HIGH_VALUE_THRESHOLD": (
                HIGH_VALUE_THRESHOLD
            ),

            "VALUABLE_THRESHOLD": (
                VALUABLE_THRESHOLD
            ),
        },
    }

    json_path = (
        report_dir
        / "summary.json"
    )

    json_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Step G
    # 控制台打印总结
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "筛选完成"
    )

    print(
        "=" * 72
    )

    for (
        key,
        value,
    ) in sorted(
        category_counts.items()
    ):

        print(
            f"{key:24s}: {value}"
        )

    print()

    print(
        f"详细 CSV：{csv_path}"
    )

    print(
        f"汇总 JSON：{json_path}"
    )

    print()

    print(
        "建议人工处理顺序："
    )

    print(
        "1. KEEP/high_value"
        " -> 优先人工复核与标注"
    )

    print(
        "2. KEEP/valuable"
        " -> 常规复核后进入训练候选池"
    )

    print(
        "3. DROP/duplicate"
        " -> 一般无需标注，可抽查"
    )

    print(
        "4. DROP/low_value"
        " -> 一般归档，不进入本轮训练"
    )

    print()

    print(
        "注意："
    )

    print(
        "本程序是主动学习式的“候选筛选器”，"
        "不是完全替代人工判断。"
    )


# ============================================================
# 12. Python 文件直接运行入口
# ============================================================
#
# 在 Spyder / PyCharm / VS Code 里点击运行，
# 就会从这里进入 main()。
#
# 不需要命令行参数。
# ============================================================

if __name__ == "__main__":
    main()
