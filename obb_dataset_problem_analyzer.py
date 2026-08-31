# -*- coding: utf-8 -*-
"""
YOLO11-OBB 本地验证集问题样本分析脚本
适配“多个 CVAT 数据集 + val.txt 绝对路径列表”的数据组织方式。

你的数据结构可类似：
H:/train_data/
├─ images_cvat00_dataset/
│  ├─ images/train/...
│  └─ labels/train/...
├─ images_cvat01_dataset/
│  ├─ images/train/...
│  └─ labels/train/...
├─ images_cvat02_dataset/
│  └─ ...
└─ output/
   └─ val.txt

val.txt 中每一行是一张验证图片的绝对路径。

脚本会：
1. 读取 val.txt；
2. 用 best.pt 对这些验证图片推理；
3. 自动根据图片路径寻找对应 OBB 标签：
      .../images/train/xxx.jpg
   -> .../labels/train/xxx.txt
4. 只保存有问题的图片；
5. 在图片里直接标出有问题的 GT / FP；
6. 文件名直接写问题类型；
7. 输出 CSV 和中文说明，方便你决定下一轮数据集补什么。

问题类型：
FN         = 漏检
LOW_CONF   = 低置信度 / 潜在漏检
LOW_IOU    = OBB 定位质量偏低
ANGLE_ERR  = 角度误差过大
FP         = 误检
"""

import csv
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# 1. 用户参数 —— 一般只需要修改这里
# =============================================================================

# -----------------------------
# 模型
# -----------------------------
MODEL_PATH = Path(r"H:\obb_test\models\best-082704n.pt")

# -----------------------------
# 验证集列表
# 直接使用你划分脚本生成的 val.txt
# -----------------------------
VAL_TXT_PATH = Path(r"H:\train_data\val_grouped.txt")

# -----------------------------
# 输出目录
# 所有问题图片最终会放在：
# OUTPUT_DIR / RUN_NAME / problem_images
# -----------------------------
OUTPUT_DIR = Path(r"H:\train_data\obb_error_output")

# 本次分析名称
# 换模型时建议改，例如：
# "8.27.0-n_best"
# "8.30.0-s_best"
RUN_NAME = "082704n"


# =============================================================================
# 2. 推理参数
# =============================================================================

# 0 = 第一张 NVIDIA GPU
# "cpu" = CPU
DEVICE = "cpu"

# 与训练时一致即可
IMGSZ = 960

# 本地显存不够可改为 8 / 4
PRED_BATCH = 8

# 推理时故意把最低 confidence 设得很低，
# 用于发现“模型其实看到了，但 confidence 不够”的目标。
INFER_CONF = 0.01

# NMS IoU
NMS_IOU = 0.30

# 正式判断“检测成功 / 漏检 / 误检”的 confidence
# 如果部署时准备使用 0.25，可以保持这里 0.25。
EVAL_CONF = 0.25


# =============================================================================
# 3. 问题判定阈值
# =============================================================================

# GT 和预测框 IoU >= 此值才算成功匹配
MATCH_IOU = 0.3

# 已经有候选预测，但 IoU 低于这个值，标记 LOW_IOU
LOW_IOU_THRESH = 0.8

# LOW_IOU 至少要求存在一个有一定重合的候选框
# 避免完全无关的预测也被标记为 LOW_IOU
LOW_IOU_MIN_CANDIDATE = 0.10

# 已经匹配到，但 confidence < 此值，标记 LOW_CONF
LOW_CONF_THRESH = 0.8

# 角度误差阈值，单位：度
ANGLE_ERROR_THRESH_DEG = 10.0

# 长宽比太接近 1 时，矩形方向本身不稳定，
# 默认不把它判定为 ANGLE_ERR
ANGLE_MIN_ASPECT_RATIO = 1.10

# 单类别 parcel 建议保持 True
MATCH_SAME_CLASS = True


# =============================================================================
# 4. 输出设置
# =============================================================================

# 绘制线宽
BOX_THICKNESS = 3

# 顶部是否显示该图片的问题统计
DRAW_INFO_PANEL = True

# 是否另外保存一份没有画框的原图
# 一般没必要，False 即可
SAVE_ORIGINAL_PROBLEM_IMAGE = False

# -----------------------------------------------------------------------------
# 运行进度显示
# -----------------------------------------------------------------------------

# 是否显示实时进度条
SHOW_PROGRESS = True

# 是否单独显示“模型推理了多少张”
# True 时会按 batch 明确显示：
#   正在推理 1~2 / 275
#   推理完成 2 / 275
SHOW_INFERENCE_PROGRESS = True

# 每处理多少张图片刷新一次进度
# 本地 CPU 推理建议设为 1，这样每张图都会刷新
PROGRESS_UPDATE_EVERY = 1

# 终端进度条宽度
PROGRESS_BAR_WIDTH = 32

# 是否在进度条中显示当前图片文件名
SHOW_CURRENT_FILENAME = True


# =============================================================================
# 5. 问题说明
# =============================================================================

ISSUE_NAME_CN = {
    "FN": "漏检",
    "LOW_CONF": "低置信度",
    "LOW_IOU": "OBB定位偏差",
    "ANGLE_ERR": "角度误差",
    "FP": "误检",
}

ISSUE_ADVICE = {
    "FN": (
        "漏检：优先级最高。人工查看该目标是否属于遮挡、贴靠、边缘、反光、"
        "软袋/异形、小目标、运动模糊等困难情况；下一轮补充同类真实困难样本。"
    ),
    "LOW_CONF": (
        "低置信度：模型已经看到目标但不够确定，属于潜在漏检。"
        "增加与该包裹外观、尺度、姿态、遮挡程度和背景相似的真实样本。"
    ),
    "LOW_IOU": (
        "OBB定位偏差：先检查 GT 框是否贴合目标、四点和标注规则是否一致；"
        "若标注正确，再补充相似形状、姿态、贴靠和遮挡样本。"
    ),
    "ANGLE_ERR": (
        "角度误差：先检查 GT 角度标注是否一致；若 GT 正确，"
        "增加相似旋转角度、长宽比、边界不清晰及遮挡目标。"
    ),
    "FP": (
        "误检：检查该区域是否存在漏标；如果确实是背景/设备，"
        "则把这种外观作为 hard negative 补充到数据集中。"
    ),
}


# =============================================================================
# 6. 数据路径工具
# =============================================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_val_list():
    """读取 val.txt 中的绝对路径。"""
    if not VAL_TXT_PATH.exists():
        raise FileNotFoundError(f"val.txt 不存在：{VAL_TXT_PATH}")

    image_paths = []

    with VAL_TXT_PATH.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip().strip('"').strip("'")

            if not text:
                continue

            p = Path(text)

            if not p.exists():
                print(f"[警告] val.txt 第 {line_no} 行图片不存在：{p}")
                continue

            image_paths.append(p)

    if not image_paths:
        raise RuntimeError(f"没有从 val.txt 读取到有效图片：{VAL_TXT_PATH}")

    return image_paths


def image_to_label_path(image_path: Path) -> Path:
    """
    根据你的 CVAT 数据集结构自动找标签：

    H:/train_data/images_cvat00_dataset/images/train/a/b/001.jpg
                           ↓
    H:/train_data/images_cvat00_dataset/labels/train/a/b/001.txt

    不要求提前知道是 cvat00 / cvat01 / cvat02 ...
    """
    parts = list(image_path.parts)

    # 从后往前找 images，防止前面目录名里也包含 images 字样
    images_index = None

    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            images_index = i
            break

    if images_index is None:
        raise ValueError(
            f"无法根据图片路径推断标签路径，因为路径中没有独立的 'images' 目录：\n"
            f"{image_path}"
        )

    parts[images_index] = "labels"

    label_path = Path(*parts).with_suffix(".txt")
    return label_path


def safe_output_name(image_path: Path, issue_tags):
    """
    例：
    FN_LOWCONF__images_cvat00_dataset__images__train__000123.jpg

    即使不同 CVAT 数据集存在同名图片，也不会互相覆盖。
    """
    ordered = ["FN", "LOW_CONF", "LOW_IOU", "ANGLE_ERR", "FP"]

    prefix_map = {
        "FN": "FN",
        "LOW_CONF": "LOWCONF",
        "LOW_IOU": "LOWIOU",
        "ANGLE_ERR": "ANGLE",
        "FP": "FP",
    }

    present = [
        prefix_map[tag]
        for tag in ordered
        if tag in issue_tags
    ]

    prefix = "_".join(present) if present else "OK"

    # 尽量保留最后几层目录，避免文件名过长
    parts = image_path.parts

    if len(parts) >= 4:
        tail = parts[-4:]
    else:
        tail = parts

    rel_name = "__".join(tail)

    return f"{prefix}__{rel_name}"


# =============================================================================
# 7. OBB 读取与几何计算
# =============================================================================

def read_gt_obb(label_path: Path, image_width: int, image_height: int):
    """
    YOLO OBB：
        class x1 y1 x2 y2 x3 y3 x4 y4

    坐标默认是 0~1 归一化。
    缺少标签文件时按“该图没有 GT 目标”处理，
    这与你划分脚本 REQUIRE_LABEL=False 的逻辑兼容。
    """
    gts = []

    if not label_path.exists():
        return gts

    with label_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 9:
                raise ValueError(
                    f"OBB 标签格式错误：{label_path}\n"
                    f"第 {line_no} 行共有 {len(parts)} 个字段，"
                    f"应为 9：class x1 y1 x2 y2 x3 y3 x4 y4"
                )

            cls_id = int(float(parts[0]))

            poly = np.array(
                [float(v) for v in parts[1:]],
                dtype=np.float32
            ).reshape(4, 2)

            poly[:, 0] *= image_width
            poly[:, 1] *= image_height

            gts.append({
                "cls": cls_id,
                "poly": poly.astype(np.float32),
            })

    return gts


def polygon_area(poly):
    return float(abs(cv2.contourArea(poly.astype(np.float32))))


def polygon_iou(poly1, poly2):
    p1 = poly1.astype(np.float32)
    p2 = poly2.astype(np.float32)

    a1 = polygon_area(p1)
    a2 = polygon_area(p2)

    if a1 <= 0 or a2 <= 0:
        return 0.0

    inter_area, _ = cv2.intersectConvexConvex(p1, p2)
    inter_area = max(float(inter_area), 0.0)

    union = a1 + a2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def long_axis_angle_and_ratio(poly):
    """
    长边方向，范围 [0, 180)。
    """
    p = poly.astype(np.float64)

    edges = [
        p[1] - p[0],
        p[2] - p[1],
        p[3] - p[2],
        p[0] - p[3],
    ]

    lengths = [np.linalg.norm(v) for v in edges]

    if min(lengths) <= 1e-9:
        return None, None

    long_idx = int(np.argmax(lengths))
    long_vec = edges[long_idx]
    long_len = lengths[long_idx]

    short_len = min(
        lengths[(long_idx - 1) % 4],
        lengths[(long_idx + 1) % 4]
    )

    angle = math.degrees(
        math.atan2(long_vec[1], long_vec[0])
    ) % 180.0

    ratio = long_len / max(short_len, 1e-9)

    return angle, ratio


def angle_diff_180(a, b):
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def greedy_match(gts, preds, iou_threshold):
    candidates = []

    for gi, gt in enumerate(gts):
        for pi, pred in enumerate(preds):

            if MATCH_SAME_CLASS and gt["cls"] != pred["cls"]:
                continue

            iou = polygon_iou(
                gt["poly"],
                pred["poly"]
            )

            if iou >= iou_threshold:
                candidates.append(
                    (iou, gi, pi)
                )

    candidates.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    used_gt = set()
    used_pred = set()
    matches = []

    for iou, gi, pi in candidates:

        if gi in used_gt or pi in used_pred:
            continue

        used_gt.add(gi)
        used_pred.add(pi)

        matches.append(
            (gi, pi, iou)
        )

    unmatched_gt = [
        i for i in range(len(gts))
        if i not in used_gt
    ]

    unmatched_pred = [
        i for i in range(len(preds))
        if i not in used_pred
    ]

    return matches, unmatched_gt, unmatched_pred


# =============================================================================
# 8. 绘图
# =============================================================================

def draw_poly(img, poly, color, thickness):
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)

    cv2.polylines(
        img,
        [pts],
        True,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_text(img, text, x, y, color, scale=0.52, thickness=2):
    x = max(2, int(x))
    y = max(20, int(y))

    (tw, th), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness
    )

    cv2.rectangle(
        img,
        (x - 2, y - th - 5),
        (x + tw + 4, y + baseline + 3),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_panel(img, counts):
    if not DRAW_INFO_PANEL:
        return img

    parts = []

    for tag in ["FN", "LOW_CONF", "LOW_IOU", "ANGLE_ERR", "FP"]:
        if counts[tag] > 0:
            parts.append(
                f"{tag}={counts[tag]}"
            )

    text = "PROBLEM | " + " | ".join(parts)

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (img.shape[1], 42),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.82,
        img,
        0.18,
        0,
        img
    )

    cv2.putText(
        img,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return img


# =============================================================================
# 9. 单张图片分析
# =============================================================================

def analyze_image(image_path, result):
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"无法读取图片：{image_path}"
        )

    h, w = image.shape[:2]

    label_path = image_to_label_path(
        image_path
    )

    gts = read_gt_obb(
        label_path,
        w,
        h
    )

    # 所有低阈值预测
    all_preds = []

    if result.obb is not None and len(result.obb) > 0:

        polys = (
            result.obb.xyxyxyxy
            .detach()
            .cpu()
            .numpy()
        )

        confs = (
            result.obb.conf
            .detach()
            .cpu()
            .numpy()
        )

        classes = (
            result.obb.cls
            .detach()
            .cpu()
            .numpy()
        )

        for poly, conf, cls_id in zip(
            polys,
            confs,
            classes
        ):
            all_preds.append({
                "cls": int(cls_id),
                "conf": float(conf),
                "poly": np.asarray(
                    poly,
                    dtype=np.float32
                ).reshape(4, 2),
            })

    # 正式评估阈值
    eval_preds = [
        p for p in all_preds
        if p["conf"] >= EVAL_CONF
    ]

    # 正式匹配：FN / FP
    eval_matches, fn_indices, fp_indices = greedy_match(
        gts,
        eval_preds,
        MATCH_IOU
    )

    # 低 conf 诊断匹配
    diagnostic_matches, _, _ = greedy_match(
        gts,
        all_preds,
        MATCH_IOU
    )

    diag_map = {
        gi: (pi, iou)
        for gi, pi, iou in diagnostic_matches
    }

    gt_issues = {
        gi: []
        for gi in range(len(gts))
    }

    gt_extra = {}

    # 每一个 GT 分析
    for gi, gt in enumerate(gts):

        best_pi = None
        best_iou = 0.0

        for pi, pred in enumerate(all_preds):

            if MATCH_SAME_CLASS and gt["cls"] != pred["cls"]:
                continue

            iou = polygon_iou(
                gt["poly"],
                pred["poly"]
            )

            if iou > best_iou:
                best_iou = iou
                best_pi = pi

        extra = {
            "best_iou": best_iou,
            "confidence": None,
            "angle_error": None,
            "gt_angle": None,
            "pred_angle": None,
            "aspect_ratio": None,
        }

        # FN
        if gi in fn_indices:
            gt_issues[gi].append("FN")

        # LOW_IOU
        if (
            best_pi is not None
            and LOW_IOU_MIN_CANDIDATE <= best_iou < LOW_IOU_THRESH
        ):
            gt_issues[gi].append("LOW_IOU")

        # LOW_CONF + ANGLE_ERR
        if gi in diag_map:

            pi, matched_iou = diag_map[gi]
            pred = all_preds[pi]

            extra["confidence"] = pred["conf"]

            if pred["conf"] < LOW_CONF_THRESH:
                gt_issues[gi].append(
                    "LOW_CONF"
                )

            gt_angle, ratio = long_axis_angle_and_ratio(
                gt["poly"]
            )

            pred_angle, _ = long_axis_angle_and_ratio(
                pred["poly"]
            )

            extra["gt_angle"] = gt_angle
            extra["pred_angle"] = pred_angle
            extra["aspect_ratio"] = ratio

            if (
                gt_angle is not None
                and pred_angle is not None
                and ratio is not None
                and ratio >= ANGLE_MIN_ASPECT_RATIO
            ):
                angle_error = angle_diff_180(
                    gt_angle,
                    pred_angle
                )

                extra["angle_error"] = angle_error

                if angle_error > ANGLE_ERROR_THRESH_DEG:
                    gt_issues[gi].append(
                        "ANGLE_ERR"
                    )

        gt_extra[gi] = extra

    counts = {
        "FN": sum(
            "FN" in issues
            for issues in gt_issues.values()
        ),
        "LOW_CONF": sum(
            "LOW_CONF" in issues
            for issues in gt_issues.values()
        ),
        "LOW_IOU": sum(
            "LOW_IOU" in issues
            for issues in gt_issues.values()
        ),
        "ANGLE_ERR": sum(
            "ANGLE_ERR" in issues
            for issues in gt_issues.values()
        ),
        "FP": len(fp_indices),
    }

    issue_tags = {
        tag
        for tag, n in counts.items()
        if n > 0
    }

    is_problem = bool(issue_tags)

    # -------------------------------------------------------------------------
    # 绘图
    # -------------------------------------------------------------------------

    annotated = image.copy()

    # GT
    for gi, gt in enumerate(gts):

        poly = gt["poly"]
        issues = gt_issues[gi]
        extra = gt_extra[gi]

        x = np.min(poly[:, 0])
        y = np.min(poly[:, 1])

        if issues:

            draw_poly(
                annotated,
                poly,
                (0, 0, 255),
                BOX_THICKNESS + 1
            )

            text = (
                f"GT#{gi} "
                + "+".join(issues)
            )

            if extra["confidence"] is not None:
                text += (
                    f" conf={extra['confidence']:.2f}"
                )

            text += (
                f" IoU={extra['best_iou']:.2f}"
            )

            if extra["angle_error"] is not None:
                text += (
                    f" angle={extra['angle_error']:.1f}deg"
                )

            draw_text(
                annotated,
                text,
                x,
                y - 7,
                (0, 0, 255)
            )

        else:

            draw_poly(
                annotated,
                poly,
                (0, 210, 0),
                BOX_THICKNESS
            )

    # 预测框
    matched_pred_indices = {
        pi
        for _, pi, _ in eval_matches
    }

    for pi, pred in enumerate(eval_preds):

        poly = pred["poly"]
        x = np.min(poly[:, 0])
        y = np.max(poly[:, 1])

        if pi in fp_indices:

            draw_poly(
                annotated,
                poly,
                (0, 165, 255),
                BOX_THICKNESS + 1
            )

            draw_text(
                annotated,
                f"FP conf={pred['conf']:.2f}",
                x,
                y + 20,
                (0, 165, 255)
            )

        elif pi in matched_pred_indices:

            draw_poly(
                annotated,
                poly,
                (255, 255, 0),
                BOX_THICKNESS
            )

    draw_panel(
        annotated,
        counts
    )

    return {
        "image": image,
        "annotated": annotated,
        "label_path": label_path,
        "gts": gts,
        "eval_preds": eval_preds,
        "gt_issues": gt_issues,
        "gt_extra": gt_extra,
        "fp_indices": fp_indices,
        "counts": counts,
        "issue_tags": issue_tags,
        "is_problem": is_problem,
    }


# =============================================================================
# 10. 运行进度显示
# =============================================================================

def format_seconds(seconds):
    """把秒数格式化为 HH:MM:SS 或 MM:SS。"""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--"

    seconds = int(round(seconds))
    hours, remain = divmod(seconds, 3600)
    minutes, secs = divmod(remain, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def print_progress(
    current,
    total,
    start_time,
    problem_count,
    totals,
    current_image=None,
):
    """
    单行实时刷新进度，例如：

    [██████████░░░░]  72/274  26.3% | elapsed 03:18 | ETA 09:15
    problem=18 | FN=4 | LOW_CONF=11 | LOW_IOU=8 | ANGLE_ERR=3 | FP=2 | xxx.jpg
    """
    if not SHOW_PROGRESS:
        return

    if total <= 0:
        return

    current = max(0, min(current, total))
    ratio = current / total

    filled = int(round(PROGRESS_BAR_WIDTH * ratio))
    filled = max(0, min(filled, PROGRESS_BAR_WIDTH))

    bar = (
        "█" * filled
        + "░" * (PROGRESS_BAR_WIDTH - filled)
    )

    elapsed = max(time.perf_counter() - start_time, 0.0)

    if current > 0:
        sec_per_image = elapsed / current
        eta = sec_per_image * (total - current)
    else:
        sec_per_image = 0.0
        eta = None

    line = (
        f"[分析 {bar}] "
        f"{current:>4}/{total:<4} "
        f"{ratio * 100:6.2f}% | "
        f"elapsed {format_seconds(elapsed)} | "
        f"ETA {format_seconds(eta)} | "
        f"{sec_per_image:5.2f}s/img | "
        f"problem={problem_count} | "
        f"FN={totals['FN']} "
        f"LOWC={totals['LOW_CONF']} "
        f"LOWIOU={totals['LOW_IOU']} "
        f"ANGLE={totals['ANGLE_ERR']} "
        f"FP={totals['FP']}"
    )

    if SHOW_CURRENT_FILENAME and current_image is not None:
        filename = current_image.name

        # 避免文件名过长导致终端一行无限延伸
        if len(filename) > 38:
            filename = "..." + filename[-35:]

        line += f" | {filename}"

    # 加一些尾部空格，用来覆盖上一轮更长的文本
    sys.stdout.write("\r" + line + " " * 12)
    sys.stdout.flush()


# =============================================================================
# 11. README
# =============================================================================

def write_readme(run_dir):
    text = f"""YOLO11-OBB 问题样本查看说明
============================================================

你的验证集来自：
{VAL_TXT_PATH}

问题图片目录：
{run_dir / "problem_images"}

============================================================
如何看文件名
============================================================

FN_xxx.jpg
    这张图存在漏检。

LOWCONF_xxx.jpg
    这张图存在低置信度目标。

LOWIOU_xxx.jpg
    这张图存在 OBB 定位偏差。

ANGLE_xxx.jpg
    这张图存在角度误差 > {ANGLE_ERROR_THRESH_DEG:.1f}° 的目标。

FP_xxx.jpg
    这张图存在误检。

如果同时存在多个问题，会组合，例如：

FN_LOWCONF_ANGLE__xxx.jpg

============================================================
如何看图片里的框
============================================================

绿色 GT：
    当前未被判定为问题的 GT 包裹。

红色 GT：
    当前存在问题的包裹。
    红框旁会显示：
        FN
        LOW_CONF
        LOW_IOU
        ANGLE_ERR
    以及 confidence / IoU / angle error 等信息。

青色框：
    正常匹配预测框。

橙色 FP：
    没有对应 GT 的误检预测。

============================================================
下一轮数据集怎么补
============================================================

1. FN 漏检
    优先级最高。
    人工判断该包裹属于：
    - 遮挡
    - 贴靠
    - 图像边缘
    - 反光
    - 软袋/异形
    - 小目标
    - 模糊
    - 特殊颜色/材质
    中的哪一种。

    下一轮重点补同类“真实困难样本”。

2. LOW_CONF
    模型已经看到了，但不够确定。
    这些是以后换工厂/光照后最容易变成漏检的目标。
    多补相似外观、尺度、姿态和背景条件。

3. LOW_IOU
    先检查 GT 标注是否准确。
    若 GT 正确，则补相似姿态、形状、遮挡和贴靠目标。

4. ANGLE_ERR
    先检查角度标注的一致性。
    若 GT 正确，则补相似旋转角度、长宽比及边界模糊目标。

5. FP
    先检查是不是漏标。
    如果确实不是包裹，就把这种背景/设备外观作为 hard negative。

============================================================
当前判定参数
============================================================

INFER_CONF = {INFER_CONF}
EVAL_CONF = {EVAL_CONF}
MATCH_IOU = {MATCH_IOU}
LOW_CONF_THRESH = {LOW_CONF_THRESH}
LOW_IOU_THRESH = {LOW_IOU_THRESH}
ANGLE_ERROR_THRESH_DEG = {ANGLE_ERROR_THRESH_DEG}
ANGLE_MIN_ASPECT_RATIO = {ANGLE_MIN_ASPECT_RATIO}

============================================================
建议查看顺序
============================================================

FN
↓
LOW_CONF
↓
ANGLE_ERR / LOW_IOU
↓
FP

详细数值同时保存在：
problem_samples.csv
"""

    (
        run_dir
        / "README_问题说明.txt"
    ).write_text(
        text,
        encoding="utf-8-sig"
    )


# =============================================================================
# 12. 分批推理工具
# =============================================================================

def iter_inference_batches(model, image_paths):
    """
    按 PRED_BATCH 分批调用 model.predict，并明确显示推理进度。

    这样不会再出现长时间停在 0/275 而不知道模型正在做什么。
    CPU 模式下尤其有用。

    每完成一个 batch，就会显示：
        [推理] 2/275 (0.73%)
        [推理] 4/275 (1.45%)
        ...
    """
    total = len(image_paths)
    batch_size = max(int(PRED_BATCH), 1)

    inference_start = time.perf_counter()
    inferred = 0

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_paths = image_paths[batch_start:batch_end]

        if SHOW_INFERENCE_PROGRESS:
            print(
                f"\n[推理中] 第 {batch_start + 1}~{batch_end} 张 / 共 {total} 张 "
                f"({batch_start / total * 100:.2f}%)",
                flush=True,
            )

        batch_start_time = time.perf_counter()

        batch_results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=IMGSZ,
            conf=INFER_CONF,
            iou=NMS_IOU,
            batch=len(batch_paths),
            device=DEVICE,
            stream=False,
            verbose=False,
        )

        batch_elapsed = time.perf_counter() - batch_start_time
        inferred = batch_end
        total_elapsed = time.perf_counter() - inference_start

        if inferred > 0:
            sec_per_image = total_elapsed / inferred
            eta = sec_per_image * (total - inferred)
        else:
            sec_per_image = 0.0
            eta = None

        if SHOW_INFERENCE_PROGRESS:
            print(
                f"[推理完成] {inferred}/{total} "
                f"({inferred / total * 100:.2f}%) | "
                f"本批 {batch_elapsed:.2f}s | "
                f"平均 {sec_per_image:.2f}s/图 | "
                f"ETA {format_seconds(eta)}",
                flush=True,
            )

        for result in batch_results:
            yield result


# =============================================================================
# 13. 主程序
# =============================================================================

def main():

    print()
    print("=" * 88)
    print("YOLO11-OBB 本地问题样本分析")
    print("=" * 88)

    print("[1/5] 检查模型与验证集路径...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型不存在：{MODEL_PATH}"
        )

    if not VAL_TXT_PATH.exists():
        raise FileNotFoundError(
            f"val.txt 不存在：{VAL_TXT_PATH}"
        )

    print("      模型路径检查通过")
    print("      val.txt 路径检查通过")

    print("[2/5] 读取验证集图片列表...")
    image_paths = read_val_list()
    print(f"      已读取 {len(image_paths)} 张有效验证图片")

    run_dir = (
        OUTPUT_DIR
        / RUN_NAME
    )

    problem_dir = (
        run_dir
        / "problem_images"
    )

    original_dir = (
        run_dir
        / "original_problem_images"
    )

    ensure_dir(problem_dir)

    if SAVE_ORIGINAL_PROBLEM_IMAGE:
        ensure_dir(original_dir)

    print(f"      输出目录：{problem_dir}")
    print()
    print("[3/5] 加载 YOLO OBB 模型...")
    model_load_start = time.perf_counter()

    model = YOLO(
        str(MODEL_PATH)
    )

    print(
        f"      模型加载完成，耗时 "
        f"{format_seconds(time.perf_counter() - model_load_start)}"
    )
    print()
    print("[4/5] 开始推理并逐张分析验证集...")
    print(
        f"      DEVICE={DEVICE}, IMGSZ={IMGSZ}, BATCH={PRED_BATCH}, "
        f"INFER_CONF={INFER_CONF}, EVAL_CONF={EVAL_CONF}"
    )
    print(
        "      程序会分别显示【推理进度】和【问题分析进度】。"
    )
    print(
        "      problem/FN/LOWC/LOWIOU/ANGLE/FP 会随着分析实时累计。"
    )
    print()

    analysis_start_time = time.perf_counter()

    # 这里不再一次性创建 stream=True 的预测流，
    # 而是按 PRED_BATCH 主动分批推理。
    # 这样每完成一批都能明确看到“已经推理了多少张”。
    results = iter_inference_batches(
        model=model,
        image_paths=image_paths,
    )

    total_problem_images = 0
    total_gt = 0

    totals = {
        "FN": 0,
        "LOW_CONF": 0,
        "LOW_IOU": 0,
        "ANGLE_ERR": 0,
        "FP": 0,
    }

    csv_rows = []

    # 先显示 0% 状态
    print_progress(
        current=0,
        total=len(image_paths),
        start_time=analysis_start_time,
        problem_count=total_problem_images,
        totals=totals,
        current_image=None,
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        image_path = Path(
            result.path
        )

        analysis = analyze_image(
            image_path,
            result
        )

        total_gt += len(
            analysis["gts"]
        )

        for tag in totals:
            totals[tag] += analysis["counts"][tag]

        output_name = ""

        if analysis["is_problem"]:

            total_problem_images += 1

            output_name = safe_output_name(
                image_path,
                analysis["issue_tags"]
            )

            cv2.imwrite(
                str(
                    problem_dir
                    / output_name
                ),
                analysis["annotated"]
            )

            if SAVE_ORIGINAL_PROBLEM_IMAGE:
                cv2.imwrite(
                    str(
                        original_dir
                        / output_name
                    ),
                    analysis["image"]
                )

        # GT 明细
        for gi, gt in enumerate(
            analysis["gts"]
        ):

            issues = analysis[
                "gt_issues"
            ][gi]

            extra = analysis[
                "gt_extra"
            ][gi]

            advice = "；".join(
                ISSUE_ADVICE[tag]
                for tag in issues
                if tag in ISSUE_ADVICE
            )

            csv_rows.append({
                "source_dataset": image_path.parents[2].name
                    if len(image_path.parents) >= 3 else "",
                "image": str(image_path),
                "label": str(analysis["label_path"]),
                "output_image": output_name,
                "object_type": "GT",
                "gt_index": gi,
                "issues": "|".join(issues),
                "confidence": (
                    round(
                        extra["confidence"],
                        6
                    )
                    if extra["confidence"] is not None
                    else ""
                ),
                "best_iou": round(
                    extra["best_iou"],
                    6
                ),
                "angle_error_deg": (
                    round(
                        extra["angle_error"],
                        4
                    )
                    if extra["angle_error"] is not None
                    else ""
                ),
                "aspect_ratio": (
                    round(
                        extra["aspect_ratio"],
                        4
                    )
                    if extra["aspect_ratio"] is not None
                    else ""
                ),
                "recommended_action": advice,
            })

        # FP 明细
        for pi in analysis["fp_indices"]:

            pred = analysis[
                "eval_preds"
            ][pi]

            csv_rows.append({
                "source_dataset": image_path.parents[2].name
                    if len(image_path.parents) >= 3 else "",
                "image": str(image_path),
                "label": str(analysis["label_path"]),
                "output_image": output_name,
                "object_type": "FP",
                "gt_index": "",
                "issues": "FP",
                "confidence": round(
                    pred["conf"],
                    6
                ),
                "best_iou": "",
                "angle_error_deg": "",
                "aspect_ratio": "",
                "recommended_action": ISSUE_ADVICE["FP"],
            })

        if (
            index % max(PROGRESS_UPDATE_EVERY, 1) == 0
            or index == len(image_paths)
        ):
            print_progress(
                current=index,
                total=len(image_paths),
                start_time=analysis_start_time,
                problem_count=total_problem_images,
                totals=totals,
                current_image=image_path,
            )

    # 结束实时进度行，避免后面的文字接在同一行
    if SHOW_PROGRESS:
        print()

    analysis_elapsed = time.perf_counter() - analysis_start_time

    print(
        f"      图片分析完成：{len(image_paths)} 张，"
        f"耗时 {format_seconds(analysis_elapsed)}，"
        f"平均 {analysis_elapsed / max(len(image_paths), 1):.2f} s/图"
    )
    print()
    print("[5/5] 正在生成 CSV、README 和 summary...")

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    csv_path = (
        run_dir
        / "problem_samples.csv"
    )

    fields = [
        "source_dataset",
        "image",
        "label",
        "output_image",
        "object_type",
        "gt_index",
        "issues",
        "confidence",
        "best_iou",
        "angle_error_deg",
        "aspect_ratio",
        "recommended_action",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(
            csv_rows
        )

    # -------------------------------------------------------------------------
    # README
    # -------------------------------------------------------------------------

    write_readme(
        run_dir
    )

    # -------------------------------------------------------------------------
    # summary
    # -------------------------------------------------------------------------

    summary = f"""YOLO11-OBB 问题样本分析汇总
============================================================

模型：
{MODEL_PATH}

验证集列表：
{VAL_TXT_PATH}

验证图片总数：
{len(image_paths)}

GT 总数：
{total_gt}

问题图片数：
{total_problem_images}

FN 漏检：
{totals['FN']}

LOW_CONF 低置信度：
{totals['LOW_CONF']}

LOW_IOU OBB定位偏差：
{totals['LOW_IOU']}

ANGLE_ERR 角度误差：
{totals['ANGLE_ERR']}

FP 误检：
{totals['FP']}

============================================================

问题图片：
{problem_dir}

详细 CSV：
{csv_path}

中文说明：
{run_dir / "README_问题说明.txt"}

建议优先查看：
FN -> LOW_CONF -> ANGLE_ERR / LOW_IOU -> FP
"""

    summary_path = (
        run_dir
        / "summary.txt"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8-sig"
    )

    print("      报告文件生成完成")
    print()
    print("=" * 88)
    print("全部分析完成")
    print("=" * 88)
    print(
        f"问题图片：{problem_dir}"
    )
    print(
        f"CSV：     {csv_path}"
    )
    print(
        f"说明：    {run_dir / 'README_问题说明.txt'}"
    )
    print(
        f"汇总：    {summary_path}"
    )
    print()
    print(
        "建议打开 problem_images 后，"
        "先按文件名前缀查看 FN -> LOWCONF -> ANGLE -> LOWIOU -> FP"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
