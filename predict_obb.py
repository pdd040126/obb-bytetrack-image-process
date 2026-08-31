#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量读取指定文件夹中的图片，使用 YOLO OBB 模型进行检测，
并将检测结果保存到指定输出文件夹。

支持两种模型输入模式：

1. square
   使用正方形输入：
   960 × 960

2. rect
   使用接近 16:9 的矩形输入：
   960 × 544

注意：
Ultralytics 中 imgsz=(height, width)，
所以矩形输入需要写成：

    (544, 960)

实际代表：

    宽 960 × 高 544

输出结构：

output/
├── images/
│   └── 带旋转检测框的结果图片
│
└── predictions/
    └── 每张图片对应的检测结果 txt

不需要：
- data.yaml
- train.txt
- labels
- 验证集
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import ultralytics
import ultralytics.nn.modules.block as block

from ultralytics import YOLO


# ============================================================
# 用户参数设置
# ============================================================


# ------------------------------------------------------------
# 正方形输入尺寸
# ------------------------------------------------------------
#
# 最终输入：
#
# 960 × 960
#

SQUARE_IMGSZ = 960

# ------------------------------------------------------------
# 模型文件
# ------------------------------------------------------------

MODEL_PATH = Path(
    r"F:\obb_test\models\best-083102s.pt"
)


# ------------------------------------------------------------
# 输入图片文件夹
# ------------------------------------------------------------

SOURCE_DIR = Path(
    r"F:\image_process_data\images_predict_input"
)


# ------------------------------------------------------------
# 输出文件夹
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"F:\image_process_data\images_predict_output"
)


# ------------------------------------------------------------
# 模型输入模式
# ------------------------------------------------------------
#
# 可选：
#
# "square"
#     使用 960 × 960 正方形输入
#
# "rect"
#     使用 960 × 544 矩形输入
#
# 推荐实验时分别跑一次：
#
# INPUT_MODE = "square"
#
# 和：
#
# INPUT_MODE = "rect"
#
# 然后比较：
# - 检测数量
# - 检测结果
# - OBB角度
# - 推理时间
# - FPS
#

INPUT_MODE = "square"


# ------------------------------------------------------------
# 矩形输入尺寸
# ------------------------------------------------------------
#
# Ultralytics 顺序：
#
# (height, width)
#
# 因此：
#
# (544, 960)
#
# 表示实际：
#
# 宽960 × 高544
#
# 1920×1080 是16:9。
#
# 当宽度缩放到960时：
#
# 1080 × 960 / 1920 = 540
#
# 再向上对齐到32的倍数：
#
# 544 = 17 × 32
#

RECT_IMGSZ = (544, 960)


# ------------------------------------------------------------
# 置信度阈值
# ------------------------------------------------------------

CONF_THRESHOLD = 0.30


# ------------------------------------------------------------
# NMS IoU阈值
# ------------------------------------------------------------

IOU_THRESHOLD = 0.30


# ------------------------------------------------------------
# 检测框绘制线宽
# ------------------------------------------------------------

LINE_WIDTH = 5


# ------------------------------------------------------------
# 推理设备
# ------------------------------------------------------------
#
# "0"：
#     使用第0张GPU
#
# "cpu"：
#     使用CPU
#
# 默认：
#     如果检测到CUDA，则使用GPU 0
#

DEVICE = "0" if torch.cuda.is_available() else "cpu"


# ============================================================
# 兼容旧模型中的 GhostBottleneck2
# ============================================================


class GhostBottleneck2(nn.Module):
    """
    用于兼容旧模型中保存的自定义 GhostBottleneck2。

    如果模型训练时使用了这个自定义模块，
    当前 Ultralytics 环境中没有对应定义，
    加载 .pt 时可能报错。

    因此在这里重新注册。
    """

    def forward(self, x):

        y = self.cv2(
            self.cv1(x)
        )

        if self.add:
            return x + y

        return y


# 把 GhostBottleneck2 注册到
# ultralytics.nn.modules.block 中
GhostBottleneck2.__module__ = block.__name__

block.GhostBottleneck2 = GhostBottleneck2


# ============================================================
# 支持的图片格式
# ============================================================

IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# 命令行参数
# ============================================================
#
# 虽然主要参数已经放到代码顶部，
# 这里仍然保留 model / source / output 的命令行覆盖能力。
#
# 平时直接运行即可，不需要输入参数。
#
# 如果以后临时需要换模型，也仍然可以：
#
# python predict_obb.py --model xxx.pt
#

def parse_args():

    parser = argparse.ArgumentParser(
        description="批量读取文件夹图片并进行 YOLO OBB 检测"
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="YOLO OBB 模型文件",
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_DIR,
        help="待检测图片文件夹",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="检测结果输出文件夹",
    )

    return parser.parse_args()


# ============================================================
# 根据 INPUT_MODE 获取模型输入尺寸
# ============================================================

def get_inference_imgsz():
    """
    根据 INPUT_MODE 返回实际传给 Ultralytics 的 imgsz。

    square：
        返回 960

    rect：
        返回 (544, 960)

    注意：
        Ultralytics tuple 顺序为
        (height, width)
    """

    mode = INPUT_MODE.strip().lower()

    if mode == "square":

        if SQUARE_IMGSZ <= 0:
            raise ValueError(
                "SQUARE_IMGSZ 必须大于0"
            )

        return SQUARE_IMGSZ

    if mode == "rect":

        if len(RECT_IMGSZ) != 2:
            raise ValueError(
                "RECT_IMGSZ 必须为 (height, width)"
            )

        height, width = RECT_IMGSZ

        if height <= 0 or width <= 0:
            raise ValueError(
                "RECT_IMGSZ 的高度和宽度必须大于0"
            )

        return RECT_IMGSZ

    raise ValueError(
        f"INPUT_MODE 设置错误：{INPUT_MODE!r}\n"
        '只能设置为 "square" 或 "rect"'
    )


# ============================================================
# 格式化模型输入尺寸
# ============================================================

def format_imgsz(imgsz):
    """
    将模型内部 imgsz 转换成常见的：

        宽 × 高

    显示方式。

    例如：

        960
        -> 960 × 960

        (544, 960)
        -> 960 × 544
    """

    if isinstance(imgsz, int):

        return f"{imgsz} × {imgsz}"

    height, width = imgsz

    return f"{width} × {height}"


# ============================================================
# 收集待检测图片
# ============================================================

def collect_images(source: Path):
    """
    递归读取 source 文件夹中的所有支持图片。

    例如：

    source/
    ├── a.jpg
    ├── b.png
    └── folder/
        └── c.jpg

    都会被读取。
    """

    images = []

    for path in source.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        images.append(
            path.resolve()
        )

    return sorted(images)


# ============================================================
# 保存 OBB 检测结果 TXT
# ============================================================

def save_prediction_txt(
    output_path: Path,
    obb,
):
    """
    将每张图片的 OBB 检测结果保存为 TXT。

    每个目标一行。

    每行格式：

    class_id
    confidence
    cx
    cy
    width
    height
    angle_rad
    x1 y1
    x2 y2
    x3 y3
    x4 y4

    即：

    class_id confidence
    cx cy width height angle_rad
    x1 y1 x2 y2 x3 y3 x4 y4

    所有坐标都是：

        原始图片像素坐标

    而不是模型输入尺寸中的坐标。

    因此：

    不管使用：
        960×960

    还是：
        960×544

    最终TXT中的坐标仍然对应原始图片。
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        # 没有检测目标时，
        # 创建一个空TXT文件。
        if obb is None or len(obb) == 0:
            return

        # 转到CPU，
        # 后续转换numpy。
        obb = obb.cpu()

        # 类别ID
        class_ids = (
            obb.cls
            .numpy()
            .astype(int)
        )

        # 置信度
        confidences = (
            obb.conf
            .numpy()
        )

        # OBB中心形式：
        #
        # cx
        # cy
        # width
        # height
        # rotation
        xywhr = (
            obb.xywhr
            .numpy()
        )

        # OBB四个角点
        corners = (
            obb.xyxyxyxy
            .numpy()
        )

        for (
            class_id,
            confidence,
            box,
            points,
        ) in zip(
            class_ids,
            confidences,
            xywhr,
            corners,
        ):

            (
                cx,
                cy,
                width,
                height,
                angle_rad,
            ) = map(
                float,
                box,
            )

            # 将：
            #
            # [[x1,y1],
            #  [x2,y2],
            #  [x3,y3],
            #  [x4,y4]]
            #
            # 展平成：
            #
            # x1 y1 x2 y2 x3 y3 x4 y4
            points = [
                float(value)
                for value
                in points.reshape(-1)
            ]

            values = [
                str(int(class_id)),
                f"{float(confidence):.6f}",
                f"{cx:.3f}",
                f"{cy:.3f}",
                f"{width:.3f}",
                f"{height:.3f}",
                f"{angle_rad:.8f}",
            ]

            values.extend(
                f"{value:.3f}"
                for value
                in points
            )

            file.write(
                " ".join(values)
                + "\n"
            )


# ============================================================
# 主程序
# ============================================================

def main():

    # --------------------------------------------------------
    # 获取命令行参数
    # --------------------------------------------------------

    args = parse_args()


    # --------------------------------------------------------
    # 根据 INPUT_MODE 选择模型输入尺寸
    # --------------------------------------------------------

    imgsz = get_inference_imgsz()


    # --------------------------------------------------------
    # 解析路径
    # --------------------------------------------------------

    model_path = (
        args.model
        .expanduser()
        .resolve()
    )

    source_dir = (
        args.source
        .expanduser()
        .resolve()
    )

    output_dir = (
        args.output
        .expanduser()
        .resolve()
    )


    # --------------------------------------------------------
    # 检查模型文件
    # --------------------------------------------------------

    if not model_path.is_file():

        raise FileNotFoundError(
            f"模型文件不存在：{model_path}"
        )


    # --------------------------------------------------------
    # 检查输入图片文件夹
    # --------------------------------------------------------

    if not source_dir.is_dir():

        raise NotADirectoryError(
            f"图片文件夹不存在：{source_dir}"
        )


    # --------------------------------------------------------
    # 检查置信度参数
    # --------------------------------------------------------

    if not 0.0 <= CONF_THRESHOLD <= 1.0:

        raise ValueError(
            "CONF_THRESHOLD 必须位于 [0, 1]"
        )


    # --------------------------------------------------------
    # 检查 IoU 参数
    # --------------------------------------------------------

    if not 0.0 <= IOU_THRESHOLD <= 1.0:

        raise ValueError(
            "IOU_THRESHOLD 必须位于 [0, 1]"
        )


    # --------------------------------------------------------
    # 收集图片
    # --------------------------------------------------------

    images = collect_images(
        source_dir
    )

    if not images:

        raise RuntimeError(
            f"文件夹中没有找到可处理图片："
            f"{source_dir}"
        )


    # --------------------------------------------------------
    # 创建输出文件夹
    # --------------------------------------------------------

    output_image_dir = (
        output_dir
        / "images"
    )

    output_prediction_dir = (
        output_dir
        / "predictions"
    )

    output_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # 打印运行配置
    # ========================================================

    print()

    print(
        "========== 运行配置 =========="
    )

    print(
        f"Python：        "
        f"{sys.version.split()[0]}"
    )

    print(
        f"PyTorch：       "
        f"{torch.__version__}"
    )

    print(
        f"Ultralytics：   "
        f"{ultralytics.__version__}"
    )

    print(
        f"CUDA 可用：     "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        try:

            gpu_name = (
                torch.cuda
                .get_device_name(0)
            )

            print(
                f"GPU：           "
                f"{gpu_name}"
            )

        except Exception:

            pass

    print(
        f"模型：          "
        f"{model_path}"
    )

    print(
        f"图片文件夹：    "
        f"{source_dir}"
    )

    print(
        f"输出文件夹：    "
        f"{output_dir}"
    )

    print(
        f"图片数量：      "
        f"{len(images)}"
    )

    print(
        f"设备：          "
        f"{DEVICE}"
    )

    print(
        f"输入模式：      "
        f"{INPUT_MODE}"
    )

    print(
        f"输入尺寸：      "
        f"{format_imgsz(imgsz)}"
    )

    print(
        f"置信度阈值：    "
        f"{CONF_THRESHOLD}"
    )

    print(
        f"NMS IoU阈值：   "
        f"{IOU_THRESHOLD}"
    )

    print(
        "=============================="
    )

    print()


    # ========================================================
    # 加载模型
    # ========================================================

    print(
        "正在加载模型..."
    )

    model = YOLO(
        str(model_path)
    )

    # 检查是不是 OBB 模型
    if model.task != "obb":

        raise RuntimeError(
            f"当前模型任务为 "
            f"{model.task!r}，"
            f"不是 OBB 模型"
        )

    print(
        "模型加载完成。"
    )

    print()


    # ========================================================
    # 开始批量检测
    # ========================================================

    total_detections = 0

    # model.predict() 总时间
    total_predict_wall_time = 0.0

    # Ultralytics内部统计
    total_preprocess_ms = 0.0
    total_inference_ms = 0.0
    total_postprocess_ms = 0.0

    script_start_time = (
        time.perf_counter()
    )


    # --------------------------------------------------------
    # 逐张图片检测
    # --------------------------------------------------------

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        # ----------------------------------------------------
        # 记录整个 model.predict() 的时间
        # ----------------------------------------------------

        predict_start = (
            time.perf_counter()
        )


        # ====================================================
        # YOLO OBB 推理
        # ====================================================

        results = model.predict(

            # 输入图片
            source=str(image_path),

            # 模型输入尺寸
            #
            # square：
            #     960
            #
            # rect：
            #     (544, 960)
            #
            imgsz=imgsz,

            # 置信度阈值
            conf=CONF_THRESHOLD,

            # NMS IoU
            iou=IOU_THRESHOLD,

            # GPU / CPU
            device=DEVICE,

            # 不打印Ultralytics每张图片详细日志
            verbose=False,
        )


        # ----------------------------------------------------
        # 结束计时
        # ----------------------------------------------------

        predict_end = (
            time.perf_counter()
        )

        predict_wall_time = (
            predict_end
            - predict_start
        )

        total_predict_wall_time += (
            predict_wall_time
        )


        # ----------------------------------------------------
        # 检查返回结果
        # ----------------------------------------------------

        if not results:

            print(
                f"[{index}/{len(images)}] "
                f"无推理结果："
                f"{image_path.name}"
            )

            continue


        # 这里只输入一张图片，
        # 所以取 results[0]
        result = results[0]


        # ----------------------------------------------------
        # 检查 OBB
        # ----------------------------------------------------

        if result.obb is None:

            raise RuntimeError(
                "模型没有返回 OBB 旋转框结果"
            )


        # ----------------------------------------------------
        # 获取检测目标数量
        # ----------------------------------------------------

        count = len(
            result.obb
        )

        total_detections += count


        # ====================================================
        # 获取 Ultralytics 内部速度统计
        # ====================================================
        #
        # result.speed 通常类似：
        #
        # {
        #     "preprocess": ...
        #     "inference": ...
        #     "postprocess": ...
        # }
        #
        # 单位：
        # ms/image
        #

        speed = (
            result.speed
            if result.speed
            else {}
        )

        preprocess_ms = float(
            speed.get(
                "preprocess",
                0.0,
            )
        )

        inference_ms = float(
            speed.get(
                "inference",
                0.0,
            )
        )

        postprocess_ms = float(
            speed.get(
                "postprocess",
                0.0,
            )
        )

        total_preprocess_ms += (
            preprocess_ms
        )

        total_inference_ms += (
            inference_ms
        )

        total_postprocess_ms += (
            postprocess_ms
        )


        # ====================================================
        # 创建输出文件名
        # ====================================================
        #
        # 添加 index，
        # 避免不同子文件夹内存在同名图片
        # 导致结果互相覆盖。
        #

        output_stem = (
            f"{index:06d}_"
            f"{image_path.stem}"
        )

        output_image_path = (
            output_image_dir
            / f"{output_stem}.jpg"
        )

        output_txt_path = (
            output_prediction_dir
            / f"{output_stem}.txt"
        )


        # ====================================================
        # 绘制 OBB 检测结果
        # ====================================================

        plotted = result.plot(

            # 显示置信度
            conf=True,

            # 显示类别标签
            labels=True,

            # 检测框线宽
            line_width=LINE_WIDTH,
        )


        # ====================================================
        # 保存检测结果图片
        # ====================================================

        success = cv2.imwrite(
            str(output_image_path),
            plotted,
        )

        if not success:

            raise RuntimeError(
                f"结果图片保存失败："
                f"{output_image_path}"
            )


        # ====================================================
        # 保存检测 TXT
        # ====================================================

        save_prediction_txt(
            output_txt_path,
            result.obb,
        )


        # ====================================================
        # 当前图片速度信息
        # ====================================================

        wall_fps = (
            1.0 / predict_wall_time
            if predict_wall_time > 0
            else 0.0
        )

        inference_fps = (
            1000.0 / inference_ms
            if inference_ms > 0
            else 0.0
        )


        # ----------------------------------------------------
        # 打印当前图片结果
        # ----------------------------------------------------

        print(

            f"[{index}/{len(images)}] "

            f"{image_path.name} "

            f"-> {count} 个目标 "

            f"| preprocess "
            f"{preprocess_ms:.2f} ms "

            f"| inference "
            f"{inference_ms:.2f} ms "

            f"| postprocess "
            f"{postprocess_ms:.2f} ms "

            f"| infer FPS "
            f"{inference_fps:.1f} "

            f"| predict总时间 "
            f"{predict_wall_time * 1000:.1f} ms "

            f"| wall FPS "
            f"{wall_fps:.1f}"
        )


    # ========================================================
    # 最终统计
    # ========================================================

    script_end_time = (
        time.perf_counter()
    )

    total_script_time = (
        script_end_time
        - script_start_time
    )


    image_count = len(images)


    # --------------------------------------------------------
    # 平均 model.predict() 总时间
    # --------------------------------------------------------

    average_predict_wall_time = (
        total_predict_wall_time
        / image_count
    )


    # --------------------------------------------------------
    # model.predict() 对应FPS
    # --------------------------------------------------------

    average_wall_fps = (
        1.0 / average_predict_wall_time
        if average_predict_wall_time > 0
        else 0.0
    )


    # --------------------------------------------------------
    # 平均预处理
    # --------------------------------------------------------

    average_preprocess_ms = (
        total_preprocess_ms
        / image_count
    )


    # --------------------------------------------------------
    # 平均纯推理
    # --------------------------------------------------------

    average_inference_ms = (
        total_inference_ms
        / image_count
    )


    # --------------------------------------------------------
    # 平均后处理
    # --------------------------------------------------------

    average_postprocess_ms = (
        total_postprocess_ms
        / image_count
    )


    # --------------------------------------------------------
    # 根据纯 inference 时间计算理论单流FPS
    # --------------------------------------------------------

    average_inference_fps = (
        1000.0
        / average_inference_ms
        if average_inference_ms > 0
        else 0.0
    )


    # ========================================================
    # 打印最终结果
    # ========================================================

    print()

    print(
        "========== 检测完成 =========="
    )

    print(
        f"输入模式：          "
        f"{INPUT_MODE}"
    )

    print(
        f"输入尺寸：          "
        f"{format_imgsz(imgsz)}"
    )

    print(
        f"处理图片：          "
        f"{image_count} 张"
    )

    print(
        f"检测目标总数：      "
        f"{total_detections}"
    )

    print()

    print(
        "---------- YOLO速度 ----------"
    )

    print(
        f"平均预处理：        "
        f"{average_preprocess_ms:.3f} ms/image"
    )

    print(
        f"平均纯推理：        "
        f"{average_inference_ms:.3f} ms/image"
    )

    print(
        f"平均后处理：        "
        f"{average_postprocess_ms:.3f} ms/image"
    )

    print(
        f"纯推理理论FPS：     "
        f"{average_inference_fps:.2f}"
    )

    print()

    print(
        "---------- 实际调用 ----------"
    )

    print(
        f"平均predict总时间： "
        f"{average_predict_wall_time * 1000:.3f} ms/image"
    )

    print(
        f"predict实际FPS：    "
        f"{average_wall_fps:.2f}"
    )

    print(
        f"程序总耗时：        "
        f"{total_script_time:.2f} s"
    )

    print()

    print(
        f"结果图片：          "
        f"{output_image_dir}"
    )

    print(
        f"检测数据：          "
        f"{output_prediction_dir}"
    )

    print(
        "=============================="
    )

    print()


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n用户中断运行。",
            file=sys.stderr,
        )

        raise SystemExit(130)

    except Exception as exc:

        print(
            f"\n运行失败：{exc}",
            file=sys.stderr,
        )

        raise