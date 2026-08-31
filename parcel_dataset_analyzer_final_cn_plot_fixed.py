# -*- coding: utf-8 -*-
"""
包裹数据集分析工具（修正版，按“目标级/图片级”分开统计）

核心修正：
1. 包裹数量分布：按“图片级”统计（一张图属于单件/2-3件/4-6件/多件）
2. 尺寸分布：按“目标级”统计（每个包裹一个样本）
3. 角度分布：按“目标级”统计（每个包裹一个样本）
4. 位置分布：按“目标级”统计（每个包裹中心点一个点）
5. 保留中文输出、进度条、原始目录配置方式

支持目录结构：

images_cvatXX_dataset
├── images
│   └── train
│       xxx.jpg
└── labels
    └── train
        xxx.txt

标签格式：
class x1 y1 x2 y2 x3 y3 x4 y4
（坐标为归一化坐标）
"""

from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt


# ============================
# 中文字体设置
# ============================
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================
# 用户配置区域
# ============================

DATASET_DIRS = [
    r"H:\train_data\images_cvat00_dataset",
    r"H:\train_data\images_cvat01_dataset",
    r"H:\train_data\images_cvat02_dataset",
    r"H:\train_data\images_cvat04_dataset",
    r"H:\train_data\images_cvat08_dataset",
]

# 按你之前的结构保留在 train_data 下
OUTPUT_DIR = r"H:\train_data\dataset_analysis"

# 0 代表全部图片
MAX_FILES = 0


# ============================
# 文件类型
# ============================

EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================
# 基础工具函数
# ============================

def read_image(path):
    """
    兼容中文路径读取图片
    """
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def find_label(img_path, root):
    """
    由图片路径匹配对应 label 路径
    例如：
    images/train/a.jpg -> labels/train/a.txt
    """
    rel = img_path.relative_to(root / "images")
    return root / "labels" / rel.with_suffix(".txt")


def read_obb(label_path, w, h):
    """
    读取 YOLO-OBB 四点标签，并从归一化坐标转换为像素坐标
    """
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()

        if len(values) < 9:
            continue

        pts = np.array(
            [float(x) for x in values[1:9]],
            dtype=np.float32
        ).reshape(4, 2)

        pts[:, 0] *= w
        pts[:, 1] *= h

        boxes.append(pts)

    return boxes


def polygon_area(poly):
    return abs(cv2.contourArea(poly))


def normalize_angle_deg(box):
    """
    将 minAreaRect 的角度统一到 [0, 180) 区间。
    """
    rect = cv2.minAreaRect(box)
    (rw, rh) = rect[1]
    angle = rect[2]

    if rw < rh:
        angle = angle + 90

    if angle < 0:
        angle = angle + 180

    if angle >= 180:
        angle = angle - 180

    return float(angle)


def size_category_by_area_ratio(area_ratio):
    """
    按目标面积占整图面积比例进行粗分类。
    """
    if area_ratio < 0.03:
        return "小尺寸"
    elif area_ratio < 0.10:
        return "中等尺寸"
    else:
        return "大尺寸"


def count_group_by_num(n):
    """
    按单张图内包裹数量分组
    """
    if n == 0:
        return "无目标"
    elif n == 1:
        return "单件包裹"
    elif n <= 3:
        return "2-3件包裹"
    elif n <= 6:
        return "4-6件包裹"
    else:
        return "多件复杂场景"


def check_overlap(boxes):
    """
    图片级遮挡/贴靠判断
    """
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            area1 = polygon_area(boxes[i])
            area2 = polygon_area(boxes[j])

            inter, _ = cv2.intersectConvexConvex(
                boxes[i].astype(np.float32),
                boxes[j].astype(np.float32)
            )

            if inter / max(1.0, min(area1, area2)) > 0.15:
                return True

    return False


def collect_images():
    """
    收集所有数据集目录中的图片
    """
    result = []

    for root in DATASET_DIRS:
        root = Path(root)

        image_root = root / "images"
        if not image_root.exists():
            continue

        for p in image_root.rglob("*"):
            if p.suffix.lower() in EXTS:
                result.append((p, root))

    result.sort(key=lambda x: str(x[0]))
    return result


# ============================
# 绘图函数
# ============================

def draw_image_count_plot(image_df, plot_dir):
    order = ["单件包裹", "2-3件包裹", "4-6件包裹", "多件复杂场景", "无目标"]
    counts = image_df["数量类别"].value_counts()
    counts = counts.reindex([x for x in order if x in counts.index], fill_value=0)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.xlabel("图片中的包裹数量类别")
    plt.ylabel("图片数量")
    plt.title("包裹数量分布（图片级）")
    plt.tight_layout()
    plt.savefig(plot_dir / "包裹数量分布.png", dpi=300)
    plt.close()


def draw_target_size_plot(target_df, plot_dir):
    order = ["小尺寸", "中等尺寸", "大尺寸"]
    counts = target_df["尺寸类别"].value_counts()
    counts = counts.reindex(order, fill_value=0)

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.xlabel("目标尺寸类别")
    plt.ylabel("包裹目标数量")
    plt.title("包裹尺寸分布（目标级）")
    plt.tight_layout()
    plt.savefig(plot_dir / "尺寸分布.png", dpi=300)
    plt.close()


def draw_target_angle_plot(target_df, plot_dir):
    bins = np.arange(0, 181, 15)

    plt.figure(figsize=(9, 5))
    plt.hist(target_df["角度(度)"], bins=bins, edgecolor="black")
    plt.xlabel("角度（度）")
    plt.ylabel("包裹目标数量")
    plt.title("包裹角度分布（目标级）")
    plt.xticks(bins)
    plt.tight_layout()
    plt.savefig(plot_dir / "角度分布.png", dpi=300)
    plt.close()


def draw_target_position_plot(target_df, plot_dir):
    plt.figure(figsize=(8, 8))
    plt.scatter(
        target_df["归一化中心X"],
        target_df["归一化中心Y"],
        s=8,
        alpha=0.7
    )

    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel("归一化X位置")
    plt.ylabel("归一化Y位置")
    plt.title("包裹空间位置分布（每个点代表1个包裹）")
    plt.tight_layout()
    plt.savefig(plot_dir / "目标位置分布.png", dpi=300)
    plt.close()


# ============================
# 主程序
# ============================

def main():
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)

    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)

    items = collect_images()

    if MAX_FILES and MAX_FILES > 0:
        items = items[:MAX_FILES]

    image_records = []
    target_records = []

    for img_path, root in tqdm(items, desc="正在分析数据集"):
        img = read_image(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]

        label_path = find_label(img_path, root)
        boxes = read_obb(label_path, w, h)

        n = len(boxes)

        image_records.append({
            "图片路径": str(img_path),
            "数据集来源": root.name,
            "图像宽度": w,
            "图像高度": h,
            "包裹数量": n,
            "数量类别": count_group_by_num(n),
            "是否存在遮挡贴靠": check_overlap(boxes)
        })

        for idx, box in enumerate(boxes):
            center_x = float(np.mean(box[:, 0]))
            center_y = float(np.mean(box[:, 1]))
            area_ratio = float(polygon_area(box) / (w * h))
            angle_deg = normalize_angle_deg(box)

            target_records.append({
                "图片路径": str(img_path),
                "数据集来源": root.name,
                "目标编号": idx,
                "图像宽度": w,
                "图像高度": h,
                "中心X(pixel)": center_x,
                "中心Y(pixel)": center_y,
                "归一化中心X": center_x / w if w > 0 else 0,
                "归一化中心Y": center_y / h if h > 0 else 0,
                "面积占比": area_ratio,
                "尺寸类别": size_category_by_area_ratio(area_ratio),
                "角度(度)": angle_deg
            })

    image_df = pd.DataFrame(image_records)
    target_df = pd.DataFrame(target_records)

    image_df.to_csv(
        output / "数据统计_图片级.csv",
        index=False,
        encoding="utf-8-sig"
    )

    target_df.to_csv(
        output / "数据统计_目标级.csv",
        index=False,
        encoding="utf-8-sig"
    )

    image_count_dist = (
        image_df["数量类别"]
        .value_counts(normalize=True)
        .rename_axis("数量类别")
        .reset_index(name="比例")
    )
    image_count_dist.to_csv(
        output / "包裹数量分布_图片级.csv",
        index=False,
        encoding="utf-8-sig"
    )

    target_size_dist = (
        target_df["尺寸类别"]
        .value_counts(normalize=True)
        .rename_axis("尺寸类别")
        .reset_index(name="比例")
    )
    target_size_dist.to_csv(
        output / "尺寸分布_目标级.csv",
        index=False,
        encoding="utf-8-sig"
    )

    angle_bins = np.arange(0, 181, 15)
    angle_labels = [f"{angle_bins[i]}-{angle_bins[i+1]}°" for i in range(len(angle_bins)-1)]
    target_df["角度区间"] = pd.cut(
        target_df["角度(度)"],
        bins=angle_bins,
        labels=angle_labels,
        right=False,
        include_lowest=True
    )
    angle_dist = (
        target_df["角度区间"]
        .value_counts(normalize=True, sort=False)
        .rename_axis("角度区间")
        .reset_index(name="比例")
    )
    angle_dist.to_csv(
        output / "角度分布_目标级.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "图片总数量": int(len(image_df)),
        "目标总数量": int(len(target_df)),
        "图片级_包裹数量分布": image_df["数量类别"].value_counts(normalize=True).to_dict(),
        "目标级_尺寸分布": target_df["尺寸类别"].value_counts(normalize=True).to_dict(),
        "图片级_遮挡贴靠比例": float(image_df["是否存在遮挡贴靠"].mean()) if len(image_df) > 0 else 0.0
    }

    with open(output / "数据分析总结.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(output / "数据采集建议.txt", "w", encoding="utf-8") as f:
        f.write("包裹数据集分析建议\n")
        f.write("====================\n\n")
        f.write(f"图片总数量：{len(image_df)}\n")
        f.write(f"目标总数量：{len(target_df)}\n\n")
        f.write("说明：\n")
        f.write("1. 包裹数量分布按“图片级”统计。\n")
        f.write("2. 尺寸、角度、位置分布按“目标级”统计，即每个包裹都是一个样本。\n\n")

        if len(image_df) > 0:
            f.write("图片级包裹数量分布：\n")
            for k, v in image_df["数量类别"].value_counts(normalize=True).to_dict().items():
                f.write(f" - {k}: {v:.2%}\n")
            f.write("\n")

        if len(target_df) > 0:
            f.write("目标级尺寸分布：\n")
            for k, v in target_df["尺寸类别"].value_counts(normalize=True).to_dict().items():
                f.write(f" - {k}: {v:.2%}\n")
            f.write("\n")

        if len(image_df) > 0:
            overlap_ratio = float(image_df["是否存在遮挡贴靠"].mean())
            f.write(f"图片级遮挡/贴靠比例：{overlap_ratio:.2%}\n")

    if len(image_df) > 0:
        draw_image_count_plot(image_df, plot_dir)

    if len(target_df) > 0:
        draw_target_size_plot(target_df, plot_dir)
        draw_target_angle_plot(target_df, plot_dir)
        draw_target_position_plot(target_df, plot_dir)

    print("\n分析完成！")
    print("输出目录：", output)
    print("图像分布图目录：", plot_dir)


if __name__ == "__main__":
    main()
