# -*- coding: utf-8 -*-
"""
根据当前多个 YOLO-OBB 数据集目录，自动生成总的：
    train_grouped.txt
    val_grouped.txt

适合后续不断增加：
    images_cvat00_dataset
    images_cvat01_dataset
    images_cvat02_dataset
    ...
    images_cvat20_dataset
    images_cvat21_dataset

目录结构要求：
DATASET_ROOT/
├─ images_cvat00_dataset/
│  ├─ images/
│  │  ├─ train/
│  │  └─ val/
│  └─ labels/
│     ├─ train/
│     └─ val/
├─ images_cvat01_dataset/
│  └─ ...
├─ ...
├─ train_grouped.txt    <- 本脚本生成
├─ val_grouped.txt      <- 本脚本生成
└─ data.yaml            <- 可选自动生成/更新

脚本功能：
1. 自动搜索所有 images_cvat*_dataset；
2. 自动收集每个数据集 images/train 和 images/val；
3. 检查对应 labels/train 和 labels/val；
4. 检查 train / val 是否重复；
5. 生成总 train_grouped.txt / val_grouped.txt；
6. 可选择 Windows 绝对路径、AutoDL 绝对路径或相对路径；
7. 可自动生成/更新总 data.yaml；
8. 打印每个子数据集 train / val 数量和缺失标签情况；
9. 生成 grouped_dataset_summary.txt 方便检查。

注意：
- 本脚本不会移动图片。
- train / val 的物理划分由 images/train、images/val 目录决定。
- 后续增加新的 images_cvatXX_dataset 后，重新运行一次即可。
"""

from pathlib import Path


# =============================================================================
# 1. 用户配置 —— 主要修改这里
# =============================================================================

# -------------------------------------------------------------------------
# 数据集总根目录
# -------------------------------------------------------------------------

# Windows 示例
DATASET_ROOT = Path(r"F:\train_data")

# AutoDL 示例：
# DATASET_ROOT = Path("/root/autodl-tmp")


# -------------------------------------------------------------------------
# 子数据集目录匹配规则
# -------------------------------------------------------------------------

# 会自动寻找：
# images_cvat00_dataset
# images_cvat01_dataset
# images_cvat10_dataset
# images_cvat20_dataset
# ...
DATASET_GLOB = "images_cvat*_dataset"


# -------------------------------------------------------------------------
# 输出文件
# -------------------------------------------------------------------------

TRAIN_GROUPED_FILE = DATASET_ROOT / "train_grouped.txt"
VAL_GROUPED_FILE = DATASET_ROOT / "val_grouped.txt"

SUMMARY_FILE = DATASET_ROOT / "grouped_dataset_summary.txt"

# 是否自动生成/更新总 data.yaml
GENERATE_DATA_YAML = True

DATA_YAML_FILE = DATASET_ROOT / "data.yaml"


# =============================================================================
# 2. grouped txt 中写什么路径
# =============================================================================

# 三种模式：
#
# "absolute"
#     按 DATASET_ROOT 当前真实路径写绝对路径。
#     Windows：
#       H:/train_data/images_cvat00_dataset/images/train/000001.jpg
#     AutoDL：
#       /root/autodl-tmp/images_cvat00_dataset/images/train/000001.jpg
#
# "relative"
#     写相对于 DATASET_ROOT 的路径：
#       images_cvat00_dataset/images/train/000001.jpg
#
# "custom_root"
#     扫描本地数据，但 grouped txt 写成另一台机器的根目录。
#     例如本地 H:\train_data 扫描，
#     但直接生成 AutoDL 路径：
#       /root/autodl-tmp/images_cvat00_dataset/images/train/000001.jpg
#
PATH_MODE = "absolute"

# PATH_MODE="custom_root" 时才使用
CUSTOM_OUTPUT_ROOT = "/root/autodl-tmp"


# =============================================================================
# 3. 标签检查
# =============================================================================

# True：
# 没有对应 label 的图片不写入 grouped txt
#
# False：
# 图片仍写入 grouped txt，但会在终端和 summary 中报告缺失标签
#
# 如果你的数据允许纯背景负样本（没有 txt），应设 False。
REQUIRE_LABEL = False

# 是否把缺失 label 当成警告显示
WARN_MISSING_LABEL = True


# =============================================================================
# 4. 图片格式
# =============================================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# =============================================================================
# 5. data.yaml 设置
# =============================================================================

# 你的任务目前是单类别 parcel。
# 如果以后增加类别，在这里修改。
CLASS_NAMES = {
    0: "parcel",
}

# data.yaml 中 train / val 默认直接引用 grouped txt。
#
# 推荐：
#   train: train_grouped.txt
#   val: val_grouped.txt
#
# 这样整个 DATASET_ROOT 搬家后，只需要 grouped txt 路径内容正确即可。
YAML_USE_RELATIVE_GROUPED_PATH = True


# =============================================================================
# 6. 基础函数
# =============================================================================

def find_dataset_dirs():
    """自动寻找所有 images_cvat*_dataset。"""
    dataset_dirs = sorted(
        [
            p
            for p in DATASET_ROOT.glob(DATASET_GLOB)
            if p.is_dir()
        ],
        key=lambda p: p.name.lower(),
    )

    if not dataset_dirs:
        raise RuntimeError(
            f"没有找到数据集目录：\n"
            f"DATASET_ROOT = {DATASET_ROOT}\n"
            f"匹配规则 = {DATASET_GLOB}"
        )

    return dataset_dirs


def collect_images(image_dir: Path):
    """递归读取某个 train/val 图片目录。"""
    if not image_dir.exists():
        return []

    return sorted(
        [
            p.resolve()
            for p in image_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.as_posix().lower(),
    )


def image_to_label_path(image_path: Path):
    """
    根据当前标准目录：
        .../images/train/a.jpg
        .../images/val/a.jpg

    自动得到：
        .../labels/train/a.txt
        .../labels/val/a.txt
    """
    parts = list(image_path.parts)

    image_index = None

    # 从后往前找独立的 images 目录
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            image_index = i
            break

    if image_index is None:
        raise ValueError(
            f"图片路径中找不到 images 目录：{image_path}"
        )

    parts[image_index] = "labels"

    return Path(*parts).with_suffix(".txt")


def make_grouped_path(image_path: Path):
    """
    根据 PATH_MODE 决定 grouped txt 里的路径写法。
    """
    if PATH_MODE == "absolute":
        return image_path.resolve().as_posix()

    if PATH_MODE == "relative":
        return (
            image_path.resolve()
            .relative_to(DATASET_ROOT.resolve())
            .as_posix()
        )

    if PATH_MODE == "custom_root":
        relative = (
            image_path.resolve()
            .relative_to(DATASET_ROOT.resolve())
            .as_posix()
        )

        return (
            CUSTOM_OUTPUT_ROOT.rstrip("/\\")
            + "/"
            + relative
        ).replace("\\", "/")

    raise ValueError(
        "PATH_MODE 只能是："
        "'absolute' / 'relative' / 'custom_root'"
    )


def write_lines(file_path: Path, values):
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path.write_text(
        "\n".join(values)
        + ("\n" if values else ""),
        encoding="utf-8",
    )


# =============================================================================
# 7. data.yaml
# =============================================================================

def make_yaml_text():
    if YAML_USE_RELATIVE_GROUPED_PATH:
        train_value = TRAIN_GROUPED_FILE.name
        val_value = VAL_GROUPED_FILE.name
    else:
        train_value = TRAIN_GROUPED_FILE.resolve().as_posix()
        val_value = VAL_GROUPED_FILE.resolve().as_posix()

    lines = [
        f"train: {train_value}",
        f"val: {val_value}",
        "",
        "names:",
    ]

    for class_id, class_name in sorted(
        CLASS_NAMES.items()
    ):
        lines.append(
            f"  {class_id}: {class_name}"
        )

    lines.append("")

    return "\n".join(lines)


def write_data_yaml():
    DATA_YAML_FILE.write_text(
        make_yaml_text(),
        encoding="utf-8",
    )


# =============================================================================
# 8. 主程序
# =============================================================================

def main():
    print("=" * 78)
    print("自动生成 YOLO-OBB grouped train / val 列表")
    print("=" * 78)
    print(f"DATASET_ROOT : {DATASET_ROOT}")
    print(f"匹配规则     : {DATASET_GLOB}")
    print(f"PATH_MODE    : {PATH_MODE}")

    if PATH_MODE == "custom_root":
        print(
            f"输出路径根目录: {CUSTOM_OUTPUT_ROOT}"
        )

    print(f"REQUIRE_LABEL: {REQUIRE_LABEL}")
    print("=" * 78)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"DATASET_ROOT 不存在：{DATASET_ROOT}"
        )

    dataset_dirs = find_dataset_dirs()

    print(
        f"\n找到 {len(dataset_dirs)} 个子数据集："
    )

    for dataset_dir in dataset_dirs:
        print(f"  {dataset_dir.name}")

    all_train_images = []
    all_val_images = []

    missing_train_labels = []
    missing_val_labels = []

    dataset_stats = []

    # -------------------------------------------------------------------------
    # 遍历每个 CVAT 数据集
    # -------------------------------------------------------------------------
    for dataset_index, dataset_dir in enumerate(
        dataset_dirs,
        start=1,
    ):
        train_image_dir = (
            dataset_dir / "images" / "train"
        )

        val_image_dir = (
            dataset_dir / "images" / "val"
        )

        train_images = collect_images(
            train_image_dir
        )

        val_images = collect_images(
            val_image_dir
        )

        valid_train = []
        valid_val = []

        train_missing = 0
        val_missing = 0

        # train
        for image_path in train_images:
            label_path = image_to_label_path(
                image_path
            )

            if not label_path.exists():
                train_missing += 1

                missing_train_labels.append(
                    (
                        image_path,
                        label_path,
                    )
                )

                if REQUIRE_LABEL:
                    continue

            valid_train.append(
                image_path
            )

        # val
        for image_path in val_images:
            label_path = image_to_label_path(
                image_path
            )

            if not label_path.exists():
                val_missing += 1

                missing_val_labels.append(
                    (
                        image_path,
                        label_path,
                    )
                )

                if REQUIRE_LABEL:
                    continue

            valid_val.append(
                image_path
            )

        all_train_images.extend(
            valid_train
        )

        all_val_images.extend(
            valid_val
        )

        dataset_stats.append({
            "name": dataset_dir.name,
            "train": len(valid_train),
            "val": len(valid_val),
            "train_missing_label": train_missing,
            "val_missing_label": val_missing,
        })

        print(
            f"[{dataset_index:>2}/{len(dataset_dirs)}] "
            f"{dataset_dir.name:<28} "
            f"train={len(valid_train):<5} "
            f"val={len(valid_val):<5} "
            f"missing_label="
            f"{train_missing + val_missing}"
        )

    # -------------------------------------------------------------------------
    # 防止重复
    # -------------------------------------------------------------------------
    train_set = {
        p.resolve()
        for p in all_train_images
    }

    val_set = {
        p.resolve()
        for p in all_val_images
    }

    overlap = (
        train_set
        & val_set
    )

    if overlap:
        print()
        print(
            "检测到 train / val 物理路径重复："
            f"{len(overlap)}"
        )

        for p in sorted(overlap)[:20]:
            print(f"  {p}")

        raise RuntimeError(
            "train / val 存在重复图片，"
            "已停止生成 grouped 文件。"
        )

    # 再检查 grouped 路径字符串是否重复
    train_grouped = [
        make_grouped_path(p)
        for p in all_train_images
    ]

    val_grouped = [
        make_grouped_path(p)
        for p in all_val_images
    ]

    train_grouped_set = set(
        train_grouped
    )

    val_grouped_set = set(
        val_grouped
    )

    grouped_overlap = (
        train_grouped_set
        & val_grouped_set
    )

    if grouped_overlap:
        raise RuntimeError(
            "生成后的 train_grouped / val_grouped "
            f"存在 {len(grouped_overlap)} 个重复路径。"
        )

    # -------------------------------------------------------------------------
    # 写 grouped
    # -------------------------------------------------------------------------
    write_lines(
        TRAIN_GROUPED_FILE,
        train_grouped,
    )

    write_lines(
        VAL_GROUPED_FILE,
        val_grouped,
    )

    # -------------------------------------------------------------------------
    # data.yaml
    # -------------------------------------------------------------------------
    if GENERATE_DATA_YAML:
        write_data_yaml()

    # -------------------------------------------------------------------------
    # summary
    # -------------------------------------------------------------------------
    summary_lines = [
        "YOLO-OBB grouped 数据集汇总",
        "=" * 72,
        "",
        f"DATASET_ROOT: {DATASET_ROOT}",
        f"PATH_MODE: {PATH_MODE}",
    ]

    if PATH_MODE == "custom_root":
        summary_lines.append(
            f"CUSTOM_OUTPUT_ROOT: {CUSTOM_OUTPUT_ROOT}"
        )

    summary_lines.extend([
        "",
        f"子数据集数量: {len(dataset_dirs)}",
        f"train 图片总数: {len(all_train_images)}",
        f"val 图片总数: {len(all_val_images)}",
        f"train/val 重复: {len(overlap)}",
        f"train 缺失 label: {len(missing_train_labels)}",
        f"val 缺失 label: {len(missing_val_labels)}",
        "",
        "各子数据集：",
        "-" * 72,
    ])

    for stat in dataset_stats:
        summary_lines.append(
            f"{stat['name']}: "
            f"train={stat['train']}, "
            f"val={stat['val']}, "
            f"train_missing_label="
            f"{stat['train_missing_label']}, "
            f"val_missing_label="
            f"{stat['val_missing_label']}"
        )

    if missing_train_labels:
        summary_lines.extend([
            "",
            "train 缺失 label 示例：",
            "-" * 72,
        ])

        for image_path, label_path in (
            missing_train_labels[:50]
        ):
            summary_lines.append(
                f"image: {image_path}"
            )
            summary_lines.append(
                f"label: {label_path}"
            )

    if missing_val_labels:
        summary_lines.extend([
            "",
            "val 缺失 label 示例：",
            "-" * 72,
        ])

        for image_path, label_path in (
            missing_val_labels[:50]
        ):
            summary_lines.append(
                f"image: {image_path}"
            )
            summary_lines.append(
                f"label: {label_path}"
            )

    # SUMMARY_FILE.write_text(
    #     "\n".join(summary_lines) + "\n",
    #     encoding="utf-8-sig",
    # )

    # -------------------------------------------------------------------------
    # 结果
    # -------------------------------------------------------------------------
    print()
    print("=" * 78)
    print("生成完成")
    print("=" * 78)
    print(
        f"train_grouped: {TRAIN_GROUPED_FILE}"
    )
    print(
        f"  图片数: {len(train_grouped)}"
    )

    print(
        f"val_grouped:   {VAL_GROUPED_FILE}"
    )
    print(
        f"  图片数: {len(val_grouped)}"
    )

    if GENERATE_DATA_YAML:
        print(
            f"data.yaml:     {DATA_YAML_FILE}"
        )

    # print(
    #     f"summary:       {SUMMARY_FILE}"
    # )

    print()
    print(
        f"train 缺失 label: "
        f"{len(missing_train_labels)}"
    )
    print(
        f"val 缺失 label:   "
        f"{len(missing_val_labels)}"
    )

    if WARN_MISSING_LABEL and (
        missing_train_labels
        or missing_val_labels
    ):
        print()
        print(
            "[提示] 检测到无标签图片。"
        )
        print(
            "如果这些是纯背景负样本，可以保留 REQUIRE_LABEL=False。"
        )
        print(
            "如果所有图片都应该包含包裹，建议检查是否存在漏标。"
        )

    print()
    print(
        "以后新增 images_cvatXX_dataset 后，"
        "只需要重新运行本脚本即可刷新 grouped 文件。"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
