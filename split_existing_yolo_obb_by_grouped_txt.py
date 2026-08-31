# -*- coding: utf-8 -*-
"""
按照 train_grouped.txt / val_grouped.txt，把多个 YOLO-OBB 数据集真正整理成：

images_cvatXX_dataset/
├─ images/
│  ├─ train/
│  └─ val/
├─ labels/
│  ├─ train/
│  └─ val/
├─ train.txt
├─ val.txt
└─ data.yaml

用途
----
你现在所有图片/标签都在 images/train 和 labels/train 中，
但 train_grouped.txt、val_grouped.txt 已经决定了哪些图片属于训练集、哪些属于验证集。

本脚本会：
1. 读取 train_grouped.txt 和 val_grouped.txt；
2. 不依赖 txt 中原来的盘符前缀，只识别：
      images_cvatXX_dataset/images/train/xxx.jpg
3. train 列表里的文件保证位于：
      images/train
      labels/train
4. val 列表里的文件移动到：
      images/val
      labels/val
5. 自动重写全局 train_grouped.txt / val_grouped.txt 中的路径；
6. 为每个 images_cvatXX_dataset 重新生成 train.txt / val.txt；
7. 保留 data.yaml 原有 names 等内容，只更新 train: 和 val:；
8. 生成 move_manifest.csv，记录每个文件最终去了哪里；
9. 操作前自动备份 txt / data.yaml 元数据。

注意：
- 默认 MOVE_FILES=True：移动，不复制。
- 不会因为某张图片没有 label 就停止；无标签图片只移动图片。
- 如果你要在 AutoDL 上运行，只需要把 DATASET_ROOT 改成：
      Path("/root/autodl-tmp")
"""

import csv
import re
import shutil
from datetime import datetime
from pathlib import Path


# =============================================================================
# 1. 用户配置 —— 主要修改这里
# =============================================================================

# 所有 images_cvatXX_dataset 所在的根目录
#
# Windows 本地：
DATASET_ROOT = Path(r"H:\train_data")
#
# AutoDL 示例：
# DATASET_ROOT = Path("/root/autodl-tmp")


# 已经划分好的全局 train / val 列表
TRAIN_LIST_FILE = Path(r"H:\train_data\gpt_dataset\train_grouped.txt")
VAL_LIST_FILE = Path(r"H:\train_data\gpt_dataset\val_grouped.txt")

# 如果 train_grouped.txt / val_grouped.txt 不在 DATASET_ROOT，
# 改成它们真实的位置即可。


# -----------------------------------------------------------------------------
# 文件处理模式
# -----------------------------------------------------------------------------

# True  = 真正移动文件，最终不会在 train/val 各保留一份
# False = 复制文件，原 train 目录中的 val 图片仍会保留
#
# 你的需求是“真正划分目录”，建议保持 True。
MOVE_FILES = True

# True  = 只检查并打印计划，不真正移动/改写文件
# False = 真正执行
#
# 第一次运行如果特别谨慎，可以先设 True 看检查结果；
# 确认后再改 False。
DRY_RUN = False

# 自动备份原来的：
# train_grouped.txt / val_grouped.txt / 每个数据集的 train.txt / val.txt / data.yaml
BACKUP_METADATA = False


# -----------------------------------------------------------------------------
# txt 路径写法
# -----------------------------------------------------------------------------

# 全局 train_grouped.txt / val_grouped.txt 写绝对路径。
# 推荐 True，和你之前的使用方式一致。
GLOBAL_LIST_USE_ABSOLUTE_PATHS = True

# 每个 images_cvatXX_dataset 内部的 train.txt / val.txt：
#
# False -> 写：
#   images/train/000001.jpg
#   images/val/000010.jpg
#
# True -> 写完整绝对路径。
#
# 推荐 False，更方便整个数据集搬到别的电脑/AutoDL。
LOCAL_LIST_USE_ABSOLUTE_PATHS = False


# -----------------------------------------------------------------------------
# data.yaml
# -----------------------------------------------------------------------------

# 是否自动更新每个数据集根目录下的 data.yaml
UPDATE_DATA_YAML = True

# 如果某个数据集没有 data.yaml，是否自动创建一个最小 data.yaml
# 不知道类别 names 时，不建议自动凭空创建，所以默认 False。
CREATE_DATA_YAML_IF_MISSING = False


# -----------------------------------------------------------------------------
# 安全检查
# -----------------------------------------------------------------------------

# 如果目标文件已存在：
# "error"     -> 直接报错停止，最安全
# "same_skip" -> 若大小相同则跳过，否则报错
EXISTING_DESTINATION_POLICY = "same_skip"

# 是否要求 train/val 中每个图片都存在
REQUIRE_IMAGE = True

# 是否要求每个图片都有 label
# 你之前允许无标签图片，因此这里默认 False。
REQUIRE_LABEL = False


# =============================================================================
# 2. 常量
# =============================================================================

DATASET_PATTERN = re.compile(
    r"(images_cvat[^/\\]+_dataset)",
    flags=re.IGNORECASE,
)

IMAGE_MARKER_PATTERN = re.compile(
    r"(images_cvat[^/\\]+_dataset)[/\\]+images[/\\]+(?:train|val)[/\\]+(.+)$",
    flags=re.IGNORECASE,
)


# =============================================================================
# 3. 基础函数
# =============================================================================

def normalize_slash(text: str) -> str:
    return text.replace("\\", "/")


def read_nonempty_lines(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"列表文件不存在：{path}")

    lines = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip().strip('"').strip("'")

            if text:
                lines.append((line_no, text))

    return lines


def parse_list_entry(text: str):
    """
    从任意前缀路径中提取：
      dataset_name
      图片在 images/train 或 images/val 后面的相对路径

    例如：
      H:/train_data/images_cvat00_dataset/images/train/a/001.jpg
    或：
      /root/autodl-tmp/images_cvat00_dataset/images/train/a/001.jpg

    都会解析成：
      dataset_name = images_cvat00_dataset
      relative_image = a/001.jpg
    """
    normalized = normalize_slash(text)

    match = IMAGE_MARKER_PATTERN.search(normalized)

    if not match:
        raise ValueError(
            "无法解析路径，要求路径中包含：\n"
            "images_cvatXX_dataset/images/train/... 或 images/val/...\n"
            f"实际：{text}"
        )

    dataset_name = match.group(1)
    relative_image = Path(match.group(2))

    return dataset_name, relative_image


def label_relative_from_image(relative_image: Path) -> Path:
    return relative_image.with_suffix(".txt")


def image_path(dataset_name: str, split: str, relative_image: Path) -> Path:
    return (
        DATASET_ROOT
        / dataset_name
        / "images"
        / split
        / relative_image
    )


def label_path(dataset_name: str, split: str, relative_image: Path) -> Path:
    return (
        DATASET_ROOT
        / dataset_name
        / "labels"
        / split
        / label_relative_from_image(relative_image)
    )


def ensure_dir(path: Path):
    if not DRY_RUN:
        path.mkdir(parents=True, exist_ok=True)


def backup_file(path: Path, backup_root: Path):
    if not BACKUP_METADATA:
        return

    if not path.exists():
        return

    try:
        rel = path.resolve().relative_to(DATASET_ROOT.resolve())
        dst = backup_root / rel
    except Exception:
        dst = backup_root / "_external" / path.name

    if DRY_RUN:
        print(f"[备份计划] {path} -> {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)


def move_or_copy(src: Path, dst: Path):
    """
    返回：
      "moved"
      "copied"
      "already_ok"
      "missing"
    """
    if dst.exists():
        if src.exists() and src.resolve() == dst.resolve():
            return "already_ok"

        if EXISTING_DESTINATION_POLICY == "same_skip":
            if src.exists() and src.stat().st_size == dst.stat().st_size:
                return "already_ok"

        raise FileExistsError(
            f"目标文件已存在，为避免覆盖已停止：\n{dst}"
        )

    if not src.exists():
        return "missing"

    if DRY_RUN:
        return "moved" if MOVE_FILES else "copied"

    dst.parent.mkdir(parents=True, exist_ok=True)

    if MOVE_FILES:
        shutil.move(str(src), str(dst))
        return "moved"

    shutil.copy2(src, dst)
    return "copied"


def locate_current_file(
    dataset_name: str,
    desired_split: str,
    relative_image: Path,
    is_label: bool,
):
    """
    因为脚本允许重复运行，所以文件可能已经在目标 split，
    也可能仍在另一个 split。

    返回：
      current_path, desired_path
    """
    folder = "labels" if is_label else "images"
    relative = (
        label_relative_from_image(relative_image)
        if is_label
        else relative_image
    )

    desired = (
        DATASET_ROOT
        / dataset_name
        / folder
        / desired_split
        / relative
    )

    other_split = "val" if desired_split == "train" else "train"

    other = (
        DATASET_ROOT
        / dataset_name
        / folder
        / other_split
        / relative
    )

    if desired.exists():
        return desired, desired

    if other.exists():
        return other, desired

    return desired, desired


# =============================================================================
# 4. 读取并检查 train / val 列表
# =============================================================================

def load_split_entries():
    train_raw = read_nonempty_lines(TRAIN_LIST_FILE)
    val_raw = read_nonempty_lines(VAL_LIST_FILE)

    train_entries = []
    val_entries = []

    for line_no, text in train_raw:
        dataset_name, rel = parse_list_entry(text)
        train_entries.append({
            "split": "train",
            "line_no": line_no,
            "original": text,
            "dataset": dataset_name,
            "relative": rel,
        })

    for line_no, text in val_raw:
        dataset_name, rel = parse_list_entry(text)
        val_entries.append({
            "split": "val",
            "line_no": line_no,
            "original": text,
            "dataset": dataset_name,
            "relative": rel,
        })

    # 用 dataset + relative 判断 train/val 是否重叠
    train_keys = {
        (x["dataset"].lower(), x["relative"].as_posix().lower())
        for x in train_entries
    }

    val_keys = {
        (x["dataset"].lower(), x["relative"].as_posix().lower())
        for x in val_entries
    }

    overlap = train_keys & val_keys

    if overlap:
        examples = list(sorted(overlap))[:10]

        raise RuntimeError(
            "train_grouped.txt 与 val_grouped.txt 存在重复图片！\n"
            f"重复数量：{len(overlap)}\n"
            f"示例：{examples}"
        )

    return train_entries, val_entries


# =============================================================================
# 5. 移动 images / labels
# =============================================================================

def process_entry(entry, manifest_rows):
    split = entry["split"]
    dataset_name = entry["dataset"]
    relative_image = entry["relative"]

    # -------------------------------------------------------------------------
    # 图片
    # -------------------------------------------------------------------------
    current_image, desired_image = locate_current_file(
        dataset_name=dataset_name,
        desired_split=split,
        relative_image=relative_image,
        is_label=False,
    )

    if current_image == desired_image and desired_image.exists():
        image_status = "already_ok"
    else:
        image_status = move_or_copy(
            current_image,
            desired_image,
        )

    if image_status == "missing" and REQUIRE_IMAGE:
        raise FileNotFoundError(
            "列表中的图片没有找到：\n"
            f"数据集：{dataset_name}\n"
            f"split：{split}\n"
            f"相对路径：{relative_image}\n"
            f"检查过：{current_image}\n"
            f"目标：{desired_image}"
        )

    # -------------------------------------------------------------------------
    # 标签
    # -------------------------------------------------------------------------
    current_label, desired_label = locate_current_file(
        dataset_name=dataset_name,
        desired_split=split,
        relative_image=relative_image,
        is_label=True,
    )

    if current_label == desired_label and desired_label.exists():
        label_status = "already_ok"
    elif current_label.exists():
        label_status = move_or_copy(
            current_label,
            desired_label,
        )
    else:
        label_status = "missing"

    if label_status == "missing" and REQUIRE_LABEL:
        raise FileNotFoundError(
            "图片对应标签没有找到：\n"
            f"{desired_label}"
        )

    manifest_rows.append({
        "split": split,
        "dataset": dataset_name,
        "relative_image": relative_image.as_posix(),
        "image_destination": desired_image.as_posix(),
        "image_status": image_status,
        "label_destination": desired_label.as_posix(),
        "label_status": label_status,
    })

    return image_status, label_status


# =============================================================================
# 6. 重写全局 grouped txt
# =============================================================================

def write_global_list(entries, split, output_path):
    lines = []

    for entry in entries:
        dataset_name = entry["dataset"]
        relative_image = entry["relative"]

        final_image = image_path(
            dataset_name,
            split,
            relative_image,
        )

        if GLOBAL_LIST_USE_ABSOLUTE_PATHS:
            value = final_image.resolve().as_posix()
        else:
            value = final_image.relative_to(DATASET_ROOT).as_posix()

        lines.append(value)

    if DRY_RUN:
        print(f"[写入计划] {output_path}：{len(lines)} 行")
        return

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 7. 生成每个数据集自己的 train.txt / val.txt
# =============================================================================

def local_list_value(dataset_root: Path, image: Path):
    if LOCAL_LIST_USE_ABSOLUTE_PATHS:
        return image.resolve().as_posix()

    return image.relative_to(dataset_root).as_posix()


def write_dataset_lists(dataset_name, train_entries, val_entries):
    dataset_root = DATASET_ROOT / dataset_name

    dataset_train = [
        x for x in train_entries
        if x["dataset"].lower() == dataset_name.lower()
    ]

    dataset_val = [
        x for x in val_entries
        if x["dataset"].lower() == dataset_name.lower()
    ]

    train_txt = dataset_root / "train.txt"
    val_txt = dataset_root / "val.txt"

    train_lines = [
        local_list_value(
            dataset_root,
            image_path(
                dataset_name,
                "train",
                x["relative"],
            ),
        )
        for x in dataset_train
    ]

    val_lines = [
        local_list_value(
            dataset_root,
            image_path(
                dataset_name,
                "val",
                x["relative"],
            ),
        )
        for x in dataset_val
    ]

    if DRY_RUN:
        print(
            f"[写入计划] {dataset_name}: "
            f"train.txt={len(train_lines)}, "
            f"val.txt={len(val_lines)}"
        )
        return

    dataset_root.mkdir(parents=True, exist_ok=True)

    train_txt.write_text(
        "\n".join(train_lines) + ("\n" if train_lines else ""),
        encoding="utf-8",
    )

    val_txt.write_text(
        "\n".join(val_lines) + ("\n" if val_lines else ""),
        encoding="utf-8",
    )


# =============================================================================
# 8. 更新 data.yaml
# =============================================================================

def update_yaml_train_val(yaml_path: Path):
    """
    只修改/补充：
        train: train.txt
        val: val.txt

    其他内容，例如 names / nc / path 等全部尽量原样保留。
    """
    if not yaml_path.exists():
        if not CREATE_DATA_YAML_IF_MISSING:
            print(f"[提示] 没有 data.yaml，跳过：{yaml_path}")
            return

        content = (
            "train: train.txt\n"
            "val: val.txt\n"
        )

        if DRY_RUN:
            print(f"[创建计划] {yaml_path}")
            return

        yaml_path.write_text(
            content,
            encoding="utf-8",
        )
        return

    text = yaml_path.read_text(
        encoding="utf-8-sig"
    )

    lines = text.splitlines()

    new_lines = []
    found_train = False
    found_val = False

    for line in lines:
        stripped = line.lstrip()
        indent = line[:len(line) - len(stripped)]

        if stripped.startswith("train:"):
            new_lines.append(
                f"{indent}train: train.txt"
            )
            found_train = True

        elif stripped.startswith("val:"):
            new_lines.append(
                f"{indent}val: val.txt"
            )
            found_val = True

        else:
            new_lines.append(line)

    if not found_train:
        new_lines.append("train: train.txt")

    if not found_val:
        new_lines.append("val: val.txt")

    if DRY_RUN:
        print(f"[更新计划] {yaml_path}: train/val")
        return

    yaml_path.write_text(
        "\n".join(new_lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# 9. 清理空目录
# =============================================================================

def remove_empty_directories(root: Path):
    if not root.exists() or DRY_RUN:
        return

    dirs = sorted(
        [p for p in root.rglob("*") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    )

    for directory in dirs:
        try:
            directory.rmdir()
        except OSError:
            pass


# =============================================================================
# 10. 主程序
# =============================================================================

def main():
    print("=" * 78)
    print("按照 train_grouped / val_grouped 整理 YOLO-OBB 数据集")
    print("=" * 78)
    print(f"DATASET_ROOT : {DATASET_ROOT}")
    print(f"TRAIN_LIST   : {TRAIN_LIST_FILE}")
    print(f"VAL_LIST     : {VAL_LIST_FILE}")
    print(f"MOVE_FILES   : {MOVE_FILES}")
    print(f"DRY_RUN      : {DRY_RUN}")
    print("=" * 78)

    train_entries, val_entries = load_split_entries()

    print(f"train 图片数：{len(train_entries)}")
    print(f"val 图片数：  {len(val_entries)}")
    print(f"总图片数：    {len(train_entries) + len(val_entries)}")

    dataset_names = sorted({
        x["dataset"]
        for x in train_entries + val_entries
    })

    print(f"涉及数据集：  {len(dataset_names)}")
    for name in dataset_names:
        n_train = sum(
            x["dataset"].lower() == name.lower()
            for x in train_entries
        )
        n_val = sum(
            x["dataset"].lower() == name.lower()
            for x in val_entries
        )
        print(
            f"  {name:<28} "
            f"train={n_train:<5} val={n_val:<5}"
        )

    # -------------------------------------------------------------------------
    # 备份元数据
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = (
        DATASET_ROOT
        / f"_split_backup_{timestamp}"
    )

    if BACKUP_METADATA:
        print()
        print("正在备份 txt / data.yaml ...")

        backup_file(
            TRAIN_LIST_FILE,
            backup_root,
        )

        backup_file(
            VAL_LIST_FILE,
            backup_root,
        )

        for dataset_name in dataset_names:
            dataset_root = DATASET_ROOT / dataset_name

            for name in [
                "train.txt",
                "val.txt",
                "data.yaml",
            ]:
                backup_file(
                    dataset_root / name,
                    backup_root,
                )

    # -------------------------------------------------------------------------
    # 整理文件
    # -------------------------------------------------------------------------
    print()
    print("开始整理 images / labels ...")

    manifest_rows = []

    all_entries = train_entries + val_entries
    total = len(all_entries)

    image_missing = 0
    label_missing = 0

    for index, entry in enumerate(
        all_entries,
        start=1,
    ):
        image_status, label_status = process_entry(
            entry,
            manifest_rows,
        )

        if image_status == "missing":
            image_missing += 1

        if label_status == "missing":
            label_missing += 1

        if (
            index % 50 == 0
            or index == total
        ):
            print(
                f"[{index:>5}/{total}] "
                f"{index / total * 100:6.2f}% | "
                f"missing_image={image_missing} | "
                f"missing_label={label_missing}"
            )

    # -------------------------------------------------------------------------
    # 重写 grouped txt
    # -------------------------------------------------------------------------
    print()
    print("重写全局 train_grouped / val_grouped ...")

    write_global_list(
        train_entries,
        "train",
        TRAIN_LIST_FILE,
    )

    write_global_list(
        val_entries,
        "val",
        VAL_LIST_FILE,
    )

    # -------------------------------------------------------------------------
    # 每个数据集 train.txt / val.txt + data.yaml
    # -------------------------------------------------------------------------
    print()
    print("更新各数据集 train.txt / val.txt / data.yaml ...")

    for dataset_name in dataset_names:
        dataset_root = DATASET_ROOT / dataset_name

        ensure_dir(
            dataset_root / "images" / "train"
        )
        ensure_dir(
            dataset_root / "images" / "val"
        )
        ensure_dir(
            dataset_root / "labels" / "train"
        )
        ensure_dir(
            dataset_root / "labels" / "val"
        )

        write_dataset_lists(
            dataset_name,
            train_entries,
            val_entries,
        )

        if UPDATE_DATA_YAML:
            update_yaml_train_val(
                dataset_root / "data.yaml"
            )

    # -------------------------------------------------------------------------
    # manifest
    # -------------------------------------------------------------------------
    manifest_path = (
        DATASET_ROOT
        / "split_move_manifest.csv"
    )

    if not DRY_RUN:
        with manifest_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "split",
                    "dataset",
                    "relative_image",
                    "image_destination",
                    "image_status",
                    "label_destination",
                    "label_status",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

    # 清理移动后遗留的空子目录
    if MOVE_FILES:
        for dataset_name in dataset_names:
            remove_empty_directories(
                DATASET_ROOT
                / dataset_name
                / "images"
            )
            remove_empty_directories(
                DATASET_ROOT
                / dataset_name
                / "labels"
            )

    print()
    print("=" * 78)
    print("处理完成" if not DRY_RUN else "DRY RUN 检查完成，未实际修改文件")
    print("=" * 78)
    print(f"train：{len(train_entries)}")
    print(f"val：  {len(val_entries)}")
    print(f"缺失图片：{image_missing}")
    print(f"缺失标签：{label_missing}")

    if not DRY_RUN:
        print(f"操作记录：{manifest_path}")

        if BACKUP_METADATA:
            print(f"元数据备份：{backup_root}")

    print()
    print("最终目录示例：")
    print(
        "images_cvat00_dataset/\n"
        "├─ images/\n"
        "│  ├─ train/\n"
        "│  └─ val/\n"
        "├─ labels/\n"
        "│  ├─ train/\n"
        "│  └─ val/\n"
        "├─ train.txt\n"
        "├─ val.txt\n"
        "└─ data.yaml"
    )


if __name__ == "__main__":
    main()
