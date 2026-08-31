# -*- coding: utf-8 -*-
"""
清理每个 images_cvat*_dataset 内部的 train.txt / val.txt
以及对应的孤立 labels。

IMPORTANT：
- 本脚本【不会读取、不会修改】根目录的 train_grouped.txt / val_grouped.txt。
- 只处理每个子数据集自己的：
      images_cvatXX_dataset/train.txt
      images_cvatXX_dataset/val.txt
      images_cvatXX_dataset/labels/train/*.txt
      images_cvatXX_dataset/labels/val/*.txt

主要处理逻辑：

1) 图片已经被删掉，但：
   - 对应 label 还在
   - 子数据集 train.txt / val.txt 里还写着这张图

   处理：
   - 从该子数据集 train.txt / val.txt 中删除这一行
   - 删除（或备份后移走）对应残余 label

2) 图片还在，但没有对应 label：
   - 可能是合法的纯背景负样本

   处理：
   - 图片保留
   - train.txt / val.txt 中对应行也保留
   - 不因为 label 不存在而删除该样本

本脚本不会：
- 修改 train_grouped.txt / val_grouped.txt
- 自动把“未写入 train.txt/val.txt 的图片”补进去
- 检查任何 OBB 坐标、类别、格式或数值合法性
- 删除图片
- 重新划分 train / val

建议：
第一次运行 DRY_RUN=True 看结果；
确认后改成 DRY_RUN=False 正式清理。
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


# =============================================================================
# 1. 用户配置
# =============================================================================

# 所有 images_cvat*_dataset 所在根目录
DATASET_ROOT = Path(r"H:\train_data")

# 自动匹配：
# images_cvat00_dataset
# images_cvat01_dataset
# images_cvat02_dataset
# ...
DATASET_GLOB = "images_cvat*_dataset"

# True  = 只显示将做什么，不真正改文件
# False = 真正执行
DRY_RUN = False

# 是否备份每个子数据集原来的 train.txt / val.txt
BACKUP_SUBDATASET_TXT = False

# 对“图片已删除但 label 还存在”的孤立 label：
# True  = 移到备份目录，推荐
# False = 直接永久删除
BACKUP_ORPHAN_LABELS = False

# 是否清理操作后产生的空 labels 子目录
REMOVE_EMPTY_DIRS = False

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}


# =============================================================================
# 2. 路径工具
# =============================================================================

def find_dataset_dirs():
    dataset_dirs = sorted(
        [
            p for p in DATASET_ROOT.glob(DATASET_GLOB)
            if p.is_dir()
        ],
        key=lambda p: p.name.lower(),
    )

    if not dataset_dirs:
        raise RuntimeError(
            f"没有找到：{DATASET_ROOT / DATASET_GLOB}"
        )

    return dataset_dirs


def normalize_txt_path(text: str) -> str:
    return text.strip().strip('"').strip("'").replace("\\", "/")


def resolve_txt_image_path(
    text: str,
    dataset_dir: Path,
) -> Path:
    """
    兼容子数据集 train.txt / val.txt 中常见写法：

    images/train/000001.jpg
    ./images/train/000001.jpg
    H:/train_data/images_cvat00_dataset/images/train/000001.jpg
    /root/autodl-tmp/images_cvat00_dataset/images/train/000001.jpg

    即使 txt 里还是旧机器前缀，也会尽量映射到当前 dataset_dir。
    """
    normalized = normalize_txt_path(text)

    p = Path(normalized)

    # 当前机器上的绝对路径
    if p.is_absolute() and p.exists():
        return p.resolve()

    # 相对子数据集根目录
    candidate = dataset_dir / Path(normalized)

    if candidate.exists():
        return candidate.resolve()

    # txt 可能是旧 Windows / AutoDL 绝对路径
    # 从 /images/train/ 或 /images/val/ 开始截取
    lower = normalized.lower()

    markers = [
        "/images/train/",
        "/images/val/",
    ]

    for marker in markers:
        pos = lower.find(marker)

        if pos >= 0:
            rel = normalized[pos + 1:]  # 去掉最前面的 /
            return (dataset_dir / Path(rel)).resolve()

    # 即使不存在，也返回一个合理候选供后续判断
    return candidate.resolve()


def image_to_label_path(image_path: Path) -> Path:
    """
    .../images/train/a.jpg -> .../labels/train/a.txt
    .../images/val/a.jpg   -> .../labels/val/a.txt
    """
    parts = list(image_path.parts)
    idx = None

    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            idx = i
            break

    if idx is None:
        raise ValueError(
            f"图片路径中找不到 images 目录：{image_path}"
        )

    parts[idx] = "labels"

    return Path(*parts).with_suffix(".txt")


def label_to_possible_images(label_path: Path):
    """
    .../labels/train/a.txt
        ->
    .../images/train/a.jpg
    .../images/train/a.png
    ...
    """
    parts = list(label_path.parts)
    idx = None

    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "labels":
            idx = i
            break

    if idx is None:
        raise ValueError(
            f"label 路径中找不到 labels 目录：{label_path}"
        )

    parts[idx] = "images"
    base = Path(*parts)

    return [
        base.with_suffix(ext)
        for ext in IMAGE_EXTENSIONS
    ]


# =============================================================================
# 3. TXT 处理
# =============================================================================

def read_txt_rows(txt_path: Path):
    if not txt_path.exists():
        return []

    rows = []

    with txt_path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            original = line.rstrip("\r\n")
            normalized = normalize_txt_path(original)

            if not normalized:
                continue

            rows.append({
                "line_no": line_no,
                "original": original,
                "normalized": normalized,
            })

    return rows


def write_txt_rows(
    txt_path: Path,
    kept_original_lines,
):
    """
    保留原 txt 中每行原来的路径写法。
    只删除失效行，不主动改成别的路径格式。
    """
    content = "\n".join(
        kept_original_lines
    )

    if kept_original_lines:
        content += "\n"

    if DRY_RUN:
        return

    txt_path.write_text(
        content,
        encoding="utf-8",
    )


# =============================================================================
# 4. 备份
# =============================================================================

def backup_file(
    src: Path,
    backup_root: Path,
):
    if not src.exists():
        return

    rel = src.resolve().relative_to(
        DATASET_ROOT.resolve()
    )

    dst = backup_root / rel

    if DRY_RUN:
        print(
            f"    [备份计划] {src} -> {dst}"
        )
        return

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        src,
        dst,
    )


def move_orphan_label(
    label_path: Path,
    orphan_backup_root: Path,
):
    if BACKUP_ORPHAN_LABELS:
        rel = label_path.resolve().relative_to(
            DATASET_ROOT.resolve()
        )

        dst = (
            orphan_backup_root
            / rel
        )

        if DRY_RUN:
            print(
                f"    [孤立label移走计划] "
                f"{label_path} -> {dst}"
            )
            return

        dst.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.move(
            str(label_path),
            str(dst),
        )

    else:
        if DRY_RUN:
            print(
                f"    [孤立label删除计划] "
                f"{label_path}"
            )
            return

        label_path.unlink()


# =============================================================================
# 5. 清理某个 train.txt / val.txt
# =============================================================================

def clean_one_split_txt(
    dataset_dir: Path,
    split: str,
    txt_backup_root: Path,
):
    txt_path = (
        dataset_dir
        / f"{split}.txt"
    )

    rows = read_txt_rows(
        txt_path
    )

    if not txt_path.exists():
        print(
            f"  {split}.txt：不存在，跳过"
        )

        return {
            "original": 0,
            "kept": 0,
            "removed_missing_image": 0,
            "removed_missing_label": 0,
        }

    if BACKUP_SUBDATASET_TXT:
        backup_file(
            txt_path,
            txt_backup_root,
        )

    kept_lines = []

    removed_missing_image = 0
    removed_missing_label = 0

    for row in rows:
        image_path = resolve_txt_image_path(
            row["normalized"],
            dataset_dir,
        )

        # 情况 1：图片已经删除
        if not image_path.exists():
            removed_missing_image += 1
            continue

        # 图片存在时，无论 label 是否存在，都保留这一行。
        #
        # 原因：
        # YOLO 数据集中“没有对应 label 文件”的图片可以作为纯背景负样本。
        # 因此不能仅因为 label 不存在，就从 train.txt / val.txt 删除图片。
        #
        # 这里只把“图片本身已经不存在”的条目清掉。
        kept_lines.append(
            row["original"]
        )

    write_txt_rows(
        txt_path,
        kept_lines,
    )

    print(
        f"  {split}.txt："
        f"原 {len(rows)} 行 -> "
        f"保留 {len(kept_lines)} 行 | "
        f"删除(图片不存在) {removed_missing_image} | "
        f"保留无label图片（可作为负样本）"
    )

    return {
        "original": len(rows),
        "kept": len(kept_lines),
        "removed_missing_image": removed_missing_image,
        "removed_missing_label": removed_missing_label,
    }


# =============================================================================
# 6. 清理孤立 label
# =============================================================================

def clean_orphan_labels(
    dataset_dir: Path,
    orphan_backup_root: Path,
):
    checked = 0
    orphan_count = 0

    for split in ("train", "val"):
        label_dir = (
            dataset_dir
            / "labels"
            / split
        )

        if not label_dir.exists():
            continue

        labels = sorted(
            [
                p for p in label_dir.rglob("*.txt")
                if p.is_file()
            ],
            key=lambda p: p.as_posix().lower(),
        )

        for label_path in labels:
            checked += 1

            image_exists = any(
                p.exists()
                for p in label_to_possible_images(
                    label_path
                )
            )

            if image_exists:
                continue

            orphan_count += 1

            move_orphan_label(
                label_path,
                orphan_backup_root,
            )

    print(
        f"  孤立 label：检查 {checked} 个，"
        f"发现 {orphan_count} 个"
    )

    return orphan_count


# =============================================================================
# 7. 清理空目录
# =============================================================================

def remove_empty_dirs(root: Path):
    if DRY_RUN or not root.exists():
        return

    dirs = sorted(
        [
            p for p in root.rglob("*")
            if p.is_dir()
        ],
        key=lambda p: len(p.parts),
        reverse=True,
    )

    for directory in dirs:
        try:
            directory.rmdir()
        except OSError:
            pass


# =============================================================================
# 8. 主程序
# =============================================================================

def main():
    print("=" * 92)
    print("清理 images_cvat*_dataset 内部 train.txt / val.txt 与残余 labels")
    print("=" * 92)
    print(f"DATASET_ROOT             : {DATASET_ROOT}")
    print(f"DRY_RUN                  : {DRY_RUN}")
    print(f"BACKUP_SUBDATASET_TXT    : {BACKUP_SUBDATASET_TXT}")
    print(f"BACKUP_ORPHAN_LABELS     : {BACKUP_ORPHAN_LABELS}")
    print()
    print("注意：本脚本不会读取或修改 train_grouped.txt / val_grouped.txt。")
    print("=" * 92)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"DATASET_ROOT 不存在：{DATASET_ROOT}"
        )

    dataset_dirs = find_dataset_dirs()

    print(
        f"\n找到 {len(dataset_dirs)} 个子数据集："
    )

    for d in dataset_dirs:
        print(
            f"  {d.name}"
        )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    cleanup_root = (
        DATASET_ROOT
        / f"_subdataset_cleanup_{stamp}"
    )

    txt_backup_root = (
        cleanup_root
        / "backup_train_val_txt"
    )

    orphan_backup_root = (
        cleanup_root
        / "backup_orphan_labels"
    )

    totals = {
        "orphan_labels": 0,

        "train_original": 0,
        "train_kept": 0,
        "train_rm_image": 0,
        "train_rm_label": 0,

        "val_original": 0,
        "val_kept": 0,
        "val_rm_image": 0,
        "val_rm_label": 0,
    }

    print()

    for index, dataset_dir in enumerate(
        dataset_dirs,
        start=1,
    ):
        print(
            f"[{index}/{len(dataset_dirs)}] "
            f"{dataset_dir.name}"
        )

        # 先清理孤立 label
        orphan_count = clean_orphan_labels(
            dataset_dir,
            orphan_backup_root,
        )

        totals["orphan_labels"] += orphan_count

        # 再清理该数据集自己的 train.txt
        train_stats = clean_one_split_txt(
            dataset_dir,
            "train",
            txt_backup_root,
        )

        totals["train_original"] += train_stats["original"]
        totals["train_kept"] += train_stats["kept"]
        totals["train_rm_image"] += train_stats["removed_missing_image"]
        totals["train_rm_label"] += train_stats["removed_missing_label"]

        # 再清理该数据集自己的 val.txt
        val_stats = clean_one_split_txt(
            dataset_dir,
            "val",
            txt_backup_root,
        )

        totals["val_original"] += val_stats["original"]
        totals["val_kept"] += val_stats["kept"]
        totals["val_rm_image"] += val_stats["removed_missing_image"]
        totals["val_rm_label"] += val_stats["removed_missing_label"]

        if REMOVE_EMPTY_DIRS:
            remove_empty_dirs(
                dataset_dir / "labels"
            )

        print()

    print("=" * 92)

    if DRY_RUN:
        print("检查完成（DRY_RUN=True，没有真正修改任何文件）")
    else:
        print("清理完成")

    print("=" * 92)

    print(
        f"孤立 labels：{totals['orphan_labels']}"
    )

    print(
        f"train.txt："
        f"原 {totals['train_original']} -> "
        f"保留 {totals['train_kept']} | "
        f"去掉图片不存在 {totals['train_rm_image']} | "
        f"去掉label不存在 {totals['train_rm_label']}"
    )

    print(
        f"val.txt：  "
        f"原 {totals['val_original']} -> "
        f"保留 {totals['val_kept']} | "
        f"去掉图片不存在 {totals['val_rm_image']} | "
        f"去掉label不存在 {totals['val_rm_label']}"
    )

    if DRY_RUN:
        print()
        print(
            "确认统计符合预期后，把 DRY_RUN=False 再运行一次。"
        )
    else:
        print()
        print(
            f"备份目录：{cleanup_root}"
        )
        print()
        print(
            "现在每个 images_cvat*_dataset 内部的 "
            "train.txt / val.txt 已与实际 image/label 更一致。"
        )
        print(
            "接下来再运行 grouped 生成脚本即可。"
        )

    print("=" * 92)


if __name__ == "__main__":
    main()
