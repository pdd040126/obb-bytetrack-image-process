from pathlib import Path
import random


# =========================
# 用户配置
# =========================

DATASET_DIRS = [
    Path("H:/train_data/images_cvat00_dataset"),
    Path("H:/train_data/images_cvat01_dataset"),
    Path("H:/train_data/images_cvat02_dataset"),
    Path("H:/train_data/images_cvat04_dataset"),
    Path("H:/train_data/images_cvat08_dataset"),
    Path("H:/train_data/images_cvat10_dataset"),
    Path("H:/train_data/images_cvat20_dataset"),
]

OUTPUT_DIR = Path("H:/image_process_data")

TRAIN_RATIO = 0.80
RANDOM_SEED = 24

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

REQUIRE_LABEL = False


def collect_dataset_images(dataset_dir: Path):
    """读取单个数据集中的图片，并检查对应标签。"""

    image_dir = dataset_dir / "images" / "train"
    label_dir = dataset_dir / "labels" / "train"

    if not image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    if not label_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {label_dir}")

    images = sorted(
        path.resolve()
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(f"没有找到图片: {image_dir}")

    valid_images = []
    missing_labels = []

    for image_path in images:
        relative_path = image_path.relative_to(image_dir.resolve())
        label_path = (label_dir / relative_path).with_suffix(".txt")

        if label_path.exists():
            valid_images.append(image_path)
        else:
            missing_labels.append((image_path, label_path))

            if not REQUIRE_LABEL:
                valid_images.append(image_path)

    if missing_labels:
        print(f"\n[{dataset_dir.name}] 缺少标签的图片: {len(missing_labels)}")

        for image_path, label_path in missing_labels[:10]:
            print(f"  图片: {image_path}")
            print(f"  标签: {label_path}")

        if len(missing_labels) > 10:
            print(f"  ... 另外还有 {len(missing_labels) - 10} 个")

        if REQUIRE_LABEL:
            print("  REQUIRE_LABEL=True，因此这些图片不会参与划分。")

    return valid_images


def split_dataset(images, train_ratio, rng):
    """将一个数据集随机划分为训练集和验证集。"""

    images = images.copy()
    rng.shuffle(images)

    total = len(images)

    if total == 1:
        return images, []

    train_count = int(total * train_ratio)

    train_count = max(1, train_count)
    train_count = min(total - 1, train_count)

    train_images = images[:train_count]
    val_images = images[train_count:]

    return train_images, val_images


def write_list(file_path: Path, image_paths):
    """将图片绝对路径写入 txt 文件。"""

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8", newline="\n") as file:
        for image_path in image_paths:
            file.write(f"{image_path.as_posix()}\n")


def main():
    rng = random.Random(RANDOM_SEED)

    all_train = []
    all_val = []

    print("=" * 60)
    print("开始划分多个 YOLO/OBB 数据集")
    print("=" * 60)

    for dataset_dir in DATASET_DIRS:
        images = collect_dataset_images(dataset_dir)

        train_images, val_images = split_dataset(
            images=images,
            train_ratio=TRAIN_RATIO,
            rng=rng,
        )

        all_train.extend(train_images)
        all_val.extend(val_images)

        print(
            f"\n[{dataset_dir.name}] "
            f"总数={len(images)}, "
            f"train={len(train_images)}, "
            f"val={len(val_images)}"
        )

    rng.shuffle(all_train)
    rng.shuffle(all_val)

    train_txt = OUTPUT_DIR / "train.txt"
    val_txt = OUTPUT_DIR / "val.txt"

    write_list(train_txt, all_train)
    write_list(val_txt, all_val)

    train_set = set(all_train)
    val_set = set(all_val)

    overlap = train_set & val_set

    print("\n" + "=" * 60)
    print("划分完成")
    print("=" * 60)
    print(f"总训练图片: {len(all_train)}")
    print(f"总验证图片: {len(all_val)}")
    print(f"训练/验证重复图片: {len(overlap)}")
    print(f"train.txt: {train_txt}")
    print(f"val.txt:   {val_txt}")

    if overlap:
        raise RuntimeError("检测到 train 和 val 中存在重复图片，请检查。")

    print("\n可以在 data.yaml 中使用:")
    print(f"train: {train_txt}")
    print(f"val: {val_txt}")


if __name__ == "__main__":
    main()
