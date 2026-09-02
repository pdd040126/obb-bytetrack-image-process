from pathlib import Path
import shutil
from collections import defaultdict


# ============================================================
# 1. 用户参数
# ============================================================

DATASET_ROOT = Path(
    r"F:\train_data\images_cvat13_dataset"
)

# 是否真正执行 label 移动
#
# False:
#     只检查，不修改任何文件
#
# True:
#     真正移动 label，并重新生成 train.txt / val.txt
#
DRY_RUN = False


# 支持的图片格式
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# 2. 数据集路径
# ============================================================

IMAGES_ROOT = DATASET_ROOT / "images"
LABELS_ROOT = DATASET_ROOT / "labels"

IMAGE_TRAIN_DIR = IMAGES_ROOT / "train"
IMAGE_VAL_DIR = IMAGES_ROOT / "val"

LABEL_TRAIN_DIR = LABELS_ROOT / "train"
LABEL_VAL_DIR = LABELS_ROOT / "val"

TRAIN_TXT = DATASET_ROOT / "train.txt"
VAL_TXT = DATASET_ROOT / "val.txt"


# ============================================================
# 3. 基础函数
# ============================================================

def is_image(path: Path):
    """
    判断文件是否为支持的图片格式。
    """
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_images(folder: Path):
    """
    递归寻找文件夹中的所有图片。
    """
    if not folder.exists():
        return []

    return sorted(
        [
            p
            for p in folder.rglob("*")
            if is_image(p)
        ],
        key=lambda p: str(p).lower()
    )


def print_title(text):
    print()
    print("=" * 75)
    print(text)
    print("=" * 75)


# ============================================================
# 4. 检查基本目录
# ============================================================

print_title("YOLO Train / Val 自动同步工具")

print(f"数据集目录：{DATASET_ROOT}")

if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        f"数据集不存在：\n{DATASET_ROOT}"
    )

if not IMAGES_ROOT.exists():
    raise FileNotFoundError(
        f"images 文件夹不存在：\n{IMAGES_ROOT}"
    )

if not LABELS_ROOT.exists():
    raise FileNotFoundError(
        f"labels 文件夹不存在：\n{LABELS_ROOT}"
    )


# 自动创建目标 label 文件夹
if not DRY_RUN:
    LABEL_TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LABEL_VAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 5. 扫描 train / val 图片
# ============================================================

train_images = get_images(IMAGE_TRAIN_DIR)
val_images = get_images(IMAGE_VAL_DIR)


print()
print(f"Train 图片数：{len(train_images)}")
print(f"Val   图片数：{len(val_images)}")
print(f"总图片数     ：{len(train_images) + len(val_images)}")


# ============================================================
# 6. 检查 train / val 是否存在同名图片
#
# 注意：
# YOLO label 依赖图片 stem，例如：
#
#   abc.jpg
#   abc.txt
#
# 因此：
#
# images/train/abc.jpg
# images/val/abc.png
#
# 也会造成 label 冲突。
# ============================================================

train_stems = defaultdict(list)
val_stems = defaultdict(list)


for image in train_images:
    train_stems[image.stem].append(image)

for image in val_images:
    val_stems[image.stem].append(image)


duplicate_between_splits = (
    set(train_stems.keys())
    &
    set(val_stems.keys())
)


if duplicate_between_splits:

    print_title("错误：Train / Val 中存在同名图片")

    for stem in sorted(duplicate_between_splits):

        print(f"\n文件名：{stem}")

        print("Train:")
        for p in train_stems[stem]:
            print(f"    {p}")

        print("Val:")
        for p in val_stems[stem]:
            print(f"    {p}")

    raise RuntimeError(
        "\nTrain 和 Val 中存在相同 stem 的图片，"
        "无法安全判断 label 应属于哪一边。"
    )


# ============================================================
# 7. 建立所有 label 的索引
#
# 无论 label 当前位于：
#
# labels/
# labels/train/
# labels/val/
# labels/xxx/
#
# 都扫描出来。
# ============================================================

all_labels = sorted(
    [
        p
        for p in LABELS_ROOT.rglob("*.txt")
        if p.is_file()
    ],
    key=lambda p: str(p).lower()
)


label_index = defaultdict(list)


for label in all_labels:
    label_index[label.stem].append(label)


print()
print(f"找到 Label：{len(all_labels)} 个")


# ============================================================
# 8. Label 同步函数
# ============================================================

moved_labels = []

missing_labels = []

duplicate_labels = []

already_correct = []


def sync_one_image(
    image_path: Path,
    split: str
):
    """
    根据图片所在 split，
    自动将对应 label 移到正确目录。

    图片不会被移动。
    """

    stem = image_path.stem

    # --------------------------------------------------------
    # 图片相对于 images/train 或 images/val 的路径
    #
    # 例如：
    #
    # images/train/camera01/day1/0001.jpg
    #
    # relative:
    #
    # camera01/day1/0001.jpg
    #
    # label 最终对应：
    #
    # labels/train/camera01/day1/0001.txt
    # --------------------------------------------------------

    if split == "train":

        relative_image = image_path.relative_to(
            IMAGE_TRAIN_DIR
        )

        destination_root = LABEL_TRAIN_DIR

    elif split == "val":

        relative_image = image_path.relative_to(
            IMAGE_VAL_DIR
        )

        destination_root = LABEL_VAL_DIR

    else:
        raise ValueError(
            f"未知 split：{split}"
        )


    relative_label = (
        relative_image
        .with_suffix(".txt")
    )

    destination = (
        destination_root
        /
        relative_label
    )


    # --------------------------------------------------------
    # 目标 label 已经存在
    # --------------------------------------------------------

    if destination.exists():

        already_correct.append(
            (
                image_path,
                destination
            )
        )

        return


    # --------------------------------------------------------
    # 根据 stem 查找 label
    # --------------------------------------------------------

    candidates = label_index.get(
        stem,
        []
    )


    # 排除不存在的路径
    #
    # 因为前面某个 label 可能已经被 move 了
    candidates = [
        p
        for p in candidates
        if p.exists()
    ]


    # --------------------------------------------------------
    # 没找到
    # --------------------------------------------------------

    if len(candidates) == 0:

        missing_labels.append(
            image_path
        )

        return


    # --------------------------------------------------------
    # 找到多个同名 label
    # --------------------------------------------------------

    if len(candidates) > 1:

        duplicate_labels.append(
            (
                image_path,
                candidates
            )
        )

        return


    # --------------------------------------------------------
    # 唯一匹配
    # --------------------------------------------------------

    source = candidates[0]


    if source == destination:

        already_correct.append(
            (
                image_path,
                destination
            )
        )

        return


    moved_labels.append(
        (
            source,
            destination
        )
    )


    print(
        f"[MOVE] "
        f"{source.relative_to(DATASET_ROOT)}"
        f"  ->  "
        f"{destination.relative_to(DATASET_ROOT)}"
    )


    if not DRY_RUN:

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.move(
            str(source),
            str(destination)
        )


# ============================================================
# 9. 根据图片位置同步 Train Label
# ============================================================

print_title("同步 Train Labels")

for image in train_images:

    sync_one_image(
        image,
        "train"
    )


# ============================================================
# 10. 根据图片位置同步 Val Label
# ============================================================

print_title("同步 Val Labels")

for image in val_images:

    sync_one_image(
        image,
        "val"
    )


# ============================================================
# 11. 生成 train.txt / val.txt
#
# 采用：
#
# images/train/xxx.jpg
#
# images/val/xxx.jpg
#
# 这样的相对路径。
#
# 使用 "/"，兼容 Windows 和 Linux。
# ============================================================

def generate_split_txt(
    images,
    txt_path
):

    lines = []

    for image in images:

        relative_path = image.relative_to(
            DATASET_ROOT
        )

        lines.append(
            relative_path.as_posix()
        )


    if not DRY_RUN:

        with txt_path.open(
            "w",
            encoding="utf-8",
            newline="\n"
        ) as f:

            for line in lines:
                f.write(line + "\n")


    return lines


print_title("重新生成 Train / Val TXT")


train_lines = generate_split_txt(
    train_images,
    TRAIN_TXT
)

val_lines = generate_split_txt(
    val_images,
    VAL_TXT
)


print(
    f"train.txt：{len(train_lines)} 行"
)

print(
    f"val.txt  ：{len(val_lines)} 行"
)


# ============================================================
# 12. 再次扫描 labels
# ============================================================

final_labels = []

if LABELS_ROOT.exists():

    final_labels = [
        p
        for p in LABELS_ROOT.rglob("*.txt")
        if p.is_file()
    ]


# ============================================================
# 13. 查找孤立 Label
#
# 即：
#
# 有 label，
# 但是 train / val 都没有对应图片。
# ============================================================

all_image_stems = (
    set(train_stems.keys())
    |
    set(val_stems.keys())
)


orphan_labels = [
    label
    for label in final_labels
    if label.stem not in all_image_stems
]


# ============================================================
# 14. 输出结果
# ============================================================

print_title("同步结果")


print(
    f"Train 图片       ：{len(train_images)}"
)

print(
    f"Val 图片         ：{len(val_images)}"
)

print(
    f"已正确 Label     ：{len(already_correct)}"
)

print(
    f"需要移动 Label   ：{len(moved_labels)}"
)

print(
    f"缺失 Label       ：{len(missing_labels)}"
)

print(
    f"重复 Label       ：{len(duplicate_labels)}"
)

print(
    f"孤立 Label       ：{len(orphan_labels)}"
)


# ============================================================
# 15. 显示缺失 Label
# ============================================================

if missing_labels:

    print_title("警告：以下图片没有 Label")

    for image in missing_labels:

        print(
            image.relative_to(
                DATASET_ROOT
            )
        )


# ============================================================
# 16. 显示重复 Label
# ============================================================

if duplicate_labels:

    print_title("警告：存在多个候选 Label")

    for image, labels in duplicate_labels:

        print()

        print(
            "图片：",
            image.relative_to(DATASET_ROOT)
        )

        print("候选 Label：")

        for label in labels:

            print(
                "    ",
                label.relative_to(
                    DATASET_ROOT
                )
            )


# ============================================================
# 17. 显示孤立 Label
# ============================================================

if orphan_labels:

    print_title("警告：孤立 Label")

    print(
        "下面这些 Label 没有找到对应图片，"
        "程序不会自动删除："
    )

    print()

    for label in orphan_labels:

        print(
            label.relative_to(
                DATASET_ROOT
            )
        )


# ============================================================
# 18. 显示 txt 示例
# ============================================================

print_title("train.txt 前 5 行")

for line in train_lines[:5]:
    print(line)


print_title("val.txt 前 5 行")

for line in val_lines[:5]:
    print(line)


# ============================================================
# 19. 完成
# ============================================================

print_title("完成")


if DRY_RUN:

    print(
        "当前为 DRY_RUN=True。"
    )

    print(
        "没有修改任何 Label，也没有修改 train.txt / val.txt。"
    )

    print(
        "确认输出无误后，将 DRY_RUN 改为 False 再运行。"
    )

else:

    print(
        "Label 已根据图片位置完成同步。"
    )

    print(
        "train.txt 和 val.txt 已重新生成。"
    )

    print(
        "图片文件没有被移动。"
    )