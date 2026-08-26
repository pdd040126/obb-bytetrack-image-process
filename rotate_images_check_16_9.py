from pathlib import Path
import cv2


# ============================================================
# 参数设置
# ============================================================

# 输入图片文件夹
INPUT_DIR = Path(r"H:\image_data\images_rotate_input")

# 输出图片文件夹
OUTPUT_DIR = Path(r"H:\image_data\images_rotate_output")

# 是否递归处理子文件夹
RECURSIVE = True

# True  = 竖图顺时针旋转90°
# False = 竖图逆时针旋转90°
ROTATE_CLOCKWISE = True

# JPEG保存质量
JPEG_QUALITY = 95

# 标准16:9比例
ASPECT_RATIO_16_9 = 16.0 / 9.0

# 允许误差
# 0.005 = 0.5%
ASPECT_RATIO_TOLERANCE = 0.005

# ============================================================
# 新增参数
# ============================================================

# True:
#   超出16:9允许误差的图片直接过滤掉，不保存到输出目录
#
# False:
#   超出误差的图片仍然保存，只在 not_16_9.txt 中记录
#
# 注意：
#   不会删除 INPUT_DIR 中的原始图片
REMOVE_NOT_16_9 = True


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
# 判断是否为图片文件
# ============================================================

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


# ============================================================
# 判断是否接近16:9
# ============================================================

def is_near_16_by_9(width: int, height: int) -> bool:

    if width <= 0 or height <= 0:
        return False

    ratio = width / height

    error = (
        abs(ratio - ASPECT_RATIO_16_9)
        / ASPECT_RATIO_16_9
    )

    return error <= ASPECT_RATIO_TOLERANCE


# ============================================================
# 保存为JPG
# ============================================================

def save_jpeg(path: Path, image) -> bool:

    try:

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        return cv2.imwrite(
            str(path),
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ],
        )

    except Exception:
        return False


# ============================================================
# 处理单张图片
# ============================================================

def process_image(
    input_path: Path,
    output_path: Path,
    warning_file,
    stats: dict,
) -> bool:

    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_COLOR
    )

    if image is None:

        print(
            f"[FAILED] Cannot read: {input_path}"
        )

        return False


    original_height, original_width = image.shape[:2]

    final_image = image

    final_width = original_width
    final_height = original_height


    print(
        f"{input_path.name:<35} "
        f"{original_width} x {original_height}",
        end="",
    )


    # ========================================================
    # 竖图 -> 横图
    # ========================================================

    if original_height > original_width:

        if ROTATE_CLOCKWISE:

            final_image = cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE,
            )

        else:

            final_image = cv2.rotate(
                image,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )


        final_height, final_width = final_image.shape[:2]

        stats["rotated"] += 1


        print(
            f"  -> rotate -> "
            f"{final_width} x {final_height}",
            end="",
        )


    # ========================================================
    # 已经是横图
    # ========================================================

    else:

        stats["kept"] += 1

        print(
            "  -> keep",
            end=""
        )


    # ========================================================
    # 检查最终比例
    # ========================================================

    final_ratio = (
        final_width
        / final_height
    )


    if not is_near_16_by_9(
        final_width,
        final_height
    ):

        stats["warning"] += 1


        print(
            f"  [WARNING: Not 16:9]"
            f"  ratio={final_ratio:.4f}",
            end="",
        )


        # ----------------------------------------------------
        # 写入异常比例记录
        # ----------------------------------------------------

        warning_file.write(
            f"{input_path}\n"
            f"    Original : "
            f"{original_width} x {original_height}\n"
            f"    Final    : "
            f"{final_width} x {final_height}\n"
            f"    Ratio    : "
            f"{final_ratio:.6f}\n"
        )


        # ----------------------------------------------------
        # 如果开启过滤，直接不保存
        # ----------------------------------------------------

        if REMOVE_NOT_16_9:

            stats["removed"] += 1

            print(
                "  -> REMOVED"
            )

            warning_file.write(
                "    Action   : REMOVED\n\n"
            )

            return True


        else:

            warning_file.write(
                "    Action   : KEPT\n\n"
            )


    print()


    # ========================================================
    # 统一保存为JPG
    # ========================================================

    if not save_jpeg(
        output_path,
        final_image
    ):

        print(
            f"[FAILED] Cannot save: "
            f"{output_path}"
        )

        return False


    stats["saved"] += 1

    return True


# ============================================================
# 主函数
# ============================================================

def main():

    print(
        "============================================"
    )

    print(
        "Batch Portrait -> Landscape Converter"
    )

    print(
        "Output Format: JPG"
    )

    print(
        "============================================\n"
    )


    print(
        f"Input : {INPUT_DIR}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        f"Remove Not 16:9: {REMOVE_NOT_16_9}"
    )

    print()


    # ========================================================
    # 检查输入目录
    # ========================================================

    if not INPUT_DIR.exists():

        print(
            "[ERROR] Input directory does not exist:\n"
            f"{INPUT_DIR}"
        )

        return 1


    # ========================================================
    # 创建输出目录
    # ========================================================

    try:

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    except OSError as e:

        print(
            "[ERROR] Cannot create output directory:\n"
            f"{e}"
        )

        return 1


    # ========================================================
    # 创建非16:9图片记录文件
    # ========================================================

    warning_file_path = (
        OUTPUT_DIR
        / "not_16_9.txt"
    )


    stats = {

        "total": 0,

        "rotated": 0,

        "kept": 0,

        "warning": 0,

        "removed": 0,

        "saved": 0,

        "failed": 0,
    }


    try:

        warning_file = warning_file_path.open(
            "w",
            encoding="utf-8",
        )

    except OSError as e:

        print(
            "[ERROR] Cannot create warning file:\n"
            f"{warning_file_path}\n"
            f"Reason: {e}"
        )

        return 1


    with warning_file:

        warning_file.write(
            "Images that are NOT close to 16:9\n"
            f"Tolerance: "
            f"{ASPECT_RATIO_TOLERANCE * 100.0}%\n"
            f"Remove: "
            f"{REMOVE_NOT_16_9}\n\n"
        )


        # ====================================================
        # 遍历文件
        # ====================================================

        try:

            if RECURSIVE:

                file_iterator = (
                    INPUT_DIR.rglob("*")
                )

            else:

                file_iterator = (
                    INPUT_DIR.iterdir()
                )


            for input_path in file_iterator:


                if not input_path.is_file():
                    continue


                if not is_image_file(input_path):
                    continue


                stats["total"] += 1


                try:

                    relative_path = (
                        input_path.relative_to(
                            INPUT_DIR
                        )
                    )

                except ValueError:

                    relative_path = Path(
                        input_path.name
                    )


                output_path = (
                    OUTPUT_DIR
                    / relative_path
                ).with_suffix(".jpg")


                if not process_image(
                    input_path,
                    output_path,
                    warning_file,
                    stats,
                ):

                    stats["failed"] += 1


        except OSError as e:

            print(
                "[ERROR] Directory traversal failed:\n"
                f"{e}"
            )

            return 1


    # ========================================================
    # 最终统计
    # ========================================================

    print()

    print(
        "============================================"
    )

    print(
        "Finished"
    )

    print(
        "============================================"
    )

    print(
        f"Total        : {stats['total']}"
    )

    print(
        f"Rotated      : {stats['rotated']}"
    )

    print(
        f"Landscape    : {stats['kept']}"
    )

    print(
        f"Not 16:9     : {stats['warning']}"
    )

    print(
        f"Removed      : {stats['removed']}"
    )

    print(
        f"Saved        : {stats['saved']}"
    )

    print(
        f"Failed       : {stats['failed']}"
    )

    print(
        f"JPEG Quality : {JPEG_QUALITY}"
    )

    print(
        f"Output       : {OUTPUT_DIR}"
    )

    print(
        f"Warning file : {warning_file_path}"
    )


    return 0


if __name__ == "__main__":
    raise SystemExit(main())