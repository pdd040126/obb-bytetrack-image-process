"""Batch image undistortion with explicit field-of-view policies.

Default behavior matches the supplied reference output:
OpenCV's default optimal-new-camera-matrix policy, alpha=0 and
centerPrincipalPoint=False.  Use ``--mode preserve-geometry`` when object
aspect ratio is more important than maximizing the border-free image area.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".ppm",
}


@dataclass(frozen=True)
class Calibration:
    image_size: tuple[int, int]
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms: float
    distortion_model: str


@dataclass(frozen=True)
class UndistortionContext:
    camera_matrix: np.ndarray
    new_camera_matrix: np.ndarray
    valid_roi: tuple[int, int, int, int]
    map1: np.ndarray
    map2: np.ndarray


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    ======================== 用户最常修改的参数 ========================
    下面这些参数都可以直接改 ``default=...``，这样双击/直接运行脚本时
    就会采用你写在代码里的默认值；也可以运行脚本时通过命令行覆盖。

    对你当前“包裹 OBB + 几何比例要保持准确”的用途，最重要的是：

    1. --input / --calibration / --output
       改成你自己的输入目录、标定 YAML、输出目录。

    2. --mode
       - match-reference:
         使用 OpenCV 默认的 ``centerPrincipalPoint=False``。
         它会优先争取较大的无黑边有效区域，但 x、y 方向可能采用不同缩放，
         因此真实正方形可能在去畸变后变成长方形。
       - preserve-geometry:
         使用 ``centerPrincipalPoint=True``，更重视保持横纵几何比例。
         对 OBB、姿态角、尺寸/长宽比分析，建议使用这个模式。

       当前代码为了与旧参考结果一致，默认仍是 ``match-reference``。
       如果你以后都做 OBB，建议把下面的 default 改成：
           default="preserve-geometry"

    3. --alpha
       OpenCV 的自由缩放系数，范围 [0, 1]：
       - 0.0：尽量不出现黑边，通常裁掉更多边缘视场；
       - 1.0：尽量保留全部视场，边缘可能出现黑边；
       - 0.3~0.7：二者折中。
       注意：alpha 主要控制“保留多少视场/黑边多少”，不是图像清晰度参数。

    4. --allow-size-scaling
       只有当输入图和标定图“宽高比完全相同，只做了等比例缩放”时才应开启。
       如果图像发生过裁剪、非等比 resize、旋转后尺寸变化，不要开启。
       对固定相机、固定 2560x1440 图像，建议保持关闭。

    5. --jpeg-quality
       输出 JPG 压缩质量，范围 0~100。
       95 左右适合保留边缘细节；90 文件更小；越低压缩痕迹越明显。
       这不会改变几何映射，但过低会影响 OBB 边缘观测。

    6. --recursive
       是否递归处理输入目录下所有子目录。默认关闭。

    7. --start-index / --filename-width
       控制输出文件编号，例如 start-index=1、filename-width=6 会生成：
       000001.jpg、000002.jpg、...

    8. --stop-on-error
       默认某一张失败时继续处理后面的图片；开启后遇到第一张失败就停止。
    ====================================================================
    """
    parser = argparse.ArgumentParser(
        description="Read OpenCV YAML calibration and batch-undistort images."
    )

    # -------------------- 1. 输入图片目录 --------------------
    # 改这里：放“原始、尚未去畸变”的图片。
    # 不要把输出目录放回输入目录，否则容易重复处理已经去畸变的图。
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(r"H:\image_process_data\images_calibration_input"),
        help="Input image directory.",
    )

    # -------------------- 2. 相机标定 YAML -------------------
    # 改这里：指向与你当前相机、镜头、分辨率对应的标定结果。
    # YAML 中至少应包含 image_width、image_height、camera_matrix、
    # distortion_coefficients。换相机/镜头后应重新标定并更换该文件。
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(r"H:\image_proccess\camera_single.yml"),
        help="OpenCV calibration YAML file.",
    )

    # -------------------- 3. 去畸变输出目录 -------------------
    # 改这里：处理后的 JPG 会写入该目录。
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"H:\image_process_data\images_calibration_output"),
        help="Output image directory.",
    )

    # -------------------- 4. 几何保持模式【非常重要】 ----------
    # match-reference：尽量匹配旧参考输出，但可能改变 x/y 缩放比例。
    # preserve-geometry：保持物体横纵比例，更适合 OBB/姿态/尺寸分析。
    #
    # 对你的用途建议：把 default="match-reference" 改为
    #                  default="preserve-geometry"
    # 如果暂时要和旧数据保持完全一致，则继续保留 match-reference。
    parser.add_argument(
        "--mode",
        choices=("match-reference", "preserve-geometry"),
        default="preserve-geometry",
        help=(
            "match-reference keeps more horizontal content without black borders "
            "but changes the x/y scale; preserve-geometry keeps object proportions."
        ),
    )

    # -------------------- 5. alpha：视场与黑边的权衡 ------------
    # 允许范围：0.0 ~ 1.0
    # 0.0：尽量没有黑边，但会裁掉更多外围画面。
    # 1.0：尽量保留全部视场，但边缘可能出现黑边。
    # 对固定工业相机通常可从 0.0 开始；如边缘内容很重要，再试 0.2/0.5/1.0。
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="OpenCV free-scaling parameter in [0, 1]. Default: 0.",
    )

    # -------------------- 6. 是否递归处理子目录 -----------------
    # 默认 False。命令行加 --recursive 后才会递归。
    parser.add_argument("--recursive", action="store_true")

    # -------------------- 7. 是否允许按分辨率缩放内参【谨慎】 -----
    # 默认关闭。只有“输入图相对标定图做了等比例 resize，且没有裁剪”时才可开启。
    # 例如标定 2560x1440，输入严格等比例缩成 1280x720，可以开启。
    # 如果输入变成 1920x1200、做过 crop、非等比 resize，则不能开启。
    parser.add_argument(
        "--allow-size-scaling",
        action="store_true",
        help="Scale camera intrinsics when image size changes but aspect ratio is unchanged.",
    )

    # -------------------- 8. JPG 输出质量 -----------------------
    # 0~100，数值越高越清晰、文件越大。
    # OBB/边缘标注建议 90~95；如果只追求体积可以再降低。
    parser.add_argument("--jpeg-quality", type=int, default=90)

    # -------------------- 9. 输出编号 ---------------------------
    # start-index=1、filename-width=6 -> 000001.jpg 起步。
    # 已存在编号不会被覆盖，程序会继续寻找下一个可用编号。
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--filename-width", type=int, default=6)

    # -------------------- 10. 单张失败是否中止整个批次 ----------
    # 默认关闭：某张图失败只报错，继续处理剩余图片。
    # 若希望严格模式，运行时加 --stop-on-error。
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def require_file_storage_matrix(
    storage: cv2.FileStorage, name: str
) -> np.ndarray:
    matrix = storage.getNode(name).mat()
    if matrix is None or matrix.size == 0:
        raise ValueError(f"Calibration YAML has no valid '{name}'.")
    return np.asarray(matrix, dtype=np.float64)


def load_calibration(path: Path) -> Calibration:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration YAML does not exist: {path}")

    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise OSError(f"OpenCV cannot open calibration YAML: {path}")

    try:
        width = int(storage.getNode("image_width").real())
        height = int(storage.getNode("image_height").real())
        rms = float(storage.getNode("rms").real())
        model = storage.getNode("distortion_model").string() or "standard"
        camera_matrix = require_file_storage_matrix(storage, "camera_matrix")
        dist_coeffs = require_file_storage_matrix(
            storage, "distortion_coefficients"
        ).reshape(-1, 1)
    finally:
        storage.release()

    if width <= 0 or height <= 0:
        raise ValueError("Calibration YAML image_width/image_height is invalid.")
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3, got {camera_matrix.shape}.")
    if dist_coeffs.size not in (4, 5, 8, 12, 14):
        raise ValueError(
            f"Unsupported distortion coefficient count: {dist_coeffs.size}."
        )

    return Calibration(
        image_size=(width, height),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms=rms,
        distortion_model=model,
    )


def read_image(path: Path) -> np.ndarray:
    # imdecode/fromfile keeps non-ASCII Windows paths reliable.
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"OpenCV cannot read image: {path}")
    return image


def write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise OSError(f"JPEG encoding failed: {path}")
    encoded.tofile(path)


def collect_images(directory: Path, recursive: bool) -> list[Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {directory}")
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    paths = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    def sort_key(path: Path) -> tuple[int, int | str, str]:
        try:
            return (0, int(path.stem), path.name.lower())
        except ValueError:
            return (1, path.name.lower(), path.name.lower())

    return sorted(paths, key=sort_key)


def scaled_camera_matrix(
    calibration: Calibration,
    image_size: tuple[int, int],
    allow_size_scaling: bool,
) -> np.ndarray:
    """
    根据当前输入图尺寸决定是否需要缩放相机内参 K。

    关键原则：
    - 标定尺寸 == 当前图片尺寸：直接使用原始 K，最安全。
    - 尺寸不同但严格等比例 resize：允许按同一比例缩放 fx、fy、cx、cy。
    - 宽高缩放比例不同：立即报错，因为这意味着非等比拉伸，继续处理会破坏几何。

    这里的 fx/fy 不需要等于图像的 16:9。现代相机像素通常近似正方形，
    因此 fx 与 fy 往往非常接近；16:9 描述的是整张图的宽/高，而不是 fx/fy。
    """
    if image_size == calibration.image_size:
        return calibration.camera_matrix.copy()
    if not allow_size_scaling:
        raise ValueError(
            f"Image size {image_size} differs from calibration size "
            f"{calibration.image_size}. Pass --allow-size-scaling only if the "
            "image was resized without cropping."
        )

    scale_x = image_size[0] / calibration.image_size[0]
    scale_y = image_size[1] / calibration.image_size[1]
    if abs(scale_x - scale_y) > 1e-6:
        raise ValueError(
            "Image and calibration aspect ratios differ; intrinsics-only scaling "
            "would be invalid."
        )

    matrix = calibration.camera_matrix.copy()
    matrix[0, 0] *= scale_x
    matrix[0, 2] *= scale_x
    matrix[1, 1] *= scale_y
    matrix[1, 2] *= scale_y
    return matrix


def create_context(
    calibration: Calibration,
    image_size: tuple[int, int],
    mode: str,
    alpha: float,
    allow_size_scaling: bool,
) -> UndistortionContext:
    """
    为某一种分辨率创建去畸变映射表。

    这部分决定最终图像几何是否会被横向/纵向非等比例拉伸，是整段代码最关键的地方。

    ``camera_matrix`` 是标定得到（或按分辨率等比缩放后）的原始内参 K；
    ``new_camera_matrix`` 是 OpenCV 为去畸变输出重新选择的内参 K'。

    当 mode == "preserve-geometry" 时：
        centerPrincipalPoint=True
    更强调保持 x/y 几何尺度一致，适合 OBB、角度、长宽比和测量用途。

    当 mode == "match-reference" 时：
        centerPrincipalPoint=False
    与 OpenCV 常见默认策略一致，可能换取更大的无黑边区域，但也可能让 fx'/fy'
    相对原始 fx/fy 发生明显变化，从而把正方形拉成长方形。
    """
    camera_matrix = scaled_camera_matrix(
        calibration, image_size, allow_size_scaling
    )
    # preserve-geometry -> True：优先保持几何比例。
    # match-reference    -> False：优先匹配旧参考/无黑边区域，可能造成 x/y 不同缩放。
    center_principal_point = mode == "preserve-geometry"

    # 生成去畸变后的新内参 K'。
    # alpha 决定“保留视场”与“黑边/裁剪”的折中；
    # centerPrincipalPoint 则直接关系到横纵几何比例是否可能被不同程度缩放。
    new_camera_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        calibration.dist_coeffs,
        image_size,
        alpha,
        image_size,
        centerPrincipalPoint=center_principal_point,
    )

    # 原始 fx/fy 与新 fx'/fy' 的比例变化，是判断“是否把正方形拉成长方形”的
    # 一个非常直接的诊断量。
    # 注意：这里比较的是“变化前后”的 fx/fy，而不是拿它和 16:9 比。
    original_ratio = camera_matrix[0, 0] / camera_matrix[1, 1]
    new_ratio = new_camera_matrix[0, 0] / new_camera_matrix[1, 1]
    ratio_change = abs(new_ratio / original_ratio - 1.0)

    print(f"\nFirst image size: {image_size[0]} x {image_size[1]}")
    print(f"Mode: {mode}; alpha={alpha:g}")
    print("Camera matrix:\n", camera_matrix)
    print("New camera matrix:\n", new_camera_matrix)
    print(f"Valid ROI (not cropped): {valid_roi}")
    print(f"fx/fy relative change: {ratio_change * 100.0:.3f}%")
    # 超过 1% 就给出警告。对于 OBB/几何测量，通常应切换 preserve-geometry。
    # 这里目前只警告、不强制停止，是为了保持原代码行为不变。
    if ratio_change > 0.01:
        print(
            "WARNING: x/y use different scale factors, so object aspect ratio "
            "changes. Use --mode preserve-geometry for geometrically proportional "
            "output.",
            file=sys.stderr,
        )

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        calibration.dist_coeffs,
        None,
        new_camera_matrix,
        image_size,
        cv2.CV_16SC2,
    )
    return UndistortionContext(
        camera_matrix=camera_matrix,
        new_camera_matrix=new_camera_matrix,
        valid_roi=tuple(int(value) for value in valid_roi),
        map1=map1,
        map2=map2,
    )


def occupied_indices(directory: Path) -> set[int]:
    result: set[int] = set()
    for path in directory.glob("*.jpg"):
        if path.stem.isdecimal():
            result.add(int(path.stem))
    return result


def next_available_index(occupied: set[int], candidate: int) -> int:
    while candidate in occupied:
        candidate += 1
    return candidate


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1].")
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [0, 100].")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.filename_width <= 0:
        raise ValueError("--filename-width must be positive.")

    input_resolved = args.input.resolve()
    output_resolved = args.output.resolve()
    if input_resolved == output_resolved:
        raise ValueError("Input and output directories must differ.")
    if args.recursive and output_resolved.is_relative_to(input_resolved):
        raise ValueError("With --recursive, output cannot be inside input.")


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    calibration = load_calibration(args.calibration)
    input_paths = collect_images(args.input, args.recursive)
    if not input_paths:
        raise RuntimeError(f"No supported images found in: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    occupied = occupied_indices(args.output)
    output_index = next_available_index(occupied, args.start_index)
    contexts: dict[tuple[int, int], UndistortionContext] = {}

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Images: {len(input_paths)}")
    print(f"Calibration RMS: {calibration.rms:.6f} px")
    print(f"Distortion model: {calibration.distortion_model}")

    succeeded = 0
    failed = 0
    for position, input_path in enumerate(input_paths, start=1):
        try:
            distorted = read_image(input_path)
            image_size = (distorted.shape[1], distorted.shape[0])
            context = contexts.get(image_size)
            if context is None:
                context = create_context(
                    calibration,
                    image_size,
                    args.mode,
                    args.alpha,
                    args.allow_size_scaling,
                )
                contexts[image_size] = context

            # 根据预计算映射表执行真正的去畸变。
            # INTER_LINEAR 是常用双线性插值：速度和画质平衡较好。
            # 这里通常不需要修改；它影响重采样质量，不决定几何比例。
            undistorted = cv2.remap(
                distorted,
                context.map1,
                context.map2,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

            output_index = next_available_index(occupied, output_index)
            output_name = f"{output_index:0{args.filename_width}d}.jpg"
            output_path = args.output / output_name
            write_jpeg(output_path, undistorted, args.jpeg_quality)
            occupied.add(output_index)
            output_index += 1
            succeeded += 1
            print(f"[{position}/{len(input_paths)}] {input_path.name} -> {output_name}")
        except Exception as error:  # Continue batch unless explicitly requested.
            failed += 1
            print(f"[{position}/{len(input_paths)}] FAILED {input_path}: {error}", file=sys.stderr)
            if args.stop_on_error:
                raise

    print(f"\nDone: succeeded={succeeded}, failed={failed}")
    return 0 if succeeded else 1


def main() -> int:
    try:
        return run(parse_args())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
