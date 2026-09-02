import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# ============================================================
# 配置区
# ============================================================

# 检测结果图片所在目录
INPUT_DIR = r"F:\image_process_data\video_frames_output"

# 输出视频
OUTPUT_VIDEO = r"F:\image_process_data\detection_result_960.mp4"

# 图片格式
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# 是否递归搜索子文件夹
RECURSIVE = False

# ============================================================
# FPS 设置
# ============================================================

# "manual"       = 手动指定 FPS
# "source_video" = 从原始视频自动读取 FPS
FPS_MODE = "manual"

# 手动模式时使用
OUTPUT_FPS = 30.0

# 自动读取原视频 FPS 时使用
ORIGINAL_VIDEO = r""

# 如果原视频每 N 帧抽 1 张做检测，
# 想保持原视频实际播放速度：
# 输出 FPS = 原视频 FPS / FRAME_INTERVAL
FRAME_INTERVAL = 1

# ============================================================
# FFmpeg
# ============================================================

FFMPEG_EXE = r"F:\image_process_data\ffmpeg\bin\ffmpeg.exe"
FFPROBE_EXE = r"F:\image_process_data\ffmpeg\bin\ffprobe.exe"

# 没加入 PATH 时可以写完整路径，例如：
# FFMPEG_EXE = r"F:\ffmpeg\bin\ffmpeg.exe"
# FFPROBE_EXE = r"F:\ffmpeg\bin\ffprobe.exe"

# H.264 编码质量
CRF = 18
PRESET = "fast"
PIX_FMT = "yuv420p"


def natural_key(path):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", path.name)
    ]


def executable_exists(exe):
    if Path(exe).is_file():
        return True
    return shutil.which(exe) is not None


def check_environment():
    if not executable_exists(FFMPEG_EXE):
        print("[错误] 找不到 ffmpeg。")
        print("请安装 FFmpeg，或设置 FFMPEG_EXE 的完整路径。")
        return False

    if FPS_MODE == "source_video" and not executable_exists(FFPROBE_EXE):
        print("[错误] 找不到 ffprobe。")
        return False

    return True


def find_images():
    input_dir = Path(INPUT_DIR)

    if not input_dir.exists():
        raise FileNotFoundError(f"图片目录不存在：{input_dir}")

    iterator = input_dir.rglob("*") if RECURSIVE else input_dir.iterdir()

    images = [
        p for p in iterator
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    images.sort(key=natural_key)
    return images


def get_source_fps(video_path):
    command = [
        FFPROBE_EXE,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "json",
        str(video_path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    if result.returncode != 0:
        raise RuntimeError("读取原视频 FPS 失败：\n" + result.stderr)

    data = json.loads(result.stdout)
    streams = data.get("streams", [])

    if not streams:
        raise RuntimeError("原视频没有视频流。")

    stream = streams[0]
    fps_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate")

    if not fps_text or fps_text == "0/0":
        raise RuntimeError("无法获取原视频 FPS。")

    a, b = fps_text.split("/")
    return float(a) / float(b)


def determine_fps():
    if FPS_MODE == "manual":
        fps = float(OUTPUT_FPS)
        if fps <= 0:
            raise ValueError("OUTPUT_FPS 必须大于 0。")
        return fps

    if FPS_MODE == "source_video":
        if not ORIGINAL_VIDEO:
            raise ValueError("请填写 ORIGINAL_VIDEO。")

        source_fps = get_source_fps(ORIGINAL_VIDEO)
        interval = max(1, int(FRAME_INTERVAL))
        fps = source_fps / interval

        print(f"原视频 FPS：{source_fps:.6f}")
        print(f"抽帧间隔：每 {interval} 帧取 1 张")
        print(f"输出 FPS：{fps:.6f}")

        return fps

    raise ValueError(f"未知 FPS_MODE：{FPS_MODE}")


def escape_concat_path(path):
    text = str(path.resolve()).replace("\\", "/")
    return text.replace("'", "'\\''")


def create_concat_file(images, fps):
    frame_duration = 1.0 / fps

    f = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
        newline="\n"
    )

    temp_path = Path(f.name)

    try:
        for image in images:
            p = escape_concat_path(image)
            f.write(f"file '{p}'\n")
            f.write(f"duration {frame_duration:.12f}\n")

        # concat 模式下最后一帧再写一次，确保最后 duration 生效
        if images:
            p = escape_concat_path(images[-1])
            f.write(f"file '{p}'\n")
    finally:
        f.close()

    return temp_path


def images_to_video():
    if not check_environment():
        return

    images = find_images()

    if not images:
        print(f"[错误] 没有找到图片：{INPUT_DIR}")
        return

    fps = determine_fps()

    output_video = Path(OUTPUT_VIDEO)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    duration = len(images) / fps

    print("=" * 72)
    print("检测结果图片 -> MP4")
    print("=" * 72)
    print(f"图片目录：{INPUT_DIR}")
    print(f"图片数量：{len(images)}")
    print(f"第一张：{images[0].name}")
    print(f"最后一张：{images[-1].name}")
    print(f"输出 FPS：{fps:.6f}")
    print(f"预计时长：{duration:.2f} 秒")
    print(f"输出视频：{OUTPUT_VIDEO}")
    print("=" * 72)

    concat_file = create_concat_file(images, fps)

    command = [
        FFMPEG_EXE,
        "-hide_banner",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-r", f"{fps:.12f}",
        "-c:v", "libx264",
        "-preset", PRESET,
        "-crf", str(CRF),
        "-pix_fmt", PIX_FMT,
        "-movflags", "+faststart",
        str(output_video)
    ]

    print()
    print("[FFmpeg] 开始生成视频...")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        stderr_lines = []

        for line in process.stderr:
            stderr_lines.append(line)

            if "frame=" in line:
                print("\r" + line.rstrip(), end="", flush=True)

        return_code = process.wait()
        print()

        if return_code == 0 and output_video.exists():
            size_mb = output_video.stat().st_size / 1024 / 1024

            print()
            print("=" * 72)
            print("[完成] 视频生成成功")
            print(f"视频：{output_video}")
            print(f"大小：{size_mb:.2f} MB")
            print(f"FPS：{fps:.6f}")
            print(f"图片：{len(images)} 张")
            print(f"预计时长：{duration:.2f} 秒")
            print("=" * 72)

        else:
            print("[错误] FFmpeg 生成视频失败。")
            print("".join(stderr_lines[-30:]))

    finally:
        try:
            concat_file.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    images_to_video()
