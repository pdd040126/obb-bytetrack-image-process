import json
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path


# ============================================================
# 配置区 —— 一般只需要修改这里
# ============================================================

# MP4 视频所在目录
INPUT_DIR = r"F:\image_process_data\video_test"

# JPG 图片保存目录
OUTPUT_DIR = r"F:\image_process_data\video_frames_output"


# ============================================================
# FFmpeg / FFprobe
# ============================================================

# 如果 ffmpeg 和 ffprobe 已经加入 Windows PATH，保持下面这样即可
FFMPEG_EXE = r"F:\image_process_data\ffmpeg\bin\ffmpeg.exe"
FFPROBE_EXE = r"F:\image_process_data\ffmpeg\bin\ffprobe.exe"

# 如果没有加入 PATH，可以改成完整路径，例如：
# FFMPEG_EXE = r"D:\ffmpeg\bin\ffmpeg.exe"
# FFPROBE_EXE = r"D:\ffmpeg\bin\ffprobe.exe"


# ============================================================
# 抽帧设置
# ============================================================

# "frame"  = 每隔固定“成功解码帧数”保存一张
# "second" = 按时间间隔保存一张
EXTRACT_MODE = "frame"

# EXTRACT_MODE = "frame" 时使用
# 1 = 每一帧都保存
# 5 = 每 5 个成功解码帧保存一张
FRAME_INTERVAL = 1

# EXTRACT_MODE = "second" 时使用
# 1.0 = 大约每秒保存一张
SECOND_INTERVAL = 1.0


# ============================================================
# JPG 设置
# ============================================================

# FFmpeg MJPEG 的 q 值：
# 2 = 很高质量
# 3 = 高质量
# 4~5 = 文件更小
#
# 它和 OpenCV 的 JPG_QUALITY=90 不是同一套数值。
JPG_QSCALE = 3


# ============================================================
# 输出设置
# ============================================================

# 是否每个视频创建独立文件夹
#
# True:
# video_frames_output/
#   video1/
#       video1_img_000001.jpg
#   video2/
#       video2_img_000001.jpg
#
# False:
# 所有 JPG 放在同一个目录
CREATE_VIDEO_SUBFOLDER = False

# 重新运行时，是否先删除当前视频以前生成的 JPG
CLEAR_EXISTING_FRAMES = True

# 是否保存 FFmpeg 日志
SAVE_FFMPEG_LOG = True

LOG_DIR_NAME = "_ffmpeg_logs"


# ============================================================
# 容错设置
# ============================================================

# 忽略可恢复的解码错误
IGNORE_DECODE_ERRORS = True

# 丢弃明显损坏的数据包
DISCARD_CORRUPT_PACKETS = True

# 生成缺失时间戳
GENERATE_PTS = True


# ============================================================
# 工具函数
# ============================================================

def executable_exists(exe):
    """
    检查 ffmpeg / ffprobe 是否可执行。
    """
    if Path(exe).is_file():
        return True

    return shutil.which(exe) is not None


def check_environment():
    """
    检查 FFmpeg 和 FFprobe。
    """
    print("=" * 72)
    print("检查运行环境")
    print("=" * 72)

    ffmpeg_ok = executable_exists(FFMPEG_EXE)
    ffprobe_ok = executable_exists(FFPROBE_EXE)

    print(f"FFmpeg : {'OK' if ffmpeg_ok else '未找到'}")
    print(f"FFprobe: {'OK' if ffprobe_ok else '未找到'}")

    if not ffmpeg_ok:
        print()
        print("[错误] 找不到 ffmpeg。")
        print("请把 ffmpeg.exe 加入 PATH，或者修改 FFMPEG_EXE 为完整路径。")
        return False

    if not ffprobe_ok:
        print()
        print("[错误] 找不到 ffprobe。")
        print("请把 ffprobe.exe 加入 PATH，或者修改 FFPROBE_EXE 为完整路径。")
        return False

    try:
        result = subprocess.run(
            [FFMPEG_EXE, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )

        first_line = result.stdout.splitlines()[0] if result.stdout else ""
        if first_line:
            print(first_line)

    except Exception:
        pass

    print()
    return True


def probe_video(video_path):
    """
    使用 ffprobe 获取视频基本信息。
    Python 不使用 OpenCV 打开视频。
    """
    command = [
        FFPROBE_EXE,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,codec_long_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,time_base",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except Exception as e:
        print(f"[警告] ffprobe 执行失败：{e}")
        return {}

    if result.returncode != 0:
        print("[警告] ffprobe 无法完整读取视频信息。")
        if result.stderr.strip():
            print(result.stderr.strip())
        return {}

    try:
        return json.loads(result.stdout)
    except Exception:
        return {}


def parse_fraction(value):
    """
    将 30000/1001 之类的 FFmpeg 帧率字符串转成 float。
    """
    if not value:
        return None

    try:
        f = Fraction(value)
        if f.denominator == 0:
            return None
        return float(f)
    except Exception:
        return None


def print_video_info(video_path, info):
    streams = info.get("streams", [])
    fmt = info.get("format", {})

    print()
    print("=" * 72)
    print(f"视频：{video_path.name}")

    if not streams:
        print("ffprobe 未取得完整视频流信息。")
        return

    s = streams[0]

    codec_name = s.get("codec_name", "unknown")
    codec_long_name = s.get("codec_long_name", "")
    width = s.get("width", "?")
    height = s.get("height", "?")

    fps = parse_fraction(s.get("avg_frame_rate"))
    if not fps or fps <= 0:
        fps = parse_fraction(s.get("r_frame_rate"))

    duration = s.get("duration")
    if duration in (None, "N/A"):
        duration = fmt.get("duration")

    nb_frames = s.get("nb_frames", "N/A")

    print(f"编码：{codec_name} {codec_long_name}".rstrip())
    print(f"分辨率：{width} × {height}")

    if fps:
        print(f"帧率：{fps:.3f} FPS")
    else:
        print("帧率：未知")

    if duration not in (None, "N/A"):
        try:
            print(f"时长：{float(duration):.3f} 秒")
        except Exception:
            print(f"时长：{duration}")

    print(f"容器报告帧数：{nb_frames}")
    print("=" * 72)


def get_output_dir(output_root, video_name):
    output_root = Path(output_root)

    if CREATE_VIDEO_SUBFOLDER:
        output_dir = output_root / video_name
    else:
        output_dir = output_root

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def clear_old_frames(output_dir, video_name):
    """
    删除当前视频以前生成的图片，避免重新运行后残留旧帧。
    """
    if not CLEAR_EXISTING_FRAMES:
        return

    pattern = f"{video_name}_img_*.jpg"
    old_files = list(output_dir.glob(pattern))

    if not old_files:
        return

    print(f"[清理] 删除以前生成的 {len(old_files)} 张 JPG...")

    failed = 0

    for file in old_files:
        try:
            file.unlink()
        except Exception:
            failed += 1

    if failed:
        print(f"[警告] 有 {failed} 张旧图片未能删除。")


def build_video_filter():
    """
    构建 FFmpeg 的视频过滤器。

    注意：
    FFmpeg 的 n 是“成功送入过滤器的解码帧序号”。

    因此对于真正已经丢失的 HEVC 帧，不会伪造原始帧号。
    """

    if EXTRACT_MODE == "frame":
        interval = max(1, int(FRAME_INTERVAL))

        if interval == 1:
            # 不需要 select，所有成功解码出的帧都输出
            return None

        # 逗号需要在 FFmpeg filter expression 中转义
        return f"select=not(mod(n\\,{interval}))"

    elif EXTRACT_MODE == "second":
        interval = float(SECOND_INTERVAL)

        if interval <= 0:
            raise ValueError("SECOND_INTERVAL 必须大于 0")

        # 第一个有效帧输出；
        # 后续只有距离上一次选中帧达到指定秒数才输出。
        return (
            "select="
            f"isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval})"
        )

    else:
        raise ValueError(
            f"未知 EXTRACT_MODE：{EXTRACT_MODE}，"
            '只能是 "frame" 或 "second"'
        )


def count_existing_frames(output_dir, video_name):
    return len(list(output_dir.glob(f"{video_name}_img_*.jpg")))


def analyze_ffmpeg_log(log_text):
    """
    从 FFmpeg 输出中统计常见 HEVC / 解码错误。

    这些错误不一定意味着整个任务失败；
    很多时候 FFmpeg 会在下一个关键帧后继续恢复。
    """

    patterns = {
        "missing_reference": r"Could not find ref with POC",
        "corrupt": r"corrupt",
        "decode_error": r"Error while decoding|error while decoding",
        "invalid_data": r"Invalid data found",
    }

    counts = {}

    for name, pattern in patterns.items():
        counts[name] = len(
            re.findall(pattern, log_text, flags=re.IGNORECASE)
        )

    return counts


# ============================================================
# 核心：FFmpeg 直接解码并直接输出 JPG
# ============================================================

def extract_frames_with_ffmpeg(video_path, output_root):
    """
    整个视频解码过程由 FFmpeg/libavcodec 完成。

    Python 不再调用：
        cv2.VideoCapture(...)
        cap.read()

    所以不会因为 OpenCV 在某个损坏 GOP 上 ret=False
    就把视频后半段直接丢掉。
    """

    video_path = Path(video_path)
    video_name = video_path.stem

    output_dir = get_output_dir(
        output_root,
        video_name
    )

    clear_old_frames(
        output_dir,
        video_name
    )

    info = probe_video(video_path)
    print_video_info(video_path, info)

    vf = build_video_filter()

    output_pattern = (
        output_dir /
        f"{video_name}_img_%06d.jpg"
    )

    command = [
        FFMPEG_EXE,
        "-hide_banner",
        "-y",

        # ----------------------------------------------------
        # 明确使用软件解码路径
        # 不指定 cuda / qsv / d3d11va 等硬件加速
        # ----------------------------------------------------
        "-hwaccel", "none",
    ]

    # --------------------------------------------------------
    # HEVC/H.265 容错
    # --------------------------------------------------------

    if IGNORE_DECODE_ERRORS:
        command += [
            "-err_detect",
            "ignore_err",
        ]

    fflags = []

    if DISCARD_CORRUPT_PACKETS:
        fflags.append("discardcorrupt")

    if GENERATE_PTS:
        fflags.append("genpts")

    if fflags:
        command += [
            "-fflags",
            "+" + "+".join(fflags),
        ]

    # --------------------------------------------------------
    # 输入
    # --------------------------------------------------------

    command += [
        "-i",
        str(video_path),

        # 只处理第一个视频流
        "-map",
        "0:v:0",

        # 不处理音频
        "-an",
    ]

    # --------------------------------------------------------
    # 抽帧过滤
    # --------------------------------------------------------

    if vf:
        command += [
            "-vf",
            vf,
        ]

    # --------------------------------------------------------
    # 非恒定帧率输出：
    # 不为了补时间轴人为复制帧。
    # 对损坏视频非常重要。
    # --------------------------------------------------------

    command += [
        "-fps_mode",
        "vfr",

        # JPG 编码
        "-c:v",
        "mjpeg",

        "-q:v",
        str(JPG_QSCALE),

        # 第一张编号为 000000
        "-start_number",
        "0",

        str(output_pattern),
    ]

    print()
    print("[FFmpeg] 开始直接解码并抽取 JPG")
    print("[FFmpeg] 不使用 OpenCV VideoCapture")
    print()

    # 不使用 shell=True，Windows 路径中有空格也更安全
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    log_lines = []

    # FFmpeg 的状态和错误主要从 stderr 输出
    for line in process.stderr:
        log_lines.append(line)

        stripped = line.rstrip()

        # 把真正有价值的信息显示到控制台，
        # 避免 FFmpeg 输出过于冗长。
        lower = stripped.lower()

        if (
            "could not find ref with poc" in lower
            or "corrupt" in lower
            or "error while decoding" in lower
            or "invalid data" in lower
        ):
            print(f"[FFmpeg 解码警告] {stripped}")

        elif "frame=" in lower:
            # FFmpeg 的常规处理进度
            print(f"\r{stripped}", end="", flush=True)

    return_code = process.wait()

    print()
    print()

    log_text = "".join(log_lines)

    # --------------------------------------------------------
    # 保存日志
    # --------------------------------------------------------

    if SAVE_FFMPEG_LOG:
        log_dir = Path(output_root) / LOG_DIR_NAME
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / f"{video_name}_ffmpeg.log"

        log_path.write_text(
            log_text,
            encoding="utf-8",
            errors="replace"
        )
    else:
        log_path = None

    # --------------------------------------------------------
    # 统计结果
    # --------------------------------------------------------

    saved_count = count_existing_frames(
        output_dir,
        video_name
    )

    warning_counts = analyze_ffmpeg_log(
        log_text
    )

    print("=" * 72)
    print(f"[完成] {video_path.name}")
    print(f"FFmpeg 返回码：{return_code}")
    print(f"成功输出 JPG：{saved_count} 张")

    if warning_counts["missing_reference"] > 0:
        print(
            "HEVC 缺失参考帧警告："
            f"{warning_counts['missing_reference']} 次"
        )

    if warning_counts["corrupt"] > 0:
        print(
            "包含 corrupt 相关警告："
            f"{warning_counts['corrupt']} 次"
        )

    if warning_counts["decode_error"] > 0:
        print(
            "包含 decoding error："
            f"{warning_counts['decode_error']} 次"
        )

    if warning_counts["invalid_data"] > 0:
        print(
            "包含 invalid data："
            f"{warning_counts['invalid_data']} 次"
        )

    if log_path:
        print(f"FFmpeg 日志：{log_path}")

    print(f"图片位置：{output_dir}")

    # FFmpeg 遇到坏 HEVC 时，有时会打印警告但最终仍返回 0。
    # 判断是否“可用”时，输出出 JPG 比是否完全无警告更重要。
    success = saved_count > 0

    if success:
        if any(warning_counts.values()):
            print()
            print(
                "[说明] 原视频存在损坏/丢包，但 FFmpeg 已尽可能"
                "跳过损坏区域并继续解码后续可恢复画面。"
            )
        else:
            print("[状态] 未检测到明显的解码损坏警告。")
    else:
        print()
        print("[失败] 没有成功输出任何 JPG。")

    print("=" * 72)

    return {
        "success": success,
        "return_code": return_code,
        "saved_count": saved_count,
        "warnings": warning_counts,
        "log_path": str(log_path) if log_path else None,
    }


# ============================================================
# 主程序
# ============================================================

def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not check_environment():
        return

    if not input_dir.exists():
        print(f"[错误] 输入目录不存在：{input_dir}")
        return

    videos = sorted(
        file
        for file in input_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() == ".mp4"
    )

    if not videos:
        print(f"[提示] 没有找到 MP4：{input_dir}")
        return

    print()
    print("=" * 72)
    print(f"发现 {len(videos)} 个 MP4 视频")
    print(f"输入目录：{input_dir}")
    print(f"输出目录：{output_dir}")
    print(f"抽帧模式：{EXTRACT_MODE}")

    if EXTRACT_MODE == "frame":
        print(f"每隔 {FRAME_INTERVAL} 个成功解码帧保存一张")
    else:
        print(f"大约每隔 {SECOND_INTERVAL} 秒保存一张")

    print("=" * 72)

    success_count = 0
    failed_count = 0
    failed_videos = []

    total_saved = 0

    for i, video in enumerate(
        videos,
        start=1
    ):
        print()
        print()
        print("#" * 72)
        print(f"[{i}/{len(videos)}] {video.name}")
        print("#" * 72)

        try:
            result = extract_frames_with_ffmpeg(
                video,
                output_dir
            )

            total_saved += result["saved_count"]

            if result["success"]:
                success_count += 1
            else:
                failed_count += 1
                failed_videos.append(video.name)

        except KeyboardInterrupt:
            print()
            print("[用户中断] 已停止。")
            break

        except Exception as e:
            failed_count += 1
            failed_videos.append(video.name)

            print()
            print(f"[严重错误] {video.name}")
            print(e)
            print("继续处理下一个视频。")

    print()
    print()
    print("=" * 72)
    print("全部处理结束")
    print("=" * 72)
    print(f"视频数量：{len(videos)}")
    print(f"成功：{success_count}")
    print(f"失败：{failed_count}")
    print(f"总输出 JPG：{total_saved} 张")
    print(f"输出目录：{output_dir}")

    if failed_videos:
        print()
        print("需要人工检查的视频：")

        for name in failed_videos:
            print(f"  - {name}")

    print("=" * 72)


if __name__ == "__main__":
    main()
