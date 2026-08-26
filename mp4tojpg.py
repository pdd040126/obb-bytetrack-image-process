import cv2
import os
from pathlib import Path


# ============================================================
# 配置区 —— 一般只需要修改这里
# ============================================================

# MP4视频所在目录
INPUT_DIR = r"H:\image_data\video_test"

# JPG图片保存目录
OUTPUT_DIR = r"H:\image_data\video_frames_output"

# 抽帧模式：
# "frame"  = 每隔固定帧数保存一张
# "second" = 每隔固定秒数保存一张
EXTRACT_MODE = "second"

# 如果 EXTRACT_MODE = "frame"
# 每隔多少帧保存一张
FRAME_INTERVAL = 30

# 如果 EXTRACT_MODE = "second"
# 每隔多少秒保存一张
SECOND_INTERVAL = 1.0

# JPG质量，范围 0~100
# 数据集一般 90~95 足够
JPG_QUALITY = 90

# 是否给每个视频单独创建文件夹
# True:
# video_frames/
#   video1/
#   video2/
#
# False:
# 所有图片直接放到 video_frames/
CREATE_VIDEO_SUBFOLDER = False


# ============================================================
# 抽帧函数
# ============================================================

def extract_frames(video_path, output_root):

    video_path = Path(video_path)
    video_name = video_path.stem

    if CREATE_VIDEO_SUBFOLDER:
        output_dir = Path(output_root) / video_name
    else:
        output_dir = Path(output_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[错误] 无法打开视频：{video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print(f"[错误] 无法获取视频FPS：{video_path}")
        cap.release()
        return

    duration = total_frames / fps

    print("=" * 70)
    print(f"视频：{video_path.name}")
    print(f"FPS：{fps:.2f}")
    print(f"总帧数：{total_frames}")
    print(f"时长：{duration:.2f} 秒")

    if EXTRACT_MODE == "frame":
        interval_frames = max(1, int(FRAME_INTERVAL))

    elif EXTRACT_MODE == "second":
        interval_frames = max(1, int(round(fps * SECOND_INTERVAL)))

    else:
        print(f"[错误] 未知抽帧模式：{EXTRACT_MODE}")
        cap.release()
        return

    print(f"实际每隔 {interval_frames} 帧保存一张")

    frame_index = 0
    save_index = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % interval_frames == 0:

            time_seconds = frame_index / fps

            # 文件名同时包含：
            # 视频名称
            # 图片序号
            # 原视频帧号
            # 时间
            filename = (
                f"{video_name}"
                f"_img_{save_index:06d}"
                f"_frame_{frame_index:08d}"
                f"_time_{time_seconds:010.3f}s.jpg"
            )

            save_path = output_dir / filename

            cv2.imwrite(
                str(save_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY]
            )

            save_index += 1

        frame_index += 1

        # 显示进度
        if frame_index % 500 == 0:
            progress = frame_index / total_frames * 100
            print(
                f"\r处理进度：{progress:6.2f}% "
                f"| 已保存 {save_index} 张",
                end=""
            )

    cap.release()

    print()
    print(f"[完成] {video_path.name}")
    print(f"共保存：{save_index} 张 JPG")
    print(f"保存位置：{output_dir}")


# ============================================================
# 主程序
# ============================================================

def main():

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"[错误] 输入目录不存在：{input_dir}")
        return

    # 搜索 MP4，大小写都兼容
    videos = []

    for file in input_dir.iterdir():
        if file.is_file() and file.suffix.lower() == ".mp4":
            videos.append(file)

    videos.sort()

    if not videos:
        print(f"[提示] 没有找到 MP4 视频：{input_dir}")
        return

    print(f"发现 {len(videos)} 个 MP4 视频。")

    for i, video in enumerate(videos, start=1):

        print()
        print(f"[{i}/{len(videos)}] 开始处理")

        extract_frames(
            video,
            output_dir
        )

    print()
    print("=" * 70)
    print("全部视频处理完成。")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()