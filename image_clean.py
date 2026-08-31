from pathlib import Path


# ============================================================
# 参数设置
# ============================================================

# 要批量清空的目标文件夹
# 可以一直往下面添加
TARGET_DIRS = [
    Path(r"H:\image_process_data\images_calibration_input"),
    Path(r"H:\image_process_data\images_calibration_output"),
    Path(r"H:\image_process_data\images_filter_input"),
    Path(r"H:\image_process_data\images_filter_output"),
    Path(r"H:\image_process_data\images_predict_input"),
    Path(r"H:\image_process_data\images_predict_output"),
    Path(r"H:\image_process_data\images_rotate_input"),
    Path(r"H:\image_process_data\images_rotate_output"),
    # Path(r"H:\image_process_data\xxx"),
]


# 是否递归删除子文件夹中的文件
# True  = 删除目标文件夹及所有子文件夹中的文件
# False = 只删除目标文件夹第一层中的文件
RECURSIVE = True


# 是否同时删除清空后的空子文件夹
# True  = 文件删除后，把空子文件夹也删除
# False = 保留原有文件夹结构
#
# 注意：
# 不管这里设置什么，都不会删除 TARGET_DIRS 中填写的目标文件夹本身
DELETE_EMPTY_DIRS = True


# 安全模式
# True  = 只显示准备删除哪些文件，不真正删除
# False = 真正执行删除
DRY_RUN = False


# ============================================================
# 清空单个文件夹
# ============================================================

def clear_directory(target_dir: Path):

    print()
    print("=" * 80)
    print(f"开始处理：{target_dir}")
    print("=" * 80)

    # --------------------------------------------------------
    # 检查目标路径
    # --------------------------------------------------------

    if not target_dir.exists():
        print(f"[跳过] 目标文件夹不存在：{target_dir}")
        return 0, 0

    if not target_dir.is_dir():
        print(f"[跳过] 目标路径不是文件夹：{target_dir}")
        return 0, 0


    # --------------------------------------------------------
    # 查找文件
    # --------------------------------------------------------

    if RECURSIVE:

        # rglob("*") 会递归查找所有子目录
        files = [
            p
            for p in target_dir.rglob("*")
            if p.is_file()
        ]

    else:

        # iterdir() 只处理当前文件夹第一层
        files = [
            p
            for p in target_dir.iterdir()
            if p.is_file()
        ]


    print(f"找到文件数量：{len(files)}")
    print()


    # --------------------------------------------------------
    # 删除文件
    # --------------------------------------------------------

    deleted_count = 0
    failed_count = 0


    for file_path in files:

        try:

            if DRY_RUN:

                print(f"[预览] 将删除：{file_path}")

            else:

                file_path.unlink()

                print(f"[已删除] {file_path}")

            deleted_count += 1


        except Exception as e:

            failed_count += 1

            print(f"[删除失败] {file_path}")
            print(f"原因：{e}")


    # --------------------------------------------------------
    # 删除空子文件夹
    # --------------------------------------------------------

    if DELETE_EMPTY_DIRS and RECURSIVE:

        directories = [
            p
            for p in target_dir.rglob("*")
            if p.is_dir()
        ]


        # 必须从最深层文件夹开始删除
        # 否则父文件夹里面还有子文件夹时无法删除
        directories.sort(
            key=lambda p: len(p.parts),
            reverse=True
        )


        for directory in directories:

            try:

                # 判断当前文件夹是否已经为空
                if not any(directory.iterdir()):

                    if DRY_RUN:

                        print(
                            f"[预览] 将删除空文件夹："
                            f"{directory}"
                        )

                    else:

                        directory.rmdir()

                        print(
                            f"[已删除空文件夹] "
                            f"{directory}"
                        )


            except Exception as e:

                print(
                    f"[文件夹处理失败] "
                    f"{directory}"
                )

                print(f"原因：{e}")


    # --------------------------------------------------------
    # 当前文件夹统计
    # --------------------------------------------------------

    print()

    if DRY_RUN:

        print(
            f"[预览完成] "
            f"{target_dir}"
        )

        print(
            f"准备删除文件："
            f"{deleted_count}"
        )

    else:

        print(
            f"[处理完成] "
            f"{target_dir}"
        )

        print(
            f"成功删除："
            f"{deleted_count}"
        )

        print(
            f"删除失败："
            f"{failed_count}"
        )


    return deleted_count, failed_count


# ============================================================
# 批量清空文件夹
# ============================================================

def clear_directories(target_dirs):

    total_deleted = 0
    total_failed = 0


    print()
    print("#" * 80)
    print("开始批量清空文件夹")
    print("#" * 80)

    print(f"目标文件夹数量：{len(target_dirs)}")

    if DRY_RUN:
        print("当前模式：DRY_RUN=True，仅预览，不真正删除")
    else:
        print("当前模式：DRY_RUN=False，将真正删除文件")

    print()


    # --------------------------------------------------------
    # 逐个处理目标文件夹
    # --------------------------------------------------------

    for index, target_dir in enumerate(
        target_dirs,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(target_dirs)}] "
            f"{target_dir}"
        )

        deleted_count, failed_count = clear_directory(
            target_dir
        )

        total_deleted += deleted_count
        total_failed += failed_count


    # --------------------------------------------------------
    # 最终统计
    # --------------------------------------------------------

    print()
    print("#" * 80)
    print("批量处理完成")
    print("#" * 80)

    print(f"目标文件夹数量：{len(target_dirs)}")


    if DRY_RUN:

        print(
            f"共找到准备删除的文件："
            f"{total_deleted}"
        )

        print()
        print(
            "当前 DRY_RUN=True，"
            "没有真正删除任何文件。"
        )

        print(
            "确认无误后，把："
        )

        print(
            "DRY_RUN = True"
        )

        print(
            "改成："
        )

        print(
            "DRY_RUN = False"
        )

    else:

        print(
            f"成功删除文件："
            f"{total_deleted}"
        )

        print(
            f"删除失败文件："
            f"{total_failed}"
        )


    print("#" * 80)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    clear_directories(TARGET_DIRS)