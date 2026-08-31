# -*- coding: utf-8 -*-
"""
批量修改 train/val txt 中的数据集路径前缀。

例如把：
H:/train_data/images_cvat00_dataset/images/train/000001.jpg

改成：
/root/autodl-tmp/images_cvat00_dataset/images/train/000001.jpg

支持：
- 多个 txt 文件一起处理
- Windows 的 / 和 \ 路径
- 默认生成新文件，不覆盖原文件
- 输出修改数量和未匹配数量
"""

from pathlib import Path


# ============================================================
# 1. 参数设置 —— 一般只改这里
# ============================================================

# 要处理的 txt 文件
INPUT_FILES = [
    Path(r"H:\image_process_data\train_grouped.txt"),
    Path(r"H:\image_process_data\val_grouped.txt"),
]

# 原来的公共前缀
OLD_PREFIX = "H:/train_data/"

# AutoDL 上的新前缀
NEW_PREFIX = "/root/autodl-tmp/"

# 是否直接覆盖原文件
# False：生成 xxx_autodl.txt
# True ：直接修改原文件
OVERWRITE = False

# 不覆盖时，新文件名后缀
OUTPUT_SUFFIX = "_autodl"

# 是否把所有反斜杠 \ 统一转换成 /
NORMALIZE_SLASH = True

# 是否显示前几条修改示例
SHOW_EXAMPLES = True

# 最多显示几条示例
MAX_EXAMPLES = 5


# ============================================================
# 2. 路径转换
# ============================================================

def convert_path(line: str):
    """
    返回：
        new_line, changed
    """
    original = line.strip()

    if not original:
        return "", False

    path_text = original

    if NORMALIZE_SLASH:
        path_text = path_text.replace("\\", "/")

    old_prefix = OLD_PREFIX.replace("\\", "/")
    new_prefix = NEW_PREFIX.replace("\\", "/")

    # 只替换开头，避免误改路径中间的同名字符串
    if path_text.lower().startswith(old_prefix.lower()):
        new_path = new_prefix + path_text[len(old_prefix):]
        return new_path, True

    return path_text, False


# ============================================================
# 3. 单个文件处理
# ============================================================

def process_file(input_file: Path):
    if not input_file.exists():
        print(f"[跳过] 文件不存在：{input_file}")
        return

    lines = input_file.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    output_lines = []
    changed_count = 0
    unchanged_count = 0
    examples = []

    for line in lines:
        new_line, changed = convert_path(line)

        output_lines.append(new_line)

        if changed:
            changed_count += 1

            if len(examples) < MAX_EXAMPLES:
                examples.append(
                    (line.strip(), new_line)
                )
        elif line.strip():
            unchanged_count += 1

    if OVERWRITE:
        output_file = input_file
    else:
        output_file = input_file.with_name(
            input_file.stem
            + OUTPUT_SUFFIX
            + input_file.suffix
        )

    output_file.write_text(
        "\n".join(output_lines) + "\n",
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print(f"输入文件：{input_file}")
    print(f"输出文件：{output_file}")
    print(f"总行数：  {len(lines)}")
    print(f"已修改：  {changed_count}")
    print(f"未匹配：  {unchanged_count}")

    if SHOW_EXAMPLES and examples:
        print()
        print("修改示例：")

        for old, new in examples:
            print(f"  原：{old}")
            print(f"  新：{new}")
            print()

    if unchanged_count > 0:
        print(
            "[提示] 有未匹配行。请检查这些路径是否不是以 "
            f"{OLD_PREFIX!r} 开头。"
        )


# ============================================================
# 4. 主程序
# ============================================================

def main():
    print("=" * 70)
    print("批量修改数据集路径前缀")
    print("=" * 70)
    print(f"旧前缀：{OLD_PREFIX}")
    print(f"新前缀：{NEW_PREFIX}")
    print(f"覆盖原文件：{OVERWRITE}")

    for file_path in INPUT_FILES:
        process_file(file_path)

    print()
    print("=" * 70)
    print("全部处理完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
