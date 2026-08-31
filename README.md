# 图像处理与 YOLO OBB 工具集

本目录保存图像预处理、视频抽帧、YOLO OBB 数据集整理、训练、预测和质量分析脚本。

> 文档版本：`v0.1.0`  
> 最近更新：`2026-08-31`  
> Git 当前基线：`8b14161 图像处理`

## 快速索引

| 阶段 | 脚本 | 作用 |
|---|---|---|
| 视频预处理 | `mp4tojpg.py` | MP4 按帧或按时间抽取 JPG |
| 图像预处理 | `image_calibration.py` | 根据 OpenCV 标定文件批量去畸变 |
| 图像预处理 | `rotate_images_check_16_9.py` | 检查 16:9，必要时旋转竖图 |
| 图像清理 | `image_clean.py` | 按配置批量清空目录 |
| 数据过滤 | `parcel_dataset_filter.py` | 用 YOLO OBB 和图像质量规则筛选样本 |
| 数据统计 | `parcel_dataset_analyzer_final_cn_plot_fixed.py` | 分析包裹数量、尺寸、角度并绘图 |
| 数据集划分 | `split_multi_obb_dataset.py` | 多个 OBB 数据集随机划分 train/val |
| 分组清单 | `generate_grouped_yolo_obb.py` | 生成分组训练/验证清单和 `data.yaml` |
| 数据集落盘 | `split_existing_yolo_obb_by_grouped_txt.py` | 按已有清单复制或移动数据集 |
| 路径迁移 | `change_dataset_path_prefix.py` | 批量替换 TXT 清单路径前缀 |
| 数据审计 | `audit_and_sync_yolo_obb_dataset.py` | 检查并同步图片、标签和清单 |
| OBB 质量分析 | `obb_dataset_problem_analyzer.py` | 识别漏检、误检、低 IoU、角度异常等问题 |
| 模型训练 | `train.py` | 使用 Ultralytics 训练 YOLO11 OBB |
| 模型预测 | `predict_obb.py` | 批量执行 YOLO OBB 预测并保存可视化结果 |

## 推荐工作流

```text
视频/原始图片
    ├─ mp4tojpg.py
    ├─ image_calibration.py
    └─ rotate_images_check_16_9.py
             ↓
parcel_dataset_filter.py（筛选样本）
             ↓
split_multi_obb_dataset.py / generate_grouped_yolo_obb.py
             ↓
train.py（训练）
             ↓
predict_obb.py / obb_dataset_problem_analyzer.py（预测与问题分析）
```

如果已经有分组后的 `train_grouped.txt` 和 `val_grouped.txt`，可以跳过随机划分，直接使用 `split_existing_yolo_obb_by_grouped_txt.py` 整理数据。

## 运行前准备

建议使用 Python 3.10+，并安装项目实际使用的依赖：

```text
opencv-python
numpy
pandas
matplotlib
tqdm
ultralytics
torch
```

大多数脚本采用“直接修改文件顶部配置”的方式；`image_calibration.py` 和 `predict_obb.py` 同时支持命令行参数。运行前重点检查：数据根目录、模型路径、输出目录、`DEVICE`、是否覆盖原文件。

## 脚本说明

### 1. `mp4tojpg.py`

- 功能：读取 `INPUT_DIR` 下的 MP4 视频并抽帧为 JPG。
- 主要配置：`EXTRACT_MODE` 可选 `frame` 或 `second`；分别使用 `FRAME_INTERVAL` 或 `SECOND_INTERVAL`。
- 默认路径：`H:\image_process_data\video_test` → `H:\image_process_data\video_frames_output`。
- 输出：文件名包含视频名、图片序号、帧号和时间戳；`CREATE_VIDEO_SUBFOLDER` 控制是否按视频分目录。
- 适用场景：视频抽帧、构建后续标定或检测输入。

### 2. `image_calibration.py`

- 功能：根据 OpenCV YAML 标定参数批量去畸变。
- 关键输入：原始图片目录、`camera_single.yml`、输出目录。
- 关键参数：`--mode match-reference|preserve-geometry`、`--recursive`、`--jpeg-quality`、`--start-index`、`--stop-on-error`。
- 默认路径：`H:\image_process_data\images_calibration_input` → `H:\image_process_data\images_calibration_output`。
- 注意：更换相机、镜头或分辨率时，应更换对应 YAML；不要把输出目录放回输入目录。

### 3. `rotate_images_check_16_9.py`

- 功能：检查图片是否为 16:9；对竖图按配置旋转，并将不符合比例的文件记录到警告文件。
- 关键配置：`ROTATE_CLOCKWISE`、`JPEG_QUALITY`、`ASPECT_RATIO_TOLERANCE`、`REMOVE_NOT_16_9`。
- 默认路径：`H:\image_process_data\images_rotate_input` → `H:\image_process_data\images_rotate_output`。
- 注意：`REMOVE_NOT_16_9=True` 会移除不符合要求的图片，执行前应确认备份和输出策略。

### 4. `image_clean.py`

- 功能：批量删除配置目录内的文件，可递归处理子目录。
- 关键配置：`TARGET_DIRS`、`RECURSIVE`、`DELETE_EMPTY_DIRS`、`DRY_RUN`。
- 安全建议：首次执行将 `DRY_RUN=True`，确认目标目录无误后再改为 `False`。

### 5. `parcel_dataset_filter.py`

- 功能：结合 YOLO OBB 预测、清晰度、亮度、目标位置、尺寸、角度和重复图像规则筛选样本。
- 默认输入：`H:\image_process_data\images_filter_input`。
- 默认输出：`H:\image_process_data\images_filter_output`。
- 关键配置：`MODEL_PATH`、`DEVICE`、`IMGSZ`、`BATCH_SIZE`、`COPY_MODE`、`DRY_RUN`。
- 输出内容：保留/排除样本、原因统计及筛选报告（以脚本实际输出为准）。

### 6. `parcel_dataset_analyzer_final_cn_plot_fixed.py`

- 功能：分析 YOLO OBB 数据集中的图片数量、目标数量、目标尺寸、角度和场景类别，并生成中文统计图。
- 关键配置：`DATASET_DIRS`、`OUTPUT_DIR`、`MAX_FILES`。
- 输入格式：数据集目录需包含 `images` 和对应标签目录；标签按 YOLO OBB 格式读取。
- 输出：分析表格、JSON/CSV 数据及 `plots` 图表目录。

### 7. `split_multi_obb_dataset.py`

- 功能：收集多个 `images_cvatXX_dataset` 数据集，按固定随机种子划分 train/val，并生成路径清单。
- 关键配置：`DATASET_DIRS`、`OUTPUT_DIR`、`TRAIN_RATIO`、`RANDOM_SEED`、`REQUIRE_LABEL`。
- 默认划分比例：`0.80`；默认随机种子：`24`。
- 注意：固定随机种子可保证重复运行时划分稳定；数据变化后应重新记录版本。

### 8. `generate_grouped_yolo_obb.py`

- 功能：自动发现多个 CVAT 数据集，生成分组 `train_grouped.txt`、`val_grouped.txt`、汇总文件和可选 `data.yaml`。
- 关键配置：`DATASET_ROOT`、`DATASET_GLOB`、`PATH_MODE`、`CUSTOM_OUTPUT_ROOT`、`REQUIRE_LABEL`、`CLASS_NAMES`。
- 典型根目录：`H:\train_data` 或 AutoDL 的 `/root/autodl-tmp`。
- 注意：确认 `PATH_MODE` 与训练机器一致；跨 Windows/AutoDL 使用时配合路径替换脚本。

### 9. `split_existing_yolo_obb_by_grouped_txt.py`

- 功能：按已有 train/val 分组清单，把图片和标签复制或移动到目标数据集结构。
- 关键配置：`DATASET_ROOT`、`TRAIN_LIST_FILE`、`VAL_LIST_FILE`、`MOVE_FILES`、`DRY_RUN`、`UPDATE_DATA_YAML`。
- 默认行为：`MOVE_FILES=True`，会移动文件；预演时必须先设置 `DRY_RUN=True`。
- 注意：脚本支持备份元数据；跨环境清单可通过绝对/相对路径配置控制。

### 10. `change_dataset_path_prefix.py`

- 功能：把 TXT 清单中的公共路径前缀从 Windows 路径替换为 AutoDL/Linux 路径。
- 默认替换：`H:/train_data/` → `/root/autodl-tmp/`。
- 关键配置：`INPUT_FILES`、`OLD_PREFIX`、`NEW_PREFIX`、`OVERWRITE`、`OUTPUT_SUFFIX`。
- 默认行为：不覆盖原文件，生成带 `_autodl` 后缀的新文件；只有确认无误后才使用 `OVERWRITE=True`。

### 11. `audit_and_sync_yolo_obb_dataset.py`

- 功能：审计分组数据集、清单、图片和标签的一致性，并按配置同步或清理异常项。
- 关键配置：`DATASET_ROOT`、`DATASET_GLOB`、`DRY_RUN`、备份选项、空目录处理选项。
- 推荐执行顺序：先 `DRY_RUN=True` 查看计划，再决定是否实际修改。
- 注意：这是会影响文件的维护脚本，执行前应保留 Git 提交或数据备份。

### 12. `obb_dataset_problem_analyzer.py`

- 功能：使用模型对验证集做 OBB 评估，定位漏检、误检、低置信度、低 IoU 和角度误差等问题。
- 关键配置：`MODEL_PATH`、`VAL_TXT_PATH`、`OUTPUT_DIR`、`DEVICE`、`IMGSZ`、阈值参数。
- 默认输出：`H:\train_data\obb_error_output`。
- 结果用途：根据问题图像回查数据标注、图像质量和模型训练效果。

### 13. `train.py`

- 功能：使用 Ultralytics YOLO11 OBB 训练模型。
- 默认模型：`yolo11n-obb.pt`。
- 默认数据：`./data.yaml`；训练轮数 `300`，`imgsz=960`，`batch=16`，`device=0`，优化器为 `AdamW`。
- 增强重点：旋转、平移、缩放、翻转和 HSV；`mosaic`、`mixup`、`cutmix` 当前为关闭状态。
- 注意：修改数据集、模型、设备或关键超参数时，应在版本记录中写明。

### 14. `predict_obb.py`

- 功能：批量读取图片并执行 YOLO OBB 预测，保存检测可视化结果。
- 关键配置：`MODEL_PATH`、`SOURCE_DIR`、`OUTPUT_DIR`、`INPUT_MODE`、`RECT_IMGSZ`、`CONF_THRESHOLD`、`IOU_THRESHOLD`、`DEVICE`。
- 支持输入模式：脚本内定义的 `square` 或矩形推理模式；运行时也可用 `--model`、`--source`、`--output` 覆盖路径。
- 注意：预测结果目录不要作为下一次输入目录，避免重复处理结果图。

## 版本管理规范

采用 `v主版本.次版本.修订版本`：

- 主版本：目录结构、数据格式或运行方式不兼容变化。
- 次版本：新增脚本、新流程或兼容性功能。
- 修订版本：修复 bug、调整阈值、补充文档或小范围优化。

每次更新只需：

1. 修改本文顶部“文档版本”和“最近更新”。
2. 在下面的更新记录最上方新增一条版本说明。
3. 提交信息使用 `vX.Y.Z 简短说明`。
4. 需要发布节点时创建同名 Git tag。

版本记录模板：

```markdown
### vX.Y.Z - YYYY-MM-DD

- 新增：
- 修改：
- 修复：
- 删除：
- 影响脚本/目录：
- 运行验证：
- 兼容性说明：
```

Git 标签示例：

```text
git add README.md <changed-files>
git commit -m "v0.2.0 新增预测流程说明"
git tag -a v0.2.0 -m "v0.2.0 新增预测流程说明"
git push origin main --tags
```

## 更新记录

### v0.1.0 - 2026-08-31

- 新增：建立本 README。
- 新增：登记当前 14 个 Python 脚本的功能、输入、输出、主要配置和注意事项。
- 新增：补充推荐工作流、危险操作提示和版本记录模板。
- 说明：未修改任何现有 Python 文件；当前工作区原有已修改和未跟踪文件均保留。

<!-- 新版本请在此处上方追加，保持时间倒序。 -->

