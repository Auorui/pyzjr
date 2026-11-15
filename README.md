<div align="center">
  <img src="resources/pyzjr-logo.png" alt="pyzjr logo" width="520">
  <p><strong>面向计算机视觉与深度学习的 Python 工具包</strong></p>
  <p>
    <a href="https://pypi.org/project/pyzjr/"><img src="https://img.shields.io/pypi/v/pyzjr?color=4f46e5&label=PyPI" alt="PyPI version"></a>
    <a href="https://pypi.org/project/pyzjr/"><img src="https://img.shields.io/pypi/pyversions/pyzjr?color=0891b2" alt="Python versions"></a>
    <a href="https://github.com/Auorui/pyzjr/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Auorui/pyzjr?color=0f766e" alt="License"></a>
    <a href="https://pepy.tech/project/pyzjr"><img src="https://static.pepy.tech/badge/pyzjr?color=7c3aed" alt="Downloads"></a>
  </p>
</div>

`pyzjr` 是一个持续积累的计算机视觉工具包，把日常图像处理、数据集构建、增强、评估和 PyTorch 训练工具整理到一个统一的 Python 包中。它适合个人研究、实验原型和小型视觉项目，也方便在已有工程中按模块复用。

## 特性

- **图像输入输出**：OpenCV、PIL、NumPy 与 PyTorch Tensor 之间的转换，图片/视频读取、写入、显示和拼接。
- **图像增强**：缩放、裁剪、旋转、翻转、颜色扰动、模糊、噪声、二值化、轮廓和骨架提取。
- **数据与文件工具**：常用数据集基类、分类/分割/去雾/变化检测数据集，以及 YAML、JSON、文本和路径操作。
- **训练组件**：损失函数、分类与分割指标、质量评价、优化器、学习率策略、训练统计和 checkpoint 管理。
- **视觉模块**：常用注意力模块、卷积与归一化层、池化、DropPath/DropBlock 和权重初始化。
- **跨平台开发体验**：模块按功能组织，公开 API 可直接从 `pyzjr` 导入。

## 安装

```bash
pip install pyzjr
```

从源码安装：

```bash
git clone https://github.com/Auorui/pyzjr.git
cd pyzjr
pip install -e .
```

项目依赖包括 `opencv-python`、`numpy`、`scikit-image`、`scikit-learn`、`torch`、`torchvision`、`matplotlib`、`PyYAML`、`tqdm` 和 `psutil`。如果只需要查看或使用某一个子模块，也可以根据实际环境单独管理依赖。

## 快速开始

### 读取图像并完成基础处理

```python
import pyzjr

image = pyzjr.read_image("path/to/image.jpg")
gray = pyzjr.to_gray(image)
small = pyzjr.smart_resize(gray, (640, 480))
pyzjr.write_image("outputs/gray.jpg", small)
```

### 组合数据增强

```python
from pyzjr.aug import Compose, Resize, RandomFlip, ToTensor

transform = Compose([
    Resize((256, 256)),
    RandomFlip(),
    ToTensor(),
])

image_tensor, mask_tensor = transform(image, mask)
```

### 创建优化器与学习率策略

```python
from pyzjr.nn import get_optimizer, get_lr_scheduler

optimizer = get_optimizer(model, optimizer_type="adamw", init_lr=1e-3)
scheduler = get_lr_scheduler(
    optimizer,
    scheduler_type="cos",
    init_lr=1e-3,
    total_epochs=50,
)
```

### 使用指标与损失函数

```python
from pyzjr.nn import DiceLoss, SegmentationMetric, calculate_psnr, calculate_seg_confusion_matrix

loss_fn = DiceLoss()
loss = loss_fn(logits, targets)

tp, fn, fp, tn = calculate_seg_confusion_matrix(
    logits, targets, mode="multiclass", num_classes=2
)
metric = SegmentationMetric(tp, fn, fp, tn, reduction="micro")
print(metric.get_index())
print(calculate_psnr(predicted_image, target_image))
```

## 模块概览

| 模块 | 内容 |
| --- | --- |
| `pyzjr.io` | 图像/视频读写、色彩空间、绘图、文本和 Matplotlib 辅助函数 |
| `pyzjr.aug` | 几何变换、颜色增强、模糊、噪声、二值化、轮廓和骨架提取 |
| `pyzjr.data` | DataLoader、数据集、文件操作、路径和标注辅助函数 |
| `pyzjr.nn` | 损失、指标、优化器、学习率策略、训练引擎和神经网络层 |
| `pyzjr.constants` | 图像/视频格式、颜色和常用常量 |
| `pyzjr.utils` | 计时、随机字符串、终端输出和装饰器 |

## 资源

仓库中的示例图片位于 [`resources/images`](resources/images)，可以用于快速测试读写、增强和质量评价功能；项目标识的矢量源文件与 PNG 文件分别是 [`resources/pyzjr-logo.svg`](resources/pyzjr-logo.svg) 和 [`resources/pyzjr-logo.png`](resources/pyzjr-logo.png)。

- PyPI：<https://pypi.org/project/pyzjr/>
- GitHub：<https://github.com/Auorui/pyzjr>
- 问题反馈：<https://github.com/Auorui/pyzjr/issues>

## 开发

安装开发版本后，可以直接修改 `pyzjr/` 下的源码并在本地验证：

```bash
pip install -e .
python -c "import pyzjr; print(pyzjr.__version__)"
```

提交问题或贡献代码时，请附上 Python 版本、操作系统、最小复现代码和完整错误信息。

## 版本

当前版本：**1.4.23**

版本变更会同步更新 `pyzjr/__init__.py` 与 `setup.py`。

## 许可

本项目使用 [MIT License](LICENSE)。

---

如果这个工具包对你的项目有帮助，欢迎在 GitHub 点一个 ⭐，也欢迎提交 Issue 或 Pull Request。
