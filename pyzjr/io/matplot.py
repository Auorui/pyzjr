"""
Copyright (c) 2025, Auorui.
All rights reserved.
"""
import math
import platform
from PIL import Image
from matplotlib import pyplot as plt


def matplotlib_patch():
    # Detect the operating system type
    import matplotlib
    matplotlib.use("TkAgg")
    system_name = platform.system()
    if system_name == "Windows":
        plt.rcParams['font.family'] = ['Microsoft YaHei']
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'sans-serif']
    elif system_name == "Linux":
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif']
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback',
                                           'sans-serif']
    elif system_name == "Darwin":
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'sans-serif']
    else:
        print("Unknown OS, using fallback font settings.")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


def plot_images(images, titles=None, stacked_shape=None, figsize_scale=4, cmap=None, save_path=None):
    """
    Universally display single or multiple images with automatic grid layout.

    Args:
        images: A single image, a 1D list of images, or a 2D list of images.
        titles: A list of titles for each image (optional).
        stacked_shape (tuple): (rows, cols). If None, it's automatically calculated.
        figsize_scale (int): Size of each subplot (width and height).
        cmap (str): Colormap for matplotlib (e.g., 'gray').
        save_path (str): If provided, save the figure to this path.
    """
    matplotlib_patch()
    if not isinstance(images, (list, tuple)):
        plt.figure(figsize=(figsize_scale, figsize_scale))
        plt.imshow(images, cmap=cmap)
        plt.axis('off')
        if titles:
            plt.title(titles)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        return

    if isinstance(images[0], (list, tuple)):
        flat_images = [img for row in images for img in row]
        rows = len(images)
        cols = len(images[0])
    else:
        flat_images = images
        total = len(flat_images)
        if stacked_shape:
            rows, cols = stacked_shape
        else:
            cols = math.ceil(math.sqrt(total))
            rows = math.ceil(total / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_scale, rows * figsize_scale))

    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(rows * cols):
        ax = axes[i]
        if i < len(flat_images):
            img = flat_images[i]
            if isinstance(img, Image.Image) and img.mode == 'L':
                cmap = 'gray'
            ax.imshow(img, cmap=cmap)
            if titles and i < len(titles):
                ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    image_path = r"/resources/images/aircraft.jpg"

    img1 = Image.open(r"/resources/images/aircraft.jpg")
    img2 = Image.open(r"/resources/images/aircraft_mask.png")
    img3 = Image.open(r"/resources/images/bird.jpg")
    img4 = Image.open(r"/resources/images/cat.png")
    img5 = Image.open(r"/resources/images/dog.png")
    img6 = Image.open(r"/resources/images/lena.png")

    # plot_images(pil_image, titles='aircraft')

    # plot_images([img1, img2, img3, img4], titles=['飞机', '飞机掩膜', '鸟', '猫'], figsize_scale=3)

    plot_images([[img1, img2, img3], [img4, img5, img6]], stacked_shape=(2, 3))



