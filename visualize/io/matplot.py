"""
Copyright (c) 2025, Auorui.
All rights reserved.
"""
import platform
from PIL import Image
from matplotlib import pyplot as plt


def matplotlib_patch():
    # 检测操作系统类型
    import matplotlib
    matplotlib.use("TkAgg")
    system_name = platform.system()
    if system_name == "Windows":
        # Windows 系统使用 Times New Roman 和 SimHei
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.sans-serif'] = ['SimHei']
    elif system_name == "Linux":
        # Linux 系统使用 DejaVu Serif 和 SimHei（一般 Linux 上没有 Times New Roman）
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif']  # DejaVu Serif 是 Linux 上常见的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
    elif system_name == "Darwin":  # macOS 系统
        # macOS 上可以使用默认的字体和 SimHei
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.sans-serif'] = ['SimHei']
    else:
        # 如果无法识别的操作系统，默认使用 DejaVu Serif
        print("Unknown OS, using fallback font settings.")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


def imshowplt(image, stacked_shape=None, title=None, cmap=None, save_path=None):
    """
    Display image and add title
    Examples:
        color_image = Image.open("dog.png")
        imshowplt([color_image,color_image], stacked_shape=(2,1), title='PIL Color Image', cmap=None)
    """
    matplotlib_patch()
    if isinstance(image, list):
        image = StackedpltV1(image, stacked_shape)
    else:
        if isinstance(image, Image.Image):
            if image.mode == 'L' or (image.mode == 'LA' and image.getbands()[1] == 'L'):
                cmap = 'gray'
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def StackedpltV1(images, stacked_shape):
    """
    Stack multiple images according to the specified shape
    :param images: A list containing multiple PIL Image objects
    :param stacked_shape: The number of rows and columns for displaying the images, e.g., (2, 3)
    :return: A stacked PIL Image object
    """
    rows, cols = stacked_shape
    new_images = []
    for im in images:
        if isinstance(im, Image.Image):
            new_images.append(im)
    assert all(img.size == new_images[0].size for img in new_images), "All images must be the same size"
    width, height = new_images[0].size
    new_img_width = width * cols
    new_img_height = height * rows
    new_img = Image.new('RGB', (new_img_width, new_img_height), color=(255, 255, 255))
    x_offset = 0
    y_offset = 0
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(new_images):
                img = new_images[index]
                new_img.paste(img, (x_offset, y_offset))
                x_offset += width
        x_offset = 0
        y_offset += height

    return new_img

def StackedpltV2(imgs, stacked_shape, titles=None, scale=None, camp=None):
    """
    Display images in a grid format.
    :param imgs: A list containing the images.
    :param stacked_shape: The number of rows and columns for displaying the images, e.g., (2, 3)
    :param titles: A list of titles for the images. Defaults to None.
    :param scale: The scaling factor for the images. Defaults to None, which means no scaling is applied.
    :param camp: 'gray'
    :return:
    Examples:
    ```
        imgs = [image1, image2, image3, image4]
        titles = ['Image 1', 'Image 2', 'Image 3','Image 4']
        axes = Stackedtorch(imgs, 2, 2, titles=titles, scale=5)
        plt.savefig('my_plot.jpg')
        plt.show()
    ```
    """
    matplotlib_patch()
    num_rows, num_cols = stacked_shape
    if scale:
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    else:
        _, axes = plt.subplots(num_rows, num_cols)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # PIL Image
        ax.imshow(img, cmap=camp)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])

    return axes

if __name__ == "__main__":
    from pyzjr.visualize.io.readio import imattributes
    image_path = r"E:\PythonProject\pyzjrPyPi\pyzjr\test.png"
    image = Image.open(image_path)
    image.show()
    imshowplt([image, image], (2, 1))

    stackimgs = [image, image, image, image]
    titles = ['Image 1', 'Image 2', 'Image 3','Image 4']
    axes = StackedpltV2(stackimgs, (2, 2), titles=titles)
    plt.show()
    print(imattributes(image))
