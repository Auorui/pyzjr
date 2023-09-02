import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def StackedCV2(scale, imgList, cols):
    """
    将多张图像堆叠在一起并在单个窗口中显示
    :param scale: 图像的缩放比例,大于1表示放大,小于1表示缩小
    :param imgList: 要堆叠的图像列表
    :param cols: 每行显示的图像数量
    :return: 堆叠后的图像
    """
    totalImages = len(imgList)
    rows = totalImages // cols if totalImages // cols * cols == totalImages else totalImages // cols + 1
    blankImages = cols * rows - totalImages

    width = imgList[0].shape[1]
    height = imgList[0].shape[0]
    imgBlank = np.zeros((height, width, 3), np.uint8)
    imgList.extend([imgBlank] * blankImages)
    for i in range(cols * rows):
        imgList[i] = cv2.resize(imgList[i], (0, 0), None, scale, scale)
        if len(imgList[i].shape) == 2:
            imgList[i] = cv2.cvtColor(imgList[i], cv2.COLOR_GRAY2BGR)
    hor = [imgBlank] * rows
    for y in range(rows):
        line = []
        for x in range(cols):
            line.append(imgList[y * cols + x])
        hor[y] = np.hstack(line)
    ver = np.vstack(hor)
    return ver


def StackedImages(scale,imgArray):
    """
    :param scale:图片的规模,可使用小数,1为原始图像
    :param imgArray: 图像组成的列表，来表示行列排列
    :return: 生成的图像可按输入列表的行列排序展示,拼接展示图像
    """
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: 
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def Stackedplt(images, shape, spacing=0.1):
    """
    按照指定形状堆叠并显示多张图片——plt
    :param images: 包含多张图片数据的列表或数组。每张图片的形状应相同
    :param shape: 展示的图片的行列数,如(2, 3)
    :param spacing: 图片之间的间隔,默认为0.1
    :return:
    """
    rows, cols = shape
    image_height, image_width, _ = images[0].shape
    stacked_height = int(rows * image_height + (rows - 1) * image_height * spacing)
    stacked_width = int(cols * image_width + (cols - 1) * image_width * spacing)
    stacked_image = np.ones((stacked_height, stacked_width, 3), dtype=np.uint8) * 255
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols

        x = int(col * (image_width + spacing * image_width))
        y = int(row * (image_height + spacing * image_height))

        stacked_image[y:y+image_height, x:x+image_width, :] = image

    plt.imshow(stacked_image)
    plt.axis('off')  # 关闭坐标轴显示
    plt.show()


def Stackedtorch(imgs, num_rows, num_cols, titles=None, scale=None, camp=None):
    """
    堆叠显示Tensor图像或PIL图像
    :param imgs: 包含图像的列表,可以包含Tensor图像或PIL图像
    :param num_rows: 图像的行数
    :param num_cols: 图像的列数
    :param titles: 图像的标题列表,默认为None
    :param scale: 图像的缩放比例,默认为None
    :param camp: 'gray'
    :return:
    example:
        imgs = [image1, image2, image3, image4]
        titles = ['Image 1', 'Image 2', 'Image 3','Image 4']
        axes = Stackedtorch(imgs, 2, 2, titles=titles, scale=5)
        plt.savefig('my_plot.jpg')
        plt.show()
    """
    if scale:
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    else:
        _, axes = plt.subplots(num_rows, num_cols)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy(),cmap=camp)
        else:
            # PIL Image
            ax.imshow(img,cmap=camp)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def plot_line(x, y, xlabel, ylabel, title):
    """
    绘制折线图
    :param x: x轴的数据列表
    :param y: y轴的数据列表
    :param xlabel: x轴的标签
    :param ylabel: y轴的标签
    :param title: 图形的标题
    :return:
    """
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def bar_chart(x, y, xlabel, ylabel, title):
    """绘制柱状图"""
    plt.bar(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def scatter_plot(x, y, xlabel, ylabel, title):
    """绘制散点图"""
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


if __name__=="__main__":
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    xlabel = 'X'
    ylabel = 'Y'
    title = '折线图示例'
    plot_line(x, y, xlabel, ylabel, title)
    bar_chart(x, y, xlabel, ylabel, title)
    scatter_plot(x, y, xlabel, ylabel, title)