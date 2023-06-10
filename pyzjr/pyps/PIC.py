from tqdm import tqdm
import requests
import cv2
import os, shutil
from matplotlib import pyplot as plt

def download_file(url):
    """
    :param url:下载文件所在url链接
    :return: 下载的位置处于根目录
    """
    print("------", "Start download with urllib")
    name = url.split("/")[-1]
    resp = requests.get(url, stream=True)
    content_size = int(resp.headers['Content-Length']) / 1024  # 确定整个安装包的大小
    # 下载到上一级目录
    path = os.path.abspath(os.path.dirname(os.getcwd())) + "\\" + name
    # 下载到该目录
    path = os.getcwd() + "\\" + name
    print("File path:  ", path)
    with open(path, "wb") as file:
        print("File total size is:  ", content_size)
        for data in tqdm(iterable=resp.iter_content(1024), total=content_size, unit='k', desc=name):
            file.write(data)
    print("------", "finish download with urllib\n\n")

def getPhotopath(paths):
    """
    * 批量读取文件夹下的图片
    :param paths:输入图片所在位置
    :return: 该目录下所有的图片
    """
    imgfile = []
    file_list = os.listdir(paths)
    for i in file_list:
        newph = os.path.join(paths, i)
        imgfile.append(newph)
    return imgfile


def Pic_rename(img_folder,object='Crack',format='.jpg',num=0):
    """
    * 用于批量修改图像的命名
    :param img_folder:存放图片的路径
    :param object: 图像的对象
    :param format: 图片格式，可自行命名，这里给出jpg
    :param num: 对图片进行计数
    :return: 用dst替换src
    """
    for img_name in os.listdir(img_folder):
        src = os.path.join(img_folder, img_name)
        dst = os.path.join(img_folder, object+ str(num) + format)
        num= num+1
        os.rename(src, dst)

def read_resize_image(path, size=1.0, show=False, space=False):
    """
    :param path:图片的路径
    :param size: 希望得到图片的大小,cat.0为原始图像
    :param show: 如果为真，返回原始图像和修改后图像的长、宽、通道
    :param space: 如果为真，返回图像的灰度图
    :return: 返回修改大小后的图像
    """
    if space==False:
        originalimage = cv2.imread(path)
    else:
        originalimage = cv2.imread(path,0)
    if size != 1.0:
        height, width = originalimage.shape[:2]
        size = (int(width * size), int(height * size))
        originalimage = cv2.resize(originalimage, size, interpolation=cv2.INTER_AREA)
    if not show:
        pass
    else:
        print(originalimage)
        print(originalimage.shape)
    return originalimage


def load_images_from_folder(folder_path):
    """
    加载一个文件夹下的图片，并存入列表中并返回
    :param folder_path: 目标图片路径
    :return: 含有该文件夹下的所有图片
    """
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def save_images(images, folder_path):
    """
    批量保存一图片
    :param images: load_images_from_folder的返回值
    :param folder_path: 保存路径
    :return: 无返回
    """
    os.makedirs(folder_path, exist_ok=True)
    for i, img in enumerate(images):
        img_path = os.path.join(folder_path, f"image_{i}.jpg")
        cv2.imwrite(img_path, img)
