import os
from pyzjr.data.base import timestr
from shutil import rmtree


def logdir(dir="logs", simple=True, prefix="", suffix=""):
    """
    Logging generator
    :param dir: Default "logs"
    :param simple: '%Y_%m_%d_%H_%M_%S' or '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    :param prefix: Add folder name before time string
    :param suffix: After adding the folder name to the time string
    :return:
    """
    time_str = timestr(simple)
    folder_names = [prefix, time_str, suffix]
    folder_names = [folder for folder in folder_names if folder]
    log_dir = os.path.join(dir, *folder_names)
    os.makedirs(log_dir, exist_ok=True)

    return log_dir

def multi_makedirs(*args):
    """
    为给定的多个路径创建目录, 如果路径不存在, 则创建它
    """
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)

def unique_makedirs(root_path, *paths, inc=False):
    """
    如果构建的输出目录路径不存在，它将创建该目录。
    如果 inc 为 True，它将检查是否已经存在相同的目录。如果存在，将在目录名称末尾附加一个数字后缀，直到找到一个不存在的目录。
    返回最终的输出目录路径。
    """
    outdir = os.path.join(root_path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '_' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '_' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir

def datatime_makedirs(root_path, *paths, use_simple_format=True):
    """
    创建含有独特时间的路径
    """
    time_path = timestr(use_simple_format)
    outdir = os.path.join(root_path, time_path, *paths)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def loss_weights_dirs(root_path='./logs', use_simple_format=True):
    """训练代码常用路径, 保存loss信息, 以及保存最佳模型"""
    time_str = timestr(use_simple_format)
    time_dir = os.path.join(root_path, time_str)
    loss_log_dir = os.path.join(time_dir, 'loss')
    save_model_dir = os.path.join(time_dir, 'weights')
    multi_makedirs(loss_log_dir, save_model_dir)
    return loss_log_dir, save_model_dir, time_dir

def rm_makedirs(file_path: str):
    # 如果文件夹存在，则先删除原文件夹在重新创建
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

if __name__=="__main__":
    root_path = 'test_output'
    subdir_name = 'test_subdir'

    # outdir1 = unique_makedirs(root_path, subdir_name)
    # print(f"Created directory: {outdir1}")
    #
    # outdir2 = unique_makedirs(root_path, subdir_name, inc=True)
    # print(f"Created directory with inc: {outdir2}")
    #
    # outdir3 = unique_makedirs(root_path, subdir_name, inc=True)
    # print(f"Created directory with inc (second time): {outdir3}")
    # cleanup_test_dir(root_path)

    datatime_makedirs(root_path, "loss")
    datatime_makedirs(root_path, "weight")