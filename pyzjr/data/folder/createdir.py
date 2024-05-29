import os
import shutil

def cleanup_test_dir(test_dir):
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

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

if __name__=="__main__":
    root_path = 'test_output'
    subdir_name = 'test_subdir'

    outdir1 = unique_makedirs(root_path, subdir_name)
    print(f"Created directory: {outdir1}")

    outdir2 = unique_makedirs(root_path, subdir_name, inc=True)
    print(f"Created directory with inc: {outdir2}")

    outdir3 = unique_makedirs(root_path, subdir_name, inc=True)
    print(f"Created directory with inc (second time): {outdir3}")
    cleanup_test_dir(root_path)