"""
Copyright (c) 2024, Auorui.
All rights reserved.

Operations related to files
"""
import os
import shutil
from datetime import datetime


def remove_dir(dir_path):
    """Recursively delete a directory and its contents"""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"'{dir_path}' has been deleted.")


def reset_dir(dir_path: str):
    """Delete then recreate a directory."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def clear_dir_content(dir_path):
    """Clear all contents inside a directory but keep the directory itself."""
    if not os.path.exists(dir_path):
        print(f"Error: The directory {dir_path} does not exist.")
        return
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        for dir in dirs:
            dir_paths = os.path.join(root, dir)
            try:
                os.rmdir(dir_paths)
            except Exception as e:
                print(f"Error deleting directory {dir_paths}: {e}")


def multi_makedirs(*paths):
    """Create directories for the given multiple paths, ignoring them if they already exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def unique_makedirs(dir_path, *paths, inc=True):
    """Create a non-repeating directory. If inc is True, append _1, _2... to the end of the directory name if it already exists."""
    outdir = os.path.join(dir_path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        return outdir

    if inc:
        count = 1
        while count < 1000:
            outdir_inc = f"{outdir}_{count}"
            if not os.path.exists(outdir_inc):
                os.makedirs(outdir_inc, exist_ok=True)
                return outdir_inc
            count += 1
        raise RuntimeError(f"Cannot find a unique name for {outdir} after 1000 attempts.")

    return outdir


def datetime_makedirs(dir_path, *paths, use_simple_format=True):
    """Create a path with unique timestamps"""
    fmt = "%Y_%m_%d_%H_%M_%S" if use_simple_format else "%A_%d_%B_%Y_%Hh_%Mm_%Ss"
    time_str = f"{datetime.now().strftime(fmt)}"
    outdir = os.path.join(dir_path, time_str, *paths)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def logdir(dir_path="logs", prefix="", suffix="", use_simple_format=True):
    """
    Create a timestamped logging directory.

    Generates a directory path in the format: {dir_path}/{prefix}_{time_str}_{suffix}.
    Empty prefix or suffix will be automatically ignored.

    Args:
        dir_path (str): Base directory path. Defaults to "logs".
        prefix (str): String to prepend before the timestamp. Defaults to "".
        suffix (str): String to append after the timestamp. Defaults to "".
        use_simple_format (bool): If True, uses '%Y_%m_%d_%H_%M_%S' (e.g., 2023_10_27_15_30_00).
                                  If False, uses a detailed format (e.g., Friday_27_October_2023_15h_30m_00s).
                                  Defaults to True.

    Returns:
        str: The absolute or relative path of the created directory.
    """
    fmt = "%Y_%m_%d_%H_%M_%S" if use_simple_format else "%A_%d_%B_%Y_%Hh_%Mm_%Ss"
    time_str = f"{datetime.now().strftime(fmt)}"
    folder_keys_names = [str(prefix), time_str, str(suffix)]
    folder_names = [folder for folder in folder_keys_names if folder]

    log_dir = os.path.join(dir_path, *folder_names)
    log_dir = os.path.normpath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def loss_weights_dirs(dir_path='./logs', prefix="", suffix="", use_simple_format=True):
    """
    Create directories for saving training logs and model weights.
    This function automatically creates a timestamped root folder, and inside it,
    two subdirectories: 'loss' and 'weights'.

    Args:
        dir_path (str): Base directory path. Defaults to './logs'.
        prefix (str): String to prepend to the timestamp folder name. Defaults to "".
        suffix (str): String to append to the timestamp folder name. Defaults to "".
        use_simple_format (bool): Format of the timestamp. Defaults to True.

    Returns:
        tuple: A tuple of (loss_log_dir, save_model_dir, time_dir), representing
               the paths for loss logs, saved weights, and the timestamped root folder respectively.
    """
    time_dir = logdir(dir_path, prefix=prefix, suffix=suffix, use_simple_format=use_simple_format)
    loss_log_dir = os.path.normpath(os.path.join(time_dir, 'loss'))
    save_model_dir = os.path.normpath(os.path.join(time_dir, 'weights'))
    multi_makedirs(loss_log_dir, save_model_dir)

    return loss_log_dir, save_model_dir, time_dir


def copy_files(file_list, target_dir):
    """
    Copy multiple files to the specified save folder

    Args:
        file_list (list): List of file paths to be copied (e.g., [path1, path2, ...])
        target_dir (str): Target save folder path
    """
    try:
        os.makedirs(target_dir, exist_ok=True)
        for file_path in file_list:
            base_file_name = os.path.basename(file_path)
            destination_path = os.path.join(target_dir, base_file_name)
            shutil.copy2(file_path, destination_path)
            print(f"Successfully copied: {file_path} -> {target_dir}")
    except Exception as e:
        print(f"Error copying files: {e}")


def move_files(file_list, target_dir):
    """
    Move multiple files to a specified directory (preserving metadata and automatically creating the
    target directory if it does not exist

    Args:
        file_list (list): List of file paths to be moved (e.g., [path1, path2, ...])
        target_dir (str): Target save folder path
    """
    try:
        os.makedirs(target_dir, exist_ok=True)
        for file_path in file_list:
            base_name = os.path.basename(file_path)
            dest_path = os.path.join(target_dir, base_name)
            shutil.move(file_path, dest_path)
            print(f"Successfully moved: {file_path} -> {dest_path}")
    except Exception as e:
        print(f"Error moving files: {str(e)}")



if __name__=="__main__":
    target_path = r'D:\software\Nutbit\NutBit'
    clear_dir_content(target_path)