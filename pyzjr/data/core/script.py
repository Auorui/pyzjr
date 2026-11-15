import os
import math
import random
import shutil

from typing import Tuple, List
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor

from pyzjr.data.core.path import get_image_path


def split_dataset(
        dir_path: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        return_abs_path: bool = True,
        shuffle: bool = True,
        seed: int = 0
) -> Tuple[List[str], List[str], List[str]]:
    assert train_ratio + val_ratio <= 1.0, "train_ratio + val_ratio cannot exceed 1.0"
    test_ratio = 1.0 - (train_ratio + val_ratio)
    images_paths = get_image_path(dir_path, return_abs_path=return_abs_path)

    if shuffle:
        random.seed(int(seed))
        random.shuffle(images_paths)

    num_images = len(images_paths)
    num_train = round(num_images * train_ratio)
    num_val = num_images - num_train if test_ratio == 0 else math.ceil(num_images * val_ratio)
    num_test = 0 if test_ratio == 0 else num_images - (num_train + num_val)

    train_list = images_paths[:num_train]
    val_list = images_paths[num_train:num_train + num_val]
    test_list = images_paths[num_train + num_val:]

    print(f"✅ Successfully split {num_images} images into: "
          f"{len(train_list)} train, {len(val_list)} val, {len(test_list)} test.")

    return train_list, val_list, test_list


def batch_rename_images(
        dir_path: str,
        save_dir: str,
        start_index: int = None,
        prefix: str = '',
        suffix: str = '',
        format: str = None,
        target_shape: tuple = None,
        num_type: int = 3,
        num_workers: int = 4,
):
    """
    Batch rename, resize, and convert image formats with multi-threading support.

    This function processes all images in a source directory and saves them to a
    destination directory. It applies EXIF orientation correction, handles alpha
    channels (flattening transparent backgrounds for formats like JPEG), resizes
    images (if specified), and can rename files with sequential numbering.

    Args:
        dir_path (str): Path to the source directory containing the images.
        save_dir (str): Path to the destination directory where processed images will be saved.
        start_index (int, optional): Starting number for sequential renaming (e.g., 1 for 001.png).
                                     If None, the original filenames are preserved. Defaults to None.
        prefix (str): String to prepend to the new filename (e.g., "train_"). Defaults to ''.
        suffix (str): String to append to the new filename (e.g., "_aug"). Defaults to ''.
        format (str, optional): Target image format (e.g., 'jpg', 'png', 'webp').
                                If None, the original format is retained. Defaults to None.
        target_shape (tuple, optional): Target dimensions as (width, height) for resizing.
                                        If None, the original dimensions are retained. Defaults to None.
        num_type (int): Zero-padding length for the sequential number (e.g., 3 -> "001").
                        Defaults to 3.
        num_workers (int): Number of threads to use for parallel processing. Defaults to 4.

    Returns:
        None. The processed images are saved directly to `save_dir`, and progress is printed to the console.
    """
    os.makedirs(save_dir, exist_ok=True)
    images_paths = get_image_path(dir_path, return_abs_path=True)
    total_nums = len(images_paths)

    if total_nums == 0:
        print("⚠️ No images found.")
        return

    # Transparent background flattening
    def _flatten(image):
        has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
        if not has_alpha:
            return image
        rgba = image.convert("RGBA")
        base = Image.new("RGB", rgba.size, (255, 255, 255))
        base.paste(rgba, mask=rgba.getchannel("A"))
        return base

    # Processing logic for a single image
    def _process(args):
        img_path, new_path, t_format, t_shape, s_format = args

        # Directly copy in the same format
        if t_format is None or t_format.upper() == s_format.upper():
            if os.path.abspath(img_path) == os.path.abspath(new_path):
                return False, "The input and output paths are the same"
            shutil.copy2(img_path, new_path)
            return True, None

        try:
            with Image.open(img_path) as source:
                source.seek(0)
                img = ImageOps.exif_transpose(source)
                kwargs = {}

                if t_shape:
                    img = img.resize(t_shape, Image.Resampling.LANCZOS)

                if t_format == "JPEG":
                    img = _flatten(img).convert("RGB")
                    kwargs.update(quality=95, optimize=True)

                elif t_format == "WEBP":
                    if img.mode not in ("RGB", "RGBA"):
                        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")
                    kwargs.update(quality=95, method=6)

                elif t_format == "PNG":
                    if img.mode == "CMYK":
                        img = img.convert("RGB")
                    kwargs.update(optimize=True)

                elif t_format == "BMP":
                    img = _flatten(img).convert("RGB")

                elif t_format == "TIFF":
                    if img.mode == "P":
                        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
                    kwargs.update(compression="tiff_deflate")

                img.save(new_path, format=t_format, **kwargs)
            return True, None
        except Exception as e:
            return False, str(e)

    t_format = None
    if format:
        fmt_lower = format.lower()
        t_format = {'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG',
                    'webp': 'WEBP', 'bmp': 'BMP', 'tif': 'TIFF', 'tiff': 'TIFF'}.get(fmt_lower)

    tasks = []
    for i, img_path in enumerate(images_paths):
        img_name = os.path.basename(img_path)
        name, ext = os.path.splitext(img_name)
        s_format = ext.lstrip('.').upper()
        re_ext = f".{format.lower()}" if format else ext.lower()

        if start_index is not None:
            padded_i = str(start_index).zfill(num_type)
            start_index += 1
            new_name = f"{prefix}{padded_i}{suffix}{re_ext}"
        else:
            new_name = f"{prefix}{name}{suffix}{re_ext}"

        tasks.append((img_path, os.path.join(save_dir, new_name), t_format, target_shape, s_format))

    success_count = 0
    print(f"🚀 Starting processing {total_nums} images...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(_process, tasks)

        for i, (success, error_msg) in enumerate(results):
            src_path = tasks[i][0]
            dst_path = tasks[i][1]

            if success:
                success_count += 1
                print(f"✅ [{i+1}/{total_nums}] {src_path} -> {dst_path}")
            else:
                print(f"❌ [{i+1}/{total_nums}] Error: {src_path} -> {error_msg}")

    print(f"\n🎉 Processing completed! Success: {success_count}/{total_nums}")
    print(f"📁 Saved to: {save_dir}")


if __name__=="__main__":
    data_dir = r'E:\PythonProject\pyzjrPyPi\resources\images'
    save_dir = r'E:\PythonProject\pyzjrPyPi\resources\images2'

    train_list, val_list, test_list = \
        split_dataset(data_dir, train_ratio=.8, val_ratio=.1, return_abs_path=False)

    print(train_list, val_list, test_list)

    batch_rename_images(data_dir, save_dir,
                        start_index=1,
                        format='jpeg',
                        prefix='re_',
                        suffix='new',
                        num_type=4)