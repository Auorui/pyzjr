import torch
import numpy as np

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

def imagenet_denormal(image):
    # 对RGB图像x进行ImageNet反规范化, i.e. = x * std + mean
    return image * IMAGENET_STD + IMAGENET_MEAN

def imagenet_normal(image):
    # 对RGB图像x进行ImageNet规范化, i.e. = (x - mean) / std
    image = image.astype('float') / 255
    normal_image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return normal_image

def min_max_normal(np_image):
    """
    Min-Max normalization, scales pixel values to a specific range [0, 1].
    """
    min_val = np.min(np_image)
    max_val = np.max(np_image)
    min_and_max = (np_image - min_val) / (max_val - min_val)
    return min_and_max

def z_score_normal(np_image):
    """
    Z-score, computes the standard scores of given data.
    """
    mean = np.mean(np_image)
    std = np.std(np_image)
    z_scores = (np_image - mean) / std
    return z_scores

def linear_normal(np_image):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = np_image.astype('float')
    for i in range(3):
        minval = arr[..., i].min()
        maxval = arr[..., i].max()
        if minval != maxval:
            arr[..., i] -= minval
            arr[..., i] *= (255.0 / (maxval - minval))
    return arr

def zero_centered_normal(np_image):
    """Zero-centered (centered)"""
    np_image = np_image.astype('float32')
    mean = np.mean(np_image)
    np_image -= mean
    return np_image

class Normalizer():
    def __init__(self, normal_type='imagenet'):
        self.normal_type = normal_type

    def __call__(self, image):
        if isinstance(image, np.ndarray):
            self.image = image
        else:
            self.image = np.array(image)

        if self.normal_type == 'imagenet':
            return imagenet_normal(self.image)
        elif self.normal_type == 'min_max':
            return min_max_normal(self.image)
        elif self.normal_type == 'z_score':
            return z_score_normal(self.image)
        elif self.normal_type == 'linear':
            return linear_normal(self.image)
        elif self.normal_type == 'zero_centered':
            return zero_centered_normal(self.image)
        else:
            raise ValueError(f"Unsupported normalization type: {self.normal_type}, "
                             f"You can choose to use these parameters"
                             f"' imagenet ',' min_max ',' z_score ',' linear ', and ' zero_centered '")


def denormalize(image, mean, std):
    """
    Denormalization operation, converts normalized data back to the original range.

    Parameters:
        - image: Input data (NumPy array or Torch tensor)
        - mean: Mean used for normalization
        - std: Standard deviation used for normalization

    Returns:
        - Denormalized data
    """
    if isinstance(image, np.ndarray):
        # NumPy arrays
        denormalized_data = image * std + mean
    elif torch.is_tensor(image):
        # Torch tensors
        denormalized_data = image.clone()
        for d, m, s in zip(denormalized_data, mean, std):
            d.mul_(s).add_(m)
    else:
        raise ValueError("Unsupported input type. Only NumPy arrays and Torch tensors are supported.")

    return denormalized_data


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from PIL import Image

    image_path = r"../../data/scripts/fire.jpg"
    image = Image.open(image_path)

    to_tensor = transforms.ToTensor()
    torch_image = to_tensor(image)

    imagenet_normalized = Normalizer('imagenet')
    min_max_normalized = Normalizer('min_max')
    z_score_normalized = Normalizer('z_score')
    linear_normalized = Normalizer('linear')
    zero_centered = Normalizer('zero_centered')


    print(imagenet_normalized(image))
    print(min_max_normalized(image))
    print(z_score_normalized(image))
    print(linear_normalized(image))
    print(zero_centered(image))