import torch
import numpy as np

class Normalize():
    def __init__(self, image):
        if isinstance(image, np.ndarray):
            self.image = image
        elif isinstance(image, torch.Tensor):
            self.image = image.numpy()
        else:
            raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")

    @property
    def min_max(self):
        """
        Min-Max normalization, scales pixel values to a specific range [0, 1].
        """
        min_val = np.min(self.image)
        max_val = np.max(self.image)
        min_and_max = (self.image - min_val) / (max_val - min_val)
        return min_and_max

    @property
    def z_score(self):
        """
        Z-score, computes the standard scores of given data.
        """
        mean = np.mean(self.image)
        std = np.std(self.image)
        z_scores = (self.image - mean) / std
        return z_scores

    @property
    def linear(self):
        """
        Linear normalization
        http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
        """
        arr = self.image.astype('float')
        for i in range(3):
            minval = arr[..., i].min()
            maxval = arr[..., i].max()
            if minval != maxval:
                arr[..., i] -= minval
                arr[..., i] *= (255.0 / (maxval - minval))
        return arr

    @property
    def zero_centered(self):
        """Zero-centered (centered)"""
        mean = np.mean(self.image)
        self.image -= mean
        return self.image

def denormalize(data, mean, std):
    """
    Denormalization operation, converts normalized data back to the original range.

    Parameters:
        - data: Input data (NumPy array or Torch tensor)
        - mean: Mean used for normalization
        - std: Standard deviation used for normalization

    Returns:
        - Denormalized data
    """
    if isinstance(data, np.ndarray):
        # NumPy arrays
        denormalized_data = data * std + mean
    elif torch.is_tensor(data):
        # Torch tensors
        denormalized_data = data.clone()
        for d, m, s in zip(denormalized_data, mean, std):
            d.mul_(s).add_(m)
    else:
        raise ValueError("Unsupported input type. Only NumPy arrays and Torch tensors are supported.")

    return denormalized_data

if __name__ == "__main__":
    import torchvision.transforms as transforms
    from PIL import Image

    image_path = r"img.png"
    image = Image.open(image_path)

    to_tensor = transforms.ToTensor()
    torch_image = to_tensor(image)

    normalizer = Normalize(torch_image)

    min_max_normalized = normalizer.min_max
    z_score_normalized = normalizer.z_score
    linear_normalized = normalizer.linear
    zero_centered = normalizer.zero_centered
    print(min_max_normalized,z_score_normalized,linear_normalized,zero_centered)
