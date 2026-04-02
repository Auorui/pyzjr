import torch

def generate_class_weights(labels, num_classes, strategy='inverse_frequency', device=None):
    """
    Generate class weights based on a given strategy for each class.

    Parameters:
        labels (Tensor): A tensor of labels with shape (b, c, h, w), where b is the batch size,
                            c is the number of classes, and h and w are the height and width of the image.
        num_classes (int): The number of classes (c).
        strategy (str): The strategy for computing weights, options are ['uniform', 'inverse_frequency', 'normalize_frequency', 'manual'].
        device (torch.device, optional): The device on which to perform computations. Default is None, meaning it uses the device of the input tensor.

    Returns:
        class_weights (Tensor): A tensor of class weights with shape (c,). Each element represents the weight for a particular class.
    """

    if device is not None:
        labels = labels.to(device)

    b, c, h, w = labels.shape
    class_weights = torch.zeros((b, c), device=device)  # 初始化权重张量

    if strategy == 'uniform':
        # 均匀权重：每个类别的权重相同
        class_weights.fill_(1.0)

    elif strategy == 'inverse_frequency':
        # 逆频率加权：频率低的类别得到更大的权重
        for i in range(b):
            class_counts = torch.bincount(labels[i].view(-1), minlength=num_classes)
            total_pixels = labels[i].numel()
            class_weights[i] = total_pixels / (num_classes * class_counts.float())
            class_weights[i][class_counts == 0] = 0  # 对于未出现的类别，设定权重为 0

    elif strategy == 'normalize_frequency':
        # 归一化频率：类别出现频率的归一化权重
        for i in range(b):
            class_freq = torch.bincount(labels[i].view(-1), minlength=num_classes)
            class_weights[i] = class_freq.float() / class_freq.sum().float()

    elif strategy == 'manual':
        # 手动设置权重
        class_weights.fill_(1.0)

    else:
        raise ValueError("Unknown strategy. Choose from ['uniform', 'inverse_frequency', 'normalize_frequency', 'manual']")

    if b == 1:
        return class_weights.squeeze(0)  # 如果只有一个样本，返回单一类别的权重
    else:
        return class_weights.mean(dim=0)  # 返回所有样本的类别权重平均值

if __name__=="__main__":
    labels = torch.randint(0, 3, (2, 3, 4, 4))
    num_classes = 3
    class_weights = generate_class_weights(labels, num_classes, strategy='inverse_frequency')
    print(class_weights)
