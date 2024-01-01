




def autopad(kernel, padding=None, dilation=1):
    """自动计算填充大小，以使输出具有与输入相同的形状
    :param k: kernel
    :param p: padding
    :param d: dilation
    :return: 自动计算得到的填充大小
    """
    k, p, d = kernel, padding, dilation
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p