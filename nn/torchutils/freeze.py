import torch.nn as nn

def freeze_model(module: nn.Module, freeze_parameters=True, freeze_bn=True):
    """
    Change 'requires_grad' value for module and it's child modules and
    optionally freeze batchnorm modules.
    :param module: Module to change
    :param freeze_parameters: True to freeze parameters; False - to enable parameters optimization.
        If None - current state is not changed.
    :param freeze_bn: True to freeze batch norm; False - to enable BatchNorm updates.
        If None - current state is not changed.
    :return: None
    """
    bn_types = nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm

    if freeze_parameters is not None:
        for param in module.parameters():
            param.requires_grad = not freeze_parameters

    if freeze_bn is not None:
        if isinstance(module, bn_types):
            module.track_running_stats = not freeze_bn

        for m in module.modules():
            if isinstance(m, bn_types):
                module.track_running_stats = not freeze_bn

    return module