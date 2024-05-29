"""
Training strategies for deep learning
- author: Auorui (夏天是冰红茶)
- creation time:2023.9
- pyzjr.dlearn is a part of the model strategy used for deep learning
- WeChat: z15583909992
- Email: zjricetea@gmail.com
- Note: Currently still being updated, please refer to the latest version for any changes that may occur
"""
from .strategy import *
from .callbacks import *
from .callbacklog import (
    LossHistory,
    LossMonitor,
    ErrorRateMonitor
)
from .tools import (
    Runcodes,
    LoadingBar,
    show_config,
    time_sync,
    profile
)
from .save_pth import (
    save_model_to_pth_best,
    save_model_to_pth_best_metrics,
    save_model_to_pth_simplify
)
from .Trainer import *


numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)