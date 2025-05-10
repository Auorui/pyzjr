"""
Copyright (c) 2024, Auorui.
All rights reserved.

Weighted average, reference from <https://pytorch.org/docs/stable/optim.html#putting-it-all-together-ema>
指数移动平均（EMA）   常见
随机加权平均（SWA）   https://arxiv.org/abs/1803.05407
Tanh自适应指数移动EMA算法（T_ADEMA）    Originating from https://kns.cnki.net/kcms2/article/abstract?v=vdPasdvfHvuuN-gB4G4neXcKqbiX3vnPwH2QfVTNb4OQCyQJO2HTgHpa6C3EDWqlnkrcNyjnTtxYjmozsntFZzru-e7vk_X4Fq9NCmtavFoJFkztVbWX1vr5qj9w2djSGdHx0-RWLb0=&uniplatform=NZKPT&flag=copy

https://blog.csdn.net/m0_62919535/article/details/135482009?spm=1001.2014.3001.5501

time 2024-01-09
"""
import torch
import itertools
from copy import deepcopy
from torch.nn import Module

__all__ = ["AveragingBaseModel", "EMAModel", "SWAModel", "T_ADEMAModel",
           "get_ema_avg_fn", "get_swa_avg_fn", "get_t_adema_fn"]

def get_ema_avg_fn(decay=0.999):
    @torch.no_grad()
    def ema_update(ema_param, current_param, num_averaged):
        return decay * ema_param + (1 - decay) * current_param
    return ema_update

def get_swa_avg_fn():
    @torch.no_grad()
    def swa_update(averaged_param, current_param, num_averaged):
        return averaged_param + (current_param - averaged_param) / (num_averaged + 1)
    return swa_update

def get_t_adema_fn(alpha=0.9):
    num_averaged = [0]
    @torch.no_grad()
    def t_adema_update(averaged_param, current_param, num_averageds):
        num_averaged[0] += 1
        decay = alpha * torch.tanh(torch.tensor(num_averaged[0], dtype=torch.float32))
        tadea_update = decay * averaged_param + (1 - decay) * current_param
        return tadea_update
    return t_adema_update


class AveragingBaseModel(Module):
    def __init__(self, model, cuda=True, avg_fn=None, use_buffers=False):
        super(AveragingBaseModel, self).__init__()
        device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.module = deepcopy(de_parallel(model))
        self.module = self.module.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        self.avg_fn = avg_fn
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update(self, model):
        self_param = itertools.chain(self.module.parameters(), self.module.buffers() if self.use_buffers else [])
        model_param = itertools.chain(model.parameters(), model.buffers() if self.use_buffers else [])

        self_param_detached = [p.detach() for p in self_param]
        model_param_detached = [p.detach().to(p_averaged.device) for p, p_averaged in zip(model_param, self_param_detached)]

        if self.n_averaged == 0:
            for p_averaged, p_model in zip(self_param_detached, model_param_detached):
                p_averaged.copy_(p_model)

        if self.n_averaged > 0:
            for p_averaged, p_model in zip(self_param_detached, model_param_detached):
                n_averaged = self.n_averaged.to(p_averaged.device)
                p_averaged.copy_(self.avg_fn(p_averaged, p_model, n_averaged))

        if not self.use_buffers:
            for b_swa, b_model in zip(self.module.buffers(), model.buffers()):
                b_swa.copy_(b_model.to(b_swa.device).detach())

        self.n_averaged += 1

class EMAModel(AveragingBaseModel):
    def __init__(self, model, cuda=False, decay=0.99, use_buffers=False):
        super().__init__(model=model, cuda=cuda, avg_fn=get_ema_avg_fn(decay), use_buffers=use_buffers)

class SWAModel(AveragingBaseModel):
    def __init__(self, model, cuda=False, use_buffers=False):
        super().__init__(model=model, cuda=cuda, avg_fn=get_swa_avg_fn(), use_buffers=use_buffers)

class T_ADEMAModel(AveragingBaseModel):
    def __init__(self, model, cuda=False, alpha=0.99, use_buffers=False):
        super().__init__(model=model, cuda=cuda, avg_fn=get_t_adema_fn(alpha), use_buffers=use_buffers)


if __name__=="__main__":
    # 创建 ResNet18 模型
    import torchvision.models as models
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    class RandomDataset(torch.utils.data.Dataset):
        def __init__(self, size=224):
            self.data = torch.randn(size, 3, 224, 224)
            self.labels = torch.randint(0, 2, (size,))

        def __getitem__(self, index):
            return self.data[index], self.labels[index]

        def __len__(self):
            return len(self.data)


    model = models.resnet18(pretrained=False)
    # model = model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # 创建数据加载器
    train_dataset = RandomDataset()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 定义权重平均模型
    swa_model = SWAModel(model, cuda=True)
    ema_model = EMAModel(model, cuda=True)
    t_adema_model = T_ADEMAModel(model, cuda=True)

    for epoch in range(5):
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{5}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 更新权重平均模型
            ema_model.update(model)
            swa_model.update(model)
            t_adema_model.update(model)

    # 测试模型
    test_dataset = RandomDataset(size=100)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    def evaluate(model):
        model.eval()  # 切换到评估模式
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"模型准确率：{accuracy * 100:.2f}%")

    # 原模型测试
    print("Model Evaluation:")
    evaluate(model.to('cuda'))   #
    # 测试权重平均模型
    print("SWAModel Evaluation:")
    evaluate(swa_model.to('cuda'))

    print("EMAModel Evaluation:")
    evaluate(ema_model.to('cuda'))

    print("T-ADEMAModel Evaluation:")
    evaluate(t_adema_model.to('cuda'))

    # 仅仅测试是否能够跑通，过程中也有比元模型要低的时候，而且权值平均主要是用与训练中后期
    # Model Evaluation:
    # 模型准确率：46.00 %
    # SWAModel Evaluation:
    # 模型准确率：54.00% %
    # EMAModel Evaluation:
    # 模型准确率：58.00% %
    # T - ADEMAModel Evaluation:
    # 模型准确率：58.00% %
