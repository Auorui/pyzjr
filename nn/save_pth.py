import os
import torch
import torch.nn as nn
from collections import OrderedDict

_torch_save = torch.save  # copy to avoid recursion errors
best_val_loss = float('inf')
init_metrics = float('-inf')

def get_single_device_state_dict(model: nn.Module):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module.state_dict()
    if hasattr(model, 'module') and not isinstance(model, nn.Sequential):
        return model.module.state_dict()
    if len({param.device for param in model.parameters()}) > 1:
        device = next(model.parameters()).device
        single_gpu_model = type(model)().to(device)
        single_gpu_model.load_state_dict(model.state_dict())
        return single_gpu_model.state_dict()
    return model.state_dict()

def save_model_simplify(net, save_dir, epoch, save_period=30):
    """
    Save the model according to training rounds

    Args:
        net: model to be saved
        save_dir: path that the model would be saved
        epoch: current epoch
        save_period: The frequency of saving the model during training.
                    The default is 10.
    """
    net = get_single_device_state_dict(net)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if epoch % save_period == 0:
        try:
            _torch_save(net, os.path.join(save_dir, f"model_epoch_{epoch}.pth"))
            print(f'\033[34mSave model to {os.path.join(save_dir, f"model_epoch_{epoch}.pth")}')
        except Exception as e:
            print(f"\033[31mFailed to save model at epoch {epoch}: {e}")


def save_model_best_loss(net, save_dir, val_loss, epoch, save_period=None):
    """
    Save the model based on training rounds (optional),
    and save the best model based on validation loss.

    Args:
        net: model to be saved
        save_dir: path that the model would be saved
        val_loss (float): Verification losses for the current period. Usually, certain
                        indicators can also be used, such as dice, accuracy, etc.
        epoch: current epoch
        save_period (int, optional): The frequency of saving the model during training.
                                    The default is None.
    """
    global best_val_loss
    os.makedirs(save_dir, exist_ok=True)
    net = get_single_device_state_dict(net)
    if epoch % save_period == 0 and save_period is not None:
        _torch_save(net,
                   os.path.join(save_dir, f'epoch{epoch}_loss{val_loss:.4}.pth'))
        print(f"\033[34mmodel saved at epoch —— {epoch}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join(save_dir, "best_loss_model.pth")
        _torch_save(net, best_model_path)
        print(f"\033[34mBest model saved at epoch —— {epoch} loss —— {val_loss}")
        print(f'\033[34mSave best model to {best_model_path}')


def save_model_best_metrics(net, save_dir, metric, epoch, minlimit=None):
    """
    Save the model if the current metric is the best observed so far.

    Args:
        net: model to be saved.
        save_dir: path that the model would be saved.
        metric (float): The current metric value.
        epoch (int): The current epoch.
        minlimit (float, optional): The minimum threshold for the metric. If the
                                     metric is less than or equal to `minlimit`,
                                     the model will not be saved.
    """
    global init_metrics
    os.makedirs(save_dir, exist_ok=True)
    net = get_single_device_state_dict(net)
    if minlimit is not None and metric <= minlimit:
        print(f"\033[33mMetric {metric} is below the minimum limit of {minlimit}, model will not be saved this epoch.")
        return
    if epoch <= 1 or metric > init_metrics:
        init_metrics = metric
        model_path = os.path.join(save_dir, f'best_metric_model.pth')
        _torch_save(net, model_path)
        print(f"\033[34mBest model saved at epoch —— {epoch}, metric —— {init_metrics}")
        print(f'\033[34mSave best model to {model_path}')


def SaveModelPth(net, save_dir, epoch, total_epochs=None, val_loss=None, metric=None, save_period=80, extra_name=None):
    """
    Save the model at specified intervals or if the model achieves the best validation loss/metric.
    Also saves the model at the last epoch.

    Args:
        net: model to be saved
        save_dir: path that the model would be saved
        epoch: current epoch
        total_epochs (int, optional): Total number of epochs. If provided, the model will be saved at the last epoch.
        val_loss (float, optional): Validation loss for the current epoch.
        metric (float, optional): The current metric value.
        save_period (int, optional): The frequency of saving the model during training.
        extra_name (int, optional): Prevent them from being saved as the same file and overlapping with each other.
    """
    global best_val_loss, init_metrics
    os.makedirs(save_dir, exist_ok=True)
    net = get_single_device_state_dict(net)
    extra_str = f"_{extra_name}" if extra_name else ""
    # Save model at specified intervals
    if epoch % save_period == 0:
        try:
            torch.save(net, os.path.join(save_dir, f"model_epoch_{epoch}{extra_str}.pth"))
            print(f'\033[34mModel saved at epoch {epoch} to {os.path.join(save_dir, f"model_epoch_{epoch}.pth")}')
        except Exception as e:
            print(f"\033[31mFailed to save model at epoch {epoch}: {e}")

    # Save model if it achieves the best validation loss
    if val_loss is not None and val_loss < best_val_loss:
        best_val_loss = val_loss
        best_loss_model_path = os.path.join(save_dir, f"best_loss_model{extra_str}.pth")
        torch.save(net, best_loss_model_path)
        print(f"\033[34mBest model saved at epoch —— {epoch} loss —— {val_loss}")
        print(f'\033[34mSave best model to {best_loss_model_path}')

    # Save model if it achieves the best metric
    if metric is not None:
        if metric > init_metrics:
            init_metrics = metric
            best_metric_model_path = os.path.join(save_dir, f"best_metric_model{extra_str}.pth")
            torch.save(net, best_metric_model_path)
            print(f"\033[34mBest model saved at epoch —— {epoch}, metric —— {init_metrics}")
            print(f'\033[34mSave best model to {best_metric_model_path}')

    # Save model at the last epoch
    if total_epochs is not None and epoch >= total_epochs // 5:
        last_model_path = os.path.join(save_dir, f"last{extra_str}.pth")
        torch.save(net, last_model_path)


def load_partial_weights(net, pretrained_path=None, pretrained_dict=None, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
    加载部分权重到模型中，只加载匹配的权重，并且返回更新后的模型。

    Args:
        net (torch.nn.Module): 要加载部分权重的当前模型。
        pretrained_path (str, optional): 预训练权重文件的路径。如果提供了此路径，将从该路径加载权重。默认为None。
        pretrained_dict (dict, optional): 预训练的权重字典。如果提供了此字典，则直接从该字典加载权重，忽略pretrained_path。默认为None。

    Returns:
        torch.nn.Module: 加载了部分权重的模型。
    """
    net = net.to(device)
    model_dict = get_single_device_state_dict(net)
    if pretrained_path is not None:
        pretrained_dict = torch.load(pretrained_path, map_location=device)
    else:
        pretrained_dict = pretrained_dict
    # 创建存储匹配、未匹配和缺失权重的列表
    load_key, no_load_key, missing_keys, temp_dict = [], [], [], {}

    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
                print(f"\033[31mKey: {k}, Model Shape: {model_dict[k].shape}, Pretrained Shape: {v.shape}")
        else:
            missing_keys.append(k)
            print(f"\033[31mKey: {k} not in model_dict.")

    model_dict.update(temp_dict)
    net.load_state_dict(model_dict)
    if not no_load_key and not missing_keys:
        print(f"\033[34m\nAll {len(load_key)} keys were successfully loaded.")
    else:
        print(f"\033[31m\nSuccessful Load Keys:{str(load_key)[:300]}……\n"
              f"Successful Load Key Num:{len(load_key)}")
        print(f"\033[31m\nFail To Load Keys:{str(no_load_key)[:100]}……\n"
              f"Fail To Load Key Num:{len(no_load_key)}")
        print(f"\033[31m\nMissing Keys:{str(missing_keys)[:100]}……\n"
              f"Missing Key Num:{len(missing_keys)}")
    return net


def save_checkpoint(net=None, optimizer=None, epoch=None, train_losses=None, train_metric=None, val_loss=None,
                    val_metric=None, check_loss=None, savepath=None, model_name=None):
    """
    保存模型检查点，包括模型权重、优化器状态、训练损失、验证损失等信息。
    """
    net_weights = get_single_device_state_dict(net)
    save_json = {
        'net_state_dict': net_weights,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'train_metric': train_metric,
        'val_loss': val_loss,
        'val_metric': val_metric
    }
    if val_loss < check_loss:
        savepath = savepath + '/{}_best_params.pth'.format(model_name)
        check_loss = val_loss
    else:
        savepath = savepath + '/{}_epoch_{}.pth'.format(model_name, epoch)
    torch.save(save_json, savepath)
    print("checkpoint of {}th epoch saved at {}".format(epoch, savepath))

    return check_loss


def load_checkpoint(model=None, optimizer=None, checkpoint_path=None,  losses_flag=False):
    """
    从检查点文件中加载模型权重、优化器状态和训练信息。
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    if losses_flag:
        losses = checkpoint['train_losses']
        return model, optimizer, start_epoch, losses
    else:
        return model, optimizer, start_epoch

def load_checkpoint_model(model, weights, map_location):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, map_location=map_location)
    return model

def load_model_weights(model, weights, strict=True, map_location=None):
    """加载模型权重并自动处理多卡训练带来的键名前缀问题"""
    checkpoint = torch.load(weights, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # 支持多种前缀格式
        if k.startswith("module."):
            name = k[7:]  # 移除"module."
        elif k.startswith("_orig_mod."):
            name = k[10:]  # 处理torch.compile添加的前缀
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=strict)
    return model
