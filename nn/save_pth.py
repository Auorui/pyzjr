import os
import torch
from collections import OrderedDict

_torch_save = torch.save  # copy to avoid recursion errors
best_val_loss = float('inf')
init_metrics = float('-inf')

def _process_net_state_dict(net, use_devices=True):
    if torch.cuda.device_count() > 1 and use_devices:
        return net.module.state_dict()
    else:
        return net.state_dict()

def SaveModelPthSimplify(net, save_dir, epoch, save_period=30, use_devices=True):
    """
    Save the model according to training rounds

    Args:
        net: model to be saved
        save_dir: path that the model would be saved
        epoch: current epoch
        save_period: The frequency of saving the model during training.
                    The default is 10.
    """
    net = _process_net_state_dict(net, use_devices)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if epoch % save_period == 0:
        try:
            _torch_save(net, os.path.join(save_dir, f"model_epoch_{epoch}.pth"))
            print(f'\033[34mSave model to {os.path.join(save_dir, f"model_epoch_{epoch}.pth")}')
        except Exception as e:
            print(f"\033[31mFailed to save model at epoch {epoch}: {e}")


def SaveModelPthBestloss(net, save_dir, val_loss, epoch, save_period=None, use_devices=True):
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
    net = _process_net_state_dict(net, use_devices)
    if epoch % save_period == 0 and save_period is not None:
        _torch_save(net,
                   os.path.join(save_dir, f'epoch{epoch}_loss{val_loss:.4}.pth'))
        print(f"\033[34mmodel saved at epoch —— {epoch}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join(save_dir, "best_model.pth")
        _torch_save(net, best_model_path)
        print(f"\033[34mBest model saved at epoch —— {epoch} loss —— {val_loss}")
        print(f'\033[34mSave best model to {best_model_path}')


def SaveModelPthBestMetrics(net, save_dir, metric, epoch, minlimit=None, use_devices=True):
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
    net = _process_net_state_dict(net, use_devices)
    if minlimit is not None and metric <= minlimit:
        print(f"\033[33mMetric {metric} is below the minimum limit of {minlimit}, model will not be saved this epoch.")
        return
    if epoch <= 1 or metric > init_metrics:
        init_metrics = metric
        model_path = os.path.join(save_dir, f'best_model.pth')
        _torch_save(net, model_path)
        print(f"\033[34mBest model saved at epoch —— {epoch}, metric —— {init_metrics}")
        print(f'\033[34mSave best model to {model_path}')


def SaveModelPth(net, save_dir, epoch, total_epochs=None, val_loss=None, metric=None, save_period=80, use_devices=True):
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
    """
    global best_val_loss, init_metrics
    os.makedirs(save_dir, exist_ok=True)
    net = _process_net_state_dict(net, use_devices)
    # Save model at specified intervals
    if epoch % save_period == 0:
        try:
            torch.save(net, os.path.join(save_dir, f"model_epoch_{epoch}.pth"))
            print(f'\033[34mModel saved at epoch {epoch} to {os.path.join(save_dir, f"model_epoch_{epoch}.pth")}')
        except Exception as e:
            print(f"\033[31mFailed to save model at epoch {epoch}: {e}")

    # Save model if it achieves the best validation loss
    if val_loss is not None and val_loss < best_val_loss:
        best_val_loss = val_loss
        best_loss_model_path = os.path.join(save_dir, "best_loss_model.pth")
        torch.save(net, best_loss_model_path)
        print(f"\033[34mBest model saved at epoch —— {epoch} loss —— {val_loss}")
        print(f'\033[34mSave best model to {best_loss_model_path}')

    # Save model if it achieves the best metric
    if metric is not None:
        if metric > init_metrics:
            init_metrics = metric
            best_metric_model_path = os.path.join(save_dir, "best_metric_model.pth")
            torch.save(net, best_metric_model_path)
            print(f"\033[34mBest model saved at epoch —— {epoch}, metric —— {init_metrics}")
            print(f'\033[34mBest model saved to {best_metric_model_path}')

    # Save model at the last epoch
    if total_epochs is not None and epoch >= total_epochs // 2:
        last_model_path = os.path.join(save_dir, "last.pth")
        torch.save(net, last_model_path)


def load_partial_weights(net, pretrained_path=None, pretrained_dict=None, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), use_devices=True):
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
    model_dict = _process_net_state_dict(net, use_devices)
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
                    val_metric=None, check_loss=None, savepath=None, model_name=None, use_devices=True):
    """
    保存模型检查点，包括模型权重、优化器状态、训练损失、验证损失等信息。
    """
    net_weights = _process_net_state_dict(net, use_devices)
    save_json = {
        'net_state_dict': net_weights,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_losses': train_losses,
        'train_metric': train_metric,
        'val_loss': val_loss,
        'val_metric': val_metric
    }
    if  val_loss < check_loss:
        savepath = savepath + '/{}_best_params.pkl'.format(model_name)
        check_loss = val_loss
    else:
        savepath = savepath + '/{}_epoch_{}.pkl'.format(model_name, epoch)
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

def load_single_weights(model, model_path):
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model