import os
import gc
import torch
import torch.nn as nn
from collections import OrderedDict

_torch_save = torch.save  # copy to avoid recursion errors
best_loss = float('inf')
best_metrics = float('-inf')

def de_parallel(model):
    """
    Convert parallel models (DataParallel or DistributedDataParallel) to single-GPU models.
    """
    return model.module if isinstance(model, (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model


def get_device(device_id=0):
    """
    Return appropriate device objects based on whether the system supports CUDA.
    """
    return torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


def release_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_single_device_state_dict(model: nn.Module):
    """Strip the multi-GPU training (DP/DDP) module. Return the pure model weight dictionary."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module.state_dict()
    if hasattr(model, 'module') and not isinstance(model, nn.Sequential):
        return model.module.state_dict()
    return model.state_dict()


def SaveModelPth(model, save_dir, epoch, total_epochs=None, val_loss=None, metric=None, save_period=80, extra_name=None):
    """
    Save the model at specified intervals or if the model achieves the best validation loss/metric.
    Also saves the model at the last epoch.

    Args:
        model: model to be saved
        save_dir: path that the model would be saved
        epoch: current epoch
        total_epochs (int, optional): Total number of epochs. If provided, the model will be saved at the last epoch.
        val_loss (float, optional): Validation loss for the current epoch.
        metric (float, optional): The current metric value.
        save_period (int, optional): The frequency of saving the model during training.
        extra_name (int, optional): Prevent them from being saved as the same file and overlapping with each other.
    """
    global best_loss, best_metrics
    os.makedirs(save_dir, exist_ok=True)
    model = get_single_device_state_dict(model)
    extra_str = f"_{extra_name}" if extra_name else ""

    # 1. Save regularly
    if epoch % save_period == 0:
        path = os.path.join(save_dir, f"model_epoch_{epoch}{extra_str}.pth")
        _torch_save(model, path)
        print(f'\033[34mModel saved at epoch {epoch} to {path}')

    # 2. Save the optimal loss
    if val_loss is not None and val_loss < best_loss:
        best_loss = val_loss
        path = os.path.join(save_dir, f"best_loss_model{extra_str}.pth")
        _torch_save(model, path)
        print(f"\033[34mBest loss model saved at epoch {epoch}, loss: {val_loss:.4f} -> {path}")

    # 3. Save the optimal metric
    if metric is not None and metric > best_metrics:
        best_metrics = metric
        path = os.path.join(save_dir, f"best_metric_model{extra_str}.pth")
        _torch_save(model, path)
        print(f"\033[34mBest metric model saved at epoch {epoch}, metric: {metric:.4f} -> {path}")

    # 3. Save the last one
    if total_epochs is not None and epoch >= total_epochs // 5 and epoch % 5 == 0:
        last_model_path = os.path.join(save_dir, f"last{extra_str}.pth")
        _torch_save(model, last_model_path)


def load_partial_model_weights(model, pretrained_path=None, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """
    Load partial weights into the model. Only loads matching keys (same name and shape)
    and skips unmatched or mismatched keys (e.g., classification heads with different classes).

    Args:
        net (torch.nn.Module): The current model to load weights into.
        pretrained_path (str, optional): Path to the pretrained weights file. Defaults to None.
        device (torch.device, optional): Device to map the weights to. Defaults to CUDA if available, else CPU.

    Returns:
        torch.nn.Module: The model with partially updated weights.
    """
    model = model.to(device)
    model_dict = model.state_dict()

    # Load pretrained weights from file or use the provided dictionary
    pretrained = torch.load(pretrained_path, map_location=device)
    pretrained_dict = pretrained.get("state_dict", pretrained.get("model", pretrained))

    # Handling the prefix of pre-trained weights (compatible with DDP / compile)
    clean_dict = {}
    for k, v in pretrained_dict.items():
        name = k[7:] if k.startswith("module.") else (k[10:] if k.startswith("_orig_mod.") else k)
        clean_dict[name] = v

    # Filter out layers that match both the name and the shape
    loaded_keys, skipped_keys = [], []
    for k, v in clean_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            model_dict[k] = v
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)

    # Load the updated dictionary
    model.load_state_dict(model_dict)

    print(f"\033[34m\n[Partial Load Summary]")
    print(f"\033[32mSuccessfully loaded {len(loaded_keys)} keys.")
    print(f"\033[33mSkipped {len(skipped_keys)} keys (shape mismatch or not in model).")
    if skipped_keys:
        print(f"\033[33mSkipped keys (first 10): {skipped_keys[:10]}")

    return model


def load_model_weights(model, weights, map_location=None, strict=True):
    """Load model weights and automatically handle key prefixes from multi-GPU training."""
    # PyTorch 2.6+ 默认 weights_only=True，这里需显式设为 False 以加载完整 checkpoint
    checkpoint = torch.load(weights, map_location=map_location, weights_only=False)

    # Extract the state dict, compatible with full checkpoint dicts or raw weight dicts
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        # Handle multiple prefix formats
        if k.startswith("module."):
            name = k[7:]  # Remove "module." prefix (from DataParallel/DDP)
        elif k.startswith("_orig_mod."):
            name = k[10:]  # Remove "_orig_mod." prefix (from torch.compile)
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=strict)
    return model


def save_checkpoint(save_dir, filename, epoch, model, optimizer, lr_scheduler,
                    val_loss, best_metric, **kwargs):
    """
    Save a checkpoint containing all essential training states as a .pth.tar file.

    Args:
        save_dir (str): Directory to save the checkpoint.
        filename (str): Base filename (e.g., 'checkpoint_epoch_10'). The '.pth.tar' will be appended.
        epoch (int): Current training epoch.
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        lr_scheduler: Learning rate scheduler.
        val_loss (float): Current validation loss.
        best_metric (float): The best metric observed so far (e.g., Dice or mIoU).
        **kwargs: Any extra info (e.g., model_config=config, notes="...").
    """
    os.makedirs(save_dir, exist_ok=True)
    final_path = filename + '.pth.tar'
    save_path = os.path.join(save_dir, final_path)
    # Pack all essential training states
    state = {
        'epoch': epoch,
        'state_dict': get_single_device_state_dict(model),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'val_loss': val_loss,
        'best_metric': best_metric,
    }
    # Pack any extra variables (e.g., your model config)
    state.update(kwargs)
    _torch_save(state, save_path)
    print(f'\033[34m[Epoch {epoch}] Checkpoint securely packed and saved to {save_path}\033[0m')


def load_checkpoint(checkpoint_path, map_location=None):
    """
    Load a .pth.tar checkpoint. Restores model weights and returns the full dictionary.
    Note: Optimizer and lr_scheduler should be restored externally from the returned checkpoint.

    Args:
        checkpoint_path (str): Path to the .pth.tar file.
        map_location: Device mapping for `torch.load`.

    Returns:
        tuple: (model, checkpoint_dict)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    # Unpack the tar file
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    return checkpoint