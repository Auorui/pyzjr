import os
import torch
from pyzjr.core.general import is_not_none

_torch_save = torch.save  # copy to avoid recursion errors
best_val_loss = float('inf')
init_metrics = float('-inf')

def save_model_to_pth_best(model, save_dir, val_loss, epoch, save_period=None):
    """
    Save the model based on training rounds (optional),
    and save the best model based on validation loss.

    Args:
        model: model to be saved
        save_dir: path that the model would be saved
        val_loss (float): Verification losses for the current period. Usually, certain
                        indicators can also be used, such as dice, accuracy, etc.
        epoch: the epoch the model finished training
        save_period (int, optional): The frequency of saving the model during training.
                                    The default is None.
    """
    global best_val_loss
    os.makedirs(save_dir, exist_ok=True)
    if epoch % save_period == 0 and is_not_none(save_period):
        _torch_save(model.state_dict(),
                   os.path.join(save_dir, f'model_epoch_{epoch}_loss_{val_loss:.4}.pth'))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join(save_dir, "best_model.pth")
        _torch_save(model.state_dict(), best_model_path)
        print(f"Best model saved at epoch —— {epoch} loss —— {val_loss}")
        print(f'Save best model to {best_model_path}')


def save_model_to_pth_best_metrics(model, save_dir, metric, epoch):
    """
    Save the model with the best performance metrics to a. pth file.
    Args:
        model: model to be saved
        save_dir: path that the model would be saved
        metric: current metric
        epoch: the epoch the model finished training
    """
    global init_metrics
    os.makedirs(save_dir, exist_ok=True)
    if epoch <= 1 or metric > init_metrics:
        init_metrics = metric
        model_path = os.path.join(save_dir, f'best_model.pth')
        _torch_save(model.state_dict(), model_path)
        print(f"Best model saved at epoch —— {epoch}, metric —— {init_metrics}")
        print(f'Save best model to {model_path}')


def save_model_to_pth_simplify(model, save_dir, epoch, save_period=10):
    """
    Save the model according to training rounds

    Args:
        model: model to be saved
        save_dir: path that the model would be saved
        epoch: the epoch the model finished training
        save_period: The frequency of saving the model during training.
                    The default is 10.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if epoch % save_period == 0:
        _torch_save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth"))
