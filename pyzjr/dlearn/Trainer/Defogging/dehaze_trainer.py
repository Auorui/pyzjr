import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from pyzjr.dlearn import AverageMeter, get_lr
from pyzjr.devices import load_owned_device, release_gpu_memory


class DefoggingTrainEpoch():
    def __init__(self,
                 model,
                 loss_function,
                 optimizer,
                 lr_scheduler,
                 total_epoch,
                 use_amp=False,
                 device=load_owned_device()):
        super().__init__()
        self.device = device
        self.model = model.to(device)
        self.model_train = self.model.train()
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.total_epoch = total_epoch
        self.scaler = None
        if use_amp:
            self.scaler = GradScaler()

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        losses = AverageMeter()
        release_gpu_memory()
        with tqdm(total=self.total_epoch, desc=f'Epoch {epoch}/{self.total_epoch}', postfix=dict, mininterval=0.3) as pbar:
            for batch in train_loader:
                source_img, target_img = batch[0].type(torch.FloatTensor).cuda(), batch[1].type(torch.FloatTensor).cuda()
                self.optimizer.zero_grad()
                if self.scaler is None:
                    outputs = self.model(source_img)
                    loss = self.loss_function(outputs, target_img)
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                else:
                    with autocast():
                        outputs = self.model(source_img)
                        loss = self.loss_function(outputs, target_img)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                losses.update(loss.item())

                pbar.set_postfix(**{'total_loss': losses.avg,
                                    'lr': get_lr(self.optimizer)})
                pbar.update(1)
        return losses.avg

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        PSNR = AverageMeter()
        release_gpu_memory()
        with tqdm(total=self.total_epoch, desc=f'Epoch {epoch}/{self.total_epoch}', postfix=dict, mininterval=0.3) as pbar:
            for batch in val_loader:
                source_img, target_img = batch[0].cuda(), batch[1].cuda()
                with torch.no_grad():
                    output = self.model(source_img).clamp_(-1, 1)

                mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
                psnr = 10 * torch.log10(1 / mse_loss).mean()
                PSNR.update(psnr.item(), source_img.size(0))
                pbar.set_postfix(**{'psnr': PSNR.avg})
                pbar.update(1)
        return PSNR.avg