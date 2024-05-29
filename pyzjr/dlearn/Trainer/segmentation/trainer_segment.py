import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pyzjr.devices import load_owned_device
from pyzjr.dlearn import get_lr
import logging
logging.basicConfig(level=logging.INFO, format='\033[34m%(message)s')
logger = logging.getLogger("loggers")

class SegmentationTrainEpoch():
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
        total_loss = 0
        total_batches = len(train_loader)
        logger.info('Start Train')
        with tqdm(total=self.total_epoch, desc=f'Epoch {epoch}/{self.total_epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(train_loader):
                if iteration >= self.total_epoch:
                    break
                imgs, pngs, labels = batch

                with torch.no_grad():
                    imgs = imgs.type(torch.FloatTensor)
                    pngs = pngs.long()
                    labels = labels.type(torch.FloatTensor)
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        pngs = pngs.cuda()
                        labels = labels.cuda()

                self.optimizer.zero_grad()
                if self.scaler is None:
                    outputs = self.model_train(imgs)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                else:
                    with autocast():
                        outputs = self.model_train(imgs)
                        loss = self.loss_function(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                total_loss += loss.item()

                pbar.set_postfix(**{'total_loss': total_loss / (iteration+1),
                                    'lr': get_lr(self.optimizer)})
                pbar.update(1)
        logger.info('Finish Train')
        return total_loss / total_batches

    def evaluate(self, val_loader, epoch):
        val_loss = 0
        self.model_train.eval()
        logger.info('Start Validation')
        total_batches = len(val_loader)
        with tqdm(total=self.total_epoch, desc=f'Epoch {epoch}/{self.total_epoch}', postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(val_loader):
                if iteration >= self.total_epoch:
                    break
                imgs, pngs, labels = batch

                with torch.no_grad():
                    imgs = imgs.type(torch.FloatTensor)
                    pngs = pngs.long()
                    labels = labels.type(torch.FloatTensor)
                    if torch.cuda.is_available():
                        imgs = imgs.cuda()
                        pngs = pngs.cuda()
                        labels = labels.cuda()

                    outputs = self.model_train(imgs)
                    loss = self.loss_function(outputs, labels)
                    val_loss += loss.item()
                pbar.set_postfix(**{'total_loss': val_loss / (iteration+1)})
                pbar.update(1)

        logger.info('Finish Validation')
        return val_loss / total_batches

