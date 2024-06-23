import os
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from pyzjr.dlearn import get_lr

def accuracy_all_classes(output, label):
    batch_size = label.size(0)
    _, pred = output.max(1)
    accuracy = pred.eq(label).float().sum().item() / batch_size
    return accuracy

class ClassificationTrainEpoch():
    """
    用于训练和评估分类模型的工具类
    """
    def __init__(self, net, train_loader, val_loader, epochs, device,
                 loss_function, optimizer, lr_scheduler, scaler):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_epochs = epochs
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.current_valepoch = 0  # 新增一个内部变量来跟踪当前轮次
        self.current_trepoch = 0

    def evaluate(self):
        self.current_valepoch += 1
        self.net.eval()
        total_test_loss = 0
        total_correct_predictions = 0
        total_samples = 0
        total_batches = len(self.val_loader)
        with tqdm(self.val_loader, desc=f'Epoch {self.total_epochs}/{self.current_valepoch} / Test', postfix=dict, mininterval=0.3) as pbar:
            for batch_idx, data in enumerate(self.val_loader):
                img, label = data

                with torch.no_grad():
                    img = img.to(self.device)
                    label = label.to(self.device)
                    out = self.net(img)
                    test_loss = self.loss_function(out, label)
                    total_test_loss += test_loss.item()

                    _, predicted = out.max(1)
                    total_correct_predictions += predicted.eq(label).float().sum().item()
                    total_samples += label.size(0)

                pbar.set_postfix(**{'val_loss': total_test_loss / (batch_idx + 1)})
                pbar.update(1)

        self.average_test_loss = total_test_loss / total_batches
        accuracy = total_correct_predictions / total_samples

        return self.average_test_loss, accuracy


    def train_one_epoch(self):
        self.current_trepoch += 1
        self.net.train()
        total_train_loss = 0.0
        total_batches = len(self.train_loader)

        total_correct_predictions = 0
        total_samples = 0

        with tqdm(self.train_loader, desc=f'Epoch {self.total_epochs}/{self.current_trepoch} / Train', postfix=dict, mininterval=0.3) as pbar:
            for batch_idx, data in enumerate(self.train_loader):
                img, label = data
                with torch.no_grad():
                    img = img.to(self.device)
                    label = label.to(self.device)

                self.optimizer.zero_grad()
                if self.scaler is not None and torch.cuda.is_available():
                    with autocast():
                        output = self.net(img)
                        train_loss = self.loss_function(output, label)
                    self.scaler.scale(train_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output = self.net(img)
                    train_loss = self.loss_function(output, label)
                    train_loss.backward()
                    self.optimizer.step()

                total_train_loss += train_loss.item()
                _, predicted = output.max(1)
                total_correct_predictions += predicted.eq(label).float().sum().item()
                total_samples += label.size(0)

                pbar.set_postfix(**{'total_loss': total_train_loss / (batch_idx + 1),
                                    'accuracy': total_correct_predictions / total_samples,
                                    'lr': get_lr(self.optimizer)})
                pbar.update(1)
            self.lr_scheduler.step()

        self.average_train_loss = total_train_loss / total_batches
        average_accuracy = total_correct_predictions / total_samples

        return self.average_train_loss, average_accuracy

    def save_model(self, model, save_dir='./logs', save_period=10):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.current_valepoch % save_period == 0:
            torch.save(model.state_dict(), os.path.join(save_dir,
        f"model_epoch_{self.current_valepoch}_train_{self.average_train_loss:.2f}_val_{self.average_test_loss:.2f}.pth"))

if __name__=="__main__":
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.cuda.amp import GradScaler
    from torch.optim.lr_scheduler import StepLR
    # A simple CNN written casually, you should replace it with your own
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.fc1 = nn.Linear(128 * 8 * 8, 512)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

    # replace with your actual dataset
    dummy_dataset = ...

    # Set up your DataLoader
    train_loader = DataLoader(dummy_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dummy_dataset, batch_size=64, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up your model, loss function, optimizer, learning rate scheduler, and scaler
    model = SimpleCNN()
    model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    scaler = GradScaler()

    Epochs = 10

    classification_trainer = ClassificationTrainEpoch(
        net=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Epochs,
        device=device,
        loss_function=loss_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler
    )

    for epoch in range(Epochs):
        # Set up the Classification_train_and_eval instance
        train_loss, train_accuracy = classification_trainer.train_one_epoch()
        print(f"Epoch {epoch}/{Epochs} - Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        val_loss, val_accuracy = classification_trainer.evaluate()
        print(f"Epoch {epoch}/{Epochs} - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        classification_trainer.save_model(model)