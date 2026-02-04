import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy


class ModelCoreWOscheduler(pl.LightningModule):
    def __init__(self, model, num_epochs):
        super().__init__()
        self.model = model
        self.num_epochs= num_epochs
        self.lr = 1e-6

        # Freeze all layers except layer4
        # for name, param in self.model.named_parameters():
        #     if "layer4" not in name:
        #         param.requires_grad = False

        # Replace final layer for binary classification
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Log to TensorBoard and progress bar
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        # return [optimizer], [scheduler]
        return optimizer
    

class ModelCore(pl.LightningModule):
    def __init__(self, model, num_epochs, lr=5e-5):
        super().__init__()
        self.model = model
        self.num_epochs= num_epochs
        self.lr = lr

        # Freeze all layers except layer4
        # for name, param in self.model.named_parameters():
        #     if "layer4" not in name:
        #         param.requires_grad = False

        # Replace final layer for binary classification
        try:
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        except :
            self.model.classifier[-1]= torch.nn.Linear(self.model.classifier[-1].in_features, 2)
        self.loss_fn = torch.nn.CrossEntropyLoss() 

        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        acc = self.train_acc(logits.softmax(dim=-1), y)
        # Log to TensorBoard and progress bar
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Accuracy
        acc = self.val_acc(logits.softmax(dim=-1), y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        # return [optimizer], [scheduler]
        return optimizer
