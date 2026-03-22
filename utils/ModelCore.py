import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
    

class ModelCore(pl.LightningModule):
    def __init__(self,model_name, model, fine_tuning, num_epochs, lr=5e-5):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.num_epochs= num_epochs
        self.lr = lr

        # Freeze all layers except layer4
        if fine_tuning:
            for name, param in self.model.named_parameters():
                if ("resnet" in model_name and "layer4" in name) \
                or ("efficientnet" in model_name and any(sub in name for sub in [ "features.7", "features.8", ]))\
                or ("mobilenet" in model_name and any(sub in name for sub in [ "features.16", "features.16", ])):
                    param.requires_grad = False
                        
                    
                        

        # Replace final layer for binary classification

        if ("resnet" in model_name):
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)

        elif ("efficientnet" in model_name):
            self.model.classifier[-1]= torch.nn.Linear(self.model.classifier[-1].in_features, 2)

        elif ("mobilenet" in model_name):
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
        
        return optimizer
