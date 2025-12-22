from helpers import get_TBlogger, Settings, setup_logging
from models import DataModel
from utils import ModelCore, ModelCoreWOscheduler

import torchvision.transforms as trns
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from pytorch_lightning import Trainer

import torchvision.models as models

import logging

setup_logging()

logger = logging.getLogger(__name__)
logger.info("Application started")

settings = Settings()

TBlogger = get_TBlogger(settings.LOGGING_DIR, settings.EXPERIMENT_NAME)


dataModel = DataModel(settings.DATASET_DIR, 
                      settings.CSV_FILE, 
                      settings.FUNDUS_DIR, 
                      settings.TRAIN_SIZE, 
                      settings.INCLUDE_TEST)


trasnformations = trns.Compose([
    trns.ToTensor(),
])

train_loader, val_loader, test_loader = dataModel.get_loaders(settings.BATCH_SIZE, trasnformations)


resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model = ModelCoreWOscheduler(resnet50, settings.NUM_EPOCHS)

trainer = Trainer(
    max_epochs=settings.NUM_EPOCHS,
    logger=TBlogger,
    accelerator="cuda",
    devices=1,
    enable_progress_bar=True,
)

tuner = Tuner(trainer)



lr_finder = tuner.lr_find(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    min_lr=1e-9,
    max_lr=1
)
best_lr = lr_finder.suggestion()
model.lr = best_lr
print(best_lr)


model = ModelCore(resnet50, settings.NUM_EPOCHS)

trainer.fit(model= model, train_dataloaders=train_loader, val_dataloaders=val_loader)





