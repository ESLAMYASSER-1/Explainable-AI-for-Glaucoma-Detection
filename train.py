from helpers import get_TBlogger, Settings, setup_logging
from models import DataModel
from utils import ModelCore

import torchvision.transforms as trns
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

import torchvision.models as models

import logging

setup_logging()

logger = logging.getLogger(__name__)
logger.info("Application started")

settings = Settings()



dataModel = DataModel(settings.DATASET_DIR, 
                      settings.CSV_FILE, 
                      settings.FUNDUS_DIR, 
                      settings.TRAIN_SIZE, 
                      settings.INCLUDE_TEST)


# trasnformations = trns.Compose([
#     trns.RandomResizedCrop(224, scale=(0.9, 1.0)),
#     trns.RandomHorizontalFlip(p=0.5),
#     trns.RandomRotation(10),
#     trns.ColorJitter(0.1, 0.1, 0.05, 0.01),
#     trns.ToTensor(),
#     trns.Normalize(mean=[0.485, 0.456, 0.406],
#                    std=[0.229, 0.224, 0.225]),
# ])

trasnformations =A.Compose([
    A.RandomResizedCrop((224, 224), scale=(0.9, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.7),
    A.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.05,
        hue=0.01,
        p=0.5
    ),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

train_loader, val_loader, test_loader = dataModel.get_loaders(settings.BATCH_SIZE, trasnformations)
img = next(iter(train_loader))[0][0]

resnet50 = models.resnet50(weights=None)
resnet101 = models.resnet101(weights=None)
resnet152 = models.resnet152(weights=None)
mobileNetv3 = models.mobilenet_v3_large(weights= None)
model_lst = [("resnet50", resnet50), ("resnet101", resnet101), ("resnet152", resnet152), ("mobileNetv3", mobileNetv3)]


for model_name, backbone in model_lst:
    TBlogger = get_TBlogger(settings.LOGGING_DIR, settings.EXPERIMENT_NAME, version=model_name)

    model = ModelCore(backbone, settings.NUM_EPOCHS,2e-4)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",    # metric to monitor
        patience=10,            # number of epochs with no improvement
        mode="min",            # minimize the monitored metric
        verbose=True
    )

    trainer = Trainer(
        max_epochs=settings.NUM_EPOCHS,
        logger=TBlogger,
        accelerator="cuda",
        devices=1,
        enable_progress_bar=True,
        callbacks=[early_stop_callback],
    )

    # tuner = Tuner(trainer)



    # lr_finder = tuner.lr_find(
    #     model,
    #     train_dataloaders=train_loader,
    #     val_dataloaders=val_loader,
    #     min_lr=3e-5,
    #     max_lr=1
    # )
    # best_lr = lr_finder.suggestion()
    # model.lr = best_lr
    # print(best_lr)

    trainer.fit(model= model, train_dataloaders=train_loader, val_dataloaders=val_loader)



