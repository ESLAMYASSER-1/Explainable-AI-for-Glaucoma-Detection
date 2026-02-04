from helpers import get_TBlogger, Settings
from models import DataModel
from utils import ModelCore


import torch
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

import logging

logger = logging.getLogger(__name__)

settings = Settings()

TBlogger = get_TBlogger(settings.LOGGING_DIR, settings.EXPERIMENT_NAME)


dataModel = DataModel(settings.DATASET_DIR, 
                      settings.CSV_FILE, 
                      settings.FUNDUS_DIR, 
                      settings.TRAIN_SIZE, 
                      settings.INCLUDE_TEST)


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


resnet50 = models.resnet50(weights=None)
resnet101 = models.resnet101(weights=None)
resnet152 = models.resnet152(weights=None)
mobileNetv3 = models.mobilenet_v3_large(weights= None)



resnet50 = ModelCore.load_from_checkpoint("../logs/seventh_try/resnet50/checkpoints/epoch=44-step=13005.ckpt",model = resnet50, num_epochs= settings.NUM_EPOCHS)
resnet101 = ModelCore.load_from_checkpoint("../logs/seventh_try/resnet101/checkpoints/epoch=53-step=15606.ckpt",model = resnet101, num_epochs= settings.NUM_EPOCHS)
resnet152 = ModelCore.load_from_checkpoint("../logs/seventh_try/resnet152/checkpoints/epoch=68-step=19941.ckpt",model = resnet152, num_epochs= settings.NUM_EPOCHS)
mobileNetv3 = ModelCore.load_from_checkpoint("../logs/seventh_try/mobileNetv3/checkpoints/epoch=37-step=10982.ckpt",model = mobileNetv3, num_epochs= settings.NUM_EPOCHS)

model_lst = [("resnet50", resnet50), ("resnet101", resnet101), ("resnet152", resnet152), ("mobileNetv3", mobileNetv3)]
for model_name, model in model_lst:
    
    with torch.inference_mode():
        model.to("cuda")
        model.eval()
        TP = 0
        TOTP = 0
        for i, l in val_loader:
            l= l.long()
            logits= model.model(i.to("cuda"))
            logits = logits.argmax(axis=1)
            TP += sum(logits.to("cpu") == l)
            TOTP += len(l)
        print(f"{"#"*20} Model Name: {model_name} {"#"*20}\nTrue predictions: {TP},\nTotal predictions: {TOTP},\nAccuracy: {(TP/TOTP)}%")
        print()




