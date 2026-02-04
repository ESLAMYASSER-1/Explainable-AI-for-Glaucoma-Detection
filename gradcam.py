from helpers import get_TBlogger, Settings
from models import DataModel
from utils import ModelCore

import glob
import torch
from torchinfo import summary
import torch.nn.functional as F
import torchvision.models as models
from torchcam.methods import GradCAMpp
import numpy as np
import matplotlib.pyplot as plt
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

if settings.USE_ALL_EXPERIMENTS_TO_PREDICT:
    experiments = glob.glob("../logs/*")
else:
    experiments = [f"../logs/{settings.PREDICT_EXPERIMENT_NAME}"]

for PREDICT_EXPERIMENT_NAME in experiments:
    print("#"*40, "Experiment Name:", PREDICT_EXPERIMENT_NAME.split("/")[-1], "#"*40)

    resnet50 = models.resnet50(weights=None)
    resnet101 = models.resnet101(weights=None)
    resnet152 = models.resnet152(weights=None)
    mobileNetv3 = models.mobilenet_v3_large(weights= None)
    passed_models = []
    try:
        resnet50 = ModelCore.load_from_checkpoint(glob.glob(f"{PREDICT_EXPERIMENT_NAME}/resnet50/checkpoints/*.ckpt")[0],model = resnet50, num_epochs= settings.NUM_EPOCHS)
    except:
        print(f"❌ resnet50 not available in Experiment: {PREDICT_EXPERIMENT_NAME.split("/")[-1]}")
        passed_models.append("resnet50")

    try:
        resnet101 = ModelCore.load_from_checkpoint(glob.glob(f"{PREDICT_EXPERIMENT_NAME}/resnet101/checkpoints/*.ckpt")[0],model = resnet101, num_epochs= settings.NUM_EPOCHS)
    except:
        print(f"❌ resnet101 not available in Experiment: {PREDICT_EXPERIMENT_NAME.split("/")[-1]}")
        passed_models.append("resnet101")

    try:
        resnet152 = ModelCore.load_from_checkpoint(glob.glob(f"{PREDICT_EXPERIMENT_NAME}/resnet152/checkpoints/*.ckpt")[0],model = resnet152, num_epochs= settings.NUM_EPOCHS)
    except:
        print(f"❌ resnet152 not available in Experiment: {PREDICT_EXPERIMENT_NAME.split("/")[-1]}")
        passed_models.append("resnet152")

    try:
        mobileNetv3 = ModelCore.load_from_checkpoint(glob.glob(f"{PREDICT_EXPERIMENT_NAME}/mobileNetv3/checkpoints/*.ckpt")[0],model = mobileNetv3, num_epochs= settings.NUM_EPOCHS)
    except:
        print(f"❌ mobileNetv3 not available in Experiment: {PREDICT_EXPERIMENT_NAME.split("/")[-1]}")
        passed_models.append("mobileNetv3")

    model_lst = [("resnet50", resnet50), ("resnet101", resnet101), ("resnet152", resnet152), ("mobileNetv3", mobileNetv3)]
    for model_name, model in model_lst:
        if model_name not in passed_models:
            
            model.to("cuda")
            model.eval()
            # summary(model, input_size=(settings.BATCH_SIZE, 3, 224, 224))
            cam_extractor = GradCAMpp(model.model, target_layer='layer4')
            TP = 0
            TOTP = 0
            for images, labels in val_loader:
                images = images.to("cuda")
                labels = labels.long().to("cuda")
                
                # Forward pass
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                
                for i in range(len(images)):
                    cam = cam_extractor(preds[i].item(), outputs[i].unsqueeze(0), retain_graph=True)
                    # cam is a 2D tensor (H, W), resize to original image size if needed
                    
                    cam = cam[0][0].unsqueeze(0).unsqueeze(0)
                    
                    cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)

                    cam = cam.squeeze().cpu().numpy()
                    # Visualize overlay
                    img = images[i].permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Denormalize
                    img = np.clip(img, 0, 1)
                    
                    plt.imshow(img)
                    plt.imshow(cam, cmap='jet', alpha=0.5)  # Overlay CAM
                    plt.title(f"Pred: {preds[i].item()}, True: {labels[i].item()}")
                    plt.axis('off')
                    plt.show()
                TP += sum(preds.to("cpu") == labels.to("cpu"))
                TOTP += len(labels)
            print(f"{"#"*20} Model Name: {model_name} {"#"*20}\nTrue predictions: {TP},\nTotal predictions: {TOTP},\nAccuracy: {(TP/TOTP)}%")
            print()
            print()