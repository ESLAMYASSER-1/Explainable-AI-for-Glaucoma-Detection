from helpers import get_TBlogger, Settings
from models import DataModel
from utils import ModelCore

import glob
from pathlib import Path
import torch
# from torchinfo import summary
import torch.nn.functional as F
import torchvision.models as models
from torchcam.methods import GradCAMpp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

import numpy as np

def normalize_image(img):
    """
    Normalize an image to range 0..1.
    
    img: np.array (H,W,C) or (C,H,W), dtype uint8 or float
    Returns: float32 image in 0..1
    """
    img = img.astype(np.float32)  # ensure float
    img_min = img.min()
    img_max = img.max()
    
    if img_max > img_min:  # avoid division by zero
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = img - img_min  # all zeros
    
    return img_norm

def normalize_cam(cam):
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam

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
    efficientnet_b4 = models.efficientnet_b4(weights= models.EfficientNet_B4_Weights.DEFAULT)

    passed_models = ["mobileNetv3", "efficientnet_b4"] # TODO: always except efficientnet_b4, and mobileNetv3 not good at axplainability 
   
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

    try:
        efficientnet_b4 = ModelCore.load_from_checkpoint(glob.glob(f"{PREDICT_EXPERIMENT_NAME}/efficientnet_b4/checkpoints/*.ckpt")[0],model = efficientnet_b4, num_epochs= settings.NUM_EPOCHS)
    except:
        print(f"❌ efficientnet_b4 not available in Experiment: {PREDICT_EXPERIMENT_NAME.split("/")[-1]}")
        passed_models.append("efficientnet_b4")


    model_lst = [("resnet152", resnet152), ("resnet50", resnet50), ("efficientnet_b4", efficientnet_b4), ("mobileNetv3", mobileNetv3), ("resnet101", resnet101),]
    
    for model_name, model in model_lst:
        if model_name not in passed_models:
            # print([f"layer name: {name}\n" for (name, module) in model.model.named_children()], sep="\n")
            
            dirPath = Path(f"./assets/gradcam_{PREDICT_EXPERIMENT_NAME.split("/")[-1]}_{model_name}_{len(glob.glob("./assets/*"))}")
            Path.mkdir(dirPath)

            model.to("cuda")
            model.eval()

            try:
                cam_extractor = GradCAMpp(model.model, target_layer='layer4')
            except:
                target_layer = model.model.features[-2]
                cam_extractor = GradCAMpp(model.model, target_layer = target_layer)
           
            TP = 0
            TOTP = 0

            batch= 0
            for images, labels in val_loader:
                batch += 1
                images = images.to("cuda")
                labels = labels.long().to("cuda")

                # ---- Accuracy (batch, no CAM here) ----
                
                outputs = model.model(images)
                preds = outputs.argmax(dim=1)

                TP += (preds == labels).sum().item()
                TOTP += labels.size(0)
                
                # ---- Grad-CAM++ (one image at a time) ----
                for i in range(len(images)):
                    img = images[i].unsqueeze(0)   # [1, C, H, W]
                    label = labels[i].item()

                    # Fresh forward pass (NO no_grad)
                    output = model.model(img)
                    pred = output.argmax(dim=1).item()

                    # Compute CAM
                    cam = cam_extractor(pred, output)   # [1, 7, 7]
                    
                    # Resize CAM
                    cam = F.interpolate(
                        cam[0].unsqueeze(1),              # [1,1,7,7]
                        size=(224, 224),
                        mode='bilinear',
                        align_corners=False
                    )
                    cam = cam.squeeze().cpu().numpy()   # [224,224]
                    
                    # ---- Image for visualization ----
                    img_np = img.squeeze(0).permute(1,2,0).cpu().numpy()
                    img_np = img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
                    img_np = np.clip(img_np, 0, 1)

                    # Save raw image
                    plt.imsave(
                        dirPath/f"glaucoma_{batch}_{i}.png",
                        img_np,
                        dpi=300
                    )

                    cam = normalize_cam(cam)
                    # Overlay CAM
                    fig, ax = plt.subplots(figsize=(6,6))
                    ax.imshow(img_np)
                    im = ax.imshow(cam, cmap='jet', alpha=0.5)
                    ax.set_title(f"Pred: {pred}, True: {label}")
                    ax.axis('off')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)

                    fig.savefig(
                        dirPath/f"glaucoma_{batch}_{i}_CAM.png",
                        bbox_inches='tight',
                        dpi=300
                    )
                    plt.close(fig)
                TP += sum(preds.to("cpu") == labels.to("cpu"))
                TOTP += len(labels)
            
            with open(dirPath/f"metrics.txt", "w") as f:
                f.write(f"{"#"*20} Model Name: {model_name} {"#"*20}\nTrue predictions: {TP},\nTotal predictions: {TOTP},\nAccuracy: {(TP/TOTP)*100}%")
            
            print(f"{"#"*20} Model Name: {model_name} {"#"*20}\nTrue predictions: {TP},\nTotal predictions: {TOTP},\nAccuracy: {(TP/TOTP)*100}%")
            print()
            print()