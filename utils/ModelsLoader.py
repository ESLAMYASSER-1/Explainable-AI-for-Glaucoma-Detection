import glob
import torch
import torchvision.models as models
from utils import ModelCore
from helpers import Settings



def load_models(modelList:list = [], pretrained:bool= False, EXPERIMENT_NAME: str = None):
    """Model loading for training and transfer learning of specific models 

    Args:
        modelList (list): specify list of models to load
        pretrained (bool): pretrained from old expreiment of not 
        EXPERIMENT_NAME (str): Old Experimant name to load pretrained model

    Returns:
        model_lst (list[tuple]): list of ('model name', Model)
        passed_models (list): list of models that couldn't be loaded
    """
    settings = Settings()
    passed_models = []

    if not pretrained :
        resnet50 = models.resnet50(weights=None)
        resnet101 = models.resnet101(weights=None)
        resnet152 = models.resnet152(weights=None)
        mobilenetv3 = models.mobilenet_v3_large(weights= None)
        efficientnet_b4 = models.efficientnet_b4(weights= None)
        model_lst = {"resnet50": resnet50, "resnet101": resnet101, "resnet152": resnet152, "mobileNetv3": mobilenetv3, "efficientnet_b4": efficientnet_b4}

    else:
        resnet50 = models.resnet50(weights=None)
        resnet101 = models.resnet101(weights=None)
        resnet152 = models.resnet152(weights=None)
        mobilenetv3 = models.mobilenet_v3_large(weights= None)
        efficientnet_b4 = models.efficientnet_b4(weights= None)
        model_lst = {"resnet50": resnet50, "resnet101": resnet101, "resnet152": resnet152, "mobileNetv3": mobilenetv3, "efficientnet_b4": efficientnet_b4}
        model_lst = {name: model for name, model in model_lst.items() if name in modelList}

        for name, model in model_lst.items():

            try:
                model_lst[name] = ModelCore.load_from_checkpoint(glob.glob(f"{EXPERIMENT_NAME}/{name}/checkpoints/*.ckpt")[0],model_name= name, model = model, fine_tuning= True, num_epochs= settings.NUM_EPOCHS)
            except Exception as e:
                print(e)
                print(f"❌ {name} not available in Experiment: {EXPERIMENT_NAME}")
                passed_models.append(name)

    passed_models = [name for name in passed_models if name in modelList]
    return model_lst.items(), passed_models


