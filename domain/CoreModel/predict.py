from helpers import get_TBlogger, Settings
from models import DataModel

import glob
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn import metrics
import numpy as np
import pandas as pd 

from utils import load_models, bootstrap_cls

from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

settings = Settings()

TBlogger = get_TBlogger(settings.LOGGING_DIR, settings.EXPERIMENT_NAME)

def predict():
    dataModel = DataModel(settings.DATASET_DIR, 
                        settings.CSV_FILE, 
                        settings.FUNDUS_DIR, 
                        settings.TRAIN_SIZE, 
                        settings.INCLUDE_TEST)


    trasnformations =A.Compose([
        A.RandomResizedCrop((224, 224), scale=(0.9, 1.0)),
        #A.HorizontalFlip(p=0.5),
        #A.Rotate(limit=10, p=0.5),
        #A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.7),
        #A.ColorJitter(
        #     brightness=0.1,
        #     contrast=0.1,
        #     saturation=0.05,
        #     hue=0.01,
        #     p=0.5
        # ),
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

        model_lst, passed_models = load_models(modelList=["resnet152", "resnet101", "resnet50", "mobileNetv3", "efficientnet_b4"], pretrained=True, EXPERIMENT_NAME=PREDICT_EXPERIMENT_NAME)
        val_results_df = []
        test_results_df= []
    
        for model_name, model in model_lst:
            if model_name not in passed_models:
                with torch.inference_mode():
                    model.to("cuda")
                    model.eval()
                    label_list = []
                    prob_list = []
                    for i, l in tqdm(val_loader): 
                        l= l.long()
                        logits= model.model(i.to("cuda"))
                        logits = torch.softmax(logits, dim=1)[:, 1]   

                        label_list = np.concatenate((label_list, l), axis=None)
                        prob_list = np.concatenate((prob_list, logits.detach().cpu().numpy()), axis=None)
                    

                    # Calculate accuracy, auroc, sensitivity, specificity
                    fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
                    precision, recall, thresholds_ = metrics.precision_recall_curve(label_list, prob_list)
                    thres_val = round(thresholds[np.argmax(tpr - fpr)], 3)
                    prob_list[np.where(prob_list >= thres_val)] = 1
                    prob_list[np.where(prob_list != 1)] = 0

                    print(sum(prob_list == label_list)/ len(label_list))

                    tn, fp, fn, tp = metrics.confusion_matrix(label_list, prob_list).ravel()
                    print(f'AUROC on the validation set is {round(metrics.auc(fpr, tpr), 3)}')
                    print(f'AUPRC on the validation set is {round(metrics.auc(recall, precision), 3)}')
                    print(f'Threshold determined by Youden index is {thres_val}')
                    print(f'Accuracy on the validation set is {round((tp + tn) / (tp + tn + fp + fn), 3)}')
                    print(f'Sensitivity on the validation set is {round(tp / (tp + fn), 3)}')
                    print(f'Specificity on the validation set is {round(tn / (tn + fp), 3)}')
                    print(f'PPV on the validation set is {round(tp / (tp + fp), 3)}')
                    print(f'NPV on the validation set is {round(tn / (tn + fn), 3)}')

                    auc_std, prc_std, acc_std, sen_std, spe_std, ppv_std, npv_std = \
                            bootstrap_cls(prob_list=prob_list, label_list=label_list, threshold=thres_val, times=100)

                    val_results_df.append([f"{model_name}", f"{PREDICT_EXPERIMENT_NAME.split("/")[-1]}", f"{thres_val}",
                                            f"{format(metrics.auc(fpr, tpr), '.3f')} ({auc_std})",
                                            f"{format(metrics.auc(recall, precision), '.3f')} ({prc_std})",
                                            f"{format((tp + tn) / (tp + tn + fp + fn), '.3f')} ({acc_std})",
                                            f"{format(tp / (tp + fn), '.3f')} ({sen_std})",
                                            f"{format(tn / (tn + fp), '.3f')} ({spe_std})",
                                            f"{format(tp / (tp + fp), '.3f')} ({ppv_std})",
                                            f"{format(tn / (tn + fn), '.3f')} ({npv_std})",
                                            ])
                    
                    
                    
                prob_list = []
                label_list = []
                with torch.no_grad():
                    for data in tqdm(test_loader):
                        images, labels = data[0].float().to("cuda"), data[1].cpu().numpy()
                        outputs = model(images)
                        label_list = np.concatenate((label_list, labels), axis=None)
                        prob_list = np.concatenate((prob_list, torch.sigmoid(outputs)[:, 1].detach().cpu().numpy()), axis=None)

                    fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
                    precision, recall, thresholds_ = metrics.precision_recall_curve(label_list, prob_list)
                    auc_std, prc_std, acc_std, sen_std, spe_std, ppv_std, npv_std = \
                        bootstrap_cls(prob_list=prob_list, label_list=label_list, threshold=thres_val, times=100)

                    prob_list[np.where(prob_list >= thres_val)] = 1
                    prob_list[np.where(prob_list != 1)] = 0
                    tn, fp, fn, tp = metrics.confusion_matrix(label_list, prob_list).ravel()

                    print(f'AUROC on the test set is {round(metrics.auc(fpr, tpr), 3)}')
                    print(f'AUPRC on the test set is {round(metrics.auc(recall, precision), 3)}')
                    print(f'Threshold determined by Youden index is {thres_val}')
                    print(f'Accuracy on the test set is {round((tp + tn) / (tp + tn + fp + fn), 3)}')
                    print(f'Sensitivity on the test set is {round(tp / (tp + fn), 3)}')
                    print(f'Specificity on the test set is {round(tn / (tn + fp), 3)}')
                    print(f'PPV on the test set is {round(tp / (tp + fp), 3)}')
                    print(f'NPV on the test set is {round(tn / (tn + fn), 3)}')


                    
                    test_results_df.append([f"{model_name}", f"{PREDICT_EXPERIMENT_NAME.split("/")[-1]}", f"{thres_val}",
                                        f"{format(metrics.auc(fpr, tpr), '.3f')} ({auc_std})",
                                        f"{format(metrics.auc(recall, precision), '.3f')} ({prc_std})",
                                        f"{format((tp + tn) / (tp + tn + fp + fn), '.3f')} ({acc_std})",
                                        f"{format(tp / (tp + fn), '.3f')} ({sen_std})",
                                        f"{format(tn / (tn + fp), '.3f')} ({spe_std})",
                                        f"{format(tp / (tp + fp), '.3f')} ({ppv_std})",
                                        f"{format(tn / (tn + fn), '.3f')} ({npv_std})",
                                            ])
                        
        val_results_df = pd.DataFrame(val_results_df)
        val_results_df.columns = ['Model', 'EXPERIMENT_NAME', 'Threshold', 'AUROC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity',
                            'PPV', 'NPV']
        
        test_results_df = pd.DataFrame(test_results_df)
        test_results_df.columns = ['Model', 'EXPERIMENT_NAME', 'Threshold', 'AUROC', 'AUPRC', 'Accuracy', 'Sensitivity', 'Specificity',
                            'PPV', 'NPV']
        
        val_results_df.to_csv("val_results_cls.csv", index=False, encoding="cp1252")
        test_results_df.to_csv("test_results_cls.csv", index=False, encoding="cp1252")

        
        print(val_results_df)
        print(test_results_df)




