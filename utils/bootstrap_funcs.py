import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix
)

def bootstrap_cls(prob_list, label_list, threshold=0.5, times=100, random_state=None):
    rng = np.random.default_rng(random_state)
    
    prob_list = np.asarray(prob_list)
    label_list = np.asarray(label_list)
    n = len(label_list)
    
    # Store metrics across bootstrap samples
    aucs, prcs = [], []
    accs, sens, spes = [], [], []
    ppvs, npvs = [], []
    
    for _ in range(times):
        # Sample with replacement
        indices = rng.integers(0, n, n)
        probs_sample = prob_list[indices]
        labels_sample = label_list[indices]
        
        # Convert probabilities to predicted labels
        preds = (probs_sample >= threshold).astype(int)
        
        # --- Metrics ---
        # AUC (handle edge case: only one class present)
        try:
            auc = roc_auc_score(labels_sample, probs_sample)
        except ValueError:
            auc = np.nan
        
        # PRC AUC
        try:
            prc = average_precision_score(labels_sample, probs_sample)
        except ValueError:
            prc = np.nan
        
        # Accuracy
        acc = accuracy_score(labels_sample, preds)
        
        # Confusion matrix: tn, fp, fn, tp
        tn, fp, fn, tp = confusion_matrix(labels_sample, preds, labels=[0, 1]).ravel()
        
        # Sensitivity (Recall)
        sen = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        
        # Specificity
        spe = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        
        # PPV (Precision)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        
        # NPV
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        
        # Collect
        aucs.append(auc)
        prcs.append(prc)
        accs.append(acc)
        sens.append(sen)
        spes.append(spe)
        ppvs.append(ppv)
        npvs.append(npv)
    
    # Convert to arrays and compute std (ignore NaNs)
    def safe_std(x):
        return np.nanstd(x, ddof=1)
    
    return (
        safe_std(aucs),
        safe_std(prcs),
        safe_std(accs),
        safe_std(sens),
        safe_std(spes),
        safe_std(ppvs),
        safe_std(npvs),
    )