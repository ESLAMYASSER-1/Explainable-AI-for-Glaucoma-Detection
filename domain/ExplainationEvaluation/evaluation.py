import torch
import numpy as np

def deletion_metric(model, image_tensor, heatmap, target_class=1, num_steps=20, baseline=0.0):
    """
    Evaluates the explanation heatmap using the deletion metric.
    Successively removes pixels of highest attribution and records the drop in prediction probability.
    
    model: PyTorch model
    image_tensor: original preprocessed image tensor [1, 3, H, W]
    heatmap: 2D numpy array [H, W] normalized to [0, 1]
    target_class: index of target class (e.g. 1 for Glaucoma)
    num_steps: number of steps to divide deletion (e.g., 20 steps = 5% pixels deleted per step)
    baseline: constant value to replace deleted pixels with (e.g., 0.0)
    
    Returns:
        steps_frac: list of fractions of deleted pixels
        probs: list of target class probabilities
        auc_score: Area Under the Deletion Curve (AODC). Lower is better.
    """
    model.eval()
    H, W = heatmap.shape
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(flat_heatmap)[::-1].copy()  # descending order of attribution
    
    probs = []
    
    # Get initial probability
    with torch.no_grad():
        init_out = model(image_tensor)
        init_prob = torch.softmax(init_out, dim=1)[0, target_class].item()
        probs.append(init_prob)
        
    steps_frac = [0.0]
    total_pixels = len(flat_heatmap)
    
    for step in range(1, num_steps + 1):
        fraction = step / num_steps
        steps_frac.append(fraction)
        num_to_delete = int(fraction * total_pixels)
        indices_to_delete = sorted_indices[:num_to_delete]
        
        # Modify image
        with torch.no_grad():
            modified_image = image_tensor.clone()
            B, C, _, _ = modified_image.shape
            flat_img = modified_image.view(B, C, H * W)
            flat_img[:, :, indices_to_delete] = baseline
            
            # Predict
            out = model(modified_image)
            prob = torch.softmax(out, dim=1)[0, target_class].item()
            probs.append(prob)
            
    # Compute AUC
    auc_score = np.trapz(probs, steps_frac)
    
    return steps_frac, probs, float(auc_score)


def insertion_metric(model, image_tensor, heatmap, target_class=1, num_steps=20, baseline=0.0):
    """
    Evaluates the explanation heatmap using the insertion metric.
    Starts with a baseline image and successively restores pixels of highest attribution.
    
    model: PyTorch model
    image_tensor: original preprocessed image tensor [1, 3, H, W]
    heatmap: 2D numpy array [H, W] normalized to [0, 1]
    target_class: index of target class
    num_steps: number of steps to divide insertion
    baseline: starting value of pixels (e.g. 0.0)
    
    Returns:
        steps_frac: list of fractions of inserted pixels
        probs: list of target class probabilities
        auc_score: Area Under the Insertion Curve (AOIC). Higher is better.
    """
    model.eval()
    H, W = heatmap.shape
    flat_heatmap = heatmap.flatten()
    sorted_indices = np.argsort(flat_heatmap)[::-1].copy()  # descending order
    
    probs = []
    
    # Get initial baseline probability (all pixel values set to baseline)
    with torch.no_grad():
        baseline_img = torch.full_like(image_tensor, baseline)
        init_out = model(baseline_img)
        init_prob = torch.softmax(init_out, dim=1)[0, target_class].item()
        probs.append(init_prob)
        
    steps_frac = [0.0]
    total_pixels = len(flat_heatmap)
    
    for step in range(1, num_steps + 1):
        fraction = step / num_steps
        steps_frac.append(fraction)
        num_to_insert = int(fraction * total_pixels)
        indices_to_insert = sorted_indices[:num_to_insert]
        
        # Modify image
        with torch.no_grad():
            modified_image = torch.full_like(image_tensor, baseline)
            B, C, _, _ = modified_image.shape
            
            flat_orig = image_tensor.view(B, C, H * W)
            flat_mod = modified_image.view(B, C, H * W)
            
            flat_mod[:, :, indices_to_insert] = flat_orig[:, :, indices_to_insert]
            
            # Predict
            out = model(modified_image)
            prob = torch.softmax(out, dim=1)[0, target_class].item()
            probs.append(prob)
            
    # Compute AUC
    auc_score = np.trapz(probs, steps_frac)
    
    return steps_frac, probs, float(auc_score)
