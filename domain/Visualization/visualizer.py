import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def overlay_heatmap(image_np, heatmap_np, alpha=0.5, colormap_name='jet'):
    """
    Overlays a 2D heatmap on top of a 3D image.
    image_np: numpy array of shape [H, W, 3], range [0, 1] or [0, 255]
    heatmap_np: numpy array of shape [H, W], range [0, 1]
    alpha: blending weight
    colormap_name: matplotlib colormap name (e.g. 'jet', 'viridis', 'hot')
    
    Returns:
        overlay: blended RGB image as numpy float32 [0, 1]
    """
    # Force image to [0, 1] float
    img = image_np.copy().astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
        
    # Get colormap and apply to normalized heatmap
    colormap = plt.colormaps[colormap_name]
    heatmap_colored = colormap(heatmap_np.astype(np.float32))[:, :, :3]
    
    # Blend images
    overlay = (1.0 - alpha) * img + alpha * heatmap_colored
    overlay = np.clip(overlay, 0.0, 1.0)
    
    return overlay

def create_comparison_plot(image_np, explanations_dict, colormap_name='jet', alpha=0.5):
    """
    Creates a single figure with the original image and all overlay explanations side by side.
    image_np: original RGB image [H, W, 3]
    explanations_dict: dictionary mapping method_name -> 2D heatmap
    
    Returns:
        fig: matplotlib Figure object
    """
    num_explanations = len(explanations_dict)
    fig, axes = plt.subplots(1, num_explanations + 1, figsize=(4 * (num_explanations + 1), 4))
    
    # Force original image to [0, 1] float
    img = image_np.copy().astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
        
    # Plot original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Plot each overlay heatmap
    for idx, (method_name, heatmap) in enumerate(explanations_dict.items(), start=1):
        overlay = overlay_heatmap(img, heatmap, alpha=alpha, colormap_name=colormap_name)
        axes[idx].imshow(overlay)
        axes[idx].set_title(method_name)
        axes[idx].axis('off')
        
    plt.tight_layout()
    return fig
