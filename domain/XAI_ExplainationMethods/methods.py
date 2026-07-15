import torch
import numpy as np
import torch.nn.functional as F
from torchcam.methods import GradCAM, GradCAMpp

class SaliencyExplainer:
    def __init__(self, model):
        self.model = model

    def attribute(self, image_tensor, target_class):
        """
        Compute Vanilla Saliency map (gradients of target score w.r.t input image).
        image_tensor: shape [1, 3, H, W]
        target_class: integer class index
        """
        # Ensure the model is in eval mode
        self.model.eval()
        
        # Clone and require grad
        input_img = image_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_img)
        
        # Compute score for target class
        score = output[0, target_class]
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Obtain absolute gradients and take the max across channels
        grads = input_img.grad.data.abs()
        saliency_map = grads[0].max(dim=0)[0].cpu().numpy()
        
        # Normalize to [0, 1]
        saliency_min, saliency_max = saliency_map.min(), saliency_map.max()
        if saliency_max > saliency_min:
            saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min + 1e-8)
        else:
            saliency_map = np.zeros_like(saliency_map)
            
        return saliency_map

class IntegratedGradientsExplainer:
    def __init__(self, model, steps=50):
        self.model = model
        self.steps = steps

    def attribute(self, image_tensor, target_class, baseline_tensor=None):
        """
        Compute Integrated Gradients attribution map.
        image_tensor: shape [1, 3, H, W]
        target_class: integer class index
        """
        self.model.eval()
        
        if baseline_tensor is None:
            baseline_tensor = torch.zeros_like(image_tensor)
            
        device = image_tensor.device
        baseline_tensor = baseline_tensor.to(device)
        
        # Linear interpolation path between baseline and input image
        alphas = np.linspace(0, 1, self.steps)
        grads_list = []
        
        for alpha in alphas:
            # Interpolated image
            interpolated = baseline_tensor + alpha * (image_tensor - baseline_tensor)
            interpolated = interpolated.clone().detach().requires_grad_(True)
            
            # Forward and backward
            output = self.model(interpolated)
            score = output[0, target_class]
            
            self.model.zero_grad()
            score.backward()
            
            grads_list.append(interpolated.grad.data.cpu().numpy()[0])
            
        # Average the gradients across all steps
        avg_grads = np.mean(np.array(grads_list), axis=0) # [3, H, W]
        
        # Calculate delta (input - baseline)
        delta = (image_tensor - baseline_tensor).cpu().detach().numpy()[0] # [3, H, W]
        
        # Multiply delta by average gradients
        integrated_grad = delta * avg_grads
        
        # Sum or take max absolute across channels
        ig_map = np.max(np.abs(integrated_grad), axis=0) # [H, W]
        
        # Normalize to [0, 1]
        ig_min, ig_max = ig_map.min(), ig_map.max()
        if ig_max > ig_min:
            ig_map = (ig_map - ig_min) / (ig_max - ig_min + 1e-8)
        else:
            ig_map = np.zeros_like(ig_map)
            
        return ig_map

class GradCAMExplainer:
    def __init__(self, model, model_name, use_pp=True):
        self.model = model.model  # access inner backbone model
        self.model_name = model_name.lower()
        self.use_pp = use_pp
        self.cam_extractor = None
        self._init_extractor()

    def _init_extractor(self):
        # Determine target layer based on model name
        target_layer = None
        if "resnet" in self.model_name:
            target_layer = "layer4"
        elif "efficientnet" in self.model_name:
            # For EfficientNet, the last convolutional features are under features
            target_layer = self.model.features[-1]
        elif "mobilenet" in self.model_name:
            target_layer = self.model.features[-1]
        
        # Instantiate GradCAM or GradCAM++ extractor
        cls = GradCAMpp if self.use_pp else GradCAM
        try:
            if target_layer:
                self.cam_extractor = cls(self.model, target_layer=target_layer)
            else:
                self.cam_extractor = cls(self.model)
        except Exception as e:
            # Fallback to automatic layer detection
            self.cam_extractor = cls(self.model)

    def attribute(self, image_tensor, target_class):
        """
        Compute Grad-CAM or Grad-CAM++ heatmap.
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(image_tensor)
        
        # Compute CAM
        cam = self.cam_extractor(target_class, output)
        
        # Resize to original image size
        _, _, H, W = image_tensor.shape
        cam_resized = F.interpolate(
            cam[0].unsqueeze(1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        cam_map = cam_resized.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam_map.min(), cam_map.max()
        if cam_max > cam_min:
            cam_map = (cam_map - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam_map = np.zeros_like(cam_map)
            
        return cam_map

    def clean(self):
        if self.cam_extractor:
            self.cam_extractor.remove_hooks()


def get_explanation(model_core, image_tensor, method, target_class=1):
    """
    Unified helper function to extract explainability heatmaps.
    model_core: ModelCore instance
    image_tensor: Shape [1, 3, H, W]
    method: 'saliency', 'integrated_gradients', 'gradcam', or 'gradcampp'
    """
    method = method.lower()
    if method == 'saliency':
        explainer = SaliencyExplainer(model_core)
        return explainer.attribute(image_tensor, target_class)
    elif method == 'integrated_gradients':
        explainer = IntegratedGradientsExplainer(model_core)
        return explainer.attribute(image_tensor, target_class)
    elif method == 'gradcam':
        explainer = GradCAMExplainer(model_core, model_core.model_name, use_pp=False)
        heatmap = explainer.attribute(image_tensor, target_class)
        explainer.clean()
        return heatmap
    elif method == 'gradcampp':
        explainer = GradCAMExplainer(model_core, model_core.model_name, use_pp=True)
        heatmap = explainer.attribute(image_tensor, target_class)
        explainer.clean()
        return heatmap
    else:
        raise ValueError(f"Unknown explanation method: {method}")
