import os
import glob
from pathlib import Path
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
import gradio as gr
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Project imports
from helpers import Settings
from utils import ModelCore
from domain.XAI_ExplainationMethods import get_explanation
from domain.ExplainationEvaluation import deletion_metric, insertion_metric
from domain.Visualization import overlay_heatmap

# Load settings
settings = Settings()

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Scan available experiments
LOGS_DIR = Path("../logs")
available_experiments = []
if LOGS_DIR.exists():
    available_experiments = [d.name for d in LOGS_DIR.iterdir() if d.is_dir()]
if not available_experiments:
    available_experiments = [settings.PREDICT_EXPERIMENT_NAME]

# Keep track of loaded models to avoid reloading
loaded_models_cache = {}

def get_model(experiment_name, model_name):
    cache_key = f"{experiment_name}_{model_name}"
    if cache_key in loaded_models_cache:
        return loaded_models_cache[cache_key]
    
    # Instantiate backbone
    model_name_lower = model_name.lower()
    if "resnet50" in model_name_lower:
        backbone = models.resnet50(weights=None)
    elif "resnet101" in model_name_lower:
        backbone = models.resnet101(weights=None)
    elif "resnet152" in model_name_lower:
        backbone = models.resnet152(weights=None)
    elif "mobilenet" in model_name_lower:
        backbone = models.mobilenet_v3_large(weights=None)
    elif "efficientnet" in model_name_lower:
        backbone = models.efficientnet_b4(weights=None)
    else:
        raise ValueError(f"Unknown backbone for model: {model_name}")
        
    # Find checkpoint
    ckpt_pattern = f"../logs/{experiment_name}/{model_name}/checkpoints/*.ckpt"
    ckpt_files = glob.glob(ckpt_pattern)
    if not ckpt_files:
        # Try relative to current dir
        ckpt_pattern = f"logs/{experiment_name}/{model_name}/checkpoints/*.ckpt"
        ckpt_files = glob.glob(ckpt_pattern)
        
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint (.ckpt) found for {model_name} in experiment {experiment_name}.")
        
    ckpt_path = ckpt_files[0]
    print(f"Loading checkpoint: {ckpt_path}")
    
    model_core = ModelCore.load_from_checkpoint(
        ckpt_path,
        model_name=model_name,
        model=backbone,
        fine_tuning=True,
        num_epochs=settings.NUM_EPOCHS
    )
    model_core.to(device)
    model_core.eval()
    
    loaded_models_cache[cache_key] = model_core
    return model_core

# Define input transformation for inference
inference_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])
Resize = A.Compose([
    A.Resize(224, 224)
])


def diagnose_and_explain(image, experiment_name, model_name, alpha_blend):
    if image is None:
        return "Please upload an image.", None, None, None, None, None, None
        
    try:
        # Load the model
        model_core = get_model(experiment_name, model_name)
    except Exception as e:
        return f"Error loading model: {str(e)}", None, None, None, None, None, None

    # Preprocess image
    h, w, c = image.shape
    # Resize original image to 224x224 for display and XAI mapping
    image_resized = Resize(image=image)["image"]
    
    transformed = inference_transforms(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(device) # [1, 3, 224, 224]

    # Model inference
    with torch.no_grad():
        logits = model_core(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        
    prob_healthy = probs[0].item()
    prob_glaucoma = probs[1].item()
    
    diagnosis_text = (
        f"### Diagnosis Results\n"
        f"**Prediction**: {'🔴 Glaucoma' if pred_class == 1 else '🟢 Healthy'}\n\n"
        f"**Confidence Scores**:\n"
        f"- Glaucoma: `{prob_glaucoma * 100:.2f}%`\n"
        f"- Healthy: `{prob_healthy * 100:.2f}%`"
    )
    
    # Generate explanations
    methods = ['saliency', 'integrated_gradients', 'gradcam', 'gradcampp']
    heatmaps = {}
    overlays = {}
    
    for method in methods:
        try:
            heatmap = get_explanation(model_core, image_tensor, method, target_class=1)
            heatmaps[method] = heatmap
            # Blend heatmap with image_resized
            overlays[method] = overlay_heatmap(image_resized, heatmap, alpha=alpha_blend, colormap_name='jet')
        except Exception as e:
            print(f"Error computing {method}: {e}")
            overlays[method] = np.zeros((224, 224, 3))
            heatmaps[method] = np.zeros((224, 224))
            
    # Quantitative evaluation of explanations
    eval_results = []
    for method in methods:
        heatmap = heatmaps[method]
        # Deletion
        _, _, del_auc = deletion_metric(model_core, image_tensor, heatmap, target_class=1, num_steps=10)
        # Insertion
        _, _, ins_auc = insertion_metric(model_core, image_tensor, heatmap, target_class=1, num_steps=10)
        
        eval_results.append({
            "Method": method.upper().replace("GRADCAM", "Grad-CAM"),
            "Deletion AUC ↓ (Lower = Better)": round(del_auc, 4),
            "Insertion AUC ↑ (Higher = Better)": round(ins_auc, 4)
        })
        
    df_eval = pd.DataFrame(eval_results)
    
    return (
        diagnosis_text,
        overlays['saliency'],
        overlays['integrated_gradients'],
        overlays['gradcam'],
        overlays['gradcampp'],
        df_eval,
        image_resized
    )

# Build Gradio UI
with gr.Blocks(title="Glaucoma Diagnosis & Explainable AI (XAI) Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 👁️ Glaucoma Diagnosis & Explainable AI (XAI) Dashboard
        Upload a fundus image to predict the probability of glaucoma and inspect the decision regions of different deep learning models using state-of-the-art XAI attributions.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Fundus Image", type="numpy")
            
            with gr.Group():
                gr.Markdown("### Model Configuration")
                experiment_dd = gr.Dropdown(
                    choices=available_experiments,
                    value=settings.PREDICT_EXPERIMENT_NAME if settings.PREDICT_EXPERIMENT_NAME in available_experiments else available_experiments[0],
                    label="Experiment Checkpoint"
                )
                model_dd = gr.Dropdown(
                    choices=["resnet50", "resnet101", "resnet152", "mobileNetv3", "efficientnet_b4"],
                    value="resnet50",
                    label="Model Backbone"
                )
                alpha_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Heatmap Alpha Blend"
                )
            
            submit_btn = gr.Button("Analyze & Explain", variant="primary")
            
        with gr.Column(scale=2):
            output_diagnosis = gr.Markdown("### Diagnosis Results\n*Please upload an image and click Analyze.*")
            
            with gr.Tabs():
                with gr.TabItem("XAI Visual Overlays"):
                    with gr.Row():
                        with gr.Column():
                            img_orig = gr.Image(label="Preprocessed Image", show_label=True, interactive=False)
                            img_saliency = gr.Image(label="Saliency Map Overlay", show_label=True, interactive=False)
                        with gr.Column():
                            img_ig = gr.Image(label="Integrated Gradients Overlay", show_label=True, interactive=False)
                            img_gcam = gr.Image(label="Grad-CAM Overlay", show_label=True, interactive=False)
                    with gr.Row():
                        with gr.Column():
                            img_gcampp = gr.Image(label="Grad-CAM++ Overlay", show_label=True, interactive=False)
                        with gr.Column():
                            pass  # spacer
                            
                with gr.TabItem("Quantitative Faithfulness Evaluation"):
                    gr.Markdown(
                        """
                        ### 📊 Explanation Faithfulness Evaluation
                        * **Deletion AUC**: Sequentially removes pixels in order of highest attribution. A steeper drop (lower AUC) is better, indicating the explainer correctly identified the most critical pixels.
                        * **Insertion AUC**: Sequentially inserts pixels into a blank canvas in order of highest attribution. A faster rise (higher AUC) is better, showing that restoring these pixels quickly recreates model confidence.
                        """
                    )
                    output_table = gr.Dataframe(headers=["Method", "Deletion AUC ↓", "Insertion AUC ↑"], interactive=False)
                    
    submit_btn.click(
        fn=diagnose_and_explain,
        inputs=[input_image, experiment_dd, model_dd, alpha_slider],
        outputs=[output_diagnosis, img_saliency, img_ig, img_gcam, img_gcampp, output_table, img_orig]
    )

if __name__ == "__main__":
    demo.launch(server_name="http://localhost", server_port=7861, share=False)
