import sys
import torch
import numpy as np

# Add src to path if needed (but we run from src)
from helpers import Settings
from utils import ModelCore
from domain.XAI_ExplainationMethods import get_explanation
from domain.ExplainationEvaluation import deletion_metric, insertion_metric
from domain.Visualization import overlay_heatmap

def run_test():
    print("Starting verification of the XAI, Evaluation, and Visualization pipeline...")
    
    # 1. Test settings loading
    settings = Settings()
    print(f"Loaded settings for project: {settings.PROJECT_NAME}")
    
    # 2. Try loading a trained model
    experiment_name = settings.PREDICT_EXPERIMENT_NAME # nighnth_try_pretrained
    model_name = "resnet50"
    print(f"Loading model {model_name} from experiment {experiment_name}...")
    
    import torchvision.models as models
    import glob
    
    backbone = models.resnet50(weights=None)
    ckpt_pattern = f"../logs/{experiment_name}/{model_name}/checkpoints/*.ckpt"
    ckpt_files = glob.glob(ckpt_pattern)
    if not ckpt_files:
        print(f"Error: Could not find checkpoint for verification. Checkpoint pattern: {ckpt_pattern}")
        sys.exit(1)
        
    ckpt_path = ckpt_files[0]
    print(f"Found checkpoint: {ckpt_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for verification: {device}")
    try:
        model_core = ModelCore.load_from_checkpoint(
            ckpt_path,
            model_name=model_name,
            model=backbone,
            fine_tuning=True,
            num_epochs=settings.NUM_EPOCHS
        )
        model_core.to(device)
        model_core.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
        
    # 3. Create a dummy image tensor
    dummy_image = torch.rand(1, 3, 224, 224).to(device)
    print(f"Generated dummy input image of shape: {dummy_image.shape}")
    
    # 4. Test each explanation method
    methods = ['saliency', 'integrated_gradients', 'gradcam', 'gradcampp']
    heatmaps = {}
    
    for method in methods:
        try:
            print(f"Generating explanation using: {method}...")
            heatmap = get_explanation(model_core, dummy_image, method, target_class=1)
            assert heatmap.shape == (224, 224), f"Heatmap shape is {heatmap.shape}, expected (224, 224)"
            assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0, f"Heatmap values not in [0, 1]: min={heatmap.min()}, max={heatmap.max()}"
            heatmaps[method] = heatmap
            print(f"✅ {method} generated successfully!")
        except Exception as e:
            print(f"❌ Failed generating {method}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    # 5. Test quantitative metrics
    for method, heatmap in heatmaps.items():
        try:
            print(f"Evaluating {method} using deletion and insertion metrics...")
            del_steps, del_probs, del_auc = deletion_metric(model_core, dummy_image, heatmap, target_class=1, num_steps=5)
            ins_steps, ins_probs, ins_auc = insertion_metric(model_core, dummy_image, heatmap, target_class=1, num_steps=5)
            
            assert len(del_steps) == 6, f"Expected 6 steps for 5-step deletion, got {len(del_steps)}"
            assert len(ins_steps) == 6, f"Expected 6 steps for 5-step insertion, got {len(ins_steps)}"
            assert 0.0 <= del_auc <= 1.0, f"Deletion AUC out of bounds: {del_auc}"
            assert 0.0 <= ins_auc <= 1.0, f"Insertion AUC out of bounds: {ins_auc}"
            
            print(f"✅ {method} metrics computed: Deletion AUC = {del_auc:.4f}, Insertion AUC = {ins_auc:.4f}")
        except Exception as e:
            print(f"❌ Failed evaluation for {method}: {e}")
            sys.exit(1)
            
    # 6. Test visualization
    try:
        print("Testing heatmap overlay blending...")
        dummy_img_np = np.random.rand(224, 224, 3)
        overlay = overlay_heatmap(dummy_img_np, heatmaps['resnet50' if 'resnet50' in heatmaps else 'saliency'], alpha=0.5)
        assert overlay.shape == (224, 224, 3), f"Overlay shape is {overlay.shape}, expected (224, 224, 3)"
        assert overlay.min() >= 0.0 and overlay.max() <= 1.0, f"Overlay out of bounds: min={overlay.min()}, max={overlay.max()}"
        print("✅ Heatmap overlay test passed!")
    except Exception as e:
        print(f"❌ Failed overlay test: {e}")
        sys.exit(1)
        
    print("\n🎉 ALL PIPELINE VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_test()
