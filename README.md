# Explainable AI for Glaucoma Detection


This repository contains a comprehensive system for glaucoma detection from fundus images using deep learning. It features an interactive dashboard built with Gradio that not only provides a diagnosis but also explains the model's decision-making process using various Explainable AI (XAI) techniques. The project includes the full pipeline for data handling, model training, evaluation, and visualization.

## Features

*   **Multi-Model Diagnosis**: Utilizes several deep learning architectures for glaucoma classification, including ResNet50, ResNet101, ResNet152, MobileNetV3, and EfficientNet-B4.
*   **Interactive XAI Dashboard**: An easy-to-use web interface (`app.py`) to upload a fundus image and receive:
    *   A diagnostic prediction (Healthy or Glaucoma).
    *   Model confidence scores.
    *   Visual explanations (heatmaps) from multiple XAI methods overlaid on the image.
*   **State-of-the-Art XAI Methods**:
    *   **Saliency Maps**: Highlights pixels that most significantly impact the output score.
    *   **Integrated Gradients**: Attributes the prediction to input features by integrating gradients along a path from a baseline.
    *   **Grad-CAM**: Generates a coarse localization map of the important regions in the image.
    *   **Grad-CAM++**: An improved version of Grad-CAM that provides better object localization.
*   **Quantitative Faithfulness Evaluation**: The dashboard automatically calculates and displays metrics to measure how faithful the explanations are to the model's reasoning:
    *   **Deletion AUC**: Measures the drop in model confidence as the most "important" pixels (according to the heatmap) are progressively removed. A lower score is better.
    *   **Insertion AUC**: Measures the rise in model confidence as the most "important" pixels are progressively added to a blank image. A higher score is better.
*   **Training & Evaluation Pipeline**: The project is built with PyTorch Lightning, providing a structured workflow for training models, logging experiments with TensorBoard, and performing rigorous evaluation.

## Model Performance

The models were trained and evaluated on a private dataset. The performance on the test set is summarized below, with metrics reported as `mean (± standard deviation)` calculated via bootstrapping over 100 runs. The optimal classification threshold was determined using the Youden index on the validation set.

| Model           | AUROC           | AUPRC           | Accuracy        | Sensitivity     | Specificity     |
| --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| **ResNet50**    | 0.986 (±0.003)  | 0.983 (±0.004)  | 0.949 (±0.006)  | 0.958 (±0.008)  | 0.943 (±0.008)  |
| **ResNet101**   | 0.983 (±0.003)  | 0.978 (±0.003)  | 0.949 (±0.005)  | 0.927 (±0.010)  | 0.962 (±0.006)  |
| **ResNet152**   | 0.985 (±0.002)  | 0.979 (±0.003)  | 0.939 (±0.006)  | 0.930 (±0.009)  | 0.945 (±0.007)  |
| **EfficientNet-B4** | 0.860 (±0.009)  | 0.813 (±0.015)  | 0.753 (±0.009)  | 0.846 (±0.012)  | 0.694 (±0.013)  |

## Project Structure

The repository is organized into a modular structure to separate concerns:

```
├── app.py                      # Main Gradio dashboard application
├── main.py                     # Script to run prediction and evaluation
├── requirements.txt            # Project dependencies
├── domain/
│   ├── CoreModel/              # Model training and prediction logic
│   ├── ExplainationEvaluation/ # Implementation of Deletion/Insertion metrics
│   ├── Visualization/          # Heatmap and plotting utilities
│   └── XAI_ExplainationMethods/  # Saliency, IG, Grad-CAM implementations
├── utils/
│   ├── ModelCore.py            # PyTorch Lightning wrapper for models
│   ├── ModelsLoader.py         # Utility to load trained models
│   └── SMGDdataClass.py        # Custom PyTorch Dataset
├── helpers/
│   ├── settings.py             # Pydantic settings management (.env)
│   └── logger.py               # Logging configuration
├── models/
│   └── DataModel.py            # Data controller to create data loaders
└── logs/                       # Directory for TensorBoard logs and model checkpoints
```

## Setup and Usage

### Prerequisites

*   Python 3.8+
*   pip
*   CUDA-enabled GPU (recommended for performance)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/eslamyasser-1/eng_mariam_el-desoky_xai_gluacoma.git
    cd eng_mariam_el-desoky_xai_gluacoma
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration:**
    Create a `.env` file in the root directory and populate it with the necessary paths and settings. An example based on `helpers/settings.py` is:
    ```env
    PROJECT_NAME="XAI_Glaucoma"
    VERSION="1.0"
    LOGGING_DIR="../logs"
    EXPERIMENT_NAME="my_experiment"
    PREDICT_EXPERIMENT_NAME="nighnth_try_pretrained"
    DATASET_DIR="../path/to/your/dataset/"
    NUM_EPOCHS=100
    LR=0.0002
    ```
    **Note:** You will need to have pre-trained model checkpoints located in the directory specified by `LOGGING_DIR` and `PREDICT_EXPERIMENT_NAME` (e.g., `../logs/nighnth_try_pretrained/resnet50/checkpoints/*.ckpt`) for the application to work.

### Running the Application

All commands should be run from the root of the project directory.

#### Launch the Interactive Dashboard

To start the Gradio web interface for diagnosis and explanation:

```bash
python app.py
```

Navigate to the local URL displayed in the terminal (e.g., `http://localhost:7861`) to access the dashboard.

#### Run Model Training

To start a new training session for the models defined in `utils/ModelsLoader.py`:

```bash
python domain/CoreModel/train.py
```

Training progress and metrics will be logged to TensorBoard in the directory specified by `LOGGING_DIR` and `EXPERIMENT_NAME`.

#### Run Model Evaluation

To evaluate the performance of pre-trained models on the validation and test sets:

```bash
python main.py
```

This will generate `val_results_cls.csv` and `test_results_cls.csv` with detailed performance metrics.
