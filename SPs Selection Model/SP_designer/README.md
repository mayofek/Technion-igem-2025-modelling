# Signal Peptide Prediction Project Guide

## Overview
This project implements deep learning models (CNN, Transformer, LSTM, MLP) for signal peptide prediction, supporting both classification (high/low secretion) and regression (quantitative activity prediction) tasks. The project includes dataset preparation, model training, and prediction workflows for bromelain expression optimization in *Bacillus subtilis*.

---

## Project Structure

```
SP_designer/
├── data/                           # Dataset files and creation guide
│   ├── Dataset_Creation_Guide.md   # Dataset format and requirements
│   ├── SP1.xlsx, SP2.xlsx         # datasets (B. amyloliquefaciens)
│   └── BSP1.xlsx, BSP2.xlsx, BSP3.xlsx  # datasets (B. subtilis)
├── classification/                 # Classification model training
│   ├── CNN/, Transformer/, LSTM/, MLP/  # Architecture-specific folders
│   └── dataset.py                  # Classification dataset loader
├── regression/                     # Regression model training
│   ├── CNN/, Transformer/, LSTM/, MLP/  # Architecture-specific folders
│   └── dataset.py                  # Regression dataset loader
└── prediction/                     # Unified prediction pipeline
    ├── Analysis_Commands_Guide.md  # Prediction and ranking analysis
    ├── unified_sp_prediction.py    # Main prediction script
    └── models/                     # Trained model files and results
```

---

## 1. Dataset Preparation

Before training models, prepare your datasets according to the required 7-column format with proper normalization and validation. The Dataset Creation Guide contains complete specifications for data structure, quality requirements, and available datasets (BSP1-3, SP1-2).

📖 **[Dataset Creation Guide](data/Dataset_Creation_Guide.md)**

---

## 2. Model Training

### Environment Setup

⚠️ **Google Colab Limitations**: Google Colab may have issues with file access, updating files on Google Drive, and handling long training sessions. For optimal results, **transfer learning and model predictions are strongly recommended to be performed locally on your PC**.

📁 **Directory Navigation**: You must navigate to the correct directory before running training scripts. The file paths in the commands must be adjusted according to your environment setup:
- **Google Colab**: Use `/content/drive/MyDrive/your_project_path/` for all file paths
- **Local PC**: Use your full local path like `C:\Users\user\Documents\igem\MODEL\COPY2\SP_designer\` for all file paths

### Training Strategies
1. **Only [Dataset]**: Train and test on same dataset
2. **[Source] validate [Target]**: Train on source, test on target
3. **[Source] and [Target]**: Train on combined data, test on target
4. **[Source] transfer [Target]**: Pretrain on source, finetune on target

### Architecture Options
- **CNN**: Convolutional Neural Network for local pattern recognition
- **Transformer**: Self-attention mechanism for global dependencies
- **LSTM**: Long Short-Term Memory for sequential patterns
- **MLP**: Multi-Layer Perceptron with attention mechanism

### Training Workflow

Each architecture (CNN, Transformer, LSTM, MLP) has dedicated folders under `classification/` and `regression/` with four training strategy subdirectories:
- `only_SP1/`, `only_SP2/`, `SP2_and_SP1/`, `SP2_trans_SP1/`

Training scripts are provided for each strategy:
- `train_[Architecture].py` - Direct training
- `pretrain_[Architecture].py` - Pretraining phase for transfer learning
- `finetune_[Architecture].py` - Finetuning phase for transfer learning

### Example Training Commands

#### Google Colab Setup
```python
import torch
print("✅ CUDA available:", torch.cuda.is_available())
from google.colab import drive
drive.mount('/content/drive')
```

#### Google Colab Training (CNN Classification)
```bash
%cd /content/drive/MyDrive/Colab\ Notebooks/repo/SP_designer/classification/CNN/only_SP1
!python3 train_CNN.py --train_path "/content/drive/MyDrive/Colab Notebooks/repo/SP_designer/data/BSP1.xlsx" --model_path "./CharCNN_best.pth.tar" --saved_values_path "./saved_values.xlsx" --save_folder "./"
```

#### Local PC Training (CNN Classification)
```bash
cd "C:\Users\user\Documents\igem\MODEL\COPY2\SP_designer\classification\CNN\only_SP1" && python3 train_CNN.py --train_path "C:\Users\user\Documents\igem\MODEL\COPY2\SP_designer\data\BSP1.xlsx" --model_path "./CharCNN_best.pth.tar" --saved_values_path "./saved_values.xlsx" --save_folder "./" --lr 0.001 --epochs 300 --batch_size 64 --optimizer Adam
```

For all other architectures and strategies, navigate to the corresponding directories and adjust the path arguments to match your environment.

---

## 3. Model Prediction and Analysis

After training your models, use the unified prediction pipeline to generate predictions and analyze results. The Analysis Commands Guide provides complete instructions for generating prediction files, extracting rankings, calculating statistical measures, and creating prioritized candidate lists.

📖 **[Analysis Commands Guide](prediction/Analysis_Commands_Guide.md)**

### Quick Example
```bash
cd "C:\Users\user\Documents\igem\MODEL\COPY2\SP_designer\prediction"
python3 unified_sp_prediction.py --input_path bromelain_sp.xlsx --output_path models/results.xlsx --cla_model_path models/classification_model.pth.tar --reg_model_path models/regression_model.pth.tar
```

---

## 4. Expected Outputs

### Training Outputs
- **Model files**: `.pth.tar` containing trained weights
- **Performance metrics**: `saved_values.xlsx` with accuracy, correlation, and loss values
- **Training logs**: Loss curves, validation metrics, confusion matrices

### Prediction Outputs
- **Prediction files**: Excel files with classification scores and regression values
- **Ranking analysis**: Prioritized signal peptide candidate lists
- **Statistical summaries**: Mean ranks, uncertainty measures, and selection criteria

---

## Summary

This project provides a complete signal peptide prediction pipeline:
1. **Dataset preparation** → Use Dataset Creation Guide for proper formatting
2. **Model training** → Train multiple architectures with various strategies (local PC recommended)
3. **Prediction & analysis** → Use Analysis Commands Guide for ranking and selection

The modular design supports experimentation with different architectures and training strategies to optimize signal peptide selection for protein expression tasks.