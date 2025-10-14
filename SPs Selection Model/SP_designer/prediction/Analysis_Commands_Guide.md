# Signal Peptide Analysis Commands Guide

## Overview
Commands to generate ranking summaries with error propagation from raw model predictions.

---

## 1. Generate Prediction Files

### Run Predictions on Models
Use classification and regression model paths to generate prediction results:

```bash
# CNN best models
python3 unified_sp_prediction.py --input_path bromelain_sp.xlsx --output_path models/bromelain_sp_cnn_best_models.xlsx --cla_model_path models/cnn_best_models/fine_CharCNN_best_BSP3_BSP1_CLASSIFICATION.pth.tar --reg_model_path models/cnn_best_models/CharCNN_best_SP2_BSP1_REGRESSION.pth.tar

# CNN BSP1-only 
python3 unified_sp_prediction.py --input_path bromelain_sp.xlsx --output_path models/bromelain_sp_cnn_only_bsp1.xlsx --cla_model_path models/cnn_only_bsp1/CharCNN_best_ONLY_BSP1_CLASSIFICATION.pth.tar --reg_model_path models/cnn_only_bsp1/CharCNN_best_ONLY_BSP1_REGRESSION.pth.tar

# Transformer model
python3 unified_sp_prediction.py --input_path bromelain_sp.xlsx --output_path models/bromelain_sp_transformer_sp2_to_bsp1.xlsx --cla_model_path models/transformer_sp2_trans_bsp1/fine_CharTransformer_best_SP2_BSP1_CLASSIFICATION.pth.tar --reg_model_path models/transformer_sp2_trans_bsp1/fine_CharTransformer_best_SP2_BSP1_REGRESSION.pth.tar

# LSTM model
python3 unified_sp_prediction.py --input_path bromelain_sp.xlsx --output_path models/bromelain_sp_lstm_sp2_bsp1.xlsx --cla_model_path models/lstm_sp2_bsp1/Classification_CharLSTM_best.pth.tar --reg_model_path models/lstm_sp2_bsp1/Regression_CharLSTM_best.pth.tar

# MLP model
python3 unified_sp_prediction.py --input_path bromelain_sp.xlsx --output_path models/bromelain_sp_mlp_bsp3_trans_bsp1.xlsx --cla_model_path models/mlp_bsp3_trans_bsp1/Classification_fine_CharMLP_best.pth.tar --reg_model_path models/mlp_bsp3_trans_bsp1/Regression_fine_CharMLP_best.pth.tar
```

**Produces files:** `models/bromelain_sp_[model_name].xlsx`

---

## 2. Extract Rankings from Prediction Files

### Create Individual Ranking Files
Extract rankings from each prediction file:

**Input Files:**
- `models/bromelain_sp_cnn_best_models.xlsx`
- `models/bromelain_sp_cnn_only_bsp1.xlsx` 
- `models/bromelain_sp_transformer_sp2_to_bsp1.xlsx`
- `models/bromelain_sp_lstm_sp2_bsp1.xlsx`
- `models/bromelain_sp_mlp_bsp3_trans_bsp1.xlsx`
- `models/bromelain_sp_old_models.xlsx`

**Process:** Sort by activity scores, assign ranks 1-N

**Produces files:** Individual ranking summaries per model

---

## 3. Create Combined Rankings

### Merge All Model Rankings
Combine individual rankings into unified structure:

**Target Structure:**
```
sp_name | full_sequence | cnn_rank | cnn_bsp1_rank | transformer_rank | lstm_rank | mlp_rank | old_model_rank
```

**Produces file:** `models/prediction_ranking_summary.xlsx`

---

## 4. Calculate Mean Rankings

### Method 1: Mean Rank Across Models
```python
mean_rank_i = (cnn_rank + cnn_bsp1_rank + transformer_rank + lstm_rank + mlp_rank + old_model_rank) / 6
```

### Method 2: Deviation from Mean Score
```python
# Step 1: Calculate mean rank for each model
model_means = {model: mean(all_model_ranks)}

# Step 2: Calculate deviation for each SP
deviation_score_i = mean([abs(rank_ij - model_mean_j) for j in models])
```

**Produces file:** `models/prediction_only_ranking_summary.xlsx`

---

## 5. Error Propagation and Uncertainty

### Calculate Rank Uncertainty
```python
import numpy as np

# For each signal peptide with ranks from 6 models:
ranks = [cnn_rank, cnn_bsp1_rank, transformer_rank, lstm_rank, mlp_rank, old_model_rank]
uncertainty_score = np.std(ranks) / np.sqrt(6)
```

**Interpretation:**
- uncertainty < 2.0: High consensus
- 2.0 ≤ uncertainty < 4.0: Medium consensus  
- uncertainty ≥ 4.0: Low consensus

**Produces file:** `models/prediction_only_ranking_summary_top3.xlsx`

---

## 6. Final Output Files

**File Progression:**
1. Model predictions → `models/bromelain_sp_[model].xlsx`
2. Combined rankings → `models/prediction_ranking_summary.xlsx`
3. Mean ranks + uncertainty → `models/prediction_only_ranking_summary.xlsx`
4. Top candidates → `models/prediction_only_ranking_summary_top3.xlsx`