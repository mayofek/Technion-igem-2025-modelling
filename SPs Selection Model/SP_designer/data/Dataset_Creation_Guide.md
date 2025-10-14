# Dataset Creation Guide for Signal Peptide Models

## Overview
This guide describes how to create properly formatted datasets for training CNN, Transformer, LSTM, and MLP models for signal peptide prediction. The models perform both classification (high/low secretion) and regression (quantitative activity prediction).

---

## 1. Required Dataset Structure

### Standard Column Format
All datasets must contain exactly 7 columns in this order:

| Column Name | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| `seq` | String | Full amino acid sequence (signal peptide + protein) | `MKRWKTVCMLCFAFLLLAAMDSNGNQEING...` |
| `sp_seq` | String | Signal peptide sequence only | `MKRWKTVCMLCFAFLLLAA` |
| `pro_seq` | String | Protein sequence only (without signal peptide) | `MDSNGNQEINGKEKLSVNDSKLK...` |
| `source_protein` | String | Source organism or protein name | `Chryseobacterium proteolyticum` |
| `experimental yield` | Float | Raw experimental secretion yield value | `0.0`, `2.5`, `1.8` |
| `target` | Float | Normalized target value for regression | `-1.017`, `0.543`, `2.1` |
| `label` | String | Classification label (Chinese characters) | `低` (low) or `高` (high) |

---

## 2. Data Requirements and Constraints

### Sequence Requirements
- **Full sequence (`seq`)**: Must equal `sp_seq + pro_seq`
- **Signal peptide (`sp_seq`)**: Typically 15-35 amino acids, starts with methionine (M)
- **Protein sequence (`pro_seq`)**: Target protein without signal peptide
- **Valid amino acids only**: Standard 20 amino acids (ACDEFGHIKLMNPQRSTVWY)

### Target Value Normalization
- **`experimental yield`**: Raw laboratory measurement values
- **`target`**: Normalized values using z-score normalization
- **Formula**: `target = (experimental_yield - mean) / std`
- **Target should have**: Mean ≈ 0, Standard deviation ≈ 1

### Classification Labels
- **`低`** (dī): Low secretion activity (typically target < threshold)
- **`高`** (gāo): High secretion activity (typically target ≥ threshold)
- **Threshold**: Usually determined by median or domain knowledge

---

## 3. Dataset Sources and Types

### Available Datasets

#### SP Datasets (B. amyloliquefaciens)
- **SP1.xlsx**: 151 experimentally validated signal peptides tested for AmyQ (α-amylase) secretion in B. amyloliquefaciens - controlled benchmark for single-protein secretion patterns
- **SP2.xlsx**: 783 cleaned entries from SPSED database and Zhang et al. (2016), spanning 15 different proteins across 10 microbial species, all secreted in B. amyloliquefaciens

#### BSP Datasets (B. subtilis focus)
- **BSP1.xlsx**: 148 signal peptides tested with cutinase in B. subtilis from Brockmeier et al. (2006) - corresponds to Takara kit sequences
- **BSP2.xlsx**: 237 cutinase secretion entries across B. subtilis TEB1030 and Corynebacterium glutamicum from SPSED database
- **BSP3.xlsx**: 638 entries testing 12 different proteins across 8 genetically distinct B. subtilis strains from SPSED database

#### Combined Datasets
- **SP1_and_SP2.xlsx**: Merged source datasets for broader training

### Dataset Summary Table
| Dataset | Host Organism(s) | # Entries | # Secreted Proteins | # Host Strains | Data Type |
|---------|------------------|-----------|-------------------|----------------|-----------|
| SP1 | B. amyloliquefaciens | 151 | 1 (AmyQ) | 1 | Experimental |
| SP2 | B. amyloliquefaciens | 783 | 15 | 1 | Literature-derived |
| BSP1 | B. subtilis | 148 | 1 (Cutinase) | 1 | Experimental |
| BSP2 | B. subtilis, C. glutamicum | 237 | 1 (Cutinase) | 2 | SPSED (filtered) |
| BSP3 | 8 strains of B. subtilis | 638 | 12 | 8 | SPSED (curated) |

---

## 4. Data Quality Requirements

### Sequence Requirements
- **Full sequence (`seq`)**: Must equal `sp_seq + pro_seq`
- **Signal peptide (`sp_seq`)**: Typically 15-35 amino acids, starts with methionine (M)
- **Protein sequence (`pro_seq`)**: Target protein without signal peptide
- **Valid amino acids only**: Standard 20 amino acids (ACDEFGHIKLMNPQRSTVWY)

### Target Value Normalization
- **`experimental yield`**: Raw laboratory measurement values
- **`target`**: Normalized values using z-score normalization
- **Formula**: `target = (experimental_yield - mean) / std`
- **Target should have**: Mean ≈ 0, Standard deviation ≈ 1

### Classification Labels
- **`低`** (dī): Low secretion activity (typically target < threshold)
- **`高`** (gāo): High secretion activity (typically target ≥ threshold)
- **Threshold**: Usually determined by median or domain knowledge

---

## Summary

Proper dataset creation is critical for model performance. Key requirements:
- **7-column structure** with consistent naming
- **Normalized targets** with mean≈0, std≈1  
- **Binary classification** using Chinese labels (低/高)
- **Sequence consistency** between full and component sequences

Follow this guide to ensure datasets are compatible with the existing model training pipeline and achieve optimal prediction performance.