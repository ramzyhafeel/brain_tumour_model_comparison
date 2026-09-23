# Comparative Analysis of Deep Learning Architectures for Multi-Class Brain Tumour MRI Classification

**Module:** SE4050 – Deep Learning 2026  
**Degree:** BSc (Hons) in Information Technology  
**Assignment:** Supervised Deep Learning Multi-Model Study  

---

## 1. Project Overview

This repository contains a comprehensive, leak-free, scientifically controlled deep learning study evaluating four distinct neural network architectures for multi-class brain tumour classification from magnetic resonance imaging (MRI) scans:

1. **Custom CNN (from scratch):** 4-block convolutional network (32 → 64 → 128 → 256), MaxPooling, Dense(256), Dropout, and 4-class softmax.
2. **VGG16 (Transfer Learning):** Deep sequential convolutional feature extractor with Global Average Pooling and a dense classifier head (14.98M parameters).
3. **ResNet50 (Transfer Learning):** Deep residual network utilizing bottleneck residual blocks and identity shortcut connections to mitigate vanishing gradients (24.11M parameters).
4. **DenseNet121 (Transfer Learning):** Densely connected convolutional network featuring dense feature concatenation blocks and transition layers for maximal feature reuse (7.30M parameters).

### Diagnostic Target Classes (4)
- `0: glioma`
- `1: meningioma`
- `2: notumor`
- `3: pituitary`

---

## 2. Master Comparison Results

Final four-model results are intentionally **not hard-coded** in the repository. Run `01_custom_cnn.ipynb`, `02_vgg16.ipynb`, `03_resnet50.ipynb`, and `04_densenet121.ipynb` to generate measured comparison rows, then run `05_final_comparison.ipynb` to create the master table and plots.

## 3. Dataset & Frozen Split Specifications

- **Dataset:** Masoud Nickparvar — Brain Tumor MRI Dataset
- **Pinned Kaggle Handle:** `masoudnickparvar/brain-tumor-mri-dataset/versions/1`
- **Dataset Fingerprint:** `061cbffb7341abf85e57a4f99de571e8fa4702ec8236c929e35543ab0547217d`
- **Data Integrity Audit:**
  - 191 duplicate copies in Training excluded.
  - 79 Training images matching Testing excluded to strictly prevent train-test leakage.
  - 0 corrupt images.
- **Frozen Split Counts & Checksums (`splits/`):**
  - **Train:** 4,353 images (`train.csv` — `7273ef5fc2cb605c5b03b22a23cda0cb383ac0d9ce5949ac3c95649b5a4270cb`)
  - **Validation:** 1,089 images (`val.csv` — `37b24456cfcd1b69df939e36603958eae6a9124b794c32c83a532b9018c6c88f`)
  - **Held-Out Test:** 1,311 images (`test.csv` — `9afc40a38949eb4f46f8c9591d3b1f51cdd11aa991bfe2cf1d15c386c5537d39`)

---

## 4. Repository Structure

```
brain-tumour-model-comparison/
├── README.md                                  # Repository overview and instructions
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Git tracking exclusions
├── config/
│   ├── class_to_index.json                    # Label-to-index mapping (0 to 3)
│   ├── environment.json                       # Hardware/software runtime manifest
│   ├── phase1_config.json                     # Shared pipeline configurations
│   └── shared_protocol.json                   # Experimental protocol contract
├── docs/
│   └── SHARED_DATA_HANDOFF.md                 # Data handoff and leakage prevention doc
├── models/
│   ├── README.md                              # Model storage documentation
│   ├── cnn/custom_cnn_final.keras             # Trained Custom CNN binary
│   ├── vgg16/vgg16_final.keras                # Generated after VGG16 notebook completes
│   ├── resnet50/resnet50_final.keras          # Generated after ResNet50 notebook completes
│   └── densenet121/densenet121_final.keras    # Generated after DenseNet121 notebook completes
├── notebooks/
│   ├── 00_data_preparation.ipynb
│   ├── 01_custom_cnn.ipynb                    # Member 1: Custom CNN from scratch
│   ├── 02_vgg16.ipynb                         # Member 2: VGG16 Transfer Learning
│   ├── 03_resnet50.ipynb                      # Member 3: ResNet50 Transfer Learning
│   ├── 04_densenet121.ipynb                   # Member 4: DenseNet121 Transfer Learning
│   └── 05_final_comparison.ipynb             # Multi-Model Group Synthesis Notebook
├── results/
│   ├── data_audit/                            # Dataset EDA, class balance, duplicate audit
│   ├── cnn/                                   # Member 1 metrics, curves, MODEL_CARD.md
│   ├── vgg16/                                 # Member 2 metrics, curves, MODEL_CARD.md
│   ├── resnet50/                              # Member 3 metrics, curves, MODEL_CARD.md
│   ├── densenet121/                           # Member 4 metrics, curves, MODEL_CARD.md
│   └── comparison/
│       ├── MASTER_COMPARISON_TABLE.csv        # Generated after all four measured rows exist
│       ├── FINAL_SYNTHESIS_REPORT.md          # Generated from measured rows only
│       └── plots/                             # Generated comparative visualizations
└── splits/
    ├── train.csv                              # Frozen training split manifest (4,353 images)
    ├── val.csv                                # Frozen validation split manifest (1,089 images)
    ├── test.csv                               # Frozen held-out test split manifest (1,311 images)
    └── split_checksums.json                   # SHA-256 integrity hashes
```

---

## 5. Execution Instructions (Google Colab / Local GPU)

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd brain_tumour_model_comparison
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Notebooks in Sequence:**
   - Execute `notebooks/01_custom_cnn.ipynb` to train/evaluate Custom CNN.
   - Execute `notebooks/02_vgg16.ipynb` to train/evaluate VGG16.
   - Execute `notebooks/03_resnet50.ipynb` to train/evaluate ResNet50.
   - Execute `notebooks/04_densenet121.ipynb` to train/evaluate DenseNet121.
   - Execute `notebooks/05_final_comparison.ipynb` to generate the master comparison table and synthesis plots.

---

## 6. Scientific Interpretation & Limitations

Do not state comparative findings until all four model notebooks have been executed and `05_final_comparison.ipynb` has generated the measured master table. The repository intentionally avoids pre-populating transfer-learning performance values before execution.

Preserved limitations:

1. **Pretraining asymmetry:** the Custom CNN is trained from scratch, while VGG16, ResNet50 and DenseNet121 use ImageNet weights. This is a practical training-approach comparison, not a perfectly isolated architecture-only experiment.
2. **Patient-level independence:** reliable patient identifiers are not available, so patient-level separation cannot be independently verified.
3. **2D image limitation:** the task uses individual 2D MRI images and does not model full 3D volumetric context.
4. **Academic artifact:** benchmark performance does not establish clinical readiness or cross-hospital generalization.
