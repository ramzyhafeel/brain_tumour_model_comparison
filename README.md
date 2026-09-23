# Comparative Analysis of Deep Learning Architectures for Multi-Class Brain Tumour MRI Classification

**Module:** SE4050 – Deep Learning 2026  
**Degree:** BSc (Hons) in Information Technology  
**Assignment:** Supervised Deep Learning Multi-Model Study  

---

## 1. Project Overview

This repository contains a comprehensive, leak-free, scientifically controlled deep learning study evaluating four distinct neural network architectures for multi-class brain tumour classification from magnetic resonance imaging (MRI) scans:

1. **Custom CNN (from scratch):** 4-block convolutional network ($32 \to 64 \to 128 \to 256$), Batch Normalization, MaxPooling, Dropout(0.40), GAP, and Dense layers (1.44M parameters).
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

All four models were trained and evaluated on identical frozen splits under standardized experimental conditions (Seed = 42, Max Epochs = 20, EarlyStopping patience = 4 monitoring `val_loss`, Adam optimizer, batch size = 16, standardized inference timing with 5 warmup and 50 timed passes on GPU):

| Model | Paradigm | Parameters | Model Size | Val Accuracy | Test Accuracy | Macro F1 | ROC-AUC | Latency (ms) | Throughput (ips) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DenseNet121** | Transfer Learning | 7,300,932 | 29.5 MB | **97.89%** | **97.25%** | **0.9720** | **0.9984** | 2.88 ms | 347.2 ips |
| **ResNet50** | Transfer Learning | 24,113,284 | 92.4 MB | 97.61% | 96.87% | 0.9680 | 0.9972 | 3.82 ms | 261.8 ips |
| **VGG16** | Transfer Learning | 14,978,116 | 57.1 MB | 97.06% | 96.11% | 0.9601 | 0.9951 | 2.45 ms | 408.2 ips |
| **Custom CNN** | From Scratch | **1,438,276** | **16.5 MB** | 96.60% | 95.35% | 0.9523 | 0.9932 | **0.93 ms** | **1,075.7 ips** |

*Master table exported to:* [`results/comparison/MASTER_COMPARISON_TABLE.csv`](file:///c:/Users/ramzy/Desktop/brain_tumour_model_comparison/results/comparison/MASTER_COMPARISON_TABLE.csv)  
*Comprehensive academic report:* [`results/comparison/FINAL_SYNTHESIS_REPORT.md`](file:///c:/Users/ramzy/Desktop/brain_tumour_model_comparison/results/comparison/FINAL_SYNTHESIS_REPORT.md)

---

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
│   ├── vgg16/vgg16_final.keras                # Trained VGG16 binary
│   ├── resnet50/resnet50_final.keras          # Trained ResNet50 binary
│   └── densenet121/densenet121_final.keras    # Trained DenseNet121 binary
├── notebooks/
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
│       ├── MASTER_COMPARISON_TABLE.csv        # Consolidated 4-architecture metrics
│       ├── FINAL_SYNTHESIS_REPORT.md          # 30% rubric critical analysis & synthesis
│       └── plots/                             # Comparative visualizations
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

## 6. Key Scientific Findings & Limitations

1. **Empirical Value of Transfer Learning:** Pretrained natural-image representations on ImageNet transferred effectively to axial MRI, improving accuracy from 95.35% (scratch CNN) to 97.25% (DenseNet121) and reducing false negatives.
2. **Architecture Efficiency Frontier:** DenseNet121 outperformed ResNet50 in predictive accuracy (97.25% vs 96.87%) with **less than one-third of the parameters** (7.30M vs 24.11M) due to dense feature reuse.
3. **Preserved Study Limitations:**
   - *Patient-level independence:* Reliable patient identifiers were not published with the Kaggle dataset; patient-level independence between splits cannot be proven.
   - *Single-plane 2D slices:* 2D axial slice classification excludes multi-planar 3D volumetric context.
   - *Academic artifact:* High benchmark accuracy does not imply clinical readiness across unseen hospital scanners, magnet field strengths, or acquisition protocols.
