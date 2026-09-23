# ResNet50 Model Card

## Project

Comparative Analysis of Deep Learning Architectures for Multi-Class Brain Tumour MRI Classification  
**SE4050 – Deep Learning 2026**

---

## Model Overview

- **Architecture:** ResNet50 (Residual Network, 50-layer deep convolutional neural network)
- **Paradigm:** Transfer Learning with Pretrained ImageNet Weights
- **Backbone Source:** `tensorflow.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))`
- **Model Role:** Member 3 of the 4-model comparative study (Custom CNN, VGG16, ResNet50, DenseNet121).
- **Selection Basis:** The winning configuration was selected using training and validation data only (`val_loss` primary, `val_macro_f1` secondary). The held-out test set was evaluated only after the configuration and checkpoint were permanently frozen.

---

## Dataset & Frozen Splits

- **Dataset:** Masoud Nickparvar — Brain Tumor MRI Dataset
- **Pinned Kaggle Handle:** `masoudnickparvar/brain-tumor-mri-dataset/versions/1`
- **Dataset Fingerprint:** `061cbffb7341abf85e57a4f99de571e8fa4702ec8236c929e35543ab0547217d`
- **Classes (4):**
  1. `glioma` (Index 0)
  2. `meningioma` (Index 1)
  3. `notumor` (Index 2)
  4. `pituitary` (Index 3)
- **Frozen Split Counts:**
  - **Training:** 4,353 images (`splits/train.csv`)
  - **Validation:** 1,089 images (`splits/val.csv`)
  - **Held-Out Test:** 1,311 images (`splits/test.csv`)
- **Data Integrity Audit:**
  - 191 exact duplicate copies in original Training were excluded by Member 1.
  - 79 Training images matching original Testing were excluded to prevent train-test contamination.
  - 0 corrupt images detected.

---

## Preprocessing & Data Pipeline

- **Image Dimensions:** $224 \times 224 \times 3$
- **Batch Size:** 16
- **Architecture-Specific Preprocessing:**
  - Uses `keras.applications.resnet50.preprocess_input`.
  - Raw pixel values are decoded in $[0, 255]$ float format.
  - Converts channels from RGB to BGR and zero-centers each channel by subtracting ImageNet means: $[103.939, 116.779, 123.68]$.
  - **Critical Design Choice:** `Rescaling(1./255)` is **NOT** applied before `preprocess_input` to avoid improper double normalization.
- **Training-Only Data Augmentation:**
  - Random Rotation: factor $\approx 0.03$ ($\approx \pm 10.8^\circ$)
  - Random Zoom: $\pm 8\%$
  - Random Translation: $5\%$ height/width
  - No augmentation applied during validation or testing.

---

## Classifier Head Architecture

To adapt the pretrained ResNet50 features to the 4-class brain tumour classification task, a lightweight classification head was attached:
1. `GlobalAveragePooling2D()`: Spatial pooling reducing $(7 \times 7 \times 2048)$ feature maps to 2,048 units (avoiding the 100,352-parameter explosion of `Flatten`).
2. `Dense(256, activation='relu')`: Task-specific representation layer adapting 2,048 residual features down to 256 non-linear activations.
3. `Dropout(rate=0.30 or 0.50)`: Regularization to combat co-adaptation.
4. `Dense(4, activation='softmax')`: Normalized multi-class probability output.

---

## Training Configuration & Protocol

- **Optimizer:** Adam
- **Loss Function:** `SparseCategoricalCrossentropy()`
- **Head Training Learning Rate:** $\eta = 10^{-3}$ ($0.001$)
- **Fine-Tuning Learning Rate:** $\eta = 10^{-5}$ ($0.00001$)
- **Maximum Epoch Budget:** 20 epochs
- **Early Stopping:** Monitored `val_loss`, patience = 4 epochs, `restore_best_weights = True`
- **Random Seed:** 42 across Python, NumPy, and TensorFlow
- **Determinism:** TensorFlow op determinism enabled

---

## Controlled Experiments Summary (Validation Only)

Three controlled experiments were conducted on the frozen training/validation split:

1. **ResNet50-A (Baseline Frozen):**
   - Backbone frozen, Dropout 0.30, Adam $\eta = 10^{-3}$.
   - Evaluated baseline transfer-learning representation stability.
2. **ResNet50-B (Controlled Regularization):**
   - Backbone frozen, Dropout 0.50, Adam $\eta = 10^{-3}$.
   - Tested whether stronger dropout reduces the generalization gap.
3. **ResNet50-C (Controlled Fine-Tuning):**
   - Final residual block `conv5_block3` conv layers unfrozen (`conv5_block3_1_conv`, `conv5_block3_2_conv`, `conv5_block3_3_conv`).
   - Stages 1–4, `conv5_block1`/`conv5_block2`, and all BatchNorm layers kept frozen.
   - Recompiled with small learning rate $\eta = 10^{-5}$ to refine high-level MRI features without destroying low-level representations.

The winning configuration was automatically selected via minimum validation loss and preserved in `results/resnet50/phase3_experiments/selected_resnet50_best.keras`.

---

## Performance & Complexity Summary

- **Total Parameters:** ~24.1M (24,113,284)
- **Trainable Parameters:** 525,572 (Frozen) / ~4.98M (Fine-Tuned `conv5_block3`)
- **Model Checkpoint Size:** ~92.4 MB
- **Inference Timing Protocol:** Benchmarked on GPU using 16-image decoded batch, 5 warmup passes, 50 timed passes with `.numpy()` synchronization.
- **Detailed Metrics:** Persisted in `results/resnet50/phase4_final_test/resnet50_final_metrics.json` and `resnet50_final_metrics.csv`.

---

## Important Scientific & Practical Limitations

1. **Pretraining Asymmetry:** ResNet50 initializes from ImageNet weights, whereas the Custom CNN was trained from scratch. The overall comparison is a study of practical engineering methodologies (pretrained transfer learning vs. custom training), not an isolated kernel architecture benchmark.
2. **Batch Normalization Freezing:** In fine-tuning, keeping BatchNorm layers frozen is essential to maintain ImageNet mean/variance statistics and avoid optimization instability on small datasets.
3. **Patient-Level Independence:** Patient identifiers were not available in the public Kaggle dataset. Consequently, patient-level independence between training, validation, and test splits could not be independently proven.
4. **Clinical Non-Equivalence:** The model is an academic research artifact. High validation and test accuracy on this benchmark do not imply clinical readiness, diagnostic reliability, or cross-institutional generalization across unseen hospital scanners and protocols.

---

## Reproducibility & Artifact Manifest

- **Pinned Seed:** 42
- **Final Model File:** `models/resnet50/resnet50_final.keras`
- **Standardized Comparison Record:** `results/resnet50/phase5_finalization/resnet50_comparison_row.csv`
- **Full Reproducibility Manifest:** `results/resnet50/phase5_finalization/resnet50_reproducibility_config.json`
- **Artifact Hashes:** `results/resnet50/phase5_finalization/artifact_checksums.json`
