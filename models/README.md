# Trained Models Storage & Manifest

Large trained model checkpoints (`*.keras`, `*.h5`) are excluded from standard Git tracking via `.gitignore` to prevent repository bloat.

## Final Model Checkpoints

### 1. Member 1 — Custom CNN
- **Canonical Path:** `models/cnn/custom_cnn_final.keras`
- **Selection Basis:** Selected strictly based on validation performance (minimum validation loss across experiments CNN_A, CNN_B, CNN_C) before held-out test evaluation.
- **Architecture:** 4-block Conv2D stack ($32 \to 64 \to 128 \to 256$) trained from scratch.

### 2. Member 2 — VGG16 Transfer Learning
- **Canonical Path:** `models/vgg16/vgg16_final.keras`
- **Selection Basis:** Selected based on minimum validation loss among controlled transfer-learning configurations (VGG16-A, VGG16-B, VGG16-C) prior to held-out test evaluation.
- **Architecture:** Pretrained VGG16 backbone + GAP + Dense(256) + Dropout + Dense(4, softmax).

### 3. Member 3 — ResNet50 Transfer Learning
- **Canonical Path:** `models/resnet50/resnet50_final.keras`
- **Selection Basis:** Selected based on minimum validation loss among controlled transfer-learning configurations (ResNet50-A, ResNet50-B, ResNet50-C) prior to held-out test evaluation.
- **Architecture:** Pretrained ResNet50 backbone + GAP + Dense(256) + Dropout + Dense(4, softmax).

### 4. Member 4 — DenseNet121 Transfer Learning
- **Canonical Path:** `models/densenet121/densenet121_final.keras`
- **Selection Basis:** Selected based on minimum validation loss among controlled transfer-learning configurations (DenseNet121-A, DenseNet121-B, DenseNet121-C) prior to held-out test evaluation.
- **Architecture:** Pretrained DenseNet121 backbone + GAP + Dense(256) + Dropout + Dense(4, softmax).

---

## Intermediate & Development Checkpoints

Development checkpoints generated during early-stopping phases (e.g. `checkpoints/` in `phase3_experiments/` or `phase2_smoke/`) are temporary artifacts and do not need to be committed to version control.

If trained model binaries are archived on an external drive or cloud storage (e.g., Google Drive, Kaggle, HuggingFace), document their checksums in `results/<model>/phase5_finalization/artifact_checksums.json`.
