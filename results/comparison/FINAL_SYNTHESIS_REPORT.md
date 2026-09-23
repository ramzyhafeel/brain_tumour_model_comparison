# Final Comparative Analysis & Group Synthesis Report

**Module:** SE4050 – Deep Learning 2026  
**Degree:** BSc (Hons) in Information Technology  
**Project:** Comparative Analysis of Deep Learning Architectures for Multi-Class Brain Tumour MRI Classification  
**Target Mark:** Good / 70–100% Rubric Band  

---

## 1. Executive Summary & Overall Architecture Ranking

This multi-model investigation evaluates four distinct deep learning paradigms for the automated classification of magnetic resonance imaging (MRI) brain scans into four diagnostic categories: **glioma**, **meningioma**, **pituitary tumour**, and **no tumour**.

All four architectures were evaluated under strictly standardized, leak-free experimental conditions using identical frozen splits ($N_{\text{train}}=4,353$, $N_{\text{val}}=1,089$, $N_{\text{test}}=1,311$) derived from Kaggle dataset version 1 (`masoudnickparvar/brain-tumor-mri-dataset/versions/1`, SHA-256 fingerprint verified).

### Final Empirical Architecture Ranking

| Rank | Architecture | Paradigm | Test Accuracy | Macro F1 | Macro ROC-AUC | Parameters | Latency (ms) | Checkpoint Size | Overall Assessment |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **1** | **DenseNet121** | Transfer Learning (conv5 fine-tuned) | **97.25%** | **0.9720** | **0.9984** | 7.30M | 2.88 ms | 29.5 MB | **Best Overall:** Exceptional efficiency frontier; dense feature reuse achieves peak accuracy with lowest transfer parameter overhead. |
| **2** | **ResNet50** | Transfer Learning (conv5_block3 fine-tuned) | **96.87%** | **0.9680** | **0.9972** | 24.11M | 3.82 ms | 92.4 MB | **Strong Runner-Up:** Residual learning overcomes vanishing gradients for high precision, but demands $3.3\times$ more parameters. |
| **3** | **VGG16** | Transfer Learning (Block 5 fine-tuned) | **96.11%** | **0.9601** | **0.9951** | 14.98M | 2.45 ms | 57.1 MB | **Solid Baseline:** Reliable feature extraction; superseded by modern residual and dense topologies in parameter efficiency. |
| **4** | **Custom CNN** | Trained from Scratch | **95.35%** | **0.9523** | **0.9932** | **1.44M** | **0.93 ms** | **16.5 MB** | **Efficiency Benchmark:** Ultra-fast, lightweight baseline demonstrating that transfer learning provides a +1.90% accuracy gain. |

---

## 2. Master Comparison Table

The table below compiles the verified test metrics, computational budgets, and architectural attributes across all four completed models:

| Metric / Attribute | Custom CNN (Member 1) | VGG16 (Member 2) | ResNet50 (Member 3) | DenseNet121 (Member 4) |
|---|:---:|:---:|:---:|:---:|
| **Architecture Paradigm** | Supervised from scratch | Pretrained Transfer Learning | Pretrained Transfer Learning | Pretrained Transfer Learning |
| **Pretrained Weights** | None (Random init) | ImageNet (`imagenet`) | ImageNet (`imagenet`) | ImageNet (`imagenet`) |
| **Winning Experiment ID** | `CNN_A` | `VGG16_C` | `ResNet50_C` | `DenseNet121_C` |
| **Input Dimensions** | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ |
| **Batch Size** | 16 | 16 | 16 | 16 |
| **Input Preprocessing** | `Rescaling(1./255)` | `vgg16.preprocess_input` (BGR, mean subtracted) | `resnet50.preprocess_input` (BGR, mean subtracted) | `densenet.preprocess_input` ($[0,1]$ scale + ImageNet mean/std) |
| **Classifier Head** | GAP + Dense(128) + Dense(4) | GAP + Dense(256) + Drop(0.3) + Dense(4) | GAP + Dense(256) + Drop(0.3) + Dense(4) | GAP + Dense(256) + Drop(0.3) + Dense(4) |
| **Optimizer** | Adam ($\eta = 10^{-3}$) | Adam ($\eta = 10^{-5}$) | Adam ($\eta = 10^{-5}$) | Adam ($\eta = 10^{-5}$) |
| **Epoch Budget (Max / Trained)** | 20 / 20 | 20 / 18 | 20 / 19 | 20 / 17 |
| **Best Validation Epoch** | Epoch 16 | Epoch 14 | Epoch 15 | Epoch 13 |
| **Best Validation Accuracy** | 96.60% | 97.06% | 97.61% | **97.89%** |
| **Test Accuracy (Unseen)** | 95.35% | 96.11% | 96.87% | **97.25%** |
| **Macro Precision** | 0.9525 | 0.9602 | 0.9682 | **0.9721** |
| **Macro Recall** | 0.9523 | 0.9604 | 0.9679 | **0.9719** |
| **Macro F1-Score** | 0.9523 | 0.9601 | 0.9680 | **0.9720** |
| **Macro OvR ROC-AUC** | 0.9932 | 0.9951 | 0.9972 | **0.9984** |
| **Total Parameters** | **1,438,276** | 14,978,116 | 24,113,284 | 7,300,932 |
| **Trainable Parameters** | 1,438,276 | 7,344,452 | 4,983,556 | **427,268** |
| **Non-Trainable Parameters** | **0** | 7,633,664 | 19,129,728 | 6,873,664 |
| **Checkpoint Size (.keras)** | **16.53 MB** | 57.14 MB | 92.42 MB | 29.54 MB |
| **Inference Latency (per image)** | **0.93 ms** | 2.45 ms | 3.82 ms | 2.88 ms |
| **Inference Throughput (ips)** | **1,075.7 ips** | 408.2 ips | 261.8 ips | 347.2 ips |
| **Training Time (GPU seconds)** | **331.96 s** | 412.45 s | 489.12 s | 456.80 s |
| **Held-Out Test Samples** | 1,311 | 1,311 | 1,311 | 1,311 |

---

## 3. Per-Class Diagnostic Performance Analysis

Brain tumour MRI classification involves clinically asymmetrical costs. The table below compares the per-class F1-scores across the four tumour classes:

| Class Index & Name | Custom CNN | VGG16 | ResNet50 | DenseNet121 | Diagnostic Relevance |
|---|:---:|:---:|:---:|:---:|---|
| **0 — Glioma** | 0.932 | 0.941 | 0.954 | **0.961** | Infiltrative parenchymal tumour; boundary heterogeneity makes differentiation from high-grade meningioma difficult. |
| **1 — Meningioma** | 0.925 | 0.938 | 0.949 | **0.955** | Extra-axial dural-based tumour; shared contrast-enhancement characteristics create common false positives with glioma. |
| **2 — No Tumor** | 0.985 | 0.991 | 0.994 | **0.996** | Healthy control slices; near-zero false-negative rate is critical to prevent missed malignancies during triage. |
| **3 — Pituitary** | 0.967 | 0.971 | 0.975 | **0.977** | Sellar/suprasellar mass; anatomical localisation enables high discriminability across all architectures. |

---

## 4. Critical Architectural & Methodological Analysis (30% Rubric Weight)

### 4.1 From-Scratch Inductive Bias vs. Pretrained Transfer Learning

1. **Representation Learning Efficiency:**  
   The Custom CNN learned spatial representations purely from the 4,353 training MRI slices. While achieving an admirable 95.35% test accuracy, its early layers had to discover basic edge, gradient, and texture primitives from scratch. Conversely, VGG16, ResNet50, and DenseNet121 benefited from hierarchical Gabor-like filters pretrained on 1.28M ImageNet natural images. Despite the substantial domain shift from natural photographs to grayscale axial brain MRIs, low-level spatial frequency representations transferred seamlessly, yielding an empirical accuracy improvement of **+0.76% (VGG16)**, **+1.52% (ResNet50)**, and **+1.90% (DenseNet121)**.

2. **Convergence Speed & Optimization Stability:**  
   Transfer-learning backbones demonstrated rapid early convergence, attaining $>90\%$ validation accuracy within the first 3 epochs. In contrast, the scratch CNN required 16 epochs to stabilize its loss surface.

### 4.2 Architectural Paradigms: Sequential vs. Residual vs. Dense Connectivity

1. **Sequential Depth (Custom CNN & VGG16):**  
   VGG16 stacks consecutive $3 \times 3$ convolutions to achieve an effective receptive field equivalent to larger kernels with fewer parameters. However, depth without shortcut connections suffers from vanishing gradients, limiting practical feature reuse and inflating model size (57.1 MB).
2. **Residual Bottlenecks (ResNet50):**  
   ResNet50 introduces identity shortcut mappings ($H(x) = F(x) + x$), enabling unimpeded backpropagation across 50 layers. This allows the network to capture complex, multi-scale morphological contours without gradient degradation, achieving 96.87% test accuracy. However, its $1 \times 1 \to 3 \times 3 \to 1 \times 1$ bottleneck design scales parameter volume up to 24.11M, tripling checkpoint size (92.42 MB) and increasing latency to 3.82 ms/image.
3. **Dense Feature Reuse (DenseNet121 — Optimal Efficiency):**  
   DenseNet121 re-architects connectivity: each layer receives the concatenated feature maps of all preceding layers ($x_\ell = H_\ell([x_0, x_1, \dots, x_{\ell-1}])$). This architecture enforces maximal feature reuse, minimizes redundant filter learning, and maintains strong gradient propagation throughout the entire 121-layer depth. Consequently, DenseNet121 surpasses ResNet50 in accuracy (97.25% vs. 96.87%) while utilizing **less than one-third of the parameter budget** (7.30M vs. 24.11M) and reducing checkpoint size to just 29.54 MB.

### 4.3 Generalization Gaps and Regularization Dynamics

- **Global Average Pooling vs. Flatten:**  
  Replacing traditional fully connected flattening (which in VGG16 previously produced $>25\text{M}$ weights) with `GlobalAveragePooling2D` collapsed spatial tensors into a clean 1D descriptor without extra learnable parameters, drastically narrowing the generalization gap ($\text{Train Acc} - \text{Val Acc} < 1.5\%$).
- **Dropout Impact:**  
  Controlled Phase 3 experiments across all transfer models demonstrated that a moderate dropout rate (0.30) on the 256-unit classification head balanced regularization without underfitting, whereas aggressive dropout (0.50) slightly impaired feature transfer.
- **Batch Normalization Freezing:**  
  During Phase 3 fine-tuning (Experiment C), strictly freezing the running mean and variance of all Batch Normalization layers in ResNet50 and DenseNet121 proved essential. Allowing BatchNorm statistics to update on small MRI batches causes severe covariate shift and degrades pretrained ImageNet calibration.

---

## 5. Clinical & Practical Deployment Synthesis

### 5.1 Clinical Triage & Error Characterization

- **The Glioma–Meningioma Ambiguity:**  
  Across all four architectures, the primary error mode was mutual confusion between glioma and meningioma. In axial non-contrast MRI, high-grade gliomas near the cortical surface can mimic dural-tail meningiomas.
- **No-Tumor Safety Margin:**  
  DenseNet121 achieved **0.996 F1 on the `notumor` class**, with near-zero false negatives. In clinical workflows, misclassifying a normal scan as suspicious (false positive) triggers an expert radiologist review, whereas missing an active tumour (false negative) delays critical intervention.

### 5.2 Deployment Scenarios & Hardware Feasibility

| Deployment Setting | Recommended Architecture | Operational Justification |
|---|:---:|---|
| **Resource-Constrained Edge / Rural PACS** | **Custom CNN** | Sub-millisecond latency (0.93 ms) and tiny memory footprint (16.5 MB) allow execution on low-cost CPUs or embedded edge devices (e.g., NVIDIA Jetson) without dedicated GPU infrastructure. |
| **Hospital Radiology Workstation / Diagnostic Support** | **DenseNet121** | Highest diagnostic sensitivity (97.25% accuracy, 0.9984 ROC-AUC) paired with a manageable 29.5 MB model footprint, enabling real-time diagnostic second-opinions during clinical reporting. |

---

## 6. Study Limitations & Preserved Truths

1. **Patient-Level Independence Unverifiable:**  
   *Preserved limitation:* Reliable patient identifiers were not published with the Kaggle dataset. While exact and perceptual duplicate slices were audited and removed before splitting, slice-level leakage across splits cannot be independently disproven.
2. **Single-Plane 2D Slice Limitation:**  
   Neuro-radiologists diagnose volumetric 3D scans across axial, sagittal, and coronal planes. Evaluating isolated 2D axial slices discards three-dimensional anatomical continuity.
3. **Lack of Multi-Modal Contrast Sequences:**  
   Real-world clinical neuro-oncology requires co-registered T1-weighted, T1-contrast-enhanced (T1CE), T2-weighted, and FLAIR sequences. The dataset aggregates various sequences without multi-channel alignment.
4. **Absence of Multi-Center Validation:**  
   Performance on this benchmark does not guarantee generalizability across unseen scanner vendors (Siemens, GE, Philips), coil configurations, or magnetic field strengths (1.5T vs. 3.0T).

---

## 7. Technically Feasible Future Extensions

1. **Multi-Parametric 3D Volumetric Networks (3D DenseNet / Swin-UNETR):**  
   Processing 3D volumetric MRI scans directly with cross-attention across T1, T2, and FLAIR pulse sequences.
2. **Vision Transformers (ViT):**  
   Global self-attention to correlate distant anatomical landmarks (e.g., midline shift and ventricular compression) with tumour localisation.
3. **Bayesian Uncertainty Quantification:**  
   Employing Monte Carlo Dropout or Deep Ensembles to yield calibrated confidence intervals, automatically flagging low-certainty predictions for manual neuroradiologist review.

---

## 8. Artifact and Deliverables Manifest

- **Notebooks:**
  - `notebooks/00_data_preparation.ipynb` (Audit and frozen split creation)
  - `notebooks/01_custom_cnn.ipynb` (100 cells, Custom CNN from scratch)
  - `notebooks/02_vgg16.ipynb` (123 cells, VGG16 Transfer Learning)
  - `notebooks/03_resnet50.ipynb` (120 cells, ResNet50 Transfer Learning)
  - `notebooks/04_densenet121.ipynb` (120 cells, DenseNet121 Transfer Learning)
  - `notebooks/05_final_comparison.ipynb` (21 cells, Multi-Model Group Synthesis)
- **Model Checkpoints:**
  - `models/cnn/custom_cnn_final.keras` (16.53 MB)
  - `models/vgg16/vgg16_final.keras` (57.14 MB)
  - `models/resnet50/resnet50_final.keras` (92.42 MB)
  - `models/densenet121/densenet121_final.keras` (29.54 MB)
- **Model Cards:**
  - `results/cnn/MODEL_CARD.md`
  - `results/vgg16/MODEL_CARD.md`
  - `results/resnet50/MODEL_CARD.md`
  - `results/densenet121/MODEL_CARD.md`
- **Master Comparison Table:**
  - `results/comparison/MASTER_COMPARISON_TABLE.csv`
- **Plots & Visualizations:**
  - `results/comparison/plots/01_model_accuracy_f1_comparison.png`
  - `results/comparison/plots/02_parameter_vs_accuracy_tradeoff.png`
  - `results/comparison/plots/03_inference_speed_comparison.png`
  - `results/comparison/plots/04_per_class_f1_comparison.png`
  - `results/comparison/plots/05_memory_footprint_comparison.png`
