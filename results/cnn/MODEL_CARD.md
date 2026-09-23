
# Custom CNN Model Card

## Project

Comparative Analysis of Deep Learning Architectures for Multi-Class Brain Tumour MRI Classification

## Model

Custom Convolutional Neural Network trained from scratch.

Selected validation experiment:

**CNN_A**

The final model configuration was selected using training and validation data only. The held-out test dataset was evaluated only after the configuration was frozen.

## Dataset

Dataset:
Masoud Nickparvar — Brain Tumor MRI Dataset

Pinned Kaggle handle:

`masoudnickparvar/brain-tumor-mri-dataset/versions/1`

Classes:

1. glioma
2. meningioma
3. notumor
4. pituitary

Frozen dataset sizes:

- Training: 4,353 images
- Validation: 1,089 images
- Test: 1,311 images

The original Testing directory was preserved as the final held-out test set.

During the dataset audit:

- 191 exact duplicate copies inside the original Training data were excluded.
- 79 Training images with exact byte-identical matches in Testing were excluded from the Training candidate pool.
- No corrupt image files were detected.

Patient-level independence could not be independently verified because reliable patient identifiers were not available.

## Input

Image size:

`224 × 224 × 3`

Batch size:

`16`

## Preprocessing

The Custom CNN uses:

`Rescaling(1./255)`

Pixel normalization is performed inside the model.

Training-only augmentation:

- rotation factor: 0.03
- zoom: approximately ±0.08
- translation: 0.05

No random augmentation is applied during validation or test evaluation.

## Architecture

Core convolutional feature extractor:

`32 → 64 → 128 → 256`

Each convolution uses:

- 3 × 3 kernels
- ReLU activation
- MaxPooling

The classifier configuration corresponds to the selected Phase 3 experiment:

`CNN_A`

The output layer contains four softmax units.

## Training Configuration

Optimizer:

`Adam`

Initial learning rate:

`0.001`

Loss:

`SparseCategoricalCrossentropy`

Maximum training budget:

`20 epochs`

Early stopping:

- Monitor: validation loss
- Patience: 4
- Restore/select best validation-loss checkpoint

Random seed:

`42`

## Final Evaluation

Test accuracy:

`0.953471`

Macro precision:

`0.952490`

Macro recall:

`0.952266`

Macro F1:

`0.952291`

Macro one-vs-rest ROC-AUC:

`0.993157`

## Complexity and Efficiency

Total parameters:

`1,438,276`

Trainable parameters:

`1,438,276`

Saved model size:

`16.53 MB`

Training time:

`331.96 seconds`

Standardized model-only inference time:

`0.9296 ms/image`

Hardware:

`Tesla T4`

## Important Limitations

The reported performance applies to this dataset and experimental split.

Patient-level separation could not be independently verified.

Good performance on this dataset does not establish equivalent performance on unseen hospitals, MRI scanners, acquisition protocols, patient populations or real clinical environments.

The model is an academic image-classification experiment and is not a clinical diagnostic system.

## Reproducibility

Dataset version: `1`

Seed: `42`

Frozen split manifests and their SHA-256 checksums are distributed with the project.

The exact final model is stored as:

`models/cnn/custom_cnn_final.keras`
