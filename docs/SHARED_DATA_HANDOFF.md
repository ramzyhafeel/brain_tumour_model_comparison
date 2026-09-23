
# Shared Dataset Handoff

## Dataset

Masoud Nickparvar — Brain Tumor MRI Dataset

Pinned version:

masoudnickparvar/brain-tumor-mri-dataset/versions/1

Do NOT use the unversioned latest dataset handle.

## Frozen Splits

Training:
4,353 images

Validation:
1,089 images

Test:
1,311 images

Members 2, 3 and 4 MUST load these manifests directly:

splits/train.csv
splits/val.csv
splits/test.csv

They MUST NOT execute train_test_split() again.

## SHA-256 Manifest Checksums

train.csv

7273ef5fc2cb605c5b03b22a23cda0cb383ac0d9ce5949ac3c95649b5a4270cb

val.csv

37b24456cfcd1b69df939e36603958eae6a9124b794c32c83a532b9018c6c88f

test.csv

9afc40a38949eb4f46f8c9591d3b1f51cdd11aa991bfe2cf1d15c386c5537d39

If any checksum differs, stop before model development.

## Class Mapping

glioma = 0
meningioma = 1
notumor = 2
pituitary = 3

Do not change the mapping.

## Shared Experimental Protocol

Image size:
224 × 224 × 3

Batch size:
16

Random seed:
42

Maximum epoch budget:
20

Early stopping:
monitor val_loss
patience 4
restore/select best validation checkpoint

Training augmentation:
same conservative rotation, zoom and translation policy used by CNN.

Testing:
held out until each member has selected their final configuration using validation results.

## Architecture-Specific Preprocessing

Custom CNN:
Rescaling(1./255)

VGG16:
keras.applications.vgg16.preprocess_input

ResNet50:
keras.applications.resnet50.preprocess_input

DenseNet121:
keras.applications.densenet.preprocess_input

Do NOT apply Rescaling(1./255) before an application-specific preprocess_input unless the architecture explicitly requires it.

## Standard Inference Timing

Use:

batch size = 16
warm-up runs = 5
timed runs = 50
training=False

Time already-decoded tensors so model computation is compared rather than filesystem speed.

Use the same GPU where possible and always record the actual hardware.

## Dataset Limitations

Exact duplicate leakage was audited before the frozen split was created.

Reliable patient identifiers were not established.

Therefore patient-level independence must not be claimed.
