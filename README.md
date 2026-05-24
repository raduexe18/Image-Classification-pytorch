# Image Classification with Neural Networks

PyTorch implementation of MLP, CNN, ResNet-18 and MobileNet-v2 for image classification 
on two imbalanced datasets: skin lesion diagnosis (7 classes, 10k images) and facial 
expression recognition (7 classes, 15k images).

## Results
| Model | Dataset | F1-macro |
|---|---|---|
| ResNet-18 FT | Skin Moles | 0.766 |
| ResNet-18 FT | Face Expressions | 0.743 |

## Techniques
- WeightedRandomSampler for class imbalance (58:1 ratio)
- OneCycleLR scheduler, Label Smoothing, AdamW
- Ablation studies: Dropout, BatchNorm, augmentation configs, LR warmup
