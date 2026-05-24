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

## Dataset

The datasets are not included in this repository.

- **Skin Moles** - [Vai de Pielea Mea Kaggle Competition](https://www.kaggle.com/competitions/vai-de-pielea-mea)
- **Skin Moles** - [Train & Test](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)
- **Face Expressions** - [You're on Candid Camera Kaggle Competition](https://www.kaggle.com/competitions/youre-on-candid-camera)
- **Face Expressions** - [Train & Test](https://www.kaggle.com/datasets/msambare/fer2013)

To run the notebook, download the datasets from Kaggle and update `ROOT_DIR` in the Setup cell to point to your local data folder.
