# Project 2: Deep Learning Applications

Introduction to Applied AI | BS 8th Semester | Department of Electronics
Author: Muhammad Raheel Ijaz

## Project 2, Week 5: Scene Classification with Transfer Learning

- Dataset: Intel Image Classification, 6 classes, 14,034 train / 3,000 test (split 11,929 train / 2,105 val / 3,000 test)
- Model: ResNet-18 pretrained on ImageNet, trained on CPU
- Feature extraction (3,078 trainable parameters): val accuracy 0.907
- Fine-tuning layer4 (about 8.4M trainable): val accuracy 0.928 (+2.09 points)
- Test accuracy (used once): 0.929; weakest class: glacier (recall 0.863), confused with mountain (65 images)
- Most instructive error: a black-and-white alley between tall stone buildings, labelled "buildings", was predicted "street" with probability 1.00, because the photo is taken from street level, so the two classes overlap and confidence is not correctness.
- Next: my own domain in Week 6: crop disease classification from leaf images (see proposal below)

### Results at a glance

| Phase | Epochs | Trainable params | Best val accuracy |
|---|---|---|---|
| Feature extraction (frozen body, new head) | 3 | 3,078 | 0.907 |
| Fine-tuning (layer4 + head, lr 1e-4 / 1e-3) | 3 | 8,396,806 | 0.928 |

Test accuracy 0.929 (2,787 / 3,000). The two biggest confusions are glacier vs mountain (65 + 37 errors) and street vs buildings (43 + 19 errors); together they are 77% of all 213 errors. Training showed mild overfitting at epoch 6 (training loss falling, validation loss rising), so the best epoch (epoch 5) was kept.

### Files

| File | Purpose |
|---|---|
| `Week5_Transfer_Learning.ipynb` | Full notebook (Parts 1 to 7, written answers, Check Your Understanding) |
| `images/` | Sample grid, edge detection, learning curves, confusion matrix, error gallery |
| `models/resnet18_intel.pt` | Fine-tuned ResNet-18 weights (44.8 MB) |
| `requirements.txt` | Python packages |
| `data/` | Dataset (not uploaded to GitHub, see below) |

### How to run locally (Windows)

```
cd "D:\Documents-D\7th Semester\Intro to AI\Project-2\Deep Learning Applications"
python -m venv venv
venv\Scripts\activate
pip install torch torchvision
pip install numpy matplotlib pillow scikit-learn jupyter
jupyter notebook
```

Download the dataset from https://www.kaggle.com/datasets/puneet6060/intel-image-classification , extract it into `data/`, open `Week5_Transfer_Learning.ipynb` and run the cells from top to bottom. With an NVIDIA GPU each epoch takes about a minute; on a CPU it takes 17 to 50 minutes.

## Week 6 Proposal

**Problem and domain:** I will build an image classifier for plant leaf diseases (crop disease detection) using the same transfer-learning pipeline as Week 5 (ResNet-18, augmentation on the training set only, feature extraction then fine-tuning). **Who benefits:** farmers and agricultural extension workers, who could identify a disease early from a photo of a leaf and treat it before it spreads, which reduces crop loss. **Candidate dataset:** PlantVillage (Hughes and Salathe), 54,303 leaf images in 38 classes of crop species and healthy/diseased states, available on Kaggle: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset . License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/), which allows non-commercial use with attribution and share-alike, so it is suitable for this coursework. Every class has hundreds of labelled images, which satisfies the "at least a few hundred per class" requirement.
