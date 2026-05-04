# DMSFA-Net: Frequency-Aware Multi-Scale Visual Learning for LDH Classification

# Related Publication
This repository contains the official implementation of the following manuscript:
"Frequency-Aware Multi-Scale Visual Feature Learning for MRI-Based Lumbar Disc Herniation Classification"
Submitted to: The Visual Computer (Springer)

# Important Notice
This code is directly associated with the above manuscript.
If you use this code or any part of this implementation, please cite our paper.

# Overview
This repository implements DMSFA-Net, a deep learning framework designed for:
-Lumbar Disc Herniation (LDH) classification
-Multi-scale feature extraction
-Frequency-aware representation learning
-Attention-based lesion localization

# Model Components
- EfficientNetB0 backbone
- Multi-scale convolutional fusion
- Frequency-domain decomposition
- Hybrid attention mechanisms
- Grad-CAM alignment loss

# Project Structure
├── config.py
├── data_loader.py
├── dataset.py
├── model.py
├── train.py
├── evaluation.py
├── gradcam.py
├── main.py
├── requirements.txt
└── README.md

# Dataset
The experiments are conducted on:
- Sagittal-LDH-MRI Dataset (Binary Classification)
- Axial-LDH-MRI Dataset (Multi-class Classification)

# Note:
- Datasets are publicly available
- Due to privacy, datasets are not included in this repository

# Dataset Structure
To use your own dataset, follow this structure:
```bash
dataset/ 
|- class_0/ 
|     |- img1.jpg 
|     |- img2.jpg 
|- class_1/ 
|     |-  img1.jpg  
|- class_2/ 
|- class_3/
```
# Dataset Path
Update dataset path in config.py:
DATA_DIRECTORY = "path/to/dataset"

# Supported Modes
- Binary classification (2 classes)
- Multi-class classification (4 LDH stages)

# Installation
Clone the repository:
git clone https://github.com/aruna-research/DMSFA-Net-LDH.git
cd DMSFA-Net-LDH

# Install dependencies:
pip install -r requirements.txt

# Training
Run the training pipeline:
python main.py

# Results
The proposed model achieves strong performance :
- 97.38% accuracy for binary classification
- 96.71% accuracy for multi-class classification

# Grad-CAM Visualization
The model provides interpretability using Grad-CAM:
- Highlights important regions
- Aligns with lesion maps
- Improves clinical trust

# Reproducibility
To ensure reproducibility:
- Fixed random seed
- Input size: 224 × 224
- Standard preprocessing and normalization
- Data augmentation:
  - Rotation (~±20°)
  - Zoom (0.8–1.2)
  - Translation and flipping
 
# Hardware & Software
-	TensorFlow 2.15
- NVIDIA GPU (A100)
- Python 3.8+

## Citation
If you use this work, please cite:
```bibtex
@article{chouhan2026dmsfanet,
  title={ Frequency-Aware Multi-Scale Visual Feature Learning for MRI-Based Lumbar Disc Herniation Classification},
  author={Chouhan, A., Chauhan, K., Mitharwal, R. and Varma, T.},
  journal={The Visual Computer},
  year={2026}
}

# DOI and Archival Link
A permanent archived version of this repository is available at:
https://doi.org/xxxxxx

# Contact
Aruna Chouhan
Email: 2019rec9037@mnit.ac.in







