# DMSFA-Net: Frequency-Aware Multi-Scale Visual Learning for LDH Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20065126.svg)](https://doi.org/10.5281/zenodo.20065126)

## Related Publication
This repository contains the official implementation of the following manuscript:
"Frequency-Aware Multi-Scale Visual Feature Learning for MRI-Based Lumbar Disc Herniation Classification"
Submitted to: The Visual Computer (Springer)

## Important Notice
This code is directly associated with the above manuscript.
If you use this code or any part of this implementation, please cite our paper.

## Overview
This repository implements DMSFA-Net, a deep learning framework designed for:
-Lumbar Disc Herniation (LDH) classification
-Multi-scale feature extraction
-Frequency-aware representation learning
-Attention-based lesion localization

## Model Components
- EfficientNetB0 backbone
- Multi-scale convolutional fusion
- Frequency-domain decomposition
- Hybrid attention mechanisms
- Grad-CAM alignment loss

## Project Structure
```bash
├── config.py
├── data_loader.py
├── dataset.py
├── evaluation.py
├── gradcam.py
├── main.py
├── model.py 
├── train.py
├── requirements.txt
└── README.md
```
## Installation
Clone the repository:
```bash
git clone https://github.com/aruna-research/DMSFA-Net-LDH.git
cd DMSFA-Net-LDH
```

## Install dependencies:
```bash
pip install -r requirements.txt
```

## Hardware & Software Requirements
-	TensorFlow 2.15
- NVIDIA GPU (T4)
- Python 3.8+

## Dataset
The experiments are conducted on:
- Sagittal-LDH-MRI Dataset (Binary Classification)
- Axial-LDH-MRI Dataset (Multi-class Classification)

## Dataset Information
The MRI datasets used in this work were obtained from publicly available sources. For the experiments, the original MRI images were organized into sagittal binary-class and axial multi-class LDH classification subsets based on the study requirements.
Image preprocessing is performed dynamically within the training pipeline using the provided source code. Since these dataset splits were specifically prepared for this study, they are not redistributed separately via the repository.
Researchers can reproduce the experiments using the publicly available datasets and the implementation details provided in the manuscript and repository documentation.

## Original Dataset Reference
S. Sudirman \textit{et al.}, “MATLAB source code for developing ground truth dataset, semantic segmentation, and evaluation for the lumbar spine MRI dataset,” \textit{Mendeley Data}, 2019., publicly available lumbar spine MRI dataset.

## Note:
- Datasets are publicly available
- Due to privacy, datasets are not included in this repository

## Dataset Structure
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
## Dataset Path
Update dataset path in config.py:
DATA_DIRECTORY = "path/to/dataset"

## Supported Modes
- Binary classification (2 classes)
- Multi-class classification (4 LDH stages)

## Training
Run the training pipeline using:
```bash
python main.py
```

## Evaluation
Run evaluation using:
```bash
python evaluation.py
```

## Reproducibility
To ensure reproducibility:
- Fixed random seed
- Input image size: 224 × 224
- Standard preprocessing and normalization
- Data augmentation:
  - Rotation (~±20°)
  - Zoom (0.8–1.2)
  - Translation and flipping
 
## Results
The proposed model achieves strong performance :
- 97.38% accuracy for binary classification
- 96.71% accuracy for multi-class classification

## Grad-CAM Visualization
The proposed framework provides interpretable lesion-aware visualizations using Grad-CAM alignment.
The generated attention maps:
- Highlight diagnostically relevant regions
- Improve lesion localization interpretability
- Enhance clinical reliability

## DOI and Archival Link
A permanent archived version of this repository is available at:
https://doi.org/10.5281/zenodo.20065126

### Citation
If you use this work, please cite:
```bibtex
@article{chouhan2026dmsfanet,
  title={Frequency-Aware Multi-Scale Visual Feature Learning for MRI-Based Lumbar Disc Herniation Classification},
  author={Chouhan, Aruna and Chauhan, Krishna and Mitharwal, Rajendra and Varma, Tarun},
  note={Submitted to The Visual Computer},
  year={2026}
}
```
##License
This project is released under the MIT License.

## Contact
Aruna Chouhan
Email: 2019rec9037@mnit.ac.in







