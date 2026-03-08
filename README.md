# 🫁 3D Volumetric Lung Nodule Segmentation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![MONAI](https://img.shields.io/badge/MONAI-Medical_AI-882B94)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end medical imaging pipeline designed to automate the detection and analysis of pulmonary nodules in 3D Computed Tomography (CT) scans. This system bridges the gap between deep learning segmentation and clinical **Radiomics**, providing both localized masks and quantifiable biological markers.

---

## 🔬 Scientific & Engineering Design

This pipeline handles high-stakes medical data complexity through a rigorous multi-stage architecture:

### 1. Hounsfield Unit (HU) Normalization
Raw DICOM values are rescaled using slope/intercept metadata and windowed to `[-1000, 400] HU`. This isolates lung tissue and ensures consistent density features across heterogeneous CT scanner hardware.

### 2. Volumetric 3D U-Net
Unlike standard 2D CNNs, this architecture utilizes **3D Convolutions** to capture the spherical morphology of nodules across the Z-axis (depth). This spatial awareness significantly reduces false positives inherent in slice-by-slice analysis.



### 3. Stochastic 3D Patching
To manage the massive memory footprint of 3D CT volumes, I implemented a custom `LungDataset` that performs random `64x64x64` patch extraction. This allows the model to train on high-resolution local features without exceeding GPU VRAM.

### 4. Radiomic Feature Extraction
Post-segmentation, the pipeline uses **PyRadiomics** to extract Shape, Texture (GLCM), and Intensity features. This transforms "black box" masks into structured datasets for downstream malignancy prediction.

---

## 🛠 Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Deep Learning** | PyTorch + MONAI | 3D U-Net implementation and Dice loss. |
| **Image Processing** | SimpleITK / Nibabel | Handling NIfTI/DICOM volumetric formats. |
| **Radiomics** | PyRadiomics | Clinical feature extraction (GLCM, GLRLM). |
| **Optimization** | Dice Similarity Coefficient | Loss function optimized for extreme class imbalance. |
| **Management** | Conda + Bash | Reproducible environment and setup scripts. |

---

## 🚀 Key Engineering Achievements

* **Robust Preprocessing Pipeline:** Handles the transition from raw medical metadata to normalized NIfTI files, including slice sorting by spatial position and pixel-array rescaling.
* **High-Precision Evaluation:** Leveraged the **Dice Similarity Coefficient (DSC)**. In medical segmentation, where the target may occupy $< 0.1\%$ of the volume, DSC provides a more representative performance metric than standard accuracy.
* **Edge-Case Mitigation:** Implemented smooth factors in metric calculations to prevent division-by-zero errors during training on "empty" patches, ensuring stable gradient descent.

---

## 🏁 Getting Started

### 1. Initialize Environment
```bash
# Clone the repo and run setup
git clone [https://github.com/your-username/lung-nodule-segmentation.git](https://github.com/your-username/lung-nodule-segmentation.git)
cd lung-nodule-segmentation
bash setup.sh
```
### 2. Pipeline Execution

1. Preprocess: Place LIDC-IDRI data in data/ and run:
   python preprocess.py
   
2. Train:  ```bash
   python train.py
   
3. Extract Features:
   python radiomics.py
   
---

## 📊 Results

| Metric | Value |
|--------|-------|
| Dice Score | ~0.78 (example) |
