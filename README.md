## 🫁 3D Volumetric Lung Nodule Segmentation & Radiomics Pipeline

An end-to-end medical imaging pipeline designed to automate the detection and analysis of pulmonary nodules in 3D Computed Tomography (CT) scans. This system bridges the gap between deep learning segmentation and clinical Radiomics, providing both a localized mask and quantifiable biological markers.

---

## 🔬 Scientific & Engineering Design

This pipeline handles the high-stakes complexity of medical data through several rigorous stages:

- Hounsfield Unit (HU) Normalization: Raw DICOM values are rescaled using slope/intercept metadata and windowed to [-1000, 400] HU. This isolates lung tissue and ensures the model learns consistent density features regardless of the CT scanner brand.
- Volumetric 3D U-Net: Unlike standard 2D CNNs, this architecture uses 3D convolutions and spatial_dims=3 to capture the spherical morphology of nodules across the Z-axis (depth), significantly reducing false positives compared to slice-by-slice analysis.
- Stochastic 3D Patching: To manage the massive memory footprint of 3D CT volumes, I implemented a custom LungDataset that performs random 64x64x64 patch extraction, allowing the model to train on high-resolution local features without exceeding GPU VRAM.
- Radiomic Feature Extraction: Once segmented, the pipeline uses PyRadiomics to extract Shape, Texture (GLCM), and Intensity features. This transforms a "black box" mask into a structured dataset for downstream malignancy prediction.

---

## 🛠 Tech Stack

Component,Technology,Role
Deep Learning,PyTorch + MONAI,3D U-Net implementation and Dice loss.
Image Processing,SimpleITK / Nibabel,Handling NIfTI/DICOM volumetric formats.
Radiomics,PyRadiomics,"Clinical feature extraction (GLCM, GLRLM)."
Optimization,Dice Similarity Coefficient,Loss function optimized for class imbalance.
Management,Conda + Bash,Reproducible environment and setup scripts.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Dice Score | ~0.78 (example) |

---

## 🛠 Installation

```bash
git clone https://github.com/YOUR_USERNAME/lung-nodule-segmentation.git
cd lung-nodule-segmentation
bash setup.sh

