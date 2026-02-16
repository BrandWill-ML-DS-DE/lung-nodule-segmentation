# 3D Volumetric Lung Nodule Segmentation & Radiomics Pipeline

End-to-end deep learning pipeline for pulmonary nodule segmentation and radiomic feature extraction using the LIDC-IDRI dataset.

---

## 🔬 Overview

This project implements:

- DICOM → Hounsfield Unit normalization
- 3D Patch-Based U-Net segmentation
- Dice Similarity Coefficient evaluation
- Radiomic feature extraction (shape + texture)
- 3D visualization using 3D Slicer

Dataset: LIDC-IDRI (TCIA)

---

## 🧠 Architecture

- Framework: PyTorch + MONAI
- Model: 3D U-Net
- Patch size: 64×64×64
- Loss: Dice Loss
- Radiomics: PyRadiomics (GLCM, Shape, GLRLM)

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

