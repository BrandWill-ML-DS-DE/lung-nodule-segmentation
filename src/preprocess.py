import os
import numpy as np
import pydicom
import SimpleITK as sitk

def load_scan(path):
    slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices]).astype(np.int16)

    for i in range(len(slices)):
        intercept = slices[i].RescaleIntercept
        slope = slices[i].RescaleSlope

        if slope != 1:
            image[i] = slope * image[i].astype(np.float64)
            image[i] = image[i].astype(np.int16)

        image[i] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def normalize_hu(image, min_bound=-1000.0, max_bound=400.0):
    image = (image - min_bound) / (max_bound - min_bound)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def save_nifti(image, output_path):
    sitk_img = sitk.GetImageFromArray(image)
    sitk.WriteImage(sitk_img, output_path)

def process_patient(patient_path, output_path):
    slices = load_scan(patient_path)
    image = get_pixels_hu(slices)
    image = normalize_hu(image)
    save_nifti(image, output_path)

if __name__ == "__main__":
    input_root = "data/LIDC-IDRI/"
    output_root = "processed/"

    os.makedirs(output_root, exist_ok=True)

    for patient in os.listdir(input_root):
        patient_path = os.path.join(input_root, patient)
        for subdir in os.listdir(patient_path):
            scan_path = os.path.join(patient_path, subdir)
            if os.path.isdir(scan_path):
                output_path = os.path.join(output_root, f"{patient}.nii.gz")
                process_patient(scan_path, output_path)
                print(f"Processed {patient}")
                break

