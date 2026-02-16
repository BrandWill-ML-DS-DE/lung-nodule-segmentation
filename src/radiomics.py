from radiomics import featureextractor
import SimpleITK as sitk
import pandas as pd

extractor = featureextractor.RadiomicsFeatureExtractor()

image = sitk.ReadImage("sample_image.nii.gz")
mask = sitk.ReadImage("sample_mask.nii.gz")

features = extractor.execute(image, mask)

df = pd.DataFrame([features])
df.to_csv("radiomics_features.csv", index=False)

print("Radiomics features extracted.")

