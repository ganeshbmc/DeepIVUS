import napari
import numpy as np
import nibabel as nib
import pydicom
import os

sample_id = input("Enter sample ID (3-digit, e.g., 001): ").strip()

dicom_path = os.path.join("sample_data", sample_id, f"{sample_id}_VL1_cropped.dcm")
img = pydicom.dcmread(dicom_path).pixel_array
volume = img[:, :, :, 0]  # (871, 512, 512)

label_path = os.path.join("sample_output", sample_id, f"{sample_id}_VL1_cropped_pred_like_gt_hwf.nii.gz")
labels = nib.load(label_path).get_fdata()
labels = np.transpose(labels, (2, 0, 1))  # (871, 512, 512)

viewer = napari.Viewer()
viewer.add_image(volume, name="IVUS")
viewer.add_labels(labels, name="Labels")

napari.run()
