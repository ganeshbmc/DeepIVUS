import numpy as np
import os
import pydicom
import nibabel as nib

from IVUS_prediction import predict


dicom_path = "sample_data/001/dicom-00001.dcm"
gt_path = "sample_gt/001/dicom-00001.dcm.nii.gz"
out_dir = "sample_output"
case_id = "001"


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return (2.0 * inter / denom) if denom > 0 else 1.0


ds = pydicom.dcmread(dicom_path, force=True)
img = ds.pixel_array  # (H,W) or (F,H,W) or (F,H,W,C)
print(
    "dicom shape/dtype/min/max:",
    getattr(img, "shape", None),
    img.dtype,
    float(np.min(img)),
    float(np.max(img)),
)

# Ensure frame axis exists
if img.ndim == 2:
    img = img[None, :, :]

# Normalize to uint8 if needed (the model code subtracts a ~60 intensity mean)
if img.dtype != np.uint8:
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    x = np.clip((x - lo) / (hi - lo + 1e-6) * 255.0, 0, 255)
    img = x.astype(np.uint8)

pred = predict(img)  # (F,H,W) int32 labels
print("pred shape:", pred.shape, "labels:", np.unique(pred))

gt_nii = nib.load(gt_path)
gt = gt_nii.get_fdata().astype(np.int32)
print("gt shape:", gt.shape, "labels:", np.unique(gt))

# Align axes if needed (common GT layout is (H,W,F))
if gt.shape == pred.shape:
    pred_aligned = pred
elif gt.shape == (pred.shape[1], pred.shape[2], pred.shape[0]):
    pred_aligned = np.transpose(pred, (1, 2, 0))
else:
    raise RuntimeError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

os.makedirs(out_dir, exist_ok=True)
pred_out_path = os.path.join(out_dir, f"{case_id}_pred_like_gt.nii.gz")
pred_raw_out_path = os.path.join(out_dir, f"{case_id}_pred_raw_fhw.nii.gz")

hdr = gt_nii.header.copy()
hdr.set_data_dtype(np.uint8)

# Save a prediction volume that matches GT axis order/shape
nib.save(nib.Nifti1Image(pred_aligned.astype(np.uint8), gt_nii.affine, hdr), pred_out_path)
print("wrote:", pred_out_path)

# Also save the raw model output (F,H,W). Use identity affine because GT
# affine may not correspond to this axis order.
nib.save(nib.Nifti1Image(pred.astype(np.uint8), np.eye(4)), pred_raw_out_path)
print("wrote:", pred_raw_out_path)

for lab in np.unique(gt):
    d = dice(pred_aligned == lab, gt == lab)
    print("dice", int(lab), "=", float(d))
