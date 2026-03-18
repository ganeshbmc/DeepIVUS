import argparse
from pathlib import Path

import numpy as np
import pydicom


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on all sample DICOM datasets and save masks."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("sample_data"),
        help="Directory containing case subfolders with DICOM files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample_output"),
        help="Directory where predicted masks are written.",
    )
    parser.add_argument(
        "--save-nifti",
        action="store_true",
        help="Also save predictions as .nii.gz (requires nibabel).",
    )
    return parser.parse_args()


def collect_dicom_files(data_dir: Path):
    dicom_files = sorted([p for p in data_dir.glob("*/*.dcm") if p.is_file()])
    if not dicom_files:
        dicom_files = sorted([p for p in data_dir.rglob("*.dcm") if p.is_file()])
    return dicom_files


def prepare_input_array(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = img[None, :, :]

    if img.dtype != np.uint8:
        x = img.astype(np.float32)
        lo, hi = np.percentile(x, [1, 99])
        x = np.clip((x - lo) / (hi - lo + 1e-6) * 255.0, 0, 255)
        img = x.astype(np.uint8)

    return img


def maybe_save_nifti(pred: np.ndarray, path: Path):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "nibabel is not installed. Install it or run without --save-nifti."
        ) from exc

    nifti = nib.Nifti1Image(pred.astype(np.uint8), np.eye(4))
    nib.save(nifti, str(path))


def main():
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    from IVUS_prediction import predict

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    dicom_files = collect_dicom_files(data_dir)
    if not dicom_files:
        raise RuntimeError(f"No DICOM files found under: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(dicom_files)} DICOM file(s) under {data_dir}")
    success = 0
    failed = 0

    for idx, dicom_path in enumerate(dicom_files, start=1):
        try:
            case_id = dicom_path.parent.name
            stem = dicom_path.stem
            case_out_dir = output_dir / case_id
            case_out_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{idx}/{len(dicom_files)}] Processing {dicom_path}")
            ds = pydicom.dcmread(str(dicom_path), force=True)
            img = prepare_input_array(ds.pixel_array)

            pred = predict(img)
            pred_like_gt = np.transpose(pred, (1, 2, 0)).astype(np.uint8)

            npy_path = case_out_dir / f"{stem}_pred_like_gt_hwf.npy"
            np.save(npy_path, pred_like_gt)
            print(f"  wrote: {npy_path}")

            if args.save_nifti:
                nii_path = case_out_dir / f"{stem}_pred_like_gt_hwf.nii.gz"
                maybe_save_nifti(pred_like_gt, nii_path)
                print(f"  wrote: {nii_path}")

            success += 1
        except Exception as exc:
            failed += 1
            print(f"  ERROR for {dicom_path}: {exc}")

    print("Inference complete")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
