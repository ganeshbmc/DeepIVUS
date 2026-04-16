"""Legacy exploratory DICOM cropper.

This script was an early one-off utility and is kept for historical reference.
For current raw DICOM cropping, use `convert_raw_cropped.py` instead.

Why `convert_raw_cropped.py` replaced this script:
- accepts CLI arguments for input/output/excel paths
- supports case-range filtering (`--case-start/--case-end`)
- writes outputs matching the repo naming convention (`<case>_<VL>_cropped.dcm`)
- supports `VL1`/`VL2`/`VL3` crop columns from Excel
- writes structured CSV logs and supports `--dry-run`

Do not use this legacy script for new dataset preparation unless you are
intentionally reproducing the old exploratory behavior.
"""

import os
from pathlib import Path

import pandas as pd
import pydicom
from pydicom.encaps import encapsulate
from pydicom.uid import ExplicitVRLittleEndian


# =========================
# USER SETTINGS
# =========================
excel_path = "100_segmentation.xlsx"
dicom_root = "volumes"                    # e.g. volumes/1, volumes/2, volumes/3
output_root = "cropped_ultrasound_dicom"

study_col = "STUDY_NO."
first_col = "ANN_1STFRAME_VL1"
last_col = "ANN_LASTFRAME_VL1"

# True if Excel frame numbering starts at 1
excel_is_1_based = True

# process only available folders
process_only_existing_folders = True

# only first few cases for testing; set None for all
max_cases = 3

save_log_csv = True
log_csv_path = "logs/cropping_log.csv"


# =========================
# HELPERS
# =========================
def safe_str(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    if x.endswith(".0"):
        x = x[:-2]
    return x


def ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def is_dicom_file(filepath):
    try:
        pydicom.dcmread(filepath, stop_before_pixels=True)
        return True
    except Exception:
        return False


def find_dicom_files(folder):
    dicom_files = []
    for root, _, files in os.walk(folder):
        for fname in files:
            fpath = os.path.join(root, fname)
            if is_dicom_file(fpath):
                dicom_files.append(fpath)
    return dicom_files


def convert_excel_frame_to_python_index(frame_number, is_1_based=True):
    frame_number = int(frame_number)
    return frame_number - 1 if is_1_based else frame_number


def get_number_of_frames(ds):
    try:
        return int(getattr(ds, "NumberOfFrames", 1))
    except Exception:
        return 1


# =========================
# MAIN
# =========================
def main():
    os.makedirs(output_root, exist_ok=True)
    if save_log_csv:
        ensure_parent_dir(log_csv_path)

    df = pd.read_excel(excel_path)

    required_cols = [study_col, first_col, last_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in Excel. Found: {list(df.columns)}")

    logs = []
    processed = 0

    for idx, row in df.iterrows():
        study_no = safe_str(row[study_col])
        first_frame = row[first_col]
        last_frame = row[last_col]

        if study_no is None or pd.isna(first_frame) or pd.isna(last_frame):
            logs.append({
                "row_index": idx,
                "study_no": study_no,
                "status": "SKIPPED",
                "reason": "Missing study/frame values"
            })
            continue

        study_folder = os.path.join(dicom_root, study_no)

        if not os.path.isdir(study_folder):
            if process_only_existing_folders:
                continue
            logs.append({
                "row_index": idx,
                "study_no": study_no,
                "status": "FAILED",
                "reason": f"Study folder not found: {study_folder}"
            })
            print(f"[ERROR] Study folder not found: {study_folder}")
            continue

        dicom_files = find_dicom_files(study_folder)

        if len(dicom_files) == 0:
            logs.append({
                "row_index": idx,
                "study_no": study_no,
                "status": "FAILED",
                "reason": "No DICOM files found"
            })
            print(f"[ERROR] No DICOM files found in {study_folder}")
            continue

        if len(dicom_files) > 1:
            print(f"[WARNING] Study {study_no}: found {len(dicom_files)} DICOM files. Using the first one.")

        dicom_path = dicom_files[0]

        try:
            ds = pydicom.dcmread(dicom_path)
        except Exception as e:
            logs.append({
                "row_index": idx,
                "study_no": study_no,
                "status": "FAILED",
                "reason": f"Could not read DICOM: {e}"
            })
            print(f"[ERROR] Could not read DICOM for study {study_no}: {e}")
            continue

        total_frames = get_number_of_frames(ds)

        start_idx = convert_excel_frame_to_python_index(first_frame, excel_is_1_based)
        end_idx = convert_excel_frame_to_python_index(last_frame, excel_is_1_based)

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        if start_idx < 0 or end_idx >= total_frames:
            logs.append({
                "row_index": idx,
                "study_no": study_no,
                "status": "FAILED",
                "reason": f"Out of bounds. start={start_idx}, end={end_idx}, total_frames={total_frames}"
            })
            print(f"[ERROR] Out of bounds for study {study_no}: start={start_idx}, end={end_idx}, total_frames={total_frames}")
            continue

        # Decode pixel array and crop frames
        try:
            arr = ds.pixel_array
        except Exception as e:
            logs.append({
                "row_index": idx,
                "study_no": study_no,
                "status": "FAILED",
                "reason": f"Could not decode pixel array: {e}"
            })
            print(f"[ERROR] Could not decode pixel array for study {study_no}: {e}")
            continue

        # Expected ultrasound cine shapes:
        # grayscale: (frames, rows, cols)
        # color:     (frames, rows, cols, channels)
        if arr.ndim < 3:
            logs.append({
                "row_index": idx,
                "study_no": study_no,
                "status": "FAILED",
                "reason": f"Unexpected pixel_array shape: {arr.shape}"
            })
            print(f"[ERROR] Unexpected shape for study {study_no}: {arr.shape}")
            continue

        cropped = arr[start_idx:end_idx + 1]
        cropped_frames = cropped.shape[0]

        # Make a copy of original metadata
        out_ds = ds.copy()

        # Replace pixel data with cropped data
        out_ds.PixelData = cropped.tobytes()
        out_ds.NumberOfFrames = str(cropped_frames)

        # Update file meta / transfer syntax for safe writing
        if not hasattr(out_ds, "file_meta") or out_ds.file_meta is None:
            out_ds.file_meta = pydicom.dataset.FileMetaDataset()

        out_ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        out_ds.is_little_endian = True
        out_ds.is_implicit_VR = False

        # optional: update instance UID so output is a new object
        out_ds.SOPInstanceUID = pydicom.uid.generate_uid()
        if hasattr(out_ds.file_meta, "MediaStorageSOPInstanceUID"):
            out_ds.file_meta.MediaStorageSOPInstanceUID = out_ds.SOPInstanceUID

        out_dir = os.path.join(output_root, study_no)
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, f"{study_no}_cropped.dcm")
        out_ds.save_as(out_path, write_like_original=False)

        logs.append({
            "row_index": idx,
            "study_no": study_no,
            "status": "SUCCESS",
            "dicom_path": dicom_path,
            "total_frames_original": total_frames,
            "start_frame_excel": int(first_frame),
            "end_frame_excel": int(last_frame),
            "start_index_python": start_idx,
            "end_index_python": end_idx,
            "saved_frames": cropped_frames,
            "output_file": out_path
        })

        print(f"[OK] Study {study_no}: total_frames={total_frames}, saved_frames={cropped_frames} -> {out_path}")

        processed += 1
        if max_cases is not None and processed >= max_cases:
            break

    if save_log_csv:
        log_df = pd.DataFrame(logs)
        log_df.to_csv(log_csv_path, index=False)
        print(f"\nLog saved to: {log_csv_path}")


if __name__ == "__main__":
    main()
