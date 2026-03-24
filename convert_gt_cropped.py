import argparse
import shutil
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Segmentation.seg.nrrd files to dense labelmaps with cropping."
    )
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=Path("sample_excel/100_segmentation.xlsx"),
        help="Path to Excel file with crop parameters.",
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("sample_gt_uncropped"),
        help="Directory containing uncropped segmentation NRRD files.",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        default=Path("sample_gt_cropped"),
        help="Output directory for cropped labelmaps.",
    )
    parser.add_argument(
        "--save-nifti",
        action="store_true",
        help="Also save outputs as .nii.gz.",
    )
    return parser.parse_args()


def load_excel_crop_params(excel_path: Path) -> dict:
    """Load crop parameters from Excel file.
    
    Returns dict: {case_id: {'VL1': (start, end), 'VL2': (start, end), ...}}
    All frame indices are 1-indexed from Excel.
    """
    try:
        import openpyxl
    except ImportError:
        raise RuntimeError("openpyxl is required. Install with: pip install openpyxl")
    
    wb = openpyxl.load_workbook(str(excel_path), data_only=True)
    ws = wb["Sheet1"]
    
    params = {}
    
    # Read header to find column indices
    header = next(ws.iter_rows(max_row=1, values_only=True))
    col_idx = {name: i for i, name in enumerate(header)}
    
    # Expected columns
    vl_runs = []
    for suffix in ["VL1", "VL2", "VL3"]:
        first_col = f"ANN_1STFRAME_{suffix}"
        last_col = f"ANN_LASTFRAME_{suffix}"
        if first_col in col_idx and last_col in col_idx:
            vl_runs.append((suffix, col_idx[first_col], col_idx[last_col]))
    
    # Read data rows
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            break
        
        study_no = row[0]
        case_id = f"{int(study_no):03d}"
        
        case_params = {}
        for vl_name, first_idx, last_idx in vl_runs:
            first_val = row[first_idx]
            last_val = row[last_idx]
            if first_val is not None and last_val is not None:
                # Excel is 1-indexed, convert to 0-indexed for Python (end is inclusive)
                case_params[vl_name] = (int(first_val) - 1, int(last_val) - 1)
        
        if case_params:
            params[case_id] = case_params
    
    return params


def load_seg_nrrd(path: Path) -> np.ndarray:
    """Load .seg.nrrd file and return the labelmap volume.
    
    Returns: np.ndarray of shape (segments, H, W, F) - segmentation layers stacked.
    """
    try:
        import nrrd
    except ImportError:
        raise RuntimeError("pynrrd is required. Install with: pip install pynrrd")
    
    data, header = nrrd.read(str(path))
    return data


def merge_segments_to_multiclass(seg_volume: np.ndarray) -> np.ndarray:
    """Merge segment layers into a single multi-class dense volume.
    
    The .seg.nrrd contains multiple binary labelmaps stacked along axis 0.
    We merge them into one volume where each pixel has a class ID:
    - 0 = background (no segment)
    - 1 = first segment (layer 0)
    - 2 = second segment (layer 1)
    - etc.
    
    If multiple segments overlap at a pixel, the highest layer index wins.
    """
    # seg_volume shape: (num_segments, H, W, F)
    # Create output: start with 0 (background)
    num_segs, H, W, F = seg_volume.shape
    multiclass = np.zeros((H, W, F), dtype=np.uint8)
    
    # For each segment layer, add its contribution
    # Higher layer index = higher class label
    for layer_idx in range(num_segs):
        layer = seg_volume[layer_idx]
        # Where this layer has label (non-zero), set class = layer_idx + 1
        # Use max to handle overlaps (higher layer wins)
        multiclass = np.maximum(multiclass, layer * (layer_idx + 1))
    
    return multiclass


def save_labelmap(label: np.ndarray, path: Path):
    """Save labelmap as .npy (and optionally .nii.gz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, label.astype(np.uint8))
    print(f"  wrote: {path}")


def save_nifti(label: np.ndarray, path: Path):
    """Save labelmap as .nii.gz."""
    try:
        import nibabel as nib
    except ImportError:
        raise RuntimeError("nibabel is required for NIfTI output.")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    nifti = nib.Nifti1Image(label.astype(np.uint8), np.eye(4))
    nib.save(nifti, str(path))
    print(f"  wrote: {path}")


def main():
    args = parse_args()
    
    # Load crop parameters from Excel
    print(f"Loading crop parameters from {args.excel_path}...")
    crop_params = load_excel_crop_params(args.excel_path)
    print(f"Found crop parameters for {len(crop_params)} cases")
    
    # Find all segmentation files in src_dir
    seg_files = sorted(args.src_dir.rglob("*_seg.nrrd"))
    if not seg_files:
        raise RuntimeError(f"No segmentation files found in {args.src_dir}")
    
    print(f"Found {len(seg_files)} segmentation file(s)")
    
    success = 0
    failed = 0
    
    for seg_path in seg_files:
        try:
            case_id = seg_path.parent.name  # e.g., "044"
            
            # Determine VL run from filename: 044_VL1_seg.nrrd -> VL1
            filename = seg_path.name  # e.g., "044_VL1_seg.nrrd"
            # Extract VL name: "044_VL1_seg.nrrd" -> "VL1"
            vl_name = None
            for suffix in ["VL1", "VL2", "VL3"]:
                if f"_{suffix}_seg.nrrd" in filename:
                    vl_name = suffix
                    break
            
            if vl_name is None:
                # Single run case: 001_seg.nrrd -> use VL1 as default
                vl_name = "VL1"
            
            # Get crop parameters for this case + VL
            if case_id not in crop_params:
                print(f"Warning: No crop params for case {case_id}, skipping")
                failed += 1
                continue
            
            case_params = crop_params[case_id]
            if vl_name not in case_params:
                print(f"Warning: No crop params for {case_id}/{vl_name}, skipping")
                failed += 1
                continue
            
            start_frame, end_frame = case_params[vl_name]
            print(f"Processing {seg_path.name} (frames {start_frame+1}-{end_frame})")
            
            # Load segmentation
            seg_volume = load_seg_nrrd(seg_path)
            print(f"  loaded shape: {seg_volume.shape}")
            
            # Crop along last axis (frames)
            # end_frame is inclusive, so slice to end_frame+1
            seg_cropped = seg_volume[:, :, :, start_frame:end_frame+1]
            print(f"  cropped shape: {seg_cropped.shape}")
            
            # Merge segments into multi-class
            label = merge_segments_to_multiclass(seg_cropped)
            # Now shape is (H, W, F_cropped)
            print(f"  merged shape: {label.shape}")
            
            # Output filename: 044_VL1_label_hwf.npy
            output_stem = f"{case_id}_{vl_name}_label_hwf"
            
            # Save .npy
            npy_path = args.dst_dir / case_id / f"{output_stem}.npy"
            save_labelmap(label, npy_path)
            
            # Save .nii.gz if requested
            if args.save_nifti:
                nii_path = args.dst_dir / case_id / f"{output_stem}.nii.gz"
                save_nifti(label, nii_path)
            
            success += 1
            
        except Exception as e:
            print(f"  ERROR for {seg_path}: {e}")
            failed += 1
    
    print(f"\nConversion complete")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()