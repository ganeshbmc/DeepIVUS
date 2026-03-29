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
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze NRRD files structure and write report to gt_conversion_log.md.",
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


def analyze_nrrd_structure(seg_volume: np.ndarray) -> dict:
    """Analyze NRRD structure and return info about each layer.
    
    Returns dict with:
    - num_layers: number of layers
    - layer_info: list of dicts with 'unique_values' for each layer
    - patterns: detected patterns ('binary', 'non-binary', 'missing')
    """
    num_layers = seg_volume.shape[0]
    layer_info = []
    patterns = []
    
    for layer_idx in range(num_layers):
        layer = seg_volume[layer_idx]
        unique_vals = sorted(set(layer.flatten()))
        layer_info.append({
            'layer_idx': layer_idx,
            'unique_values': unique_vals,
            'is_binary': unique_vals in [[0, 1], [0]],
        })
        
        if unique_vals in [[0, 1], [0]]:
            patterns.append('binary')
        else:
            patterns.append('non_binary')
    
    while len(patterns) < 3:
        patterns.append('missing')
    
    return {
        'num_layers': num_layers,
        'layer_info': layer_info,
        'patterns': patterns,
    }


def merge_segments_to_multiclass(seg_volume: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Merge segment layers with priority: lumen > vessel > plaque.
    
    Auto-detects layer structure and applies appropriate logic:
    - Binary layers (values [0,1]): use == 1
    - Non-binary layers (values [0,1,2]): use > 0
    - Missing layers: skip
    
    Priority rules:
    - Lumen wins over everything (label = 1)
    - Vessel wall wins over plaque (label = 2)
    - Plaque gets reassigned to lumen or vessel if overlaps exist
    - Plaque-only pixels become background (label = 0)
    """
    num_segs, H, W, F = seg_volume.shape
    
    layer_analysis = analyze_nrrd_structure(seg_volume)
    
    if verbose:
        print(f"  Detected: {layer_analysis['num_layers']} layers, patterns: {layer_analysis['patterns']}")
    
    multiclass = np.zeros((H, W, F), dtype=np.uint8)
    
    if num_segs >= 1:
        lumen = seg_volume[0]
        layer0_unique = sorted(set(lumen.flatten()))
        
        if layer0_unique in [[0, 1], [0]]:
            lumen_mask = (lumen == 1)
            logic = "binary (==1)"
        else:
            lumen_mask = (lumen > 0)
            logic = f"non-binary (>0), values={layer0_unique}"
        
        if verbose:
            print(f"  Layer 0 (lumen): using {logic}")
        multiclass = np.where(lumen_mask, 1, multiclass)
    
    if num_segs >= 2:
        vessel = seg_volume[1]
        vessel_unique = sorted(set(vessel.flatten()))
        
        if vessel_unique in [[0, 1], [0]]:
            vessel_mask = (vessel == 1)
            logic = "binary (==1)"
        else:
            vessel_mask = (vessel > 0)
            logic = f"non-binary (>0), values={vessel_unique}"
        
        if verbose:
            print(f"  Layer 1 (vessel): using {logic}")
        multiclass = np.where(vessel_mask, 2, multiclass)
    
    if num_segs >= 3:
        if verbose:
            print(f"  Layer 2 (plaque): absorbed by lumen/vessel")
    
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


def analyze_all_files(src_dir: Path, excel_path: Path, output_log: Path):
    """Analyze all NRRD files and write report to log file."""
    import openpyxl
    
    crop_params = load_excel_crop_params(excel_path)
    
    seg_files = sorted(src_dir.rglob("*_seg.nrrd"))
    
    report_lines = []
    report_lines.append("# NRRD Structure Analysis Report\n")
    report_lines.append(f"Source directory: {src_dir}")
    report_lines.append(f"Total NRRD files found: {len(seg_files)}\n")
    
    report_lines.append("## File Structure Details\n")
    report_lines.append("| Case | Layers | Layer 0 Values | Layer 1 Values | Layer 2 Values |")
    report_lines.append("|------|--------|----------------|----------------|----------------|")
    
    stats = {
        '2_layers': 0,
        '3_layers': 0,
        'layer0_binary': 0,
        'layer0_nonbinary': 0,
        'layer1_binary': 0,
        'layer1_nonbinary': 0,
    }
    
    for seg_path in seg_files:
        case_id = seg_path.parent.name
        filename = seg_path.name
        
        vl_name = "VL1"
        for suffix in ["VL1", "VL2", "VL3"]:
            if f"_{suffix}_seg.nrrd" in filename:
                vl_name = suffix
                break
        
        try:
            seg_volume = load_seg_nrrd(seg_path)
            analysis = analyze_nrrd_structure(seg_volume)
            
            num_layers = analysis['num_layers']
            
            if num_layers == 2:
                stats['2_layers'] += 1
            elif num_layers == 3:
                stats['3_layers'] += 1
            
            layer0_vals = analysis['layer_info'][0]['unique_values']
            layer1_vals = analysis['layer_info'][1]['unique_values'] if num_layers >= 2 else []
            layer2_vals = analysis['layer_info'][2]['unique_values'] if num_layers >= 3 else []
            
            if layer0_vals in [[0, 1], [0]]:
                stats['layer0_binary'] += 1
            else:
                stats['layer0_nonbinary'] += 1
            
            if layer1_vals in [[0, 1], [0]]:
                stats['layer1_binary'] += 1
            else:
                stats['layer1_nonbinary'] += 1
            
            row = f"| {case_id}_{vl_name} | {num_layers} | {layer0_vals} | {layer1_vals} | {layer2_vals} |"
            report_lines.append(row)
            
        except Exception as e:
            report_lines.append(f"| {case_id}_{vl_name} | ERROR: {e} |")
    
    report_lines.append("\n## Summary Statistics\n")
    report_lines.append(f"- 2-layer files: {stats['2_layers']}")
    report_lines.append(f"- 3-layer files: {stats['3_layers']}")
    report_lines.append(f"- Layer 0 (lumen) binary: {stats['layer0_binary']}")
    report_lines.append(f"- Layer 0 (lumen) non-binary: {stats['layer0_nonbinary']}")
    report_lines.append(f"- Layer 1 (vessel) binary: {stats['layer1_binary']}")
    report_lines.append(f"- Layer 1 (vessel) non-binary: {stats['layer1_nonbinary']}")
    
    report_lines.append("\n## Conversion Logic Applied\n")
    report_lines.append("- Binary layer (values [0,1]): uses == 1")
    report_lines.append("- Non-binary layer (values [0,1,2]): uses > 0")
    report_lines.append("- Missing layer: skipped")
    
    report_content = "\n".join(report_lines)
    
    with open(output_log, 'w') as f:
        f.write(report_content)
    
    print(f"Analysis report written to: {output_log}")
    print(report_content)


def main():
    args = parse_args()
    
    if args.analyze:
        output_log = Path("gt_conversion_log.md")
        print("Running analysis mode...")
        analyze_all_files(args.src_dir, args.excel_path, output_log)
        return
    
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
            
            # Merge segments into multi-class (with adaptive logic)
            label = merge_segments_to_multiclass(seg_cropped, verbose=True)
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