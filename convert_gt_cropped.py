import argparse
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np


KNOWN_SEGMENTS = ("lumen", "vessel_ob", "plaque")


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


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


def create_log_path() -> Path:
    log_dir = Path("sample_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"convert_gt_cropped_log_{datetime.now().strftime('%d%m%Y_%H%M')}.txt"


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
    header = next(ws.iter_rows(max_row=1, values_only=True))
    col_idx = {name: i for i, name in enumerate(header)}

    vl_runs = []
    for suffix in ["VL1", "VL2", "VL3"]:
        first_col = f"ANN_1STFRAME_{suffix}"
        last_col = f"ANN_LASTFRAME_{suffix}"
        if first_col in col_idx and last_col in col_idx:
            vl_runs.append((suffix, col_idx[first_col], col_idx[last_col]))

    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            break

        case_id = f"{int(row[0]):03d}"
        case_params = {}
        for vl_name, first_idx, last_idx in vl_runs:
            first_val = row[first_idx]
            last_val = row[last_idx]
            if first_val is not None and last_val is not None:
                case_params[vl_name] = (int(first_val) - 1, int(last_val) - 1)

        if case_params:
            params[case_id] = case_params

    return params


def load_seg_nrrd(path: Path):
    """Load a segmentation NRRD and return data plus header."""
    try:
        import nrrd
    except ImportError:
        raise RuntimeError("pynrrd is required. Install with: pip install pynrrd")

    data, header = nrrd.read(str(path))
    return data, header


def normalize_segment_name(name: str | None) -> str | None:
    if name is None:
        return None

    normalized = "".join(ch.lower() for ch in str(name) if ch.isalnum())
    if not normalized:
        return None

    if "lumen" in normalized:
        return "lumen"
    if any(token in normalized for token in ["vessel", "vesel", "veseel", "vesssel"]):
        return "vessel_ob"
    if any(token in normalized for token in ["plaque", "plauqe", "palque"]):
        return "plaque"
    return None


def extract_segment_entries(header: dict) -> list[dict]:
    entries = []
    index = 0
    while f"Segment{index}_Name" in header:
        entries.append(
            {
                "index": index,
                "name": header.get(f"Segment{index}_Name"),
                "layer": header.get(f"Segment{index}_Layer"),
                "label_value": header.get(f"Segment{index}_LabelValue"),
            }
        )
        index += 1
    return entries


def assign_canonical_segment_names(entries: list[dict]) -> tuple[list[dict], list[str]]:
    notes = []
    assigned = []
    used_names = set()

    for entry in entries:
        canonical_name = normalize_segment_name(entry["name"])
        assigned_entry = dict(entry)
        assigned_entry["canonical_name"] = canonical_name
        assigned.append(assigned_entry)
        if canonical_name is not None:
            used_names.add(canonical_name)
            if canonical_name != entry["name"]:
                notes.append(
                    f"mapped segment '{entry['name']}' to canonical name '{canonical_name}'"
                )

    missing_names = [name for name in KNOWN_SEGMENTS if name not in used_names]
    unknown_entries = [entry for entry in assigned if entry["canonical_name"] is None]

    if len(unknown_entries) == 1 and len(missing_names) == 1:
        entry = unknown_entries[0]
        inferred_name = missing_names[0]
        entry["canonical_name"] = inferred_name
        notes.append(
            f"inferred unknown segment '{entry['name']}' as '{inferred_name}' because it was the only missing semantic segment"
        )

    for entry in assigned:
        if entry["canonical_name"] is None:
            notes.append(
                f"could not map segment '{entry['name']}' to a known semantic segment"
            )

    return assigned, notes


def parse_int(value, field_name: str) -> int:
    if value is None:
        raise ValueError(f"missing {field_name}")
    return int(value)


def extract_segment_mask(seg_data: np.ndarray, entry: dict) -> np.ndarray:
    label_value = parse_int(entry["label_value"], "label value")

    if seg_data.ndim == 4:
        layer_index = parse_int(entry["layer"], "layer index")
        if not (0 <= layer_index < seg_data.shape[0]):
            raise ValueError(
                f"layer index {layer_index} out of bounds for data with {seg_data.shape[0]} layers"
            )
        return seg_data[layer_index] == label_value

    if seg_data.ndim == 3:
        return seg_data == label_value

    raise ValueError(f"unsupported NRRD data shape: {seg_data.shape}")


def crop_segmentation_data(seg_data: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
    if seg_data.ndim == 4:
        return seg_data[:, :, :, start_frame : end_frame + 1]
    if seg_data.ndim == 3:
        return seg_data[:, :, start_frame : end_frame + 1]
    raise ValueError(f"unsupported NRRD data shape: {seg_data.shape}")


def count_nonzero(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def reconstruct_semantic_masks(seg_data: np.ndarray, header: dict) -> tuple[dict, dict]:
    entries = extract_segment_entries(header)
    if not entries:
        raise RuntimeError("no Segment{i} entries found in NRRD header")

    entries, notes = assign_canonical_segment_names(entries)
    masks = {name: None for name in KNOWN_SEGMENTS}
    raw_segment_details = []

    for entry in entries:
        canonical_name = entry["canonical_name"]
        if canonical_name is None:
            raw_segment_details.append(
                {
                    "index": entry["index"],
                    "name": entry["name"],
                    "canonical_name": None,
                    "layer": entry["layer"],
                    "label_value": entry["label_value"],
                    "voxels": None,
                }
            )
            continue

        mask = extract_segment_mask(seg_data, entry)
        raw_segment_details.append(
            {
                "index": entry["index"],
                "name": entry["name"],
                "canonical_name": canonical_name,
                "layer": entry["layer"],
                "label_value": entry["label_value"],
                "voxels": count_nonzero(mask),
            }
        )

        if masks[canonical_name] is None:
            masks[canonical_name] = mask
        else:
            masks[canonical_name] = np.logical_or(masks[canonical_name], mask)
            notes.append(
                f"merged multiple segments for canonical name '{canonical_name}'"
            )

    spatial_shape = seg_data.shape[-3:]
    for name in KNOWN_SEGMENTS:
        if masks[name] is None:
            masks[name] = np.zeros(spatial_shape, dtype=bool)

    info = {
        "entries": raw_segment_details,
        "notes": notes,
        "missing_required": [
            name for name in ("lumen", "vessel_ob") if count_nonzero(masks[name]) == 0
        ],
    }
    return masks, info


def build_final_labelmap(masks: dict) -> tuple[np.ndarray, dict]:
    lumen_mask = masks["lumen"]
    vessel_mask = masks["vessel_ob"]
    plaque_mask = masks["plaque"]

    final_labelmap = np.zeros(lumen_mask.shape, dtype=np.uint8)
    final_labelmap[lumen_mask] = 1
    vessel_wall_mask = np.logical_and(vessel_mask, np.logical_not(lumen_mask))
    final_labelmap[vessel_wall_mask] = 2

    qc = {
        "lumen_voxels": count_nonzero(lumen_mask),
        "vessel_voxels": count_nonzero(vessel_mask),
        "plaque_voxels": count_nonzero(plaque_mask),
        "vessel_wall_voxels": count_nonzero(vessel_wall_mask),
        "lumen_outside_vessel": count_nonzero(np.logical_and(lumen_mask, np.logical_not(vessel_mask))),
        "plaque_outside_vessel": count_nonzero(np.logical_and(plaque_mask, np.logical_not(vessel_mask))),
        "plaque_overlap_lumen": count_nonzero(np.logical_and(plaque_mask, lumen_mask)),
        "plaque_overlap_vessel_wall": count_nonzero(np.logical_and(plaque_mask, vessel_wall_mask)),
        "plaque_inside_vessel": count_nonzero(np.logical_and(plaque_mask, vessel_mask)),
    }
    return final_labelmap, qc


def analyze_nrrd_structure(seg_data: np.ndarray, header: dict) -> dict:
    entries, notes = assign_canonical_segment_names(extract_segment_entries(header))
    entry_info = []
    for entry in entries:
        try:
            mask = extract_segment_mask(seg_data, entry) if entry["canonical_name"] else None
            voxels = count_nonzero(mask) if mask is not None else None
        except Exception as exc:
            voxels = f"ERROR: {exc}"

        entry_info.append(
            {
                "index": entry["index"],
                "name": entry["name"],
                "canonical_name": entry["canonical_name"],
                "layer": entry["layer"],
                "label_value": entry["label_value"],
                "voxels": voxels,
            }
        )

    return {
        "shape": tuple(seg_data.shape),
        "ndim": seg_data.ndim,
        "entries": entry_info,
        "notes": notes,
    }


def save_labelmap(label: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, label.astype(np.uint8))
    print(f"  wrote: {path}")


def save_nifti(label: np.ndarray, path: Path):
    try:
        import nibabel as nib
    except ImportError:
        raise RuntimeError("nibabel is required for NIfTI output.")

    path.parent.mkdir(parents=True, exist_ok=True)
    nifti = nib.Nifti1Image(label.astype(np.uint8), np.eye(4))
    nib.save(nifti, str(path))
    print(f"  wrote: {path}")


def determine_vl_name(seg_path: Path) -> str:
    filename = seg_path.name
    for suffix in ["VL1", "VL2", "VL3"]:
        if f"_{suffix}_seg.nrrd" in filename:
            return suffix
    return "VL1"


def analyze_all_files(src_dir: Path, output_log: Path):
    seg_files = sorted(src_dir.rglob("*_seg.nrrd"))

    report_lines = []
    report_lines.append("# NRRD Structure Analysis Report")
    report_lines.append("")
    report_lines.append(f"Source directory: {src_dir}")
    report_lines.append(f"Total NRRD files found: {len(seg_files)}")
    report_lines.append("")

    for seg_path in seg_files:
        case_id = seg_path.parent.name
        vl_name = determine_vl_name(seg_path)
        report_lines.append(f"## {case_id}_{vl_name}")
        report_lines.append(f"- file: {seg_path}")
        try:
            seg_data, header = load_seg_nrrd(seg_path)
            analysis = analyze_nrrd_structure(seg_data, header)
            report_lines.append(f"- shape: {analysis['shape']}")
            report_lines.append(f"- ndim: {analysis['ndim']}")
            for entry in analysis["entries"]:
                report_lines.append(
                    "- segment "
                    f"{entry['index']}: name={entry['name']}, canonical={entry['canonical_name']}, "
                    f"layer={entry['layer']}, label={entry['label_value']}, voxels={entry['voxels']}"
                )
            if analysis["notes"]:
                for note in analysis["notes"]:
                    report_lines.append(f"- note: {note}")
        except Exception as exc:
            report_lines.append(f"- ERROR: {exc}")
        report_lines.append("")

    output_log.write_text("\n".join(report_lines) + "\n")
    print(f"Analysis report written to: {output_log}")


def convert_all(args):
    print(f"Loading crop parameters from {args.excel_path}...")
    crop_params = load_excel_crop_params(args.excel_path)
    print(f"Found crop parameters for {len(crop_params)} cases")

    seg_files = sorted(args.src_dir.rglob("*_seg.nrrd"))
    if not seg_files:
        raise RuntimeError(f"No segmentation files found in {args.src_dir}")

    print(f"Found {len(seg_files)} segmentation file(s)")

    success = 0
    failed = 0

    for seg_path in seg_files:
        try:
            case_id = seg_path.parent.name
            vl_name = determine_vl_name(seg_path)

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
            print(f"Processing {seg_path.name} (frames {start_frame + 1}-{end_frame + 1})")

            seg_data, header = load_seg_nrrd(seg_path)
            print(f"  loaded shape: {seg_data.shape}")
            seg_cropped = crop_segmentation_data(seg_data, start_frame, end_frame)
            print(f"  cropped shape: {seg_cropped.shape}")

            masks, info = reconstruct_semantic_masks(seg_cropped, header)
            for entry in info["entries"]:
                print(
                    "  segment "
                    f"{entry['index']}: name={entry['name']}, canonical={entry['canonical_name']}, "
                    f"layer={entry['layer']}, label={entry['label_value']}, voxels={entry['voxels']}"
                )
            for note in info["notes"]:
                print(f"  note: {note}")

            if info["missing_required"]:
                raise RuntimeError(
                    "missing required semantic segment(s): " + ", ".join(info["missing_required"])
                )

            label, qc = build_final_labelmap(masks)
            print(f"  merged shape: {label.shape}")
            print(
                "  QC: "
                f"lumen={qc['lumen_voxels']}, vessel_ob={qc['vessel_voxels']}, "
                f"vessel_wall={qc['vessel_wall_voxels']}, plaque={qc['plaque_voxels']}"
            )
            print(
                "  QC overlaps: "
                f"lumen_outside_vessel={qc['lumen_outside_vessel']}, "
                f"plaque_outside_vessel={qc['plaque_outside_vessel']}, "
                f"plaque_overlap_lumen={qc['plaque_overlap_lumen']}, "
                f"plaque_overlap_vessel_wall={qc['plaque_overlap_vessel_wall']}"
            )

            output_stem = f"{case_id}_{vl_name}_label_hwf"
            npy_path = args.dst_dir / case_id / f"{output_stem}.npy"
            save_labelmap(label, npy_path)

            if args.save_nifti:
                nii_path = args.dst_dir / case_id / f"{output_stem}.nii.gz"
                save_nifti(label, nii_path)

            success += 1

        except Exception as exc:
            print(f"  ERROR for {seg_path}: {exc}")
            failed += 1

    print("\nConversion complete")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")


def main():
    args = parse_args()
    log_path = create_log_path()

    with log_path.open("w", encoding="utf-8") as log_file:
        tee = TeeStream(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            print(f"Logging to: {log_path}")
            print(f"Started at: {datetime.now().isoformat(timespec='seconds')}")

            if args.analyze:
                output_log = Path("gt_conversion_log.md")
                print("Running analysis mode...")
                analyze_all_files(args.src_dir, output_log)
                return

            convert_all(args)


if __name__ == "__main__":
    main()
