#!/usr/bin/env python
"""
Evaluation script for IVUS segmentation predictions vs ground truth.

Default behavior evaluates both lumen and vessel wall and writes a wide,
case-level CSV that is easier to use for downstream statistical analysis.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


LABEL_MAPPINGS = {
    "identity": {0: 0, 1: 1, 2: 2, 3: 0},
    "swap_1_2": {0: 0, 1: 2, 2: 1, 3: 0},
}

CLASS_CONFIG = {
    "lumen": {"id": 1, "prefix": "lumen", "display": "lumen"},
    "vessel_wall": {"id": 2, "prefix": "vessel_wall", "display": "vessel wall"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate IVUS segmentation predictions against ground truth."
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("sample_output"),
        help="Directory containing prediction .npy files.",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("sample_gt_cropped"),
        help="Directory containing GT .npy files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("sample_results.csv"),
        help="Output CSV path for results.",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="identity",
        choices=sorted(LABEL_MAPPINGS),
        help="Label mapping to apply to prediction labels.",
    )
    parser.add_argument(
        "--lumen-only",
        action="store_true",
        help="Evaluate lumen only.",
    )
    parser.add_argument(
        "--vessel-only",
        action="store_true",
        help="Evaluate vessel wall only.",
    )
    args = parser.parse_args()

    if args.lumen_only and args.vessel_only:
        parser.error("--lumen-only and --vessel-only cannot be used together")

    return args


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    denominator = pred.sum() + gt.sum()
    if denominator == 0:
        return 1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0
    return (2.0 * intersection) / denominator


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return intersection / union


def precision_score(pred: np.ndarray, gt: np.ndarray) -> float:
    true_positives = np.logical_and(pred, gt).sum()
    predicted_positives = pred.sum()
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives


def recall_score(pred: np.ndarray, gt: np.ndarray) -> float:
    true_positives = np.logical_and(pred, gt).sum()
    actual_positives = gt.sum()
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives


def compute_metrics_for_class(pred: np.ndarray, gt: np.ndarray, class_id: int) -> dict:
    pred_binary = (pred == class_id).astype(np.uint8)
    gt_binary = (gt == class_id).astype(np.uint8)
    return {
        "dice": dice_score(pred_binary, gt_binary),
        "iou": iou_score(pred_binary, gt_binary),
        "precision": precision_score(pred_binary, gt_binary),
        "recall": recall_score(pred_binary, gt_binary),
    }


def determine_classes_to_evaluate(args) -> list[str]:
    if args.lumen_only:
        return ["lumen"]
    if args.vessel_only:
        return ["vessel_wall"]
    return ["lumen", "vessel_wall"]


def remap_prediction_labels(pred: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    pred_remapped = pred.copy()
    for model_label, gt_label in mapping.items():
        pred_remapped[pred == model_label] = gt_label
    return pred_remapped


def stem_from_prediction_path(pred_path: Path) -> str:
    stem = pred_path.stem.replace("_pred_like_gt_hwf", "")
    return stem.replace("_cropped", "")


def find_matching_files(pred_dir: Path, gt_dir: Path):
    pred_files = sorted(pred_dir.rglob("*_pred_like_gt_hwf.npy"))

    matches = []
    missing_gt = []
    for pred_path in pred_files:
        case_id = pred_path.parent.name
        stem = stem_from_prediction_path(pred_path)
        gt_filename = f"{stem}_label_hwf.npy"
        gt_path = gt_dir / case_id / gt_filename

        if gt_path.exists():
            matches.append((pred_path, gt_path, case_id, stem))
        else:
            missing_gt.append((pred_path, gt_filename))

    return matches, missing_gt


def format_shape(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def build_csv_row(case_id: str, stem: str, pred_path: Path, gt_path: Path, mapping_name: str, shape: tuple[int, ...], selected_classes: list[str], metrics_by_class: dict) -> dict:
    row = {
        "case": case_id,
        "stem": stem,
        "mapping": mapping_name,
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
        "shape": format_shape(shape),
    }

    for class_name, config in CLASS_CONFIG.items():
        prefix = config["prefix"]
        class_metrics = metrics_by_class.get(class_name)
        row[f"{prefix}_dice"] = class_metrics["dice"] if class_metrics else ""
        row[f"{prefix}_iou"] = class_metrics["iou"] if class_metrics else ""
        row[f"{prefix}_precision"] = class_metrics["precision"] if class_metrics else ""
        row[f"{prefix}_recall"] = class_metrics["recall"] if class_metrics else ""

    return row


def write_results_csv(output_csv: Path, rows: list[dict]):
    fieldnames = [
        "case",
        "stem",
        "mapping",
        "pred_path",
        "gt_path",
        "shape",
        "lumen_dice",
        "lumen_iou",
        "lumen_precision",
        "lumen_recall",
        "vessel_wall_dice",
        "vessel_wall_iou",
        "vessel_wall_precision",
        "vessel_wall_recall",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_case_table(rows: list[dict], selected_classes: list[str]):
    if selected_classes == ["lumen"]:
        print("\n" + "=" * 68)
        print(f"{'Case':<8} {'Stem':<15} {'Lumen Dice':>12} {'Lumen IoU':>12} {'Lumen Rec':>12}")
        print("=" * 68)
        for row in rows:
            print(
                f"{row['case']:<8} {row['stem']:<15} {float(row['lumen_dice']):>12.4f} "
                f"{float(row['lumen_iou']):>12.4f} {float(row['lumen_recall']):>12.4f}"
            )
        print("=" * 68)
        return

    if selected_classes == ["vessel_wall"]:
        print("\n" + "=" * 74)
        print(f"{'Case':<8} {'Stem':<15} {'Vessel Dice':>12} {'Vessel IoU':>12} {'Vessel Rec':>12}")
        print("=" * 74)
        for row in rows:
            print(
                f"{row['case']:<8} {row['stem']:<15} {float(row['vessel_wall_dice']):>12.4f} "
                f"{float(row['vessel_wall_iou']):>12.4f} {float(row['vessel_wall_recall']):>12.4f}"
            )
        print("=" * 74)
        return

    print("\n" + "=" * 94)
    print(
        f"{'Case':<8} {'Stem':<15} {'Lumen Dice':>12} {'Lumen IoU':>12} "
        f"{'Vessel Dice':>12} {'Vessel IoU':>12}"
    )
    print("=" * 94)
    for row in rows:
        print(
            f"{row['case']:<8} {row['stem']:<15} {float(row['lumen_dice']):>12.4f} "
            f"{float(row['lumen_iou']):>12.4f} {float(row['vessel_wall_dice']):>12.4f} "
            f"{float(row['vessel_wall_iou']):>12.4f}"
        )
    print("=" * 94)


def print_summary(rows: list[dict], selected_classes: list[str], missing_gt: list, shape_mismatches: list):
    print(f"\nMissing GT files: {len(missing_gt)}")
    print(f"Shape mismatches skipped: {len(shape_mismatches)}")

    for class_name in selected_classes:
        prefix = CLASS_CONFIG[class_name]["prefix"]
        display = CLASS_CONFIG[class_name]["display"]
        dice_values = [float(row[f"{prefix}_dice"]) for row in rows]
        iou_values = [float(row[f"{prefix}_iou"]) for row in rows]
        precision_values = [float(row[f"{prefix}_precision"]) for row in rows]
        recall_values = [float(row[f"{prefix}_recall"]) for row in rows]

        print(f"\nAverage metrics ({display}):")
        print(f"  Dice: {np.mean(dice_values):.4f}")
        print(f"  IoU: {np.mean(iou_values):.4f}")
        print(f"  Precision: {np.mean(precision_values):.4f}")
        print(f"  Recall: {np.mean(recall_values):.4f}")


def main():
    args = parse_args()

    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Prediction dir not found: {args.pred_dir}")
    if not args.gt_dir.exists():
        raise FileNotFoundError(f"GT dir not found: {args.gt_dir}")

    selected_classes = determine_classes_to_evaluate(args)
    label_mapping = LABEL_MAPPINGS[args.mapping]

    print(f"Scanning {args.pred_dir} and {args.gt_dir}...")
    matches, missing_gt = find_matching_files(args.pred_dir, args.gt_dir)
    print(f"Found {len(matches)} matching prediction-GT pairs")
    print(f"Using label mapping: {args.mapping} -> {label_mapping}")
    print(f"Evaluating classes: {', '.join(CLASS_CONFIG[name]['display'] for name in selected_classes)}")

    for pred_path, gt_filename in missing_gt:
        print(f"Warning: No GT found for {pred_path.name} (expected {gt_filename})")

    if not matches:
        print("No matching files found. Exiting.")
        return

    rows = []
    shape_mismatches = []

    for pred_path, gt_path, case_id, stem in matches:
        try:
            pred = np.load(pred_path)
            gt = np.load(gt_path)
        except Exception as exc:
            print(f"Error processing {pred_path.name}: {exc}")
            continue

        if pred.shape != gt.shape:
            shape_mismatches.append((case_id, pred.shape, gt.shape))
            print(f"Shape mismatch for {case_id}: pred {pred.shape} vs gt {gt.shape}")
            continue

        pred_remapped = remap_prediction_labels(pred, label_mapping)
        metrics_by_class = {}
        for class_name in selected_classes:
            class_id = CLASS_CONFIG[class_name]["id"]
            metrics_by_class[class_name] = compute_metrics_for_class(pred_remapped, gt, class_id)

        rows.append(
            build_csv_row(
                case_id=case_id,
                stem=stem,
                pred_path=pred_path,
                gt_path=gt_path,
                mapping_name=args.mapping,
                shape=pred.shape,
                selected_classes=selected_classes,
                metrics_by_class=metrics_by_class,
            )
        )

    if not rows:
        print("No results to save.")
        return

    write_results_csv(args.output_csv, rows)
    print(f"\nResults saved to: {args.output_csv}")
    print_case_table(rows, selected_classes)
    print_summary(rows, selected_classes, missing_gt, shape_mismatches)


if __name__ == "__main__":
    main()
