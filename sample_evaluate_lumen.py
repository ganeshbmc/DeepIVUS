#!/usr/bin/env python
"""
Evaluation script for IVUS lumen segmentation.

Computes Dice, IoU, Precision, and Recall for lumen (class 1) only.

Output: Console summary + sample_results_lumen.csv
"""

import argparse
from pathlib import Path

import numpy as np


# Only evaluate lumen (class 1)
CLASS_TO_EVALUATE = 1

LABEL_NAME = 'lumen'


# Label mapping options:
#   identity: {0: 0, 1: 1, 2: 2, 3: 0} - current (no change)
#   swap_1_2: {0: 0, 1: 2, 2: 1, 3: 0} - swap lumen and vessel
LABEL_MAPPINGS = {
    'identity': {0: 0, 1: 1, 2: 2, 3: 0},
    'swap_1_2': {0: 0, 1: 2, 2: 1, 3: 0},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate IVUS lumen segmentation predictions against ground truth."
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
        default=Path("sample_results_lumen.csv"),
        help="Output CSV path for results.",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default='identity',
        choices=['identity', 'swap_1_2'],
        help="Label mapping to apply (identity or swap_1_2).",
    )
    return parser.parse_args()


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient."""
    intersection = np.logical_and(pred, gt).sum()
    denominator = pred.sum() + gt.sum()
    if denominator == 0:
        return 1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0
    return (2.0 * intersection) / denominator


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU / Jaccard index."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return intersection / union


def precision_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Precision."""
    true_positives = np.logical_and(pred, gt).sum()
    predicted_positives = pred.sum()
    if predicted_positives == 0:
        return 0.0
    return true_positives / predicted_positives


def recall_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Recall / Sensitivity."""
    true_positives = np.logical_and(pred, gt).sum()
    actual_positives = gt.sum()
    if actual_positives == 0:
        return 0.0
    return true_positives / actual_positives


def compute_metrics(pred: np.ndarray, gt: np.ndarray, class_id: int) -> dict:
    """Compute metrics for a class."""
    pred_binary = (pred == class_id).astype(np.uint8)
    gt_binary = (gt == class_id).astype(np.uint8)

    return {
        'dice': dice_score(pred_binary, gt_binary),
        'iou': iou_score(pred_binary, gt_binary),
        'precision': precision_score(pred_binary, gt_binary),
        'recall': recall_score(pred_binary, gt_binary),
    }


def find_matching_files(pred_dir: Path, gt_dir: Path):
    """Find matching prediction and GT file pairs."""
    pred_files = sorted(pred_dir.rglob("*_pred_like_gt_hwf.npy"))
    
    matches = []
    for pred_path in pred_files:
        case_id = pred_path.parent.name
        stem = pred_path.stem.replace("_pred_like_gt_hwf", "")
        stem = stem.replace("_cropped", "")
        
        gt_filename = f"{stem}_label_hwf.npy"
        gt_path = gt_dir / case_id / gt_filename
        
        if gt_path.exists():
            matches.append((pred_path, gt_path, case_id, stem))
        else:
            print(f"Warning: No GT found for {pred_path.name} (expected {gt_filename})")
    
    return matches


def main():
    args = parse_args()
    
    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Prediction dir not found: {args.pred_dir}")
    if not args.gt_dir.exists():
        raise FileNotFoundError(f"GT dir not found: {args.gt_dir}")
    
    print(f"Scanning {args.pred_dir} and {args.gt_dir}...")
    matches = find_matching_files(args.pred_dir, args.gt_dir)
    print(f"Found {len(matches)} matching prediction-GT pairs")
    
    if not matches:
        print("No matching files found. Exiting.")
        return
    
    # Get the label mapping
    MODEL_TO_GT_MAPPING = LABEL_MAPPINGS[args.mapping]
    print(f"Using label mapping: {args.mapping} -> {MODEL_TO_GT_MAPPING}")
    
    results = []
    
    for pred_path, gt_path, case_id, stem in matches:
        try:
            pred = np.load(pred_path)
            gt = np.load(gt_path)
            
            if pred.shape != gt.shape:
                print(f"Shape mismatch for {case_id}: pred {pred.shape} vs gt {gt.shape}")
                continue
            
            # Apply label mapping: remap model labels to GT labels
            pred_remapped = pred.copy()
            for model_label, gt_label in MODEL_TO_GT_MAPPING.items():
                pred_remapped[pred == model_label] = gt_label
            
            # Evaluate only lumen (class 1)
            metrics = compute_metrics(pred_remapped, gt, CLASS_TO_EVALUATE)
            
            results.append({
                'case': case_id,
                'stem': stem,
                'dice': metrics['dice'],
                'iou': metrics['iou'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
            })
        
        except Exception as e:
            print(f"Error processing {pred_path.name}: {e}")
    
    # Compute summary statistics
    if results:
        import csv
        
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = ['case', 'stem', 'dice', 'iou', 'precision', 'recall']
        
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to: {args.output_csv}")
        
        # Print summary table
        print("\n" + "=" * 60)
        print(f"{'Case':<8} {'Stem':<15} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Rec':>8}")
        print("=" * 60)
        
        for r in results:
            print(f"{r['case']:<8} {r['stem']:<15} {r['dice']:>8.4f} {r['iou']:>8.4f} "
                  f"{r['precision']:>8.4f} {r['recall']:>8.4f}")
        
        print("=" * 60)
        
        # Overall average
        all_dice = [r['dice'] for r in results]
        all_iou = [r['iou'] for r in results]
        all_prec = [r['precision'] for r in results]
        all_rec = [r['recall'] for r in results]
        
        print(f"\nOverall average ({LABEL_NAME}):")
        print(f"  Dice: {np.mean(all_dice):.4f}")
        print(f"  IoU: {np.mean(all_iou):.4f}")
        print(f"  Precision: {np.mean(all_prec):.4f}")
        print(f"  Recall: {np.mean(all_rec):.4f}")
    
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
