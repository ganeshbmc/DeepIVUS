import shutil
from pathlib import Path

# Update source and destination paths as needed
src_root = Path("/mnt/e/dishas_thesis_samplevolumes_and_labels/OUTPUT_AI_IVUS")
dst_root = Path("/home/ganeshbmc/github/DeepIVUS/sample_gt_uncropped")
dst_root.mkdir(parents=True, exist_ok=True)

copied = 0
missing = 0

for i in range(1, 101):
    case = f"{i:03d}"
    src_case = src_root / case
    
    if not src_case.exists():
        print(f"Missing folder: {case}")
        missing += 1
        continue
    
    # Check if case has VL subfolders (multi-run)
    vl_subdirs = [d.name for d in src_case.iterdir() if d.is_dir() and d.name.startswith("VL")]
    
    if vl_subdirs:
        # Multi-run case: copy each VL's Segmentation.seg.nrrd
        for vl_name in sorted(vl_subdirs):
            src_file = src_case / vl_name / "Segmentation.seg.nrrd"
            if src_file.exists():
                dst_file = dst_root / case / f"{case}_{vl_name}_seg.nrrd"
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                print(f"Copied: {case}/{vl_name}")
                copied += 1
            else:
                print(f"Missing: {case}/{vl_name}/Segmentation.seg.nrrd")
                missing += 1
    else:
        # Single-run case: copy directly if exists
        src_file = src_case / "Segmentation.seg.nrrd"
        if src_file.exists():
            dst_file = dst_root / case / f"{case}_seg.nrrd"
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {case}")
            copied += 1
        else:
            print(f"Missing: {case}/Segmentation.seg.nrrd")
            missing += 1

print(f"\nDone. Copied: {copied}, Missing: {missing}")