**GT Conversion Approach**

The core fix is to stop interpreting raw NRRDs by array index alone. `convert_gt_cropped.py` currently does that in `merge_segments_to_multiclass()`, and that breaks on shared-layer and reordered files.

What the converter should do instead:

1. Read segment metadata from the NRRD header for every `Segment{i}`.
2. Canonicalize each segment to one of `lumen`, `vessel_ob`, `plaque` using the segment name first, with typo-tolerant matching.
3. Reconstruct each semantic mask from `Segment{i}_Layer` and `Segment{i}_LabelValue`, not from `seg_volume[i]` blindly.
4. Crop frames using the existing Excel frame ranges.
5. Build final labels as:
   - `1` where `lumen_mask`
   - `2` where `vessel_ob_mask AND NOT lumen_mask`
   - `0` everywhere else
6. Use plaque only as a QA/input signal, not as a final class.
   - Plaque inside lumen becomes `1` automatically because lumen wins.
   - Plaque inside vessel wall becomes `2` automatically because vessel wall is `vessel_ob - lumen`.
   - Plaque outside both becomes `0`.

**Why this works**

It covers all raw storage variants I found:

- Normal 4D files: 3 separate layers.
- Shared-layer files: plaque stored in the same underlying labelmap layer as lumen or vessel, distinguished by `LabelValue`.
- Reordered files: segment order is not reliable.
- Typo cases: `vesel_ob`, `veseel_ob`, `vesssel_ob`, `plauqe`, `palque`.
- One merged 3D labelmap case: `002`, where all segments are encoded in a single volume with label values `1/2/3`.

**Concrete reconstruction rule**

For each segment entry in the header:

- If NRRD is 4D (`dimension=4`, `kinds[0]=list`):
  - Segment mask = `data[layer_index] == label_value`
- If NRRD is 3D (`dimension=3`):
  - Segment mask = `data == label_value`

This is the key fix. It correctly handles cases like:

- `003`, `049`, `072`, `084`: plaque shares layer 0 with lumen, using `LabelValue=2`
- `052`: vessel is layer 0 value 2, lumen is layer 1 value 1, plaque is layer 1 value 2
- `004`: segments are reordered entirely
- `002`: all three segments are in one merged 3D labelmap

**What I found in the raw data**

- `101` raw `_seg.nrrd` files total.
- `100` are 4D segmented labelmaps.
- `1` is a 3D merged labelmap: `sample_gt_uncropped/002/002_seg.nrrd`.

Most common header patterns:

- `44` files: separate layers `lumen`, `vessel_ob`, `plaque`
- `44` files: plaque shares lumen’s layer via a different `LabelValue`
- Several typo/name anomalies
- Outliers:
  - `004`: reordered names and labels
  - `052`: vessel/lumen/plaque arranged differently
  - `084`: vessel named `Segment_1`
  - `002`: merged 3D labelmap

**Recommended validation checks**

After reconstruction, the converter should also report per file:

- whether all 3 semantic segments were identified
- lumen pixels outside vessel
- plaque pixels outside vessel
- plaque overlap with lumen
- plaque overlap with vessel wall
- unknown/unmapped segment names

That gives a QC report without blocking conversion.

**Recommended implementation plan**

1. Replace index-based merge logic with header-driven segment reconstruction.
2. Add name normalization for common typos and `Segment_1`.
3. Build lumen/vessel/plaque masks from `Layer + LabelValue`.
4. Generate final cropped labelmaps with `lumen=1`, `vessel wall=2`, `background=0`.
5. Emit a conversion/QC report so anomalous files are visible.
6. Re-run full conversion for all raw files into `sample_gt_cropped/`.
7. Verify file count, shapes, and a few known anomalies (`002`, `003`, `004`, `052`, `084`).

**Recommendation**
Auto-convert anomalous files when they can be reconstructed from metadata, and only skip a file if a required semantic mask cannot be identified at all.

One question before implementation: for ambiguous files like `084` where `Segment_1` is unnamed, do you want me to treat it as `vessel_ob` automatically with a warning? That is my recommended behavior.
