#!/usr/bin/env python3
"""
DepthWizard Standalone Object Detection Test Harness
===================================================

Isolated from the depth/mesh/calibration pipeline. Depends only on
Ultralytics YOLO and standard-library / Pillow / numpy.

Usage examples
--------------
  # Single image, one checkpoint
  python test_detection_standalone.py --checkpoint depthwizard_v2_obb.pt --image DJI_0009.JPG

  # Compare two checkpoints side-by-side
  python test_detection_standalone.py --checkpoint depthwizard_v2_obb.pt --compare yolov8n-obb.pt --image DJI_0009.JPG

  # Batch a whole directory
  python test_detection_standalone.py --checkpoint depthwizard_v2_obb.pt --image-dir ./sample_data/

  # Check DOTA training-label balance
  python test_detection_standalone.py --checkpoint depthwizard_v2_obb.pt --check-training-balance --labels-dir /path/to/dota/labels/train
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


# ===================================================================
# Path Resolution Helpers
# ===================================================================

def _resolve_file_path(file_path: str, search_subdirs: list = None) -> str:
    """
    Resolve file path flexibly so users can invoke the script from repo root,
    backend/, or backend/tools/ using relative names.
    """
    if not file_path:
        return file_path

    # If it exists as given, return absolute
    if os.path.isfile(file_path):
        return os.path.abspath(file_path)

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(tools_dir, ".."))
    repo_dir = os.path.abspath(os.path.join(backend_dir, ".."))

    candidate_dirs = [
        tools_dir,
        backend_dir,
        os.path.join(backend_dir, "sample_data"),
        repo_dir,
        os.getcwd(),
        os.path.join(os.getcwd(), "backend"),
        os.path.join(os.getcwd(), "backend", "sample_data"),
        os.path.join(os.getcwd(), "sample_data"),
    ]
    if search_subdirs:
        candidate_dirs.extend(search_subdirs)

    for d in candidate_dirs:
        candidate = os.path.join(d, file_path)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    return os.path.abspath(file_path)


def _resolve_dir_path(dir_path: str) -> str:
    """Resolve directory path flexibly."""
    if not dir_path:
        return dir_path
    if os.path.isdir(dir_path):
        return os.path.abspath(dir_path)

    tools_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.abspath(os.path.join(tools_dir, ".."))
    repo_dir = os.path.abspath(os.path.join(backend_dir, ".."))

    candidate_dirs = [
        tools_dir,
        backend_dir,
        os.path.join(backend_dir, "sample_data"),
        repo_dir,
        os.getcwd(),
        os.path.join(os.getcwd(), "backend"),
        os.path.join(os.getcwd(), "backend", "sample_data"),
        os.path.join(os.getcwd(), "sample_data"),
    ]
    for d in candidate_dirs:
        candidate = os.path.join(d, dir_path)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)

    return os.path.abspath(dir_path)


# ===================================================================
# Image Channel & Format Validation
# ===================================================================

def _validate_and_load_image(image_path: str):
    """
    Inspect image channels, mode, and bit depth before inference.
    Prevents PyTorch Conv2D RuntimeError when processing 1-channel rasters.

    Returns:
        (ready_image_or_path, skip_message)
    """
    from PIL import Image
    basename = os.path.basename(image_path)
    try:
        with Image.open(image_path) as img:
            mode = img.mode
            bands = img.getbands()
            num_channels = len(bands)

            # 1. Single-channel (grayscale, float DEM, 1-bit, integer elevation)
            if num_channels == 1 or mode in ("L", "F", "I", "1"):
                return None, (
                    f"Skipping {basename}: single-channel image, not suitable for RGB "
                    f"object detection (likely a DEM/LiDAR reference raster, not an optical photo)"
                )

            # 2. 4-channel RGBA / RGBX - convert defensively to standard 3-channel RGB
            if mode in ("RGBA", "RGBX") or num_channels == 4:
                rgb_img = img.convert("RGB")
                return rgb_img, None

            # 3. Standard 3-channel RGB
            if mode == "RGB" and num_channels == 3:
                return image_path, None

            # 4. Other multi-channel formats (e.g. CMYK, YCbCr, 16-bit multi-band)
            try:
                rgb_img = img.convert("RGB")
                return rgb_img, None
            except Exception as conv_err:
                return None, (
                    f"Skipping {basename}: unsupported image mode '{mode}' with {num_channels} bands ({conv_err})"
                )

    except Exception as e:
        return None, f"Skipping {basename}: unable to open image ({e})"


# ===================================================================
# Model Loading & Inspection
# ===================================================================

def _load_model(checkpoint_path: str):
    """Load an Ultralytics YOLO model and return (model, resolved_path)."""
    from ultralytics import YOLO

    resolved = _resolve_file_path(checkpoint_path)
    if not os.path.isfile(resolved):
        print(f"\nERROR: Checkpoint file not found: {resolved}")
        print(f"Searched provided path '{checkpoint_path}' in current directory, backend/, and tools/.")
        sys.exit(1)

    model = YOLO(resolved)
    return model, resolved


def _print_model_info(model, label: str, resolved_path: str):
    """Print task, path, and class-name mapping for a loaded model."""
    print(f"\n{'─' * 72}")
    print(f"  {label}")
    print(f"{'─' * 72}")
    print(f"  Path : {resolved_path}")
    print(f"  Task : {model.task}")
    if model.task != "obb":
        print(f"  ⚠️  WARNING: model.task is '{model.task}', expected 'obb'!")
    names = model.names or {}
    print(f"  Classes ({len(names)}):")
    formatted_dict = ", ".join(f"{idx}: {names[idx]}" for idx in sorted(names.keys()))
    print(f"    Dict: {{{formatted_dict}}}")
    for idx in sorted(names.keys()):
        print(f"    {idx:>2}: {names[idx]}")
    print()
    return names


def _compare_class_maps(names_a: dict, names_b: dict, label_a: str, label_b: str):
    """Print a side-by-side comparison and flag mismatches."""
    all_indices = sorted(set(names_a.keys()) | set(names_b.keys()))
    mismatches = []

    print(f"\n{'═' * 72}")
    print(f"  CLASS-MAP COMPARISON")
    print(f"{'═' * 72}")
    print(f"  {'Idx':>3}  {'(' + label_a + ')':30s}  {'(' + label_b + ')':30s}  Match?")
    print(f"  {'───':>3}  {'─' * 30}  {'─' * 30}  ──────")
    for idx in all_indices:
        na = names_a.get(idx, "<MISSING>")
        nb = names_b.get(idx, "<MISSING>")
        match = "✅" if na == nb else "❌ MISMATCH"
        if na != nb:
            mismatches.append((idx, na, nb))
        print(f"  {idx:>3}  {na:30s}  {nb:30s}  {match}")

    if mismatches:
        print(f"\n  ⚠️  {len(mismatches)} CLASS-INDEX MISMATCH(ES) FOUND:")
        for idx, na, nb in mismatches:
            print(f"     Index {idx}: '{na}' vs '{nb}'")
        print(f"     This is the primary diagnostic signal for a mapping/head mismatch.")
    else:
        print(f"\n  ✅ All {len(all_indices)} class indices match between checkpoints.")
    print()


# ===================================================================
# Inference & Visualization
# ===================================================================

def _run_inference(model, image_input, conf: float, output_dir: str,
                   tag: str = "", imgsz: int = 1024, orig_filename: str = "") -> list:
    """
    Run OBB inference on a single image (file path or PIL Image).
    Returns list of dicts: [{class_name, class_idx, confidence, coords}, ...]
    Saves annotated image to output_dir/<tag>.jpg
    """
    from PIL import Image

    # Run prediction directly
    results = model.predict(
        source=image_input,
        conf=conf,
        imgsz=imgsz,
        save=False,
        verbose=False,
    )

    detections = []
    names = model.names or {}

    if results and len(results) > 0:
        res = results[0]

        # Save annotated image directly to output_dir
        if output_dir:
            stem = Path(orig_filename).stem if orig_filename else "detect"
            out_filename = f"{tag}.jpg" if tag else f"{stem}_detected.jpg"
            out_path = os.path.join(output_dir, out_filename)
            try:
                res.save(filename=out_path)
            except Exception:
                # Fallback manual plot save
                plot_img = res.plot()
                Image.fromarray(plot_img[..., ::-1]).save(out_path)

        # OBB model
        if hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
            for i in range(len(res.obb)):
                cls_id = int(res.obb.cls[i].item())
                conf_val = float(res.obb.conf[i].item())
                # xyxyxyxy (4-corner coords: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
                if hasattr(res.obb, "xyxyxyxy"):
                    raw_coords = res.obb.xyxyxyxy[i].cpu().numpy().tolist()
                    coords = [[round(pt[0], 1), round(pt[1], 1)] for pt in raw_coords]
                elif hasattr(res.obb, "xywhr"):
                    raw_coords = res.obb.xywhr[i].cpu().numpy().tolist()
                    coords = [round(c, 2) for c in raw_coords]
                else:
                    coords = []
                detections.append({
                    "class_name": names.get(cls_id, f"class_{cls_id}"),
                    "class_idx": cls_id,
                    "confidence": round(conf_val, 4),
                    "coords": coords,
                })
        # Fallback: AABB model
        elif hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            for i in range(len(res.boxes)):
                cls_id = int(res.boxes.cls[i].item())
                conf_val = float(res.boxes.conf[i].item())
                bbox = res.boxes.xyxy[i].cpu().numpy().tolist()
                detections.append({
                    "class_name": names.get(cls_id, f"class_{cls_id}"),
                    "class_idx": cls_id,
                    "confidence": round(conf_val, 4),
                    "coords": [round(c, 1) for c in bbox],
                })

    return detections


def _print_detections(detections: list, image_path: str, label: str = ""):
    """Pretty-print detection results for one image."""
    basename = os.path.basename(image_path)
    header = f"  {label + ' — ' if label else ''}{basename}: {len(detections)} detection(s)"
    print(header)

    if not detections:
        print("    (none)")
        return

    for i, d in enumerate(detections):
        coords_str = str(d["coords"])
        print(
            f"    [{i}] cls={d['class_idx']:>2} ({d['class_name']:20s})  "
            f"conf={d['confidence']:.4f}  coords={coords_str}"
        )


def _print_summary_table(class_counter: Counter, label: str):
    """Print aggregated per-class detection counts."""
    print(f"\n{'═' * 72}")
    print(f"  AGGREGATE DETECTION SUMMARY — {label}")
    print(f"{'═' * 72}")
    total = sum(class_counter.values())
    if total == 0:
        print("  No detections across all images.")
        return

    print(f"  {'Class':25s}  {'Count':>6}  {'%':>6}  Bar")
    print(f"  {'─' * 25}  {'─' * 6}  {'─' * 6}  {'─' * 30}")

    max_count = max(class_counter.values()) if class_counter else 1
    for cls_name, count in class_counter.most_common():
        pct = 100.0 * count / total
        bar_len = int(30 * count / max_count)
        bar = "█" * bar_len
        flag = "  ⚠️ DOMINANT" if pct > 60 and len(class_counter) > 1 else ""
        print(f"  {cls_name:25s}  {count:>6}  {pct:>5.1f}%  {bar}{flag}")

    print(f"  {'─' * 25}  {'─' * 6}")
    print(f"  {'TOTAL':25s}  {total:>6}")
    print()


# ===================================================================
# Training Label Balance Diagnostic
# ===================================================================

def _check_training_balance(labels_dir: str, class_names: dict):
    """Scan DOTA-format or YOLO-format label files and print per-class instance counts."""
    resolved_dir = _resolve_dir_path(labels_dir)
    print(f"\n{'═' * 72}")
    print(f"  TRAINING LABEL BALANCE CHECK")
    print(f"  Labels dir: {resolved_dir}")
    print(f"{'═' * 72}")

    if not os.path.isdir(resolved_dir):
        print(f"  ERROR: Directory not found: {resolved_dir}")
        return

    dota_label_aliases = {
        "plane": "plane",
        "ship": "ship",
        "storage-tank": "storage tank",
        "storage tank": "storage tank",
        "baseball-diamond": "baseball diamond",
        "baseball diamond": "baseball diamond",
        "tennis-court": "tennis court",
        "tennis court": "tennis court",
        "basketball-court": "basketball court",
        "basketball court": "basketball court",
        "ground-track-field": "ground track field",
        "ground track field": "ground track field",
        "harbor": "harbor",
        "bridge": "bridge",
        "large-vehicle": "large vehicle",
        "large vehicle": "large vehicle",
        "small-vehicle": "small vehicle",
        "small vehicle": "small vehicle",
        "helicopter": "helicopter",
        "roundabout": "roundabout",
        "soccer-ball-field": "soccer ball field",
        "soccer ball field": "soccer ball field",
        "swimming-pool": "swimming pool",
        "swimming pool": "swimming pool",
    }

    class_counter = Counter()
    total_files = 0
    total_instances = 0

    label_files = []
    for root, _, files in os.walk(resolved_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                label_files.append(os.path.join(root, f))

    for lf in label_files:
        total_files += 1
        with open(lf, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue  # skip empty or malformed lines

                canonical = None
                # YOLO format: class_idx x1 y1 x2 y2 x3 y3 x4 y4 (or class_idx x y w h r)
                if parts[0].isdigit() and int(parts[0]) in class_names:
                    cls_id = int(parts[0])
                    canonical = class_names[cls_id]
                # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 category [difficulty]
                elif len(parts) >= 9:
                    category = parts[8].lower()
                    canonical = dota_label_aliases.get(category, category)

                if canonical:
                    class_counter[canonical] += 1
                    total_instances += 1

    print(f"  Scanned {total_files} label file(s), {total_instances} total instance(s)\n")

    if total_instances == 0:
        print("  No instances found. Check label format (expected DOTA or YOLO format).")
        return

    print(f"  {'Class':25s}  {'Count':>8}  {'%':>6}  Bar")
    print(f"  {'─' * 25}  {'─' * 8}  {'─' * 6}  {'─' * 30}")

    max_count = max(class_counter.values()) if class_counter else 1
    for cls_name, count in class_counter.most_common():
        pct = 100.0 * count / total_instances
        bar_len = int(30 * count / max_count)
        bar = "█" * bar_len
        print(f"  {cls_name:25s}  {count:>8}  {pct:>5.1f}%  {bar}")

    print(f"  {'─' * 25}  {'─' * 8}")
    print(f"  {'TOTAL':25s}  {total_instances:>8}")

    # Flag extreme imbalances
    if len(class_counter) > 1:
        most_common_name, most_common_count = class_counter.most_common(1)[0]
        least_common_name, least_common_count = class_counter.most_common()[-1]
        ratio = most_common_count / max(least_common_count, 1)
        if ratio > 10:
            print(f"\n  ⚠️  SEVERE CLASS IMBALANCE: '{most_common_name}' has {ratio:.0f}x "
                  f"more instances than '{least_common_name}'")
            print(f"     This can cause class-collapse in fine-tuning.")
    print()


# ===================================================================
# Main CLI Entry Point
# ===================================================================

def main():
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt = os.path.abspath(os.path.join(tools_dir, "..", "depthwizard_v2_obb.pt"))
    if not os.path.isfile(default_ckpt):
        default_ckpt = "depthwizard_v2_obb.pt"

    parser = argparse.ArgumentParser(
        description="DepthWizard Standalone YOLO-OBB Detection Test Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", type=str, default=default_ckpt,
                        help=f"Path to primary .pt checkpoint (default: {default_ckpt})")
    parser.add_argument("--compare", type=str, default=None,
                        help="Optional second .pt checkpoint for side-by-side comparison")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single test image")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Path to a directory of test images")
    default_conf = float(os.environ.get("DEPTHWIZARD_YOLO_CONF", 0.25))
    parser.add_argument("--conf", type=float, default=default_conf,
                        help=f"Confidence threshold (default: {default_conf})")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Inference image size (default: 1024)")
    parser.add_argument("--check-training-balance", action="store_true",
                        help="Scan DOTA label files for class balance")
    parser.add_argument("--labels-dir", type=str, default=None,
                        help="Path to DOTA labels directory (for --check-training-balance)")

    args = parser.parse_args()

    output_dir = os.path.join(tools_dir, "test_outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("  DepthWizard Standalone Detection Test Harness")
    print("=" * 72)
    print(f"  Confidence threshold : {args.conf}")
    print(f"  Inference image size : {args.imgsz}")
    print(f"  Output directory     : {output_dir}")

    # ── 1. Load primary checkpoint ────────────────────────────────
    model_a, path_a = _load_model(args.checkpoint)
    names_a = _print_model_info(model_a, "PRIMARY CHECKPOINT", path_a)

    # ── 2. Load comparison checkpoint (optional) ──────────────────
    model_b, path_b, names_b = None, None, None
    if args.compare:
        model_b, path_b = _load_model(args.compare)
        names_b = _print_model_info(model_b, "COMPARISON CHECKPOINT", path_b)
        _compare_class_maps(names_a, names_b, os.path.basename(path_a), os.path.basename(path_b))

    # ── 3. Training balance check ─────────────────────────────────
    if args.check_training_balance:
        if not args.labels_dir:
            print("ERROR: --check-training-balance requires --labels-dir")
            sys.exit(1)
        _check_training_balance(args.labels_dir, names_a)

    # ── 4. Collect image paths ────────────────────────────────────
    image_paths = []
    if args.image:
        resolved_img = _resolve_file_path(args.image)
        if not os.path.isfile(resolved_img):
            print(f"ERROR: Image not found: {resolved_img}")
            print(f"Searched '{args.image}' in current directory, backend/sample_data/, and tools/.")
            sys.exit(1)
        image_paths.append(resolved_img)
    elif args.image_dir:
        resolved_dir = _resolve_dir_path(args.image_dir)
        if not os.path.isdir(resolved_dir):
            print(f"ERROR: Directory not found: {resolved_dir}")
            sys.exit(1)
        for f in sorted(os.listdir(resolved_dir)):
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(resolved_dir, f))
        print(f"\n  Found {len(image_paths)} image candidate(s) in {resolved_dir}")

    if not image_paths and not args.check_training_balance:
        print("\nNo images provided. Use --image or --image-dir.")
        sys.exit(0)

    if not image_paths:
        print("\nDone (no inference requested).")
        return

    # ── 5. Run inference ──────────────────────────────────────────
    counter_a = Counter()
    counter_b = Counter()
    processed_count = 0
    skipped_count = 0

    for img_path in image_paths:
        basename = os.path.basename(img_path)
        print(f"\n{'─' * 72}")
        print(f"  IMAGE: {basename}")
        print(f"  Path : {img_path}")
        print(f"{'─' * 72}")

        # Validate channels & format
        ready_img, skip_reason = _validate_and_load_image(img_path)
        if skip_reason:
            print(f"  ℹ️  {skip_reason}")
            skipped_count += 1
            continue

        try:
            img_stem = Path(img_path).stem

            # Primary checkpoint
            tag_a = f"primary_{img_stem}"
            dets_a = _run_inference(
                model_a, ready_img, args.conf, output_dir,
                tag=tag_a, imgsz=args.imgsz, orig_filename=basename
            )
            _print_detections(dets_a, img_path, label=os.path.basename(path_a))
            out_path_a = os.path.join(output_dir, f"{tag_a}.jpg")
            if os.path.isfile(out_path_a):
                print(f"    Annotated output: {out_path_a}")
            for d in dets_a:
                counter_a[d["class_name"]] += 1

            # Comparison checkpoint (if requested)
            if model_b is not None:
                tag_b = f"compare_{img_stem}"
                dets_b = _run_inference(
                    model_b, ready_img, args.conf, output_dir,
                    tag=tag_b, imgsz=args.imgsz, orig_filename=basename
                )
                _print_detections(dets_b, img_path, label=os.path.basename(path_b))
                out_path_b = os.path.join(output_dir, f"{tag_b}.jpg")
                if os.path.isfile(out_path_b):
                    print(f"    Annotated output: {out_path_b}")
                for d in dets_b:
                    counter_b[d["class_name"]] += 1

                # Side-by-side summary for this image
                def _fmt_dets(dets):
                    if not dets:
                        return "(none)"
                    return ", ".join(f"{d['class_name']}@{d['confidence']:.3f}" for d in dets)

                print(f"\n  Side-by-side for {basename}:")
                summary_a = _fmt_dets(dets_a)
                summary_b = _fmt_dets(dets_b)
                print(f"    {os.path.basename(path_a):30s}  →  {len(dets_a)} det(s): {summary_a}")
                print(f"    {os.path.basename(path_b):30s}  →  {len(dets_b)} det(s): {summary_b}")

            processed_count += 1

        except Exception as exc:
            print(f"  ❌ Error processing {basename}: {exc}")
            skipped_count += 1
            continue

    # ── 6. Aggregate summary table ────────────────────────────────
    print(f"\nBatch summary: {processed_count} image(s) processed, {skipped_count} skipped.")
    _print_summary_table(counter_a, os.path.basename(path_a))
    if model_b is not None:
        _print_summary_table(counter_b, os.path.basename(path_b))

    print(f"\n{'═' * 72}")
    print(f"  Annotated images saved to: {output_dir}")
    print(f"{'═' * 72}")
    print("Done.")


if __name__ == "__main__":
    main()
