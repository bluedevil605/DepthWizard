# 🔍 DIAGNOSTIC MODE: Isolating Duplicate Detection Root Cause

**Created:** 2026-09-08  
**Issue:** DJI_0009.JPG shows 15 "Building" detections for a scene with only 2-3 actual buildings, with heavy visual overlap.

## 🎯 Objective

Before attempting another fix, we need to isolate the **actual root cause** of duplicate building detections. The problem could be:

1. **Tiling/Merge Issue:** Cross-tile NMS not working correctly, causing tile-boundary duplicates
2. **Base YOLO NMS Issue:** Ultralytics' built-in NMS insufficient or misconfigured (iou=0.45 too high, agnostic_nms not enabled)
3. **Land-Cover Building Extraction:** Multiple overlapping boxes generated per rooftop from the built-up mask

---

## 🛠️ Diagnostic Features Added

### Query Parameters

Two new diagnostic query parameters have been added to the `/api/upload` endpoint:

#### 1. `disable_tiling=true`
**Purpose:** Runs YOLO inference on the full image in a single pass (no tiling)

**What it tells us:**
- If **single-pass ALSO produces ~15 buildings** → Problem is NOT tiling-related
  - Likely: Ultralytics' internal NMS with iou=0.45 is too lenient
  - Likely: Land-cover building extraction generating multiple overlapping boxes per rooftop
- If **single-pass produces 2-3 buildings** but tiled produces 15 → Problem IS tiling-related
  - The previous cross-tile merge fix isn't being called or isn't working correctly

#### 2. `diagnostic_mode=true`
**Purpose:** Enables extra verbose logging with detailed class breakdowns and per-tile statistics

**What it shows:**
- Per-tile detection counts
- Raw detections BEFORE cross-tile merge
- Merged detections AFTER cross-tile merge
- Final detections AFTER size sanity filters
- Class-by-class breakdown at each stage

---

## 📝 Testing Instructions

### STEP 1: Test Single-Pass (No Tiling)

Upload DJI_0009.JPG with tiling disabled:

```bash
# Using curl (adjust path to your image)
curl -X POST "http://localhost:8000/api/upload?disable_tiling=true&diagnostic_mode=true" \
  -F "file=@DJI_0009.JPG" \
  -F "vertical_exaggeration=1.0"
```

**Or via browser/Postman:**
```
POST http://localhost:8000/api/upload?disable_tiling=true&diagnostic_mode=true
```

**Check the backend logs for:**
- Raw detections from YOLO (after Ultralytics NMS)
- Building count in class breakdown
- Final count after size filter

**Record these numbers:**
- Raw detections from YOLO: **X detections**
- Building count: **Y buildings**

---

### STEP 2: Test Tiled Mode (Normal)

Upload DJI_0009.JPG with tiling enabled (default):

```bash
curl -X POST "http://localhost:8000/api/upload?disable_tiling=false&diagnostic_mode=true" \
  -F "file=@DJI_0009.JPG"
```

**Check the backend logs for:**
- Number of tiles generated
- Raw detections before merge
- Merged detections after WBF
- Final detections after size filter

**Record these numbers:**
- Number of tiles: **N**
- Raw detections (before merge): **M detections**, **B_raw buildings**
- Merged detections (after WBF): **P detections**, **B_merged buildings**
- Final detections: **Q detections**, **B_final buildings**

---

## 🔬 Diagnosis Decision Tree

### Case A: Single-Pass Shows 2-3 Buildings, Tiled Shows ~15

**Root Cause:** Tiling/merge problem

**Next Steps:**
1. Check if merge function is being called (look for logs)
2. Check merge effectiveness (reduction percentage)
3. Try lowering cross_tile_iou threshold for buildings

### Case B: Single-Pass ALSO Shows ~15 Buildings

**Root Cause:** Base YOLO NMS or land-cover extraction problem

**Sub-cases:**
- If YOLO raw output shows 15 buildings: Ultralytics NMS insufficient
- If land-cover extraction adds 12+ buildings: Building footprint extraction issue

**Next Steps:**
1. Check detection source breakdown in logs: `[DETECTION MERGE] aerial=X, coco=Y, buildings_from_mask=Z`
2. If aerial source has 15: Add explicit NMS with lower threshold
3. If buildings_from_mask has 15: Review land_cover.py extraction logic

---

## 🚀 Quick Test

```bash
# Single-pass test
curl -X POST "http://localhost:8000/api/upload?disable_tiling=true&diagnostic_mode=true" \
  -F "file=@DJI_0009.JPG"

# Tiled test  
curl -X POST "http://localhost:8000/api/upload?disable_tiling=false&diagnostic_mode=true" \
  -F "file=@DJI_0009.JPG"
```

Compare the building counts in backend logs between both runs.
