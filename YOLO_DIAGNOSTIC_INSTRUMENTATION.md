# YOLO Duplicate Detection - Diagnostic Instrumentation

## Purpose
Add comprehensive diagnostic logging to isolate the root cause of duplicate detections (15 "Building" labels for 2-3 actual houses in DJI_0009.JPG).

## Changes Made

### 1. Single-Pass Inference Mode (`disable_tiling=True`)
**File:** `backend/app/semantic/detector.py` (lines 546-644)

Added diagnostic mode that runs YOLO on the full image in one pass (no tiling) to isolate whether duplicates come from tiling or base YOLO NMS.

**Activation:** Set `diagnostic_disable_tiling = True` in `backend/app/main.py` line 320

**Logs Generated:**
```
🔍 DIAGNOSTIC MODE: disable_tiling=True - Running single-pass inference
   Image: 4000x3000, Model: OBB aerial, Conf: 0.35
🔍 SINGLE-PASS RAW OUTPUT: X detection(s) from YOLO
   (This is AFTER Ultralytics' built-in NMS, BEFORE any custom merge)
🔍 SINGLE-PASS FINAL: X detection(s)
   Class breakdown: {'building': Y, ...}
🔍 DIAGNOSTIC RESULT: If this count is ~2-3 buildings but tiled mode shows ~15,
   the problem is in TILING/MERGE. If this ALSO shows ~15 buildings, the problem
   is that Ultralytics' built-in NMS is insufficient or disabled.
```

### 2. Per-Tile Detection Counts (Tiled Mode)
**File:** `backend/app/semantic/detector.py` (lines 658-760)

Tracks how many detections each tile produces:

**Logs Generated:**
```
🔍 TILED RAW OUTPUT: X total raw detections from N tiles
   Per-tile counts: [3, 5, 2, 4, ...]
   (These are AFTER Ultralytics' built-in per-tile NMS, BEFORE cross-tile merge)
   Raw class breakdown: {'building': Y, 'small vehicle': Z, ...}
```

### 3. Merge Function Call Tracing
**Files:** `backend/app/semantic/detector.py` (lines 758-781, 182-220, 242-276)

Confirms merge functions are actually being called and logs input/output counts:

**Logs Generated:**
```
🔍 CALLING MERGE FUNCTION: WBF with IoU=0.25
🔍 _apply_wbf() CALLED with X input detections, IoU=0.25
🔍 _apply_wbf() RETURNING Y fused detections (from X inputs)
🔍 MERGE FUNCTION RETURNED: Y detections (X → Y)
```

### 4. Final Results Summary
**File:** `backend/app/semantic/detector.py` (lines 820-832)

Enhanced logging with class breakdown:

**Logs Generated:**
```
================================================================================
🔍 TILED INFERENCE COMPLETE (WBF, IoU=0.25):
   47 raw → 15 merged (68% reduction, ratio=3.1x) → 15 final
   Final class breakdown: {'building': 15, 'small vehicle': 2}
================================================================================
```

### 5. Ultralytics NMS Parameters
**Note:** Added documentation that Ultralytics YOLO applies internal NMS with default `iou=0.45`. No `agnostic_nms` parameter is being set, which may allow same-class boxes to pass through if they meet the IoU threshold.

## Testing Instructions

### STEP 1: Run Single-Pass Mode

1. **Enable diagnostic mode** in `backend/app/main.py` line 320:
```python
diagnostic_disable_tiling = True  # Set to True
```

2. **Restart backend** (if auto-reload doesn't pick it up):
```bash
cd backend
# Ctrl+C to stop if running
uvicorn app.main:app --reload --port 8000
```

3. **Upload DJI_0009.JPG** through frontend

4. **Check backend logs** for:
```
🔍 SINGLE-PASS FINAL: X detection(s)
   Class breakdown: {'building': Y, ...}
```

**Expected Outcomes:**

**Case A:** Single-pass shows **2-3 buildings** → Tiling/merge is the problem
- Previous WBF fix isn't working or isn't being called
- Check for "🔍 CALLING MERGE FUNCTION" log (if missing, fix not wired in)

**Case B:** Single-pass ALSO shows **~15 buildings** → Base YOLO NMS is the problem
- Ultralytics' internal NMS with IoU=0.45 is too lenient
- May need to add `agnostic_nms=True` or lower `iou` parameter
- Land-cover building extraction may be generating multiple overlapping boxes per rooftop

### STEP 2: Analyze Tiled Mode (If Case A)

1. **Disable diagnostic mode** in `backend/app/main.py` line 320:
```python
diagnostic_disable_tiling = False  # Set to False
```

2. **Upload same image** and check logs for:
```
🔍 TILED RAW OUTPUT: X total raw detections from N tiles
   Per-tile counts: [...]
🔍 CALLING MERGE FUNCTION: WBF with IoU=0.25
🔍 _apply_wbf() CALLED with X input detections
🔍 _apply_wbf() RETURNING Y fused detections
```

**If merge function logs are MISSING:**
- Fix isn't wired into request flow
- Check that updated `detect_objects()` is being called

**If merge function logs are PRESENT but ratio is low (e.g., 47→38):**
- WBF/NMS isn't merging effectively
- May need to inspect actual bounding box coordinates for overlap
- IoU calculation may be incorrect

### STEP 3: Compare Results

Log both numbers here:
- **Single-pass building count:** ___ 
- **Tiled final building count:** ___
- **Merge ratio (if tiled):** ___x

## Files Modified

1. **`backend/app/semantic/detector.py`**
   - Added `disable_tiling` parameter
   - Added single-pass inference branch (lines 546-644)
   - Added per-tile count tracking (lines 658-740)
   - Added merge function tracing (lines 182-220, 242-276, 758-781)
   - Enhanced final logging (lines 820-832)

2. **`backend/app/main.py`**
   - Added `diagnostic_disable_tiling` flag (line 320)
   - Passed flag to `detect_objects()` call

## Next Steps (After Diagnostics)

**If Case A (tiling/merge problem):**
- Verify merge function is in call stack
- Check IoU calculation logic
- Inspect actual bbox coordinates for proper translation

**If Case B (base NMS problem):**
- Test with `agnostic_nms=True` in YOLO call
- Test with lower `iou` parameter (e.g., 0.30)
- Review land-cover building extraction for duplicate generation

---

**Status:** ✅ Diagnostic instrumentation deployed  
**Date:** 2026-09-07 15:11 UTC  
**Action Required:** Set `diagnostic_disable_tiling=True`, upload DJI_0009.JPG, report log results
