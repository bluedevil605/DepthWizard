# DIAGNOSTIC INSTRUMENTATION - READY FOR TESTING

## Summary

Comprehensive diagnostic logging has been added to isolate the root cause of duplicate detections (15 "Building" labels for 2-3 actual houses).

## Quick Start - STEP 1: Test Single-Pass Mode

### Enable Diagnostic Mode

**File:** `backend/app/main.py` line 319

Change from:
```python
diagnostic_disable_tiling = False  # Set to True for testing
```

To:
```python
diagnostic_disable_tiling = True  # DIAGNOSTIC MODE ENABLED
```

### Run Test

1. **Backend auto-reload should pick up the change** (if running with `--reload`)
2. **Upload DJI_0009.JPG** through the frontend
3. **Check backend terminal logs** for diagnostic output

### Expected Log Output (Single-Pass Mode)

```
================================================================================
🔍 DIAGNOSTIC MODE: disable_tiling=True - Running single-pass inference
   Image: WxH, Model: OBB aerial, Conf: 0.35
================================================================================
🔍 SINGLE-PASS RAW OUTPUT: X detection(s) from YOLO
   (This is AFTER Ultralytics' built-in NMS, BEFORE any custom merge)
   Note: Ultralytics YOLO applies NMS internally with default iou=0.45
🔍 SINGLE-PASS FINAL: X detection(s)
   Class breakdown: {'building': Y, ...}
================================================================================
🔍 DIAGNOSTIC RESULT: If this count is ~2-3 buildings but tiled mode shows ~15,
   the problem is in TILING/MERGE. If this ALSO shows ~15 buildings, the problem
   is that Ultralytics' built-in NMS is insufficient or disabled.
================================================================================
```

### Record Results

**Building count (single-pass):** _______

---

## STEP 2: Test Tiled Mode (If Needed)

### Disable Diagnostic Mode

Change back to:
```python
diagnostic_disable_tiling = False  # Normal tiled mode
```

### Expected Log Output (Tiled Mode)

```
🔍 TILED MODE: Starting YOLO on WxH image across N tile(s)
🔍 TILED RAW OUTPUT: X total raw detections from N tiles
   Per-tile counts: [3, 5, 2, 4, ...]
   (These are AFTER Ultralytics' built-in per-tile NMS, BEFORE cross-tile merge)
   Raw class breakdown: {'building': Y, ...}
🔍 CALLING MERGE FUNCTION: WBF with IoU=0.25
🔍 _apply_wbf() CALLED with X input detections, IoU=0.25
🔍 _apply_wbf() RETURNING Y fused detections (from X inputs)
🔍 MERGE FUNCTION RETURNED: Y detections (X → Y)
================================================================================
🔍 TILED INFERENCE COMPLETE (WBF, IoU=0.25):
   X raw → Y merged (Z% reduction, ratio=R.Rx) → Y final
   Final class breakdown: {'building': Y, ...}
================================================================================
```

### Record Results

**Building count (tiled raw):** _______  
**Building count (tiled merged):** _______  
**Merge ratio:** _______x  
**Merge function called:** YES / NO

---

## Interpretation Guide

### Case A: Single-pass shows 2-3 buildings, tiled shows 15
**Diagnosis:** Tiling/merge problem
- Previous WBF fix isn't working
- Check if "🔍 CALLING MERGE FUNCTION" log appears
- If missing: Fix not wired into request flow
- If present but ratio low: WBF/NMS not merging effectively

**Next Steps:**
- Verify merge function is actually being called
- Check IoU calculation logic
- Inspect actual bbox coordinates

### Case B: Single-pass ALSO shows ~15 buildings
**Diagnosis:** Base YOLO NMS problem
- Ultralytics' internal NMS (IoU=0.45) is too lenient
- May need `agnostic_nms=True` or lower `iou` parameter
- Land-cover building extraction may be generating overlapping boxes

**Next Steps:**
- Test with `agnostic_nms=True` in YOLO inference call
- Test with lower `iou` parameter (e.g., 0.30)
- Review land-cover building extraction logic

### Case C: Merge function never called
**Diagnosis:** Code path issue
- Check that updated `detect_objects()` is being imported
- Verify `disable_tiling=False` actually triggers tiled branch
- Look for Python import caching issues

---

## Files Modified

1. **`backend/app/semantic/detector.py`**
   - Added `disable_tiling` parameter (line 506)
   - Added single-pass inference branch (lines 546-644)
   - Added per-tile detection tracking (lines 658-740)
   - Added merge function call tracing (lines 182-220, 242-276, 758-781)
   - Enhanced final result logging (lines 820-832)

2. **`backend/app/main.py`**
   - Added `diagnostic_disable_tiling` flag (line 319)
   - Passed flag to `detect_objects()` call (line 327)
   - Added diagnostic mode warning (lines 330-333)

---

## Status

✅ **Diagnostic instrumentation deployed and compiled**  
✅ **All Python syntax validated**  
✅ **Module imports successfully**  
✅ **Function signature confirmed: `disable_tiling` parameter present**

**Date:** 2026-09-07 15:12 UTC  
**Ready for testing:** YES

---

## Action Required

1. Set `diagnostic_disable_tiling = True` in `backend/app/main.py` line 319
2. Upload DJI_0009.JPG through frontend
3. Report the diagnostic log output showing:
   - Single-pass building count
   - (Optional) Tiled mode building counts and merge ratio
4. Based on results, we'll apply the appropriate fix (Case A vs Case B)
