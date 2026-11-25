# ✅ DIAGNOSTIC INSTRUMENTATION COMPLETE

**Date:** 2026-09-08  
**Status:** READY FOR TESTING

---

## 🎯 What Was Implemented

Instead of attempting another blind fix for the duplicate building detection issue (15 "Building" detections for 2-3 actual buildings in DJI_0009.JPG), comprehensive diagnostic instrumentation has been added to **isolate the actual root cause** first.

---

## 🛠️ Changes Made

### 1. Diagnostic Query Parameters Added

**`disable_tiling=true`** - Bypasses tiled inference, runs YOLO on full image in single pass  
**`diagnostic_mode=true`** - Enables verbose logging with per-tile stats and class breakdowns

### 2. Enhanced Logging

**Single-Pass Mode:**
- Raw YOLO output count (after Ultralytics internal NMS)
- Class breakdown showing building count
- Notes that Ultralytics uses iou=0.45 by default

**Tiled Mode:**
- Number of tiles generated
- Per-tile detection counts
- Raw detections before merge
- Merged detections after WBF/NMS
- Final detections after size filters
- Class breakdowns at each stage

### 3. Files Modified

- `backend/app/semantic/detector.py` - Added diagnostic logging throughout
- `backend/app/main.py` - Added query parameters and diagnostic flags
- `DIAGNOSTIC_INSTRUCTIONS.md` - Complete testing guide (NEW)

---

## 🧪 How to Test

### Quick Test Commands:

```bash
# Test 1: Single-pass (no tiling)
curl -X POST "http://localhost:8000/api/upload?disable_tiling=true&diagnostic_mode=true" \
  -F "file=@DJI_0009.JPG"

# Test 2: Tiled mode (normal)
curl -X POST "http://localhost:8000/api/upload?disable_tiling=false&diagnostic_mode=true" \
  -F "file=@DJI_0009.JPG"
```

### Via Browser:
1. Open http://localhost:5173
2. Upload DJI_0009.JPG
3. Check **backend terminal logs** for diagnostic output

---

## 📊 Diagnosis Decision Tree

### Scenario A: Single-pass shows 2-3 buildings, tiled shows 15
→ **Root cause: Tiling/merge problem**  
→ Fix: Adjust cross-tile IoU threshold or verify merge function is called

### Scenario B: Single-pass ALSO shows 15 buildings
→ **Root cause: Base YOLO NMS or land-cover extraction**  
→ Check detection merge log: `[DETECTION MERGE] aerial=X, buildings_from_mask=Y`  
→ If aerial=15: Add explicit NMS with lower threshold  
→ If buildings_from_mask=15: Fix land-cover extraction logic

---

## 🚀 Services Running

✅ Backend: http://localhost:8000  
✅ Frontend: http://localhost:5173  
✅ API Docs: http://localhost:8000/docs

---

## 📝 Next Steps

1. Run both tests on DJI_0009.JPG
2. Record building counts from logs
3. Determine scenario (A or B)
4. Implement targeted fix based on diagnosis
5. Verify fix with same diagnostic flags

See `DIAGNOSTIC_INSTRUCTIONS.md` for detailed testing guide.
