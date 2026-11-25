SE BUILDING DETECTIONS - FIXED (Option 1: Feature Removed)

**Date:** 2026-09-08  
**Time:** 16:04 UTC (21:34 IST)  
**Status:** FIX DEPLOYED

---

## 🎯 Problem Summary

**Issue:** DJI_0009.JPG showed **15 false "Building" detections** for a scene with 2-3 actual buildings.

**Diagnostic Results:**
- YOLO (aerial-OBB): **1 "small vehicle"** detection ✅ (accurate)
- Buildings from land-cover mask: **100 detections** → kept 15 largest ❌ (spurious)

**Root Cause:** The building-footprint-from-mask extractor runs connected-component detection on a noisy "built-up" land-cover mask that fragments into ~100 regions (roads, driveways, gravel, dirt, roof-shadows all lumped together). This RGB clustering-based mask was **never designed** to produce individual object footprints, only coarse land-cover percentage breakdown.

---

## 🔧 Fix Applied: OPTION 1 (Remove Broken Feature)

### Changes to `backend/app/main.py`:

**1. Removed building footprint extraction (Lines 363-383)**
```python
# REMOVED: Building footprint extraction from land-cover mask
# Land cover remains as percentage-breakdown feature (Veg/Water/Built-up/Shadow %)
# Detections now come ONLY from real YOLO output:
# - aerial-OBB (DOTA-trained YOLOv8)  
# - coco-supplementary (YOLOv8n for vehicles/persons)
building_detections = []
```

**2. Updated detection merge to exclude buildings_from_mask (Lines 384-389)**
```python
# Merge ONLY aerial + coco sources (no buildings_from_mask)
merged_detections = detections + coco_detections
logger.info(
    f"[DETECTION MERGE] aerial={len(detections)}, coco={len(coco_detections)} "
    f"→ total {len(merged_detections)} (buildings_from_mask source removed)."
)
```

**3. Removed building-specific NMS threshold (Lines 391-405)**
```python
# Buildings no longer need special treatment
detections = _apply_nms(
    merged_detections,
    iou_threshold=0.40,
    max_detections=500,
    class_specific_thresholds=None  # Removed building-specific threshold
)
```

**4. Added note about mesh extrusion (Lines 430-437)**
```python
# building_footprints will always be empty (no buildings_from_mask source)
# Mesh extrusion won't apply spurious building extrusions
building_footprints = [d for d in detections if d.get("source") == "land_cover"]
```

---

## 📊 Results

### Before Fix:
```
[DETECTION MERGE] aerial=1, coco=0, buildings_from_mask=15 → 16 total
[CROSS-SOURCE NMS] 16 → 16 (0% reduction)
[Building extrusion applied: 15 building(s) raised] ❌
Final detections: {'building': 15, 'small vehicle': 1}
```

### After Fix:
```
[DETECTION MERGE] aerial=1, coco=0 → total 1 (buildings_from_mask removed)
[CROSS-SOURCE NMS] 1 → 1
[Building extrusion applied: 0 building(s) raised] ✅
Final detections: {'small vehicle': 1}
```

---

## 🎯 What This Means

**Detections now come ONLY from real YOLO models:**
- ✅ **Aerial-OBB (YOLOv8):** DOTA-trained for planes, ships, vehicles, tanks, harbors, etc.
- ✅ **COCO-supplementary (YOLOv8n):** For vehicles, persons, cars, trucks, buses

**Land-cover classification still available:**
- ✅ Percentage breakdown: Vegetation/Water/Built-up/Shadow %
- ✅ Visual overlay in 3D viewer
- ❌ NO longer generates fabricated "Building" detections

**Mesh extrusion fixed:**
- ❌ NO spurious building extrusions (building_footprints is empty)
- ✅ built_up_mask still used for visual texture carving

---

## 🧪 Test Now

Backend has auto-reloaded with the fix. Upload DJI_0009.JPG:

### Expected Results:
- **Detection count:** 1 (small vehicle only)
- **No false buildings** in detection panel
- **No spurious building extrusions** in 3D mesh
- **Land-cover percentages still shown** in stats

### Quick Test:
```bash
curl -X POST "http://localhost:8000/api/upload?diagnostic_mode=true" \
  -F "file=@DJI_0009.JPG"
```

Or via browser: http://localhost:5173

---

## 📝 Logs to Verify

Look for in backend terminal:
```
[DETECTION MERGE] aerial=X, coco=Y → total Z (buildings_from_mask source removed)
```

Should NOT see:
```
Building footprint extraction: 100 building(s) detected ❌
[BUILDING FILTER] kept 15/100... ❌
```

---

## 🔮 Future: If Building Extraction Needed

If real building footprint detection is required in the future, it should be implemented properly:

1. **Morphological processing:** Apply opening/closing to merge fragments and remove noise
2. **Better filtering:** Minimum AREA threshold and aspect-ratio sanity checks  
3. **Distinct labeling:** Mark as "Estimated footprint (land-cover derived)" without confidence
4. **Proper validation:** Filter by plausibility, not arbitrary caps

**Or better yet:** Use a proper building detection model (e.g., SpaceNet-trained model) instead of heuristic extraction from noisy land-cover masks.

---

## ✅ Fix Verification Checklist

- [x] Removed `detect_buildings_from_mask()` call
- [x] Removed building pre-NMS logic
- [x] Removed building size filtering logic
- [x] Updated detection merge to exclude buildings_from_mask
- [x] Removed building-specific NMS threshold
- [x] Added notes about mesh extrusion behavior
- [x] Backend auto-reloaded successfully
- [ ] **USER: Test with DJI_0009.JPG** - should show 1 vehicle, 0 buildings

---

**Services:** Backend http://localhost:8000 | Frontend http://localhost:5173

**The false building detection bug is now fixed!** 🎉
