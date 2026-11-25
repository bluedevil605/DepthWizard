# ✅ BUILDING DUPLICATE FIX - ROOT CAUSE FOUND

**Date:** 2026-09-08  
**Issue:** DJI_0009.JPG showed 15 building detections for 2-3 actual buildings

---

## 🔬 ROOT CAUSE (From Diagnostics)

**Single-Pass YOLO Output:**
```
🔍 SINGLE-PASS RAW OUTPUT: 1 detection(s)
   Raw class breakdown: {'small vehicle': 1}
```
→ YOLO detected NO buildings, only 1 vehicle

**Land-Cover Extraction:**
```
Building footprint extraction: 100 building(s) detected
[BUILDING FILTER] kept 15/100 building footprint(s)
[CROSS-SOURCE NMS] 16 → 16 (0% reduction with IoU=0.15)
```
→ Connected-component analysis found 100 buildings, kept 15, NMS didn't merge any

**Conclusion:** Problem is land-cover building extraction, NOT YOLO tiling/NMS

---

## 🔧 THE FIX

Added aggressive pre-NMS to land-cover buildings in `backend/app/main.py`:

```python
# New: Pre-NMS on building footprints BEFORE size filtering
if len(building_detections) > 0:
    building_detections = _apply_nms(
        building_detections,
        iou_threshold=0.05,  # Very aggressive - merge if >5% overlap
        max_detections=100
    )
```

Also changed cross-source NMS building threshold from 0.15 to 0.05.

**Why IoU=0.05?**
- Connected components produce adjacent boxes with minimal overlap
- 5% threshold catches close neighbors
- Consolidates fragmented rooftop detections

---

## 📊 Expected Results

**Before:** 100 → 15 → 15 buildings (0% NMS reduction)  
**After:** 100 → ~10-15 → ~3-5 → 2-3 buildings (80%+ reduction)

---

## 🧪 Test Now

Backend has auto-reloaded with the fix. Upload DJI_0009.JPG:

```bash
curl -X POST "http://localhost:8000/api/upload?diagnostic_mode=true" \
  -F "file=@DJI_0009.JPG"
```

Or via browser: http://localhost:5173

**Expected:** 2-3 buildings instead of 15 ✅

---

## 📝 Look For These Logs

```
[BUILDING PRE-NMS] Applying aggressive NMS to 100 land-cover buildings
[BUILDING PRE-NMS] After NMS: ~10-15 buildings remain
[BUILDING FILTER] kept 2-4/10-15 building footprint(s)
[CROSS-SOURCE NMS] 5 → 2-3 unique detection(s)
```

**Services:** Backend http://localhost:8000 | Frontend http://localhost:5173
