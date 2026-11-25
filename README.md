# DepthWizard 🛰️🏔️

> **Converting Single-View Optical RGB Remote-Sensing Images into 3D-Navigable Elevation Models**  
> *Built for hackathon demonstrations with Python FastAPI, Hugging Face Depth Anything V2, Rasterio, Trimesh, React, and Three.js (React-Three-Fiber).*

---

## 🌟 Executive Summary

**DepthWizard** is a prototype pipeline that transforms monocular 2D optical aerial/satellite imagery into metric 3D digital surface models (DSMs) and interactive WebGL experiences. By pairing cutting-edge vision foundation models (**Depth Anything V2**) with geospatial reference DEMs (**NASA SRTM 30m**), DepthWizard bridges the gap between dimensionless monocular depth and physical ground elevations without requiring stereo image pairs or expensive LiDAR missions.

---

## 🏗️ Architecture & Pipeline Flow

```
                                  [Optical RGB Input]
                        (GeoTIFF / TIFF / JPG / PNG / Drone Photo)
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
[Georeference Inspector]          [Depth Anything V2]              [Semantic Intelligence Layer]
Checks EXIF GPS or GeoTIFF      Hugging Face Transformers          ├── [YOLOv8-OBB Aerial Detector]
         │                                 │                       │   DOTA aerial weights (vehicles,
         │                                 ▼                       │   buildings, tanks, ships, planes)
         │                          [Relative Depth]               └── [RGB Land-Cover Differentiator]
         │                       Normalized [0.0..1.0]                 Excess Green (ExG = 2G-R-B)
         │                                 │                           + K-Means (k=4): Veg, Water,
         │                                 │                           Built-up, Shadow PNG Map
         ▼                                 │                                 │
  [Georeferenced?]                         │                                 │
   ├── YES ────────────────────────────────┴─────────────────────────────────┤
   │   ▼                                                                     ▼
   │   [SRTM DEM Spatial Alignment]                      [Class-Stratified Linear Regression]
   │   Reprojected & resampled (Rasterio)                Elevation = α_c * Depth + β_c
   │                                                     Fit per land-cover class (fallback if <20pts)
   │                                                                         │
   │                                                                         ▼
   │                                                                 [Metric GeoTIFF DSM]
   │                                                                 Calibrated Absolute Model
   │
   └── NO ─────────────────────────────────┬─────────────────────────────────┤
       ▼                                   │                                 │
   [Object Scale Inference]                │                                 │
   Uses detected vehicle (~1.5m) or        │                                 │
   building (~3m) to derive scale factor   │                                 │
   ├── Succeeded ──────────────────────────┤                                 │
   │   ▼                                   │                                 │
   │   [Approximate Metric DSM]            │                                 │
   │   Labeled "object-scale-inference"    │                                 │
   └── Failed / No qualifying object ──────┘                                 │
       ▼                                                                     │
   [rDSM Relative Depth Model]                                               │
       │                                                                     │
       └───────────────────────────────────┬─────────────────────────────────┘
                                           ▼
                             [3D Heightfield & UV Draping]
                           Downsampled grid + Texture Map
                                           │
                                           ▼
                             [Single-File Binary GLB Export]
                                           │
                                           ▼
                        [Interactive Three.js / R3F 3D Viewer]
                    ├── Layer Toggle: RGB Texture | Land Cover | Object Detections
                    ├── 3D Detection Boxes & Interactive Target Inspection Tags
                    ├── Orbit Controls & First-Person Drone Flight (WASD)
                    ├── Live Surface Raycasting & Elevation HUD (Meters / Slope)
                    ├── Normal-derived Slope Heatmap (0° Green → 15° Yellow → 35°+ Red)
                    ├── Transparent Calibration Methodology Telemetry
                    └── Optional LiDAR Ground-Truth Validation (RMSE, MAE, Pearson r)
```

---

## 🚀 Quickstart Guide

### Prerequisites
- **Python 3.10+** (tested on Python 3.11 with `uv` or `pip`)
- **Node.js v18+** & `npm`

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies (CPU torch is sufficient and fast)
pip install -r requirements-dev.txt

# Run backend API server (runs on http://127.0.0.1:8000)
python run.py
```

### 2. Frontend Setup

```bash
# Open a new terminal and navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start Vite development server (runs on http://localhost:5175)
npm run dev
```

Visit **`http://localhost:5175`** in your browser.

## ☁️ Vercel Deployment

The frontend and FastAPI backend are separate Vercel projects. The backend runs as a Python Function and uses Vercel Blob for uploaded imagery and generated assets; local development continues to use the Vite proxy and local files.

### Create the projects

1. Import this repository twice in Vercel.
2. Set the first project's **Root Directory** to `frontend` and the second to `backend`.
3. Connect the same **public** Vercel Blob store to both projects. The frontend needs the store's `BLOB_READ_WRITE_TOKEN`; the backend uses its Vercel OIDC connection.
4. Set the backend project's Python version to **3.12** (also specified in `backend/.python-version`), enable **Large Functions**, and set `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`.
5. Configure **All Deployments** protection for both projects, including Production and Preview. Standard Protection alone leaves production domains public. Password Protection is plan-dependent (Pro or Enterprise; not Hobby); Vercel Authentication restricts access to team members rather than public demo users.

If your Vercel plan cannot protect the production deployments for public demo users, leave `BLOB_UPLOADS_ENABLED` unset/false until application-level authentication is added.

### Environment variables

Set these in the Vercel project settings:

**Frontend**

```text
VITE_API_BASE_URL=https://<backend-project>.vercel.app
VITE_USE_BLOB_UPLOAD=true
BLOB_UPLOADS_ENABLED=true
BLOB_READ_WRITE_TOKEN=<read-write token from the connected Blob store>
```

`BLOB_READ_WRITE_TOKEN` must remain a server-side variable; do not prefix it with `VITE_`.

**Backend**

```text
CORS_ORIGINS=https://<frontend-project>.vercel.app
BLOB_UPLOADS_ENABLED=true
HF_HOME=/tmp/huggingface
YOLO_CONFIG_DIR=/tmp/depthwizard-ultralytics
```

Include any frontend Preview/custom domains that need API access in `CORS_ORIGINS`. The frontend sends credentials so Vercel deployment protection can guard cross-project API requests. Do not set `BLOB_UPLOADS_ENABLED=true` unless both deployments are protected.

### Deploy and verify

- Deploy the backend first, then set its deployment URL as `VITE_API_BASE_URL` in the frontend project and deploy the frontend.
- Visit both protected deployment URLs in a browser and complete Vercel's protection prompt before testing cross-project API calls.
- Check `<backend-url>/api/health`, then try a sample and an image upload.
- The depth model is downloaded on the first processing request; cold starts can take longer. The model's cache is placed in `/tmp`, and generated files are uploaded to Blob because Vercel Function filesystems are ephemeral.
- Uploaded imagery and generated outputs are publicly accessible through the connected Blob store. Do not upload sensitive imagery.
- `backend/.vercelignore` excludes the local virtual environment, test/tool artifacts, uploads, and stale generated outputs while retaining the small precomputed Mount St. Helens demo bundle. It does not delete anything from the working tree.

---

## 🎮 Interactive Features & Controls

1. **One-Click Offline Hackathon Demos**:
   - **Mount St. Helens Crater (GeoTIFF)**: Full georeferenced pipeline, SRTM DEM linear regression, and paired mock LiDAR ground-truth accuracy validation.
   - **Alpine Drone Flight (Optical)**: Non-georeferenced optical flight demonstrating relative elevation (`rDSM`) mode.
2. **Dual Navigation Modes**:
   - **Orbit Mode**: Left-click drag to rotate around the terrain center, right-click drag to pan, scroll wheel to zoom.
   - **Drone Fly Mode (WASD)**: Click canvas to lock mouse pointer. Use **W/A/S/D** to fly like a survey drone, **Space/E** to ascend, **Shift/Q** to descend, and mouse to steer. Press **ESC** to exit.
3. **Surface Shading Toggles**:
   - **Optical RGB**: Draped photorealistic optical texture.
   - **Slope Heatmap**: Real-time normal gradient color ramp (Green = Flat $0^\circ$, Yellow = Moderate $15^\circ$, Red = Steep Precipice $>35^\circ$).
   - **Wireframe**: Geometric polygon topology inspection.
4. **Live Elevation & Slope HUD**:
   - Hover the mouse anywhere on the 3D model to see live elevation in meters, slope angle in degrees, and local terrain coordinates via real-time raycasting.
5. **Topographic Accuracy Validation**:
   - If a reference raster (e.g. LiDAR DEM) is uploaded, DepthWizard resamples it to the generated DSM and computes **RMSE**, **MAE**, and **Pearson Correlation ($r$)**.
6. **Export Artifacts**:
   - One-click downloads for the exported textured **`.glb` 3D model** and the calibrated **`.tif` GeoTIFF**.

---

## 🎯 Supported Input Types & Domain Safety

DepthWizard is purpose-engineered for **nadir (direct top-down) optical remote-sensing imagery**. Ground-level or oblique perspective photographs produce nonsensical elevation displacement and false-positive aerial detections because foundation models assume a top-down geometry.

| Input Modality | Supported Formats | Sensor Geometry | Pipeline Behavior |
| :--- | :--- | :--- | :--- |
| **Mapping Satellite / Aerial** | `.tif`, `.tiff` (GeoTIFF) | Nadir ($90^\circ$ top-down) with valid CRS/transform | **Full Absolute Calibration (DSM)** against SRTM DEM; immediate nadir bypass ($98\%$ confidence). |
| **Drone Survey Orthomosaic** | `.jpg`, `.jpeg`, `.png`, `.tif` | Gimbal pitch $\approx -90^\circ$ (top-down) | **Relative (`rDSM`) or Object Scale DSM**; passes nadir heuristic gate ($>50\%$). |
| **Ground-Level / Oblique Photo** | Any (`.jpg`, `.png`, `.webp`) | Horizontal or tilted perspective ($\text{pitch} > -45^\circ$) | **Intercepted by Domain Safety Gate**. Returns warning with diagnostic cues and "Process Anyway" override. |

### 🛡️ Why Domain Validation Matters
1. **Monocular Elevation Mismatch**: Depth models (Depth Anything V2) trained on natural scenes map foreground-to-background perspective distance as vertical relief. When applied to ground photos (e.g. rubble or building facades), distant features turn into giant vertical spikes.
2. **Aerial Object Detector Hallucination**: YOLO models trained on DOTAv1 aerial datasets expect bounding footprints seen from above. A patch of blue sky or a blue tarp in a ground photo is frequently misdetected as a `swimming pool`.

### ⚡ Pre-Inference Validation Layer (<100ms)
Before loading heavy neural networks, DepthWizard executes a fast classical-CV heuristic scoring suite (`assess_nadir_likelihood`):
- **Sky & Horizon Detection**: Detects uniform blue, overcast, or sunset regions in the upper frame and horizon edges dividing sky from ground.
- **Vanishing-Point / Perspective Cues**: Extracts Hough lines and identifies converging diagonals characteristic of ground-level linear perspective.
- **Georeference & Drone EXIF Metadata**: Authenticates mapping-grade GeoTIFFs and reads drone gimbal pitch (`-90°` nadir vs `0°` horizontal).
- **Defense-in-Depth on Forced Override**: When a user forces processing on a non-nadir image (`force=true`), the pipeline enforces a stricter YOLO confidence floor ($0.65$), applies a class-level size/aspect-ratio sanity filter, visually de-emphasizes detections in the 3D overlay, and marks the output with persistent caution badges.

---

## 🔬 Scientific Methodology

### 1. Georeference Detection
- **GeoTIFF**: Rasterio queries coordinate reference systems (`src.crs`) and affine transform matrices (`src.transform`).
- **Standard Optical (JPG/PNG)**: PIL reads EXIF metadata (`GPSLatitude`, `GPSLongitude`) and calculates geographic bounds in WGS84.

### 2. Monocular Relative Depth (`rDSM`)
- Leverages `"depth-anything/Depth-Anything-V2-Small-hf"` to extract relative inverse-depth / disparity.
- Normalizes output to a consistent `[0.0, 1.0]` grid representing relative relief.
- Includes an offline relief heuristic fallback to guarantee demo resilience under network interruptions.

### 3. SRTM Calibration
- The corresponding SRTM 30m tile is reprojected and resampled to the optical image's exact pixel dimensions using bilinear interpolation.
- A robust linear regression $\text{Elevation} = \alpha \cdot \text{Depth}_{\text{rel}} + \beta$ is fit over valid non-void samples.
- The calibrated surface is written as a float32 GeoTIFF retaining original georeferencing.

### 4. 3D Mesh Generation & Draping
- The elevation grid is downsampled to max $512 \times 512$ (default $256 \times 256$) for 60 FPS WebGL rendering.
- A regular heightfield $(X, Y, Z)$ is constructed with UV coordinates mapping to the draped optical texture.
- The mesh and texture are packaged into a self-contained binary `.glb` via Trimesh.

---

## 🧠 Semantic Intelligence Layer

DepthWizard couples monocular depth with a parallel computer vision semantic pipeline:

### 1. Aerial Object Detection (Ultralytics YOLOv8-OBB)
- **Active model:** `depthwizard_v2_obb.pt` — **fine-tuned** from the base `yolov8n-obb.pt` checkpoint on **DOTAv1-obb** for **30 epochs** (`lr0=0.001`, `imgsz=1024`).
  - **Fine-tuned validation metrics:** Precision **0.781**, Recall **0.585**, mAP50 **0.620**, mAP50-95 **0.448** (compared to the original untuned baseline).
  - **Why fine-tuning was needed:** The pretrained base `yolov8n-obb.pt` weights showed weak confidence and low recall on real aerial/drone objects during integration testing (e.g., only marginally confident vehicle detections at ~0.40). This was an independent detection quality issue, separate from the tiling, cross-source merge, NMS, and land-cover cross-validation pipeline fixes applied earlier in development.
- **Model selection toggle:** The `DETECTION_MODEL_VERSION` environment variable (or constant at top of `detector.py`) controls which checkpoint is loaded:
  - `v2` (default) → `depthwizard_v2_obb.pt` (fine-tuned)
  - `v1` / `original` / `baseline` → `yolov8n-obb.pt` (original pretrained)
  - Any custom filename or absolute path is also supported.
- Captures oriented bounding boxes (OBBs) suited for nadir and oblique drone/satellite observations across 15 DOTA classes.
- Detects key infrastructure & transit classes: vehicles (`small vehicle`, `large vehicle`), storage tanks, harbors, bridges, ships, and planes.
- Gracefully falls back to standard COCO weights (`yolov8n.pt`) with clear logging if aerial checkpoints are unavailable offline.

### 2. RGB Land-Cover Differentiator (ExG + K-Means)
- Calculates the **Excess Green Index** ($\text{ExG} = 2G - R - B$) per pixel to highlight photosynthetic vegetative matter without requiring multi-spectral NIR bands.
- Clusters 4-dimensional features $[R, G, B, \text{ExG}]$ using fast MiniBatch $K$-means ($k=4$) into canonical terrain strata:
  - 🟩 **Vegetation** (High ExG)
  - 🟦 **Water** (Low luminance, blue dominance, negative ExG)
  - ⬜ **Built-Up / Bare Earth** (High luminance, neutral reflectance)
  - ⬛ **Shadow / Deep Relief** (Lowest perceived luminance)
- Produces a full-resolution 2D class mask and a color-coded draped PNG texture.

### 3. Class-Stratified Metric Calibration & Fallbacks
- **Georeferenced Mode (`per-class`)**: Splits regression sample points by land-cover class and fits a separate slope and intercept ($\alpha_c, \beta_c$) per class. This corrects for forest canopy thickness vs. bare ground penetrability. If any class contains fewer than 20 sample points, it falls back to the robust global fit.
- **Non-Georeferenced Mode (`object-scale-inference`)**: When a vehicle (~1.5m) or building (~3.0m/storey) is detected with high confidence ($>0.5$), the relative depth relief step ($\Delta d$) across the object boundary is used to derive an approximate physical metric scale factor ($S = H_{\text{ref}} / \Delta d$).

---

## ⚠️ Limitations & Accuracy Caveats

1. **Object-Based Scale Inference vs. DEM Ground Control**:
   - Object-based vertical scale inference is an **uncalibrated optical heuristic**. It relies on nominal height assumptions (standard civilian vehicle ≈1.5m, single building storey ≈3.0m) and assumes local planar ground around the target.
   - It **must never be treated with the same confidence as DEM-calibrated (SRTM) or LiDAR ground-truth absolute DSM**. In the UI and API telemetry, it is strictly flagged as `"object-scale-inference"` with explicit disclaimer warnings.
2. **Nadir Monocular Foreshortening**:
   - Monocular vision transformers (Depth Anything V2) estimate relative depth from luminance, aerial haze, texture gradients, and focal cues. Complex multi-tier terrain or extreme shadowed canyons can exhibit localized non-linear scale distortion.
3. **Land-Cover Spectral Limitations of RGB**:
   - Without near-infrared (NIR) or short-wave infrared (SWIR) bands (as found on Sentinel-2 or Landsat), RGB Excess Green is a physical proxy. Highly reflective green roofs or artificial turf can occasionally be grouped with vegetation.

---

## 🛠️ Production Roadmap (Enterprise Upgrades)

*This prototype is intentionally designed for lightweight hackathon demonstration. A production-grade defense or civil photogrammetry deployment would implement:*

1. **Higher-Capacity Vision Models**:
   - Swap `Depth-Anything-V2-Small` for `Depth-Anything-V2-Large` or specialized satellite stereo-optical fusion architectures trained on high-resolution Earth observation data (Maxar WorldView-3, PlanetScope, Sentinel-2).
2. **Ground Control Point (GCP) Bundle Adjustment**:
   - Integrate RTK-GPS survey ground control points and multi-ray bundle block adjustment rather than single-parameter linear regression.
3. **Continuous Multiresolution Level-of-Detail (LOD)**:
   - Stream massive gigapixel terrains via **Cesium 3D Tiles** or **Quantized Mesh (`tin-terrain`)** quadtrees, enabling smooth navigation from continental orbit down to millimeter ground level.
4. **Cloud-Optimized DEM Ingestion**:
   - Query USGS 3DEP 1m LiDAR or Copernicus 30m Global DEM directly using HTTP range-requests on Cloud Optimized GeoTIFFs (COGs) rather than local tiles.
5. **GPU Texture Compression**:
   - Compress draped orthomosaics into **KTX2 / Basis Universal** GPU formats with mipmapping to minimize VRAM usage across mobile and desktop clients.

---

## 📁 Repository Structure

```
SIH/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI server, CORS, /api/upload (nadir gate & force override)
│   │   ├── depth.py             # Depth Anything V2 loader & relative depth estimator
│   │   ├── geo.py               # EXIF GPS and GeoTIFF CRS extraction, SRTM alignment
│   │   ├── calibration.py       # Per-class SRTM regression & object-scale inference
│   │   ├── mesh.py              # 3D heightfield builder, UV texture draping, GLB exporter
│   │   ├── sample_generator.py  # Procedural Mount St. Helens GeoTIFF + LiDAR generator
│   │   ├── validation/          # Domain safety & topographic accuracy package
│   │   │   ├── __init__.py      # Re-exports assess_nadir_likelihood & validate_against_reference
│   │   │   ├── nadir_check.py   # Heuristic scoring (sky/horizon, vanishing lines, metadata)
│   │   │   └── accuracy.py      # LiDAR benchmark accuracy metrics (RMSE, MAE, Pearson r)
│   │   ├── validation_accuracy.py # Compatibility wrapper
│   │   └── semantic/            # Semantic intelligence module
│   │       ├── __init__.py      # Re-exports detector and land cover functions
│   │       ├── detector.py      # Ultralytics YOLOv8-OBB aerial object detection + size filter
│   │       └── land_cover.py    # ExG + KMeans 4-class land-cover differentiator
│   ├── validation/              # Direct module re-export wrapper (/backend/validation)
│   ├── semantic/                # Direct module re-export wrapper (/backend/semantic)
│   ├── sample_data/             # Pre-generated sample datasets for offline demo
│   ├── static/outputs/          # Generated GLB models, preview PNGs, land-cover maps
│   ├── tests/                   # Pytest unit & integration test suite (19 tests)
│   │   ├── test_nadir_check.py  # Unit & integration tests for nadir check & force override
│   │   ├── test_pipeline.py     # End-to-end depth estimation & mesh export tests
│   │   └── test_semantic.py     # YOLO detection, land cover, and calibration tests
│   ├── requirements.txt         # Pinned backend dependencies (torch, ultralytics, etc.)
│   ├── depthwizard_v2_obb.pt      # Fine-tuned DOTA aerial OBB checkpoint (active, DOTAv1 30ep)
│   ├── yolov8n-obb.pt           # Original baseline DOTA aerial OBB checkpoint (fallback)
│   └── run.py                   # Development server launcher
├── frontend/
│   ├── index.html               # Web application shell with Inter & JetBrains Mono fonts
│   ├── package.json             # React 19, R3F, Drei, Three.js, Lucide-React
│   ├── vite.config.js           # Vite config with backend proxy
│   └── src/
│       ├── main.jsx             # React entrypoint
│       ├── App.jsx              # Main UI shell (Layer toggles, 3D Canvas, Drawers)
│       ├── styles/index.css     # Cyber-Geospatial Dark-Tech design system
│       ├── components/
│       │   ├── Header.jsx       # Branding, mode pill, and Layer Toggle Panel
│       │   ├── UploadPanel.jsx  # Drag-and-drop file upload & sample dataset cards
│       │   ├── Viewer3D.jsx     # Three.js canvas, OrbitControls, and drone flight
│       │   ├── TerrainMesh.jsx  # Dynamic material swapping (RGB vs Land Cover vs Slope)
│       │   ├── Detection3DOverlay.jsx # 3D bounding boxes & floating inspection labels
│       │   ├── ControlsBar.jsx  # Orbit vs Fly, Optical vs Slope, Wireframe, Downloads
│       │   ├── ElevationHUD.jsx # Real-time cursor elevation (meters) & slope indicator
│       │   └── StatsPanel.jsx   # Telemetry, calibration methodology, semantic breakdown
│       └── utils/api.js         # API client
└── README.md                    # System documentation
```

---

## 📜 License
MIT License. Built with passion for open geospatial science and 3D remote-sensing innovation.

### GAMUS AGL + SRTM baseline calibration

TerraLift provides a parallel `gamus_agl_baseline` calibration method. It adds the current GAMUS v2 fine-tuned metric Above Ground Level (AGL) prediction to a resampled SRTM ground baseline per pixel: `absolute DSM = SRTM ground + GAMUS AGL`. This is not an SRTM regression and therefore does not use an R² threshold. Select it with `calibration_method=gamus_agl_baseline`; `auto` only prefers it for conservative Washington-DC-like urban/suburban, aerial, small-footprint inputs. All other imagery continues through the existing SRTM regression, relative, or object-scale paths.

The production checkpoint is `backend/models/depth_anything_gamus_v2_finetuned` (400 training tiles and 80 geographically held-out validation tiles). Training unfroze the final two backbone transformer layers plus the depth head (25.3% of parameters) and used L1 plus an edge-aware gradient loss for sharper urban/suburban structural boundaries. On the held-out validation set, v2 achieved a 54.2% MAE improvement over baseline, compared with 38.3% for v1. The one-line `GAMUS_CHECKPOINT_VERSION` constant in `backend/app/depth.py` can be changed from `"v2"` to `"v1"` for rollback/reference runs. This model is intended for GAMUS-applicable US-style urban/suburban aerial imagery.

GAMUS has a real domain gap: it was trained on Washington DC urban/suburban aerial 1024×1024 tiles. It may not generalize well to dense forest, mountains, or non-US urban layouts. The API and telemetry expose the chosen calibration method rather than hiding this decision.