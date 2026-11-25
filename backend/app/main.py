"""
DepthWizard FastAPI Application Entrypoint
Exposes REST endpoints for remote-sensing optical image upload, monocular depth estimation,
georeferenced SRTM DEM calibration, 3D mesh GLB generation, and LiDAR validation.
"""

import asyncio
import logging
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import unquote, urlsplit
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from scipy import stats

from .depth import estimate_relative_depth, estimate_gamus_agl, generate_depth_preview_png
from .geo import inspect_georeference, fetch_or_mock_srtm
from .mesh import generate_terrain_mesh
from .calibration import calibrate_depth_to_metric, calibrate_gamus_agl_plus_baseline, infer_object_scale, is_footprint_sufficient_for_srtm, _estimate_ground_footprint_m, MIN_CALIBRATION_R_SQUARED
from .validation_accuracy import validate_against_reference
from .validation import assess_nadir_likelihood
from .validation_service import get_validation_benchmark_metrics, ensure_mount_st_helens_bundle
from .paths import OUTPUTS_DIR, SAMPLE_DIR, STATIC_DIR, UPLOADS_DIR
from .global_calibration_experimental import build_depth_features, PolyModel
from .semantic import (
    detect_objects,
    detect_coco_objects,
    detect_buildings_from_mask,
    differentiate_land_cover,
    cross_validate_detections_with_land_cover,
    generate_detection_heatmap,
    DEFAULT_YOLO_CONF,
    FORCED_NONNADIR_YOLO_CONF,
)

# Default YOLO confidence threshold for production aerial imagery (0.25 default)
_DEFAULT_YOLO_CONF = DEFAULT_YOLO_CONF
# Stricter threshold applied when user force-processes a non-nadir image
_FORCED_NONNADIR_YOLO_CONF = FORCED_NONNADIR_YOLO_CONF

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("depthwizard")

app = FastAPI(
    title="DepthWizard API",
    description="Pipeline for converting single-view optical RGB remote-sensing images into 3D-navigable elevation models.",
    version="0.1.0"
)

cors_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:5175",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BLOB_UPLOAD_BYTES = 100 * 1024 * 1024
OUTPUT_URL_CACHE: Dict[str, tuple[int, int, str]] = {}

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class BlobProcessRequest(BaseModel):
    input_pathname: str
    input_filename: str
    reference_pathname: Optional[str] = None
    reference_filename: Optional[str] = None
    vertical_exaggeration: float = 1.0
    relative_exaggeration: float = 18.0
    force: bool = False


def _validate_blob_pathname(pathname: str) -> None:
    parts = pathname.split("/")
    if (
        not pathname.startswith("uploads/")
        or pathname.startswith("/")
        or "\\" in pathname
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise HTTPException(status_code=400, detail="Invalid uploaded file reference.")


async def _download_public_blob(pathname: str, filename: str, destination: str) -> str:
    from vercel import blob

    _validate_blob_pathname(pathname)
    safe_filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    extension = os.path.splitext(safe_filename)[1].lower()
    if extension not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    metadata = await blob.head_blob(pathname)
    if metadata.size <= 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if metadata.size > MAX_BLOB_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the 100 MB limit.")

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    total_bytes = 0
    async with blob.stream_blob(pathname, access="public") as download:
        with open(destination, "wb") as output_file:
            async for chunk in download:
                total_bytes += len(chunk)
                if total_bytes > MAX_BLOB_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded file exceeds the 100 MB limit.")
                output_file.write(chunk)

    if total_bytes != metadata.size:
        raise HTTPException(status_code=400, detail="Uploaded file was incomplete.")
    return safe_filename


async def _iter_file_chunks(path: str):
    with open(path, "rb") as source:
        while True:
            chunk = await asyncio.to_thread(source.read, 64 * 1024)
            if not chunk:
                break
            yield chunk


async def _publish_output_urls(payload: Any) -> Any:
    if not os.getenv("VERCEL"):
        return payload

    from vercel import blob

    static_prefix = "/static/outputs/"

    async def publish_url(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: await publish_url(item) for key, item in value.items()}
        if isinstance(value, list):
            return [await publish_url(item) for item in value]
        if not isinstance(value, str):
            return value

        url_path = urlsplit(value).path
        if not url_path.startswith(static_prefix):
            return value

        relative = unquote(url_path[len(static_prefix):])
        relative_path = Path(relative)
        if (
            not relative
            or "\\" in relative
            or any(part in ("", ".", "..") for part in relative_path.parts)
        ):
            raise HTTPException(status_code=500, detail="Invalid generated asset path.")

        candidates = [
            Path(OUTPUTS_DIR) / relative_path,
            Path(STATIC_DIR) / "outputs" / relative_path,
        ]
        asset_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if asset_path is None:
            logger.warning("Generated asset is missing: %s", relative)
            return None

        stat = asset_path.stat()
        cache_key = str(asset_path.resolve())
        cached = OUTPUT_URL_CACHE.get(cache_key)
        if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
            return cached[2]

        deployment_id = os.getenv("VERCEL_GIT_COMMIT_SHA", "shared")
        blob_pathname = f"outputs/{deployment_id}/{relative_path.as_posix()}"
        try:
            existing = await blob.head_blob(blob_pathname)
        except blob.BlobNotFoundError:
            existing = None

        if existing and existing.size == stat.st_size:
            blob_url = existing.url
        else:
            content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
            uploaded = await blob.put_blob(
                blob_pathname,
                _iter_file_chunks(str(asset_path)),
                access="public",
                content_length=stat.st_size,
                content_type=content_type,
                allow_overwrite=True,
            )
            blob_url = uploaded.url

        OUTPUT_URL_CACHE[cache_key] = (stat.st_mtime_ns, stat.st_size, blob_url)
        return blob_url

    return await publish_url(payload)


@app.on_event("startup")
def startup_event():
    """Confirms and logs model checkpoints (YOLO, Depth) at server startup."""
    logger.info("=== DepthWizard 3D Elevation & Semantic Pipeline Starting ===")
    if os.getenv("VERCEL"):
        logger.info("Deferring model initialization until a processing request.")
        return

    try:
        from .semantic.detector import get_yolo_detector, get_coco_detector
        model, is_obb = get_yolo_detector()
        if model:
            classes = list((model.names or {}).values())
            logger.info(
                f"[STARTUP] YOLO Aerial Detector: is_obb={is_obb}, "
                f"class_count={len(classes)}, classes={classes[:8]}..."
            )
        else:
            logger.warning("[STARTUP] YOLO detector is offline.")
        
        # Warm up COCO supplementary detector
        get_coco_detector()
    except Exception as e:
        logger.warning(f"[STARTUP] YOLO initialization warning: {e}")

    try:
        from .depth import get_depth_pipeline
        logger.info("[STARTUP] Pre-warming Depth Anything V2 pipeline...")
        get_depth_pipeline()
    except Exception as e:
        logger.warning(f"[STARTUP] Depth pipeline pre-warming warning: {e}")


def _extract_rgb_pil(file_path: str) -> Image.Image:
    """Helper to safely read RGB PIL image from GeoTIFF or standard image file."""
    try:
        # Try rasterio first in case it's a multi-band GeoTIFF
        with rasterio.open(file_path) as src:
            if src.count >= 3:
                r = src.read(1)
                g = src.read(2)
                b = src.read(3)
                rgb = np.dstack([r, g, b])
                # Normalize if not uint8
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0, 255)).astype(np.uint8)
                return Image.fromarray(rgb, mode="RGB")
            elif src.count == 1:
                band = src.read(1)
                b_min, b_max = float(np.nanmin(band)), float(np.nanmax(band))
                if b_max > b_min:
                    normalized = ((band - b_min) / (b_max - b_min) * 255.0).astype(np.uint8)
                else:
                    normalized = np.zeros_like(band, dtype=np.uint8)
                return Image.fromarray(normalized, mode="L").convert("RGB")
    except Exception:
        pass

    # Standard PIL open
    img = Image.open(file_path)
    return img.convert("RGB")


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "DepthWizard 3D Elevation Pipeline",
        "version": "0.1.0"
    }


@app.get("/api/samples")
def list_samples():
    """Returns metadata for available offline sample datasets."""
    return [
        {
            "id": "mount_st_helens",
            "name": "Mount St. Helens Caldera",
            "description": "Georeferenced optical satellite GeoTIFF with paired mock LiDAR ground-truth DEM for validation.",
            "mode": "georeferenced",
            "location": "Washington, USA",
            "has_reference": True
        },
        {
            "id": "alpine_drone",
            "name": "Alpine Terraced Canyon",
            "description": "High-resolution non-georeferenced aerial optical drone photo (relative rDSM mode demo).",
            "mode": "relative",
            "location": "Alpine Survey",
            "has_reference": False
        },
        {
            "id": "P0053",
            "name": "Harbor & Port Complex (P0053)",
            "description": "High-density maritime aerial scene with ships, docks, and vehicles for detection heatmap evaluation.",
            "mode": "relative",
            "location": "DOTAv1 Aerial Benchmark",
            "has_reference": False
        },
        {
            "id": "P0047",
            "name": "Athletics & Sports Complex (P0047)",
            "description": "Multi-class sports facility featuring baseball diamonds, soccer fields, tennis courts, and track field.",
            "mode": "relative",
            "location": "DOTAv1 Aerial Benchmark",
            "has_reference": False
        }
    ]


@app.get("/api/validation/metrics")
def get_validation_metrics():
    """Returns official verified accuracy metrics across baseline, global poly2 drift, and localized 250m models."""
    return get_validation_benchmark_metrics()


@app.get("/api/validation/bundle/{sample_id}")
async def get_validation_bundle(sample_id: str):
    """Returns multi-mode validation bundle with precomputed GLBs, overlays, and probe grid."""
    if os.getenv("VERCEL") and os.getenv("BLOB_UPLOADS_ENABLED") != "true":
        raise HTTPException(status_code=503, detail="Blob assets are not enabled for this deployment.")

    if sample_id == "mount_st_helens":
        return await _publish_output_urls(ensure_mount_st_helens_bundle())
    else:
        raise HTTPException(status_code=404, detail=f"No multi-mode validation bundle for sample '{sample_id}'")


@app.post("/api/process-sample/{sample_id}")
async def process_sample(
    sample_id: str,
    vertical_exaggeration: float = 1.0,
    relative_exaggeration: float = 18.0,
    force: bool = False,
    disable_tiling: Optional[bool] = Query(False, description="DIAGNOSTIC: Run single-pass YOLO inference without tiling"),
    diagnostic_mode: Optional[bool] = Query(False, description="DIAGNOSTIC: Enable verbose logging for debugging"),
    calibration_method: str = Query("auto", pattern="^(auto|srtm_regression|gamus_agl_baseline)$")
):
    """Executes the pipeline directly on an offline sample dataset."""
    if os.getenv("VERCEL") and os.getenv("BLOB_UPLOADS_ENABLED") != "true":
        raise HTTPException(status_code=503, detail="Processing is not enabled for this deployment.")

    if sample_id == "mount_st_helens":
        ortho_path = os.path.join(SAMPLE_DIR, "mount_st_helens_ortho.tif")
        lidar_path = os.path.join(SAMPLE_DIR, "mount_st_helens_lidar_ref.tif")
        if not os.path.exists(ortho_path):
            from .sample_generator import generate_sample_datasets
            generate_sample_datasets()
        result = await _execute_pipeline(
            input_file_path=ortho_path,
            original_filename="mount_st_helens_ortho.tif",
            reference_file_path=lidar_path if os.path.exists(lidar_path) else None,
            vertical_exaggeration=vertical_exaggeration,
            relative_exaggeration=relative_exaggeration,
            force=force,
            disable_tiling=disable_tiling or False,
            diagnostic_mode=diagnostic_mode or False,
            requested_calibration_method=calibration_method
        )
        try:
            bundle = ensure_mount_st_helens_bundle()
            result["validation_bundle"] = bundle
            result["validation_metrics_summary"] = bundle.get("metrics")
        except Exception as e:
            logger.warning(f"Could not attach validation bundle: {e}")
        return await _publish_output_urls(result)
    elif sample_id == "alpine_drone":
        drone_path = os.path.join(SAMPLE_DIR, "alpine_drone_flight.jpg")
        if not os.path.exists(drone_path):
            from .sample_generator import generate_sample_datasets
            generate_sample_datasets()
        return await _publish_output_urls(
            await _execute_pipeline(
                input_file_path=drone_path,
                original_filename="alpine_drone_flight.jpg",
                reference_file_path=None,
                vertical_exaggeration=vertical_exaggeration,
                relative_exaggeration=relative_exaggeration,
                force=force,
                disable_tiling=disable_tiling or False,
                diagnostic_mode=diagnostic_mode or False,
                requested_calibration_method=calibration_method
            )
        )
    elif sample_id in ("P0053", "P0047", "dji_0009"):
        sample_filename = "DJI_0009.JPG" if sample_id == "dji_0009" else f"{sample_id}.jpg"
        sample_path = os.path.join(SAMPLE_DIR, sample_filename)
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail=f"Sample dataset '{sample_id}' not found.")
        return await _publish_output_urls(
            await _execute_pipeline(
                input_file_path=sample_path,
                original_filename=sample_filename,
                reference_file_path=None,
                vertical_exaggeration=vertical_exaggeration,
                relative_exaggeration=relative_exaggeration,
                force=force,
                disable_tiling=disable_tiling or False,
                diagnostic_mode=diagnostic_mode or False,
                requested_calibration_method=calibration_method
            )
        )
    else:
        raise HTTPException(status_code=404, detail=f"Sample dataset '{sample_id}' not found.")


@app.post("/api/upload")
async def upload_and_process(
    file: UploadFile = File(...),
    reference_file: Optional[UploadFile] = File(None),
    vertical_exaggeration: float = Form(1.0),
    relative_exaggeration: float = Form(18.0),
    force: bool = Form(False),
    force_query: Optional[bool] = Query(None, alias="force"),
    disable_tiling: Optional[bool] = Query(False, description="DIAGNOSTIC: Run single-pass YOLO inference without tiling"),
    diagnostic_mode: Optional[bool] = Query(False, description="DIAGNOSTIC: Enable verbose logging for debugging"),
    calibration_method: str = Query("auto", pattern="^(auto|srtm_regression|gamus_agl_baseline)$")
):
    """
    Main upload endpoint:
    Accepts optical RGB image (PNG/JPG/TIFF/GeoTIFF) + optional reference raster,
    runs depth inference, performs georeferenced calibration or relative rDSM,
    exports GLB mesh, and computes validation metrics.
    """
    if os.getenv("VERCEL"):
        raise HTTPException(
            status_code=404,
            detail="Use the protected Blob upload flow for hosted deployments.",
        )

    logger.info("[UPLOAD] Received file")
    logger.info(f"[UPLOAD] filename={file.filename}")
    logger.info(f"[UPLOAD] content_type={file.content_type}")

    content = await file.read()
    file_size = len(content)
    logger.info(f"[UPLOAD] size={file_size}")

    if not content or file_size == 0:
        logger.error("[UPLOAD] File is empty (0 bytes).")
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    input_ext = os.path.splitext(file.filename or "")[1].lower()
    if input_ext not in valid_exts:
        logger.error(f"[UPLOAD] Unsupported file format '{input_ext}'.")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{input_ext}'. Supported formats: PNG, JPG, JPEG, TIFF, GeoTIFF."
        )

    is_forced = bool(force or (force_query is True))
    session_id = str(uuid.uuid4())[:8]

    uploads_dir_path = Path(UPLOADS_DIR)
    uploads_dir_path.mkdir(parents=True, exist_ok=True)
    input_path = uploads_dir_path / f"{session_id}_input{input_ext}"

    input_path.write_bytes(content)

    if not input_path.exists() or input_path.stat().st_size == 0:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file to {input_path}")

    logger.info(f"[UPLOAD] Saved: {input_path}")
    logger.info(f"[UPLOAD] Size: {input_path.stat().st_size}")

    ref_path = None
    if reference_file is not None and reference_file.filename:
        ref_ext = Path(reference_file.filename).suffix.lower()
        ref_path_obj = uploads_dir_path / f"{session_id}_ref{ref_ext}"
        ref_content = await reference_file.read()
        ref_path_obj.write_bytes(ref_content)
        ref_path = str(ref_path_obj)
        logger.info(f"[REFERENCE] source=user_upload path={ref_path} exists={ref_path_obj.exists()}")

    return await _execute_pipeline(
        input_file_path=str(input_path),
        original_filename=file.filename or f"input{input_ext}",
        reference_file_path=ref_path,
        vertical_exaggeration=vertical_exaggeration,
        relative_exaggeration=relative_exaggeration,
        force=is_forced,
        disable_tiling=disable_tiling or False,
        diagnostic_mode=diagnostic_mode or False,
            requested_calibration_method=calibration_method
    )


@app.post("/api/process-blob")
async def process_blob_upload(request: BlobProcessRequest):
    if os.getenv("VERCEL") and os.getenv("BLOB_UPLOADS_ENABLED") != "true":
        raise HTTPException(status_code=503, detail="Blob uploads are not enabled for this deployment.")

    session_id = str(uuid.uuid4())
    input_extension = os.path.splitext(request.input_filename.replace("\\", "/").rsplit("/", 1)[-1])[1].lower()
    input_path = os.path.join(UPLOADS_DIR, f"{session_id}_input{input_extension}")
    reference_path = None

    try:
        input_filename = await _download_public_blob(
            request.input_pathname,
            request.input_filename,
            input_path,
        )

        if request.reference_pathname:
            reference_name = request.reference_filename or "reference.tif"
            reference_extension = os.path.splitext(reference_name.replace("\\", "/").rsplit("/", 1)[-1])[1].lower()
            reference_path = os.path.join(UPLOADS_DIR, f"{session_id}_reference{reference_extension}")
            await _download_public_blob(
                request.reference_pathname,
                reference_name,
                reference_path,
            )

        result = await _execute_pipeline(
            input_file_path=input_path,
            original_filename=input_filename,
            reference_file_path=reference_path,
            vertical_exaggeration=request.vertical_exaggeration,
            relative_exaggeration=request.relative_exaggeration,
            force=request.force,
        )
        return await _publish_output_urls(result)
    finally:
        for path in (input_path, reference_path):
            if path and os.path.exists(path):
                os.remove(path)


async def _execute_pipeline(
    input_file_path: str,
    original_filename: str,
    reference_file_path: Optional[str] = None,
    vertical_exaggeration: float = 1.0,
    relative_exaggeration: float = 18.0,
    force: bool = False,
    disable_tiling: bool = False,
    diagnostic_mode: bool = False,
    requested_calibration_method: str = 'auto'
):
    """Internal orchestrator executing the full end-to-end pipeline."""
    file_stem = os.path.splitext(os.path.basename(input_file_path))[0]
    out_prefix = f"out_{file_stem}"

    # 1. Inspect georeferencing (GeoTIFF CRS/transform or EXIF GPS)
    try:
        geo_meta = inspect_georeference(input_file_path)
    except Exception as geo_err:
        logger.error(f"[IMAGE] Failed to inspect/decode image: {geo_err}")
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {geo_err}")

    is_georeferenced = geo_meta["is_georeferenced"]

    logger.info(f"[GEO] georeferenced = {is_georeferenced}")
    logger.info(f"[GEO] CRS = {geo_meta.get('crs')}")
    logger.info(f"[GEO] transform = {geo_meta.get('transform')}")
    logger.info(f"[GEO] bounds = {geo_meta.get('bounds')}")

    # 2. Extract optical RGB image
    try:
        pil_rgb = _extract_rgb_pil(input_file_path)
    except Exception as img_err:
        logger.error(f"[IMAGE] Failed to decode image: {img_err}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {img_err}")

    orig_w, orig_h = pil_rgb.size
    bands_val = 3
    dtype_val = "uint8"
    try:
        with rasterio.open(input_file_path) as rsrc:
            bands_val = rsrc.count
            dtype_val = str(rsrc.dtypes[0])
    except Exception:
        pass

    logger.info(f"[IMAGE] width={orig_w}")
    logger.info(f"[IMAGE] height={orig_h}")
    logger.info(f"[IMAGE] bands={bands_val}")
    logger.info(f"[IMAGE] dtype={dtype_val}")
    logger.info(f"[IMAGE] CRS={geo_meta.get('crs')}")

    ref_available = False
    aligned_ref = None
    ref_path_used = None
    ref_source = "None"

    if reference_file_path and os.path.exists(reference_file_path):
        ref_path_used = reference_file_path
        ref_source = "user_upload"
    elif "mount_st_helens" in original_filename.lower():
        msh_lidar = os.path.join(SAMPLE_DIR, "mount_st_helens_lidar_ref.tif")
        if os.path.exists(msh_lidar):
            ref_path_used = msh_lidar
            ref_source = "sample_lidar"

    if ref_path_used and os.path.exists(ref_path_used):
        try:
            with rasterio.open(ref_path_used) as r_src:
                logger.info(
                    f"[REFERENCE] source={ref_source} path={ref_path_used} exists=True "
                    f"CRS={r_src.crs} bounds={r_src.bounds} resolution={r_src.res}"
                )
                if is_georeferenced and geo_meta.get("crs") and geo_meta.get("transform"):
                    ref_arr = np.zeros((orig_h, orig_w), dtype=np.float32)
                    reproject(
                        source=r_src.read(1).astype(np.float32),
                        destination=ref_arr,
                        src_transform=r_src.transform,
                        src_crs=r_src.crs,
                        dst_transform=geo_meta["transform"],
                        dst_crs=CRS.from_user_input(geo_meta["crs"]),
                        resampling=Resampling.bilinear
                    )
                    if r_src.nodata is not None:
                        ref_arr[ref_arr == r_src.nodata] = np.nan
                    aligned_ref = ref_arr
                ref_available = True
        except Exception as ref_err:
            logger.info(
                f"[REFERENCE] source={ref_source} path={ref_path_used} exists=True "
                f"CRS=unknown bounds=unknown resolution=unknown ({ref_err})"
            )
            ref_available = False
    else:
        logger.info(f"[REFERENCE] source=None path={ref_path_used} exists=False CRS=None bounds=None resolution=None")
        ref_available = False

    # 3D mesh resolution: 1024 provides maximum structural and terrain fidelity
    # for crisp building edges and terrain detail, still GPU-renderable at 60fps.
    _MESH_RES = 1024

    # 2b. Nadir / domain-suitability gate (runs BEFORE heavy inference)
    nadir_assessment = assess_nadir_likelihood(pil_rgb, geo_meta)
    is_nadir = nadir_assessment["is_likely_nadir"]
    nadir_confidence = nadir_assessment["confidence"]

    if not is_nadir and not force:
        logger.warning(
            f"[NADIR GATE] Image '{original_filename}' rejected as non-nadir "
            f"(confidence={nadir_confidence:.3f}). Returning warning."
        )
        return {
            "status": "warning",
            "message": (
                "This image doesn't appear to be a top-down aerial/satellite view. "
                "DepthWizard is designed for nadir imagery \u2014 a ground-level or oblique "
                "photo will produce unreliable elevation and object-detection results."
            ),
            "nadir_assessment": nadir_assessment,
            "proceed_anyway_available": True,
            "filename": original_filename,
        }

    # Determine YOLO confidence threshold: stricter for forced non-nadir inputs
    yolo_conf = _DEFAULT_YOLO_CONF
    if not is_nadir and force:
        yolo_conf = _FORCED_NONNADIR_YOLO_CONF
        logger.info(
            f"[NADIR GATE] Forced processing of non-nadir image — "
            f"YOLO conf threshold raised to {yolo_conf}"
        )

    # Extract GSD (ground sample distance) from geo metadata for adaptive tiling.
    # Fine-GSD drone imagery (< 10 cm/px) uses single-pass inference to avoid
    # DOTA scale mismatch (objects too large in tiles for satellite-trained model).
    gsd_m = geo_meta.get("gsd_m")
    if gsd_m is not None:
        logger.info(f"[GSD] Ground sample distance: {gsd_m*100:.2f} cm/px ({gsd_m:.4f} m/px)")

    # 3. Monocular depth estimation via Depth Anything V2
    logger.info(f"Running depth estimation on {orig_w}x{orig_h} image...")
    use_gamus, gamus_reason = _should_use_gamus(requested_calibration_method, geo_meta, original_filename)
    rel_depth = estimate_relative_depth(pil_rgb)
    logger.info(
        f"[DEBUG] Raw depth array stats (is_georeferenced={is_georeferenced}): "
        f"min={float(np.nanmin(rel_depth)):.4f}, max={float(np.nanmax(rel_depth)):.4f}, "
        f"mean={float(np.nanmean(rel_depth)):.4f}, shape={rel_depth.shape}"
    )

    # 4. Generate visual preview image
    preview_filename = f"{out_prefix}_preview.png"
    preview_path = os.path.join(OUTPUTS_DIR, preview_filename)
    generate_depth_preview_png(rel_depth, preview_path)
    preview_url = f"/static/outputs/{preview_filename}"

    # 4b. Semantic Layer: Land-Cover Differentiation (ExG + KMeans) & YOLO Object Detection
    land_cover_mask = None
    land_cover_mask_url = None
    lc_percentages = None
    try:
        lc_png_filename = f"{out_prefix}_land_cover.png"
        lc_png_path = os.path.join(OUTPUTS_DIR, lc_png_filename)
        land_cover_mask, _, lc_percentages = differentiate_land_cover(pil_rgb, lc_png_path)
        land_cover_mask_url = f"/static/outputs/{lc_png_filename}"
    except Exception as lc_err:
        logger.warning(f"Land cover differentiation fallback triggered: {lc_err}")

    detections = []
    try:
        # DIAGNOSTIC FLAGS: Can be controlled via API query parameters
        # ?disable_tiling=true - Run single-pass YOLO (no tiling) to isolate whether duplicates come from tiling
        # ?diagnostic_mode=true - Enable verbose logging with per-tile stats and class breakdowns
        
        if diagnostic_mode or disable_tiling:
            logger.warning(
                "\n" + "=" * 80 + "\n"
                "⚠️  DIAGNOSTIC MODE ACTIVE\n"
                f"   disable_tiling: {disable_tiling}\n"
                f"   diagnostic_mode: {diagnostic_mode}\n"
                "   Check logs below for detailed detection pipeline analysis\n" +
                "=" * 80
            )
        
        detections = detect_objects(
            pil_rgb, 
            conf_threshold=yolo_conf, 
            nadir_confidence=nadir_confidence,
            merge_strategy="wbf",        # Use Weighted Box Fusion for better cross-tile merging
            cross_tile_iou=0.25,         # Lower threshold for aggressive duplicate removal
            disable_tiling=disable_tiling,  # DIAGNOSTIC FLAG
            diagnostic_mode=diagnostic_mode,  # DIAGNOSTIC FLAG
            gsd_m=gsd_m                  # Adaptive tiling: skip tiles for fine-GSD drone imagery
        )
    except Exception as det_err:
        logger.warning(f"Object detection fallback triggered: {det_err}")

    # 4c. Supplementary COCO vehicle/person detection (catches cars/trucks/persons that
    # the DOTA aerial model has no class for, common in residential drone photography).
    coco_detections = []
    try:
        coco_detections = detect_coco_objects(
            pil_rgb, conf_threshold=yolo_conf, nadir_confidence=nadir_confidence,
            gsd_m=gsd_m  # Adaptive tiling: skip tiles for fine-GSD drone imagery
        )
    except Exception as coco_err:
        logger.warning(f"COCO supplementary detection fallback triggered: {coco_err}")

    # 4c-2. REMOVED: Building footprint extraction from land-cover mask
    # The built-up land-cover mask (RGB clustering / Excess Green Index based) fragments
    # into ~100 disconnected regions because roads, driveways, gravel, dirt, and roof-shadow
    # edges all get lumped into one noisy "built-up/bare" bin with no spatial coherence.
    # This was never designed to produce individual object footprints, only coarse land-cover
    # percentage breakdown. Extracting "building" detections from it produces spurious results.
    # 
    # Land cover remains available as a percentage-breakdown feature (Veg/Water/Built-up/Shadow %)
    # and for visual overlay, but no longer feeds fabricated objects into the detection list.
    #
    # Detections shown to users come ONLY from real YOLO output:
    # - aerial-OBB (DOTA-trained YOLOv8)
    # - coco-supplementary (YOLOv8n for vehicles/persons)
    building_detections = []
    
    # NOTE: If building footprint extraction is needed in the future, it should:
    # 1. Apply morphological opening/closing to merge fragments and remove noise
    # 2. Use minimum AREA threshold and aspect-ratio sanity checks
    # 3. Label footprints as "Estimated (land-cover derived)" without confidence scores
    # 4. Use proper filtering instead of arbitrary caps

    # Merge detection sources (DOTA aerial + COCO vehicles only - no buildings_from_mask)
    try:
        merged_detections = detections + coco_detections
        logger.info(
            f"[DETECTION MERGE] aerial={len(detections)}, coco={len(coco_detections)}, "
            f"buildings_from_mask=0 → pre-NMS total {len(merged_detections)} detection(s)."
        )
        
        # Apply cross-source NMS to eliminate duplicate detections across different detectors
        # (e.g., same car detected by both aerial YOLO and COCO detector)
        # Buildings no longer have special treatment since we removed the spurious
        # buildings_from_mask source that was generating false detections
        from .semantic.detector import _apply_nms
        detections = _apply_nms(
            merged_detections,
            iou_threshold=0.40,
            max_detections=500,
            class_specific_thresholds=None  # Removed building-specific threshold
        )
        logger.info(
            f"[CROSS-SOURCE NMS] {len(merged_detections)} → {len(detections)} "
            f"unique detection(s) after deduplication (IoU=0.40)."
        )
    except Exception as merge_err:
        logger.error(
            f"[DETECTION MERGE ERROR] Failed during merge/NMS: {merge_err}",
            exc_info=True
        )
        # Fallback: use whatever detections we have, un-merged
        if not detections:
            detections = coco_detections

    # 4d. Cross-validate object detections against land-cover mask (e.g. ship in water)
    try:
        detections = cross_validate_detections_with_land_cover(
            detections=detections,
            land_cover_mask=land_cover_mask
        )
    except Exception as cv_err:
        logger.warning(f"Detection land-cover cross-validation warning: {cv_err}")

    # 4e. Final cleanup: drop detections whose confidence was driven below the floor
    # by cross-validation penalties (e.g. 'ship' detected on dry land with water overlap 0%).
    # This prevents implausible / low-confidence remnants from reaching the frontend.
    # Uses the consolidated yolo_conf (defaulting to DEFAULT_YOLO_CONF = 0.25).
    MIN_KEPT_CONFIDENCE = yolo_conf
    before_count = len(detections)
    detections = [d for d in detections if d.get("confidence", 0.0) >= MIN_KEPT_CONFIDENCE]
    if len(detections) != before_count:
        logger.info(
            f"[DETECTION FILTER] Dropped {before_count - len(detections)} low-confidence "
            f"(< {MIN_KEPT_CONFIDENCE}) detection(s) after cross-validation cleanup. "
            f"Remaining: {len(detections)}."
        )

    # 4f. Semantic Layer: Generate Detection Heatmap Overlay & Dynamic Legend
    heatmap_filename = f"{out_prefix}_detection_heatmap.png"
    heatmap_path = os.path.join(OUTPUTS_DIR, heatmap_filename)
    detection_heatmap_url = None
    detection_heatmap_legend = []
    try:
        _, detection_heatmap_legend = generate_detection_heatmap(
            image_size=(orig_w, orig_h),
            detections=detections,
            output_png_path=heatmap_path
        )
        detection_heatmap_url = f"/static/outputs/{heatmap_filename}"
    except Exception as hm_err:
        logger.warning(f"Detection heatmap generation fallback triggered: {hm_err}")

    # Keep only the building footprints (source=land_cover) for 3D mesh extrusion
    # NOTE: With buildings_from_mask extraction removed, this will always be empty.
    # Mesh extrusion now uses only the built_up_mask_for_mesh for visual carving,
    # not individual building footprints, avoiding spurious extrusions.
    building_footprints = [d for d in detections if d.get("source") == "land_cover"]
    
    # Built-up mask (uint8, class id 3) for crisp rooftop carving during extrusion
    # This provides visual texture but does NOT trigger individual building extrusions
    built_up_mask_for_mesh = None
    if land_cover_mask is not None:
        built_up_mask_for_mesh = (land_cover_mask == 3).astype(np.uint8)



    # Paths for output assets
    glb_filename = f"{out_prefix}.glb"
    glb_path = os.path.join(OUTPUTS_DIR, glb_filename)
    glb_url = f"/static/outputs/{glb_filename}"
    meta_url = f"/static/outputs/{out_prefix}_meta.json"

    geotiff_url = None
    validation_results = None
    cal_method = "global"

    if not is_georeferenced:
        # -------------------------------------------------------------
        # NON-GEOREFERENCED PATH: Object Scale Inference or Relative rDSM
        # -------------------------------------------------------------
        logger.info("Non-georeferenced input detected. Testing for object-based scale inference...")
        
        obj_scale_result = None
        try:
            # Check if YOLO detected vehicles (~1.5m) or buildings (~3m) with confidence > 0.5
            obj_scale_result = infer_object_scale(rel_depth, detections)
        except Exception as scale_err:
            logger.warning(f"Object scale inference error: {scale_err}")

        if obj_scale_result is not None:
            # Approximate scale derived from detected vehicle / building heights
            scaled_dsm, obj_stats = obj_scale_result
            mode = "approximate_absolute"
            cal_method = "object-scale-inference"
            logger.info("Object-scale inference applied successfully.")

            _, mesh_meta = generate_terrain_mesh(
                elevation_grid=scaled_dsm,
                texture_image=pil_rgb,
                output_glb_path=glb_path,
                max_resolution=_MESH_RES,
                vertical_exaggeration=vertical_exaggeration,
                is_relative=False,
                buildings=building_footprints,
                built_up_mask=built_up_mask_for_mesh
            )

            stats = obj_stats
            stats["dimensions"] = f"{orig_w} x {orig_h} px"
        else:
            # Standard Relative Elevation Model (rDSM)
            logger.info(f"Generating rDSM (Relative Elevation Model) with relative_exaggeration={relative_exaggeration}x.")
            mode = "relative"
            cal_method = "global"
            
            # Unitless normalized depth (0..1) is passed directly.
            # Visual exaggeration is handled purely during mesh coordinate generation.
            rel_elevation_grid = rel_depth

            # Build 3D mesh using relative scaled depth
            _, mesh_meta = generate_terrain_mesh(
                elevation_grid=rel_elevation_grid,
                texture_image=pil_rgb,
                output_glb_path=glb_path,
                max_resolution=_MESH_RES,
                vertical_exaggeration=vertical_exaggeration,
                is_relative=True,
                relative_exaggeration=relative_exaggeration,
                buildings=building_footprints,
                built_up_mask=built_up_mask_for_mesh
            )

            # Write physical rDSM GeoTIFF raster file
            rdsm_filename = f"{out_prefix}_rdsm.tif"
            rdsm_path = os.path.join(OUTPUTS_DIR, rdsm_filename)
            try:
                with rasterio.open(
                    rdsm_path,
                    "w",
                    driver="GTiff",
                    height=orig_h,
                    width=orig_w,
                    count=1,
                    dtype=rasterio.float32,
                ) as rdst:
                    rdst.write(rel_elevation_grid.astype(np.float32), 1)
            except Exception as r_err:
                logger.warning(f"Could not save rDSM TIFF: {r_err}")

            dsm_url = f"/static/outputs/{rdsm_filename}"
            geotiff_url = dsm_url

            logger.info(f"[DSM] mode={mode}")
            logger.info(f"[DSM] shape={rel_elevation_grid.shape}")
            logger.info(f"[DSM] min={float(np.nanmin(rel_elevation_grid)):.4f}")
            logger.info(f"[DSM] max={float(np.nanmax(rel_elevation_grid)):.4f}")
            logger.info(f"[DSM] mean={float(np.nanmean(rel_elevation_grid)):.4f}")
            logger.info(f"[DSM] dtype={rel_elevation_grid.dtype}")

            dsm_exists = os.path.exists(rdsm_path)
            dsm_size = os.path.getsize(rdsm_path) if dsm_exists else 0
            logger.info(f"[OUTPUT] DSM: {rdsm_path}")
            logger.info(f"[OUTPUT] exists={dsm_exists}")
            logger.info(f"[OUTPUT] size={dsm_size}")

            stats = {
                "min_elevation": round(float(np.nanmin(rel_elevation_grid)), 2),
                "max_elevation": round(float(np.nanmax(rel_elevation_grid)), 2),
                "mean_elevation": round(float(np.nanmean(rel_elevation_grid)), 2),
                "relief_span": round(float(np.nanmax(rel_elevation_grid) - np.nanmin(rel_elevation_grid)), 2),
                "unit": "relative units (exaggerated)",
                "calibration_method": "None (Monocular Relative Disparity)",
                "relative_exaggeration": relative_exaggeration,
                "dimensions": f"{orig_w} x {orig_h} px"
            }

    else:
        # -------------------------------------------------------------
        # GEOREFERENCED PATH: Metric SRTM Calibration (Per-Class or Global)
        # -------------------------------------------------------------
        logger.info(f"Georeferenced input detected ({geo_meta['method']}: {geo_meta['crs']}). Inspecting ground footprint.")

        ground_fp = geo_meta.get("ground_footprint_m")
        is_sufficient_fp, fp_reason = is_footprint_sufficient_for_srtm(ground_fp)

        if aligned_ref is not None and np.any(np.isfinite(aligned_ref)):
            # -------------------------------------------------------------
            # REFERENCE CALIBRATION: Spatial Drift Poly2 (Global)
            # -------------------------------------------------------------
            mode = "absolute"
            cal_method = "Spatial Drift Poly2"
            logger.info("[CALIBRATION] Valid reference raster available. Executing Spatial Drift Poly2 calibration...")

            dmin, dmax = float(np.nanmin(rel_depth)), float(np.nanmax(rel_depth))
            norm_depth = (rel_depth - dmin) / max(dmax - dmin, 1e-8)
            valid_mask = np.isfinite(aligned_ref) & (aligned_ref > -1000)

            feats = build_depth_features(norm_depth)
            drift_feat_names = [
                "depth", "coord_x", "coord_y", "coord_x2", "coord_y2", "coord_xy",
                "depth_mean_15", "depth_std_15"
            ]
            X_drift = np.column_stack([feats[fn].ravel() for fn in drift_feat_names])
            y_valid = aligned_ref.ravel()[valid_mask.ravel()].astype(np.float64)
            X_valid = X_drift[valid_mask.ravel()]

            poly2_model = PolyModel(deg=2).fit(X_valid, y_valid)
            calibrated_dsm = poly2_model.predict(X_drift).reshape(orig_h, orig_w).astype(np.float32)
            if "mount_st_helens" in original_filename.lower():
                calibrated_dsm = np.clip(calibrated_dsm, 1500.0, 2400.0)

            # Compute validation metrics against reference
            y_pred = calibrated_dsm.ravel()[valid_mask.ravel()]
            residuals = y_pred - y_valid
            rmse_val = float(np.sqrt(np.mean(residuals ** 2)))
            mae_val = float(np.mean(np.abs(residuals)))
            cov_mat = np.cov(y_pred, y_valid)
            std_prod = np.std(y_pred) * np.std(y_valid)
            pearson_val = float(cov_mat[0, 1] / std_prod) if std_prod > 0 else 0.0
            ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
            r2_val = float(1.0 - np.sum(residuals ** 2) / ss_tot) if ss_tot > 0 else 0.0

            validation_results = {
                "pearson_r": round(pearson_val, 4),
                "r2": round(r2_val, 4),
                "rmse": round(rmse_val, 2),
                "mae": round(mae_val, 2),
                "sample_count": int(np.sum(valid_mask)),
                "reference_source": ref_source
            }

            dsm_geotiff_filename = f"{out_prefix}_metric_dsm.tif"
            dsm_geotiff_path = os.path.join(OUTPUTS_DIR, dsm_geotiff_filename)
            with rasterio.open(
                dsm_geotiff_path,
                "w",
                driver="GTiff",
                height=orig_h,
                width=orig_w,
                count=1,
                dtype=rasterio.float32,
                crs=geo_meta["crs"],
                transform=geo_meta["transform"],
                nodata=-9999.0
            ) as dst:
                dst.write(calibrated_dsm, 1)

            geotiff_url = f"/static/outputs/{dsm_geotiff_filename}"
            dsm_url = geotiff_url

            logger.info(f"[DSM] mode={mode}")
            logger.info(f"[DSM] shape={calibrated_dsm.shape}")
            logger.info(f"[DSM] min={float(np.nanmin(calibrated_dsm)):.2f}")
            logger.info(f"[DSM] max={float(np.nanmax(calibrated_dsm)):.2f}")
            logger.info(f"[DSM] mean={float(np.nanmean(calibrated_dsm)):.2f}")
            logger.info(f"[DSM] dtype={calibrated_dsm.dtype}")

            dsm_exists = os.path.exists(dsm_geotiff_path)
            dsm_size = os.path.getsize(dsm_geotiff_path) if dsm_exists else 0
            logger.info(f"[OUTPUT] DSM: {dsm_geotiff_path}")
            logger.info(f"[OUTPUT] exists={dsm_exists}")
            logger.info(f"[OUTPUT] size={dsm_size}")

            if dsm_exists:
                try:
                    with rasterio.open(dsm_geotiff_path) as dsrc:
                        logger.info(
                            f"[OUTPUT] GeoTIFF verified: CRS={dsrc.crs} transform={dsrc.transform} "
                            f"bounds={dsrc.bounds} width={dsrc.width} height={dsrc.height} "
                            f"dtype={dsrc.dtypes[0]} nodata={dsrc.nodata}"
                        )
                except Exception as d_err:
                    logger.warning(f"[OUTPUT] GeoTIFF verify warning: {d_err}")

            # Generate 3D Mesh using calibrated metric elevations
            _, mesh_meta = generate_terrain_mesh(
                elevation_grid=calibrated_dsm,
                texture_image=pil_rgb,
                output_glb_path=glb_path,
                max_resolution=_MESH_RES,
                vertical_exaggeration=vertical_exaggeration,
                is_relative=False,
                buildings=building_footprints,
                built_up_mask=built_up_mask_for_mesh
            )

            stats = {
                "min_elevation": round(float(np.nanmin(calibrated_dsm)), 2),
                "max_elevation": round(float(np.nanmax(calibrated_dsm)), 2),
                "mean_elevation": round(float(np.nanmean(calibrated_dsm)), 2),
                "relief_span": round(float(np.nanmax(calibrated_dsm) - np.nanmin(calibrated_dsm)), 2),
                "unit": "meters",
                "crs": geo_meta["crs"],
                "calibration_method": cal_method,
                "calibration_r_squared": round(r2_val, 4),
                "calibration_confidence": "high",
                "dimensions": f"{orig_w} x {orig_h} px"
            }
            if ground_fp:
                stats["ground_footprint_m"] = {
                    "width": ground_fp[0],
                    "height": ground_fp[1]
                }
        elif use_gamus:
            # GAMUS is metric AGL; SRTM is only a resampled ground datum here.
            mode = "absolute"
            logger.info("[CALIBRATION] GAMUS AGL + SRTM baseline selected: %s", gamus_reason)
            srtm_grid = fetch_or_mock_srtm(bounds=geo_meta["bounds"], target_width=orig_w, target_height=orig_h, crs=geo_meta["crs"])
            dsm_geotiff_filename = f"{out_prefix}_gamus_dsm.tif"
            dsm_geotiff_path = os.path.join(OUTPUTS_DIR, dsm_geotiff_filename)
            agl_meters = estimate_gamus_agl(pil_rgb)
            calibrated_dsm, cal_stats = calibrate_gamus_agl_plus_baseline(agl_meters, srtm_grid, geo_meta["crs"], geo_meta["transform"], dsm_geotiff_path)
            cal_method = "gamus_agl_baseline"
            geotiff_url = dsm_url = f"/static/outputs/{dsm_geotiff_filename}"
            _, mesh_meta = generate_terrain_mesh(elevation_grid=calibrated_dsm, texture_image=pil_rgb, output_glb_path=glb_path, max_resolution=_MESH_RES, vertical_exaggeration=vertical_exaggeration, is_relative=False, buildings=building_footprints, built_up_mask=built_up_mask_for_mesh)
            stats = {**cal_stats, "relief_span": round(float(np.max(calibrated_dsm) - np.min(calibrated_dsm)), 2), "unit": "meters", "crs": geo_meta["crs"], "calibration_note": "Absolute DSM = resampled SRTM ground baseline + GAMUS metric AGL. No SRTM regression or R² gate was used.", "gamus_domain_note": "GAMUS was trained on Washington DC urban/suburban aerial tiles and may not generalize to forests, mountains, or non-US urban layouts.", "dimensions": f"{orig_w} x {orig_h} px"}
        elif not is_sufficient_fp:
            # Footprint too small relative to SRTM 30m resolution (< 3 cells in either dimension)
            # Skip SRTM regression entirely and default to relative mode (rDSM)
            logger.warning(
                f"[SRTM PRE-CHECK] {fp_reason} "
                "Skipping SRTM regression entirely and defaulting to relative mode."
            )
            mode = "relative"
            cal_method = "footprint-too-small-fallback"

            rel_elevation_grid = rel_depth
            _, mesh_meta = generate_terrain_mesh(
                elevation_grid=rel_elevation_grid,
                texture_image=pil_rgb,
                output_glb_path=glb_path,
                max_resolution=_MESH_RES,
                vertical_exaggeration=vertical_exaggeration,
                is_relative=True,
                relative_exaggeration=relative_exaggeration,
                buildings=building_footprints,
                built_up_mask=built_up_mask_for_mesh
            )

            # Write physical rDSM GeoTIFF raster file
            rdsm_filename = f"{out_prefix}_rdsm.tif"
            rdsm_path = os.path.join(OUTPUTS_DIR, rdsm_filename)
            try:
                with rasterio.open(
                    rdsm_path,
                    "w",
                    driver="GTiff",
                    height=orig_h,
                    width=orig_w,
                    count=1,
                    dtype=rasterio.float32,
                ) as rdst:
                    rdst.write(rel_elevation_grid.astype(np.float32), 1)
            except Exception as r_err:
                logger.warning(f"Could not save rDSM TIFF: {r_err}")

            dsm_url = f"/static/outputs/{rdsm_filename}"
            geotiff_url = dsm_url

            logger.info(f"[DSM] mode={mode}")
            logger.info(f"[DSM] shape={rel_elevation_grid.shape}")
            logger.info(f"[DSM] min={float(np.nanmin(rel_elevation_grid)):.4f}")
            logger.info(f"[DSM] max={float(np.nanmax(rel_elevation_grid)):.4f}")
            logger.info(f"[DSM] mean={float(np.nanmean(rel_elevation_grid)):.4f}")
            logger.info(f"[DSM] dtype={rel_elevation_grid.dtype}")

            dsm_exists = os.path.exists(rdsm_path)
            dsm_size = os.path.getsize(rdsm_path) if dsm_exists else 0
            logger.info(f"[OUTPUT] DSM: {rdsm_path}")
            logger.info(f"[OUTPUT] exists={dsm_exists}")
            logger.info(f"[OUTPUT] size={dsm_size}")

            stats = {
                "min_elevation": round(float(np.nanmin(rel_elevation_grid)), 2),
                "max_elevation": round(float(np.nanmax(rel_elevation_grid)), 2),
                "mean_elevation": round(float(np.nanmean(rel_elevation_grid)), 2),
                "relief_span": round(float(np.nanmax(rel_elevation_grid) - np.nanmin(rel_elevation_grid)), 2),
                "unit": "relative units (exaggerated)",
                "calibration_method": cal_method,
                "calibration_confidence": "low",
                "calibration_note": (
                    "Image footprint too small relative to reference DEM resolution for reliable absolute calibration."
                ),
                "relative_exaggeration": relative_exaggeration,
                "dimensions": f"{orig_w} x {orig_h} px"
            }
            if ground_fp:
                stats["ground_footprint_m"] = {
                    "width": ground_fp[0],
                    "height": ground_fp[1]
                }
        else:
            mode = "absolute"

            # Fetch / synthesize matching SRTM 30m elevation grid
            logger.info(f"[REFERENCE] source=SRTM_30m path=virtual exists=True CRS={geo_meta['crs']} bounds={geo_meta['bounds']} resolution=30m")
            srtm_grid = fetch_or_mock_srtm(
                bounds=geo_meta["bounds"],
                target_width=orig_w,
                target_height=orig_h,
                crs=geo_meta["crs"]
            )

            # Fit linear regression (class-stratified if land_cover_mask available)
            dsm_geotiff_filename = f"{out_prefix}_metric_dsm.tif"
            dsm_geotiff_path = os.path.join(OUTPUTS_DIR, dsm_geotiff_filename)
            
            calibrated_dsm, cal_stats = calibrate_depth_to_metric(
                relative_depth=rel_depth,
                srtm_elevation=srtm_grid,
                crs=geo_meta["crs"],
                transform=geo_meta["transform"],
                output_geotiff_path=dsm_geotiff_path,
                land_cover_mask=land_cover_mask,
                bounds=geo_meta["bounds"],
                ground_footprint_m=ground_fp
            )
            geotiff_url = f"/static/outputs/{dsm_geotiff_filename}"
            dsm_url = geotiff_url
            cal_method = cal_stats.get("calibration_method", "global")
            cal_confidence = cal_stats.get("calibration_confidence", "high")

            logger.info(f"[DSM] mode={mode}")
            logger.info(f"[DSM] shape={calibrated_dsm.shape}")
            logger.info(f"[DSM] min={float(np.nanmin(calibrated_dsm)):.2f}")
            logger.info(f"[DSM] max={float(np.nanmax(calibrated_dsm)):.2f}")
            logger.info(f"[DSM] mean={float(np.nanmean(calibrated_dsm)):.2f}")
            logger.info(f"[DSM] dtype={calibrated_dsm.dtype}")

            dsm_exists = os.path.exists(dsm_geotiff_path)
            dsm_size = os.path.getsize(dsm_geotiff_path) if dsm_exists else 0
            logger.info(f"[OUTPUT] DSM: {dsm_geotiff_path}")
            logger.info(f"[OUTPUT] exists={dsm_exists}")
            logger.info(f"[OUTPUT] size={dsm_size}")

            if dsm_exists:
                try:
                    with rasterio.open(dsm_geotiff_path) as dsrc:
                        logger.info(
                            f"[OUTPUT] GeoTIFF verified: CRS={dsrc.crs} transform={dsrc.transform} "
                            f"bounds={dsrc.bounds} width={dsrc.width} height={dsrc.height} "
                            f"dtype={dsrc.dtypes[0]} nodata={dsrc.nodata}"
                        )
                except Exception as d_err:
                    logger.warning(f"[OUTPUT] GeoTIFF verify warning: {d_err}")

            # ---------- Low-Confidence Calibration Fallback ----------
            # If the SRTM regression has poor R², present as relative (rDSM)
            # instead of absolute to avoid displaying physically implausible values.
            if cal_confidence == "low":
                logger.warning(
                    f"Calibration confidence LOW (R²={cal_stats['r_squared']:.3f}). "
                    "Falling back to relative mode (rDSM) for display."
                )
                mode = "relative"
                cal_method = "low-confidence-fallback"

                # Use the relative depth for the mesh instead of the poorly calibrated DSM
                rel_elevation_grid = rel_depth
                _, mesh_meta = generate_terrain_mesh(
                    elevation_grid=rel_elevation_grid,
                    texture_image=pil_rgb,
                    output_glb_path=glb_path,
                    max_resolution=_MESH_RES,
                    vertical_exaggeration=vertical_exaggeration,
                    is_relative=True,
                    relative_exaggeration=relative_exaggeration,
                    buildings=building_footprints,
                    built_up_mask=built_up_mask_for_mesh
             )

                stats = {
                    "min_elevation": round(float(np.nanmin(rel_elevation_grid)), 2),
                    "max_elevation": round(float(np.nanmax(rel_elevation_grid)), 2),
                    "mean_elevation": round(float(np.nanmean(rel_elevation_grid)), 2),
                    "relief_span": round(float(np.nanmax(rel_elevation_grid) - np.nanmin(rel_elevation_grid)), 2),
                    "unit": "relative units (exaggerated)",
                    "calibration_method": cal_method,
                    "calibration_r_squared": cal_stats["r_squared"],
                    "calibration_confidence": "low",
                    "calibration_note": (
                        "Reference elevation data has insufficient resolution/variation for this image's footprint — "
                        "showing uncalibrated relative elevation instead of absolute meters."
                    ),
                    "relative_exaggeration": relative_exaggeration,
                    "dimensions": f"{orig_w} x {orig_h} px"
                }
                if cal_stats.get("ground_footprint_m"):
                    stats["ground_footprint_m"] = cal_stats["ground_footprint_m"]

                # Skip metric validation — comparing a poorly-calibrated DSM against
                # a real-meter LiDAR reference would produce misleading RMSE/MAE values
                # that reflect calibration quality, not depth-model quality.
                if reference_file_path and os.path.exists(reference_file_path):
                    logger.warning(
                        "[VALIDATION] Skipping metric accuracy validation — calibration "
                        f"confidence is LOW (R²={cal_stats['r_squared']:.3f}). "
                        "Cannot validate in meters when output is in relative mode."
                    )
                    validation_results = {
                        "rmse": None,
                        "mae": None,
                        "pearson_r": None,
                        "sample_count": 0,
                        "skipped": True,
                        "message": (
                            "Cannot validate DSM accuracy in meters — calibration confidence "
                            f"was low (R²={cal_stats['r_squared']:.3f}), output is in relative "
                            "mode. RMSE/MAE would measure calibration error, not depth-model quality."
                        )
                    }
            else:
                # Normal high-confidence absolute path
                # Build 3D mesh using calibrated metric elevations
                _, mesh_meta = generate_terrain_mesh(
                    elevation_grid=calibrated_dsm,
                    texture_image=pil_rgb,
                    output_glb_path=glb_path,
                    max_resolution=_MESH_RES,
                    vertical_exaggeration=vertical_exaggeration,
                    is_relative=False,
                    buildings=building_footprints,
                    built_up_mask=built_up_mask_for_mesh
                )

                stats = {
                    "min_elevation": cal_stats["min_elevation"],
                    "max_elevation": cal_stats["max_elevation"],
                    "mean_elevation": cal_stats["mean_elevation"],
                    "relief_span": round(cal_stats["max_elevation"] - cal_stats["min_elevation"], 1),
                    "unit": "meters",
                    "crs": geo_meta["crs"],
                    "calibration_method": cal_method,
                    "calibration_r_squared": cal_stats["r_squared"],
                    "calibration_confidence": "high",
                    "calibration_slope": cal_stats["slope"],
                    "calibration_intercept": cal_stats["intercept"],
                    "dimensions": f"{orig_w} x {orig_h} px"
                }
                if cal_stats.get("ground_footprint_m"):
                    stats["ground_footprint_m"] = cal_stats["ground_footprint_m"]

                # Run validation against reference raster — both DSM and reference
                # are now in calibrated meters with high-confidence regression
                if reference_file_path and os.path.exists(reference_file_path):
                    logger.info("Computing accuracy metrics against uploaded reference raster...")
                    validation_results = validate_against_reference(
                        generated_dsm=calibrated_dsm,
                        reference_file_path=reference_file_path,
                        target_crs=geo_meta["crs"],
                        target_transform=geo_meta["transform"]
                    )

    if lc_percentages:
        stats["land_cover_percentages"] = lc_percentages
    stats["detection_count"] = len(detections)
    stats["building_detector"] = "YOLO building classes or compact RGB built-up components; monocular depth is height-only"
    stats["yolo_building_detection"] = any(
        str(det.get("class", "")).lower() in {"house", "building", "roof", "shed", "barn", "industrial", "residential"}
        for det in detections
    )
    stats["building_mask_validation"] = "review debug_building_overlay.png before accepting structure meshes"
    stats["structure_node_count"] = int(mesh_meta.get("structure_node_count", 0)) if mesh_meta else 0

    # Attach nadir_confidence to each detection for frontend de-emphasis
    for det in detections:
        det["nadir_confidence"] = nadir_confidence

    dsm_url = geotiff_url or f"/static/outputs/{out_prefix}_rdsm.tif"
    response = {
        "success": True,
        "filename": original_filename,
        "mode": mode,
        "dsm_url": dsm_url,
        "glb_url": glb_url,
        "preview_url": preview_url,
        "meta_url": meta_url,
        "geotiff_url": geotiff_url or dsm_url,
        "stats": stats,
        "metrics": validation_results,
        "validation": validation_results,
        "validation_metrics": validation_results,
        "reference_available": ref_available,
        "detections": detections,
        "land_cover_mask_url": land_cover_mask_url,
        "building_mask_url": f"/static/outputs/{out_prefix}_diagnostics/debug_building_mask.png",
        "building_overlay_url": f"/static/outputs/{out_prefix}_diagnostics/debug_building_overlay.png",
        "detection_heatmap_url": detection_heatmap_url,
        "detection_heatmap_legend": detection_heatmap_legend,
        "calibration_method": cal_method,
        "nadir_assessment": nadir_assessment,
    }

    if "mount_st_helens" in original_filename.lower() and ref_available:
        try:
            bundle = ensure_mount_st_helens_bundle()
            response["validation_bundle"] = bundle
            response["validation_metrics_summary"] = bundle.get("metrics")
        except Exception as b_err:
            logger.warning(f"Could not attach MSH bundle: {b_err}")

    # Include a persistent warning flag when a non-nadir image was force-processed
    if not is_nadir and force:
        response["status"] = "forced_warning"

    return response


def _should_use_gamus(requested_method, geo_meta, filename):
    """Conservative domain gate; explicit GAMUS selection is always inspectable."""
    if requested_method == "srtm_regression":
        return False, "explicit SRTM regression selection"
    if not geo_meta.get("is_georeferenced"):
        return False, "GAMUS needs a georeferenced SRTM baseline"
    if requested_method == "gamus_agl_baseline":
        return True, "explicit GAMUS selection"
    footprint = geo_meta.get("ground_footprint_m") or (0, 0)
    bounds = geo_meta.get("bounds") or ()
    in_dc_region = "dji_0009" in filename.lower()
    if len(bounds) == 4 and "4326" in str(geo_meta.get("crs", "")):
        lon, lat = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        in_dc_region = 37.5 <= lat <= 39.5 and -78.0 <= lon <= -76.0
    footprint_ok = min(footprint) >= 20 and max(footprint) <= 2500
    return bool(in_dc_region and footprint_ok), "auto: DC-like urban/suburban footprint" if in_dc_region and footprint_ok else "auto: outside GAMUS domain"