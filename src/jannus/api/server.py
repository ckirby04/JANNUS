"""
FastAPI server for BrainMetScan segmentation and clinical reporting.
"""

import collections
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask

from .._version import __version__
from ..core import paths as _paths
from ..rag.recist import RECISTMeasurer
from ..segmentation.ensemble import SmartEnsemble
from ..segmentation.longitudinal import LongitudinalTracker
from ..segmentation.model_registry import ModelRegistry
from .auth import check_permission, get_api_key_info, set_db
from .database import Database
from .dicom_handler import DICOMIngester
from .logging_config import RequestTimer, get_logger, setup_logging
from .schema import (
    BatchPredictionResponse,
    ComparisonLesionMatch,
    ComparisonResponse,
    HealthResponse,
    LesionDetail,
    ModelInfo,
    PredictionResponse,
    SegmentationResult,
)

# Global state
_ensemble: SmartEnsemble | None = None
_registry: ModelRegistry | None = None
_db: Database | None = None
_dicom_ingester = DICOMIngester()
_longitudinal_tracker = LongitudinalTracker()
_recist = RECISTMeasurer()

# LRU-style bounded cache: max 50 predictions to prevent memory leaks
_MAX_CACHE_SIZE = 50
_predictions_cache: collections.OrderedDict = collections.OrderedDict()

logger = get_logger("api")


def _get_project_root() -> Path:
    return _paths.repo_root()


def _cache_put(job_id: str, data: dict) -> None:
    """Insert into bounded prediction cache, evicting oldest if full."""
    _predictions_cache[job_id] = data
    while len(_predictions_cache) > _MAX_CACHE_SIZE:
        _predictions_cache.popitem(last=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global _ensemble, _registry, _db

    # Setup structured logging
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_file = os.environ.get("LOG_FILE")
    setup_logging(log_level=log_level, log_file=log_file)

    project_root = _get_project_root()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize database
    db_path = os.environ.get("DATABASE_PATH")
    _db = Database(db_path)
    set_db(_db)
    logger.info("Database initialized", extra={"endpoint": "startup"})

    _registry = ModelRegistry(project_root)
    config_path = project_root / "configs" / "models.yaml"

    if config_path.exists():
        logger.info("Loading ensemble models", extra={"endpoint": "startup"})
        _ensemble = SmartEnsemble.from_config(str(config_path), device=device)
        n_models = len(_ensemble.models) if _ensemble else 0
        logger.info("Models loaded", extra={"endpoint": "startup", "lesion_count": n_models})
    else:
        logger.warning("configs/models.yaml not found. No models loaded.", extra={"endpoint": "startup"})

    yield  # App runs

    # Cleanup
    _predictions_cache.clear()
    logger.info("Server shutdown complete", extra={"endpoint": "shutdown"})


app = FastAPI(
    title="JANNUS API",
    description="Brain metastasis segmentation and clinical reporting service",
    version=__version__,
    lifespan=lifespan,
)

# CORS. v1.50 SECURITY CHANGE — the default is now "no cross-origin access"
# rather than "*". A wildcard here meant any web page a clinician happened to
# visit could issue cross-origin requests against a server on their network;
# combined with the old anonymous-by-default auth, that reached /predict and
# /admin/*. Sites that need browser access must name their origins explicitly:
#     CORS_ORIGINS=https://viewer.hospital.example
_cors_origins = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    gpu = torch.cuda.is_available()
    n_models = len(_ensemble.models) if _ensemble else 0
    names = list(_ensemble.names) if _ensemble else []
    return HealthResponse(
        status="healthy" if n_models > 0 else "no_models",
        gpu_available=gpu,
        models_loaded=n_models,
        model_names=names,
    )


@app.get("/models", response_model=list[ModelInfo])
async def list_models():
    if _registry is None:
        return []
    models = _registry.list_models()
    # patch_size and threshold are absent on entries that don't need them
    # (e.g. nnU-Net, where the predictor owns its own preprocessing). Use
    # 0 / 0.5 as sentinel defaults so the API still serialises cleanly.
    return [
        ModelInfo(
            name=m["name"],
            patch_size=int(m.get("patch_size", 0)),
            threshold=float(m.get("threshold", 0.5)),
            architecture=m.get("architecture", "unknown"),
            exists=m["exists"],
        )
        for m in models
    ]


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    files: list[UploadFile] = File(...),
    input_format: str = Form("nifti"),
    threshold: float = Form(0.5),
    use_tta: bool = Form(False),
    generate_report: bool = Form(False),
    case_id: str = Form(""),
    api_key_info: dict | None = Depends(get_api_key_info),
):
    """
    Run segmentation on uploaded NIfTI or DICOM files.

    - **nifti**: Upload 4 NIfTI files named t1_pre.nii.gz, t1_gd.nii.gz, flair.nii.gz, t2.nii.gz
    - **dicom**: Upload DICOM files; sequences auto-identified from SeriesDescription
    """
    check_permission(api_key_info, "predict")

    if _ensemble is None or len(_ensemble.models) == 0:
        raise HTTPException(status_code=503, detail="No models loaded")

    key_id = api_key_info["key_id"] if api_key_info else None

    with RequestTimer() as timer:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                if input_format == "dicom":
                    tensor, spacing = await _process_dicom_upload(files, tmpdir)
                else:
                    tensor, spacing = await _process_nifti_upload(files, tmpdir)

                # Run prediction
                result = _ensemble.predict_volume(
                    tensor,
                    threshold=threshold,
                    use_tta=use_tta,
                    voxel_spacing=spacing,
                )

            elapsed = timer.duration_ms / 1000.0

            # Build response
            lesions = [LesionDetail(**ld) for ld in result["lesion_details"]]
            total_vol_voxels = sum(lesion.volume_voxels for lesion in lesions)
            total_vol_mm3 = sum(lesion.volume_mm3 for lesion in lesions)

            cid = case_id or f"case_{uuid.uuid4().hex[:8]}"
            job_id = uuid.uuid4().hex

            seg_result = SegmentationResult(
                case_id=cid,
                lesion_count=result["lesion_count"],
                total_volume_voxels=total_vol_voxels,
                total_volume_mm3=total_vol_mm3,
                lesions=lesions,
                processing_time_seconds=round(elapsed, 2),
                job_id=job_id,
            )

            # Cache for mask download (bounded)
            _cache_put(job_id, {
                "probability_map": result["probability_map"],
                "binary_mask": result["binary_mask"],
            })

            # Record in database
            if _db:
                _db.record_prediction(
                    job_id=job_id,
                    case_id=cid,
                    status="completed",
                    lesion_count=result["lesion_count"],
                    total_volume_mm3=total_vol_mm3,
                    processing_time=round(elapsed, 2),
                    threshold=threshold,
                    use_tta=use_tta,
                    api_key_id=key_id,
                )
                _db.log_event(
                    "prediction",
                    endpoint="/predict",
                    api_key_id=key_id,
                    details={"job_id": job_id, "lesion_count": result["lesion_count"]},
                    ip_address=request.client.host if request.client else None,
                    status_code=200,
                )

            logger.info(
                "Prediction completed",
                extra={
                    "endpoint": "/predict",
                    "job_id": job_id,
                    "case_id": cid,
                    "lesion_count": result["lesion_count"],
                    "duration_ms": timer.duration_ms,
                    "api_key_id": key_id,
                },
            )

            # Optional RAG report
            rag_report = None
            if generate_report:
                rag_report = _generate_rag_report(result, cid)

            return PredictionResponse(
                status="success",
                result=seg_result,
                rag_report=rag_report,
            )

        except Exception as e:
            logger.error(
                "Prediction failed",
                extra={"endpoint": "/predict", "error_detail": str(e), "api_key_id": key_id},
            )
            if _db:
                _db.log_event(
                    "prediction_error",
                    endpoint="/predict",
                    api_key_id=key_id,
                    details={"error": str(e)},
                    ip_address=request.client.host if request.client else None,
                    status_code=500,
                )
            return PredictionResponse(status="error", error=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    threshold: float = Form(0.5),
    use_tta: bool = Form(False),
    api_key_info: dict | None = Depends(get_api_key_info),
):
    """Batch prediction: each file group is a separate case (files named case_N_seq.nii.gz)."""
    check_permission(api_key_info, "predict")

    single = await predict(
        request=request,
        files=files,
        input_format="nifti",
        threshold=threshold,
        use_tta=use_tta,
        api_key_info=api_key_info,
    )
    return BatchPredictionResponse(
        status="success",
        total_cases=1,
        completed=1 if single.status == "success" else 0,
        failed=0 if single.status == "success" else 1,
        results=[single],
    )


@app.get("/predict/{job_id}/mask")
async def download_mask(
    job_id: str,
    output_format: str = "nifti",
    api_key_info: dict | None = Depends(get_api_key_info),
):
    """Download segmentation mask for a completed prediction.

    v1.50 SECURITY CHANGE — this endpoint now requires authentication. In
    v1.40 it was the only handler with no auth dependency at all, so it served
    patient segmentation masks unauthenticated even when AUTH_REQUIRED=true.
    Job IDs are uuid4 and not guessable, but they are disclosed in prediction
    responses, in logs, and via /admin/predictions, so obscurity was the only
    control.
    """
    check_permission(api_key_info, "predict")

    if job_id not in _predictions_cache:
        raise HTTPException(status_code=404, detail="Prediction not found")

    cached = _predictions_cache[job_id]
    mask = cached["binary_mask"]

    # v1.50: mkdtemp() here leaked a directory containing a patient
    # segmentation on every download. FileResponse needs the file to outlive
    # this function, so the cleanup is registered as a background task rather
    # than done with a context manager.
    tmpdir = tempfile.mkdtemp()
    cleanup = BackgroundTask(shutil.rmtree, tmpdir, ignore_errors=True)

    if output_format == "numpy":
        tmp_path = Path(tmpdir) / "segmentation_mask.npy"
        np.save(str(tmp_path), mask)
        return FileResponse(str(tmp_path), media_type="application/octet-stream",
                            filename="segmentation_mask.npy", background=cleanup)
    else:
        tmp_path = Path(tmpdir) / "segmentation_mask.nii.gz"
        nii = nib.Nifti1Image(mask.astype(np.float32), np.eye(4))
        nib.save(nii, str(tmp_path))
        return FileResponse(str(tmp_path), media_type="application/gzip",
                            filename="segmentation_mask.nii.gz", background=cleanup)


@app.get("/predict/{job_id}/report")
async def download_report(
    job_id: str,
    api_key_info: dict | None = Depends(get_api_key_info),
):
    """Download a PDF clinical report for a completed prediction."""
    if job_id not in _predictions_cache:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Get prediction record from DB for metadata
    pred_record = _db.get_prediction(job_id) if _db else None
    case_id = pred_record["case_id"] if pred_record else "unknown"

    _predictions_cache[job_id]

    try:
        from .pdf_report import PDFReportGenerator

        generator = PDFReportGenerator()

        # Build result dict from cached data
        lesion_details = []
        if pred_record and pred_record.get("result_json"):
            lesion_details = pred_record["result_json"].get("lesions", [])

        result = {
            "lesion_count": pred_record.get("lesion_count", 0) if pred_record else 0,
            "lesion_details": lesion_details,
            "processing_time_seconds": pred_record.get("processing_time_seconds", 0) if pred_record else 0,
        }

        pdf_bytes = generator.generate(result=result, case_id=case_id)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="report_{job_id[:8]}.pdf"'},
        )

    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="PDF generation requires reportlab. Install with: pip install 'jannus[api]'",
        ) from exc


@app.post("/compare", response_model=ComparisonResponse)
async def compare_timepoints(
    request: Request,
    baseline_files: list[UploadFile] = File(...),
    followup_files: list[UploadFile] = File(...),
    threshold: float = Form(0.5),
    baseline_case_id: str = Form("baseline"),
    followup_case_id: str = Form("followup"),
    api_key_info: dict | None = Depends(get_api_key_info),
):
    """Compare segmentation between baseline and follow-up timepoints."""
    check_permission(api_key_info, "predict")

    if _ensemble is None or len(_ensemble.models) == 0:
        raise HTTPException(status_code=503, detail="No models loaded")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_tensor, spacing = await _process_nifti_upload(baseline_files, tmpdir + "/baseline")
            followup_tensor, _ = await _process_nifti_upload(followup_files, tmpdir + "/followup")

            baseline_result = _ensemble.predict_volume(
                baseline_tensor, threshold=threshold, voxel_spacing=spacing
            )
            followup_result = _ensemble.predict_volume(
                followup_tensor, threshold=threshold, voxel_spacing=spacing
            )

            comparison = _longitudinal_tracker.compare_timepoints(
                baseline_result, followup_result, voxel_spacing=spacing
            )

        logger.info(
            "Comparison completed",
            extra={
                "endpoint": "/compare",
                "duration_ms": 0,
                "api_key_id": api_key_info["key_id"] if api_key_info else None,
            },
        )

        return ComparisonResponse(
            status="success",
            baseline_case_id=baseline_case_id,
            followup_case_id=followup_case_id,
            response_category=comparison["response_category"],
            matched_lesions=[
                ComparisonLesionMatch(**m) for m in comparison["matched_lesions"]
            ],
            new_lesions=comparison["new_lesions"],
            resolved_lesions=comparison["resolved_lesions"],
            sum_of_diameters_baseline_mm=comparison["sum_of_diameters_baseline_mm"],
            sum_of_diameters_followup_mm=comparison["sum_of_diameters_followup_mm"],
        )

    except Exception as e:
        logger.error("Comparison failed", extra={"endpoint": "/compare", "error_detail": str(e)})
        return ComparisonResponse(status="error", error=str(e))


# --- Admin Endpoints ---

@app.get("/admin/stats")
async def admin_stats(
    days: int = 30,
    api_key_info: dict | None = Depends(get_api_key_info),
):
    """Get usage statistics. Requires 'admin' permission if auth is enabled."""
    check_permission(api_key_info, "admin")

    if _db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return _db.get_stats(days=days)


@app.get("/admin/predictions")
async def admin_list_predictions(
    limit: int = 50,
    offset: int = 0,
    api_key_info: dict | None = Depends(get_api_key_info),
):
    """List recent predictions. Requires 'admin' permission if auth is enabled."""
    check_permission(api_key_info, "admin")

    if _db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return _db.list_predictions(limit=limit, offset=offset)


@app.get("/admin/keys")
async def admin_list_keys(
    api_key_info: dict | None = Depends(get_api_key_info),
):
    """List all API keys (without actual key values). Requires 'admin' permission."""
    check_permission(api_key_info, "admin")

    if _db is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    return _db.list_api_keys()


# --- Helpers ---

async def _process_nifti_upload(files: list[UploadFile], tmpdir: str) -> tuple[torch.Tensor, tuple]:
    """Save uploaded NIfTI files and load as tensor."""
    tmpdir_path = Path(tmpdir)
    tmpdir_path.mkdir(parents=True, exist_ok=True)

    nifti_paths: dict[str, str] = {}
    for f in files:
        name = Path(f.filename).name
        for seq in ["t1_pre", "t1_gd", "flair", "t2"]:
            if seq in name.lower():
                save_path = tmpdir_path / f"{seq}.nii.gz"
                content = await f.read()
                save_path.write_bytes(content)
                nifti_paths[seq] = str(save_path)
                break

    if len(nifti_paths) < 4:
        required = {"t1_pre", "t1_gd", "flair", "t2"}
        missing = sorted(required - set(nifti_paths.keys()))
        raise ValueError(f"Missing sequences: {missing}. Got: {sorted(nifti_paths.keys())}")

    tensor, spacing = _dicom_ingester.load_nifti_as_tensor(nifti_paths)
    return tensor, spacing


async def _process_dicom_upload(files: list[UploadFile], tmpdir: str) -> tuple[torch.Tensor, tuple]:
    """Save uploaded DICOM files, identify sequences, and load as tensor."""
    tmpdir_path = Path(tmpdir)
    dicom_dir = tmpdir_path / "dicom"
    dicom_dir.mkdir(parents=True, exist_ok=True)

    # v1.50 SECURITY FIX (critical) — the uploaded filename is attacker
    # controlled and was previously joined to `dicom_dir` unsanitised. Two
    # working attacks existed:
    #   1. traversal:      filename="../../../etc/cron.d/pwn"
    #   2. absolute path:  pathlib discards the left operand entirely, so
    #                      Path("/tmp/x") / "/etc/cron.d/pwn" == "/etc/cron.d/pwn"
    # Either gave arbitrary file write as the server user — root in the shipped
    # container — i.e. straightforward remote code execution. `Path(...).name`
    # strips both, matching what the sibling NIfTI handler already did.
    file_paths = []
    for position, f in enumerate(files):
        safe_name = Path(f.filename or "").name
        if not safe_name or safe_name in {".", ".."}:
            # Never fall back to the client-supplied string; synthesise instead.
            safe_name = f"upload_{position:05d}.dcm"
        save_path = dicom_dir / safe_name
        content = await f.read()
        save_path.write_bytes(content)
        file_paths.append(str(save_path))

    dicom_images = _dicom_ingester.ingest_dicom_files(file_paths)
    if len(dicom_images) < 4:
        raise ValueError(
            f"Could not identify all 4 sequences from DICOM. Found: {list(dicom_images.keys())}"
        )

    nifti_dir = tmpdir_path / "nifti"
    nifti_paths = _dicom_ingester.dicom_to_nifti(dicom_images, str(nifti_dir))
    tensor, spacing = _dicom_ingester.load_nifti_as_tensor(nifti_paths)
    return tensor, spacing


def _generate_rag_report(result: dict, case_id: str) -> str | None:
    """Generate a RAG report if the query module is available."""
    try:
        from ..rag.feature_extractor import RadiomicFeatureExtractor

        extractor = RadiomicFeatureExtractor()
        features = extractor.extract_lesion_features(result["binary_mask"])
        features["case_id"] = case_id

        # Try v2 hybrid retrieval for richer context
        kb_facts = []
        try:
            from pathlib import Path as _Path
            v2_db_path = _Path("outputs/rag/chromadb_v2")
            bm25_path = _Path("outputs/rag/bm25_index.pkl")
            if v2_db_path.exists():
                from ..rag.query import retrieve_literature
                query_text = f"brain metastases {features.get('num_lesions', 0)} lesions"
                retrieval_results = retrieve_literature(
                    query_text,
                    v2_db_path,
                    bm25_index_path=str(bm25_path) if bm25_path.exists() else None,
                    k=5,
                )
                kb_facts = [r.text for r in retrieval_results]
        except Exception:
            pass

        from ..rag.query import generate_report_local
        return generate_report_local(features, [], kb_facts)
    except Exception:
        return None
