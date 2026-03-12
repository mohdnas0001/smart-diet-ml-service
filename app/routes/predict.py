from fastapi import APIRouter, UploadFile, File, HTTPException
from app.schemas.response import AnalysisResponse
from app.utils.image_utils import validate_image_file, load_image_from_bytes, fix_exif_rotation
from app.utils.logger import logger

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/api/predict", response_model=AnalysisResponse)
async def predict(file: UploadFile = File(...)):
    """
    Analyse a food image and return detected food items with nutrient estimates.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Use JPEG or PNG.",
        )

    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB)")

    try:
        validate_image_file(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        image = load_image_from_bytes(data)
        image = fix_exif_rotation(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not open image: {exc}")

    from app.main import pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Analysis pipeline not initialised")

    try:
        result = await pipeline.run(image)
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    return result
