from fastapi import APIRouter, Depends

from app.dependencies import get_model
from core.inference import DetectorService

router = APIRouter()


@router.get("/health")
def health(model: DetectorService = Depends(get_model)) -> dict[str, object]:
    return {
        "status": "ok" if model.is_ready else "degraded",
        "model_loaded": model.checkpoint_loaded,
        "device": str(model.device),
        "threshold": model.threshold,
        "image_size": model.image_size,
        "frequency_mode": model.frequency_mode,
    }
