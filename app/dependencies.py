from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from core.inference import DetectorService

_service: DetectorService | None = None


def build_service(settings: Settings) -> DetectorService:
    return DetectorService.from_checkpoint(
        checkpoint_path=settings.model_path,
        device=settings.device,
        threshold=settings.threshold,
        allow_missing=True,
    )


def get_model(settings: Annotated[Settings, Depends(get_settings)]) -> DetectorService:
    global _service
    if _service is None:
        _service = build_service(settings)
    return _service


def reload_model(settings: Settings | None = None) -> DetectorService:
    global _service
    settings = settings or get_settings()
    _service = build_service(settings)
    return _service
