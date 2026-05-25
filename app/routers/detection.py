from fastapi import HTTPException, UploadFile, status
from PIL import UnidentifiedImageError

from app.config import Settings, get_settings
from app.schemas.detection import DetectionResponse, VideoDetectionResponse
from core.inference import DetectorService

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


async def _read_upload(file: UploadFile, settings: Settings) -> bytes:
    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_mb} MB limit.",
        )
    return data


def _upload_suffix(filename: str | None, default: str = ".mp4") -> str:
    if filename and "." in filename:
        return f".{filename.rsplit('.', 1)[-1].lower()}"
    return default


def _is_video_upload(filename: str | None, content_type: str | None) -> bool:
    if content_type and content_type.startswith("video/"):
        return True
    return _upload_suffix(filename, default="") in VIDEO_EXTENSIONS


def classify_upload_bytes(
    data: bytes,
    filename: str | None,
    content_type: str | None,
    service: DetectorService,
    frames_per_scene: int = 6,
    scene_threshold: float = 0.55,
    max_scenes: int = 12,
) -> DetectionResponse | VideoDetectionResponse:
    if _is_video_upload(filename, content_type):
        try:
            result = service.predict_video_bytes(
                data,
                suffix=_upload_suffix(filename),
                frames_per_scene=frames_per_scene,
                scene_threshold=scene_threshold,
                max_scenes=max_scenes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        return VideoDetectionResponse(filename=filename, **result.__dict__)

    try:
        result = service.predict_bytes(data)
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image file.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return DetectionResponse(filename=filename, result=result)
