import asyncio
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.config import Settings, get_settings
from app.dependencies import get_model
from app.routers.detection import _read_upload, classify_upload_bytes
from app.schemas.detection import SubmitResponse, TaskStatusResponse
from core.inference import DetectorService

router = APIRouter()

TASKS: dict[str, TaskStatusResponse] = {}
RUNNING_TASKS: set[asyncio.Task] = set()


async def _run_detection_task(
    task_id: str,
    data: bytes,
    filename: str | None,
    content_type: str | None,
    service: DetectorService,
) -> None:
    try:
        result = await asyncio.to_thread(
            classify_upload_bytes,
            data,
            filename,
            content_type,
            service,
        )
    except HTTPException as exc:
        TASKS[task_id] = TaskStatusResponse(task_id=task_id, status="failed", error=str(exc.detail))
        return
    except Exception as exc:
        TASKS[task_id] = TaskStatusResponse(task_id=task_id, status="failed", error=str(exc))
        return

    TASKS[task_id] = TaskStatusResponse(task_id=task_id, status="completed", result=result)


@router.post("/submit", response_model=SubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_detection(
    file: Annotated[UploadFile, File(description="Image or video file to classify.")],
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[DetectorService, Depends(get_model)],
) -> SubmitResponse:
    data = await _read_upload(file, settings)
    task_id = uuid4().hex
    TASKS[task_id] = TaskStatusResponse(task_id=task_id, status="processing")
    task = asyncio.create_task(
        _run_detection_task(
            task_id,
            data,
            file.filename,
            file.content_type,
            service,
        )
    )
    RUNNING_TASKS.add(task)
    task.add_done_callback(RUNNING_TASKS.discard)
    return SubmitResponse(task_id=task_id)


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found.")
    return task
