import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.face_detection import detect_face, expand_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create face crop metadata from a unified raw manifest.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="data/processed_metadata.csv")
    parser.add_argument("--output-dir", default="data/raw/processed_crops")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--face-detector", default=None, help="Default: INSWAPPER_FACE_DETECTOR or insightface.")
    parser.add_argument("--workers", type=int, default=1, help="Number of face-crop workers. Try 1 or 2 for InsightFace.")
    parser.add_argument("--log-every", type=int, default=100, help="Print progress every N input frames.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows for a smoke test.")
    parser.add_argument("--verbose", action="store_true", help="Print one line for every processed frame.")
    return parser.parse_args()


def infer_metadata(row) -> dict[str, object]:
    source = str(row.get("source", "unknown")).lower()
    fake_type = str(row.get("fake_type", "real" if int(row["label"]) == 0 else source)).lower()
    return {
        "fake_type": fake_type,
        "is_inswapper": int("inswapper" in fake_type or "inswapper" in source),
        "boundary_label": int(row.get("boundary_label", row["label"])),
        "quality_label": int(row.get("quality_label", 0)),
        "source": source,
        "video_id": row.get("video_id", row.get("video_path", "")),
        "identity_id": row.get("identity_id", ""),
    }


def process_frame(task: dict[str, object]) -> dict[str, object]:
    index = int(task["index"])
    image_path = Path(str(task["path"]))
    output_dir = Path(str(task["output_dir"]))
    started_at = time.time()
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        detection = detect_face(image, backend=task.get("face_detector"))
        if detection is None:
            return {
                "ok": False,
                "path": str(image_path),
                "reason": "face_not_found",
                "elapsed_sec": round(time.time() - started_at, 3),
            }
        rows: list[dict[str, object]] = []
        for crop_name, scale in [("tight", 1.1), ("expanded", 1.5), ("scene", 2.0)]:
            crop_box = expand_box(detection.box, scale, image.width, image.height)
            crop = image.crop(crop_box).resize(
                (int(task["image_size"]), int(task["image_size"])),
                Image.Resampling.BILINEAR,
            )
            crop_path = output_dir / f"{index:08d}_{crop_name}.jpg"
            crop.save(crop_path, quality=95)
            rows.append(
                {
                    "path": str(crop_path),
                    "label": int(task["label"]),
                    "crop_type": crop_name,
                    "box_x1": crop_box[0],
                    "box_y1": crop_box[1],
                    "box_x2": crop_box[2],
                    "box_y2": crop_box[3],
                    "face_confidence": round(detection.confidence, 6),
                    "face_detector": detection.backend,
                    "fake_type": task["fake_type"],
                    "is_inswapper": int(task["is_inswapper"]),
                    "boundary_label": int(task["boundary_label"]),
                    "quality_label": int(task["quality_label"]),
                    "source": task["source"],
                    "video_id": task["video_id"],
                    "identity_id": task["identity_id"],
                }
            )
    return {
        "ok": True,
        "path": str(image_path),
        "rows": rows,
        "elapsed_sec": round(time.time() - started_at, 3),
        "face_confidence": round(detection.confidence, 6),
    }


def build_tasks(frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for index, row in frame.iterrows():
        metadata = infer_metadata(row)
        tasks.append(
            {
                "index": int(index),
                "path": str(row["path"]),
                "label": int(row["label"]),
                "output_dir": str(output_dir),
                "image_size": int(args.image_size),
                "face_detector": args.face_detector,
                **metadata,
            }
        )
    return tasks


def log_progress(
    completed: int,
    total: int,
    crop_rows: int,
    failures: int,
    started_at: float,
    *,
    prefix: str = "progress",
) -> None:
    elapsed = max(time.time() - started_at, 1e-6)
    frames_per_minute = completed / (elapsed / 60)
    eta_min = ((total - completed) / max(frames_per_minute, 1e-6)) if completed else 0.0
    print(
        f"{prefix} [{completed}/{total}] crop_rows={crop_rows} failures={failures} "
        f"elapsed={elapsed / 60:.1f}m speed={frames_per_minute:.1f} frames/min eta={eta_min:.1f}m",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    if args.limit is not None:
        frame = frame.head(args.limit)
    required = {"path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input manifest missing columns: {sorted(missing)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(frame, output_dir, args)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    started_at = time.time()
    total = len(tasks)
    log_every = max(1, int(args.log_every))
    workers = max(1, min(int(args.workers), os.cpu_count() or 1))
    if workers > 2 and (args.face_detector is None or args.face_detector == "insightface"):
        print(
            f"warning: workers={workers} means each process loads InsightFace; start with 1 or 2 if RAM is tight",
            flush=True,
        )
    print(
        f"creating face crops from {total} frames image_size={args.image_size} workers={workers}",
        flush=True,
    )
    log_progress(0, total, len(rows), len(failures), started_at, prefix="start")

    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            result = process_frame(task)
            if result["ok"]:
                rows.extend(result["rows"])
                if args.verbose:
                    print(
                        f"ok frame={result['path']} crops={len(result['rows'])} "
                        f"conf={result['face_confidence']} elapsed={result['elapsed_sec']}s",
                        flush=True,
                    )
            else:
                failures.append({"path": result["path"], "reason": result["reason"]})
                print(f"failed frame={result['path']} reason={result['reason']}", flush=True)
            if completed % log_every == 0 or completed == total:
                log_progress(completed, total, len(rows), len(failures), started_at)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_frame, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    result = future.result()
                    if result["ok"]:
                        rows.extend(result["rows"])
                        if args.verbose:
                            print(
                                f"ok frame={result['path']} crops={len(result['rows'])} "
                                f"conf={result['face_confidence']} elapsed={result['elapsed_sec']}s",
                                flush=True,
                            )
                    else:
                        failures.append({"path": result["path"], "reason": result["reason"]})
                        print(f"failed frame={result['path']} reason={result['reason']}", flush=True)
                except Exception as exc:
                    failures.append({"path": str(task["path"]), "reason": repr(exc)})
                    print(f"failed frame={task['path']} reason={exc!r}", flush=True)
                if completed % log_every == 0 or completed == total:
                    log_progress(completed, total, len(rows), len(failures), started_at)

    rows.sort(key=lambda item: (str(item.get("video_id", "")), str(item["path"])))

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["path", "label", "source"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    failure_path = Path(args.output_csv).with_suffix(".failures.csv")
    with open(failure_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    elapsed = max(time.time() - started_at, 1e-6)
    print(
        f"wrote {len(rows)} crop rows to {args.output_csv}; "
        f"failures={len(failures)} failure_log={failure_path}; elapsed={elapsed / 60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
