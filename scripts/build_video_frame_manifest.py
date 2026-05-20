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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.video import sample_scene_aware_frames_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample scene-aware video frames into an image manifest.")
    parser.add_argument("--videos", required=True, help="CSV with path,label,source columns.")
    parser.add_argument("--output-csv", default="data/raw_manifest.csv")
    parser.add_argument("--output-dir", default="data/raw/video_frames")
    parser.add_argument("--frames-per-scene", type=int, default=6)
    parser.add_argument("--scene-threshold", type=float, default=0.55)
    parser.add_argument("--max-scenes", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos for a smoke test.")
    parser.add_argument("--log-every", type=int, default=25, help="Print progress every N videos.")
    parser.add_argument("--workers", type=int, default=1, help="Number of video workers. Try 4, 6, or 8 first.")
    parser.add_argument("--verbose", action="store_true", help="Print one line for every processed video.")
    return parser.parse_args()


def process_video(task: dict[str, object]) -> dict[str, object]:
    video_index = int(task["video_index"])
    video_path = Path(str(task["path"]))
    output_dir = Path(str(task["output_dir"]))
    started_at = time.time()
    frames, scene_count = sample_scene_aware_frames_from_path(
        video_path,
        frames_per_scene=int(task["frames_per_scene"]),
        scene_threshold=float(task["scene_threshold"]),
        max_scenes=int(task["max_scenes"]),
    )
    stem = video_path.stem
    rows: list[dict[str, object]] = []
    for sample_index, frame in enumerate(frames):
        frame_path = output_dir / f"{video_index:06d}_{stem}_s{frame.scene_index:03d}_f{frame.frame_index:06d}.jpg"
        frame.image.save(frame_path, quality=95)
        rows.append(
            {
                "path": str(frame_path),
                "label": int(task["label"]),
                "source": task["source"],
                "fake_type": task["fake_type"],
                "identity_id": task["identity_id"],
                "video_id": task["video_id"],
                "video_path": str(video_path),
                "video_index": video_index,
                "scene_count": scene_count,
                "scene_index": frame.scene_index,
                "frame_index": frame.frame_index,
                "timestamp_sec": frame.timestamp_sec,
                "sample_index": sample_index,
            }
        )
    return {
        "ok": True,
        "video_index": video_index,
        "path": str(video_path),
        "rows": rows,
        "scene_count": scene_count,
        "frame_count": len(rows),
        "elapsed_sec": round(time.time() - started_at, 3),
    }


def build_tasks(videos: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for video_index, row in videos.iterrows():
        video_path = Path(str(row["path"]))
        tasks.append(
            {
                "video_index": int(video_index),
                "path": str(video_path),
                "label": int(row["label"]),
                "source": row["source"],
                "fake_type": row.get("fake_type", row["source"]),
                "identity_id": row.get("identity_id", ""),
                "video_id": row.get("video_id", video_path.stem),
                "output_dir": str(output_dir),
                "frames_per_scene": int(args.frames_per_scene),
                "scene_threshold": float(args.scene_threshold),
                "max_scenes": int(args.max_scenes),
            }
        )
    return tasks


def log_progress(
    completed: int,
    total: int,
    rows_count: int,
    failures_count: int,
    started_at: float,
    *,
    prefix: str = "progress",
) -> None:
    elapsed = max(time.time() - started_at, 1e-6)
    videos_per_minute = completed / (elapsed / 60)
    eta_min = ((total - completed) / max(videos_per_minute, 1e-6)) if completed else 0.0
    print(
        f"{prefix} [{completed}/{total}] rows={rows_count} failures={failures_count} "
        f"elapsed={elapsed / 60:.1f}m speed={videos_per_minute:.1f} videos/min eta={eta_min:.1f}m",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    videos = pd.read_csv(args.videos)
    if args.limit is not None:
        videos = videos.head(args.limit)
    required = {"path", "label", "source"}
    missing = required - set(videos.columns)
    if missing:
        raise ValueError(f"Video manifest missing columns: {sorted(missing)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(videos, output_dir, args)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    started_at = time.time()
    total_videos = len(tasks)
    log_every = max(1, int(args.log_every))
    cpu_count = os.cpu_count() or 1
    workers = max(1, min(int(args.workers), cpu_count))
    if workers >= 8:
        print(
            f"warning: workers={workers} can be disk/CPU heavy; reduce it if the machine becomes unresponsive",
            flush=True,
        )
    print(
        f"extracting scene-aware frames from {total_videos} videos "
        f"frames_per_scene={args.frames_per_scene} max_scenes={args.max_scenes} workers={workers}",
        flush=True,
    )
    log_progress(0, total_videos, len(rows), len(failures), started_at, prefix="start")

    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            try:
                result = process_video(task)
                video_rows = result["rows"]
                rows.extend(video_rows)
                if args.verbose:
                    print(
                        f"ok video={result['path']} frames={result['frame_count']} "
                        f"scenes={result['scene_count']} elapsed={result['elapsed_sec']}s",
                        flush=True,
                    )
            except Exception as exc:
                failures.append({"path": str(task["path"]), "reason": repr(exc)})
                print(f"failed video={task['path']} reason={exc!r}", flush=True)
            if completed % log_every == 0 or completed == total_videos:
                log_progress(completed, total_videos, len(rows), len(failures), started_at)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_video, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    result = future.result()
                    video_rows = result["rows"]
                    rows.extend(video_rows)
                    if args.verbose:
                        print(
                            f"ok video={result['path']} frames={result['frame_count']} "
                            f"scenes={result['scene_count']} elapsed={result['elapsed_sec']}s",
                            flush=True,
                        )
                except Exception as exc:
                    failures.append({"path": str(task["path"]), "reason": repr(exc)})
                    print(f"failed video={task['path']} reason={exc!r}", flush=True)
                if completed % log_every == 0 or completed == total_videos:
                    log_progress(completed, total_videos, len(rows), len(failures), started_at)

    rows.sort(key=lambda item: (int(item["video_index"]), int(item["sample_index"])))

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)
    failure_path = Path(args.output_csv).with_suffix(".failures.csv")
    with open(failure_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    elapsed = max(time.time() - started_at, 1e-6)
    print(
        f"wrote {len(rows)} frame rows to {args.output_csv}; "
        f"failures={len(failures)} failure_log={failure_path}; elapsed={elapsed / 60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
