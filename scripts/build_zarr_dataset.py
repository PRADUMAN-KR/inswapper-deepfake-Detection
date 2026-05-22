import argparse
import csv
import json
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


NUMERIC_COLUMNS = [
    "label",
    "is_inswapper",
    "boundary_label",
    "quality_label",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack processed face crops and labels into a chunked Zarr dataset.")
    parser.add_argument("--metadata", required=True, help="CSV with processed crop paths and labels.")
    parser.add_argument("--output", required=True, help="Output .zarr directory.")
    parser.add_argument("--root-dir", default=None, help="Base directory for relative image paths. Defaults to metadata parent.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1, help="Image decode workers. Try 2 or 4 for Zarr packing.")
    parser.add_argument("--log-every", type=int, default=500, help="Print progress every N successfully packed samples.")
    parser.add_argument("--verbose", action="store_true", help="Print one line for every packed sample.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def import_zarr():
    try:
        import zarr
    except ImportError as exc:
        raise SystemExit("zarr is not installed. Run: uv sync --extra train") from exc
    return zarr


def resolve_image_path(path: str, root_dir: Path) -> Path:
    image_path = Path(path)
    if image_path.is_absolute():
        return image_path
    return root_dir / image_path


def load_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def load_sample(index: int, path: str, root_dir: Path, image_size: int) -> dict[str, Any]:
    image_path = resolve_image_path(path, root_dir)
    try:
        image = load_image(image_path, image_size)
    except Exception as exc:
        return {"ok": False, "index": index, "path": image_path, "reason": repr(exc)}
    return {"ok": True, "index": index, "path": image_path, "image": image}


def create_array(group: Any, name: str, *, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: Any) -> Any:
    if hasattr(group, "create_array"):
        return group.create_array(name, shape=shape, chunks=chunks, dtype=dtype)
    return group.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype)


def write_metadata_copy(frame: pd.DataFrame, output: Path) -> None:
    frame.to_csv(output / "metadata.csv", index=False)


def log_progress(written: int, total: int, failures: int, started_at: float, *, prefix: str = "progress") -> None:
    elapsed = max(time.time() - started_at, 1e-6)
    samples_per_minute = written / (elapsed / 60)
    eta_min = ((total - written) / max(samples_per_minute, 1e-6)) if written else 0.0
    print(
        f"{prefix} [{written}/{total}] failures={failures} "
        f"elapsed={elapsed / 60:.1f}m speed={samples_per_minute:.1f} samples/min eta={eta_min:.1f}m",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    zarr = import_zarr()
    metadata_path = Path(args.metadata)
    output_path = Path(args.output)
    root_dir = Path(args.root_dir) if args.root_dir else metadata_path.parent

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    frame = pd.read_csv(metadata_path)
    frame = frame.dropna(how="all").reset_index(drop=True)
    required = {"path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Metadata has no samples: {metadata_path}")

    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            if column == "is_inswapper":
                fake_type = frame.get("fake_type", pd.Series(["real"] * len(frame))).astype(str).str.lower()
                source = frame.get("source", pd.Series([""] * len(frame))).astype(str).str.lower()
                frame[column] = ((fake_type.str.contains("inswapper")) | (source.str.contains("inswapper"))).astype(np.int8)
            elif column == "boundary_label":
                frame[column] = frame["label"].astype(np.int8)
            else:
                frame[column] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(output_path), mode="w")
    sample_count = len(frame)
    image_shape = (sample_count, args.image_size, args.image_size, 3)
    image_chunks = (min(args.chunk_size, sample_count), args.image_size, args.image_size, 3)

    images = create_array(group, "images", shape=image_shape, chunks=image_chunks, dtype=np.uint8)
    labels = create_array(group, "labels", shape=(sample_count,), chunks=(min(args.chunk_size, sample_count),), dtype=np.int8)
    is_inswapper = create_array(group, "is_inswapper", shape=(sample_count,), chunks=(min(args.chunk_size, sample_count),), dtype=np.int8)
    boundary = create_array(group, "boundary_label", shape=(sample_count,), chunks=(min(args.chunk_size, sample_count),), dtype=np.int8)
    quality = create_array(group, "quality_label", shape=(sample_count,), chunks=(min(args.chunk_size, sample_count),), dtype=np.int8)

    failures: list[dict[str, str]] = []
    written = 0
    log_every = max(1, int(args.log_every))
    started_at = time.time()
    workers = max(1, int(args.workers))
    print(
        f"packing {sample_count} samples into {output_path} image_size={args.image_size} "
        f"chunk_size={args.chunk_size} workers={workers}",
        flush=True,
    )
    log_progress(0, sample_count, len(failures), started_at, prefix="start")

    def write_loaded_sample(index: int, image_path: Path, image: np.ndarray) -> None:
        row = frame.iloc[index]
        images[index] = image
        labels[index] = int(row["label"])
        is_inswapper[index] = int(row["is_inswapper"])
        boundary[index] = int(row["boundary_label"])
        quality[index] = int(row["quality_label"])
        if args.verbose:
            print(f"packed sample index={index} path={image_path}", flush=True)

    if workers == 1:
        for index, row in frame.iterrows():
            image_path = resolve_image_path(str(row["path"]), root_dir)
            try:
                loaded = load_sample(index, str(row["path"]), root_dir, int(args.image_size))
                if not loaded["ok"]:
                    raise RuntimeError(str(loaded["reason"]))
                write_loaded_sample(int(loaded["index"]), Path(loaded["path"]), loaded["image"])
            except Exception as exc:
                failures.append({"index": str(index), "path": str(image_path), "reason": repr(exc)})
                print(f"failed sample index={index} path={image_path} reason={exc!r}", flush=True)
                continue
            written += 1
            if written % log_every == 0:
                log_progress(written, sample_count, len(failures), started_at)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = set()
            row_iter = frame.iterrows()
            max_pending = workers * 4

            def submit_next() -> bool:
                try:
                    index, row = next(row_iter)
                except StopIteration:
                    return False
                pending.add(executor.submit(load_sample, index, str(row["path"]), root_dir, int(args.image_size)))
                return True

            for _ in range(min(max_pending, sample_count)):
                submit_next()

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        loaded = future.result()
                        if loaded["ok"]:
                            write_loaded_sample(int(loaded["index"]), Path(loaded["path"]), loaded["image"])
                        else:
                            failures.append(
                                {
                                    "index": str(loaded["index"]),
                                    "path": str(loaded["path"]),
                                    "reason": str(loaded["reason"]),
                                }
                            )
                            print(
                                f"failed sample index={loaded['index']} path={loaded['path']} reason={loaded['reason']}",
                                flush=True,
                            )
                            submit_next()
                            continue
                    except Exception as exc:
                        failures.append({"index": "-1", "path": "", "reason": repr(exc)})
                        print(f"failed sample index=-1 path= reason={exc!r}", flush=True)
                        submit_next()
                        continue
                    written += 1
                    if written % log_every == 0:
                        log_progress(written, sample_count, len(failures), started_at)
                    submit_next()

    if failures:
        failure_path = output_path.with_suffix(".failures.csv")
        with open(failure_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "path", "reason"])
            writer.writeheader()
            writer.writerows(failures)
        raise RuntimeError(f"Failed to pack {len(failures)} samples. See {failure_path}")

    write_metadata_copy(frame, output_path)
    summary = {
        "metadata": str(metadata_path),
        "sample_count": sample_count,
        "image_size": args.image_size,
        "chunk_size": args.chunk_size,
        "label_counts": {str(key): int(value) for key, value in frame["label"].astype(int).value_counts().sort_index().items()},
        "fake_type_counts": {str(key): int(value) for key, value in frame.get("fake_type", pd.Series(["unknown"] * sample_count)).value_counts().items()},
    }
    group.attrs.update(summary)
    with open(output_path / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    log_progress(written, sample_count, len(failures), started_at, prefix="done")
    print(f"wrote {sample_count} samples to {output_path}", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
