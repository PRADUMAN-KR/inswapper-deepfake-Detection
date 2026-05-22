
# InSwapper Detector: Open-Source Deepfake Face Swap Detection Pipeline

> 🚧 This project is currently under active development.
> APIs, architecture, and training pipeline may change frequently.

![Status](https://img.shields.io/badge/status-under--development-orange)


Open-source deepfake detection pipeline for detecting InSwapper face swaps, AI-generated face manipulation, forged face videos, and face-swap artifacts using ConvNeXt-Tiny, frequency features, scene-aware video sampling, and FastAPI inference.

## Layout

- `app/`: FastAPI app, routes, settings, request/response schemas.
- `core/`: pure model, preprocessing, inference, postprocessing logic.
- `training/`: offline training and evaluation only.
- `data/`: CSV manifests only; raw images stay out of git.
- `configs/`: YAML training configs.
- `scripts/`: preprocessing, splitting, and video-frame manifest utilities.

## Setup

Use Python 3.12 for the training/serving container. Very new Python versions may not have wheels for `insightface`, `onnxruntime`, or other ML packages yet.

```bash
uv sync --extra train --extra test
cp .env.example .env
```

Pip fallback:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
```

## Data

Create `data/train.csv`, `data/val.csv`, and `data/test.csv` with:

```csv
path,label,source
data/raw/real/0001.jpg,0,celebdf
data/raw/fake/0001.jpg,1,inswapper
```

Split by identity/source video to avoid leakage.

For the current local dataset layout, build the video manifest automatically:

```bash
uv run python scripts/build_video_manifest.py \
  --data-root data \
  --output-csv data/video_manifest.csv
```

This reads:

- `data/inswapper/original_videos` as real
- `data/inswapper/inswapper` as fake INSwapper
- `data/inswapper/uniface` as fake UniFace manipulated videos

Other original-video sources are intentionally excluded so the detector learns manipulation artifacts from the same source distribution instead of unrelated dataset differences. Training balance is handled by the sampler, not by dropping fake samples from the manifest.

## Train

For image datasets, create a raw manifest such as `data/raw_manifest.csv`.

For video datasets, first convert videos into scene-aware training frames and write the same raw manifest format:

```bash
python scripts/build_video_frame_manifest.py \
  --videos data/video_manifest.csv \
  --output-csv data/raw_manifest.csv \
  --output-dir data/raw/video_train_frames \
  --frames-per-scene 6
```

Create face crops and final metadata:

```bash
python scripts/build_processed_crop_manifest.py \
  --input-csv data/raw_manifest.csv \
  --output-csv data/processed_metadata.csv \
  --output-dir data/raw/processed_crops
```

Split safely by identity/video group:

```bash
python scripts/split_metadata.py \
  --metadata data/processed_metadata.csv \
  --train data/train.csv \
  --val data/val.csv \
  --test data/test.csv
```

For faster training on large datasets, pack each split into Zarr:

```bash
python scripts/build_zarr_dataset.py \
  --metadata data/train.csv \
  --output data/zarr/train.zarr \
  --root-dir . \
  --overwrite

python scripts/build_zarr_dataset.py \
  --metadata data/val.csv \
  --output data/zarr/val.zarr \
  --root-dir . \
  --overwrite

python scripts/build_zarr_dataset.py \
  --metadata data/test.csv \
  --output data/zarr/test.zarr \
  --root-dir . \
  --overwrite
```

To train from Zarr, set the config manifests to the `.zarr` directories:

```yaml
data:
  root_dir: .
  train_manifest: data/zarr/train.zarr
  val_manifest: data/zarr/val.zarr
  test_manifest: data/zarr/test.zarr
```

Then train the final ConvNeXt-Tiny multi-task detector:

```bash
python training/train.py --config configs/convnext_tiny.yaml
```

The best checkpoint is written to `checkpoints/best_model.pt`.
Training history is written to `checkpoints/history.csv`.

Final ConvNeXt-Tiny training recipe:

- production face detection with InsightFace before crop generation and inference
- pretrained `convnext_tiny` backbone from `timm`
- RGB branch plus frequency CNN branch
- optional Zarr-backed dataset loading for high-throughput training
- multi-task heads for real/fake, InSwapper, boundary artifacts, and quality/compression
- weighted multi-task loss with automatic class-balance alpha
- balanced sampler for real/fake imbalance
- phased fine-tuning: frozen backbone, last-stage unfreeze, full-model unfreeze
- AdamW with warmup plus cosine decay
- mixed precision, gradient clipping, optional gradient accumulation
- checkpoint selection by product metric
- score fusion and validation threshold sweep

## Final Training Pipeline

This is the training pipeline the project follows for production model development.

```mermaid
flowchart TD
    A[Raw Data] --> B{Input Type}

    B -->|Images| IMG_SRC[Image Sources<br/>Real / InSwapper]
    B -->|Videos| VID_SRC[Video Sources<br/>Real / InSwapper]

    VID_SRC --> VFE[Scene-Aware Frame Extraction<br/>scripts/build_video_frame_manifest.py]
    VFE --> SCENE_DETECT[Detect Scene Changes]
    SCENE_DETECT --> SCENE_SAMPLE[Sample 5-6 Frames Per Scene]
    SCENE_SAMPLE --> FRAME_SAVE[Save Extracted Frames]
    FRAME_SAVE --> FRAME_MANIFEST[Frame Manifest CSV]

    IMG_SRC --> IMG_MANIFEST[Image Manifest CSV]
    FRAME_MANIFEST --> RAW_MANIFEST[Unified Raw Manifest]
    IMG_MANIFEST --> RAW_MANIFEST

    RAW_MANIFEST --> FACE_DETECT[Face Detection<br/>RetinaFace / MediaPipe / YOLO Face]
    FACE_DETECT --> FACE_FOUND{Face Found?}

    FACE_FOUND -->|No| FACE_SKIP[Skip Sample + Log Failure]
    FACE_FOUND -->|Yes| FACE_BOX[Face Bounding Box]

    FACE_BOX --> CROP_GEN[Scene-Aware Crop Generation]
    CROP_GEN --> CROP_TIGHT[Tight Crop 1.1x]
    CROP_GEN --> CROP_EXPANDED[Expanded Crop 1.5x]
    CROP_GEN --> CROP_SCENE[Scene Crop 2.0x]

    CROP_TIGHT --> RESIZE_256[Resize to 256x256]
    CROP_EXPANDED --> RESIZE_256
    CROP_SCENE --> RESIZE_256

    RESIZE_256 --> CROP_SAVE[Save Processed Crops]
    CROP_SAVE --> META[Create Final Metadata CSV]

    META --> META_IMAGE[image_path]
    META --> META_LABEL[label<br/>0 real / 1 fake]
    META --> META_FAKE_TYPE[fake_type<br/>real / inswapper / uniface]
    META --> META_INSWAPPER[is_inswapper]
    META --> META_BOUNDARY[boundary_label]
    META --> META_QUALITY[quality_label]
    META --> META_SOURCE[source]
    META --> META_VIDEO[video_id]
    META --> META_IDENTITY[identity_id]

    META --> SAFE_SPLIT[Identity / Video Level Split]
    SAFE_SPLIT --> TRAIN_CSV[train.csv]
    SAFE_SPLIT --> VAL_CSV[val.csv]
    SAFE_SPLIT --> TEST_CSV[test.csv]

    TRAIN_CSV --> DATASET[DeepfakeDataset<br/>training/dataset.py]
    DATASET --> AUG[Albumentations Training Pipeline]

    AUG --> AUG_RESIZE[Resize 256x256]
    AUG --> AUG_FLIP[Horizontal Flip]
    AUG --> AUG_JPEG[JPEG Compression]
    AUG --> AUG_RECOMPRESS[Resize Recompression]
    AUG --> AUG_NOISE[Noise]
    AUG --> AUG_GBLUR[Gaussian Blur]
    AUG --> AUG_MBLUR[Motion Blur]
    AUG --> AUG_COLOR[Color Jitter]
    AUG --> AUG_SHARPEN[Sharpening]
    AUG --> AUG_NORM[Normalize]

    AUG_RESIZE --> AUG_RGB[Augmented RGB Crop]
    AUG_FLIP --> AUG_RGB
    AUG_JPEG --> AUG_RGB
    AUG_RECOMPRESS --> AUG_RGB
    AUG_NOISE --> AUG_RGB
    AUG_GBLUR --> AUG_RGB
    AUG_MBLUR --> AUG_RGB
    AUG_COLOR --> AUG_RGB
    AUG_SHARPEN --> AUG_RGB
    AUG_NORM --> AUG_RGB

    AUG_RGB --> RGB_TENSOR[RGB Tensor]
    AUG_RGB --> FREQ_GEN[Frequency Map Generator<br/>FFT / DCT / High-Pass]
    FREQ_GEN --> FREQ_TENSOR[Frequency Tensor]

    RGB_TENSOR --> LOADER[DataLoader]
    FREQ_TENSOR --> LOADER

    LOADER --> IMBALANCE{Class Imbalance Handling}
    IMBALANCE -->|Enabled| SAMPLER[WeightedRandomSampler]
    IMBALANCE -->|Optional| CLASS_WEIGHTS[Class Weights / Pos Weights]

    SAMPLER --> TRAIN_BATCHES[Training Batches]
    CLASS_WEIGHTS --> TRAIN_BATCHES

    VAL_CSV --> VAL_DATASET[Validation Dataset<br/>No Heavy Augmentation]
    VAL_DATASET --> VAL_BATCHES[Validation Batches]

    TRAIN_BATCHES --> MODEL[Final Detector Model]

    MODEL --> RGB_BRANCH[RGB Branch<br/>ConvNeXt-Tiny]
    MODEL --> FREQ_BRANCH[Frequency CNN Branch]

    RGB_BRANCH --> RGB_FEATURE[RGB Feature 768-D]
    FREQ_BRANCH --> FREQ_FEATURE[Frequency Feature 256-D]

    RGB_FEATURE --> FUSION[Feature Fusion<br/>Concat 1024-D]
    FREQ_FEATURE --> FUSION

    FUSION --> FUSION_MLP[Fusion MLP<br/>1024 to 512]

    FUSION_MLP --> HEAD_RF[Real/Fake Head]
    FUSION_MLP --> HEAD_INSWAPPER[InSwapper Head]
    FUSION_MLP --> HEAD_BOUNDARY[Boundary Artifact Head]
    FUSION_MLP --> HEAD_QUALITY[Quality / Compression Head]

    MODEL --> PHASE{Training Phase}

    PHASE -->|Epochs 1-3| PHASE_HEAD[Freeze ConvNeXt Backbone<br/>Train Frequency Branch + Fusion + Heads]
    PHASE -->|Epochs 4-15| PHASE_PARTIAL[Unfreeze ConvNeXt Stage 3-4<br/>Partial Fine-Tuning]
    PHASE -->|Epochs 16+| PHASE_FULL[Unfreeze Full Model<br/>Low LR Fine-Tuning]

    PHASE_HEAD --> FORWARD[Forward Pass]
    PHASE_PARTIAL --> FORWARD
    PHASE_FULL --> FORWARD

    FORWARD --> MULTI_LOSS[Multi-Task Loss]

    HEAD_RF --> LOSS_RF[BCE / Focal Loss<br/>Real/Fake]
    HEAD_INSWAPPER --> LOSS_INSWAPPER[BCE / Focal Loss<br/>InSwapper]
    HEAD_BOUNDARY --> LOSS_BOUNDARY[BCE / Focal Loss<br/>Boundary]
    HEAD_QUALITY --> LOSS_QUALITY[CrossEntropy Loss<br/>Quality]

    LOSS_RF --> MULTI_LOSS
    LOSS_INSWAPPER --> MULTI_LOSS
    LOSS_BOUNDARY --> MULTI_LOSS
    LOSS_QUALITY --> MULTI_LOSS

    MULTI_LOSS --> TOTAL_LOSS[Total Loss<br/>1.0 RF + 0.7 InSwapper + 0.4 Boundary + 0.2 Quality]

    TOTAL_LOSS --> AMP[AMP Mixed Precision]
    AMP --> ACCUM[Gradient Accumulation]
    ACCUM --> CLIP[Gradient Clipping]
    CLIP --> OPTIM[AdamW Optimizer]
    OPTIM --> LR_SCHED[Warmup + Cosine LR Scheduler]
    LR_SCHED --> NEXT_STEP[Next Training Step]

    VAL_BATCHES --> VAL_FORWARD[Validation Forward Pass]
    VAL_FORWARD --> VAL_SCORES[Sigmoid / Softmax Scores]

    VAL_SCORES --> SCORE_FUSION[Score Fusion]
    SCORE_FUSION --> FINAL_SCORE[final_score =<br/>0.55 real_fake<br/>+ 0.30 inswapper<br/>+ 0.15 boundary]

    FINAL_SCORE --> THRESHOLD_SWEEP[Threshold Sweep]
    THRESHOLD_SWEEP --> THRESH_F1[Best F1 Threshold]
    THRESHOLD_SWEEP --> THRESH_LOW_FPR[Low FPR Threshold]
    THRESHOLD_SWEEP --> THRESH_PRODUCT[Balanced Product Threshold]

    FINAL_SCORE --> VAL_METRICS[Validation Metrics]

    VAL_METRICS --> METRIC_AUC[AUC]
    VAL_METRICS --> METRIC_ACC[Accuracy]
    VAL_METRICS --> METRIC_PREC[Precision]
    VAL_METRICS --> METRIC_RECALL[Recall]
    VAL_METRICS --> METRIC_F1[F1]
    VAL_METRICS --> METRIC_EER[EER]
    VAL_METRICS --> METRIC_FPR[False Positive Rate]
    VAL_METRICS --> METRIC_FNR[False Negative Rate]
    VAL_METRICS --> METRIC_INSWAPPER[InSwapper Recall]
    VAL_METRICS --> METRIC_COMPRESSED[Compressed Real FPR]

    VAL_METRICS --> BEST_PRODUCT{Best Product Metric?}

    BEST_PRODUCT -->|Yes| BEST_CKPT[Save Best Checkpoint<br/>checkpoints/best_model.pt]
    BEST_PRODUCT -->|No| LAST_CKPT[Save Last Checkpoint<br/>checkpoints/last_epoch.pt]

    BEST_CKPT --> HISTORY[Log History<br/>checkpoints/history.csv]
    LAST_CKPT --> HISTORY

    HISTORY --> EARLY_STOP{Early Stopping?}

    EARLY_STOP -->|No| NEXT_STEP
    EARLY_STOP -->|Yes| DONE[Training Complete]

    DONE --> FINAL_EVAL[Final Evaluation<br/>training/evaluate.py]
    TEST_CSV --> FINAL_EVAL

    FINAL_EVAL --> TEST_REPORTS[Test Reports]

    TEST_REPORTS --> REPORT_REAL[Clean Real Performance]
    TEST_REPORTS --> REPORT_COMPRESSED[Compressed Real False Positives]
    TEST_REPORTS --> REPORT_INSWAPPER[InSwapper Recall]
    TEST_REPORTS --> REPORT_QUALITY[Low Quality Robustness]
    TEST_REPORTS --> REPORT_IDENTITY[Unseen Identity Generalization]
    TEST_REPORTS --> REPORT_SOURCE[Unseen Source Generalization]

    TEST_REPORTS --> PROD[Export Production Checkpoint<br/>Loaded by API / Inference Pipeline]
```

## Evaluate

```bash
python training/evaluate.py --config configs/convnext_tiny.yaml --checkpoint checkpoints/best_model.pt
```

## Serve

```bash
uvicorn app.main:app --reload
```

Endpoints:

- `GET /health`
- `GET /ready`
- `POST /detect`
- `POST /detect/batch`
- `POST /detect/video`
- `POST /admin/reload-model`
- `POST /admin/threshold`

Example:

```bash
curl -F "file=@face.jpg" http://localhost:8000/detect
```

Local inference without the API:

```bash
python scripts/predict_video.py clip.mp4 --checkpoint checkpoints/best_model.pt
```

## Scene-Aware Video Detection

The model still predicts one frame at a time with ConvNeXt-Tiny. For videos, the pipeline:

1. Scans the video using HSV histogram distance to find scene cuts.
2. Samples up to 6 evenly spaced frames inside each detected scene.
3. Runs the image detector on every sampled frame.
4. Averages frame fake probabilities into one video-level result.

This avoids wasting all samples on near-duplicate frames from one shot and gives you frame-level evidence for review.

## Production Notes

- Put the trained checkpoint at `checkpoints/best_model.pt` or set `INSWAPPER_MODEL_PATH`.
- Tune `INSWAPPER_THRESHOLD` from validation metrics, not from the test set.
- Default face detection is `INSWAPPER_FACE_DETECTOR=insightface`. Use `opencv_haar` only for local development experiments.



topics:?
Deepfake detection, InSwapper detector, face swap detection, AI-generated face detection, forged video detection, ConvNeXt deepfake model, deepfake artifact detection, face manipulation detection, FastAPI deepfake API, PyTorch deepfake detector