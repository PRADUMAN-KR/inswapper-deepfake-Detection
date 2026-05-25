import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from core.model import load_from_checkpoint
from training.dataset import create_dataset
from training.metrics import compute_binary_metrics, fuse_detection_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector checkpoint.")
    parser.add_argument("--config", default="configs/convnext_tiny.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default="checkpoints/eval_predictions.csv")
    parser.add_argument("--verbose", action="store_true", help="Print dataset, checkpoint, and batch progress.")
    parser.add_argument("--log-every", type=int, default=50, help="Verbose progress interval in batches.")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = args.manifest or cfg["data"].get("test_manifest") or cfg["data"]["val_manifest"]
    if args.verbose:
        print(f"config              {args.config}")
        print(f"checkpoint          {args.checkpoint}")
        print(f"manifest            {manifest}")
        print(f"device              {device}")
    ds = create_dataset(
        manifest,
        image_size=cfg["model"]["image_size"],
        train=False,
        root_dir=cfg["data"].get("root_dir"),
        frequency_mode=cfg["model"].get("frequency_mode", "fft"),
    )
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"])
    if args.verbose:
        print(f"samples             {len(ds)}")
        print(f"batches             {len(loader)}")
        print(f"batch_size          {cfg['train']['batch_size']}")
    model = load_from_checkpoint(args.checkpoint, device=device, backbone=cfg["model"]["backbone"]).eval()

    labels: list[float] = []
    scores: list[float] = []
    real_fake: list[float] = []
    inswapper: list[float] = []
    boundary: list[float] = []
    log_every = max(1, args.log_every)
    for step, batch in enumerate(loader, start=1):
        outputs = model(
            batch["rgb"].to(device),
            frequency=batch["frequency"].to(device),
            return_dict=True,
        )
        labels.extend(batch["targets"]["real_fake"].tolist())
        real_fake.extend(torch.sigmoid(outputs["real_fake"]).flatten().cpu().tolist())
        inswapper.extend(torch.sigmoid(outputs["inswapper"]).flatten().cpu().tolist())
        boundary.extend(torch.sigmoid(outputs["boundary"]).flatten().cpu().tolist())
        if args.verbose and (step == 1 or step % log_every == 0 or step == len(loader)):
            print(f"evaluated batches   {step}/{len(loader)}")

    scores = fuse_detection_scores(
        real_fake=torch.tensor(real_fake).numpy(),
        inswapper=torch.tensor(inswapper).numpy(),
        boundary=torch.tensor(boundary).numpy(),
        weights=cfg.get("score_fusion"),
    ).tolist()

    metrics = compute_binary_metrics(labels, scores)
    preds = [int(score >= metrics.best_threshold) for score in scores]
    print(metrics)
    print(confusion_matrix(labels, preds))
    print(classification_report(labels, preds, target_names=["real", "fake"], zero_division=0))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "final_score", "real_fake", "inswapper", "boundary", "prediction"])
        writer.writerows(zip(labels, scores, real_fake, inswapper, boundary, preds, strict=True))
    if args.verbose:
        print(f"predictions_csv     {args.output}")


if __name__ == "__main__":
    main()
