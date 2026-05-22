import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from core.model import ConvNeXtTinyDetector
from training.dataset import DeepfakeDataset, ZarrDeepfakeDataset, create_dataset
from training.losses import MultiTaskDetectionLoss
from training.trainer import EarlyStopping, save_checkpoint, train_epoch, val_epoch
from training.utils import (
    build_warmup_cosine_scheduler,
    count_labels,
    load_yaml,
    resolve_device,
    seed_worker,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train INSwapper detector.")
    parser.add_argument("--config", default="configs/convnext_tiny.yaml")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def build_loader(dataset: DeepfakeDataset | ZarrDeepfakeDataset, cfg: dict, train: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    shuffle = train
    if train and cfg["train"].get("balanced_sampler", True):
        counts = count_labels(dataset.labels)
        weights = [1.0 / max(1, counts[label]) for label in dataset.labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
        shuffle = False

    num_workers = int(cfg["train"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def auto_focal_alpha(labels: list[int]) -> float:
    counts = count_labels(labels)
    total = max(1, counts[0] + counts[1])
    return counts[0] / total


def append_history(path: str | Path, row: dict[str, float | int | str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def phase_for_epoch(epoch: int, cfg: dict) -> str:
    phases = cfg["train"].get("phases", {})
    freeze_until = int(phases.get("freeze_backbone_until", 3))
    partial_until = int(phases.get("unfreeze_last_stages_until", 15))
    if epoch < freeze_until:
        return "freeze_backbone"
    if epoch < partial_until:
        return "unfreeze_last_stages"
    return "unfreeze_full"


def phase_schedule_text(cfg: dict) -> str:
    phases = cfg["train"].get("phases", {})
    freeze_until = int(phases.get("freeze_backbone_until", 3))
    partial_until = int(phases.get("unfreeze_last_stages_until", 15))
    return (
        f"phase_schedule: epochs 0-{max(0, freeze_until - 1)} freeze_backbone; "
        f"epochs {freeze_until}-{max(freeze_until, partial_until - 1)} unfreeze_last_stages; "
        f"epochs {partial_until}+ unfreeze_full"
    )


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def format_count(value: int) -> str:
    return f"{value / 1_000_000:.2f}M"


def print_section(title: str, rows: list[tuple[str, str]]) -> None:
    width = 76
    print("\n" + "=" * width, flush=True)
    print(title, flush=True)
    print("-" * width, flush=True)
    for key, value in rows:
        print(f"{key:<24} {value}", flush=True)
    print("=" * width, flush=True)


def print_subsection(title: str, rows: list[tuple[str, str]]) -> None:
    print("\n" + "-" * 76, flush=True)
    print(title, flush=True)
    print("-" * 76, flush=True)
    for key, value in rows:
        print(f"{key:<24} {value}", flush=True)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)
    device = resolve_device(cfg["train"].get("device", "auto"))
    torch.backends.cudnn.benchmark = bool(cfg["train"].get("cudnn_benchmark", True))
    checkpoint = torch.load(args.resume, map_location=device) if args.resume else None
    start_epoch = int(checkpoint["epoch"]) + 1 if checkpoint else 0

    train_ds = create_dataset(
        cfg["data"]["train_manifest"],
        image_size=cfg["model"]["image_size"],
        train=True,
        root_dir=cfg["data"].get("root_dir"),
        frequency_mode=cfg["model"].get("frequency_mode", "fft"),
    )
    val_ds = create_dataset(
        cfg["data"]["val_manifest"],
        image_size=cfg["model"]["image_size"],
        train=False,
        root_dir=cfg["data"].get("root_dir"),
        frequency_mode=cfg["model"].get("frequency_mode", "fft"),
    )
    train_loader = build_loader(train_ds, cfg, train=True, seed=seed)
    val_loader = build_loader(val_ds, cfg, train=False, seed=seed)

    model = ConvNeXtTinyDetector(
        pretrained=cfg["model"].get("pretrained", True),
        backbone=cfg["model"]["backbone"],
        drop_path_rate=cfg["model"].get("drop_path_rate", 0.1),
    ).to(device)
    current_phase = phase_for_epoch(start_epoch, cfg)
    model.set_training_phase(current_phase)

    alpha = cfg["loss"].get("alpha", "auto")
    if alpha == "auto":
        alpha = auto_focal_alpha(train_ds.labels)
    criterion = MultiTaskDetectionLoss(
        focal_gamma=cfg["loss"].get("focal_gamma", 2.0),
        alpha=alpha,
        weights=cfg["loss"].get("task_weights"),
    )
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        betas=tuple(cfg["optimizer"].get("betas", [0.9, 0.999])),
    )
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    update_steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum_steps))
    total_steps = update_steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = int(cfg["scheduler"].get("warmup_epochs", 1) * update_steps_per_epoch)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=cfg["scheduler"].get("min_lr_ratio", 0.05),
    )
    scaler = GradScaler(device.type, enabled=cfg["train"].get("amp", True) and device.type == "cuda")
    best_auc = -1.0
    best_product_metric = -1.0
    best_monitor_value = -1.0
    early_stopping_metric = cfg["train"].get("early_stopping_metric", "product_score")
    early_stopping = EarlyStopping(
        patience=int(cfg["train"].get("early_stopping_patience", 6)),
        min_delta=float(cfg["train"].get("early_stopping_min_delta", 0.0)),
    )

    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            print("optimizer state shape changed after backbone unfreeze; continuing with a fresh optimizer")
        if checkpoint.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except ValueError:
                print("scheduler state shape changed after backbone unfreeze; continuing with a fresh scheduler")
        best_auc = float(checkpoint.get("metrics", {}).get("auc", -1.0))
        best_product_metric = float(checkpoint.get("metrics", {}).get("product_score", best_auc))
        best_monitor_value = float(checkpoint.get("metrics", {}).get(early_stopping_metric, best_product_metric))
        early_stopping.best = best_monitor_value

    alpha_display = "none" if alpha is None else f"{float(alpha):.4f}"
    total_params, trainable_params = count_trainable_parameters(model)
    print_section(
        "Training Run",
        [
            ("model", f"{cfg['model']['backbone']}  image_size={cfg['model']['image_size']}  pretrained={cfg['model'].get('pretrained', True)}"),
            ("backbone", f"uses_timm={model.uses_timm}  pretrained_requested={model.pretrained_requested}  pretrained_loaded={model.pretrained_loaded}"),
            ("input branches", f"rgb + {cfg['model'].get('frequency_mode', 'fft')} frequency"),
            ("device", f"{device}  amp={cfg['train'].get('amp', True)}"),
            ("parameters", f"total={format_count(total_params)}  trainable={format_count(trainable_params)}  phase={current_phase}"),
            ("data", f"train={len(train_ds)}  val={len(val_ds)}"),
            ("batches", f"train_steps={len(train_loader)}  val_steps={len(val_loader)}  batch_size={cfg['train']['batch_size']}"),
            ("schedule", f"epochs={cfg['train']['epochs']}  start_epoch={start_epoch}  grad_accum={grad_accum_steps}"),
            ("optimizer", f"AdamW  lr={cfg['optimizer']['lr']}  weight_decay={cfg['optimizer']['weight_decay']}"),
            ("scheduler", f"warmup_epochs={cfg['scheduler'].get('warmup_epochs', 1)}  min_lr_ratio={cfg['scheduler'].get('min_lr_ratio', 0.05)}"),
            ("loss", f"focal_gamma={cfg['loss'].get('focal_gamma', 2.0)}  alpha={alpha_display}"),
            ("task weights", str(cfg["loss"].get("task_weights"))),
            ("score fusion", str(cfg.get("score_fusion"))),
            ("sampler", f"balanced_sampler={cfg['train'].get('balanced_sampler', True)}"),
            ("phase schedule", phase_schedule_text(cfg).removeprefix("phase_schedule: ")),
            ("early stopping", f"metric={early_stopping_metric}  patience={cfg['train'].get('early_stopping_patience', 6)}  min_delta={cfg['train'].get('early_stopping_min_delta', 0.0)}"),
        ],
    )

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        epoch_started_at = time.time()
        next_phase = phase_for_epoch(epoch, cfg)
        if next_phase != current_phase:
            current_phase = next_phase
            model.set_training_phase(current_phase)
            total_params, trainable_params = count_trainable_parameters(model)
            optimizer = torch.optim.AdamW(
                filter(lambda parameter: parameter.requires_grad, model.parameters()),
                lr=cfg["optimizer"]["lr"],
                weight_decay=cfg["optimizer"]["weight_decay"],
                betas=tuple(cfg["optimizer"].get("betas", [0.9, 0.999])),
            )
            scheduler = build_warmup_cosine_scheduler(
                optimizer,
                total_steps=max(1, (cfg["train"]["epochs"] - epoch) * update_steps_per_epoch),
                warmup_steps=warmup_steps,
                min_lr_ratio=cfg["scheduler"].get("min_lr_ratio", 0.05),
            )
            print_subsection(
                "Training Phase Changed",
                [
                    ("phase", current_phase),
                    ("trainable params", f"{format_count(trainable_params)} / {format_count(total_params)}"),
                ],
            )

        print_subsection(
            f"Epoch {epoch:03d}/{cfg['train']['epochs']}",
            [
                ("phase", current_phase),
                ("train batches", str(len(train_loader))),
                ("val batches", str(len(val_loader))),
            ],
        )

        train_loss, lr = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            cfg["train"].get("amp", True),
            scheduler=scheduler,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=cfg["train"].get("max_grad_norm"),
            log_every=cfg["train"].get("log_every", 100),
            epoch=epoch,
            total_epochs=cfg["train"]["epochs"],
            score_fusion_weights=cfg.get("score_fusion"),
        )
        val_loss, metrics = val_epoch(
            model,
            val_loader,
            criterion,
            device,
            cfg["train"].get("amp", True),
            score_fusion_weights=cfg.get("score_fusion"),
            log_every=cfg["train"].get("val_log_every", cfg["train"].get("log_every", 100)),
            epoch=epoch,
            total_epochs=cfg["train"]["epochs"],
        )

        print_subsection(
            f"Epoch {epoch:03d} Summary",
            [
                ("phase", current_phase),
                ("elapsed", f"{(time.time() - epoch_started_at) / 60:.1f}m"),
                ("lr", f"{lr:.6g}"),
                ("loss", f"train={train_loss:.4f}  val={val_loss:.4f}"),
                ("product / auc", f"{metrics.product_score:.4f} / {metrics.auc:.4f}"),
                ("acc / precision", f"{metrics.accuracy:.4f} / {metrics.precision:.4f}"),
                ("recall / f1", f"{metrics.recall:.4f} / {metrics.f1:.4f}"),
                ("eer", f"{metrics.eer:.4f}"),
                ("fpr / fnr", f"{metrics.false_positive_rate:.4f} / {metrics.false_negative_rate:.4f}"),
                ("best threshold", f"{metrics.best_threshold:.3f}"),
            ],
        )
        append_history(
            cfg["paths"]["history_csv"],
            {
                "epoch": epoch,
                "phase": current_phase,
                "lr": lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "auc": metrics.auc,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "eer": metrics.eer,
                "false_positive_rate": metrics.false_positive_rate,
                "false_negative_rate": metrics.false_negative_rate,
                "product_score": metrics.product_score,
                "best_threshold": metrics.best_threshold,
                "true_negative": metrics.true_negative,
                "false_positive": metrics.false_positive,
                "false_negative": metrics.false_negative,
                "true_positive": metrics.true_positive,
            },
        )
        save_checkpoint(cfg["paths"]["last_checkpoint"], model, optimizer, scheduler, epoch, metrics, cfg)
        print(f"checkpoint last: {cfg['paths']['last_checkpoint']}", flush=True)
        if metrics.product_score > best_product_metric:
            best_product_metric = metrics.product_score
            best_auc = metrics.auc
            save_checkpoint(cfg["paths"]["best_checkpoint"], model, optimizer, scheduler, epoch, metrics, cfg)
            print(
                f"checkpoint best: {cfg['paths']['best_checkpoint']}  product={best_product_metric:.4f}  auc={best_auc:.4f}",
                flush=True,
            )
        monitor_value = float(getattr(metrics, early_stopping_metric))
        improved_monitor = monitor_value > best_monitor_value + float(cfg["train"].get("early_stopping_min_delta", 0.0))
        if improved_monitor:
            best_monitor_value = monitor_value
        should_stop = early_stopping.step(monitor_value)
        print_subsection(
            "Validation Detail",
            [
                ("best val f1", f"{best_monitor_value:.4f}"),
                ("current val f1", f"{metrics.f1:.4f}"),
                ("best threshold", f"{metrics.best_threshold:.3f}"),
                ("confusion matrix", f"tn={metrics.true_negative}  fp={metrics.false_positive}  fn={metrics.false_negative}  tp={metrics.true_positive}"),
                ("early stopping", f"metric={early_stopping_metric}  current={monitor_value:.4f}  best={early_stopping.best:.4f}  bad_epochs={early_stopping.bad_epochs}/{early_stopping.patience}"),
            ],
        )
        if should_stop:
            print(
                f"early stopping at epoch={epoch} metric={early_stopping_metric} "
                f"best={early_stopping.best:.4f}",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()
