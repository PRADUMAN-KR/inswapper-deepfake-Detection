from pathlib import Path
import time
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from training.metrics import BinaryMetrics, compute_binary_metrics, fuse_detection_scores


def _binary_counts(scores: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> tuple[int, int, int, int]:
    preds = scores >= threshold
    labels = labels >= 0.5
    tp = int((preds & labels).sum().item())
    tn = int((~preds & ~labels).sum().item())
    fp = int((preds & ~labels).sum().item())
    fn = int((~preds & labels).sum().item())
    return tp, tn, fp, fn


def _counts_to_metrics(tp: int, tn: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    total = max(1, tp + tn + fp + fn)
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return accuracy, precision, recall, f1


def _format_loss_parts(loss_parts: dict[str, float]) -> str:
    names = ["real_fake", "inswapper", "boundary", "quality"]
    return "  ".join(f"{name}={loss_parts.get(name, 0.0):.4f}" for name in names)


def _print_progress_block(title: str, lines: list[str]) -> None:
    print(title, flush=True)
    for line in lines:
        print(f"  {line}", flush=True)


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if value > self.best + self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    amp: bool = True,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float | None = None,
    log_every: int = 100,
    epoch: int | None = None,
    total_epochs: int | None = None,
    score_fusion_weights: dict[str, float] | None = None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    grad_accum_steps = max(1, grad_accum_steps)
    last_lr = optimizer.param_groups[0]["lr"]
    started_at = time.time()
    log_every = max(0, int(log_every))
    total_steps = len(loader)
    tp = tn = fp = fn = 0
    loss_parts_running: dict[str, float] = {}
    last_grad_norm = 0.0
    for step, batch in enumerate(loader, start=1):
        rgb = batch["rgb"].to(device, non_blocking=True)
        frequency = batch["frequency"].to(device, non_blocking=True)
        targets = {
            name: value.to(device, non_blocking=True)
            for name, value in batch["targets"].items()
        }
        with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(rgb, frequency=frequency, return_dict=True)
            raw_loss = criterion(outputs, targets)
            loss = (raw_loss[0] if isinstance(raw_loss, tuple) else raw_loss) / grad_accum_steps
            loss_parts = raw_loss[1] if isinstance(raw_loss, tuple) else {"total": float(loss.detach().cpu())}
        if scaler is not None and amp and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = step % grad_accum_steps == 0 or step == len(loader)
        if should_step:
            if scaler is not None and amp and device.type == "cuda":
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    last_grad_norm = float(grad_norm.detach().cpu())
                else:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                    last_grad_norm = float(grad_norm.detach().cpu())
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    last_grad_norm = float(grad_norm.detach().cpu())
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                    last_grad_norm = float(grad_norm.detach().cpu())
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            last_lr = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * grad_accum_steps * rgb.size(0)
        for name, value in loss_parts.items():
            loss_parts_running[name] = loss_parts_running.get(name, 0.0) + float(value) * rgb.size(0)
        with torch.no_grad():
            weights = score_fusion_weights or {"real_fake": 0.55, "inswapper": 0.30, "boundary": 0.15}
            scores = (
                weights["real_fake"] * torch.sigmoid(outputs["real_fake"]).flatten()
                + weights["inswapper"] * torch.sigmoid(outputs["inswapper"]).flatten()
                + weights["boundary"] * torch.sigmoid(outputs["boundary"]).flatten()
            )
            labels = targets["real_fake"].flatten() >= 0.5
            batch_tp, batch_tn, batch_fp, batch_fn = _binary_counts(scores, labels)
            tp += batch_tp
            tn += batch_tn
            fp += batch_fp
            fn += batch_fn
        if log_every and (step == 1 or step % log_every == 0 or step == total_steps):
            elapsed = max(time.time() - started_at, 1e-6)
            steps_per_minute = step / (elapsed / 60)
            eta_min = (total_steps - step) / max(steps_per_minute, 1e-6)
            average_loss = running_loss / max(1, step * loader.batch_size)
            average_parts = {
                name: value / max(1, step * loader.batch_size)
                for name, value in loss_parts_running.items()
            }
            epoch_display = "?" if epoch is None else f"{epoch:03d}"
            epoch_total_display = "?" if total_epochs is None else str(total_epochs)
            accuracy, precision, recall, f1 = _counts_to_metrics(tp, tn, fp, fn)
            _print_progress_block(
                f"[train] epoch {epoch_display}/{epoch_total_display}  step {step}/{total_steps}",
                [
                    f"loss={average_loss:.4f}  {_format_loss_parts(average_parts)}",
                    f"acc@0.5={accuracy:.4f}  precision@0.5={precision:.4f}  recall@0.5={recall:.4f}  f1@0.5={f1:.4f}",
                    f"grad_norm={last_grad_norm:.3f}  lr={last_lr:.6g}  speed={steps_per_minute:.1f} steps/min  eta={eta_min:.1f}m",
                ],
            )
    return running_loss / len(loader.dataset), last_lr


@torch.inference_mode()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool = True,
    score_fusion_weights: dict[str, float] | None = None,
    log_every: int = 100,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> tuple[float, BinaryMetrics]:
    model.eval()
    running_loss = 0.0
    labels_all: list[float] = []
    real_fake_all: list[float] = []
    inswapper_all: list[float] = []
    boundary_all: list[float] = []
    tp = tn = fp = fn = 0
    loss_parts_running: dict[str, float] = {}
    started_at = time.time()
    log_every = max(0, int(log_every))
    total_steps = len(loader)
    for step, batch in enumerate(loader, start=1):
        rgb = batch["rgb"].to(device, non_blocking=True)
        frequency = batch["frequency"].to(device, non_blocking=True)
        targets = {
            name: value.to(device, non_blocking=True)
            for name, value in batch["targets"].items()
        }
        with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(rgb, frequency=frequency, return_dict=True)
            raw_loss = criterion(outputs, targets)
            loss = raw_loss[0] if isinstance(raw_loss, tuple) else raw_loss
            loss_parts = raw_loss[1] if isinstance(raw_loss, tuple) else {"total": float(loss.detach().cpu())}
        running_loss += loss.item() * rgb.size(0)
        for name, value in loss_parts.items():
            loss_parts_running[name] = loss_parts_running.get(name, 0.0) + float(value) * rgb.size(0)
        labels_all.extend(targets["real_fake"].flatten().detach().cpu().tolist())
        real_fake_scores = torch.sigmoid(outputs["real_fake"]).flatten()
        inswapper_scores = torch.sigmoid(outputs["inswapper"]).flatten()
        boundary_scores = torch.sigmoid(outputs["boundary"]).flatten()
        real_fake_all.extend(real_fake_scores.detach().cpu().tolist())
        inswapper_all.extend(inswapper_scores.detach().cpu().tolist())
        boundary_all.extend(boundary_scores.detach().cpu().tolist())
        weights = score_fusion_weights or {"real_fake": 0.55, "inswapper": 0.30, "boundary": 0.15}
        fused_batch = (
            weights["real_fake"] * real_fake_scores
            + weights["inswapper"] * inswapper_scores
            + weights["boundary"] * boundary_scores
        )
        batch_tp, batch_tn, batch_fp, batch_fn = _binary_counts(fused_batch, targets["real_fake"].flatten())
        tp += batch_tp
        tn += batch_tn
        fp += batch_fp
        fn += batch_fn
        if log_every and (step == 1 or step % log_every == 0 or step == total_steps):
            elapsed = max(time.time() - started_at, 1e-6)
            steps_per_minute = step / (elapsed / 60)
            eta_min = (total_steps - step) / max(steps_per_minute, 1e-6)
            average_loss = running_loss / max(1, step * loader.batch_size)
            average_parts = {
                name: value / max(1, step * loader.batch_size)
                for name, value in loss_parts_running.items()
            }
            epoch_display = "?" if epoch is None else f"{epoch:03d}"
            epoch_total_display = "?" if total_epochs is None else str(total_epochs)
            accuracy, precision, recall, f1 = _counts_to_metrics(tp, tn, fp, fn)
            _print_progress_block(
                f"[val] epoch {epoch_display}/{epoch_total_display}  step {step}/{total_steps}",
                [
                    f"loss={average_loss:.4f}  {_format_loss_parts(average_parts)}",
                    f"acc@0.5={accuracy:.4f}  precision@0.5={precision:.4f}  recall@0.5={recall:.4f}  f1@0.5={f1:.4f}",
                    f"speed={steps_per_minute:.1f} steps/min  eta={eta_min:.1f}m",
                ],
            )
    fused = fuse_detection_scores(
        real_fake=torch.tensor(real_fake_all).numpy(),
        inswapper=torch.tensor(inswapper_all).numpy(),
        boundary=torch.tensor(boundary_all).numpy(),
        weights=score_fusion_weights,
    )
    return running_loss / len(loader.dataset), compute_binary_metrics(labels_all, fused.tolist())


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    metrics: BinaryMetrics,
    config: dict[str, Any],
    threshold: float | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics.__dict__,
            "threshold": threshold if threshold is not None else metrics.best_threshold,
            "config": config,
        },
        path,
    )
