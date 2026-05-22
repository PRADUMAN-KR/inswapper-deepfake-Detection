from pathlib import Path
from typing import Any

import torch
from torch import nn


class FrequencyBranch(nn.Module):
    """Small CNN branch for FFT/DCT/high-pass style artifacts."""

    def __init__(self, in_channels: int = 3, out_features: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_features, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_features),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvNeXtTinyDetector(nn.Module):
    """ConvNeXt-Tiny RGB branch + frequency branch + INSwapper-focused heads."""

    def __init__(
        self,
        pretrained: bool = False,
        backbone: str = "convnext_tiny",
        drop_path_rate: float = 0.1,
        allow_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pretrained_requested = pretrained
        self.uses_timm = False
        self.pretrained_loaded = False
        try:
            import timm

            self.rgb_backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
                drop_path_rate=drop_path_rate,
            )
            rgb_features = int(self.rgb_backbone.num_features)
            self.uses_timm = True
            self.pretrained_loaded = pretrained
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    f"Could not create {backbone}. Install timm before training or serving production checkpoints."
                ) from exc
            rgb_features = 768
            self.rgb_backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.Conv2d(64, rgb_features, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(rgb_features),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        self.rgb_features = rgb_features
        self.frequency_features = 256
        self.frequency_branch = FrequencyBranch(out_features=self.frequency_features)
        self.fusion = nn.Sequential(
            nn.Linear(self.rgb_features + self.frequency_features, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict(
            {
                "real_fake": nn.Linear(512, 1),
                "inswapper": nn.Linear(512, 1),
                "boundary": nn.Linear(512, 1),
                "quality": nn.Linear(512, 3),
            }
        )

    def forward(
        self,
        rgb: torch.Tensor,
        frequency: torch.Tensor | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        rgb_feature = self.rgb_backbone(rgb).flatten(1)
        if frequency is None:
            frequency = torch.zeros_like(rgb)
        frequency_feature = self.frequency_branch(frequency)
        fused = self.fusion(torch.cat([rgb_feature, frequency_feature], dim=1))
        outputs = {name: head(fused) for name, head in self.heads.items()}
        if return_dict:
            return outputs
        return outputs["real_fake"]

    def set_backbone_trainable(self, trainable: bool) -> None:
        classifier_tokens = ("frequency_branch", "fusion", "heads")
        for name, parameter in self.named_parameters():
            parameter.requires_grad = trainable or any(token in name for token in classifier_tokens)

    def set_training_phase(self, phase: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True
        if phase == "freeze_backbone":
            self.set_backbone_trainable(False)
        elif phase == "unfreeze_last_stages":
            for name, parameter in self.named_parameters():
                if name.startswith(("frequency_branch", "fusion", "heads")):
                    parameter.requires_grad = True
                elif "stages.2" in name or "stages.3" in name:
                    parameter.requires_grad = True
                elif name.startswith("rgb_backbone"):
                    parameter.requires_grad = False
        elif phase == "unfreeze_full":
            for parameter in self.parameters():
                parameter.requires_grad = True
        else:
            raise ValueError(f"Unknown training phase: {phase}")


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def load_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | str,
    backbone: str = "convnext_tiny",
    strict: bool = True,
) -> ConvNeXtTinyDetector:
    path = Path(checkpoint_path)
    checkpoint: dict[str, Any] = torch.load(path, map_location=device)
    model_config = checkpoint.get("config", {}).get("model", {}) if isinstance(checkpoint, dict) else {}
    model = ConvNeXtTinyDetector(
        pretrained=False,
        backbone=model_config.get("backbone", backbone),
        drop_path_rate=model_config.get("drop_path_rate", 0.1),
        allow_fallback=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(_strip_module_prefix(state_dict), strict=strict)
    return model.to(device)
