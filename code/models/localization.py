from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ConvBnAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LocalizationNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        # Spatially-aware direct regression:
        # predict point heatmaps, then decode (x, y) with soft-argmax.
        self.decoder = nn.Sequential(
            ConvBnAct(512, 256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnAct(256, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnAct(128, 96),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBnAct(96, 64),
        )
        self.heatmap_head = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0)
        self.last_heatmaps: torch.Tensor | None = None

    @staticmethod
    def _meshgrid_xy(
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ys = torch.linspace(0.0, 1.0, steps=h, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, steps=w, device=device, dtype=dtype)
        try:
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        except TypeError:
            grid_y, grid_x = torch.meshgrid(ys, xs)
        return grid_x, grid_y

    def _decode(self, heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = heatmaps.shape
        flat = heatmaps.view(b, c, -1)
        probs = torch.softmax(flat, dim=-1).view(b, c, h, w)

        grid_x, grid_y = self._meshgrid_xy(h=h, w=w, device=heatmaps.device, dtype=heatmaps.dtype)
        grid_x = grid_x.view(1, 1, h, w)
        grid_y = grid_y.view(1, 1, h, w)

        exp_x = (probs * grid_x).sum(dim=(2, 3))
        exp_y = (probs * grid_y).sum(dim=(2, 3))

        coords = torch.stack(
            [
                exp_x[:, 0],
                exp_y[:, 0],
                exp_x[:, 1],
                exp_y[:, 1],
            ],
            dim=1,
        )

        dx = grid_x - exp_x.unsqueeze(-1).unsqueeze(-1)
        dy = grid_y - exp_y.unsqueeze(-1).unsqueeze(-1)
        var = (probs * (dx.square() + dy.square())).sum(dim=(2, 3))
        unc = torch.sqrt(torch.clamp(var, min=1e-8))
        return coords, unc

    @staticmethod
    def decode_argmax_centroid(heatmaps: torch.Tensor, radius: int = 2) -> torch.Tensor:
        probs = torch.sigmoid(heatmaps)
        b, c, h, w = probs.shape
        out = torch.zeros((b, 4), device=probs.device, dtype=probs.dtype)

        for bi in range(b):
            coord_list: list[torch.Tensor] = []
            for ci in range(min(c, 2)):
                hm = probs[bi, ci]
                idx = int(torch.argmax(hm).item())
                y0 = idx // w
                x0 = idx % w

                x1 = max(0, x0 - radius)
                x2 = min(w - 1, x0 + radius)
                y1 = max(0, y0 - radius)
                y2 = min(h - 1, y0 + radius)

                patch = hm[y1 : y2 + 1, x1 : x2 + 1]
                ys = torch.arange(y1, y2 + 1, device=probs.device, dtype=probs.dtype).view(-1, 1)
                xs = torch.arange(x1, x2 + 1, device=probs.device, dtype=probs.dtype).view(1, -1)
                weight = torch.clamp(patch.sum(), min=1e-8)
                cx = (patch * xs).sum() / weight
                cy = (patch * ys).sum() / weight

                coord_list.append(cx / float(max(w - 1, 1)))
                coord_list.append(cy / float(max(h - 1, 1)))

            out[bi, 0] = coord_list[0]
            out[bi, 1] = coord_list[1]
            out[bi, 2] = coord_list[2]
            out[bi, 3] = coord_list[3]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = self.decoder(feat)
        heatmaps = self.heatmap_head(feat)
        coords, unc = self._decode(heatmaps)
        self.last_heatmaps = heatmaps
        return torch.cat([coords, unc], dim=1)


def build_localization_model(pretrained: bool = True) -> LocalizationNet:
    return LocalizationNet(pretrained=pretrained)


def load_localization_checkpoint(path: str, device: torch.device) -> LocalizationNet:
    payload = torch.load(path, map_location=device)
    model = build_localization_model(pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model
