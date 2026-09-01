from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_pipeline.components.models.base import ModelFactory
from eval_pipeline.registry import register_component

try:
    from monai.networks.nets import SwinUNETR
except ImportError as error:  # pragma: no cover - surfaced at config load
    raise ImportError("task3_conditional_swin_unetr requires monai: pip install --no-deps monai") from error


class _ConditionalRefinement(nn.Module):
    """Re-apply the domain embedding after the last decoder stage.

    The bottleneck embedding barely survives the trip to the output. MONAI's
    decoder blocks normalise with InstanceNorm, and a per-channel constant
    added to a feature map is exactly the quantity InstanceNorm subtracts back
    out, so only convolution border effects leak through: swapping the domain
    pair moves the output by ~1% of its magnitude, against ~7% for the 2D
    ConditionalUNet.

    The answer is not to drop normalisation - that mistake cost the ViT its
    whole first run, where an unnormalised decoder grew its features 12x and
    saturated the output Tanh beyond recovery. Normalise first and apply the
    embedding as a scale and shift *afterwards*: nothing downstream can cancel
    it, and the norm still bounds what the convolutions build up. This block
    sits between the last decoder stage and MONAI's 1x1 output convolution,
    which on its own would reduce a constant to a single scalar offset.

    The modulation and the second convolution start at zero, so the network
    begins as exactly the pretrained one and learns how much conditioning it
    wants.
    """

    def __init__(self, channels: int, num_domains: int) -> None:
        super().__init__()
        self.source_embedding = nn.Embedding(num_domains, channels)
        self.target_embedding = nn.Embedding(num_domains, channels)
        groups = math.gcd(32, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.modulation = nn.Linear(channels, 2 * channels)
        for zeroed in (self.modulation, self.conv2):
            nn.init.zeros_(zeroed.weight)
            nn.init.zeros_(zeroed.bias)

    def forward(
        self, x: torch.Tensor, source_domain: torch.Tensor, target_domain: torch.Tensor
    ) -> torch.Tensor:
        conditioning = self.source_embedding(source_domain) + self.target_embedding(target_domain)
        h = self.conv1(F.leaky_relu(self.norm1(x), 0.01))
        scale, shift = self.modulation(conditioning).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.conv2(F.leaky_relu(h, 0.01))
        return x + h


class ConditionalSwinUNETR(nn.Module):
    """Swin UNETR with the same domain conditioning as ConditionalUNet.

    MONAI's forward is replicated here rather than called, so the source and
    target embeddings can be added at the bottleneck (encoder10's output, the
    deepest tensor before the decoder). That is the same injection point the
    2D ConditionalUNet uses.

    The pretrained encoder is the self-supervised SwinViT from Tang et al.
    (CVPR 2022), trained on roughly 5k *CT* volumes - not MRI. Swin UNETR was
    validated on brain MRI in the BraTS 2021 paper, but by fine-tuning these
    CT-pretrained weights, which is exactly what this class sets up.
    """

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        num_domains: int = 15,
        feature_size: int = 48,
        pretrained_path: str | None = None,
        use_v2: bool = False,
        freeze_encoder: bool = False,
        dropout_path_rate: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.net = SwinUNETR(
            in_channels=input_channels,
            out_channels=output_channels,
            feature_size=feature_size,
            spatial_dims=3,
            use_v2=use_v2,
            dropout_path_rate=dropout_path_rate,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_checkpoint=use_checkpoint,
        )
        if pretrained_path:
            self.load_pretrained(pretrained_path)

        bottleneck_channels = feature_size * 16
        # 5*m + c -> (m \in {t1, t2, t2*}) and (c \in {0.1T, 1.5T, 3T, 5T, 7T}) -> {0...14}
        self.source_embedding = nn.Embedding(num_domains, bottleneck_channels)
        self.target_embedding = nn.Embedding(num_domains, bottleneck_channels)
        self.condition_head = _ConditionalRefinement(feature_size, num_domains)
        # SwinUNETR's head is a plain conv for segmentation logits; translation
        # needs the same bounded output as the other Task 3 models.
        self.activation = nn.Tanh()

        if freeze_encoder:
            for parameter in self.net.swinViT.parameters():
                parameter.requires_grad_(False)

    def load_pretrained(self, path: str | Path) -> None:
        """Load either released Swin UNETR checkpoint format.

        Two exist and they are not interchangeable:

        * the self-supervised SwinViT (model_swinvit.pt) - encoder only,
          1-channel CT, DataParallel-prefixed keys, handled by MONAI's own
          load_from mapper;
        * a BraTS21 fold checkpoint - the full encoder *and* decoder trained on
          brain MRI, but with in_channels=4 and out_channels=3.

        For the BraTS one the head is dropped (3 tumour classes are meaningless
        for regression) and the 4-channel patch embedding is summed down to one
        channel. Summing is exact, not an approximation: feeding one image to
        all four input channels of the original conv computes precisely the
        same thing as convolving it once with the summed kernel.
        """
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        weights = checkpoint.get("state_dict", checkpoint)

        # The SSL checkpoint carries the encoder alone; a BraTS fold carries the
        # convolutional encoder/decoder stages too. That is what tells them apart.
        if not any("encoder1." in key or "decoder1." in key for key in weights):
            self.net.load_from(checkpoint if "state_dict" in checkpoint else {"state_dict": weights})
            print(f"loaded self-supervised SwinViT encoder from {Path(path).name}", flush=True)
            return

        adapted: dict[str, torch.Tensor] = {}
        target = self.net.state_dict()
        for key, value in weights.items():
            name = key[len("module."):] if key.startswith("module.") else key
            if name not in target:
                continue
            if name.startswith("out."):
                continue  # segmentation head, replaced by the regression head
            expected = target[name]
            if value.shape != expected.shape:
                # Every mismatch is on the input-channel axis: BraTS stacks four
                # MRI contrasts where this model takes one image. Summing the
                # kernel over those channels is exact, not an approximation -
                # feeding one image to all four inputs of the original
                # convolution computes precisely that. It applies to the Swin
                # patch embedding and to the convolutional stem alike.
                differs_only_in_input_channels = (
                    expected.shape[1] == 1
                    and value.shape[0] == expected.shape[0]
                    and value.shape[2:] == expected.shape[2:]
                )
                if differs_only_in_input_channels:
                    value = value.sum(dim=1, keepdim=True)
                else:
                    continue
            adapted[name] = value

        missing, unexpected = self.net.load_state_dict(adapted, strict=False)
        print(
            f"loaded {len(adapted)}/{len(target)} tensors from {Path(path).name} "
            f"(skipped {len(missing)} not in checkpoint, {len(unexpected)} unexpected)",
            flush=True,
        )

    def forward(
        self,
        image: torch.Tensor,
        target_domain: torch.Tensor,
        source_domain: torch.Tensor | None = None,
    ) -> torch.Tensor:
        net = self.net
        hidden = net.swinViT(image, net.normalize)
        enc0 = net.encoder1(image)
        enc1 = net.encoder2(hidden[0])
        enc2 = net.encoder3(hidden[1])
        enc3 = net.encoder4(hidden[2])

        bottleneck = net.encoder10(hidden[4])
        source_domain = target_domain if source_domain is None else source_domain
        conditioning = self.source_embedding(source_domain) + self.target_embedding(target_domain)
        bottleneck = bottleneck + conditioning[:, :, None, None, None]

        decoded = net.decoder5(bottleneck, hidden[3])
        decoded = net.decoder4(decoded, enc3)
        decoded = net.decoder3(decoded, enc2)
        decoded = net.decoder2(decoded, enc1)
        decoded = net.decoder1(decoded, enc0)
        decoded = self.condition_head(decoded, source_domain, target_domain)
        return self.activation(net.out(decoded))

    def optimizer_param_groups(self, learning_rate: float, encoder_scale: float) -> list[dict]:
        """Split by what the checkpoint actually restored, not by encoder/decoder.

        A BraTS fold restores the convolutional decoder too, so splitting on
        swinViT alone would hand pretrained decoder weights the full fresh-layer
        learning rate. Only the domain embeddings and the regression head are
        randomly initialised, and only those get the fast rate.
        """
        pretrained, fresh = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            is_fresh = not name.startswith("net.") or name.startswith("net.out.")
            (fresh if is_fresh else pretrained).append(parameter)
        groups = [{"params": fresh, "lr": learning_rate}]
        if pretrained:
            groups.append({"params": pretrained, "lr": learning_rate * encoder_scale})
        return groups


@register_component("task3_conditional_swin_unetr", category="model")
class ConditionalSwinUNETRFactory(ModelFactory[ConditionalSwinUNETR]):
    def build(self) -> ConditionalSwinUNETR:
        model = ConditionalSwinUNETR(
            input_channels=int(self.params.get("input_channels", 1)),
            output_channels=int(self.params.get("output_channels", 1)),
            num_domains=int(self.params.get("num_domains", 15)),
            feature_size=int(self.params.get("feature_size", 48)),
            pretrained_path=self.params.get("pretrained_path") or None,
            use_v2=bool(self.params.get("use_v2", False)),
            freeze_encoder=bool(self.params.get("freeze_encoder", False)),
            dropout_path_rate=float(self.params.get("dropout_path_rate", 0.0)),
            drop_rate=float(self.params.get("drop_rate", 0.0)),
            attn_drop_rate=float(self.params.get("attn_drop_rate", 0.0)),
            use_checkpoint=bool(self.params.get("use_checkpoint", False)),
        )
        checkpoint = self.params.get("checkpoint")
        device = self._device()
        if checkpoint:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state.get("model", state))
        return model.to(device)

    def _device(self) -> torch.device:
        configured = self.params.get("device")
        if isinstance(configured, list):
            configured = configured[0]
        if isinstance(configured, int):
            configured = f"cuda:{configured}"
        device = torch.device(configured or "cpu")
        return torch.device("cpu") if device.type == "cuda" and not torch.cuda.is_available() else device
