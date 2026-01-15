import timm
import torch
from timm.models.helpers import adapt_input_conv
from torch import nn

from training.src.models.components.loss_functions import BinaryFocalLoss
from training.src.models.components.model_class import Model

torch.use_deterministic_algorithms(True, warn_only=True)


class MobileNetV4(Model):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        freeze_backbone: bool,
        grayscale: bool,
        num_classes: int,
    ):
        super().__init__(num_classes=num_classes)
        self.criterion = BinaryFocalLoss()

        backbone = timm.create_model(
            "mobilenetv4_conv_small_050.e3000_r224_in1k", pretrained=True, num_classes=0
        )
        if grayscale:
            conv = backbone.conv_stem
            conv.weight = nn.Parameter(adapt_input_conv(1, conv.weight))
            conv.in_channels = 1

        self.feature_extractor = backbone

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(1280, self.num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import roootutils

    root = roootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mobilenetv4.yaml")
    _ = hydra.utils.instantiate(cfg)
