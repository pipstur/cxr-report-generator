import timm
import torch
from torch import nn

from training.src.models.components.loss_functions import get_loss_function
from training.src.models.components.model_class import Model

torch.use_deterministic_algorithms(True, warn_only=True)


class EfficientFormer(Model):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        freeze_backbone: bool,
        grayscale: bool,
        num_classes: int,
        loss_function: str,
        focal_loss_parameters: dict,
    ):
        super().__init__(num_classes=num_classes)
        self.criterion = get_loss_function(loss_function, focal_loss_parameters)

        backbone = timm.create_model("efficientformerv2_s0", pretrained=True, num_classes=0)
        if grayscale:
            conv = backbone.stem.conv1.conv
            backbone.stem.conv1.conv = nn.Conv2d(
                1,
                conv.out_channels,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                bias=conv.bias is not None,
            )

            with torch.no_grad():
                old_weights = backbone.stem.conv1.conv.weight
                new_weights = old_weights.mean(dim=1, keepdim=True)
                backbone.stem.conv1.conv.weight = nn.Parameter(new_weights)

        num_filters = int(backbone.num_features)
        self.feature_extractor = backbone

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(num_filters, self.num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import rootutils

    root = rootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "efficientformer.yaml")
    _ = hydra.utils.instantiate(cfg)
