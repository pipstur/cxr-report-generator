import timm
import torch
from torch import nn

from training.src.models.components.loss_functions import get_loss_function
from training.src.models.components.model_class import Model

torch.use_deterministic_algorithms(True, warn_only=True)


class ConvNeXtV2Atto(Model):
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

        backbone = timm.create_model(
            "convnextv2_atto",
            pretrained=True,
            num_classes=0,
            in_chans=1 if grayscale else 3,
        )

        self.feature_extractor = backbone

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(backbone.num_features, self.num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import rootutils

    root = rootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "convnextv2.yaml")
    _ = hydra.utils.instantiate(cfg)
