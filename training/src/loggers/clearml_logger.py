import tempfile
import warnings
from typing import Any, Dict, Optional, Union

from clearml import Task
from PIL import Image
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only


class ClearMLLogger(Logger):
    """
    Custom ClearML Logger for PyTorch Lightning 2.x
    Logs:
        - Scalars (metrics)
        - Hyperparameters
        - Model checkpoints
        - Images via .experiment.add_image(...) for compatibility
    """

    def __init__(
        self,
        project_name: str = "Default",
        task_name: str = "Experiment",
        tags: Optional[list] = None,
        reuse_last_task_id: bool = False,
    ):
        super().__init__()
        self.project_name = project_name
        self.task_name = task_name
        self.tags = tags
        self.reuse_last_task_id = reuse_last_task_id

        self._task = Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=tags,
            reuse_last_task_id=reuse_last_task_id,
        )
        self._logger = self._task.get_logger()

        # Monkey-patch `experiment.add_image` to work like TensorBoard
        self.experiment = self._task
        self.experiment.add_image = self._add_image_wrapper

    @rank_zero_only
    def _add_image_wrapper(self, tag, img_tensor, global_step=None, **kwargs):
        """
        Mimics TensorBoardLogger.add_image using ClearML report_image
        Accepts a torch.Tensor image (C,H,W) or PIL Image
        """
        # Convert torch tensor to PIL Image if needed
        self._task.mark_started()
        if hasattr(img_tensor, "shape") and len(img_tensor.shape) == 3:
            import torchvision

            img_tensor = torchvision.transforms.ToPILImage()(img_tensor)
        elif isinstance(img_tensor, Image.Image):
            pass
        else:
            raise ValueError(f"Unsupported image type: {type(img_tensor)}")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_tensor.save(tmp.name, format="PNG")
            artifact_path = tmp.name

        self._task.upload_artifact(name=tag, artifact_object=artifact_path)

    @property
    def name(self) -> str:
        return "ClearMLLogger"

    @property
    def version(self) -> str:
        return self._task.id

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Any]):
        flat_params = params if isinstance(params, dict) else vars(params)
        for k, v in flat_params.items():
            self._task.connect_configuration({"hparams": {k: v}})

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        self._task.mark_started()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for k, v in metrics.items():
                if hasattr(v, "item"):
                    try:
                        v = v.item()
                    except Exception:
                        continue
                if isinstance(v, (float, int)):
                    self._logger.report_scalar(
                        title="Metrics",
                        series=k,
                        value=v,
                        iteration=step,
                    )

    @rank_zero_only
    def log_model_checkpoint(self, filepath: str):
        self._task.update_output_model(path=filepath, model_name=filepath.split("/")[-1])

    def finalize(self, status: str):
        self._task.close()
