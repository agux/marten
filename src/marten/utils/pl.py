from tqdm.auto import tqdm
# from pytorch_lightning.callbacks import Callback, ProgressBar   ## for pl ^2.0.0
from pytorch_lightning.callbacks import ProgressBarBase   ## for pl ^1.9.4


class GlobalProgressBar(ProgressBarBase):
    """Global progress bar.
    Originally from: https://github.com/Lightning-AI/lightning/issues/765
    """

    def __init__(
        self, global_progress: bool = True, leave_global_progress: bool = True
    ):
        super().__init__()

        self.global_progress = global_progress
        self.global_desc = "Epoch: {epoch}/{max_epoch}"
        self.leave_global_progress = leave_global_progress
        self.global_pb = None
        self.step_pb = None

    def on_fit_start(self, trainer, pl_module):
        desc = self.global_desc.format(
            epoch=trainer.current_epoch + 1, max_epoch=trainer.max_epochs
        )

        self.global_pb = tqdm(
            desc=desc,
            total=trainer.max_epochs,
            initial=trainer.current_epoch,
            leave=self.leave_global_progress,
            disable=not self.global_progress,
        )

    # def on_train_epoch_start(self, trainer, pl_module):
    #     self.step_pb = tqdm(
    #         desc="Training",
    #         total=len(trainer.train_dataloader),
    #         leave=False,
    #     )

    # def on_train_epoch_end(self, trainer, pl_module):
    #     self.step_pb.close()
    #     self.step_pb = None

    #     # Set description
    #     desc = self.global_desc.format(
    #         epoch=trainer.current_epoch + 1, max_epoch=trainer.max_epochs
    #     )
    #     self.global_pb.set_description(desc)

    #     # # Set logs and metrics
    #     # logs = pl_module.logs
    #     # for k, v in logs.items():
    #     #     if isinstance(v, torch.Tensor):
    #     #         logs[k] = v.squeeze().item()
    #     # self.global_pb.set_postfix(logs)

    #     # Update progress
    #     self.global_pb.update(1)

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     self.step_pb.update(1)

    # def on_fit_end(self, trainer, pl_module):
    #     self.global_pb.close()
    #     self.global_pb = None

    # def on_validation_epoch_start(self, trainer, pl_module) -> None:
    #     pass

    # def on_validation_epoch_end(self, trainer, pl_module) -> None:
    #     pass

    # def on_test_epoch_start(
    #     self, trainer, pl_module
    # ) -> None:
    #     """Called when the test epoch begins."""

    # def on_test_epoch_end(
    #     self, trainer, pl_module
    # ) -> None:
    #     """Called when the test epoch ends."""

    # def on_validation_batch_start(
    #     self,
    #     trainer,
    #     pl_module,
    #     batch,
    #     batch_idx: int,
    #     dataloader_idx: int = 0,
    # ) -> None:
    #     """Called when the validation batch begins."""

    # def on_validation_batch_end(
    #     self,
    #     trainer,
    #     pl_module,
    #     outputs,
    #     batch,
    #     batch_idx: int,
    #     dataloader_idx: int = 0,
    # ) -> None:
    #     """Called when the validation batch ends."""

    # def on_test_batch_start(
    #     self,
    #     trainer,
    #     pl_module,
    #     batch,
    #     batch_idx: int,
    #     dataloader_idx: int = 0,
    # ) -> None:
    #     """Called when the test batch begins."""

    # def on_test_batch_end(
    #     self,
    #     trainer,
    #     pl_module,
    #     outputs,
    #     batch,
    #     batch_idx: int,
    #     dataloader_idx: int = 0,
    # ) -> None:
    #     """Called when the test batch ends."""

    # def on_validation_start(
    #     self, trainer, pl_module
    # ) -> None:
    #     """Called when the validation loop begins."""

    # def on_validation_end(
    #     self, trainer, pl_module
    # ) -> None:
    #     """Called when the validation loop ends."""
