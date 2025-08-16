import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from deep_neuronmorpho.engine import (
    ContrastiveGraphModule,
    NeuronDataModule,
)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train_model(cfg: DictConfig) -> None:
    """Train a model using a configuration file."""
    if cfg.training.random_state is not None:
        pl.seed_everything(cfg.training.random_state, workers=True)
    model = instantiate(cfg.model)
    # TODO: add loss function to lightning module
    # loss_fn = create_loss_fn(cfg.training.loss_fn, temp=cfg.training.loss_temp)
    contrastive_module: pl.LightningModule = ContrastiveGraphModule(model, cfg)
    datamodule: pl.LightningDataModule = NeuronDataModule(cfg)
    trainer: pl.Trainer = instantiate(cfg.training.trainer)
    trainer.fit(
        contrastive_module,
        datamodule=datamodule,
        ckpt_path=None,
    )


if __name__ == "__main__":
    train_model()
