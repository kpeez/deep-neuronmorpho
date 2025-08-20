import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train_model(cfg: DictConfig) -> None:
    """Train a model using a configuration file."""
    if cfg.training.random_state is not None:
        pl.seed_everything(cfg.training.random_state, workers=True)

    lightning_module = instantiate(cfg.training.lightning_module)
    datamodule = instantiate(cfg.datamodule)
    trainer = instantiate(cfg.trainer)
    trainer.fit(
        lightning_module,
        datamodule=datamodule,
        ckpt_path=None,
    )


if __name__ == "__main__":
    train_model()
