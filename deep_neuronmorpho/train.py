import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train_model(cfg: DictConfig) -> None:
    """Train a model using a configuration file."""
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed, workers=True)

    lightning_module = instantiate(cfg.training.lightning_module)
    datamodule = instantiate(cfg.data)
    trainer = instantiate(cfg.training.trainer)
    trainer.fit(
        lightning_module,
        datamodule=datamodule,
        ckpt_path=cfg.ckpt_path,
    )


if __name__ == "__main__":
    train_model()
