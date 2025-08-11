"""Hydra-based training script for GraphDINO using PyTorch Lightning."""

import hydra
import pytorch_lightning as pl
import torch
from hydra import utils
from omegaconf import DictConfig

from deep_neuronmorpho.engine import (
    GraphDINOLightningModule,
)
from deep_neuronmorpho.engine.trainer_utils import build_dataloader


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train a GraphDINO model using a Hydra DictConfig."""

    print(cfg)
    if cfg.training.get("random_state") is not None:
        pl.seed_everything(cfg.training.random_state, workers=True)

    model = utils.instantiate(cfg.model)
    callbacks = utils.instantiate(cfg.training.callbacks)
    logger = utils.instantiate(cfg.training.logger)
    dataloaders = build_dataloader(cfg)
    lightning_module = GraphDINOLightningModule(model, cfg)
    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        logger=logger,
        callbacks=callbacks,
        devices="auto",
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        deterministic=bool(getattr(cfg.training, "deterministic", False)),
    )
    trainer.fit(
        lightning_module,
        dataloaders[0],
        dataloaders[1] if len(dataloaders) > 1 else None,
        ckpt_path=None,
    )

    print(" ✅ Training completed!")


if __name__ == "__main__":
    main()
