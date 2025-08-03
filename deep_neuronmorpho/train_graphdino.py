"""
Training script for the GraphDINO model using PyTorch Lightning.
This replaces the functionality of the original ssl_trainer.py with a
Lightning-based implementation, using standard PyTorch DataLoaders
rather than torch-geometric specialized loaders.
"""

import pytorch_lightning as pl
import torch
import typer

from deep_neuronmorpho.engine import (
    GraphDINOLightningModule,
    log_hyperparameters,
    setup_callbacks,
    setup_logging,
)
from deep_neuronmorpho.engine.trainer_utils import build_dataloader
from deep_neuronmorpho.models import create_graphdino
from deep_neuronmorpho.utils import Config


def train_graphdino(config_file: str, checkpoint: str | None = None) -> None:
    """
    Train a GraphDINO model using PyTorch Lightning.

    Args:
        config_file: Path to the YAML configuration file
        checkpoint: Optional checkpoint to resume training from
    """
    cfg = Config.load(config_file=config_file)
    if hasattr(cfg.training, "random_state") and cfg.training.random_state is not None:
        pl.seed_everything(cfg.training.random_state, workers=True)
    logger, ckpts_dir = setup_logging(cfg)
    log_hyperparameters(logger, cfg)
    callbacks = setup_callbacks(cfg, ckpts_dir)
    dataloaders = build_dataloader(cfg)
    model = create_graphdino(cfg)
    lightning_module = GraphDINOLightningModule(model, cfg)
    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        logger=logger,
        callbacks=callbacks,
        devices="auto",
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        deterministic=hasattr(cfg.training, "deterministic") and cfg.training.deterministic,
    )
    trainer.fit(
        lightning_module,
        dataloaders[0],
        dataloaders[1] if len(dataloaders) > 1 else None,
        ckpt_path=checkpoint,
    )

    logger.info(f"Training completed! Checkpoints saved to {ckpts_dir}")


if __name__ == "__main__":
    app = typer.Typer()

    @app.command()
    def main(
        config_file: str = typer.Argument(..., help="The configuration file path."),
        checkpoint: str = typer.Option(
            None, "--checkpoint", "-c", help="The checkpoint file to load from."
        ),
    ) -> None:
        """Train a GraphDINO model using PyTorch Lightning."""
        train_graphdino(config_file, checkpoint)

    app()
