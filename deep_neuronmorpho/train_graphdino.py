"""
Training script for the GraphDINO model using PyTorch Lightning.
This replaces the functionality of the original ssl_trainer.py with a
Lightning-based implementation, using standard PyTorch DataLoaders
rather than torch-geometric specialized loaders.
"""

import pytorch_lightning as pl
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
    # Load configuration
    cfg = Config.load(config_file=config_file)

    # Set random seed if specified
    if hasattr(cfg.training, "random_state") and cfg.training.random_state is not None:
        pl.seed_everything(cfg.training.random_state, workers=True)

    # Setup logging and checkpoint directories
    logger, ckpts_dir = setup_logging(cfg)
    log_hyperparameters(logger, cfg)

    # Setup callbacks
    callbacks = setup_callbacks(cfg, ckpts_dir)

    # Setup dataloaders for GraphDINO
    dataloaders = build_dataloader(cfg)

    # Create model
    model = create_graphdino(cfg)

    # Create Lightning module
    lightning_module = GraphDINOLightningModule(model, cfg)

    # Create trainer
    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        logger=logger,
        callbacks=callbacks,
        devices="auto",
        accelerator="auto",
        deterministic=hasattr(cfg.training, "deterministic") and cfg.training.deterministic,
    )

    # Train the model
    trainer.fit(
        lightning_module,
        dataloaders[0],
        dataloaders[1] if len(dataloaders) > 1 else None,
        ckpt_path=checkpoint,
    )

    print(f"Training completed! Checkpoints saved to {ckpts_dir}")


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
