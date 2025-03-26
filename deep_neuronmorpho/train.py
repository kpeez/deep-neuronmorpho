import pytorch_lightning as pl
import torch
import typer

from deep_neuronmorpho.engine import (
    ContrastiveGraphModule,
    create_loss_fn,
    create_model,
    create_trainer,
    log_hyperparameters,
    setup_callbacks,
    setup_dataloaders,
    setup_logging,
)
from deep_neuronmorpho.utils import Config


def train_model(config_file: str, checkpoint: str | None = None) -> None:
    """Train a model using a configuration file."""
    cfg = Config.from_yaml(config_file=config_file)
    if cfg.training.random_state is not None:
        pl.seed_everything(cfg.training.random_state, workers=True)
    logger, ckpts_dir = setup_logging(cfg)

    log_hyperparameters(logger, cfg)
    callbacks = setup_callbacks(cfg, ckpts_dir)
    dataloaders = setup_dataloaders(
        cfg,
        datasets=["contra_train", "eval_train"],
        pin_memory=torch.cuda.is_available(),
        num_workers=9,
    )
    model = create_model(cfg.model.name, cfg.model)
    loss_fn = create_loss_fn(cfg.training.loss_fn, temp=cfg.training.loss_temp)
    contrastive_model = ContrastiveGraphModule(model, cfg, loss_fn=loss_fn)
    trainer = create_trainer(cfg, logger, callbacks)

    trainer.fit(
        contrastive_model,
        train_dataloaders=dataloaders["contra_train"],
        val_dataloaders=dataloaders["eval_train"],
        ckpt_path=checkpoint,
    )


if __name__ == "__main__":
    app = typer.Typer()

    @app.command()
    def cli_train_model(
        config_file: str = typer.Argument(..., help="The configuration file."),
        checkpoint: str = typer.Option(
            None, "--checkpoint", "-c", help="The checkpoint file to load from."
        ),
    ) -> None:
        """CLI for training a model using a configuration file."""
        train_model(config_file, checkpoint)

    app()
