from pathlib import Path

import pytorch_lightning as pl
import torch
import typer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from deep_neuronmorpho.engine import (
    ContrastiveGraphModule,
    NTXEntLoss,
    setup_dataloaders,
    setup_experiment_results,
)
from deep_neuronmorpho.models import MACGNN, MACGNNv2
from deep_neuronmorpho.utils import Config

app = typer.Typer()


def train_model(
    config_file: str,
    checkpoint: str | None = None,
) -> None:
    """Train a model using a configuration file."""
    conf = Config.from_yaml(config_file=config_file)
    if conf.training.random_state is not None:
        pl.seed_everything(conf.training.random_state, workers=True)

    # Generate unique experiment name and directory
    expt_name, expt_dir = setup_experiment_results(conf)

    dataloaders = setup_dataloaders(
        conf,
        datasets=["contra_train", "eval_train"],
        pin_memory=torch.cuda.is_available(),
        num_workers=9,
    )

    model_checkpoint = ModelCheckpoint(
        dirpath=Path(expt_dir) / "ckpts",
        filename=f"{expt_name}-{{step:07d}}-{{train_loss:.2f}}",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=10 * conf.training.logging_steps,
        monitor="train_loss",
        mode="min",
    )
    early_stopping = (
        EarlyStopping(
            monitor="train_loss",
            patience=conf.training.patience,
            mode="min",
        )
        if conf.training.patience
        else None
    )

    # load model and loss function
    model = MACGNNv2(conf.model) if "v2" in conf.model.name.lower() else MACGNN(conf.model)
    loss_fn = NTXEntLoss(conf.training.contra_loss_temp)
    # create training module and trainer
    contrastive_model = ContrastiveGraphModule(model, conf, loss_fn=loss_fn)
    trainer = pl.Trainer(
        # max_epochs=conf.training.epochs,
        max_steps=conf.training.max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        deterministic=True,
        logger=pl.loggers.TensorBoardLogger(Path(expt_dir) / "logs"),
        log_every_n_steps=conf.training.logging_steps,
        callbacks=[model_checkpoint, early_stopping],
    )
    # Train the model
    trainer.fit(
        contrastive_model,
        train_dataloaders=dataloaders["contra_train"],
        val_dataloaders=dataloaders["eval_train"],
        ckpt_path=checkpoint,
    )


@app.command()
def cli_train_model(
    config_file: str = typer.Argument(..., help="The configuration file."),
    checkpoint: str = typer.Option(
        None, "--checkpoint", "-c", help="The checkpoint file to load from."
    ),
) -> None:
    """CLI for training a model using a configuration file."""
    train_model(config_file, checkpoint)


if __name__ == "__main__":
    app()
