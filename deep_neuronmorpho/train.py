"""Training module."""

import torch
import typer

from deep_neuronmorpho.engine import ContrastiveTrainer, setup_dataloaders
from deep_neuronmorpho.models import MACGNN
from deep_neuronmorpho.utils import Config

app = typer.Typer()


def train_model(
    config_file: str,
    checkpoint: str | None = None,
    gpu: int | None = None,
) -> None:
    """Train a model using a configuration file."""
    conf = Config.from_yaml(config_file=config_file)
    device = f"cuda:{gpu}" if torch.cuda.is_available() and gpu is not None else "cpu"
    dataloaders = setup_dataloaders(
        conf,
        datasets=["contra_train", "eval_train", "eval_test"],
        pin_memory=True,
    )

    # create model and trainer
    # TODO: add support for other models (parse model name from config file)
    macgnn = MACGNN(conf.model)
    trainer = ContrastiveTrainer(
        model=macgnn,
        config=conf,
        dataloaders=dataloaders,
        device=device,
    )
    # start training
    trainer.fit(ckpt_file=checkpoint)


@app.command()
def cli_train_model(
    config_file: str = typer.Argument(..., help="The configuration file."),
    checkpoint: str = typer.Option(
        None, "--checkpoint", "-c", help="The checkpoint file to load from."
    ),
    gpu: int = typer.Option(None, "--gpu", "-g", help="The ID of the GPU to use."),
) -> None:
    """CLI for training a model using a configuration file."""
    train_model(config_file, checkpoint, gpu)


if __name__ == "__main__":
    app()
