"""Training module."""

import torch
import typer

from deep_neuronmorpho.engine import ContrastiveTrainer
from deep_neuronmorpho.engine.trainer_utils import setup_dataloaders
from deep_neuronmorpho.models import MACGNN
from deep_neuronmorpho.utils import ModelConfig

app = typer.Typer()


def train_model(
    config_file: str,
    checkpoint: str | None = None,
) -> None:
    """Train a model using a configuration file."""
    conf = ModelConfig(config_file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloaders = setup_dataloaders(conf, datasets=["contra_train", "eval_train", "eval_test"])

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
    config_file: str = typer.Argument(  # noqa: B008,
        ...,
        help="The configuration file.",
    ),
    checkpoint: str = typer.Option(  # noqa: B008,
        None,
        "--checkpoint",
        "-c",
        help="The checkpoint file to load from.",
    ),
) -> None:
    """CLI for training a model using a configuration file."""
    train_model(config_file, checkpoint)


if __name__ == "__main__":
    app()
