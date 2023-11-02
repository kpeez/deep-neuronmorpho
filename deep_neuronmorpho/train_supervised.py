"""Training module."""
import typer

from deep_neuronmorpho.data import NeuronGraphDataset
from deep_neuronmorpho.engine import SupervisedTrainer, setup_common_utilities
from deep_neuronmorpho.models import MACGNN

app = typer.Typer()


def train_model(
    config_file: str,
    checkpoint: str | None = None,
    gpu: int | None = None,
) -> None:
    """Train a model using a configuration file."""
    conf, device = setup_common_utilities(config_file, gpu)
    dataset = NeuronGraphDataset(conf.dirs.graph_data, dataset_name=conf.datasets.eval_train)
    # create model and trainer
    # TODO: add support for other models (parse model name from config file)
    model = MACGNN(conf.model)
    trainer = SupervisedTrainer(model=model, config=conf, dataset=dataset, device=device)
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
