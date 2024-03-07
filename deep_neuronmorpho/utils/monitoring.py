"""Utilities for monitoring the training process."""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime as dt
from logging import Logger
from pathlib import Path
from typing import Any, Collection

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing_extensions import TypeAlias


class ProgressBar:
    """A class that wraps the tqdm progress bar to simplify tracking progress of iterable objects.

    Args:
        iterable (Collection): An iterable object to be tracked.
        desc (str, optional): A short description of the progress bar. Defaults to "".
        percent_increment (int, optional): The percentage of progress between each update.
        Defaults to 5.
        bar_format (str, optional): The format of the progress bar.
        **kwargs: Additional keyword arguments that can be passed to the tqdm function.


    Attributes:
        iterable (Collection): The iterable object being tracked.
        desc (str): The short description of the progress bar.
        num_iterations (int): The total number of iterations in the iterable.
        increment_value (int): The number of iterations between each update.
        pbar (tqdm): The underlying tqdm progress bar object.
    """

    def __init__(
        self,
        iterable: Collection,
        desc: str = "",
        percent_increment: int = 5,
        bar_format: str = (
            "{desc}[{n_fmt}/{total_fmt}]{percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]"
        ),
        **kwargs: Any,
    ) -> None:
        self.iterable = iterable
        self.desc = desc
        self.num_iterations = len(iterable)
        self.increment_value = int(np.ceil(self.num_iterations * percent_increment / 100))
        self.pbar = tqdm(
            iterable,
            desc=desc,
            total=self.num_iterations,
            bar_format=bar_format,
            miniters=self.increment_value,
            **kwargs,
        )

    def __iter__(self) -> Any:
        """Iterate over the iterable and update the progress bar.

        Yields:
            Any: The next item in the iterable.
        """
        for item in self.pbar:
            yield item


class EventLogger:
    """Class for logging event progress."""

    def __init__(self, log_dir: Path | str, expt_name: str, to_file: bool = True) -> None:
        self.log_dir = Path(log_dir)
        self.expt_name = expt_name
        self.to_file = to_file
        self.logger: Logger | None = None

    def setup_logger(self) -> None:
        """Create a logger for logging training progress."""
        logger = logging.getLogger(self.expt_name)
        logger.setLevel(logging.INFO)
        # log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        # log to file if specified
        if self.to_file:
            # add timestamp if none in expt_name
            fname_pattern = re.compile(r"\d{4}[-_]\d{2}[-_]\d{2}[-_]\d{2}h?[-_]\d{2}m?")
            if fname_pattern.search(self.expt_name) is None:
                timestamp = dt.now().strftime("%Y_%m_%d_%Hh_%Mm")
                log_filename = f"{timestamp}-{self.expt_name}.log"
            else:
                log_filename = f"{self.expt_name}.log"
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(self.log_dir / log_filename)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self.logger = logger

    def check_logger(self) -> None:
        """Check if logger exists, if not, setup the logger."""
        if self.logger is None:
            self.setup_logger()

    def message(self, message: str, level: str = "info") -> None:
        """Log a message with the specified level."""
        self.check_logger()
        assert self.logger is not None
        if level.lower() == "info":
            self.logger.info(message)
        elif level.lower() == "warning":
            self.logger.warning(message)
        elif level.lower() == "error":
            self.logger.error(message)
        else:
            raise ValueError(f"Unknown logging level: {level}")


def extract_data(
    content: str,
    pattern: str,
    num_values: int = 1,
) -> pd.Series | tuple[pd.Series, pd.Series]:
    """Extract data from string using regex pattern.

    Args:
        content (str):  The string to extract data from.
        pattern (str): The regex pattern to use for extraction.
        num_values (int, optional): Number of values to extract per line. Defaults to 1.

    Returns:
        pd.Series | tuple[pd.Series, pd.Series]: A pd.Series object containing the extracted data.
    """
    data: tuple = tuple(pd.Series(dtype=float) for _ in range(num_values))
    for match in re.finditer(pattern, content):
        epoch = int(match.group(1))
        values = tuple(float(match.group(i)) for i in range(2, num_values + 2))
        for series, value in zip(data, values, strict=True):
            series[epoch] = value
    return data if num_values > 1 else data[0]


def plot_series(
    data: pd.Series,
    title: str,
    ylabel: str,
    ax: plt.Axes | None = None,
    **kwargs: Any,
) -> None:
    """Plot a pd.Series object."""
    if not ax:
        ax = plt.gca()

    ax.plot(data, lw=2, **kwargs)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel("Epoch", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()


@dataclass
class ContrastiveLogData:
    """Parses the log file to extract training losses and evaluation accuracies.

    Also provides a method (`plot_results()`) to visualize the extracted data.

    Attributes:
        file (Path): The path to the log file.
        train_loss (pd.Series): Series of training loss values.
        eval_acc (pd.Series): Series of evaluation accuracy values.
        expt_name (str): The name of the model.
    """

    file: Path
    train_loss: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    eval_acc: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    _expt_name: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Initialize object."""
        self.file = Path(self.file)
        if not self.file.exists() or self.file.stat().st_size == 0:
            raise ValueError(f"{self.file} is missing or empty.")
        self._train_loss_pattern = r"Epoch (\d+)/(?:\d+): Train Loss: (\d+\.\d+)"
        self._eval_acc_pattern = r"Epoch (\d+)/(?:\d+): Benchmark Test accuracy: (\d+\.\d+)"
        self._parse_log_file()

    def __repr__(self) -> str:
        """String representation of the logfile."""
        train_loss_last = self.train_loss.iat[-1] if not self.train_loss.empty else "N/A"
        eval_acc_last = self.eval_acc.iat[-1] if not self.eval_acc.empty else "N/A"
        return f"""{self.__class__.__name__}(file={self.file!r},
        expt_name={self.expt_name!r},
        last_train_loss={train_loss_last}, last_eval_acc={eval_acc_last})
        """

    def _parse_log_file(self) -> None:
        with self.file.open("r") as f:
            content = f.read()
        self.train_loss = extract_data(content=content, pattern=self._train_loss_pattern)  # type: ignore
        self.eval_acc = extract_data(content=content, pattern=self._eval_acc_pattern)  # type: ignore

    @property
    def expt_name(self) -> str:
        """Get model name from the log file name."""
        if self._expt_name is None:
            pattern = r"(?<=\d{4}_\d{2}_\d{2}_\d{2}h_\d{2}m-).+(?=.log)"
            match = re.search(pattern, self.file.name)
            self._expt_name = match.group() if match else "N/A"

        return self._expt_name

    @property
    def total_epochs(self) -> int:
        """Get the total number of epochs model was trained for."""
        return len(self.train_loss)

    def plot_results(self) -> None:
        """Plot training loss and evaluation accuracy."""
        fig, axs = plt.subplots(nrows=2, figsize=(10, 6))
        plot_series(self.train_loss, "Training Loss", "Loss", color="black", ax=axs[0])

        if not self.eval_acc.empty:
            plot_series(
                self.eval_acc,
                "Benchmark Classification Accuracy",
                "Accuracy",
                color="red",
                ax=axs[1],
            )
        else:
            fig.delaxes(axs[1])

        fig.suptitle(f"Training results: {self.expt_name}", fontsize=18)
        plt.tight_layout()


@dataclass
class SupervisedLogData:
    """Parses the log file to extract losses and evaluation accuracies.

    Also provides a method (`plot_results()`) to visualize the extracted data.

    Attributes:
        file (Path): The path to the log file.
        train_loss (pd.Series): Training loss values.
        val_loss (pd.Series): Validation loss values.
        train_acc (pd.Series): Training accuracy values.
        val_acc (pd.Series): Validation accuracy values.
        expt_name (str): The name of the model.
    """

    file: Path
    train_loss: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    val_loss: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    train_acc: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    val_acc: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    _expt_name: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Initialize object."""
        self.file = Path(self.file)
        if not self.file.exists() or self.file.stat().st_size == 0:
            raise ValueError(f"{self.file} is missing or empty.")
        self._loss_pattern = r"Epoch (\d+)/(?:\d+): Train Loss: (\d+\.\d+) \| Val Loss: (\d+\.\d+)"
        self._acc_pattern = r"Epoch (\d+)/(?:\d+): Train acc: (\d+\.\d+) \| Val acc: (\d+\.\d+)"

        self._parse_log_file()

    def __repr__(self) -> str:
        """String representation of the logfile."""
        train_loss_last = self.train_loss.iat[-1] if not self.train_loss.empty else "N/A"
        val_loss_last = self.val_loss.iat[-1] if not self.val_loss.empty else "N/A"
        train_acc_last = self.train_acc.iat[-1] if not self.train_acc.empty else "N/A"
        val_acc_last = self.val_acc.iat[-1] if not self.val_acc.empty else "N/A"

        return f"""{self.__class__.__name__}(file={self.file!r},
        expt_name={self.expt_name!r},
        last_train_loss={train_loss_last}, last_val_loss={val_loss_last},
        last_train_acc= {train_acc_last}, last_val_acc={val_acc_last})
        """

    def _parse_log_file(self) -> None:
        with self.file.open("r") as f:
            content = f.read()
        self.train_loss, self.val_loss = extract_data(
            content=content, pattern=self._loss_pattern, num_values=2
        )
        self.train_acc, self.val_acc = extract_data(
            content=content, pattern=self._acc_pattern, num_values=2
        )

    @property
    def expt_name(self) -> str:
        """Get model name from the log file name."""
        if self._expt_name is None:
            pattern = r"(?<=\d{4}_\d{2}_\d{2}_\d{2}h_\d{2}m-).+(?=.log)"
            match = re.search(pattern, self.file.name)
            self._expt_name = match.group() if match else "N/A"

        return self._expt_name

    @property
    def total_epochs(self) -> int:
        """Get the total number of epochs model was trained for."""
        return len(self.train_loss)

    def plot_results(self) -> None:
        """Plot training and validation loss and accuracy."""
        fig, axs = plt.subplots(nrows=2, figsize=(10, 6))
        plot_series(self.train_loss, "", "Loss", ax=axs[0], label="Train loss")
        plot_series(self.val_loss, "", "Loss", ax=axs[0], label="Validation loss")
        plot_series(self.train_acc, "", "Accuracy", ax=axs[1], label="Train acc.")
        plot_series(self.val_acc, "", "Accuracy", ax=axs[1], label="Validation acc.")
        axs[0].legend()
        axs[1].legend()
        fig.suptitle(f"Training results: {self.expt_name}", fontsize=18)
        plt.tight_layout()


LogData: TypeAlias = ContrastiveLogData | SupervisedLogData


@dataclass
class ExperimentResults:
    """Dataclass for loading log data from several runs of a contrastive learning experiment."""

    expt_dir: str | Path
    supervised: bool = False

    def __post_init__(self) -> None:
        """Load all log files from the experiment directory."""
        self.expt_dir = Path(self.expt_dir)
        log_files = [
            f
            for d in self.expt_dir.iterdir()
            if d.is_dir() and d.name != "old_bad_expts"
            for f in (d / "logs").glob("*.log")
        ]

        self.log_data: list[LogData] = [
            SupervisedLogData(file) if self.supervised else ContrastiveLogData(file)
            for file in log_files
        ]

        self.expts = {log.expt_name: log for log in self.log_data}
        self.expt_names = list(self.expts.keys())

    def plot_results(self) -> None:
        """Plot evaluation results for all runs in directory."""
        for log in self.log_data:
            log.plot_results()

    def __getitem__(self, key: str) -> LogData:
        """Get log data for a specific experiment name."""
        return self.expts[key]

    def __repr__(self) -> str:
        """String representation of the ExperimentResults object."""
        return f"{self.__class__.__name__}(expts: {self.expt_names})"
