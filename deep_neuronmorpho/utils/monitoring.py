"""Utilities for monitoring the training process."""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime as dt
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Collection

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


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

    def __init__(self, log_dir: Path, expt_name: str, to_file: bool = True) -> None:
        self.log_dir = log_dir
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
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            log_filename = f"{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}-{self.expt_name}.log"
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


@dataclass
class LogData:
    """Class to parse and represent training log data.

    Parses the log file to extract training losses and evaluation accuracies.
    Provides a method (`plot_results()`) to visualize the extracted data.

    Attributes:
        file (Path): The path to the log file.
        train_loss (pd.Series): Series of training loss values.
        eval_acc (pd.Series): Series of evaluation accuracy values.
        model_name (str): The name of the model.
    """

    file: Path
    train_loss: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    eval_acc: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    _model_name: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Initialize object."""
        self.file = Path(self.file)
        if not self.file.exists() or self.file.stat().st_size == 0:
            raise ValueError(f"{self.file} is missing or empty.")
        self._parse_log_file()

    def __repr__(self) -> str:
        """String representation of the logfile."""
        train_loss_last = self.train_loss.iat[-1] if not self.train_loss.empty else "N/A"
        eval_acc_last = self.eval_acc.iat[-1] if not self.eval_acc.empty else "N/A"
        return f"""{self.__class__.__name__}(file={self.file!r},
        model_name={self.model_name!r},
        last_train_loss={train_loss_last}, last_eval_acc={eval_acc_last})
        """

    def _parse_log_file(self) -> None:
        with self.file.open("r") as f:
            content = f.read()
        self._extract_data(content, self._extract_train_loss)
        self._extract_data(content, self._extract_eval_acc)

    def _extract_data(self, content: str, extraction_func: Callable) -> None:
        for match in re.finditer(extraction_func.pattern, content):
            extraction_func(match)

    def _extract_train_loss(self, match: re.Match) -> None:
        epoch, loss = int(match.group(1)), float(match.group(2))
        self.train_loss[epoch] = loss

    _extract_train_loss.pattern = r"Epoch (\d+)/(?:\d+): Train Loss: (\d+\.\d+)"

    def _extract_eval_acc(self, match: re.Match) -> None:
        epoch, acc = int(match.group(1)), float(match.group(2))
        self.eval_acc[epoch] = acc

    _extract_eval_acc.pattern = r"Epoch (\d+)/(?:\d+): Benchmark Test accuracy: (\d+\.\d+)"

    @property
    def model_name(self) -> str:
        """Get model name from the log file name."""
        if self._model_name is None:
            pattern = r"(?<=-)[^\-]+(?=-\d{4}_\d{2}_\d{2}_\d{2}h_\d{2}m.log)"
            match = re.search(pattern, self.file.name)
            self._model_name = match.group() if match else "N/A"

        return self._model_name

    @property
    def total_epochs(self) -> int:
        """Get the total number of epochs model was trained for."""
        return len(self.train_loss)

    def plot_results(self) -> None:
        """Plot training loss and evaluation accuracy."""
        fig, axs = plt.subplots(nrows=2, figsize=(10, 6))
        axs[0].plot(self.train_loss, color="black", lw=2)
        axs[0].set_title("Training Loss", fontsize=16)
        axs[0].set_ylabel("Loss", fontsize=14)
        axs[0].tick_params(axis="both", labelsize=12)

        axs[1].plot(self.eval_acc, color="red", lw=2)
        axs[1].set_title("Benchmark Classification Accuracy", fontsize=16)
        axs[1].set_ylabel("Accuracy", fontsize=14)
        axs[1].set_xlabel("Epoch", fontsize=14)
        axs[1].tick_params(axis="both", labelsize=12)

        fig.suptitle(f"Training results: {self.model_name}", fontsize=18)

        plt.tight_layout()
