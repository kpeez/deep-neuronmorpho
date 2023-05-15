"""Utilities for monitoring the training process."""
import logging
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Collection

import numpy as np
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


def setup_logger(log_dir: Path) -> logging.Logger:
    """Create a logger for logging training progress."""
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # writing log messages to a file
    log_filename = f"{dt.now().strftime('%Y-%m-%d_%H-%M-%S')}-training.log"
    file_handler = logging.FileHandler(log_dir / log_filename)
    file_handler.setLevel(logging.INFO)

    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
