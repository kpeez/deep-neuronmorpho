"""Progress bars for various tasks.

For example, loading/exporting/processing files or training/evaluating models.
"""
from typing import Any, Collection

import numpy as np
from tqdm import tqdm


class ProgressBar:
    """A class that wraps the tqdm progress bar to simplify tracking progress of iterable objects.

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
        """asdf.

        Args:
            iterable (Collection): An iterable object to be tracked.
            desc (str, optional): A short description of the progress bar. Defaults to "".
            percent_increment (int, optional): The percentage of progress between each update.
            Defaults to 5.
            bar_format (str, optional): The format of the progress bar.
            **kwargs: Additional keyword arguments that can be passed to the tqdm function.
        """
        self.iterable = iterable
        self.desc = desc
        self.num_iterations = len(iterable)
        self.increment_value = int(np.ceil(self.num_iterations * percent_increment / 100))
        self.pbar = tqdm(
            iterable,
            desc=desc,
            total=self.num_iterations,
            bar_format=bar_format,
            **kwargs,
        )

    def update(self) -> None:
        """Update the progress bar if the increment value is reached."""
        if self.pbar.n % self.increment_value == 0:
            self.pbar.update(self.increment_value)

    def __iter__(self) -> Any:
        """Iterate over the iterable and update the progress bar.

        Yields:
            Any: The next item in the iterable.
        """
        for item in self.pbar:
            self.update()
            yield item
