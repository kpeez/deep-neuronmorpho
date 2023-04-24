"""Create training, validation, and test splits of the dataset."""

import random
import shutil
from pathlib import Path
from typing import Tuple

from typer import Typer

from deep_neuronmorpho.utils.progress import ProgressBar

app = Typer()


@app.command()
def create_data_splits(
    input_dir: Path, split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 42
) -> None:
    """Create training, validation, and test data splits from a directory containing .swc files.

    Args:
        input_dir (Path): The path to the input directory containing the .swc files.
        split_ratios (Tuple[float, float, float], optional): Ratios for train, validation,
        and test splits. Defaults to (0.7, 0.2, 0.1).
        seed (int, optional): The random seed for shuffling the data. Defaults to 42.

    This function will create subdirectories named 'train', 'val', and 'test' in the parent
    of the input directory and move the .swc files into the corresponding subdirectories according
    to the specified split ratios.
    """
    input_dir = Path(input_dir)
    swc_files = list(input_dir.glob("*.swc"))

    random.seed(seed)
    random.shuffle(swc_files)

    num_files = len(swc_files)
    train_end_idx = int(num_files * split_ratios[0])
    val_end_idx = train_end_idx + int(num_files * split_ratios[1])

    train_dir = input_dir / "train"
    val_dir = input_dir / "val"
    test_dir = input_dir / "test"

    for directory in [train_dir, val_dir, test_dir]:
        directory.mkdir(exist_ok=True)

    split_mapping = {
        "train": (0, train_end_idx, train_dir),
        "val": (train_end_idx, val_end_idx, val_dir),
        "test": (val_end_idx, num_files, test_dir),
    }

    for _split, (start, end, target_dir) in split_mapping.items():
        split_files = swc_files[start:end]
        pbar = ProgressBar(split_files, desc=f"Moving {_split} files: ")
        for file in pbar:
            shutil.move(str(file), str(target_dir / file.name))


if __name__ == "__main__":
    print("Splitting dataset into train, val, and test sets...")
    app()
    print("Done splitting dataset.")
