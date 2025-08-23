from pathlib import Path

import pandas as pd
import pytest
from torch_geometric.data import Data

from deep_neuronmorpho.data import SWCData, swc_df_to_pyg_data

TESTS_ROOT_DIR = Path(__file__).parent


@pytest.fixture
def swc_file() -> Path:
    return TESTS_ROOT_DIR / "utils" / "1164438028_191812_5676-X8072-Y26215_reg.swc"


@pytest.fixture
def swc_dataframe(swc_file: Path) -> pd.DataFrame:
    return SWCData.load_swc_data(swc_file)


@pytest.fixture
def synthetic_swc_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n": [1, 2, 3, 4],
            "type": [1, 3, 3, 3],
            "x": [0.0, 1.0, 1.0, 2.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "radius": [1.0, 0.5, 0.5, 0.5],
            "parent": [-1, 1, 1, 3],
        }
    )


@pytest.fixture
def pyg_data(swc_dataframe: pd.DataFrame) -> Data:
    return swc_df_to_pyg_data(swc_dataframe)
