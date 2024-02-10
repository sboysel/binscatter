import numpy as np
import pytest

from src.binscatter import Binscatter

@pytest.fixture
def data() -> tuple:
    n_obs = 1000
    k = 4
    np.random.seed(0)
    x = np.random.rand(n_obs)
    y = np.random.rand(n_obs)
    controls = np.random.rand(n_obs, k)
    return x, y, controls


def test_binscatter(data):
    x, y, controls = data
    bs = Binscatter(x, y, controls)
