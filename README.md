# binscatter

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Fork of [esantorella/binscatter](https://github.com/esantorella/binscatter) but simply tries to prepare data for plotting.

## Install

```shell
pip install git+https://github.com/sboysel/binscatter@dev
```

## Usage

```python
from binscatter import binscatter
import numpy as np

n = 1000
d = 4
x = np.random.rand(n)
y = np.random.rand(n)
controls = np.random.rand(n, d)

x_binned, y_binned, x_smooth, y_smooth = binscatter(x, y, controls)
```
