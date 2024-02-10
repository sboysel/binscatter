# binscatter

Fork of [esantorella/binscatter](https://github.com/esantorella/binscatter) but simply tries to prepare data for plotting.

## Getting started

```shell
pip install git+https://github.com/sboysel/binscatter@dev
```

## Usage

```
import binscatter
import numpy as np

n = 1000
d = 4
x = np.random.rand(n)
y = np.random.rand(n)
controls = np.random.rand(n, d)

binscatter.Binscatter(x, y, controls, k=1, n_bins=20)
```
