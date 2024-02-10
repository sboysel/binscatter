"""binscatter
Author: Sam Boysel <sboysel@gmail.com>
Original author:  Elizabeth Santorella
Source: fork of https://github.com/esantorella/binscatter
SPDX-License-Identifier: BSD-3-Clause
"""
import numpy as np


def binscatter(x, y, controls, k=1, n_bins=None, x_linspace_num=100) -> tuple:
    """Generate data for binned scatterplot"""
    bs = Binscatter(x, y, controls, k, n_bins, x_linspace_num)
    return bs.generate()

class Binscatter:
    """Class to generate plottable data for binned scatterplot

    Parameters

    x : numpy.ndarray            (required) x values to plot 
    y : numpy.ndarray            (required) y values to plot
    controls : numpy.ndarray     (optional) control variables used for conditioning
                                 the relationship between x and y on. If provided,
                                 Binscatter.generate() will 'residualize / de-mean'
                                 both x and y using the variables in controls.
    k : int                      (required)
    n_bins: int                  (optional)
    x_linspace_num : int         (required)
    
    for data (y, x), return version with means by bin
    for data (y, x, controls), first residualize (y, x) with controls, then return
        the binned version.
    
    Returns 
    
    x_binned : numpy.ndarray
    y_binned : numpy.ndarray
    x_smooth : numpy.ndarray           linspace used to plot smoothed polynomial
                                       fit
    y_smooth : numpy.poly1d            a smoothed polynomial of dergree k

    based on binscatter by Elizabeth Santorella
    source: https://github.com/esantorella/binscatter
    """
    def __init__(self, x, y, controls, k=1, n_bins=None, x_linspace_num=100):
        """
        """
        self.x = x
        self.y = y
        self.controls = controls
        self.k = k
        self.n_bins = n_bins
        self.x_linspace_num = x_linspace_num   
        self.x_binned = None
        self.y_binned = None
        self.x_smooth = None
        self.y_smooth = None

    def generate(self):
        """
        """
        if self.controls is None:
            y_tilde = self.y
            x_tilde = self.x
        else:
            y_tilde = _residualize(self.y, self.controls)
            x_tilde = _residualize(self.x, self.controls)

        # sort observations
        argsort = np.argsort(x_tilde)
        x_tilde = x_tilde[argsort]
        y_tilde = y_tilde[argsort]

        # create bins 
        if self.n_bins is None:
            self.n_bins = _n_bins_optimal(self.x)
            
        self.bins = _get_bins(len(self.y), self.n_bins)

        # binned means
        x_binned = np.array([np.mean(x_tilde[bin_]) for bin_ in self.bins])
        y_binned = np.array([np.mean(y_tilde[bin_]) for bin_ in self.bins])

        # polynomial fit
        coef = np.polyfit(x_binned, y_binned, self.k)
        poly = np.poly1d(coef)
        x_smooth = np.linspace(self.x.min(), self.x.max(), num=self.x_linspace_num)
        y_smooth = poly(x_smooth)

        # update opject
        self.x_binned = x_binned
        self.y_binned = y_binned
        self.x_smooth = x_smooth
        self.y_smooth = y_smooth
        
        return (x_binned, y_binned, x_smooth, y_smooth)


def _ols(X, y, rcond=None):
    """Estimate β from the model 
    y = Xβ + e
    where E[e|X] = 0
    """
    beta, residuals, _, _ = np.linalg.lstsq(X, y, rcond=rcond)    
    return beta, residuals

def _predict(X, beta):
    """Calculate yhat = Xβ"""
    return np.dot(X, beta)

def _residualize(y, controls):
    """Calculate ytilde = y - yhat where yhat = Xβ from the model y = Xβ + e
    Recenter ytilde by adding back the original mean of y.
    """
    # demean y
    beta_y, _ = _ols(controls, y)
    y_hat = _predict(controls, beta_y)
    y_tilde = y - y_hat
    # recenter y
    y_tilde += np.mean(y)
    return y_tilde

def _get_bins(n_elements: int, n_bins: int) -> list:
    bin_edges = np.linspace(0, n_elements, n_bins + 1).astype(int)
    bins = [slice(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
    return bins

def _iqr(x):
    q75, q25 = np.percentile(x, [75 ,25])
    return q75 - q25

def _freedman_diaconis(x):
    """Return the optimal bin width
    https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    """
    return 2 * _iqr(x) * len(x) ** (-1/3)

def _n_bins_optimal(x):
    """
    """
    bin_width = _freedman_diaconis(x)
    return int(np.ceil((x.max() - x.min()) / bin_width)) 
