import numpy as np
import matplotlib.pyplot as plt
from utils.visualization import parity_plot, residual_hist


def test_parity_plot():
    y_true = np.arange(5.0)
    y_pred = np.arange(5.0) + 0.1
    fig = parity_plot(y_true, y_pred, label="test_output")
    assert isinstance(fig, plt.Figure)
    plt.close('all')


def test_residual_hist():
    residuals = np.random.randn(100)
    fig = residual_hist(residuals, label="log10_Ctr")
    assert isinstance(fig, plt.Figure)
    plt.close('all')
