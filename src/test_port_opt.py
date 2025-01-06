import pandas as pd
import numpy as np

import config
from port_opt import find_global_minimum_variance_portfolio


def test_find_global_minimum_variance_portfolio():

    MANUAL_DATA_DIR = config.MANUAL_DATA_DIR

    data_assets = pd.read_csv(MANUAL_DATA_DIR / 'data_assets.csv', parse_dates=['date'])
    data_assets = data_assets.set_index('date')

    rets_log = np.log(data_assets / data_assets.shift(1))
    rets = data_assets / data_assets.shift(1) - 1

    rets = rets * 100

    Sigma = np.cov(rets.dropna().T)

    wstar = find_global_minimum_variance_portfolio(Sigma)
    wstar_expected = np.array([-0.13,  0.37,  0.  ,  0.76,  0.01])

    assert np.allclose(wstar, wstar_expected, rtol=0, atol=0.01)
