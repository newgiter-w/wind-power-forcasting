"""Seeded compiled-kernel witness for the project-local QRF dependency."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / '.aris/qrf-packages'))
import numpy as np
import quantile_forest
from quantile_forest import RandomForestQuantileRegressor

model = RandomForestQuantileRegressor(n_estimators=1, bootstrap=False,
    max_samples_leaf=None, min_samples_leaf=2, random_state=0, n_jobs=1)
model.fit(np.zeros((20, 2)), np.r_[np.zeros(10), np.ones(10)])
q = model.predict(np.zeros((1, 2)), quantiles=[.05, .95])[0]
np.testing.assert_equal(q, [0., 1.])
assert model.estimators_[0].predict(np.zeros((1, 2)))[0] == .5
print('QRF_WITNESS', quantile_forest.__version__, q.tolist())
