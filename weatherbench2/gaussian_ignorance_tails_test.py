# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Numerical regressions for rare Gaussian threshold events."""

import math
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from scipy import special, stats
from weatherbench2 import metrics
from weatherbench2 import regions
from weatherbench2 import thresholds
import xarray as xr


class GaussianIgnoranceTailsTest(parameterized.TestCase):

  @parameterized.parameters(-1.0, 1.0)
  def test_ten_sigma_event_matches_erfc_reference(self, sign):
    forecast = xr.Dataset({'x': xr.DataArray(0.0), 'x_std': xr.DataArray(1.0)})
    truth = xr.Dataset({'x': xr.DataArray(sign * 11.0)})
    threshold = xr.Dataset({'x': xr.DataArray(sign * 10.0)})
    actual = metrics._compute_gaussian_ignorance_score(
        forecast, truth, threshold
    )
    expected = -math.log(math.erfc(10 / math.sqrt(2)) / 2)
    np.testing.assert_allclose(actual.x, expected, rtol=2e-14, atol=0)
    self.assertTrue(np.isfinite(actual.x))

  @parameterized.parameters(np.float32, np.float64)
  def test_tail_scores_keep_values_and_coordinate_alignment(self, dtype):
    z = np.array([-40.0, -10.0, -1.0, 0.0, 1.0, 10.0, 40.0], dtype=dtype)
    coord = {'point': np.arange(len(z))}
    mean = xr.DataArray(
        np.full(len(z), 2.0, dtype=dtype), dims='point', coords=coord
    )
    std = xr.full_like(mean, 2.0)
    forecast = xr.Dataset({'x': mean, 'x_std': std})
    threshold = xr.Dataset({'x': mean + std * z})
    before = forecast.copy(deep=True)
    for exceeded in (False, True):
      truth = xr.Dataset({'x': threshold.x + (1 if exceeded else -1)})
      actual = metrics._compute_gaussian_ignorance_score(
          forecast, truth, threshold
      )
      expected = -special.log_ndtr(-z if exceeded else z)
      np.testing.assert_allclose(
          actual.x,
          expected,
          rtol=1e-7 if dtype == np.float32 else 2e-14,
          atol=1e-15,
      )
      self.assertTrue(np.isfinite(actual.x).all())
      xr.testing.assert_identical(actual.point, mean.point)
    xr.testing.assert_identical(forecast, before)

  @parameterized.parameters(False, True)
  def test_public_threshold_metric_keeps_rare_event_scores_finite(
      self, regional
  ):
    coords = {
        'time': np.array(['2020-01-01'], dtype='datetime64[ns]'),
        'latitude': [-60.0, 0.0, 60.0],
        'longitude': [0.0, 180.0],
    }
    mean = xr.DataArray(np.zeros((1, 3, 2)), dims=tuple(coords), coords=coords)
    forecast = xr.Dataset({'x': mean, 'x_std': xr.ones_like(mean)})
    truth = xr.Dataset({'x': xr.full_like(mean, 11.0)})
    clim_mean = xr.full_like(mean.isel(time=0, drop=True), 10.0).expand_dims(
        dayofyear=[1]
    )
    climatology = xr.Dataset({'x': clim_mean, 'x_std': xr.ones_like(clim_mean)})
    threshold = thresholds.GaussianQuantileThreshold(climatology, 0.5)
    region = regions.SliceRegion(lat_slice=slice(0, 60)) if regional else None
    actual = metrics.GaussianIgnoranceScore([threshold]).compute_chunk(
        forecast, truth, region=region
    )
    expected = -math.log(math.erfc(10 / math.sqrt(2)) / 2)
    np.testing.assert_allclose(actual.x, expected, rtol=2e-14, atol=0)
    self.assertNotIn('latitude', actual.dims)
    self.assertNotIn('longitude', actual.dims)

  def test_moderate_probabilities_match_existing_formula(self):
    forecast = xr.Dataset(
        {'x': ('point', [-1.0, 0.0, 1.0]), 'x_std': ('point', [1.0, 1.0, 1.0])}
    )
    threshold = xr.Dataset({'x': ('point', [0.0, 0.0, 0.0])})
    for truth_values in ([-1.0, 0.0, 1.0], [1.0, 1.0, 1.0]):
      truth = xr.Dataset({'x': ('point', truth_values)})
      cdf = stats.norm.cdf(-forecast.x.values)
      expected = -np.log(np.where(truth.x.values > 0, 1 - cdf, cdf))
      actual = metrics._compute_gaussian_ignorance_score(
          forecast, truth, threshold
      )
      np.testing.assert_allclose(actual.x, expected, rtol=1e-14, atol=0)

  def test_missing_forecast_parameters_stay_missing(self):
    forecast = xr.Dataset(
        {'x': ('point', [np.nan, 0.0]), 'x_std': ('point', [1.0, np.nan])}
    )
    truth = xr.Dataset({'x': ('point', [1.0, 1.0])})
    threshold = xr.Dataset({'x': ('point', [0.0, 0.0])})
    actual = metrics._compute_gaussian_ignorance_score(
        forecast, truth, threshold
    )
    self.assertTrue(np.isnan(actual.x).all())


if __name__ == '__main__':
  absltest.main()
