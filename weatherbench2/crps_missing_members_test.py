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
"""CRPS uses the observed ensemble size independently at each location."""

import numpy as np
import xarray as xr
from absl.testing import absltest, parameterized

from weatherbench2 import metrics


def _reference(values, observation):
  values = np.asarray(values)
  values = values[~np.isnan(values)]
  if not values.size:
    return np.nan, np.nan
  spread = 0.0
  if values.size > 1:
    spread = np.abs(values[:, None] - values[None, :]).sum() / (
        values.size * (values.size - 1)
    )
  return spread, np.abs(values - observation).mean() - 0.5 * spread


class MissingEnsembleMembersTest(parameterized.TestCase):

  @parameterized.product(
      values=(
          (0.0, 2.0, np.nan),
          (1.0, 3.0, np.nan),
          (1.0, np.nan, np.nan),
          (2.0, 2.0, np.nan),
          (np.nan, np.nan, np.nan),
          (1.0, 2.0, 4.0),
      ),
      ensemble_dim=("realization", "member"),
  )
  def test_scores_match_pairwise_reference(self, values, ensemble_dim):
    forecast = xr.Dataset({"x": (ensemble_dim, np.asarray(values))})
    truth = xr.Dataset({"x": 1.0})
    expected_spread, expected_score = _reference(values, 1.0)
    spread = metrics.SpatialCRPSSpread(ensemble_dim).compute_chunk(
        forecast, truth, skipna=True
    )
    score = metrics.SpatialCRPS(ensemble_dim).compute_chunk(
        forecast, truth, skipna=True
    )
    np.testing.assert_allclose(spread.x, expected_spread, atol=1e-12)
    np.testing.assert_allclose(score.x, expected_score, atol=1e-12)

  def test_per_variable_per_location_counts_and_spatial_average(self):
    values = np.array(
        [
            [0.0, 2.0, np.nan, np.nan],
            [1.0, 3.0, 5.0, np.nan],
            [2.0, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]
    ).reshape(2, 2, 4)
    other = np.array(
        [
            [3.0, np.nan, 1.0, 5.0],
            [np.nan, 2.0, np.nan, 4.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, np.nan, 6.0, np.nan],
        ]
    ).reshape(2, 2, 4)
    forecast = xr.Dataset(
        {
            "x": (("latitude", "longitude", "realization"), values),
            "y": (("latitude", "longitude", "realization"), other),
        },
        coords={
            "latitude": [-45.0, 45.0],
            "longitude": [0.0, 180.0],
            "realization": [5, 10, 15, 20],
        },
    )
    truth = xr.Dataset(
        {
            name: xr.ones_like(forecast[name].isel(realization=0, drop=True))
            for name in forecast
        }
    )
    expected = truth.copy(deep=True)
    for name in forecast:
      expected[name].data = np.array(
          [
              _reference(row, 1.0)[1]
              for row in forecast[name].values.reshape(-1, 4)
          ]
      ).reshape(2, 2)
    for perm in ([0, 1, 2, 3], [3, 0, 2, 1]):
      reordered = forecast.isel(realization=perm).transpose(
          "realization", "longitude", "latitude"
      )
      actual = metrics.SpatialCRPS().compute_chunk(
          reordered, truth, skipna=True
      )
      xr.testing.assert_allclose(actual.transpose(*expected.dims), expected)
      averaged = metrics.CRPS().compute_chunk(reordered, truth, skipna=True)
      xr.testing.assert_allclose(averaged, expected.mean(skipna=True))

  def test_missing_members_preserve_translation_invariance(self):
    forecast = xr.Dataset({"x": ("realization", [1.0, 3.0, np.nan])})
    truth = xr.Dataset({"x": 2.0})
    for metric in (metrics.SpatialCRPS(), metrics.SpatialCRPSSpread()):
      expected = metric.compute_chunk(forecast, truth, skipna=True)
      actual = metric.compute_chunk(forecast + 100, truth + 100, skipna=True)
      xr.testing.assert_allclose(actual, expected)

  def test_skipna_false_still_propagates_missing_values(self):
    forecast = xr.Dataset({"x": ("realization", [1.0, 3.0, np.nan])})
    truth = xr.Dataset({"x": 2.0})
    for metric in (metrics.SpatialCRPS(), metrics.SpatialCRPSSpread()):
      self.assertTrue(np.isnan(metric.compute_chunk(forecast, truth).x.item()))


if __name__ == "__main__":
  absltest.main()
