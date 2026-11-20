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
# pyformat: mode=pyink
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from weatherbench2 import metrics
from weatherbench2 import regions
from weatherbench2 import schema
import xarray as xr


class RegionsTest(absltest.TestCase):

  def testLandRegion(self):
    # Test that non-land regions are not considered in metric computation
    forecast = schema.mock_forecast_data(
        variables_3d=[],
        variables_2d=['2m_temperature'],
        time_start='2022-01-01',
        time_stop='2022-01-02',
        lead_stop='0 day',
    )
    truth = schema.mock_truth_data(
        variables_3d=[],
        variables_2d=['2m_temperature'],
        time_start='2022-01-01',
        time_stop='2022-01-02',
    )
    forecast = forecast.where(forecast.latitude > 0, 1)
    lsm = xr.zeros_like(forecast['2m_temperature'].squeeze())
    lsm = lsm.where(lsm.latitude < 1.0, 1)
    land_region = regions.LandRegion(lsm)

    rmse = metrics.RMSESqrtBeforeTimeAvg()

    results = rmse.compute(forecast, truth, region=land_region)
    np.testing.assert_allclose(results['2m_temperature'].values, 0.0)


class ExtraTropicalThresholdTest(parameterized.TestCase):

  @parameterized.parameters(
      (0, False), (20, False), (45, False), (45, True), (90, False)
  )
  def test_configured_cutoff_preserves_weights_and_coordinates(
      self, threshold, reverse
  ):
    latitudes = np.array([-90, -60, -45, -30, -20, 0, 20, 30, 45, 60, 90])
    if reverse:
      latitudes = latitudes[::-1]
    dataset = xr.Dataset(
        {'value': (('latitude', 'longitude'), np.ones((11, 2)))},
        coords={'latitude': latitudes, 'longitude': [0, 180]},
    )
    weights = xr.DataArray(
        np.arange(1.0, 12.0), dims='latitude', coords={'latitude': latitudes}
    )
    original_dataset = dataset.copy(deep=True)
    original_weights = weights.copy(deep=True)
    result, result_weights = regions.ExtraTropicalRegion(
        threshold_lat=threshold
    ).apply(dataset, weights)
    expected = weights * (np.abs(latitudes) >= threshold)
    xr.testing.assert_identical(result, dataset)
    xr.testing.assert_equal(result_weights, expected)
    xr.testing.assert_identical(dataset, original_dataset)
    xr.testing.assert_identical(weights, original_weights)

  def test_default_retains_twenty_degree_cutoff(self):
    dataset = xr.Dataset(coords={'latitude': [-30, -20, 0, 20, 30]})
    weights = xr.ones_like(dataset.latitude, dtype=float)
    _, actual = regions.ExtraTropicalRegion().apply(dataset, weights)
    np.testing.assert_array_equal(actual, [1, 1, 0, 1, 1])

  @parameterized.parameters(0, 45)
  def test_mse_uses_requested_cutoff_in_combined_region(self, threshold):
    latitude = np.arange(-90.0, 91.0, 15.0)
    values = (np.abs(latitude) + 1)[:, None] * np.array([[1.0, 3.0, 5.0]])
    forecast = xr.Dataset(
        {'value': (('latitude', 'longitude'), values)},
        coords={'latitude': latitude, 'longitude': [0.0, 90.0, 180.0]},
    )
    truth = xr.zeros_like(forecast)
    region = regions.CombinedRegion(
        [
            regions.SliceRegion(lon_slice=slice(0.0, 90.0)),
            regions.ExtraTropicalRegion(threshold_lat=threshold),
        ]
    )
    actual = metrics.MSE().compute_chunk(forecast, truth, region=region)
    keep = np.abs(latitude) >= threshold
    weights = metrics.get_lat_weights(forecast).values[keep]
    expected = np.sum(values[keep, :2] ** 2 * weights[:, None]) / (
        2 * weights.sum()
    )
    np.testing.assert_allclose(actual['value'], expected, rtol=1e-6)


if __name__ == '__main__':
  absltest.main()
