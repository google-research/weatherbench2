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
from absl.testing import absltest
from weatherbench2 import schema
from weatherbench2 import utils
import xarray


class SchemaTest(absltest.TestCase):

  def test_mock_truth_data(self):
    ds = schema.mock_truth_data()
    expected_sizes = {
        'time': 366,
        'longitude': 36,
        'latitude': 18 + 1,
        'level': 3,
    }
    self.assertEqual(dict(ds.sizes), expected_sizes)
    expected_dims = ('time', 'level', 'longitude', 'latitude')
    self.assertEqual(ds['temperature'].dims, expected_dims)

  def test_mock_forecast_data(self):
    ds = schema.mock_forecast_data()
    expected_sizes = {
        'time': 366,
        'longitude': 36,
        'latitude': 18 + 1,
        'level': 3,
        'prediction_timedelta': 11,
    }
    self.assertEqual(dict(ds.sizes), expected_sizes)
    expected_dims = (
        'prediction_timedelta',
        'time',
        'level',
        'longitude',
        'latitude',
    )
    self.assertEqual(ds['temperature'].dims, expected_dims)

  def test_mock_climatology_data(self):
    base = schema.mock_truth_data(time_resolution='6 hours')
    expected = utils.compute_hourly_stat(
        base, window_size=3, clim_years=slice(None), hour_interval=6
    )
    actual = schema.mock_hourly_climatology_data(hour_interval=6)
    xarray.testing.assert_allclose(expected, actual)


if __name__ == '__main__':
  absltest.main()
