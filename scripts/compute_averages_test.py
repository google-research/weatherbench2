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
"""Tests for compute_averages."""

from absl.testing import absltest
from absl.testing import flagsaver
from weatherbench2 import metrics
from weatherbench2 import schema
from weatherbench2 import utils
import xarray
from xarray_beam._src import test_util

from . import compute_averages


class ComputeAveragesTest(test_util.TestCase):

  def test_average_time_longitude(self):
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds = utils.random_like(schema.mock_forecast_data(ensemble_size=3))

    # Add a variable (that will be dropped) that contains a new extra dimension
    # (that will be dropped upon VARIABLE selection).
    input_ds['extra'] = input_ds['2m_temperature'].expand_dims(extra_dim=[1])

    input_ds.chunk({'time': 31}).to_zarr(input_path)

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
        time_start='2020-06-01',
        time_stop='2020-07-01',
        averaging_dims=['time', 'longitude'],
        levels=[700],
        variables=['geopotential', 'temperature'],
    ):
      compute_averages.main([])

    output_ds = xarray.open_zarr(output_path)

    expected = input_ds.sel(
        time=slice('2020-06-01', '2020-07-01'), level=[700]
    )[['geopotential', 'temperature']].mean(['time', 'longitude'])

    xarray.testing.assert_allclose(output_ds, expected)

  def test_average_time_latitude(self):
    # Latitude requires weighted averaging, so this tests a slightly different
    # code path.
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds = utils.random_like(schema.mock_forecast_data(ensemble_size=3))

    # Add a variable (that will be dropped) that contains a new extra dimension
    # (that will be dropped upon VARIABLE selection).
    input_ds['extra'] = input_ds['2m_temperature'].expand_dims(extra_dim=[1])

    input_ds.chunk({'time': 31}).to_zarr(input_path)

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
        time_start='2020-06-01',
        time_stop='2020-07-01',
        averaging_dims=['time', 'latitude'],
        levels=None,  # Use all
        variables=['geopotential', 'temperature'],
    ):
      compute_averages.main([])

    output_ds = xarray.open_zarr(output_path)

    weights = metrics.get_lat_weights(input_ds)

    expected = (
        input_ds.sel(
            time=slice('2020-06-01', '2020-07-01'),
        )[['geopotential', 'temperature']]
        .mean(['time'])
        .weighted(weights)
        .mean(['latitude'])
    )

    xarray.testing.assert_allclose(output_ds, expected)


if __name__ == '__main__':
  absltest.main()
