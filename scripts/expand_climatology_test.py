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
from absl.testing import flagsaver
import numpy as np
import pandas as pd
from weatherbench2 import schema
import xarray

from . import expand_climatology


def random_like(dataset: xarray.Dataset, seed: int = 0) -> xarray.Dataset:
  rs = np.random.RandomState(seed)
  return dataset.copy(
      data={k: rs.normal(size=v.shape) for k, v in dataset.items()}
  )


class WB2ExpandClimatologyTest(absltest.TestCase):

  def test(self):
    climatology = schema.mock_hourly_climatology_data(
        variables_3d=['geopotential', 'temperature'],
        variables_2d=['2m_temperature'],
        hour_interval=6,
    )
    climatology = random_like(climatology)

    times = pd.date_range(
        '2019-12-01', '2020-03-30', freq=pd.Timedelta('6 hours')
    )
    times_array = xarray.DataArray(times, dims=['time'], coords={'time': times})
    expected = climatology.sel(
        dayofyear=times_array.dt.dayofyear, hour=times_array.dt.hour
    )
    del expected.coords['dayofyear']
    del expected.coords['hour']

    input_path = self.create_tempdir('input_path').full_path
    output_path = self.create_tempdir('output_path').full_path

    climatology.chunk({'dayofyear': 31, 'level': 1}).to_zarr(input_path)

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
        time_start='2019-12-01',
        time_stop='2020-03-30',
        runner='DirectRunner',
    ):
      expand_climatology.main([])

    actual = xarray.open_zarr(output_path)
    xarray.testing.assert_allclose(actual, expected)
    self.assertEqual(actual.chunks['time'][0], 4 * 31)
    self.assertEqual(actual.chunks['level'][0], 1)


if __name__ == '__main__':
  absltest.main()
