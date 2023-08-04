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
from weatherbench2 import schema
import xarray_beam

from . import regrid


class RegridTest(absltest.TestCase):

  def test_regridding(self):
    input_ds = schema.mock_truth_data(
        variables_3d=['geopotential'],
        variables_2d=['2m_temperature'],
        time_start='2021-01-01',
        time_stop='2022-01-01',
        spatial_resolution_in_degrees=5.0,
    )

    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds.chunk({'time': 40}).to_zarr(input_path)

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
        output_chunks='time=160',
        latitude_nodes=36,
        longitude_nodes=19,
        latitude_spacing='equiangular_with_poles',
        regridding_method='conservative',
        runner='DirectRunner',
    ):
      regrid.main([])

    actual_ds, actual_chunks = xarray_beam.open_zarr(output_path)
    self.assertEqual(actual_ds.keys(), input_ds.keys())
    self.assertEqual(
        dict(actual_ds.sizes),
        {'time': 365, 'level': 3, 'longitude': 36, 'latitude': 19},
    )
    self.assertEqual(
        actual_chunks,
        {'time': 160, 'level': 3, 'longitude': 36, 'latitude': 19},
    )


if __name__ == '__main__':
  absltest.main()
