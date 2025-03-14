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
from absl.testing import parameterized
import numpy as np
from weatherbench2 import regridding
from weatherbench2 import schema
import xarray_beam as xbeam

from . import regrid


LatitudeSpacing = regridding.LatitudeSpacing
LongitudeScheme = regridding.LongitudeScheme


class RegridTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          input_longitude_scheme=LongitudeScheme.CENTER_AT_ZERO,
          input_latitude_spacing=LatitudeSpacing.EQUIANGULAR_WITHOUT_POLES,
          output_longitude_scheme=LongitudeScheme.START_AT_ZERO,
          output_latitude_spacing=LatitudeSpacing.EQUIANGULAR_WITH_POLES,
      ),
      dict(
          input_longitude_scheme=LongitudeScheme.START_AT_ZERO,
          input_latitude_spacing=LatitudeSpacing.EQUIANGULAR_WITH_POLES,
          output_longitude_scheme=LongitudeScheme.CENTER_AT_ZERO,
          output_latitude_spacing=LatitudeSpacing.EQUIANGULAR_WITHOUT_POLES,
      ),
  )
  def test_regridding_gets_dimensions_correct(
      self,
      input_longitude_scheme,
      input_latitude_spacing,
      output_longitude_scheme,
      output_latitude_spacing,
  ):
    # The correctness of regridded values are checked in regridding.py
    input_ds = (
        schema.mock_truth_data(
            variables_3d=['geopotential'],
            variables_2d=['2m_temperature'],
            time_start='2021-01-01',
            time_stop='2021-01-11',
            spatial_resolution_in_degrees=1.0,
        )
        .assign_coords(
            # 181 nodes, since mock_truth_data uses equiangular_with_poles
            latitude=regridding.latitude_values(input_latitude_spacing, 181),
        )
        .assign_coords(
            longitude=regridding.longitude_values(input_longitude_scheme, 360),
        )
    )

    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds.chunk({'time': 5}).to_zarr(input_path)
    longitude_nodes = 90
    latitude_nodes = 45

    with flagsaver.as_parsed(
        input_path=input_path,
        output_path=output_path,
        output_chunks='time=2',
        # The output is at 4deg
        longitude_nodes=f'{longitude_nodes}',
        latitude_nodes=f'{latitude_nodes}',
        latitude_spacing=output_latitude_spacing.name,
        longitude_scheme=output_longitude_scheme.name,
        regridding_method='conservative',
        runner='DirectRunner',
    ):
      regrid.main([])

    output_ds, output_chunks = xbeam.open_zarr(output_path)
    self.assertEqual(output_ds.keys(), input_ds.keys())
    self.assertEqual(
        dict(output_ds.sizes),
        {'time': 10, 'level': 3, 'longitude': 90, 'latitude': 45},
    )
    self.assertEqual(
        output_chunks,
        {'time': 2, 'level': 3, 'longitude': 90, 'latitude': 45},
    )

    np.testing.assert_array_equal(
        output_ds.longitude,
        regridding.longitude_values(output_longitude_scheme, longitude_nodes),
    )

    np.testing.assert_array_equal(
        output_ds.latitude,
        regridding.latitude_values(output_latitude_spacing, latitude_nodes),
    )


if __name__ == '__main__':
  absltest.main()
