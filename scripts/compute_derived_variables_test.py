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
import xarray

from . import compute_derived_variables


class ComputeDerivedVariablesTest(absltest.TestCase):

  def test_analysis(self):
    variables_3d = [
        'temperature',
        'geopotential',
        'u_component_of_wind',
        'v_component_of_wind',
        'specific_humidity',
        'specific_cloud_liquid_water_content',
        'specific_cloud_ice_water_content',
    ]
    variables_2d = [
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2m_temperature',
    ]
    inputs = schema.mock_truth_data(
        variables_3d=variables_3d,
        variables_2d=variables_2d,
        time_start='2020-01-01',
        time_stop='2021-01-01',
    )

    input_path = self.create_tempdir('input_path').full_path
    output_path = self.create_tempdir('output_path').full_path

    inputs.chunk().to_zarr(input_path)

    derived_variables = [
        'wind_speed',
        '10m_wind_speed',
        'divergence',
        'vorticity',
        'vertical_velocity',
        'eddy_kinetic_energy',
        'geostrophic_wind_speed',
        'ageostrophic_wind_speed',
        'lapse_rate',
        'total_column_vapor',
        'total_column_liquid',
        'total_column_ice',
        'integrated_vapor_transport',
        'relative_humidity',
    ]

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
        derived_variables=derived_variables,
        runner='DirectRunner',
    ):
      compute_derived_variables.main([])

    result = xarray.open_zarr(output_path)
    self.assertEqual(set(result), set(inputs).union(derived_variables))

  def test_forecast(self):
    variables_3d = [
        'temperature',
        'geopotential',
        'u_component_of_wind',
        'v_component_of_wind',
        'specific_humidity',
    ]
    variables_2d = [
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        '2m_temperature',
        'total_precipitation',
    ]
    inputs = schema.mock_forecast_data(
        variables_3d=variables_3d,
        variables_2d=variables_2d,
        time_start='2020-01-01',
        time_stop='2020-02-01',
        time_resolution='6 hours',
        lead_resolution='6 hours',
    )

    input_path = self.create_tempdir('input_path').full_path
    output_path = self.create_tempdir('output_path').full_path

    inputs.chunk().to_zarr(input_path)

    derived_variables = [
        'wind_speed',
        '10m_wind_speed',
        'divergence',
        'vorticity',
        'vertical_velocity',
        'eddy_kinetic_energy',
        'geostrophic_wind_speed',
        'ageostrophic_wind_speed',
        'lapse_rate',
        'total_column_vapor',
        'integrated_vapor_transport',
        'relative_humidity',
        'total_precipitation_6hr',
        'total_precipitation_24hr',
    ]

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
        derived_variables=derived_variables,
        runner='DirectRunner',
    ):
      compute_derived_variables.main([])

    result = xarray.open_zarr(output_path)
    self.assertEqual(set(result), set(inputs).union(derived_variables))


if __name__ == '__main__':
  absltest.main()
