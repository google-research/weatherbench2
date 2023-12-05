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
import os

from absl.testing import absltest
from absl.testing import flagsaver
from weatherbench2 import schema
import xarray

from . import evaluate


class WB2Evaluation(absltest.TestCase):

  def _test(self, use_beam=True, input_chunks=None):
    input_chunks = input_chunks or {}
    variables_3d = [
        'geopotential',
        'u_component_of_wind',
        'v_component_of_wind',
    ]
    derived_variables = [
        'wind_speed',
        'u_component_of_ageostrophic_wind',
        'v_component_of_ageostrophic_wind',
    ]
    variables_2d = ['2m_temperature']
    truth = schema.mock_truth_data(
        variables_3d=variables_3d,
        variables_2d=variables_2d,
        time_start='2020-01-01',
        time_stop='2021-01-01',
    )
    forecast = schema.mock_forecast_data(
        variables_3d=variables_3d,
        variables_2d=variables_2d,
        time_start='2019-12-01',
        time_stop='2021-01-01',
        lead_stop='3 days',
    )
    climatology = schema.mock_hourly_climatology_data(
        variables_3d=variables_3d,
        variables_2d=variables_2d,
    )
    climatology = climatology.assign(
        wind_speed=climatology['u_component_of_wind'],
        u_component_of_ageostrophic_wind=climatology['u_component_of_wind'],
        v_component_of_ageostrophic_wind=climatology['u_component_of_wind'],
    )

    truth_path = self.create_tempdir('truth').full_path
    forecast_path = self.create_tempdir('forecast').full_path
    climatology_path = self.create_tempdir('climatology').full_path
    output_dir = self.create_tempdir('output_dir').full_path

    truth.chunk().to_zarr(truth_path)
    forecast.chunk().to_zarr(forecast_path)
    climatology.chunk().to_zarr(climatology_path)

    eval_configs = [
        'deterministic',
        'deterministic_vs_analysis',
    ]

    with flagsaver.flagsaver(
        forecast_path=forecast_path,
        obs_path=truth_path,
        climatology_path=climatology_path,
        output_dir=output_dir,
        time_start='2020-01-01',
        time_stop='2020-12-31',
        runner='DirectRunner',
        by_init=False,
        regions=['global', 'tropics', 'extra-tropics', 'europe'],
        input_chunks=input_chunks,
        eval_configs=','.join(eval_configs),
        use_beam=use_beam,
        variables=variables_3d + variables_2d,
        derived_variables=derived_variables,
    ):
      evaluate.main([])

    for config_name in eval_configs:
      expected_sizes_2d = {'metric': 4, 'lead_time': 4, 'region': 4}
      expected_sizes_3d = {'metric': 4, 'lead_time': 4, 'region': 4, 'level': 3}

      with self.subTest(config_name):
        results_path = os.path.join(output_dir, f'{config_name}.nc')
        actual = xarray.open_dataset(results_path)
        extra_out_vars = [
            'wind_speed',
            'wind_vector',
            'u_component_of_ageostrophic_wind',
            'v_component_of_ageostrophic_wind',
            'ageostrophic_wind_vector',
        ]
        self.assertEqual(
            set(actual), set(variables_3d + variables_2d + extra_out_vars)
        )
        self.assertEqual(actual['geopotential'].sizes, expected_sizes_3d)
        self.assertEqual(actual['2m_temperature'].sizes, expected_sizes_2d)
        self.assertIn('wind_vector', actual)
        self.assertIn('ageostrophic_wind_vector', actual)

  def test_in_memory(self):
    self._test(use_beam=False, input_chunks=dict(time=-1))

  def test_beam(self):
    self._test(use_beam=True, input_chunks=dict(time=-1))

  def test_beam_time_chunk_125(self):
    self._test(use_beam=True, input_chunks=dict(time=125))

  def test_beam_longitude_chunk_20(self):
    self._test(use_beam=True, input_chunks=dict(time=-1, longitude=20))


if __name__ == '__main__':
  absltest.main()
