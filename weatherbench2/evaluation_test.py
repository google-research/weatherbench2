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
from weatherbench2 import config
from weatherbench2 import evaluation
from weatherbench2 import metrics
from weatherbench2 import schema
from weatherbench2 import utils
from weatherbench2.regions import ExtraTropicalRegion
from weatherbench2.regions import SliceRegion
import xarray


class EvaluationTest(absltest.TestCase):

  def test_in_memory_and_beam_consistency(self):
    selection = config.Selection(
        variables=['geopotential'],
        levels=[500, 700, 850],
        lat_slice=slice(None),
        lon_slice=slice(None),
        time_slice=slice('2020-01-01', '2020-12-31'),
    )

    truth = schema.mock_truth_data(
        variables_3d=['geopotential'],
        time_start='2019-12-01',
        time_stop='2021-01-04',
        spatial_resolution_in_degrees=30,
        time_resolution='3 hours',
    )
    truth = utils.random_like(truth, seed=0)

    forecast = schema.mock_forecast_data(
        variables_3d=['geopotential'],
        time_start='2019-12-01',
        time_stop='2021-01-01',
        lead_stop='3 days',
        spatial_resolution_in_degrees=30,
    )
    forecast = utils.random_like(forecast, seed=1)

    climatology = schema.mock_hourly_climatology_data(
        variables_3d=['geopotential', 'temperature'],
        spatial_resolution_in_degrees=30,
    )
    climatology = utils.random_like(climatology, seed=2)

    truth_path = self.create_tempdir('truth').full_path
    forecast_path = self.create_tempdir('forecast').full_path

    output_path_1 = self.create_tempdir('output_path').full_path
    output_path_2 = self.create_tempdir('output_path').full_path

    truth.chunk().to_zarr(truth_path)
    forecast.chunk().to_zarr(forecast_path)

    paths = config.Paths(
        forecast=forecast_path,
        obs=truth_path,
        output_dir=output_path_1,
    )

    data_config = config.Data(selection=selection, paths=paths)

    regions = {
        'global': SliceRegion(),
        'tropics': SliceRegion(lat_slice=slice(-20, 20)),
        'extra-tropics': ExtraTropicalRegion(),
    }

    eval_configs = {
        'forecast_vs_era': config.Eval(
            metrics={
                'rmse': metrics.RMSESqrtBeforeTimeAvg(),
                'acc': metrics.ACC(climatology=climatology),
            },
            against_analysis=False,
        ),
        'forecast_vs_era_by_region': config.Eval(
            metrics={'rmse': metrics.RMSESqrtBeforeTimeAvg()},
            against_analysis=False,
            regions=regions,
        ),
        'forecast_vs_era_spatial': config.Eval(
            metrics={'mse': metrics.SpatialMSE()},
            against_analysis=False,
        ),
        'forecast_vs_era_temporal': config.Eval(
            metrics={'rmse': metrics.RMSESqrtBeforeTimeAvg()},
            against_analysis=False,
            temporal_mean=False,
        ),
    }

    evaluation.evaluate_in_memory(data_config, eval_configs)

    data_config.paths.output_dir = output_path_2
    evaluation.evaluate_with_beam(
        data_config,
        eval_configs,
        input_chunks={'init_time': 125},
        runner='DirectRunner',
    )

    for eval_name in eval_configs:
      with self.subTest(eval_name):
        ds1 = xarray.open_dataset(
            os.path.join(output_path_1, f'{eval_name}.nc')
        )
        ds2 = xarray.open_dataset(
            os.path.join(output_path_2, f'{eval_name}.nc')
        )
        xarray.testing.assert_allclose(ds1, ds2)


if __name__ == '__main__':
  absltest.main()
