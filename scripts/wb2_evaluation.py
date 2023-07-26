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
r"""Run WeatherBench 2 evaluation pipeline.

Example Usage:
  ```
  export BUCKET=my-bucket
  export PROJECT=my-project
  export REGION=us-central1

  python scripts/wb2_evaluation.py \
    --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_with_poles_conservative.zarr \
    --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/evals/$USER/5.625-deterministic-results/ \
    --input_chunks=time=1,lead_time=1 \
    --eval_configs=deterministic \
    --use_beam=True \
    --beam_runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --region=$REGION \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=setup.py \
    --job_name=compute-zonal-energy-spectrum-$USER
  ```
"""
import ast

from absl import app
from absl import flags
from weatherbench2 import evaluation
from weatherbench2 import flag_utils
from weatherbench2.config import DataConfig, EvalConfig, Paths, Selection  # pylint: disable=g-multiple-import
from weatherbench2.derived_variables import DERIVED_VARIABLE_DICT
from weatherbench2.metrics import ACC, Bias, CRPS, CRPSSkill, CRPSSpread, EnergyScore, EnergyScoreSkill, EnergyScoreSpread, EnsembleMeanRMSE, EnsembleStddev, RMSE, MSE, SEEPS, SpatialBias, SpatialMSE, SpatialSEEPS, WindVectorRMSE  # pylint: disable=g-multiple-import,unused-import
from weatherbench2.regions import SliceRegion, LandRegion  # pylint: disable=g-multiple-import
import xarray as xr

_DEFAULT_VARIABLES = [
    'geopotential',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'specific_humidity',
    '2m_temperature',
    'mean_sea_level_pressure',
]
_DEFAULT_DERIVED_VARIABLES = []

_DEFAULT_LEVELS = ['500', '700', '850']

FORECAST_PATH = flags.DEFINE_string(
    'forecast_path',
    None,
    help='Path to forecasts to evaluate in Zarr format',
)
OBS_PATH = flags.DEFINE_string(
    'obs_path',
    None,
    help='Path to ground-truth to evaluate in Zarr format',
)
CLIMATOLOGY_PATH = flags.DEFINE_string(
    'climatology_path',
    None,
    help='Path to climatology. Used to compute e.g. ACC.',
)
BY_INIT = flags.DEFINE_bool(
    'by_init',
    False,
    help='Specifies whether forecasts are in by-init or by-valid format.',
)
EVALUATE_PERSISTENCE = flags.DEFINE_bool(
    'evaluate_persistence',
    False,
    'Evaluate persistence forecast, i.e. forecast at t=0',
)
EVALUATE_CLIMATOLOGY = flags.DEFINE_bool(
    'evaluate_climatology',
    False,
    'Evaluate climatology forecast specified in climatology path',
)
EVALUATE_PROBABILISTIC_CLIMATOLOGY = flags.DEFINE_bool(
    'evaluate_probabilistic_climatology',
    False,
    'Evaluate probabilistic climatology,'
    'derived from using each year of the ground-truth dataset as a member',
)
PROBABILISTIC_CLIMATOLOGY_START_YEAR = flags.DEFINE_integer(
    'probabilistic_climatology_start_year',
    None,
    'First year of ground-truth to usefor probabilistic climatology',
)
PROBABILISTIC_CLIMATOLOGY_END_YEAR = flags.DEFINE_integer(
    'probabilistic_climatology_end_year',
    None,
    'Last year of ground-truth to usefor probabilistic climatology',
)
PROBABILISTIC_CLIMATOLOGY_HOUR_INTERVAL = flags.DEFINE_integer(
    'probabilistic_climatology_hour_interval',
    6,
    'Hour interval to computeprobabilistic climatology',
)
ADD_LAND_REGION = flags.DEFINE_bool(
    'add_land_region',
    False,
    help=(
        'Add land-only evaluation. `land_sea_mask` must be in observation'
        'dataset.'
    ),
)
COMPUTE_SEEPS = flags.DEFINE_bool(
    'compute_seeps', False, 'Compute SEEPS for total_precipitation_24hr.'
)
EVAL_CONFIGS = flags.DEFINE_string(
    'eval_configs',
    'deterministic',
    help='Comma-separated list of evaluation configs to run.',
)
ENSEMBLE_DIM = flags.DEFINE_string(
    'ensemble_dim',
    'number',
    help='Ensemble dimension name for ensemble metrics. Default = "number".',
)
RENAME_VARIABLES = flags.DEFINE_string(
    'rename_variables',
    None,
    help=(
        'Dictionary of variable to rename to standard names. E.g. {"2t":'
        ' "2m_temperature"}'
    ),
)
PRESSURE_LEVEL_SUFFIXES = flags.DEFINE_bool(
    'pressure_level_suffixes',
    False,
    help=(
        'Decode pressure levels as suffixes in forecast file. E.g.'
        ' temperature_850'
    ),
)
LEVELS = flags.DEFINE_list(
    'levels',
    _DEFAULT_LEVELS,
    help='Comma delimited list of pressure levels to select for evaluation',
)
VARIABLES = flags.DEFINE_list(
    'variables',
    _DEFAULT_VARIABLES,
    help='Comma delimited list of variables to select from weather.',
)
DERIVED_VARIABLES = flags.DEFINE_list(
    'derived_variables',
    [],
    help=(
        'Comma delimited list of derived variables to dynamically compute'
        'during evaluation.'
    ),
)
TIME_START = flags.DEFINE_string(
    'time_start',
    '2020-01-01',
    help='ISO 8601 timestamp (inclusive) at which to start evaluation',
)
TIME_STOP = flags.DEFINE_string(
    'time_stop',
    '2020-12-31',
    help='ISO 8601 timestamp (inclusive) at which to stop evaluation',
)
OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    help='Directory in which to save evaluation results in netCDF format',
)
OUTPUT_FILE_PREFIX = flags.DEFINE_string(
    'output_file_prefix',
    '',
    help=(
        'Prefix of results filename. If "_" or "-" is desired, please add'
        ' specifically here.'
    ),
)
INPUT_CHUNKS = flag_utils.DEFINE_chunks(
    'input_chunks',
    'time=1',
    help=(
        'chunk sizes overriding input chunks to use for loading forecast and '
        'observation data. By default, omitted dimension names are loaded in '
        'one chunk.'
    ),
)
USE_BEAM = flags.DEFINE_bool(
    'use_beam',
    False,
    'Run evaluation pipeline as beam pipeline. If False, run in memory.',
)
BEAM_RUNNER = flags.DEFINE_string('beam_runner', None, help='Beam runner')
FANOUT = flags.DEFINE_integer(
    'fanout',
    None,
    help='Beam CombineFn fanout. Might be required for large dataset.',
)


def _wind_vector_rmse():
  """Defines Wind Vector RMSEs if U/V components are in variables."""
  wind_vector_rmse = []
  if (
      'u_component_of_wind' in VARIABLES.value
      and 'v_component_of_wind' in VARIABLES.value
  ):
    wind_vector_rmse.append(
        WindVectorRMSE(
            u_name='u_component_of_wind',
            v_name='v_component_of_wind',
            vector_name='wind_vector',
        )
    )
  if (
      '10m_u_component_of_wind' in VARIABLES.value
      and '10m_v_component_of_wind' in VARIABLES.value
  ):
    wind_vector_rmse.append(
        WindVectorRMSE(
            u_name='10m_u_component_of_wind',
            v_name='10m_v_component_of_wind',
            vector_name='10m_wind_vector',
        )
    )
  return wind_vector_rmse


def main(argv: list[str]) -> None:
  """Run all WB2 metrics."""
  selection = Selection(
      variables=VARIABLES.value,
      levels=[int(level) for level in LEVELS.value],
      time_slice=slice(TIME_START.value, TIME_STOP.value),
  )

  paths = Paths(
      forecast=FORECAST_PATH.value,
      obs=OBS_PATH.value,
      climatology=CLIMATOLOGY_PATH.value,
      output_dir=OUTPUT_DIR.value,
      output_file_prefix=OUTPUT_FILE_PREFIX.value,
  )

  rename_variables = (
      ast.literal_eval(RENAME_VARIABLES.value)
      if RENAME_VARIABLES.value
      else None
  )
  data_config = DataConfig(
      selection=selection,
      paths=paths,
      by_init=BY_INIT.value,
      rename_variables=rename_variables,
      pressure_level_suffixes=PRESSURE_LEVEL_SUFFIXES.value,
  )

  # Default regions
  regions = {
      'global': SliceRegion(),
      'tropics': SliceRegion(lat_slice=slice(-20, 20)),
      'extra-tropics': SliceRegion(
          lat_slice=[slice(None, -20), slice(20, None)]
      ),
      'europe': SliceRegion(
          lat_slice=slice(35, 75),
          lon_slice=[slice(360 - 12.5, None), slice(0, 42.5)],
      ),
  }

  if ADD_LAND_REGION.value:
    lsm = xr.open_zarr(OBS_PATH.value, chunks=None)['land_sea_mask']
    regions['land'] = LandRegion(land_sea_mask=lsm)

  # Open climatology for ACC computation
  climatology = xr.open_zarr(CLIMATOLOGY_PATH.value)
  climatology = evaluation.make_latitude_increasing(climatology)

  deterministic_metrics = {
      'rmse': RMSE(wind_vector_rmse=_wind_vector_rmse()),
      'mse': MSE(),
      'acc': ACC(climatology=climatology),
  }
  spatial_metrics = {'bias': SpatialBias(), 'mse': SpatialMSE()}
  if COMPUTE_SEEPS.value:
    deterministic_metrics['seeps'] = SEEPS(climatology=climatology)
    spatial_metrics['seeps'] = SpatialSEEPS(climatology=climatology)

  derived_variables = [
      DERIVED_VARIABLE_DICT[derived_variable]
      for derived_variable in DERIVED_VARIABLES.value
  ]

  eval_configs = {
      'deterministic': EvalConfig(
          metrics=deterministic_metrics,
          against_analysis=False,
          regions=regions,
          derived_variables=derived_variables,
          evaluate_persistence=EVALUATE_PERSISTENCE.value,
          evaluate_climatology=EVALUATE_CLIMATOLOGY.value,
      ),
      'deterministic_spatial': EvalConfig(
          metrics=spatial_metrics,
          against_analysis=False,
          derived_variables=derived_variables,
          evaluate_persistence=EVALUATE_PERSISTENCE.value,
          evaluate_climatology=EVALUATE_CLIMATOLOGY.value,
          output_format='zarr',
      ),
      'deterministic_temporal': EvalConfig(
          metrics=deterministic_metrics,
          against_analysis=False,
          regions=regions,
          derived_variables=derived_variables,
          evaluate_persistence=EVALUATE_PERSISTENCE.value,
          evaluate_climatology=EVALUATE_CLIMATOLOGY.value,
          temporal_mean=False,
      ),
      # Against analysis is deprecated for by_init, since the time intervals are
      # not compatible. Still functional for by_valid
      'deterministic_vs_analysis': EvalConfig(
          metrics=deterministic_metrics,
          against_analysis=True,
          regions=regions,
          derived_variables=derived_variables,
      ),
      'probabilistic': EvalConfig(
          metrics={
              'crps': CRPS(ensemble_dim=ENSEMBLE_DIM.value),
              'ensemble_mean_rmse': EnsembleMeanRMSE(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
              'ensemble_stddev': EnsembleStddev(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
          },
          against_analysis=False,
          derived_variables=derived_variables,
          evaluate_probabilistic_climatology=EVALUATE_PROBABILISTIC_CLIMATOLOGY.value,
          probabilistic_climatology_start_year=PROBABILISTIC_CLIMATOLOGY_START_YEAR.value,
          probabilistic_climatology_end_year=PROBABILISTIC_CLIMATOLOGY_END_YEAR.value,
          probabilistic_climatology_hour_interval=PROBABILISTIC_CLIMATOLOGY_HOUR_INTERVAL.value,
      ),
      'ensemble_forecast_vs_era_experimental_metrics': EvalConfig(
          metrics={
              'crps_spread': CRPSSpread(ensemble_dim=ENSEMBLE_DIM.value),
              'crps_skill': CRPSSkill(ensemble_dim=ENSEMBLE_DIM.value),
              'energy_score': EnergyScore(ensemble_dim=ENSEMBLE_DIM.value),
              'energy_score_spread': EnergyScoreSpread(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
              'energy_score_skill': EnergyScoreSkill(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
          },
          against_analysis=False,
          derived_variables=derived_variables,
      ),
  }
  if not set(EVAL_CONFIGS.value.split(',')).issubset(eval_configs):
    raise flags.UnrecognizedFlagError(
        f'{EVAL_CONFIGS.value=} did not define a subset of '
        f'{eval_configs.keys()=}'
    )

  eval_configs = {
      k: v
      for k, v in eval_configs.items()
      if k in EVAL_CONFIGS.value.split(',')
  }

  input_chunks = flag_utils.parse_chunks(INPUT_CHUNKS.value)

  if USE_BEAM.value:
    evaluation.evaluate_with_beam(
        data_config,
        eval_configs,
        runner=BEAM_RUNNER.value,
        input_chunks=input_chunks,
        fanout=FANOUT.value,
        argv=argv,
    )
  else:
    evaluation.evaluate_in_memory(data_config, eval_configs)


if __name__ == '__main__':
  app.run(main)
  flags.mark_flag_as_required('output_path')
