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

  python scripts/evaluate.py \
    --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_with_poles_conservative.zarr \
    --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_64x32_equiangular_with_poles_conservative.zarr \
    --output_dir=gs://$BUCKET/datasets/evals/$USER/5.625-deterministic-results/ \
    --input_chunks=time=1,lead_time=1 \
    --eval_configs=deterministic \
    --use_beam=True \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --region=$REGION \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./scripts/dataflow-requirements.txt \
    --job_name=eval-$USER
  ```
"""
import ast

from absl import app
from absl import flags
from weatherbench2 import config
from weatherbench2 import evaluation
from weatherbench2 import flag_utils
from weatherbench2 import metrics
from weatherbench2 import thresholds
from weatherbench2.derived_variables import DERIVED_VARIABLE_DICT
from weatherbench2.regions import CombinedRegion
from weatherbench2.regions import LandRegion
from weatherbench2.regions import SliceRegion
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
    True,
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
    'Evaluate climatology forecast specified in climatology path. Note that'
    ' this will not work for probabilistic evaluation. Please use the'
    ' EVALUATE_PROBABILISTIC_CLIMATOLOGY flag.',
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
REGIONS = flags.DEFINE_list(
    'regions',
    None,
    help=(
        'Comma delimited list of predefined regions to evaluate. "all" for all'
        'predefined regions.'
    ),
)
LSM_DATASET = flags.DEFINE_string(
    'lsm_dataset',
    None,
    help=(
        'Dataset containing land-sea-mask at same resolution of datasets to be'
        ' evaluated. Required if region with land-sea-mask is picked. If None,'
        ' defaults to observation dataset.'
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
SKIPNA = flags.DEFINE_boolean(
    'skipna',
    False,
    help=(
        'Whether to skip NaN data points (in forecasts and observations) when'
        ' evaluating.'
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
    help=(
        'Comma delimited list of pressure levels to select for evaluation.'
        ' Ignored if level is not a dimension.'
    ),
)
VARIABLES = flags.DEFINE_list(
    'variables',
    _DEFAULT_VARIABLES,
    help='Comma delimited list of variables to select from weather.',
)
AUX_VARIABLES = flags.DEFINE_list(
    'aux_variables',
    None,
    help='Comma delimited list of auxiliary variables for metric evaluation.',
)
DERIVED_VARIABLES = flags.DEFINE_list(
    'derived_variables',
    [],
    help=(
        'Comma delimited list of derived variables to dynamically compute'
        'during evaluation.'
    ),
)
THRESHOLD_METHOD = flags.DEFINE_string(
    'threshold_method',
    'quantile',
    help=(
        'Threshold method used to binarize forecasts and observations into'
        ' dichotomous events. It can be one of "quantile", "gaussian_quantile".'
    ),
)
QUANTILE_THRESHOLDS = flags.DEFINE_list(
    'quantile_thresholds',
    [],
    help=(
        'Climatological quantile thresholds used to binarize forecasts and'
        ' observations into dichotomous events.'
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
        'chunk sizes overriding input chunks to use for loading forecast and'
        ' observation data. By default, omitted dimension names are loaded in'
        ' one chunk. Metrics should be embarrassingly parallel across chunked'
        ' dimensions. In particular, output variables should contain these'
        ' dimensions.'
    ),
)
USE_BEAM = flags.DEFINE_bool(
    'use_beam',
    False,
    'Run evaluation pipeline as beam pipeline. If False, run in memory.',
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')
FANOUT = flags.DEFINE_integer(
    'fanout',
    None,
    help='Beam CombineFn fanout. Recommended when evaluating large datasets.',
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write Zarr in parallel per worker.',
)
SHUFFLE_BEFORE_TEMPORAL_MEAN = flags.DEFINE_bool(
    'shuffle_before_temporal_mean',
    False,
    help=(
        'Shuffle before computing the temporal mean. This is a good idea when'
        ' evaluation metric outputs are small compared to the size of the'
        ' input data, such as when aggregating over space or a large ensemble.'
    ),
)


def _wind_vector_error(err_type: str):
  """Defines Wind Vector [R]MSEs if U/V components are in variables."""
  if err_type == 'mse':
    cls = metrics.WindVectorMSE
  elif err_type == 'rmse':
    cls = metrics.WindVectorRMSESqrtBeforeTimeAvg
  else:
    raise ValueError(f'Unrecognized {err_type=}')
  wind_vector_error = []
  available = set(VARIABLES.value).union(DERIVED_VARIABLES.value)
  for u_name, v_name, vector_name in [
      ('u_component_of_wind', 'v_component_of_wind', 'wind_vector'),
      ('10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_vector'),
      (
          'u_component_of_geostrophic_wind',
          'v_component_of_geostrophic_wind',
          'geostrophic_wind_vector',
      ),
      (
          'u_component_of_ageostrophic_wind',
          'v_component_of_ageostrophic_wind',
          'ageostrophic_wind_vector',
      ),
  ]:
    if u_name in available and v_name in available:
      wind_vector_error.append(
          cls(
              u_name=u_name,
              v_name=v_name,
              vector_name=vector_name,
          )
      )
  return wind_vector_error


def main(argv: list[str]) -> None:
  """Run all WB2 metrics."""
  selection = config.Selection(
      variables=VARIABLES.value,
      aux_variables=AUX_VARIABLES.value,
      levels=[int(level) for level in LEVELS.value],
      time_slice=slice(TIME_START.value, TIME_STOP.value),
  )

  paths = config.Paths(
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
  data_config = config.Data(
      selection=selection,
      paths=paths,
      by_init=BY_INIT.value,
      rename_variables=rename_variables,
      pressure_level_suffixes=PRESSURE_LEVEL_SUFFIXES.value,
  )

  # Default regions
  predefined_regions = {
      'global': SliceRegion(),
      'tropics': SliceRegion(lat_slice=slice(-20, 20)),
      'extra-tropics': SliceRegion(
          lat_slice=[slice(None, -20), slice(20, None)]
      ),
      'northern-hemisphere': SliceRegion(lat_slice=slice(20, None)),
      'southern-hemisphere': SliceRegion(lat_slice=slice(None, -20)),
      'europe': SliceRegion(
          lat_slice=slice(35, 75),
          lon_slice=[slice(360 - 12.5, None), slice(0, 42.5)],
      ),
      'north-america': SliceRegion(
          lat_slice=slice(25, 60), lon_slice=slice(360 - 120, 360 - 75)
      ),
      'north-atlantic': SliceRegion(
          lat_slice=slice(25, 65), lon_slice=slice(360 - 70, 360 - 10)
      ),
      'north-pacific': SliceRegion(
          lat_slice=slice(25, 60), lon_slice=slice(145, 360 - 130)
      ),
      'east-asia': SliceRegion(
          lat_slice=slice(25, 60), lon_slice=slice(102.5, 150)
      ),
      'ausnz': SliceRegion(
          lat_slice=slice(-45, -12.5), lon_slice=slice(120, 175)
      ),
      'arctic': SliceRegion(lat_slice=slice(60, 90)),
      'antarctic': SliceRegion(lat_slice=slice(-90, -60)),
  }
  try:
    if LSM_DATASET.value:
      land_sea_mask = xr.open_zarr(LSM_DATASET.value)['land_sea_mask'].compute()
    else:
      land_sea_mask = xr.open_zarr(OBS_PATH.value)['land_sea_mask'].compute()
    land_regions = {
        'global_land': LandRegion(land_sea_mask=land_sea_mask),
        'extra-tropics_land': CombinedRegion(
            regions=[
                SliceRegion(lat_slice=[slice(None, -20), slice(20, None)]),
                LandRegion(land_sea_mask=land_sea_mask),
            ]
        ),
        'tropics_land': CombinedRegion(
            regions=[
                SliceRegion(lat_slice=slice(-20, 20)),
                LandRegion(land_sea_mask=land_sea_mask),
            ]
        ),
    }
    predefined_regions = predefined_regions | land_regions
  except KeyError:
    print('No land_sea_mask found.')
  if REGIONS.value == ['all']:
    regions = predefined_regions
  elif REGIONS.value is None:
    regions = None
  else:
    regions = {
        k: v for k, v in predefined_regions.items() if k in REGIONS.value
    }

  # Open climatology for ACC and quantile metrics computation
  climatology = xr.open_zarr(CLIMATOLOGY_PATH.value)
  climatology = evaluation.make_latitude_increasing(climatology)

  if QUANTILE_THRESHOLDS.value:
    threshold_cls = thresholds.get_threshold_cls(THRESHOLD_METHOD.value)
    threshold_list = [
        threshold_cls(climatology=climatology, quantile=float(q))
        for q in QUANTILE_THRESHOLDS.value
    ]
  else:
    threshold_list = []

  deterministic_metrics = {
      'mse': metrics.MSE(wind_vector_mse=_wind_vector_error('mse')),
      'acc': metrics.ACC(climatology=climatology),
      'bias': metrics.Bias(),
      'mae': metrics.MAE(),
  }
  rmse_metrics = {
      'rmse_sqrt_before_time_avg': metrics.RMSESqrtBeforeTimeAvg(
          wind_vector_rmse=_wind_vector_error('rmse')
      ),
  }
  spatial_metrics = {
      'bias': metrics.SpatialBias(),
      'mse': metrics.SpatialMSE(),
      'mae': metrics.SpatialMAE(),
  }
  if COMPUTE_SEEPS.value:
    deterministic_metrics['seeps_24hr'] = metrics.SEEPS(
        climatology=climatology,
        precip_name='total_precipitation_24hr',
        dry_threshold_mm=0.25,
    )
    deterministic_metrics['seeps_6hr'] = metrics.SEEPS(
        climatology=climatology,
        precip_name='total_precipitation_6hr',
        dry_threshold_mm=0.1,
    )
    spatial_metrics['seeps_24hr'] = metrics.SpatialSEEPS(
        climatology=climatology,
        precip_name='total_precipitation_24hr',
        dry_threshold_mm=0.25,
    )
    spatial_metrics['seels_6hr'] = metrics.SpatialSEEPS(
        climatology=climatology,
        precip_name='total_precipitation_6hr',
        dry_threshold_mm=0.1,
    )

  derived_variables = {
      name: DERIVED_VARIABLE_DICT[name] for name in DERIVED_VARIABLES.value
  }

  eval_configs = {
      'deterministic': config.Eval(
          metrics=deterministic_metrics,
          against_analysis=False,
          regions=regions,
          derived_variables=derived_variables,
          evaluate_persistence=EVALUATE_PERSISTENCE.value,
          evaluate_climatology=EVALUATE_CLIMATOLOGY.value,
      ),
      'deterministic_spatial': config.Eval(
          metrics=spatial_metrics,
          against_analysis=False,
          derived_variables=derived_variables,
          evaluate_persistence=EVALUATE_PERSISTENCE.value,
          evaluate_climatology=EVALUATE_CLIMATOLOGY.value,
          output_format='zarr',
      ),
      'deterministic_temporal': config.Eval(
          metrics=deterministic_metrics | rmse_metrics,
          against_analysis=False,
          regions=regions,
          derived_variables=derived_variables,
          evaluate_persistence=EVALUATE_PERSISTENCE.value,
          evaluate_climatology=EVALUATE_CLIMATOLOGY.value,
          temporal_mean=False,
      ),
      # Against analysis is deprecated for by_init, since the time intervals are
      # not compatible. Still functional for by_valid
      'deterministic_vs_analysis': config.Eval(
          metrics=deterministic_metrics,
          against_analysis=True,
          regions=regions,
          derived_variables=derived_variables,
      ),
      'probabilistic': config.Eval(
          metrics={
              'crps': metrics.CRPS(ensemble_dim=ENSEMBLE_DIM.value),
              'crps_spread': metrics.CRPSSpread(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
              'crps_skill': metrics.CRPSSkill(ensemble_dim=ENSEMBLE_DIM.value),
              'ensemble_mean_mse': metrics.EnsembleMeanMSE(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
              'debiased_ensemble_mean_mse': metrics.DebiasedEnsembleMeanMSE(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
              'ensemble_variance': metrics.EnsembleVariance(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
          },
          regions=regions,
          against_analysis=False,
          derived_variables=derived_variables,
          evaluate_probabilistic_climatology=EVALUATE_PROBABILISTIC_CLIMATOLOGY.value,
          probabilistic_climatology_start_year=PROBABILISTIC_CLIMATOLOGY_START_YEAR.value,
          probabilistic_climatology_end_year=PROBABILISTIC_CLIMATOLOGY_END_YEAR.value,
          probabilistic_climatology_hour_interval=PROBABILISTIC_CLIMATOLOGY_HOUR_INTERVAL.value,
      ),
      'ensemble_binary': config.Eval(
          metrics={
              'brier_score': metrics.EnsembleBrierScore(
                  ensemble_dim=ENSEMBLE_DIM.value, thresholds=threshold_list
              ),
              'debiased_brier_score': metrics.DebiasedEnsembleBrierScore(
                  ensemble_dim=ENSEMBLE_DIM.value, thresholds=threshold_list
              ),
              'ignorance_score': metrics.EnsembleIgnoranceScore(
                  ensemble_dim=ENSEMBLE_DIM.value, thresholds=threshold_list
              ),
          },
          regions=regions,
          against_analysis=False,
          derived_variables=derived_variables,
          evaluate_probabilistic_climatology=EVALUATE_PROBABILISTIC_CLIMATOLOGY.value,
          probabilistic_climatology_start_year=PROBABILISTIC_CLIMATOLOGY_START_YEAR.value,
          probabilistic_climatology_end_year=PROBABILISTIC_CLIMATOLOGY_END_YEAR.value,
          probabilistic_climatology_hour_interval=PROBABILISTIC_CLIMATOLOGY_HOUR_INTERVAL.value,
      ),
      'ensemble_forecast_vs_era_experimental_metrics': config.Eval(
          metrics={
              'energy_score': metrics.EnergyScore(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
              'energy_score_spread': metrics.EnergyScoreSpread(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
              'energy_score_skill': metrics.EnergyScoreSkill(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
              'ensemble_mean_rmse_sqrt_before_time_avg': (
                  metrics.EnsembleMeanRMSESqrtBeforeTimeAvg(
                      ensemble_dim=ENSEMBLE_DIM.value
                  )
              ),
              'ensemble_stddev_sqrt_before_time_avg': (
                  metrics.EnsembleStddevSqrtBeforeTimeAvg(
                      ensemble_dim=ENSEMBLE_DIM.value
                  )
              ),
          },
          against_analysis=False,
          derived_variables=derived_variables,
      ),
      'probabilistic_spatial': config.Eval(
          metrics={
              'crps': metrics.SpatialCRPS(ensemble_dim=ENSEMBLE_DIM.value),
              'crps_spread': metrics.SpatialCRPSSpread(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
              'crps_skill': metrics.SpatialCRPSSkill(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
              'ensemble_mean_mse': metrics.SpatialEnsembleMeanMSE(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
              'debiased_ensemble_mean_mse': (
                  metrics.DebiasedSpatialEnsembleMeanMSE(
                      ensemble_dim=ENSEMBLE_DIM.value
                  )
              ),
              'ensemble_variance': metrics.SpatialEnsembleVariance(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
          },
          against_analysis=False,
          derived_variables=derived_variables,
          evaluate_probabilistic_climatology=EVALUATE_PROBABILISTIC_CLIMATOLOGY.value,
          probabilistic_climatology_start_year=PROBABILISTIC_CLIMATOLOGY_START_YEAR.value,
          probabilistic_climatology_end_year=PROBABILISTIC_CLIMATOLOGY_END_YEAR.value,
          probabilistic_climatology_hour_interval=PROBABILISTIC_CLIMATOLOGY_HOUR_INTERVAL.value,
          output_format='zarr',
      ),
      'ensemble_binary_spatial': config.Eval(
          metrics={
              'brier_score': metrics.SpatialEnsembleBrierScore(
                  ensemble_dim=ENSEMBLE_DIM.value, thresholds=threshold_list
              ),
              'debiased_brier_score': metrics.SpatialDebiasedEnsembleBrierScore(
                  ensemble_dim=ENSEMBLE_DIM.value, thresholds=threshold_list
              ),
              'ignorance_score': metrics.SpatialEnsembleIgnoranceScore(
                  ensemble_dim=ENSEMBLE_DIM.value, thresholds=threshold_list
              ),
          },
          against_analysis=False,
          derived_variables=derived_variables,
          evaluate_probabilistic_climatology=EVALUATE_PROBABILISTIC_CLIMATOLOGY.value,
          probabilistic_climatology_start_year=PROBABILISTIC_CLIMATOLOGY_START_YEAR.value,
          probabilistic_climatology_end_year=PROBABILISTIC_CLIMATOLOGY_END_YEAR.value,
          probabilistic_climatology_hour_interval=PROBABILISTIC_CLIMATOLOGY_HOUR_INTERVAL.value,
          output_format='zarr',
      ),
      'probabilistic_spatial_histograms': config.Eval(
          metrics={
              'rank_histogram': metrics.RankHistogram(
                  ensemble_dim=ENSEMBLE_DIM.value
              ),
          },
          against_analysis=False,
          derived_variables=derived_variables,
          evaluate_probabilistic_climatology=EVALUATE_PROBABILISTIC_CLIMATOLOGY.value,
          probabilistic_climatology_start_year=PROBABILISTIC_CLIMATOLOGY_START_YEAR.value,
          probabilistic_climatology_end_year=PROBABILISTIC_CLIMATOLOGY_END_YEAR.value,
          probabilistic_climatology_hour_interval=PROBABILISTIC_CLIMATOLOGY_HOUR_INTERVAL.value,
          output_format='zarr',
      ),
      'gaussian_probabilistic': config.Eval(
          metrics={
              'crps': metrics.GaussianCRPS(),
              'ensemble_variance': metrics.GaussianVariance(),
          },
          against_analysis=False,
          regions=regions,
          derived_variables=derived_variables,
      ),
      'gaussian_binary': config.Eval(
          metrics={
              'brier_score': metrics.GaussianBrierScore(
                  thresholds=threshold_list
              ),
              'ignorance_score': metrics.GaussianIgnoranceScore(
                  thresholds=threshold_list
              ),
          },
          against_analysis=False,
          regions=regions,
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

  if USE_BEAM.value:
    evaluation.evaluate_with_beam(
        data_config,
        eval_configs,
        runner=RUNNER.value,
        input_chunks=INPUT_CHUNKS.value,
        skipna=SKIPNA.value,
        fanout=FANOUT.value,
        num_threads=NUM_THREADS.value,
        shuffle_before_temporal_mean=SHUFFLE_BEFORE_TEMPORAL_MEAN.value,
        argv=argv,
    )
  else:
    evaluation.evaluate_in_memory(
        data_config, eval_configs, skipna=SKIPNA.value
    )


if __name__ == '__main__':
  app.run(main)
  flags.mark_flags_as_required(['output_path', 'obs_path'])
