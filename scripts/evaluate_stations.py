"""
"""

import ast

from absl import app
from absl import flags
# from weatherbench2.regions import CombinedRegion
# from weatherbench2.regions import LandRegion
# from weatherbench2.regions import SliceRegion
from weatherbench2 import aggregations
from weatherbench2 import config
from weatherbench2 import evaluation
from weatherbench2 import flag_utils
from weatherbench2 import metrics
from weatherbench2.derived_variables import DERIVED_VARIABLE_DICT
from weatherbench2.data_readers import SparseGroundTruthFromParquet
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
    help='Beam CombineFn fanout. Might be required for large dataset.',
)


def main(argv: list[str]) -> None:
  selection = config.Selection(
      variables=VARIABLES.value,
      time_slice=slice(TIME_START.value, TIME_STOP.value),
      chunks=INPUT_CHUNKS.value,
  )

  paths = config.Paths(
      forecast=FORECAST_PATH.value,
      obs_data_loader=SparseGroundTruthFromParquet(
          path=OBS_PATH.value,
          time_dim='timeNominal',
          variables=VARIABLES.value,
          ),
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
      rename_variables=rename_variables,
      pressure_level_suffixes=PRESSURE_LEVEL_SUFFIXES.value,
      grid2sparse=True,
      grid2sparse_method='nearest',
  )

  eval_configs = {
      'deterministic': config.Eval(
          metrics={'mse': metrics.MSE()},
          aggregations={
              'weighted': aggregations.WeightedStationAverage(skipna=True),
              'unweighted': aggregations.UnweightedAverage(
                  skipna=True, dims=['stationName'],
              ),
          },
      ),
      'deterministic_spatial': config.Eval(
          metrics={'mse': metrics.MSE()},
          aggregations={
              'none': aggregations.NoAggregation()
          },
      )
  }
  evaluation.evaluate_with_beam(
      data_config,
      eval_configs,
      runner=RUNNER.value,
      input_chunks=INPUT_CHUNKS.value,
      fanout=FANOUT.value,
      argv=argv,
  )





if __name__ == '__main__':
  app.run(main)
  flags.mark_flag_as_required('output_path')
