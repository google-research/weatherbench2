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
# pyformat: mode=pyink
r"""Add derived variables to dataset and save as new file.

Example Usage:
  ```
  export BUCKET=my-bucket
  export PROJECT=my-project
  export REGION=us-central1

  python scripts/compute_derived_variables.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/era5/$USER/1959-2022-6h-64x32_equiangular_with_poles_conservative_with_derived_vars.zarr \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --region=$REGION \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./scripts/dataflow-requirements.txt \
    --job_name=compute-derived-variables-$USER
  ```
"""
import ast
from absl import app
from absl import flags
import apache_beam as beam
from weatherbench2 import derived_variables as dvs
from weatherbench2 import flag_utils
import xarray as xr
import xarray_beam as xbeam

_DEFAULT_DERIVED_VARIABLES = [
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


INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
DERIVED_VARIABLES = flags.DEFINE_list(
    'derived_variables',
    _DEFAULT_DERIVED_VARIABLES,
    help=(
        'Comma delimited list of derived variables to dynamically compute'
        'during evaluation.'
    ),
)
PREEXISTING_VARIABLES_TO_REMOVE = flags.DEFINE_list(
    'preexisting_variables_to_remove',
    [],
    help=(
        'Comma delimited list of variables to remove from the source data, '
        'if they exist. This is useful to allow for overriding source dataset '
        'variables with dervied variables of the same name.'
    ),
)
RENAME_RAW_TP_NAME = flags.DEFINE_bool(
    'rename_raw_tp_name', False, 'Rename raw tp name to "total_precipitation".'
)
RAW_TP_NAME = flags.DEFINE_string(
    'raw_tp_name',
    'total_precipitation',
    help=(
        'Raw name of total precipitation variables. Use'
        ' "total_precipitation_6hr" for backwards compatibility.'
    ),
)
RENAME_VARIABLES = flags.DEFINE_string(
    'rename_variables',
    None,
    help=(
        'Dictionary of variable to rename to standard names. E.g. {"2t":'
        ' "2m_temperature"}'
    ),
)
WORKING_CHUNKS = flag_utils.DEFINE_chunks(
    'working_chunks',
    '',
    help=(
        'chunk sizes overriding input chunks to use for computing aggregations'
        ' e.g., "longitude=10,latitude=10". No need to add'
        ' prediction_timedelta=-1'
    ),
)
RECHUNK_ITEMSIZE = flags.DEFINE_integer(
    'rechunk_itemsize',
    4,
    help='Itemsize for rechunking.',
)
MAX_MEM_GB = flags.DEFINE_integer(
    'max_mem_gb', 1, help='Max memory for rechunking in GB.'
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)

RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


def _add_derived_variables(
    dataset: xr.Dataset, derived_variables: dict[str, dvs.DerivedVariable]
) -> xr.Dataset:
  return dataset.assign(
      {k: dv.compute(dataset) for k, dv in derived_variables.items()}
  )


def _strip_offsets(
    key: xbeam.Key, dataset: xr.Dataset
) -> tuple[xbeam.Key, xr.Dataset]:
  """Remove offsets without corresponding coordinate in dataset."""
  key = key.with_offsets(
      **{k: v if k in dataset.coords else None for k, v in key.offsets.items()}
  )
  return key, dataset


def main(argv: list[str]) -> None:
  derived_variables = {}
  for variable_name in DERIVED_VARIABLES.value:
    # Remove suffix for precipitation accumulations
    # E.g. total_precipitation_24hr_from_6hr should also be called
    # total_precipitation_24hr
    dv = dvs.DERIVED_VARIABLE_DICT[variable_name]
    if (
        variable_name.startswith('total_precipitation_')
        and '_from_' in variable_name
    ):
      variable_name = variable_name.split('_from_')[0]
      assert (
          variable_name not in DERIVED_VARIABLES.value
      ), 'Duplicate variable name after removing suffix.'
    derived_variables[variable_name] = dv

  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH.value)

  for var_name in PREEXISTING_VARIABLES_TO_REMOVE.value:
    if var_name in source_dataset:
      del source_dataset[var_name]
  source_chunks = {
      # Removing variables may remove some dims.
      k: v
      for k, v in source_chunks.items()
      if k in source_dataset.dims
  }

  # Validate and clean-up the source datset.
  if RENAME_RAW_TP_NAME.value:
    source_dataset = source_dataset.rename(
        {RAW_TP_NAME.value: 'total_precipitation'}
    )

  rename_variables = (
      ast.literal_eval(RENAME_VARIABLES.value)
      if RENAME_VARIABLES.value
      else None
  )
  if rename_variables:
    source_dataset = source_dataset.rename(rename_variables)
    source_chunks = {
        rename_variables.get(k, k): v for k, v in source_chunks.items()
    }

  for var_name, dv in derived_variables.items():
    if var_name in source_dataset:
      raise ValueError(
          f'cannot compute {var_name!r} because it already exists in the source'
          ' dataset. Consider including it in '
          '--preexisting_variables_to_remove.'
      )
    if not set(dv.base_variables) <= source_dataset.keys():
      raise ValueError(
          f'cannot compute {var_name!r} because its base variables '
          f'{dv.base_variables} are not found in the source dataset:\n'
          f'{source_dataset}'
      )

  # Add derived variables to template
  template = source_dataset.copy(deep=False)
  derived_variables_with_rechunking = {}
  derived_variables_without_rechunking = {}
  for name, dv in derived_variables.items():
    dropped_dims = dv.all_input_core_dims - set(dv.core_dims[1])
    variable = template[dv.base_variables[0]].isel(
        {k: 0 for k in dropped_dims}, drop=True
    )
    template[name] = variable
    template[name].attrs = {}  # Strip attributes
    if 'prediction_timedelta' in dv.all_input_core_dims:
      derived_variables_with_rechunking[name] = dv
    else:
      derived_variables_without_rechunking[name] = dv
  template = xbeam.make_template(template)

  working_chunks = dict(source_chunks)  # No rechunking
  working_chunks.update(WORKING_CHUNKS.value)
  if 'prediction_timedelta' in source_chunks:
    working_chunks.update({'prediction_timedelta': -1})

  # Define helper functions for branching
  rechunk_variables = []
  for dv in derived_variables_with_rechunking.values():
    rechunk_variables.extend(dv.base_variables)

  def _is_precip(kv: tuple[xbeam.Key, xr.Dataset]) -> bool:
    key, _ = kv
    assert len(key.vars) == 1, key
    (var,) = key.vars
    return var in rechunk_variables

  def _is_not_precip(kv: tuple[xbeam.Key, xr.Dataset]) -> bool:
    key, _ = kv
    assert len(key.vars) == 1, key
    (var,) = key.vars
    return var not in rechunk_variables

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    # Initial branch for computation without rechunking
    # TODO(srasp): Further optimize by splitting branches earlier
    # so that with and without rechunking can be computed in parallel
    pcoll = (
        root
        | xbeam.DatasetToChunks(
            source_dataset,
            source_chunks,
            split_vars=False,
            num_threads=NUM_THREADS.value,
        )
        | beam.MapTuple(
            lambda k, v: (  # pylint: disable=g-long-lambda
                k,
                _add_derived_variables(v, derived_variables_without_rechunking),
            )
        )
        | xbeam.SplitVariables()
        | beam.MapTuple(_strip_offsets)
    )

    if derived_variables_with_rechunking:
      # Rechunking branch: Only variables that require rechunking in lead time,
      # i.e. precipitation, will be rechunked. Others go straight to
      # ChunksToZarr.
      pcoll_rechunk = (
          pcoll
          | beam.Filter(_is_precip)
          | 'RechunkIn'
          >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
              source_dataset.sizes,
              source_chunks,
              working_chunks,
              itemsize=RECHUNK_ITEMSIZE.value,
              max_mem=2**30 * MAX_MEM_GB.value,
          )
          | beam.MapTuple(
              lambda k, v: (  # pylint: disable=g-long-lambda
                  k,
                  _add_derived_variables(v, derived_variables_with_rechunking),
              )
          )
          | 'RechunkOut'
          >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
              source_dataset.sizes,
              working_chunks,
              source_chunks,
              itemsize=RECHUNK_ITEMSIZE.value,
              max_mem=2**30 * MAX_MEM_GB.value,
          )
      )

      # Bypass branch for non-rechunk variables
      pcoll_no_rechunk = pcoll | beam.Filter(_is_not_precip)
      pcoll = (pcoll_no_rechunk, pcoll_rechunk) | beam.Flatten()

    # Combined
    _ = pcoll | xbeam.ChunksToZarr(
        OUTPUT_PATH.value,
        template,
        source_chunks,
        num_threads=NUM_THREADS.value,
    )


if __name__ == '__main__':
  app.run(main)
