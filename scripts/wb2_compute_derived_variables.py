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

  python scripts/wb2_compute_derived_variables.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/era5/$USER/1959-2022-6h-64x32_equiangular_with_poles_conservative_with_derived_vars.zarr \
    --beam_runner=DataflowRunner \
    -- \
    --project $PROJECT \
    --region $REGION \
    --temp_location gs://$BUCKET/tmp/ \
    --job_name compute-derived-variables-$USER
  ```
"""
from absl import app
from absl import flags
import apache_beam as beam
from weatherbench2 import flag_utils
from weatherbench2.derived_variables import DERIVED_VARIABLE_DICT, DerivedVariable, PrecipitationAccumulation, AggregatePrecipitationAccumulation  # pylint: disable=g-line-too-long,g-multiple-import
import xarray as xr
import xarray_beam as xbeam

_DEFAULT_DERIVED_VARIABLES = [
    'wind_speed',
    '10m_wind_speed',
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

RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


def _add_derived_variables(
    dataset: xr.Dataset, derived_variables: list[DerivedVariable]
) -> xr.Dataset:
  for dv in derived_variables:
    dataset[dv.variable_name] = dv.compute(dataset)
  return dataset


def _strip_offsets(
    key: xbeam.Key, dataset: xr.Dataset
) -> tuple[xbeam.Key, xr.Dataset]:
  """Remove offsets without corresponding coordinate in dataset."""
  key = key.with_offsets(
      **{k: v if k in dataset.coords else None for k, v in key.offsets.items()}
  )
  return key, dataset


def main(argv: list[str]) -> None:
  derived_variables = [
      DERIVED_VARIABLE_DICT[derived_variable]
      for derived_variable in DERIVED_VARIABLES.value
  ]

  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH.value)
  if RENAME_RAW_TP_NAME.value:
    source_dataset = source_dataset.rename(
        {RAW_TP_NAME.value: 'total_precipitation'}
    )

  # Add derived variables to template
  template = source_dataset
  derived_variables_with_rechunking = []
  derived_variables_without_rechunking = []
  for dv in derived_variables:
    template = template.assign(
        {dv.variable_name: template[dv.base_variables[0]]}
    )
    template[dv.variable_name].attrs = {}  # Strip attributes
    if isinstance(dv, (
        PrecipitationAccumulation,
        AggregatePrecipitationAccumulation,
    )):
      derived_variables_with_rechunking.append(dv)
    else:
      derived_variables_without_rechunking.append(dv)
  template = xbeam.make_template(template)

  working_chunks = dict(source_chunks)  # No rechunking
  if WORKING_CHUNKS.value:
    working_chunks.update(flag_utils.parse_chunks(WORKING_CHUNKS.value))
  working_chunks.update({'prediction_timedelta': -1})

  # Define helper functions for branching
  rechunk_variables = []
  for dv in derived_variables_with_rechunking:
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
        | xbeam.DatasetToChunks(source_dataset, source_chunks, split_vars=False)
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
      # Rechunking branch: Only variables that require rechunking,
      # i.e. precipitation, will be rechunked. Others go straight to
      # ChunksToZarr.
      pcoll_rechunk = (
          pcoll
          | beam.Filter(_is_precip)
          | 'RechunkIn'
          >> xbeam.Rechunk(
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
          >> xbeam.Rechunk(
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
        OUTPUT_PATH.value, template, source_chunks, num_threads=16
    )


if __name__ == '__main__':
  app.run(main)
