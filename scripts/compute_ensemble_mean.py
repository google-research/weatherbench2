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
r"""Ensemble mean (over REALIZATION dimension) of a forecast dataset.

Example Usage:
  ```
  export BUCKET=my-bucket
  export PROJECT=my-project
  export REGION=us-central1

  python scripts/compute_ensemble_mean.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/era5/$USER/1959-2022-ensemble-means.zarr \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./scripts/dataflow-requirements.txt \
    --job_name=compute-ensemble-mean-$USER
  ```
"""
import typing as t

from absl import app
from absl import flags
import apache_beam as beam
import xarray as xr
import xarray_beam as xbeam

REALIZATION = 'realization'

INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')
REALIZATION_NAME = flags.DEFINE_string(
    'realization_name',
    REALIZATION,
    'Name of realization/member/number dimension.',
)
TIME_DIM = flags.DEFINE_string(
    'time_dim', 'time', help='Name for the time dimension to slice data on.'
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
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)
VARIABLES = flags.DEFINE_list(
    'variables',
    None,
    help=(
        'Comma delimited list of variables to select from weather. By default,'
        ' all variables are selected.'
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


# pylint: disable=expression-not-assigned


def _impose_data_selection(
    ds: xr.Dataset, source_chunks: t.Mapping[str, int]
) -> tuple[xr.Dataset, t.Mapping[str, int]]:
  """Select a subset of ds and keep remaining chunk sizes."""
  selection = {
      TIME_DIM.value: slice(TIME_START.value, TIME_STOP.value),
  }
  if VARIABLES.value is not None:
    ds = ds[VARIABLES.value]
  ds = ds.sel({k: v for k, v in selection.items() if k in ds.dims})
  source_chunks = {
      # Some dimensions may be removed when we remove variables.
      k: v
      for k, v in source_chunks.items()
      if k in ds.dims
  }
  return ds, source_chunks


def main(argv: list[str]):
  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH.value)
  source_dataset, source_chunks = _impose_data_selection(
      source_dataset, source_chunks
  )
  template = xbeam.make_template(
      source_dataset.isel({REALIZATION_NAME.value: 0}, drop=True),
      # coordinates should not be lazy
      lazy_vars=source_dataset.data_vars.keys(),
  )
  target_chunks = {
      k: v for k, v in source_chunks.items() if k != REALIZATION_NAME.value
  }

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    (
        root
        | xbeam.DatasetToChunks(
            source_dataset,
            source_chunks,
            split_vars=True,
            num_threads=NUM_THREADS.value,
        )
        | xbeam.Mean(REALIZATION_NAME.value, skipna=SKIPNA.value)
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template,
            target_chunks,
            num_threads=NUM_THREADS.value,
        )
    )


if __name__ == '__main__':
  app.run(main)
