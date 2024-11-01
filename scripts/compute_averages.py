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
r"""Computes average over dimensions of a forecast dataset.

Example of getting the (average) vertical profile of temperature, by latitude.
  ```
  export BUCKET=my-bucket
  export PROJECT=my-project

  python scripts/compute_averages.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/era5/$USER/temperature-vertical-profile.zarr \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --averaging_dims=time,longitude \
    --variables=temperature \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./scripts/dataflow-requirements.txt \
    --job_name=compute-vertical-profile-$USER
  ```
"""
import typing as t

from absl import app
from absl import flags
import apache_beam as beam
from weatherbench2 import metrics
import xarray as xr
import xarray_beam as xbeam

INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')

AVERAGING_DIMS = flags.DEFINE_list(
    'averaging_dims',
    None,
    help=(
        'Comma delimited list of dimensions to average over. Required.  If'
        ' "latitude" is included, the averaging with be area weighted.'
    ),
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
LEVELS = flags.DEFINE_list(
    'levels',
    None,
    help=(
        'Comma delimited list of pressure levels to compute spectra on. If'
        ' empty, compute on all levels of --input_path'
    ),
)
VARIABLES = flags.DEFINE_list(
    'variables',
    None,
    help=(
        'Comma delimited list of data variables to include in output.  '
        'If empty, compute on all data_vars of --input_path'
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
FANOUT = flags.DEFINE_integer(
    'fanout',
    None,
    help='Beam CombineFn fanout. Might be required for large dataset.',
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)


# pylint: disable=expression-not-assigned


def _impose_data_selection(
    ds: xr.Dataset,
    source_chunks: t.Mapping[str, int],
) -> tuple[xr.Dataset, dict[str, int]]:
  """Select requested subset of data and trim chunks if needed."""
  if VARIABLES.value is not None:
    ds = ds[VARIABLES.value]
  selection = {
      TIME_DIM.value: slice(TIME_START.value, TIME_STOP.value),
  }
  if LEVELS.value:
    selection['level'] = [float(l) for l in LEVELS.value]
  ds = ds.sel({k: v for k, v in selection.items() if k in ds.dims})
  return ds, {k: v for k, v in source_chunks.items() if k in ds.dims}


def main(argv: list[str]):
  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH.value)
  source_dataset, source_chunks = _impose_data_selection(
      source_dataset, source_chunks
  )
  template = xbeam.make_template(
      source_dataset.isel({d: 0 for d in AVERAGING_DIMS.value}, drop=True)
  )
  target_chunks = {
      k: v for k, v in source_chunks.items() if k not in AVERAGING_DIMS.value
  }

  if 'latitude' in AVERAGING_DIMS.value:
    weights = metrics.get_lat_weights(source_dataset)
  else:
    weights = None

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    chunked = root | xbeam.DatasetToChunks(
        source_dataset,
        source_chunks,
        split_vars=True,
        num_threads=NUM_THREADS.value,
    )

    if weights is not None:
      chunked = chunked | beam.MapTuple(
          lambda k, v: (k, v * weights.reindex_like(v))
      )

    (
        chunked
        | xbeam.Mean(
            AVERAGING_DIMS.value, skipna=SKIPNA.value, fanout=FANOUT.value
        )
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template,
            target_chunks,
            num_threads=NUM_THREADS.value,
        )
    )


if __name__ == '__main__':
  app.run(main)
  flags.mark_flag_as_required(['averaging_dims'])
