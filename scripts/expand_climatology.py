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
r"""Expand a climatology dataset into forecasts for particular times.

Example Usage:
  ```
  export START_TIME=2017-01-01
  export STOP_TIME=2017-12-31
  export BUCKET=my-bucket
  export PROJECT=my-project
  export REGION=us-central1

  python scripts/expand_climatology.py \
    --input_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/era5-expanded-climatology/$USER/era5-expanded-climatology-2017.zarr/ \
    --time_start=$START_TIME \
    --time_stop=$STOP_TIME \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --region=$REGION \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./scripts/dataflow-requirements.txt \
    --job_name=expand-climatology-$USER
  ```
"""
from collections import abc
import math

from absl import app
from absl import flags
import apache_beam as beam
import pandas as pd
import xarray
import xarray_beam as xbeam

INPUT_PATH = flags.DEFINE_string(
    'input_path',
    None,
    help='path to hourly or daily climatology dataset',
)
OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    help='path to save outputs in Zarr format',
)
TIME_START = flags.DEFINE_string(
    'time_start',
    '2017-01-01',
    help='ISO 8601 timestamp (inclusive) at which to start outputs',
)
TIME_STOP = flags.DEFINE_string(
    'time_stop',
    '2017-12-31',
    help='ISO 8601 timestamp (inclusive) at which to stop outputs',
)
TIME_CHUNK_SIZE = flags.DEFINE_integer(
    'time_chunk_size',
    None,
    help='Desired integer chunk size. If not set, inferred from input chunks.',
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


def select_climatology(
    variable_name_and_time_slice: tuple[str, slice],
    climatology: xarray.Dataset,
    time_index: pd.DatetimeIndex,
    base_chunks: dict[str, int],
) -> abc.Iterator[tuple[xbeam.Key, xarray.Dataset]]:
  """Select climatology data matching time_index[time_slice]."""
  variable_name, time_slice = variable_name_and_time_slice
  chunk_times = time_index[time_slice]
  times_array = xarray.DataArray(
      chunk_times, dims=['time'], coords={'time': chunk_times}
  )
  if 'hour' in climatology.coords:
    sliced = climatology[[variable_name]].sel(
        dayofyear=times_array.dt.dayofyear, hour=times_array.dt.hour
    )
    del sliced.coords['dayofyear']
    del sliced.coords['hour']
  else:
    sliced = climatology[[variable_name]].sel(
        dayofyear=times_array.dt.dayofyear
    )
    del sliced.coords['dayofyear']

  key = xbeam.Key({'time': time_slice.start}, vars={variable_name})
  sliced = sliced.compute()
  target_chunks = {k: v for k, v in base_chunks.items() if k in sliced.dims}
  yield from xbeam.split_chunks(key, sliced, target_chunks)


def main(argv: list[str]) -> None:
  climatology, input_chunks = xbeam.open_zarr(INPUT_PATH.value)

  if 'hour' not in climatology.coords:
    hour_delta = 24
    time_dims = ['dayofyear']
  else:
    hour_delta = (climatology.hour[1] - climatology.hour[0]).item()
    time_dims = ['hour', 'dayofyear']

  times = pd.date_range(
      TIME_START.value, TIME_STOP.value, freq=hour_delta * pd.Timedelta('1h')
  )

  template = (
      xbeam.make_template(climatology)
      .isel({dim: 0 for dim in time_dims}, drop=True)
      .expand_dims(time=times)
  )

  if TIME_CHUNK_SIZE.value is None:
    time_chunk_size = input_chunks['dayofyear'] * input_chunks.get('hour', 1)
  else:
    time_chunk_size = TIME_CHUNK_SIZE.value

  time_chunk_count = math.ceil(times.size / time_chunk_size)
  variables = list(climatology.keys())
  base_chunks = {k: v for k, v in input_chunks.items() if k not in time_dims}
  output_chunks = dict(base_chunks)
  output_chunks['time'] = time_chunk_size

  # Beam type checking is broken with Python 3.10:
  # https://github.com/apache/beam/issues/24685
  beam.typehints.disable_type_annotations()

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | beam.Create([i * time_chunk_size for i in range(time_chunk_count)])
        | beam.Map(lambda start: slice(start, start + time_chunk_size))
        | beam.FlatMap(lambda index: [(v, index) for v in variables])
        | beam.Reshuffle()
        | beam.FlatMap(select_climatology, climatology, times, base_chunks)
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template=template,
            zarr_chunks=output_chunks,
            num_threads=NUM_THREADS.value,
        )
    )


if __name__ == '__main__':
  app.run(main)
  flags.mark_flag_as_required(['input_path', 'output_path'])
