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
r"""Switch forecasts from "initialization time" to "valid time".

When aggregating a collection of raw weather forecasts, it is most natural to
store them with a dimension corresponding to "initialization time" or "forecast
reference time," i.e., the time of the analysis from which the forecast is made.
For ECMWF's 15 day forecast, forecasts are initialized twice per day (0z and
12z).

Each forecast then makes predictions out for some time period into the future.
"Valid" time (or just "time") is the time for which the forecast predicts
weather. The difference between valid and initialization time is the "lead" time
delta or "forecast period." For example, a forecast initialized on January 1
with a +5 day lead is valid on January 6.

Given a collection of consecutive forecasts, we can make a 2D plot of available
init and valid times, where the line denotes a single forecast:

          valid time
       o----->
         o----->
  init     o----->
  time       o----->
               o----->
                 o----->

Raw forecasts are typically stored with initialization time as a dimension, but
for some evaluation purposes (e.g., to compute error from a ground-truth dataset
in WeatherBench), it can be more convenenient to look at forecasts with valid
time as a dimension, which makes them directly comparable to an analysis
dataset with a fixed set of valid times across all forecast leads. This is
illustrated below, where dots indicate missing values:

          valid time
       .
       . .
       . . .
       o | | ^
         o | | ^
  init     o | | ^
  time       o | | ^
               o | | ^
                 o | | ^
                   . . .
                     . .
                       .

This pipeline does this transformation.

Notes:
- It expects datasets with "time" and "prediction_timedelta"
  dimensions, matching the output of DeepMind's data pipeline:
  google3/learning/deepmind/research/weather/data_pipelines/hres/grib_to_zarr.py
- Initializing times are expected to be spaced at some multiple of the interval
  between forecast lead times.
- For more on forecast alignment, see the climpred documentation. This pipeline
  realigns forecasts from "same_inits" to "same_verifs":
  https://climpred.readthedocs.io/en/stable/alignment.html

Example Usage:
  ```
  export BUCKET=my-bucket
  export PROJECT=my-project
  export REGION=us-central1

  python scripts/convert_init_to_valid_time.py \
    --input_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/hres/$USER/2016-2022-0012-64x32_equiangular_with_poles_conservative_with_valid_times.zarr/ \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --region=$REGION \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./scripts/dataflow-requirements.txt \
    --job_name=init-to-valid-times-$USER
  ```
"""
from typing import Iterable, Mapping

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import pandas as pd
import xarray
import xarray_beam

INPUT_PATH = flags.DEFINE_string('input_path', None, help='zarr inputs')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='zarr outputs')
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)

TIME = 'time'
DELTA = 'prediction_timedelta'
INIT = 'init'


def get_forecast_offset_and_spacing(
    init_times: np.ndarray, lead_times: np.ndarray
) -> tuple[int, int]:
  """Calculate the offset & spacing between weather forecasts by valid time."""
  init_deltas = np.unique(np.diff(init_times))
  if init_deltas.size > 1:
    raise ValueError(f'initialization times are not equidistant: {init_deltas}')
  (init_delta,) = init_deltas

  lead_deltas = np.unique(np.diff(lead_times))
  if lead_deltas.size > 1:
    raise ValueError(f'lead times are not equidistant: {lead_deltas}')
  (lead_delta,) = lead_deltas

  forecast_spacing, remainder = divmod(init_delta, lead_delta)
  if remainder:
    raise ValueError(
        'initialization times not spaced at a multiple of lead times: '
        f'{lead_delta=}, {init_delta=}'
    )

  if lead_times[0] == np.timedelta64(0, 'h'):
    forecast_offset = 0
  else:
    forecast_offset = lead_times.tolist().index(forecast_spacing * lead_delta)

  return int(forecast_offset), int(forecast_spacing)


def get_axis(dataset: xarray.Dataset, dim: str) -> int:
  (axis,) = {array.dims.index(dim) for array in dataset.values()}
  return axis


def slice_along_timedelta_axis(
    key: xarray_beam.Key,
    chunk: xarray.Dataset,
    forecast_offset: int = 0,
    forecast_spacing: int = 1,
) -> Iterable[tuple[xarray_beam.Key, xarray.Dataset]]:
  """Select chunks to keep along the timedelta axis & update their keys."""
  offset = key.offsets[DELTA]
  new_offset, remainder = divmod(offset, forecast_spacing)
  if remainder == forecast_offset:
    new_key = key.with_offsets(**{DELTA: new_offset})
    yield new_key, chunk


def index_on_valid_time(
    key: xarray_beam.Key,
    chunk: xarray.Dataset,
) -> tuple[xarray_beam.Key, xarray.Dataset]:
  """Adjust keys and datasets in one chunk from init to valid time."""
  time_offset = key.offsets[INIT] + key.offsets[DELTA]
  new_key = key.with_offsets(init=None, time=time_offset)

  assert chunk.sizes[INIT] == 1 and chunk.sizes[DELTA] == 1
  squeezed = chunk.squeeze(INIT, drop=True)
  valid_times = chunk[INIT].data + chunk[DELTA].data
  new_chunk = squeezed.expand_dims(
      {TIME: valid_times}, axis=list(chunk.dims).index(INIT)
  )

  return new_key, new_chunk.astype(np.float32)


def iter_padding_chunks(
    _,
    template: xarray.Dataset,
    chunks: Mapping[str, int],
    source_time_index: pd.Index,
) -> Iterable[tuple[xarray_beam.Key, xarray.Dataset]]:
  """Yields all-NaN chunks for missing forecasts."""
  template_slice = template.head(time=1, prediction_timedelta=1)
  base_chunk = np.nan * xarray.zeros_like(template_slice).compute()
  chunks = {(TIME if k == INIT else k): v for k, v in chunks.items()}

  time_index = template.indexes[TIME]
  delta_index = template.indexes[DELTA]
  assert time_index.is_monotonic_increasing  # pytype: disable=attribute-error
  assert delta_index.is_monotonic_increasing  # pytype: disable=attribute-error

  def make_chunks(time, delta):
    i = time_index.get_loc(time)  # pytype: disable=attribute-error
    j = delta_index.get_loc(delta)  # pytype: disable=attribute-error
    key = xarray_beam.Key({TIME: i, DELTA: j})
    chunk = base_chunk.assign_coords(time=[time], prediction_timedelta=[delta])
    for key, chunk in xarray_beam.split_variables(key, chunk):
      key_chunks = {k: v for k, v in chunks.items() if k in chunk.dims}
      yield from xarray_beam.split_chunks(key, chunk, key_chunks)

  first_time = source_time_index[0]
  last_time = source_time_index[-1]

  for time in time_index:
    for delta in delta_index:
      init_time = time - delta
      if init_time < first_time or init_time > last_time:
        yield from make_chunks(time, delta)


def main(argv: list[str]) -> None:
  source_ds, chunks = xarray_beam.open_zarr(INPUT_PATH.value)

  # We'll use "time" only for "valid time" in this pipeline, so rename the
  # "time" dimension on the input to "init" (short for initialization time)
  source_ds = source_ds.rename({TIME: INIT})

  input_chunks = {INIT if k == TIME else k: v for k, v in chunks.items()}
  split_chunks = {
      k: 1 if k in [INIT, DELTA] else v for k, v in input_chunks.items()
  }
  output_chunks = chunks  # same as input Zarr

  forecast_offset, forecast_spacing = get_forecast_offset_and_spacing(
      source_ds.init.data, source_ds.prediction_timedelta.data
  )

  # Lead times that are not a multiple of the forecast spacing (e.g., the +6
  # hour prediction) would result in a dataset where some valid/lead time
  # combinations are missing. Instead, we drop these lead times.
  delta_slice = slice(forecast_offset, None, forecast_spacing)
  new_deltas = source_ds.prediction_timedelta.data[delta_slice]
  new_times = np.unique(
      source_ds.init.data[:, np.newaxis] + new_deltas[np.newaxis, :]
  )
  template = (
      xarray_beam.make_template(source_ds)
      .isel({INIT: 0}, drop=True)
      .expand_dims({TIME: new_times}, axis=get_axis(source_ds, INIT))
      .isel({DELTA: 0}, drop=True)
      .expand_dims({DELTA: new_deltas}, axis=get_axis(source_ds, DELTA))
      .astype(np.float32)  # ensure we can represent NaN
  )

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as p:
    padding = (
        p
        | beam.Create([None])  # dummy input
        | beam.FlatMap(
            iter_padding_chunks,
            template,
            split_chunks,
            source_ds.indexes[INIT],
        )
    )
    p |= xarray_beam.DatasetToChunks(
        source_ds, input_chunks, split_vars=True, num_threads=NUM_THREADS.value
    )
    if input_chunks != split_chunks:
      p |= xarray_beam.SplitChunks(split_chunks)
    p |= beam.FlatMapTuple(
        slice_along_timedelta_axis,
        forecast_offset=forecast_offset,
        forecast_spacing=forecast_spacing,
    )
    p |= beam.MapTuple(index_on_valid_time)
    p = (p, padding) | beam.Flatten()
    if input_chunks != split_chunks:
      p |= xarray_beam.ConsolidateChunks(output_chunks)
    p |= xarray_beam.ChunksToZarr(
        OUTPUT_PATH.value,
        template,
        output_chunks,
        num_threads=NUM_THREADS.value,
    )


if __name__ == '__main__':
  app.run(main)
