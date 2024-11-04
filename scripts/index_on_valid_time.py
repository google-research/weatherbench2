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
r"""Index forecasts on "valid time".

When aggregating a collection of raw weather forecasts, it is most natural to
store them with a dimension corresponding to "initialization time" or "forecast
reference time," and "prediction timedelta" or "lead time." i.e., the time of
the analysis from which the forecast is made, and the length of the forecast.
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

This pipeline transforms either
 (i) (init, lead) -> (valid, lead)
OR
 (ii) (init, lead) -> (init, valid)

Notes:
- It expects datasets with "time" and "prediction_timedelta"
  dimensions.
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
from typing import Iterable, Mapping, Optional

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import pandas as pd
import xarray
import xarray_beam

INPUT_PATH = flags.DEFINE_string('input_path', None, help='zarr inputs')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='zarr outputs')
DESIRED_TIME_DIMS = flags.DEFINE_enum(
    'desired_time_dims',
    'valid_and_delta',
    ['valid_and_delta', 'valid_and_init'],
    help=(
        'The output is always indexed on "valid time" (with name "time"). This '
        'FLAG determines whether the other dimension is the timedelta ("delta")'
        ' or initial time ("init").'
    ),
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)

TIME = 'time'
DELTA = 'prediction_timedelta'
INIT = 'init'

VALID_AND_DELTA = 'valid_and_delta'
VALID_AND_INIT = 'valid_and_init'


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


def slice_along_timedelta_axis_if_index_on_timedelta(
    key: xarray_beam.Key,
    chunk: xarray.Dataset,
    forecast_offset: int = 0,
    forecast_spacing: int = 1,
) -> Iterable[tuple[xarray_beam.Key, xarray.Dataset]]:
  """Select chunks to keep along the other index axis & update their keys."""
  if DESIRED_TIME_DIMS.value == VALID_AND_DELTA:
    offset = key.offsets[DELTA]
    new_offset, remainder = divmod(offset, forecast_spacing)
    if remainder == forecast_offset:
      new_key = key.with_offsets(**{DELTA: new_offset})
      yield new_key, chunk
  else:
    yield key, chunk


def index_on_valid_time(
    key: xarray_beam.Key,
    chunk: xarray.Dataset,
    dropped_dim: Optional[str] = None,
    forecast_spacing: Optional[int] = None,
) -> tuple[xarray_beam.Key, xarray.Dataset]:
  """Adjust keys and datasets in one chunk from init to valid time."""
  if DESIRED_TIME_DIMS.value == VALID_AND_DELTA:
    # Recall we kept only every forecast_spacing timedelta if indexing on
    # timedelta. This means we don't have to account for forecast spacing in the
    # offsets.
    time_offset = key.offsets[INIT] + key.offsets[DELTA]
  else:
    # In this case, we kept all timedeltas.
    time_offset = key.offsets[INIT] * forecast_spacing + key.offsets[DELTA]
  new_key = key.with_offsets(**{TIME: time_offset, dropped_dim: None})

  assert chunk.sizes[INIT] == 1 and chunk.sizes[DELTA] == 1
  squeezed = chunk.squeeze(dropped_dim, drop=True)
  valid_times = chunk[INIT].data + chunk[DELTA].data
  new_chunk = squeezed.expand_dims(
      {TIME: valid_times}, axis=list(chunk.dims).index(dropped_dim)
  )

  return new_key, new_chunk.astype(np.float32)


def iter_padding_chunks(
    _,
    template: xarray.Dataset,
    chunks: Mapping[str, int],
    source_index: pd.Index,
    other_index_dim: str,
    dropped_dim: str,
) -> Iterable[tuple[xarray_beam.Key, xarray.Dataset]]:
  """Yields all-NaN chunks for missing forecasts."""
  template_slice = template.head({TIME: 1, other_index_dim: 1})
  base_chunk = np.nan * xarray.zeros_like(template_slice).compute()
  chunks = {(TIME if k == dropped_dim else k): v for k, v in chunks.items()}

  time_index = template.indexes[TIME]
  other_index = template.indexes[other_index_dim]
  assert time_index.is_monotonic_increasing  # pytype: disable=attribute-error
  assert other_index.is_monotonic_increasing  # pytype: disable=attribute-error

  def make_chunks(time, other):
    """Make all-NaN Chunks."""
    i = time_index.get_loc(time)  # pytype: disable=attribute-error
    j = other_index.get_loc(other)  # pytype: disable=attribute-error
    key = xarray_beam.Key({TIME: i, other_index_dim: j})
    chunk = base_chunk.assign_coords(
        {TIME: [time], other_index_dim: [other]},
    )
    for key, chunk in xarray_beam.split_variables(key, chunk):
      key_chunks = {k: v for k, v in chunks.items() if k in chunk.dims}
      yield from xarray_beam.split_chunks(key, chunk, key_chunks)

  # These two blocks could be combined but would be less readable.
  if DESIRED_TIME_DIMS.value == VALID_AND_DELTA:
    first_time = source_index[0]
    last_time = source_index[-1]
    for time in time_index:
      for delta in other_index:
        init_time = time - delta
        if init_time < first_time or init_time > last_time:
          yield from make_chunks(time, delta)
  else:
    source_delta_min = source_index[0]
    source_delta_max = source_index[-1]
    for time in time_index:
      for init_time in other_index:
        delta = time - init_time
        if delta < source_delta_min or delta > source_delta_max:
          yield from make_chunks(time, init_time)


def main(argv: list[str]) -> None:
  source_ds, chunks = xarray_beam.open_zarr(INPUT_PATH.value)

  if DESIRED_TIME_DIMS.value == VALID_AND_DELTA:
    other_index_dim = DELTA
    dropped_dim = INIT
  else:
    other_index_dim = INIT
    dropped_dim = DELTA

  # We'll use "time" only for "valid time" in this pipeline, so rename the
  # "time" dimension on the input to "init" (short for initialization time)
  source_ds = source_ds.rename({TIME: INIT})

  input_chunks = {INIT if k == TIME else k: v for k, v in chunks.items()}
  split_chunks = {
      k: 1 if k in [INIT, DELTA] else v for k, v in input_chunks.items()
  }

  # Replace dropped_dim with other_index_dim (using the same chunksize),
  # unless that dim already had a chunksize.
  # This will e.g. use the INIT chunksize for DELTA if VALID_AND_DELTA.
  output_chunks = {}
  for k, v in chunks.items():
    if k == dropped_dim:
      if other_index_dim not in chunks:
        output_chunks[other_index_dim] = v
    else:
      output_chunks[k] = v

  forecast_offset, forecast_spacing = get_forecast_offset_and_spacing(
      source_ds.init.data, source_ds.prediction_timedelta.data
  )

  if DESIRED_TIME_DIMS.value == VALID_AND_DELTA:
    # Lead times that are not a multiple of the forecast spacing (e.g., the +6
    # hour prediction) would result in a dataset where many valid/lead time
    # combinations are missing. Instead, we drop these lead times.
    delta_slice = slice(forecast_offset, None, forecast_spacing)
  else:
    # If indexing on init time, we can keep every timedelta.
    delta_slice = slice(None)
  new_deltas = source_ds[DELTA].data[delta_slice]
  new_times = np.unique(
      source_ds.init.data[:, np.newaxis] + new_deltas[np.newaxis, :]
  )
  if DESIRED_TIME_DIMS.value == VALID_AND_DELTA:
    template = (
        xarray_beam.make_template(source_ds)
        .isel({INIT: 0}, drop=True)
        .expand_dims({TIME: new_times}, axis=get_axis(source_ds, INIT))
        .isel({DELTA: 0}, drop=True)
        .expand_dims({DELTA: new_deltas}, axis=get_axis(source_ds, DELTA))
        .astype(np.float32)  # ensure we can represent NaN
    )
  else:  # Else index on INIT and drop DELTA
    template = (
        xarray_beam.make_template(source_ds)
        .isel({DELTA: 0}, drop=True)
        .expand_dims({TIME: new_times}, axis=get_axis(source_ds, DELTA))
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
            source_ds.indexes[dropped_dim],
            other_index_dim=other_index_dim,
            dropped_dim=dropped_dim,
        )
    )
    p |= xarray_beam.DatasetToChunks(
        source_ds, input_chunks, split_vars=True, num_threads=NUM_THREADS.value
    )
    if input_chunks != split_chunks:
      p |= xarray_beam.SplitChunks(split_chunks)
    p |= beam.FlatMapTuple(
        slice_along_timedelta_axis_if_index_on_timedelta,
        forecast_offset=forecast_offset,
        forecast_spacing=forecast_spacing,
    )
    p |= beam.MapTuple(
        index_on_valid_time,
        dropped_dim=dropped_dim,
        forecast_spacing=forecast_spacing,
    )
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
