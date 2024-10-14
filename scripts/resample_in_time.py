# Copyright 2024 Google LLC
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
r"""CLI to resample data to new time resolution.

The output will contain statistics of input variables specified by the flags
--mean_vars, --min_vars etc...  Variables not associated with any flag will not
appear in the output.


Example Usage:

  ```
  export BUCKET=my-bucket
  export PROJECT=my-project
  export REGION=us-central1

  python scripts/resample_in_time.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/era5/$USER/2020-2021-weekly-average-temperature.zarr \
    --runner=DataflowRunner \
    --time_start=2020 \
    --time_stop=2021 \
    --period=1w \
    --mean_vars=temperature \
    --working_chunks="latitude=12,longitude=12" \
    -- \
    --project=$PROJECT \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./scripts/dataflow-requirements.txt \
    --job_name=compute-ensemble-mean-$USER
  ```
"""

from collections import abc
import functools
import typing as t

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import pandas as pd
from weatherbench2 import flag_utils
import xarray as xr
import xarray_beam as xbeam

# Command line arguments
INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path.')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path.')

RUNNER = flags.DEFINE_string(
    'runner',
    None,
    help='beam.runners.Runner',
)
METHOD = flags.DEFINE_enum(
    'method',
    'resample',
    ['resample', 'rolling'],
    help=(
        'Whether to resample to new times (spaced by --period), or use a'
        ' rolling window. In either case, output at time index T uses the'
        ' window [T, T + period]. In particular, whether using resample or'
        ' rolling, output at matching times will be the same.'
    ),
)
PERIOD = flags.DEFINE_string(
    'period',
    '1d',
    help=(
        'Convertable to pandas.Timedelta. E.g. "1d" (one day) or "1w" (one'
        ' week). See pandas.to_timedelta.'
    ),
)
MEAN_VARS = flags.DEFINE_list(
    'mean_vars',
    [],
    help=(
        'Comma-delimited list of variables to compute the mean of. Will result'
        ' in "_mean" suffix added to variables unless --add_mean_suffix=false. '
        ' Setting --mean_vars=ALL is equivalent to listing every time dependent'
        ' variable.'
    ),
)
MIN_VARS = flags.DEFINE_list(
    'min_vars',
    [],
    help=(
        'Comma-delimited list of variables to compute the minimum of. Will'
        ' result in "_min" suffix added to variables. Setting --min_vars=ALL'
        ' results is equivalent to listing every time dependent variable.'
    ),
)
MAX_VARS = flags.DEFINE_list(
    'max_vars',
    [],
    help=(
        'Comma-delimited list of variables to compute the minimum of. Will'
        ' result in "_max" suffix added to variables. Setting --max_vars=ALL is'
        ' equivalent to listing every time dependent variable.'
    ),
)
ADD_MEAN_SUFFIX = flags.DEFINE_bool(
    'add_mean_suffix',
    False,
    help='Add suffix "_mean" to variable name when computing the mean.',
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)
TIME_DIM = flags.DEFINE_string(
    'time_dim', 'time', help='Name for the time dimension to slice data on.'
)
TIME_START = flags.DEFINE_string(
    'time_start',
    None,
    help=(
        'ISO 8601 timestamp (inclusive) at which to start resampling. If None,'
        ' use the first time in --input_path.'
    ),
)
TIME_STOP = flags.DEFINE_string(
    'time_stop',
    None,
    help=(
        'ISO 8601 timestamp (inclusive) at which to stop resampling. If None,'
        ' use the last time in --input_path.'
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
WORKING_CHUNKS = flag_utils.DEFINE_chunks(
    'working_chunks',
    '',
    help=(
        'Spatial chunk sizes to use during time downsampling, e.g.,'
        ' "longitude=10,latitude=10". May not include "--time_dim".  In other'
        ' words, the entire time series for each chunk is loaded into memory at'
        ' once. So if there are many times, --working_chunks should be small in'
        ' other dimensions.'
    ),
)

_ALL = 'ALL'  # Sentinal for including all variables.


def _get_vars(
    list_of_vars: list[str], time_dependent_vars: list[t.Hashable]
) -> list[t.Hashable]:
  """Get variables for a particular statistic."""
  if not list_of_vars:
    return []
  if len(list_of_vars) == 1 and list_of_vars[0] == _ALL:
    return time_dependent_vars
  if _ALL in list_of_vars:
    raise ValueError(
        f'Cannot specify both {_ALL} and other variables. Found {list_of_vars}'
    )
  return list_of_vars


def resample_in_time_chunk(
    key: xbeam.Key,
    chunk: xr.Dataset,
    method: str,
    period: pd.Timedelta,
    time_dim: str,
    mean_vars: list[str],
    min_vars: list[str],
    max_vars: list[str],
    add_mean_suffix: bool,
    skipna: bool = False,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Resample a data chunk in time and return a requested time statistic.

  Args:
    key: An xarray beam key into a data chunk.
    chunk: The data chunk.
    method: resample or rolling.
    period: The time frequency of the resampled data.
    time_dim: Dimension indexing time in chunk.
    mean_vars: Variables to compute the mean of.
    min_vars: Variables to compute the min of.
    max_vars: Variables to compute the max of.
    add_mean_suffix: Whether to add a "_mean" suffix to variables after
      computing the mean.
    skipna: Whether to skip NaN values in both forecasts and observations during
      evaluation.

  Returns:
    The resampled data chunk and its key.
  """
  # Remove time offset because each chunk contains the entire timeseries.
  rsmp_key = key.with_offsets(**{time_dim: None})

  rsmp_chunks = []
  for chunk_var in chunk.data_vars:
    if chunk_var in mean_vars:
      rsmp_chunks.append(
          resample_in_time_core(
              chunk, method, period, 'mean', skipna=skipna
          ).rename(
              {chunk_var: f'{chunk_var}_mean' if add_mean_suffix else chunk_var}
          )
      )
    if chunk_var in min_vars:
      rsmp_chunks.append(
          resample_in_time_core(
              chunk, method, period, 'min', skipna=skipna
          ).rename({chunk_var: f'{chunk_var}_min'})
      )
    if chunk_var in max_vars:
      rsmp_chunks.append(
          resample_in_time_core(
              chunk, method, period, 'max', skipna=skipna
          ).rename({chunk_var: f'{chunk_var}_max'})
      )

  return rsmp_key, xr.merge(rsmp_chunks)


def resample_in_time_core(
    chunk: t.Union[xr.Dataset, xr.DataArray],
    method: str,
    period: pd.Timedelta,
    statistic: str,
    skipna: bool,
) -> t.Union[xr.Dataset, xr.DataArray]:
  """Core call to xarray resample or rolling."""
  if method == 'rolling':
    delta_t = pd.to_timedelta(np.diff(chunk[TIME_DIM.value][:2].data)[0])
    if period % delta_t:
      raise ValueError(
          f'{delta_t=} between chunk times did not evenly divide {period=}'
      )
    return getattr(
        chunk.rolling(
            {TIME_DIM.value: period // delta_t}, center=False, min_periods=None
        ),
        statistic,
    )(skipna=skipna)
  elif method == 'resample':
    return getattr(
        chunk.resample({TIME_DIM.value: period}, label='left'),
        statistic,
    )(skipna=skipna)
  else:
    raise ValueError(f'Unhandled {method=}')


def main(argv: abc.Sequence[str]) -> None:

  ds, input_chunks = xbeam.open_zarr(INPUT_PATH.value)
  period = pd.to_timedelta(PERIOD.value)

  if TIME_START.value is not None or TIME_STOP.value is not None:
    ds = ds.sel({TIME_DIM.value: slice(TIME_START.value, TIME_STOP.value)})

  # Select the variables needed for statistics.
  time_dependent_vars = [k for k, v in ds.items() if TIME_DIM.value in v.dims]
  nontime_vars = set(ds).difference(time_dependent_vars)
  mean_vars = _get_vars(MEAN_VARS.value, time_dependent_vars)
  min_vars = _get_vars(MIN_VARS.value, time_dependent_vars)
  max_vars = _get_vars(MAX_VARS.value, time_dependent_vars)

  keep_vars = set(mean_vars).union(min_vars).union(max_vars)
  if keep_vars.intersection(nontime_vars):
    raise ValueError(
        'Statistics asked for on some variables that did not contain'
        f' {TIME_DIM.value}: {keep_vars.intersection(nontime_vars)}'
    )
  ds = ds[keep_vars]

  # To ensure results at time T use data from [T, T + period], an offset needs
  # to be added if the method is rolling.
  # It would be wonderful if this was the default, or possible with appropriate
  # kwargs in rolling, but alas...
  if METHOD.value == 'rolling':
    delta_ts = pd.to_timedelta(np.unique(np.diff(ds[TIME_DIM.value].data)))
    if len(delta_ts) != 1:
      raise ValueError(
          f'Input data must have constant spacing. Found {delta_ts}'
      )
    delta_t = delta_ts[0]
    ds = ds.assign_coords(
        {TIME_DIM.value: ds[TIME_DIM.value] - period + delta_t}
    )

  # Make the template
  if METHOD.value == 'resample':
    rsmp_times = resample_in_time_core(
        # All stats will give the same times, so use 'mean' arbitrarily.
        ds[TIME_DIM.value],
        METHOD.value,
        period,
        statistic='mean',
        skipna=SKIPNA.value,
    )[TIME_DIM.value]
  else:
    rsmp_times = ds[TIME_DIM.value]
  assert isinstance(ds, xr.Dataset)  # To satisfy pytype.
  rsmp_template = (
      xbeam.make_template(ds)
      .isel({TIME_DIM.value: 0}, drop=True)
      .expand_dims(
          {TIME_DIM.value: rsmp_times},
      )
  )
  template_copy = rsmp_template.copy()
  rsmp_template = rsmp_template[[]]  # Drop all variables...will add in below
  for var in mean_vars:
    rsmp_template = rsmp_template.assign(
        {f'{var}_mean' if ADD_MEAN_SUFFIX.value else var: template_copy[var]}
    )
  for var in min_vars:
    rsmp_template = rsmp_template.assign({f'{var}_min': template_copy[var]})
  for var in max_vars:
    rsmp_template = rsmp_template.assign({f'{var}_max': template_copy[var]})

  # We've changed ds (e.g. dropped vars), so the dims may have changed.
  # Therefore, input_chunks may no longer be valid.
  ds_chunks = {k: v for k, v in input_chunks.items() if k in ds.dims}

  # Get the working and output chunks
  ds_chunks_without_time = {
      k: v for k, v in ds_chunks.items() if k != TIME_DIM.value
  }
  working_chunks = ds_chunks_without_time.copy()
  working_chunks.update(WORKING_CHUNKS.value)
  if TIME_DIM.value in working_chunks:
    raise ValueError('cannot include time working chunks')
  working_chunks[TIME_DIM.value] = len(ds[TIME_DIM.value])

  output_chunks = ds_chunks.copy()
  output_chunks[TIME_DIM.value] = min(
      len(rsmp_times), output_chunks[TIME_DIM.value]
  )

  itemsize = max(var.dtype.itemsize for var in rsmp_template.values())

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    # Read, rechunk, compute stats
    unused_pcoll = (
        root
        | xbeam.DatasetToChunks(
            ds, ds_chunks, split_vars=True, num_threads=NUM_THREADS.value
        )
        | 'RechunkToWorkingChunks'
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            ds.sizes,
            ds_chunks,
            working_chunks,
            itemsize=itemsize,
        )
        | 'Stats'
        >> beam.MapTuple(
            functools.partial(
                resample_in_time_chunk,
                time_dim=TIME_DIM.value,
                method=METHOD.value,
                period=period,
                mean_vars=mean_vars,
                min_vars=min_vars,
                max_vars=max_vars,
                add_mean_suffix=ADD_MEAN_SUFFIX.value,
                skipna=SKIPNA.value,
            )
        )
        | 'RechunkToOutputChunks'
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            rsmp_template.sizes,
            working_chunks,
            output_chunks,
            itemsize=itemsize,
        )
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template=rsmp_template,
            zarr_chunks=output_chunks,
            num_threads=NUM_THREADS.value,
        )
    )


if __name__ == '__main__':
  app.run(main)
