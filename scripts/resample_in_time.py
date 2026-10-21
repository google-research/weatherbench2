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
from typing import Optional
import warnings

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
SUM_VARS = flags.DEFINE_list(
    'sum_vars',
    [],
    help=(
        'Comma-delimited list of variables to compute the sum of. Will result'
        ' in a suffix (default "_sum") added to variables. Setting'
        ' --sum_vars=ALL is equivalent to listing every time dependent'
        ' variable.'
    ),
)
SUM_SUFFIX = flags.DEFINE_string(
    'sum_suffix',
    '_sum',
    help='Suffix to add to variable name when computing the sum.',
)
LABEL_SIDE = flags.DEFINE_enum(
    'label_side',
    'left',
    ['left', 'right'],
    help=(
        'Which side of the interval to label. "right" means the label T'
        ' corresponds to the interval (T - period, T]. "left" means the label T'
        ' corresponds to the interval [T, T + period].'
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
OUTPUT_TIME_START = flags.DEFINE_string(
    'output_time_start',
    None,
    help=(
        'ISO 8601 timestamp (inclusive) at which to start saving the output. '
        'Useful to trim initial NaNs when using rolling windows.'
    ),
)
OUTPUT_TIME_STOP = flags.DEFINE_string(
    'output_time_stop',
    None,
    help='ISO 8601 timestamp (inclusive) at which to stop saving the output.',
)
INPUT_SELECT_HOURS = flags.DEFINE_list(
    'input_select_hours',
    [],
    help=(
        'Comma-delimited list of hours to select from time dimension before'
        'resampling (e.g. to filter input data to specific times of day before'
        'computing statistics). E.g., --input_select_hours=0,6,12,18.'
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
OUTPUT_SELECT_HOURS = flags.DEFINE_list(
    'output_select_hours',
    [],
    help=(
        'Comma-delimited list of hours to select from time dimension after'
        ' resampling. E.g., --output_select_hours=0.'
    ),
)
USE_FAST_RESAMPLE = flags.DEFINE_bool(
    'use_fast_resample',
    True,
    help=(
        'Use map-side partial aggregation + CombinePerKey for method=resample '
        'instead of full-timeseries xbeam.Rechunk.'
    ),
)

_ALL = 'ALL'  # Sentinal for including all variables.


def _get_vars(
    list_of_vars: list[str], time_dependent_vars: list[str]
) -> list[str]:
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
    sum_vars: list[str],
    add_mean_suffix: bool,
    skipna: bool = False,
    output_select_hours: Optional[list[int]] = None,
    sum_suffix: str = '_sum',
    output_time_start: Optional[str] = None,
    output_time_stop: Optional[str] = None,
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
    sum_vars: Variables to compute the sum of.
    add_mean_suffix: Whether to add a "_mean" suffix to variables after
      computing the mean.
    skipna: Whether to skip NaN values in both forecasts and observations during
      evaluation.
    output_select_hours: List of hours to select from time dimension after
      resampling.
    sum_suffix: Suffix to add to variables when computing the sum.
    output_time_start: ISO 8601 timestamp to filter start of output.
    output_time_stop: ISO 8601 timestamp to filter end of output.

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
    if chunk_var in sum_vars:
      rsmp_chunks.append(
          resample_in_time_core(
              chunk, method, period, 'sum', skipna=skipna
          ).rename({chunk_var: f'{chunk_var}{sum_suffix}'})
      )

  merged_chunk = xr.merge(rsmp_chunks)
  if output_select_hours:
    merged_chunk = merged_chunk.sel(
        {time_dim: merged_chunk[time_dim].dt.hour.isin(output_select_hours)}
    )
  if output_time_start is not None or output_time_stop is not None:
    merged_chunk = merged_chunk.sel(
        {time_dim: slice(output_time_start, output_time_stop)}
    )

  return rsmp_key, merged_chunk


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
    if LABEL_SIDE.value == 'left':
      resampled = getattr(
          chunk.resample({TIME_DIM.value: period}, label='left'),
          statistic,
      )(skipna=skipna)
      return resampled
    elif LABEL_SIDE.value == 'right':
      resampled = getattr(
          chunk.resample(
              {TIME_DIM.value: period}, label='right', closed='right'
          ),
          statistic,
      )(skipna=skipna)
      return resampled.isel({TIME_DIM.value: slice(1, None)})
    else:
      raise ValueError(f'Unhandled {LABEL_SIDE.value=}')
  else:
    raise ValueError(f'Unhandled {method=}')
  assert False  # Should not be reached


def _emit_partial_resample(
    key: xbeam.Key,
    chunk: xr.Dataset,
    *,
    time_dim: str,
    in_to_out_idx: np.ndarray,
    out_t_chunk: int,
    mean_vars: set[str],
    min_vars: set[str],
    max_vars: set[str],
    sum_vars: set[str],
    skipna: bool,
) -> abc.Iterator[tuple[xbeam.Key, dict[int, dict[str, t.Any]]]]:
  """Pre-aggregates an input chunk into partial stats keyed by output chunk."""
  assert key.vars is not None
  var_name = str(next(iter(key.vars)))
  da = chunk[var_name]
  time_axis = da.dims.index(time_dim)
  t0 = key.offsets[time_dim]
  t_len = da.sizes[time_dim]
  out_ids = in_to_out_idx[t0 : t0 + t_len]
  valid_out_ids = np.unique(out_ids[out_ids >= 0])
  if valid_out_ids.size == 0:
    return

  arr = np.asarray(da.values)
  need_sum = (var_name in mean_vars) or (var_name in sum_vars)
  need_min = var_name in min_vars
  need_max = var_name in max_vars

  by_out_chunk: dict[int, dict[int, dict[str, t.Any]]] = {}
  for out_idx_np in valid_out_ids:
    out_idx = int(out_idx_np)
    idxs = np.flatnonzero(out_ids == out_idx)
    s_start, s_stop = int(idxs[0]), int(idxs[-1]) + 1
    if s_stop - s_start == idxs.size:
      slc: list[t.Any] = [slice(None)] * arr.ndim
      slc[time_axis] = slice(s_start, s_stop)
      sub = arr[tuple(slc)]
    else:
      sub = np.take(arr, idxs, axis=time_axis)

    stats: dict[str, t.Any] = {}
    if need_sum:
      if skipna:
        stats['sum'] = np.nansum(sub, axis=time_axis, dtype=np.float64)
        stats['count'] = np.sum(~np.isnan(sub), axis=time_axis, dtype=np.int32)
      else:
        stats['sum'] = np.sum(sub, axis=time_axis, dtype=np.float64)
        stats['count'] = int(idxs.size)
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
      if need_min:
        stats['min'] = (np.nanmin if skipna else np.min)(sub, axis=time_axis)
      if need_max:
        stats['max'] = (np.nanmax if skipna else np.max)(sub, axis=time_axis)

    out_chunk_offset = (out_idx // out_t_chunk) * out_t_chunk
    slot = out_idx - out_chunk_offset
    by_out_chunk.setdefault(out_chunk_offset, {})[slot] = stats

  for out_chunk_offset, acc in by_out_chunk.items():
    yield key.with_offsets(**{time_dim: out_chunk_offset}), acc


class _ResampleCombineFn(beam.CombineFn):
  """Combines partial per-slot statistics for an output Zarr chunk."""

  def __init__(self, skipna: bool):
    super().__init__()
    self.skipna = skipna

  def create_accumulator(self) -> dict[int, dict[str, t.Any]]:
    return {}

  def add_input(
      self,
      accumulator: dict[int, dict[str, t.Any]],
      element: dict[int, dict[str, t.Any]],
  ) -> dict[int, dict[str, t.Any]]:
    return self.merge_accumulators([accumulator, element])

  def merge_accumulators(
      self, accumulators: abc.Iterable[dict[int, dict[str, t.Any]]]
  ) -> dict[int, dict[str, t.Any]]:
    merged: dict[int, dict[str, t.Any]] = {}
    for acc in accumulators:
      for slot, stats in acc.items():
        if slot not in merged:
          merged[slot] = {
              k: v.copy() if isinstance(v, np.ndarray) else v
              for k, v in stats.items()
          }
        else:
          cur = merged[slot]
          if 'sum' in stats:
            cur['sum'] = cur['sum'] + stats['sum']
            cur['count'] = cur['count'] + stats['count']
          if 'min' in stats:
            cur['min'] = (
                np.fmin(cur['min'], stats['min'])
                if self.skipna
                else np.minimum(cur['min'], stats['min'])
            )
          if 'max' in stats:
            cur['max'] = (
                np.fmax(cur['max'], stats['max'])
                if self.skipna
                else np.maximum(cur['max'], stats['max'])
            )
    return merged

  def extract_output(
      self, accumulator: dict[int, dict[str, t.Any]]
  ) -> dict[int, dict[str, t.Any]]:
    return accumulator


def _finalize_output_chunk(
    key: xbeam.Key,
    acc: dict[int, dict[str, t.Any]],
    *,
    time_dim: str,
    rsmp_template: xr.Dataset,
    output_chunks: dict[str, int],
    mean_vars: set[str],
    min_vars: set[str],
    max_vars: set[str],
    sum_vars: set[str],
    add_mean_suffix: bool,
    sum_suffix: str,
    skipna: bool,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Builds the final output Zarr chunk Dataset from combined slot stats."""
  assert key.vars is not None
  var_name = str(next(iter(key.vars)))
  indexers = {
      dim: slice(
          offset, min(offset + output_chunks[dim], rsmp_template.sizes[dim])
      )
      for dim, offset in key.offsets.items()
  }
  chunk_tmpl = rsmp_template.isel(indexers)
  out_t_len = chunk_tmpl.sizes[time_dim]

  data_vars: dict[str, tuple[tuple[t.Hashable, ...], np.ndarray]] = {}
  specs: list[tuple[str, str]] = []
  if var_name in mean_vars:
    specs.append(('mean', f'{var_name}_mean' if add_mean_suffix else var_name))
  if var_name in min_vars:
    specs.append(('min', f'{var_name}_min'))
  if var_name in max_vars:
    specs.append(('max', f'{var_name}_max'))
  if var_name in sum_vars:
    specs.append(('sum', f'{var_name}{sum_suffix}'))

  for stat_kind, out_var_name in specs:
    tmpl_da = chunk_tmpl[out_var_name]
    time_axis = tmpl_da.dims.index(time_dim)
    orig_dtype = tmpl_da.dtype
    slice_shape = tuple(
        s for i, s in enumerate(tmpl_da.shape) if i != time_axis
    )
    slots: list[np.ndarray] = []
    for slot in range(out_t_len):
      if slot not in acc:
        slots.append(np.full(slice_shape, np.nan, dtype=orig_dtype))
        continue
      s = acc[slot]
      if stat_kind == 'mean':
        if skipna:
          val = np.divide(
              s['sum'],
              s['count'],
              out=np.full_like(s['sum'], np.nan),
              where=s['count'] > 0,
          ).astype(orig_dtype)
        else:
          val = (s['sum'] / s['count']).astype(orig_dtype)
      elif stat_kind == 'sum':
        val = s['sum'].astype(orig_dtype)
      elif stat_kind == 'min':
        val = s['min'].astype(orig_dtype)
      elif stat_kind == 'max':
        val = s['max'].astype(orig_dtype)
      else:
        raise ValueError(f'Unknown {stat_kind=}')
      slots.append(val)
    stacked = np.stack(slots, axis=time_axis)
    data_vars[out_var_name] = (tmpl_da.dims, stacked)

  out_ds = xr.Dataset(data_vars, coords=chunk_tmpl.coords)
  return key.replace(vars=set(out_ds.data_vars)), out_ds


def _compute_in_to_out_idx(
    ds: xr.Dataset,
    *,
    time_dim: str,
    method: str,
    period: pd.Timedelta,
    output_select_hours: list[int],
    output_time_slice: Optional[slice],
) -> np.ndarray:
  """Computes mapping from each input time index to its output time index."""
  idx_da = xr.DataArray(
      np.arange(len(ds[time_dim]), dtype=np.float64),
      coords={time_dim: ds[time_dim]},
      dims=[time_dim],
  )
  min_idx_da = resample_in_time_core(idx_da, method, period, 'min', skipna=True)
  max_idx_da = resample_in_time_core(idx_da, method, period, 'max', skipna=True)
  if output_select_hours:
    min_idx_da = min_idx_da.sel(
        {time_dim: min_idx_da[time_dim].dt.hour.isin(output_select_hours)}
    )
    max_idx_da = max_idx_da.sel(
        {time_dim: max_idx_da[time_dim].dt.hour.isin(output_select_hours)}
    )
  if output_time_slice is not None:
    min_idx_da = min_idx_da.sel({time_dim: output_time_slice})
    max_idx_da = max_idx_da.sel({time_dim: output_time_slice})
  assert isinstance(min_idx_da, xr.DataArray)
  assert isinstance(max_idx_da, xr.DataArray)
  in_to_out_idx = np.full(len(ds[time_dim]), -1, dtype=np.int32)
  for out_idx, (lo, hi) in enumerate(
      zip(
          np.asarray(min_idx_da.values),
          np.asarray(max_idx_da.values),
      )
  ):
    if not np.isnan(lo) and not np.isnan(hi):
      in_to_out_idx[int(lo) : int(hi) + 1] = out_idx
  return in_to_out_idx


def _fast_resample_stages(
    pcoll: beam.PCollection,
    *,
    time_dim: str,
    in_to_out_idx: np.ndarray,
    rsmp_template: xr.Dataset,
    output_chunks: dict[str, int],
    mean_vars: list[str],
    min_vars: list[str],
    max_vars: list[str],
    sum_vars: list[str],
    add_mean_suffix: bool,
    sum_suffix: str,
    skipna: bool,
) -> beam.PCollection:
  """Intermediate Beam stages for map-side partial aggregation + CombinePerKey."""
  # Instead of rechunking the full time series onto single workers before
  # computing resample statistics, this approach maps each input chunk into
  # partial per-output-slot statistics keyed by output Zarr chunk offset,
  # combines those partial statistics via CombinePerKey, and finalizes each
  # output chunk directly.
  mean_vars_set = {str(v) for v in mean_vars}
  min_vars_set = {str(v) for v in min_vars}
  max_vars_set = {str(v) for v in max_vars}
  sum_vars_set = {str(v) for v in sum_vars}
  out_t_chunk = max(1, int(output_chunks[time_dim]))
  return (
      pcoll
      | 'PartialResample'
      >> beam.FlatMapTuple(
          functools.partial(
              _emit_partial_resample,
              time_dim=time_dim,
              in_to_out_idx=in_to_out_idx,
              out_t_chunk=out_t_chunk,
              mean_vars=mean_vars_set,
              min_vars=min_vars_set,
              max_vars=max_vars_set,
              sum_vars=sum_vars_set,
              skipna=skipna,
          )
      )
      | 'CombineResample'
      >> beam.CombinePerKey(_ResampleCombineFn(skipna=skipna))
      | 'FinalizeOutputChunk'
      >> beam.MapTuple(
          functools.partial(
              _finalize_output_chunk,
              time_dim=time_dim,
              rsmp_template=rsmp_template,
              output_chunks=output_chunks,
              mean_vars=mean_vars_set,
              min_vars=min_vars_set,
              max_vars=max_vars_set,
              sum_vars=sum_vars_set,
              add_mean_suffix=add_mean_suffix,
              sum_suffix=sum_suffix,
              skipna=skipna,
          )
      )
  )


def _rechunk_resample_stages(
    pcoll: beam.PCollection,
    *,
    ds: xr.Dataset,
    ds_chunks: dict[str, int],
    working_chunks: dict[str, int],
    output_chunks: dict[str, int],
    rsmp_template: xr.Dataset,
    itemsize: int,
    time_dim: str,
    method: str,
    period: pd.Timedelta,
    mean_vars: list[str],
    min_vars: list[str],
    max_vars: list[str],
    sum_vars: list[str],
    add_mean_suffix: bool,
    skipna: bool,
    output_select_hours: list[int],
    sum_suffix: str,
    output_time_start: Optional[str],
    output_time_stop: Optional[str],
) -> beam.PCollection:
  """Intermediate Beam stages using full-timeseries Rechunk before stats."""
  # Rechunks input chunks to working_chunks spanning the full time dimension,
  # computes rolling or resample statistics on each full time series via xarray,
  # and rechunks back to output_chunks.
  return (
      pcoll
      | 'RechunkToWorkingChunks'
      >> xbeam.Rechunk(
          {str(k): v for k, v in ds.sizes.items()},
          ds_chunks,
          working_chunks,
          itemsize=itemsize,
      )
      | 'Stats'
      >> beam.MapTuple(
          functools.partial(
              resample_in_time_chunk,
              time_dim=time_dim,
              method=method,
              period=period,
              mean_vars=mean_vars,
              min_vars=min_vars,
              max_vars=max_vars,
              sum_vars=sum_vars,
              add_mean_suffix=add_mean_suffix,
              skipna=skipna,
              output_select_hours=output_select_hours,
              sum_suffix=sum_suffix,
              output_time_start=output_time_start,
              output_time_stop=output_time_stop,
          )
      )
      | 'RechunkToOutputChunks'
      >> xbeam.Rechunk(
          {str(k): v for k, v in rsmp_template.sizes.items()},
          working_chunks,
          output_chunks,
          itemsize=itemsize,
      )
  )


def main(argv: abc.Sequence[str]) -> None:

  ds, input_chunks = xbeam.open_zarr(INPUT_PATH.value)
  period = pd.to_timedelta(PERIOD.value)

  if TIME_START.value is not None or TIME_STOP.value is not None:
    ds = ds.sel({TIME_DIM.value: slice(TIME_START.value, TIME_STOP.value)})

  if INPUT_SELECT_HOURS.value:
    input_select_hours = [int(h) for h in INPUT_SELECT_HOURS.value]
    ds = ds.sel(
        {TIME_DIM.value: ds[TIME_DIM.value].dt.hour.isin(input_select_hours)}
    )

  output_select_hours = [int(h) for h in OUTPUT_SELECT_HOURS.value]
  output_time_slice = None
  if OUTPUT_TIME_START.value is not None or OUTPUT_TIME_STOP.value is not None:
    output_time_slice = slice(OUTPUT_TIME_START.value, OUTPUT_TIME_STOP.value)

  # Select the variables needed for statistics.
  time_dependent_vars = [
      str(k) for k, v in ds.items() if TIME_DIM.value in v.dims
  ]
  nontime_vars = {str(k) for k in ds}.difference(time_dependent_vars)
  mean_vars = _get_vars(MEAN_VARS.value, time_dependent_vars)
  min_vars = _get_vars(MIN_VARS.value, time_dependent_vars)
  max_vars = _get_vars(MAX_VARS.value, time_dependent_vars)
  sum_vars = _get_vars(SUM_VARS.value, time_dependent_vars)

  keep_vars = set(mean_vars).union(min_vars).union(max_vars).union(sum_vars)
  if keep_vars.intersection(nontime_vars):
    raise ValueError(
        'Statistics asked for on some variables that did not contain'
        f' {TIME_DIM.value}: {keep_vars.intersection(nontime_vars)}'
    )
  ds = ds[keep_vars]

  if METHOD.value == 'rolling':
    delta_ts = pd.to_timedelta(np.unique(np.diff(ds[TIME_DIM.value].data)))
    if len(delta_ts) != 1:
      raise ValueError(
          f'Input data must have constant spacing. Found {delta_ts}'
      )
    delta_t = delta_ts[0]
    if LABEL_SIDE.value == 'left':
      # To ensure results at time T use data from [T, T + period],
      # an offset needs to be added if the method is rolling.
      ds = ds.assign_coords(
          {TIME_DIM.value: ds[TIME_DIM.value] - period + delta_t}
      )
    elif LABEL_SIDE.value == 'right':
      # To ensure results at time T use data from (T-period, T],
      # no offset needs to be added if the method is rolling, because
      # rolling(center=False) evaluates the window [T-period+delta_t, T]
      # and assigns it to T.
      pass
    else:
      raise ValueError(f'Unhandled {LABEL_SIDE.value=}')
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

  if output_select_hours:
    rsmp_times = rsmp_times.sel(
        {TIME_DIM.value: rsmp_times.dt.hour.isin(output_select_hours)}
    )

  if output_time_slice is not None:
    rsmp_times = rsmp_times.sel({TIME_DIM.value: output_time_slice})

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
  for var in sum_vars:
    rsmp_template = rsmp_template.assign(
        {f'{var}{SUM_SUFFIX.value}': template_copy[var]}
    )

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

  in_to_out_idx = None
  if METHOD.value == 'resample' and USE_FAST_RESAMPLE.value:
    in_to_out_idx = _compute_in_to_out_idx(
        ds,
        time_dim=TIME_DIM.value,
        method=METHOD.value,
        period=period,
        output_select_hours=output_select_hours,
        output_time_slice=output_time_slice,
    )

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    pcoll = root | xbeam.DatasetToChunks(
        ds, ds_chunks, split_vars=True, num_threads=NUM_THREADS.value
    )
    if in_to_out_idx is not None:
      pcoll = _fast_resample_stages(
          pcoll,
          time_dim=TIME_DIM.value,
          in_to_out_idx=in_to_out_idx,
          rsmp_template=rsmp_template,
          output_chunks=output_chunks,
          mean_vars=mean_vars,
          min_vars=min_vars,
          max_vars=max_vars,
          sum_vars=sum_vars,
          add_mean_suffix=ADD_MEAN_SUFFIX.value,
          sum_suffix=SUM_SUFFIX.value,
          skipna=SKIPNA.value,
      )
    else:
      pcoll = _rechunk_resample_stages(
          pcoll,
          ds=ds,
          ds_chunks=ds_chunks,
          working_chunks=working_chunks,
          output_chunks=output_chunks,
          rsmp_template=rsmp_template,
          itemsize=itemsize,
          time_dim=TIME_DIM.value,
          method=METHOD.value,
          period=period,
          mean_vars=mean_vars,
          min_vars=min_vars,
          max_vars=max_vars,
          sum_vars=sum_vars,
          add_mean_suffix=ADD_MEAN_SUFFIX.value,
          skipna=SKIPNA.value,
          output_select_hours=output_select_hours,
          sum_suffix=SUM_SUFFIX.value,
          output_time_start=OUTPUT_TIME_START.value,
          output_time_stop=OUTPUT_TIME_STOP.value,
      )
    unused_pcoll = pcoll | xbeam.ChunksToZarr(
        OUTPUT_PATH.value,
        template=rsmp_template,
        zarr_chunks=output_chunks,
        num_threads=NUM_THREADS.value,
    )


if __name__ == '__main__':
  app.run(main)
