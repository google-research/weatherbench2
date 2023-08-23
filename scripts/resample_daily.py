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
r"""CLI to resample data to daily resolution."""

from collections import abc
import functools

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from weatherbench2 import flag_utils
import xarray as xr
import xarray_beam as xbeam

# Command line arguments
INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path.')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path.')

BEAM_RUNNER = flags.DEFINE_string(
    'beam_runner',
    None,
    help='beam.runners.Runner',
)
STATISTICS = flags.DEFINE_list(
    'statistics',
    ['mean'],
    help='Output resampled time statistics, from "mean", "min", or "max".',
)
ADD_STATISTIC_SUFFIX = flags.DEFINE_bool(
    'add_statistic_suffix',
    False,
    'Add suffix of statistic to variable name. Required for >1 statistic.',
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads', None, help='Number of chunks to load in parallel per worker.'
)
START_YEAR = flags.DEFINE_integer(
    'start_year', None, help='Start year (inclusive).'
)
END_YEAR = flags.DEFINE_integer('end_year', None, help='End year (inclusive).')
WORKING_CHUNKS = flag_utils.DEFINE_chunks(
    'working_chunks',
    '',
    help=(
        'Spatial chunk sizes to use during time downsampling, '
        'e.g., "longitude=10,latitude=10". They may not include "time".'
    ),
)


def resample_in_time_chunk(
    obs_key: xbeam.Key,
    obs_chunk: xr.Dataset,
    *,
    resampled_frequency: str = '1d',
    statistic: str = 'mean',
    add_statistic_suffix: bool = False,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Resample a data chunk in time and return a requested time statistic.

  Args:
    obs_key: An xarray beam key into a data chunk.
    obs_chunk: The data chunk.
    resampled_frequency: The time frequency of the resampled data.
    statistic: The statistic used for time aggregation. It can be `mean`, `min`,
      or `max`.
    add_statistic_suffix: Whether to append the statistic name as a suffix to
      all output variables.

  Returns:
    The resampled data chunk and its key.
  """
  rsmp_key = obs_key.with_offsets(time=None)
  rsmp_chunk = obs_chunk.resample(time=resampled_frequency)

  if statistic == 'mean':
    rsmp_chunk = rsmp_chunk.mean()
  elif statistic == 'min':
    rsmp_chunk = rsmp_chunk.min()
  elif statistic == 'max':
    rsmp_chunk = rsmp_chunk.max()

  # Append time statistic to var name.
  if add_statistic_suffix:
    for var in rsmp_chunk:
      rsmp_chunk = rsmp_chunk.rename({var: f'{var}_{statistic}'})
    rsmp_key = rsmp_key.replace(
        vars={f'{var}_{statistic}' for var in rsmp_key.vars}
    )
  return rsmp_key, rsmp_chunk


def main(argv: abc.Sequence[str]) -> None:
  if not ADD_STATISTIC_SUFFIX.value and len(STATISTICS.value) > 1:
    raise ValueError('--add_statistic_suffix is required for >1 statistics.')

  obs, input_chunks = xbeam.open_zarr(INPUT_PATH.value)
  if START_YEAR.value is not None and END_YEAR.value is not None:
    time_slice = (str(START_YEAR.value), str(END_YEAR.value))
    obs = obs.sel(time=slice(*time_slice))
  # drop static variables, for which time resampling would fail
  obs = obs.drop_vars([k for k, v in obs.items() if 'time' not in v.dims])

  # Get output times at daily resolution
  orig_times = obs.coords['time'].values
  daily_times = np.arange(
      orig_times.min(),
      orig_times.max() + np.timedelta64(1, 'D'),
      dtype='datetime64[D]',
  )

  input_chunks_without_time = {
      k: v for k, v in input_chunks.items() if k != 'time'
  }
  working_chunks = input_chunks_without_time.copy()
  working_chunks.update(WORKING_CHUNKS.value)
  if 'time' in working_chunks:
    raise ValueError('cannot include time in working chunks')
  in_working_chunks = dict(working_chunks, time=-1)
  out_working_chunks = dict(working_chunks, time=-1)

  output_chunks = input_chunks.copy()

  rsmp_template = (
      xbeam.make_template(obs)
      .isel(time=0, drop=True)
      .expand_dims(
          time=daily_times,
      )
  )
  if ADD_STATISTIC_SUFFIX.value:
    raw_vars = list(rsmp_template)
    # Append time statistic to var name.
    for var in raw_vars:
      for stat in STATISTICS.value:
        rsmp_template = rsmp_template.assign(
            {f'{var}_{stat}': rsmp_template[var]}
        )
      rsmp_template = rsmp_template.drop(var)

  itemsize = max(var.dtype.itemsize for var in rsmp_template.values())

  with beam.Pipeline(runner=BEAM_RUNNER.value, argv=argv) as root:
    # Read and rechunk
    pcoll = (
        root
        | xbeam.DatasetToChunks(
            obs, input_chunks, split_vars=True, num_threads=NUM_THREADS.value
        )
        | 'RechunkIn'
        >> xbeam.Rechunk(
            obs.sizes, input_chunks, in_working_chunks, itemsize=itemsize
        )
    )

    # Branches to compute statistics
    pcolls = []
    for stat in STATISTICS.value:
      pcoll_tmp = pcoll | f'{stat}' >> beam.MapTuple(
          functools.partial(
              resample_in_time_chunk,
              resampled_frequency='1d',
              statistic=stat,
              add_statistic_suffix=ADD_STATISTIC_SUFFIX.value,
          )
      )
      pcolls.append(pcoll_tmp)

    # Rechunk and write to file
    _ = (
        pcolls
        | beam.Flatten()
        | 'RechunkOut'
        >> xbeam.Rechunk(
            rsmp_template.sizes,
            out_working_chunks,
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
