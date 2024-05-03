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
r"""CLI to compute raw statistical moments of a dataset."""
import functools
from typing import Any, Optional

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import xarray as xr
import xarray_beam as xbeam


# Command line arguments
INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
START_YEAR = flags.DEFINE_integer(
    'start_year', 1990, help='Inclusive start year of climatology'
)
END_YEAR = flags.DEFINE_integer(
    'end_year', 2020, help='Inclusive end year of climatology'
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')
RECHUNK_ITEMSIZE = flags.DEFINE_integer(
    'rechunk_itemsize', 4, help='Itemsize for rechunking.'
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)


def moment_reduce(
    obs: xr.Dataset,
    order: str = 'first',
    reduce_dims: tuple[str, ...] = ('longitude', 'latitude'),
    sel_kwargs: Optional[dict[str, Any]] = None,
) -> xr.Dataset:
  """Reduce a dataset along dimensions by computing a raw statistical moment.

  The zeroth statistical moment is defined here using `np.nan^0 = np.nan`,
  and omitting all nans over the expectation. Thus, it defines the fraction
  of non-nan elements in the input. All other moments follow their standard
  definition.

  Args:
    obs: Input dataset.
    order: Order of the raw statistical moment to compute.
    reduce_dims: Dimensions over which dataset is reduced.
    sel_kwargs: Selection keyword arguments for input dataset.

  Returns:
    A dataset with all variables transformed to the given statistical moment.
  """
  obs = obs.load()
  if order == 'zeroth':
    non_nan_obs = xr.apply_ufunc(np.logical_not, xr.apply_ufunc(np.isnan, obs))
    return non_nan_obs.sel(sel_kwargs).mean(dim=reduce_dims)
  elif order == 'first':
    return obs.sel(sel_kwargs).mean(dim=reduce_dims)
  elif order == 'second':
    sq_obs = xr.apply_ufunc(np.square, obs)
    return sq_obs.sel(sel_kwargs).mean(dim=reduce_dims)
  else:
    raise NotImplementedError(f'Order {order} not implemented.')


def moment_reduce_spatial_chunk(
    obs_key: xbeam.Key,
    obs_chunk: xr.Dataset,
    *,
    order: str = 'first',
) -> tuple[xbeam.Key, xr.Dataset]:
  """Reduce a chunk by computing a statistical moment in space."""
  stat_key = obs_key.with_offsets(longitude=None, latitude=None)
  stat_key = stat_key.replace(vars={f'{var}_{order}' for var in stat_key.vars})
  for var in obs_chunk:
    obs_chunk = obs_chunk.rename({var: f'{var}_{order}'})
  stat_chunk = moment_reduce(obs_chunk, order=order)
  return stat_key, stat_chunk


def mean_reduce_time_chunk(
    obs_key: xbeam.Key,
    obs_chunk: xr.Dataset,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Reduce a chunk by averaging in time."""
  stat_key = obs_key.with_offsets(time=None, moment=0)
  # Note that all elementwise operations are performed in first reduce,
  # so order is fixed to 1 here.
  stat_chunk = moment_reduce(
      obs_chunk, reduce_dims=('time',), order='first'
  ).expand_dims('moment')
  return stat_key, stat_chunk


def main(argv: list[str]) -> None:
  orders = ['zeroth', 'first', 'second']

  obs, input_chunks = xbeam.open_zarr(INPUT_PATH.value)
  if START_YEAR.value is not None and END_YEAR.value is not None:
    time_slice = (str(START_YEAR.value), str(END_YEAR.value))
    obs = obs.sel(time=slice(*time_slice))

  # Chunks for spatial reduce
  reduce_working_chunks = {
      k: v
      for k, v in input_chunks.items()
      if k != 'longitude' and k != 'latitude'
  }
  # Spatial reduce template
  space_reduce_template = xbeam.make_template(obs).isel(
      {'longitude': 0, 'latitude': 0}, drop=True
  )

  # Chunks for time reduce
  time_working_chunks = dict(level=reduce_working_chunks['level'], time=-1)
  # Output chunks
  output_chunks = dict(level=reduce_working_chunks['level'], moment=1)

  # Add new vars to template
  raw_vars = list(space_reduce_template)
  for stat in orders:
    for var in raw_vars:
      space_reduce_template = space_reduce_template.assign(
          {f'{var}_{stat}': space_reduce_template[var]}
      )
  for var in raw_vars:
    space_reduce_template = space_reduce_template.drop(var)

  output_template = space_reduce_template.isel(
      {'time': 0}, drop=True
  ).expand_dims('moment')

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    # Read
    pcoll = root | xbeam.DatasetToChunks(
        obs, input_chunks, split_vars=True, num_threads=NUM_THREADS.value
    )

    # Branches to compute statistical moments
    pcolls = []
    for order in orders:
      # Reduce in space
      pcoll_tmp = pcoll | f'SpaceReduce_{order}' >> beam.MapTuple(
          functools.partial(moment_reduce_spatial_chunk, order=order)
      )
      # Rechunk in time
      pcoll_time = pcoll_tmp | f'RechunkTime_{order}' >> xbeam.Rechunk(
          # Convert to string to satisfy pytype.
          {str(k): v for k, v in space_reduce_template.sizes.items()},
          reduce_working_chunks,
          time_working_chunks,
          itemsize=RECHUNK_ITEMSIZE.value,
      )
      # Average in time
      pcoll_final = pcoll_time | f'TimeReduce_{order}' >> beam.MapTuple(
          mean_reduce_time_chunk
      )
      pcolls.append(pcoll_final)

    # Flatten branches and write output
    _ = (
        pcolls
        | 'Flatten' >> beam.Flatten()
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template=output_template,
            zarr_chunks=output_chunks,
            num_threads=NUM_THREADS.value,
        )
    )


if __name__ == '__main__':
  app.run(main)
