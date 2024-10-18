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
r"""CLI to compute and save climatology.

Example Usage:
  ```
  export BUCKET=my-bucket
  export PROJECT=my-project
  export REGION=us-central1

  python scripts/compute_climatology.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/ear5-hourly-climatology/$USER/1990_to_2020_1h_64x32_equiangular_with_poles_conservative.zarr \
    --working_chunks="level=1,longitude=4,latitude=4" \
    --output_chunks="level=1,hour=3" \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --region=$REGION \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --job_name=compute-climatology-$USER
  ```
"""
import ast
import functools
from typing import Callable, Optional, Union

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from weatherbench2 import flag_utils
from weatherbench2 import utils
import xarray as xr
import xarray_beam as xbeam

DEFAULT_SEEPS_THRESHOLD_MM = (
    "{'total_precipitation_24hr':0.25, 'total_precipitation_6hr':0.1}"
)


# Command line arguments
INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
FREQUENCY = flags.DEFINE_string(
    'frequency',
    'hourly',
    (
        'Frequency of the computed climatology. "hourly": Compute the'
        ' climatology per day of year and hour of day. "daily": Compute the'
        ' climatology per day of year.'
    ),
)
HOUR_INTERVAL = flags.DEFINE_integer(
    'hour_interval',
    1,
    help='Which intervals to compute hourly climatology for.',
)
WINDOW_SIZE = flags.DEFINE_integer(
    'window_size', 61, help='Window size in days to average over.'
)
START_YEAR = flags.DEFINE_integer(
    'start_year', 1990, help='Inclusive start year of climatology'
)
END_YEAR = flags.DEFINE_integer(
    'end_year', 2020, help='Inclusive end year of climatology'
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')
WORKING_CHUNKS = flag_utils.DEFINE_chunks(
    'working_chunks',
    '',
    help=(
        'Chunk sizes overriding input chunks to use for computing climatology, '
        'e.g., "longitude=10,latitude=10".'
    ),
)
OUTPUT_CHUNKS = flag_utils.DEFINE_chunks(
    'output_chunks',
    '',
    help='Chunk sizes overriding input chunks to use for storing climatology',
)
RECHUNK_ITEMSIZE = flags.DEFINE_integer(
    'rechunk_itemsize', 4, help='Itemsize for rechunking.'
)
STATISTICS = flags.DEFINE_list(
    'statistics',
    ['mean'],
    help='Statistics to compute from "mean", "std", "seeps", "quantile".',
)
QUANTILES = flags.DEFINE_list('quantiles', [], 'List of quantiles to compute.')
METHOD = flags.DEFINE_string(
    'method',
    'explicit',
    (
        'Computation method to use. "explicit": Stack years first, apply'
        ' rolling and then compute weighted statistic over (year,'
        ' rolling_window). "fast": Compute statistic over day-of-year first and'
        ' then apply weighted smoothing. Mathematically equivalent for mean but'
        ' different for nonlinear statistics.'
    ),
)
SEEPS_DRY_THRESHOLD_MM = flags.DEFINE_string(
    'seeps_dry_threshold_mm',
    DEFAULT_SEEPS_THRESHOLD_MM,
    help=(
        'Dict defining dry threshold for SEEPS quantile computation for each'
        'precipitation variable. In mm.'
    ),
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)


class Quantile:
  """Compute quantiles."""

  def __init__(self, quantiles: list[float]):
    self.quantiles = quantiles

  def compute(
      self,
      ds: xr.Dataset,
      dim: tuple[str],
      weights: Optional[xr.Dataset] = None,
  ):
    if weights is not None:
      ds = ds.weighted(weights)  # pytype: disable=wrong-arg-types
    return ds.quantile(self.quantiles, dim=dim)


class SEEPSThreshold:
  """Compute SEEPS thresholds (heav/light) and fraction of dry grid points."""

  def __init__(self, dry_threshold_mm: float, var: str):
    self.dry_threshold_m = dry_threshold_mm / 1000.0
    self.var = var

  def compute(
      self,
      ds: xr.Dataset,
      dim: tuple[str],
      weights: Optional[xr.Dataset] = None,
  ):
    """Compute SEEPS thresholds and fraction of dry grid points."""
    ds = ds[self.var]
    is_dry = ds < self.dry_threshold_m
    dry_fraction = is_dry.mean(dim=dim)
    not_dry = ds.where(~is_dry)
    heavy_threshold = not_dry
    if weights is not None:
      heavy_threshold = heavy_threshold.weighted(
          weights
      )  # pytype: disable=wrong-arg-types
    heavy_threshold = heavy_threshold.quantile(2 / 3, dim=dim)
    out = xr.Dataset(
        {
            f'{self.var}_seeps_threshold': heavy_threshold.drop('quantile'),
            f'{self.var}_seeps_dry_fraction': dry_fraction,
        }
    )  # fmt: skip
    return out


def compute_seeps_chunk(
    obs_key: xbeam.Key,
    obs_chunk: xr.Dataset,
    *,
    frequency: str,
    window_size: int,
    clim_years: slice,
    hour_interval: int,
    seeps_threshold_mm: Optional[dict[str, float]] = None,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Compute SEEPS climatology on a chunk."""
  clim_key = obs_key.with_offsets(time=None, hour=0, dayofyear=0)
  if METHOD.value != 'explicit':
    raise NotImplementedError('SEEPS only tested for explicit.')
  (var,) = clim_key.vars
  clim_key = clim_key.replace(
      vars={f'{var}_seeps_threshold', f'{var}_seeps_dry_fraction'}
  )
  stat_fn = SEEPSThreshold(seeps_threshold_mm[var], var=var).compute
  if frequency == 'hourly':
    clim_chunk = utils.compute_hourly_stat(
        obs=obs_chunk,
        window_size=window_size,
        clim_years=clim_years,
        hour_interval=hour_interval,
        stat_fn=stat_fn,
    )
  elif frequency == 'daily':
    clim_chunk = utils.compute_daily_stat(
        obs=obs_chunk,
        window_size=window_size,
        clim_years=clim_years,
        stat_fn=stat_fn,
    )
  else:
    raise NotImplementedError(f'Frequency {frequency} not implemented.')
  return clim_key, clim_chunk


def compute_stat_chunk(
    obs_key: xbeam.Key,
    obs_chunk: xr.Dataset,
    *,
    frequency: str,
    window_size: int,
    clim_years: slice,
    statistic: Union[str, Callable[..., xr.Dataset]] = 'mean',
    hour_interval: Optional[int] = None,
    quantiles: Optional[list[float]] = None,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Compute climatology on a chunk."""
  if statistic not in ['mean', 'std', 'quantile']:
    raise NotImplementedError(f'stat {statistic} not implemented.')
  offsets = dict(dayofyear=0)
  if frequency == 'hourly':
    offsets['hour'] = 0
  clim_key = obs_key.with_offsets(time=None, **offsets)
  if statistic != 'mean':
    clim_key = clim_key.replace(
        vars={f'{var}_{statistic}' for var in clim_key.vars}
    )
    for var in obs_chunk:
      obs_chunk = obs_chunk.rename({var: f'{var}_{statistic}'})
  if statistic == 'quantile':
    statistic = Quantile(quantiles).compute
  compute_kwargs = {
      'obs': obs_chunk,
      'window_size': window_size,
      'clim_years': clim_years,
      'stat_fn': statistic,
  }

  if frequency == 'hourly' and METHOD.value == 'explicit':
    clim_chunk = utils.compute_hourly_stat(
        **compute_kwargs, hour_interval=hour_interval
    )
  elif frequency == 'hourly' and METHOD.value == 'fast':
    clim_chunk = utils.compute_hourly_stat_fast(
        **compute_kwargs, hour_interval=hour_interval
    )
  elif frequency == 'daily' and METHOD.value == 'explicit':
    clim_chunk = utils.compute_daily_stat(**compute_kwargs)
  elif frequency == 'daily' and METHOD.value == 'fast':
    clim_chunk = utils.compute_daily_stat_fast(**compute_kwargs)
  else:
    raise NotImplementedError(
        f'method {METHOD.value} for climatological frequency {frequency}'
        ' not implemented.'
    )
  return clim_key, clim_chunk


def main(argv: list[str]) -> None:
  obs, input_chunks = xbeam.open_zarr(INPUT_PATH.value)

  # Convert object-type coordinates to string.
  # Required to avoid: https://github.com/pydata/xarray/issues/3476
  for coord_name, coord in obs.coords.items():
    if coord.dtype == 'object':
      obs[coord_name] = coord.astype(str)

  # TODO(shoyer): slice obs in time using START_YEAR and END_YEAR. This would
  # require some care in order to ensure input_chunks['time'] remains valid.

  # drop static variables, for which the climatology calculation would fail
  obs = obs.drop_vars([k for k, v in obs.items() if 'time' not in v.dims])

  input_chunks_without_time = {
      k: v for k, v in input_chunks.items() if k != 'time'
  }

  if FREQUENCY.value == 'daily':
    stat_kwargs = {}
    clim_chunks = dict(dayofyear=-1)
    clim_dims = dict(dayofyear=1 + np.arange(366))
  elif FREQUENCY.value == 'hourly':
    stat_kwargs = dict(hour_interval=HOUR_INTERVAL.value)
    clim_chunks = dict(hour=-1, dayofyear=-1)
    clim_dims = dict(
        hour=np.arange(0, 24, HOUR_INTERVAL.value), dayofyear=1 + np.arange(366)
    )
  else:
    raise NotImplementedError(f'frequency {FREQUENCY.value} not implemented.')

  working_chunks = input_chunks_without_time.copy()
  working_chunks.update(WORKING_CHUNKS.value)
  if 'time' in working_chunks:
    raise ValueError('cannot include time in working chunks')
  in_working_chunks = dict(working_chunks, time=-1)
  out_working_chunks = dict(working_chunks, **clim_chunks)

  output_chunks = input_chunks_without_time.copy()
  output_chunks.update(clim_chunks)
  output_chunks.update(OUTPUT_CHUNKS.value)

  clim_template = (
      xbeam.make_template(obs).isel(time=0, drop=True).expand_dims(clim_dims)
  )

  raw_vars = list(clim_template)
  seeps_dry_threshold_mm = ast.literal_eval(SEEPS_DRY_THRESHOLD_MM.value)
  if 'seeps' in STATISTICS.value:
    for v in seeps_dry_threshold_mm.keys():
      clim_template = clim_template.assign(
          {
              f'{v}_seeps_threshold': clim_template[v],
              f'{v}_seeps_dry_fraction': clim_template[v],
          }
      )  # fmt: skip

  def _compute_seeps(kv):
    k, _ = kv
    (var,) = k.vars
    return var in seeps_dry_threshold_mm.keys()

  quantiles = [float(q) for q in QUANTILES.value]
  for stat in STATISTICS.value:
    if stat not in ['seeps', 'mean']:
      for var in raw_vars:
        if stat == 'quantile':
          if not quantiles:
            raise ValueError(
                'Cannot compute stat `quantile` without specifying --quantiles.'
            )
          quantile_dim = xr.DataArray(
              quantiles, name='quantile', dims=['quantile']
          )
          temp = clim_template[var].expand_dims(quantile=quantile_dim)
          if 'hour' in temp.dims:
            temp = temp.transpose('hour', 'quantile', ...)
        else:
          temp = clim_template[var]
        clim_template = clim_template.assign({f'{var}_{stat}': temp})
  # Mean has no suffix. Delete no suffix variables if no mean required
  if 'mean' not in STATISTICS.value:
    for var in raw_vars:
      clim_template = clim_template.drop(var)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    # Read and Rechunk
    pcoll = (
        root
        | xbeam.DatasetToChunks(
            obs,
            input_chunks,
            split_vars=True,
            num_threads=NUM_THREADS.value,
        )
        | 'RechunkIn'
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            obs.sizes,
            input_chunks,
            in_working_chunks,
            itemsize=RECHUNK_ITEMSIZE.value,
        )
    )

    # Branches to compute statistics
    pcolls = []
    for stat in STATISTICS.value:
      # SEEPS branch
      if stat == 'seeps':
        pcoll_tmp = (
            pcoll
            | beam.Filter(_compute_seeps)
            | 'seeps'
            >> beam.MapTuple(
                functools.partial(
                    compute_seeps_chunk,
                    window_size=WINDOW_SIZE.value,
                    clim_years=slice(
                        str(START_YEAR.value), str(END_YEAR.value)
                    ),
                    frequency=FREQUENCY.value,
                    seeps_threshold_mm=seeps_dry_threshold_mm,
                    **stat_kwargs,
                )
            )
        )
      else:
        # Mean and Std branches
        pcoll_tmp = pcoll | f'{stat}' >> beam.MapTuple(
            functools.partial(
                compute_stat_chunk,
                frequency=FREQUENCY.value,
                window_size=WINDOW_SIZE.value,
                clim_years=slice(str(START_YEAR.value), str(END_YEAR.value)),
                statistic=stat,
                quantiles=quantiles,
                **stat_kwargs,
            )
        )
      pcolls.append(pcoll_tmp)

    # Rechunk and write output
    _ = (
        pcolls
        | beam.Flatten()
        | 'RechunkOut'
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            clim_template.sizes,
            out_working_chunks,
            output_chunks,
            itemsize=RECHUNK_ITEMSIZE.value,
        )
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template=clim_template,
            zarr_chunks=output_chunks,
            num_threads=NUM_THREADS.value,
        )
    )


if __name__ == '__main__':
  app.run(main)
