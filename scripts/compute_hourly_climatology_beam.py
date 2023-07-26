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
  export MODE=mean
  export START_YEAR=1959
  export END_YEAR=2015
  export BUCKET=my-bucket
  export PROJECT=my-project
  export REGION=us-central1

  python scripts/compute_hourly_climatology_beam.py \
    --mode=$MODE \
    --start_year=$START_YEAR \
    --end_year=$END_YEAR \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/ear5-hourly-climatology/$USER/${MODE}/${START_YEAR}_to_${END_YEAR}_6h_64x32_equiangular_conservative.zarr \
    --working_chunks="level=1,longitude=4,latitude=4" \
    --output_chunks="level=1,hour=3" \
    --beam_runner=DataflowRunner \
    -- \
    --project $PROJECT \
    --region $REGION \
    --temp_location gs://$BUCKET/tmp/ \
    --job_name compute-hourly-climatology-$USER
  ```
"""
import functools

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from weatherbench2 import flag_utils
from weatherbench2 import utils
import xarray as xr
import xarray_beam as xbeam


# Command line arguments
# TODO(shoyer): add an option for daily climatology
INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
MODE = flags.DEFINE_string('mode', 'mean', help='Climatological mean or std')
HOUR_INTERVAL = flags.DEFINE_integer(
    'hour_interval',
    1,
    help='Which intervals to compute hourly climatology for.',
)
WINDOW_SIZE = flags.DEFINE_integer('window_size', 61, help='Window size')
START_YEAR = flags.DEFINE_integer('start_year', 1990, help='Clim start year')
END_YEAR = flags.DEFINE_integer('end_year', 2020, help='Clim end year (incl.)')
BEAM_RUNNER = flags.DEFINE_string(
    'beam_runner', None, help='beam.runners.Runner'
)
WORKING_CHUNKS = flag_utils.DEFINE_chunks(
    'working_chunks',
    '',
    help=(
        'chunk sizes overriding input chunks to use for computing climatology, '
        'e.g., "longitude=10,latitude=10".'
    ),
)
OUTPUT_CHUNKS = flag_utils.DEFINE_chunks(
    'output_chunks',
    '',
    help='chunk sizes overriding input chunks to use for storing climatology',
)


def compute_hourly_climatology_mean_chunk(
    obs_key: xbeam.Key,
    obs_chunk: xr.Dataset,
    *,
    window_size: int,
    clim_years: slice,
    hour_interval: int,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Compute hourly climatology on a chunk."""
  clim_key = obs_key.with_offsets(time=None, hour=0, dayofyear=0)
  clim_chunk = utils.compute_hourly_climatology_mean_fast(
      obs=obs_chunk,
      window_size=window_size,
      clim_years=clim_years,
      hour_interval=hour_interval,
  )
  return clim_key, clim_chunk


def compute_hourly_climatology_std_chunk(
    obs_key: xbeam.Key,
    obs_chunk: xr.Dataset,
    *,
    window_size: int,
    clim_years: slice,
    hour_interval: int,
) -> tuple[xbeam.Key, xr.Dataset]:
  """Compute hourly climatology on a chunk."""
  clim_key = obs_key.with_offsets(time=None, hour=0, dayofyear=0)
  clim_chunk = utils.compute_hourly_climatology_std_fast(
      obs=obs_chunk,
      window_size=window_size,
      clim_years=clim_years,
      hour_interval=hour_interval,
  )
  return clim_key, clim_chunk


def main(argv: list[str]) -> None:
  obs, input_chunks = xbeam.open_zarr(INPUT_PATH.value)
  # TODO(shoyer): slice obs in time using START_YEAR and END_YEAR. This would
  # require some care in order to ensure input_chunks['time'] remains valid.

  # drop static variables, for which the climatology calculation would fail
  obs = obs.drop_vars([k for k, v in obs.items() if 'time' not in v.dims])

  input_chunks_without_time = {
      k: v for k, v in input_chunks.items() if k != 'time'
  }

  working_chunks = input_chunks_without_time.copy()
  working_chunks.update(flag_utils.parse_chunks(WORKING_CHUNKS.value))
  if 'time' in working_chunks:
    raise ValueError('cannot include time in working chunks')
  in_working_chunks = dict(working_chunks, time=-1)
  out_working_chunks = dict(working_chunks, hour=-1, dayofyear=-1)

  output_chunks = input_chunks_without_time.copy()
  output_chunks.update(hour=-1, dayofyear=-1)
  output_chunks.update(flag_utils.parse_chunks(OUTPUT_CHUNKS.value))

  clim_template = (
      xbeam.make_template(obs)
      .isel(time=0, drop=True)
      .expand_dims(
          hour=np.arange(0, 24, HOUR_INTERVAL.value),
          dayofyear=1 + np.arange(366),
      )
  )
  if MODE.value == 'mean':
    compute_hourly_climatology_chunk = compute_hourly_climatology_mean_chunk
  elif MODE.value == 'std':
    compute_hourly_climatology_chunk = compute_hourly_climatology_std_chunk
  else:
    raise ValueError(f'Wrong climatological mode value: {MODE.value}')

  with beam.Pipeline(runner=BEAM_RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(
            obs, input_chunks, split_vars=True, num_threads=16
        )
        | 'RechunkIn'
        >> xbeam.Rechunk(obs.sizes, input_chunks, in_working_chunks, itemsize=4)
        | beam.MapTuple(
            functools.partial(
                compute_hourly_climatology_chunk,
                window_size=WINDOW_SIZE.value,
                clim_years=slice(str(START_YEAR.value), str(END_YEAR.value)),
                hour_interval=HOUR_INTERVAL.value,
            )
        )
        | 'RechunkOut'
        >> xbeam.Rechunk(
            clim_template.sizes, out_working_chunks, output_chunks, itemsize=4
        )
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template=clim_template,
            zarr_chunks=output_chunks,
            num_threads=16,
        )
    )


if __name__ == '__main__':
  app.run(main)
