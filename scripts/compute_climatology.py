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
  python scripts/compute_climatology.py \
    --input_path='gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2' \
    --output_path='gs://$BUCKET/datasets/ear5-hourly-climatology/$USER/1990-2019_6h_1440x721.zarr' \
    --by_hour=False
  ```
"""

import typing as t

from absl import app
from absl import flags
import xarray as xr

from weatherbench2 import utils as wb2_utils

# Command line arguments
INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
BY_HOUR = flags.DEFINE_bool('by_hour', True, help='Compute by hour of day')
HOUR_INTERVAL = flags.DEFINE_integer(
    'hour_interval',
    1,
    help='Which intervals to compute hourly climatology for.',
)
WINDOW_SIZE = flags.DEFINE_integer('window_size', 61, help='Window size')
START_YEAR = flags.DEFINE_integer('start_year', 1990, help='Clim start year')
END_YEAR = flags.DEFINE_integer('end_year', 2020, help='Clim end year (incl.)')


def main(_: t.Sequence[str]) -> None:
  obs = xr.open_zarr(INPUT_PATH.value)
  if BY_HOUR.value:
    print('Compute hourly climatology.')
    clim = wb2_utils.compute_hourly_stat(
        obs=obs,
        window_size=WINDOW_SIZE.value,
        clim_years=slice(str(START_YEAR.value), str(END_YEAR.value)),
        hour_interval=HOUR_INTERVAL.value,
    )
  else:
    print('Compute daily climatology.')
    clim = wb2_utils.compute_daily_stat(
        obs=obs,
        window_size=WINDOW_SIZE.value,
        clim_years=slice(str(START_YEAR.value), str(END_YEAR.value)),
    )
  # Save
  print('Saving output file')
  print(clim)
  clim.to_zarr(OUTPUT_PATH.value)


if __name__ == '__main__':
  app.run(main)
