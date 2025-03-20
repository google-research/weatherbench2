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
r"""Run WeatherBench 2 regridding pipeline.

Only rectalinear grids (one dimensional lat/lon coordinates) on the input Zarr
file are supported, but irregular spacing is OK.

Example Usage:
  ```
  export BUCKET=my-bucket
  export PROJECT=my-project
  export REGION=us-central1

  python scripts/regrid.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
    --output_path=gs://$BUCKET/datasets/era5/$USER/1959-2022-6h-64x33.zarr \
    --output_chunks="time=100" \
    --longitude_nodes=64 \
    --latitude_nodes=33 \
    --latitude_spacing=equiangular_with_poles \
    --regridding_method=conservative \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --region=$REGION \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./scripts/dataflow-requirements.txt \
    --job_name=regrid-$USER
  ```
"""
from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from weatherbench2 import flag_utils
from weatherbench2 import regridding
import xarray_beam

INPUT_PATH = flags.DEFINE_string('input_path', None, help='zarr inputs')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='zarr outputs')
OUTPUT_CHUNKS = flag_utils.DEFINE_chunks(
    'output_chunks', '', help='desired chunking of output zarr'
)
LATITUDE_NODES = flags.DEFINE_integer(
    'latitude_nodes', None, help='number of desired latitude nodes'
)
LONGITUDE_NODES = flags.DEFINE_integer(
    'longitude_nodes', None, help='number of desired longitude nodes'
)
LATITUDE_SPACING = flags.DEFINE_enum_class(
    'latitude_spacing',
    regridding.LatitudeSpacing.EQUIANGULAR_WITH_POLES,
    regridding.LatitudeSpacing,
    help='Desired latitude spacing.',
)
LONGITUDE_SCHEME = flags.DEFINE_enum_class(
    'longitude_scheme',
    regridding.LongitudeScheme.START_AT_ZERO,
    regridding.LongitudeScheme,
    help=(
        'What values the output longitude dimension will have. With Δ = 360 /'
        ' LONGITUDE_NODES, "START_AT_ZERO" means longitude=[0, ..., 360 - Δ].'
        ' "CENTER_AT_ZERO" means longitude=[-180 + Δ/2, ..., 180 - Δ/2]'
    ),
)
REGRIDDING_METHOD = flags.DEFINE_enum(
    'regridding_method',
    'conservative',
    ['nearest', 'bilinear', 'conservative'],
    help='regridding method',
)
LATITUDE_NAME = flags.DEFINE_string(
    'latitude_name', 'latitude', help='Name of latitude dimension in dataset'
)
LONGITUDE_NAME = flags.DEFINE_string(
    'longitude_name', 'longitude', help='Name of longitude dimension in dataset'
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


def main(argv):
  source_ds, input_chunks = xarray_beam.open_zarr(INPUT_PATH.value)

  # Rename latitude/longitude names
  renames = {
      LONGITUDE_NAME.value: 'longitude',
      LATITUDE_NAME.value: 'latitude',
  }
  source_ds = source_ds.rename(renames)
  input_chunks = {renames.get(k, k): v for k, v in input_chunks.items()}

  # Lat/lon must be single chunk for regridding.
  input_chunks['longitude'] = -1
  input_chunks['latitude'] = -1

  old_lon = source_ds.coords['longitude'].data
  old_lat = source_ds.coords['latitude'].data

  new_lon = regridding.longitude_values(
      LONGITUDE_SCHEME.value,
      LONGITUDE_NODES.value,
  )
  new_lat = regridding.latitude_values(
      LATITUDE_SPACING.value,
      LATITUDE_NODES.value,
  )

  regridder_cls = {
      'nearest': regridding.NearestRegridder,
      'bilinear': regridding.BilinearRegridder,
      'conservative': regridding.ConservativeRegridder,
  }[REGRIDDING_METHOD.value]

  source_grid = regridding.Grid.from_degrees(lon=old_lon, lat=np.sort(old_lat))
  target_grid = regridding.Grid.from_degrees(lon=new_lon, lat=new_lat)
  regridder = regridder_cls(source_grid, target_grid)

  template = (
      xarray_beam.make_template(source_ds)
      .isel(longitude=0, latitude=0, drop=True)
      .expand_dims(longitude=new_lon, latitude=new_lat)
      .transpose(..., 'longitude', 'latitude')
  )
  itemsize = max(var.dtype.itemsize for var in template.values())

  output_chunks = input_chunks.copy()
  output_chunks.update(OUTPUT_CHUNKS.value)
  print('OUTPUT_CHUNKS:', repr(output_chunks))

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xarray_beam.DatasetToChunks(
            source_ds,
            input_chunks,
            split_vars=True,
            num_threads=NUM_THREADS.value,
        )
        | 'Regrid'
        >> beam.MapTuple(lambda k, v: (k, regridder.regrid_dataset(v)))
        | xarray_beam.Rechunk(
            template.sizes,
            input_chunks,
            output_chunks,
            itemsize=itemsize,
        )
        | xarray_beam.ChunksToZarr(
            OUTPUT_PATH.value,
            template,
            output_chunks,
            num_threads=NUM_THREADS.value,
        )
    )


if __name__ == '__main__':
  app.run(main)
