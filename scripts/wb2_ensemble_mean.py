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
r"""Ensemble mean (over REALIZATION dimension) of a forecast dataset.

Example Usage:
  ```
  export BUCKET=my-bucket
  export PROJECT=my-project
  export REGION=us-central1

  python scripts/wb2_ensemble_mean.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/era5/$USER/1959-2022-ensemble-means.zarr \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --region=$REGION \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --job_name=compute-ensemble-mean-$USER
  ```
"""
from absl import app
from absl import flags
import apache_beam as beam
import xarray_beam as xbeam

REALIZATION = 'realization'

INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')
REALIZATION_NAME = flags.DEFINE_string(
    'realization_name',
    REALIZATION,
    'Name of realization/member/number dimension.',
)


# pylint: disable=expression-not-assigned


def main(argv: list[str]):
  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH.value)
  template = xbeam.make_template(
      source_dataset.isel({REALIZATION_NAME.value: 0}, drop=True)
  )
  target_chunks = {
      k: v for k, v in source_chunks.items() if k != REALIZATION_NAME.value
  }

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    (
        root
        | xbeam.DatasetToChunks(source_dataset, source_chunks, split_vars=True)
        | xbeam.Mean(REALIZATION_NAME.value)
        | xbeam.ChunksToZarr(OUTPUT_PATH.value, template, target_chunks)
    )


if __name__ == '__main__':
  app.run(main)
