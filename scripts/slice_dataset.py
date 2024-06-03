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
r"""CLI to slice a Zarr file containing a xarray.Dataset.


Example Usage:

  ```
  export BUCKET=my-bucket
  export PROJECT=my-project
  export REGION=us-central1

  python scripts/resample_in_time.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/era5/$USER/2020-2021-weekly-average-temperature.zarr \
    --runner=DataflowRunner \
    --sel="prediction_timedelta_stop=15 days,latitude_start=-33.33,latitude_stop=33.33" \
    --isel="longitude_start=0,longitude_stop=180,longitude_step=40" \
    --keep_variables=geopotential,temperature \
    -- \
    --project=$PROJECT \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./scripts/dataflow-requirements.txt \
    --job_name=slice-dataset-$USER
  ```
"""

from collections import abc
import re

from absl import app
from absl import flags
import apache_beam as beam
from weatherbench2 import flag_utils
import xarray_beam as xbeam


# Command line arguments
INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path.')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path.')

SEL = flag_utils.DEFINE_dim_value_pairs(
    'sel',
    '',
    help=(
        'Selection criteria, to pass to xarray.Dataset.sel. Passed as key=value'
        ' pairs, with key = VARNAME_{start,stop,step}'
    ),
)

ISEL = flag_utils.DEFINE_dim_integer_pairs(
    'isel',
    '',
    help=(
        'Selection criteria, to pass to xarray.Dataset.isel. Passed as'
        ' key=value pairs, with key = VARNAME_{start,stop,step}'
    ),
)

DROP_VARIABLES = flags.DEFINE_list(
    'drop_variables',
    None,
    help=(
        'Comma delimited list of variables to drop. If empty, drop no'
        ' variables. List may include data variables or coords.'
    ),
)

KEEP_VARIABLES = flags.DEFINE_list(
    'keep_variables',
    None,
    help=(
        'Comma delimited list of data variables to keep. If empty, use'
        ' --drop_variables to determine which variables to keep'
    ),
)

OUTPUT_CHUNKS = flag_utils.DEFINE_chunks(
    'output_chunks', '', help='Chunk sizes overriding input chunks.'
)

RUNNER = flags.DEFINE_string(
    'runner', None, help='Beam runner. Use DirectRunner for local execution.'
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)


def _get_selections(
    isel_flag_value: dict[str, int],
    sel_flag_value: dict[str, flag_utils.DimValueType],
) -> tuple[dict[str, slice], dict[str, slice]]:
  """Gets dictionaries for `xr.isel` and `xr.sel`."""
  isel_parts = {}
  sel_parts = {}
  for parts_dict, flag_value in [
      (isel_parts, isel_flag_value),
      (sel_parts, sel_flag_value),
  ]:
    for k, v in flag_value.items():
      match = re.search(r'^(.*)_(start|stop|step)$', k)
      if not match:
        raise ValueError(f'Flag {k} did not end in _(start|stop|step)')
      dim, placement = match.groups()
      if dim not in parts_dict:
        parts_dict[dim] = [None, None, None]
      if placement == 'start':
        parts_dict[dim][0] = v
      elif placement == 'stop':
        parts_dict[dim][1] = v
      else:
        parts_dict[dim][2] = v

  overlap = set(isel_parts).intersection(sel_parts)
  if overlap:
    raise ValueError(
        f'--isel {isel_flag_value} and --sel {sel_flag_value} overlapped for'
        f' variables {overlap}'
    )
  isel = {k: slice(*v) for k, v in isel_parts.items()}
  sel = {k: slice(*v) for k, v in sel_parts.items()}
  return isel, sel


def main(argv: abc.Sequence[str]) -> None:

  ds, input_chunks = xbeam.open_zarr(INPUT_PATH.value)

  if DROP_VARIABLES.value:
    ds = ds.drop_vars(DROP_VARIABLES.value)
  elif KEEP_VARIABLES.value:
    ds = ds[KEEP_VARIABLES.value]

  isel, sel = _get_selections(ISEL.value, SEL.value)
  if isel:
    ds = ds.isel(isel)
  if sel:
    ds = ds.sel(sel)

  template = xbeam.make_template(ds)

  output_chunks = {k: v for k, v in input_chunks.items()}  # Copy
  for k in output_chunks:
    if k in OUTPUT_CHUNKS.value:
      output_chunks[k] = OUTPUT_CHUNKS.value[k]
    else:
      output_chunks[k] = min(output_chunks[k], ds.sizes[k])

  itemsize = max(var.dtype.itemsize for var in template.values())

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    # Read, rechunk, write
    unused_pcoll = (
        root
        | xbeam.DatasetToChunks(
            ds, input_chunks, split_vars=True, num_threads=NUM_THREADS.value
        )
        | xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            ds.sizes,
            input_chunks,
            output_chunks,
            itemsize=itemsize,
        )
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template=template,
            zarr_chunks=output_chunks,
            num_threads=NUM_THREADS.value,
        )
    )


if __name__ == '__main__':
  flags.mark_flags_as_required(['input_path', 'output_path'])
  flags.mark_flags_as_mutual_exclusive(['keep_variables', 'drop_variables'])
  app.run(main)
