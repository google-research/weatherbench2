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
import logging
import re
import typing as t

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

SEL = flag_utils.DEFINE_dim_value_pairs(
    'sel',
    '',
    help=(
        'Selection criteria, to pass to xarray.Dataset.sel. Passed as key=value'
        ' pairs, with key = VARNAME_{start,stop,step,list}. If key ends with'
        ' start, stop, or step, the values are used in a slice as'
        ' slice(cast(start), cast(stop), int(step)). start/stop/step default'
        ' to None. If key ends with "list", the value should be a list of "+"'
        ' delimited ints/floats/strings. Here `cast` tries to cast to numeric,'
        'but falls back to string. '
    ),
)

SEL_STRINGS = flag_utils.DEFINE_dim_value_pairs(
    'sel_strings',
    '',
    help=(
        'Selection criteria, to pass to xarray.Dataset.sel. Passed as'
        ' key=value pairs, with key = VARNAME_{start,stop,step,list}. '
        'If key ends with start, stop, or step, the values are used in a slice '
        'as slice(str(start), str(stop), int(step)). start/stop/step default to'
        ' None. If key ends with "list", the value should be '
        'a list of "+" delimited ints/floats/strings. '
        'Useful, since years should be sliced using strings like "2000". '
    ),
)

ISEL = flag_utils.DEFINE_dim_value_pairs(
    'isel',
    '',
    help=(
        'Selection criteria, to pass to xarray.Dataset.isel. Passed as'
        ' key=value pairs, with key = VARNAME_{start,stop,step,list}. '
        'If key ends with start, stop, or step, the value should be integers '
        '(defaulting to None). If key ends with "list", the value should be '
        'a list of "+" delimited ints.'
    ),
)

DROP_SEL = flag_utils.DEFINE_dim_value_pairs(
    'drop_sel',
    '',
    help=(
        'Selection criteria, to pass to xarray.Dataset.drop_sel. Passed as'
        ' key=value pairs, with key = VARNAME_{start,stop,step,list}. '
        'If key ends with start, stop, or step, the values are used in a slice '
        ' slice(cast(start), cast(stop), int(step)). start/stop/step default'
        ' to None. If key ends with "list", the value should be a list of "+"'
        ' delimited ints/floats/strings. Here `cast` tries to cast to numeric,'
        'but falls back to string. '
    ),
)

DROP_SEL_STRINGS = flag_utils.DEFINE_dim_value_pairs(
    'drop_sel_strings',
    '',
    help=(
        'Selection criteria, to pass to xarray.Dataset.drop_sel. Passed as'
        ' key=value pairs, with key = VARNAME_{start,stop,step,list}. '
        'If key ends with start, stop, or step, the values are used in a slice '
        'as slice(str(start), str(stop), int(step)). start/stop/step default to'
        ' None. If key ends with "list", the value should be '
        'a list of "+" delimited ints/floats/strings.'
        'Useful, since years should be sliced using strings like "2000". '
    ),
)

DROP_ISEL = flag_utils.DEFINE_dim_value_pairs(
    'drop_isel',
    '',
    help=(
        'Selection criteria, to pass to xarray.Dataset.drop_isel. Passed as'
        ' key=value pairs, with key = VARNAME_{start,stop,step,list}. '
        'If key ends with start, stop, or step, the value should be integers '
        '(defaulting to None). If key ends with "list", the value should be '
        'a list of "+" delimited ints.'
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
MAKE_DIMS_INCREASING = flags.DEFINE_list(
    'make_dims_increasing',
    [],
    help='Dimensions to make increasing, reversing order if needed.',
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)

# pylint: disable=logging-fstring-interpolation


def _maybe_make_some_dims_increasing(ds: xr.Dataset) -> xr.Dataset:
  """Specified monotonic dims are made increasing, raise if non-monotonic."""
  for dim in MAKE_DIMS_INCREASING.value:
    x = ds[dim].data
    is_increasing = np.diff(x) > 0
    if np.all(is_increasing):
      pass  # Already increasing, great!
    elif np.all(~is_increasing):
      ds = ds.sel({dim: x[::-1]})
    else:
      raise ValueError(f'Cannot make non-monotonic dimension {dim} increasing')
  return ds


def _get_selections(
    flag_values: dict[str, flag_utils.DimValueType],
    force_string: bool,
) -> list[dict[str, t.Union[str, int, list[int], slice]]]:
  """Gets parts used to select based on flags."""

  def maybe_tostr(v):
    # This function explicitly forces a string. By default, the flag parser
    # 1. Tries casting to int..stuff like '2.2' will fail
    # 2. Tries casting to float...stuff like 'cats' will fail
    # 3. Returns a string.
    if force_string:
      return str(v)
    return v

  list_selectors = {}
  value_selectors = {}
  for k, v in flag_values.items():
    # Validate and parse.
    match = re.search(r'^(.*)_(start|stop|step|list)$', k)
    if not match:
      raise ValueError(f'Flag {k} did not end in _(start|stop|step|list)')
    dim, placement = match.groups()
    # Handle list types
    if placement == 'list':
      # Convert to string to allow .split('+') even if v was a single item list
      # that cong converted to a float or int by the flag parser.
      v = str(v)
      if '++' in v:
        raise ValueError(f'Found ambiguous "++" in {dim=} flag value {v}')
      list_selectors[dim] = [
          maybe_tostr(flag_utils.get_dim_value(v_i)) for v_i in v.split('+')
      ]
    else:  # Else handle non-list types
      v = flag_utils.get_dim_value(v)
      if dim not in value_selectors:
        value_selectors[dim] = [None, None, None]
      if placement == 'start':
        value_selectors[dim][0] = maybe_tostr(v)
      elif placement == 'stop':
        value_selectors[dim][1] = maybe_tostr(v)
      else:  # Else 'step'
        # In Xarray, step must be an int.
        # https://github.com/pydata/xarray/issues/5228
        value_selectors[dim][2] = int(v)

  selections = []
  for dim, selector in list_selectors.items():
    selections.append({dim: selector})
  for dim, selector in value_selectors.items():
    selections.append(
        {dim: slice(*selector) if isinstance(selector, list) else selector}
    )
  logging.info(f'Deduced selections {selections=} from {flag_values=}')
  return selections  # pytype: disable=bad-return-type


def main(argv: abc.Sequence[str]) -> None:

  ds, input_chunks = xbeam.open_zarr(INPUT_PATH.value)

  ds = _maybe_make_some_dims_increasing(ds)

  if DROP_VARIABLES.value:
    ds = ds.drop_vars(DROP_VARIABLES.value)
  elif KEEP_VARIABLES.value:
    ds = ds[KEEP_VARIABLES.value]
  input_chunks = {k: v for k, v in input_chunks.items() if k in ds.dims}

  for selection in _get_selections(ISEL.value, force_string=False):
    ds = ds.isel(selection)
  for selection in _get_selections(SEL.value, force_string=False):
    ds = ds.sel(selection)
  for selection in _get_selections(SEL_STRINGS.value, force_string=True):
    ds = ds.sel(selection)
  for selection in _get_selections(DROP_ISEL.value, force_string=False):
    ds = ds.drop_isel(selection)
  for selection in _get_selections(DROP_SEL.value, force_string=False):
    ds = ds.drop_sel(selection)
  for selection in _get_selections(DROP_SEL_STRINGS.value, force_string=True):
    ds = ds.drop_sel(selection)

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
