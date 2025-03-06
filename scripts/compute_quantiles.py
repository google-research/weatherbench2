r"""Computes quantiles in a Dataset.

Example of getting quantiles of temperature by latitude, longitude, and level.
So we reduce over all other dims (in this case, "time").

  ```
  export BUCKET=my-bucket
  export PROJECT=my-project

  python scripts/compute_quantiles.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/era5/$USER/temperature-quantiles.zarr \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --dim=time \
    --variables=temperature \
    --time_start="2000-01-01" \
    --time_stop="2000-12-31" \
    --working_chunks="latitude=4,longitude=4,level=1" \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./scripts/dataflow-requirements.txt \
    --job_name=compute-vertical-profile-$USER
  ```
"""

import typing as t

from absl import app
from absl import flags
import apache_beam as beam
from weatherbench2 import flag_utils
import xarray as xr
import xarray_beam as xbeam


INPUT_PATH = flags.DEFINE_string('input_path', None, help='zarr input path')
OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    help='Path to output zarr',
)
QUANTILES = flags.DEFINE_list(
    'quantiles',
    None,
    help='Comma delimited list of quantiles, 0 <= q <= 1.',
)
DIM = flags.DEFINE_list(
    name='dim',
    default=[],
    help='Comma delimited list of dimensions to reduce over.',
)
NAME_SUFFIX = flags.DEFINE_string(
    'name_suffix',
    '',
    help=(
        'Suffix to add to variable names. If you add "_quantile", then the'
        ' output can be used as climatology for Weatherbench 2 thresholded'
        ' scores.'
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

LEVELS = flags.DEFINE_list(
    'levels',
    None,
    help=(
        'Comma delimited list of pressure levels to compute spectra on. If'
        ' empty, compute on all levels of --input_path'
    ),
)
TIME_DIM = flags.DEFINE_string(
    'time_dim',
    'time',
    help=(
        'Name for the time dimension to slice data on, if TIME_START or'
        ' TIME_STOP is provided.'
    ),
)
TIME_START = flags.DEFINE_string(
    'time_start',
    '2020-01-01',
    help='ISO 8601 timestamp (inclusive) at which to start evaluation',
)
TIME_STOP = flags.DEFINE_string(
    'time_stop',
    '2020-12-31',
    help='ISO 8601 timestamp (inclusive) at which to stop evaluation',
)
VARIABLES = flags.DEFINE_list(
    'variables',
    None,
    help=(
        'Comma delimited list of data variables to include in output.  '
        'If empty, compute on all data_vars of --input_path'
    ),
)

WORKING_CHUNKS = flag_utils.DEFINE_chunks(
    'working_chunks',
    '',
    help=(
        'If provided, rechunk to this when reducing. E.g. "time=1,timedelta=5".'
        '  Keys must be a subset of dimensions not being reduced over'
        ' (preserved dims). The in process memory size is the working chunk'
        ' size, and dims not preserved cannot be  working chunks. So set this'
        ' carefully. For that reason, the default value for all preserved dims'
        ' is 1.'
    ),
)
OUTPUT_CHUNKS = flag_utils.DEFINE_chunks(
    'output_chunks',
    '',
    help=(
        'If provided, rechunk output to this after reducing. E.g.'
        ' "time=1,timedelta=1". By default, re-use the input dataset chunk'
        ' sizes.'
    ),
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


def _get_preserve_dims(ds: xr.Dataset) -> set[t.Hashable]:
  """Dims in ds that are preserved."""
  return set([d for d in ds.dims if d not in DIM.value])


def _impose_data_selection(
    ds: xr.Dataset,
    chunks: t.Mapping[str, int],
) -> tuple[xr.Dataset, dict[str, int]]:
  """Select subset of data and chunks as requested by FLAGS."""
  if VARIABLES.value is not None:
    ds = ds[VARIABLES.value]
  selection = {
      TIME_DIM.value: slice(TIME_START.value, TIME_STOP.value),
  }
  if LEVELS.value:
    selection['level'] = [float(l) for l in LEVELS.value]
  ds = ds.sel({k: v for k, v in selection.items() if k in ds.dims})
  chunks = {k: v for k, v in chunks.items() if k in ds.dims}
  return ds, chunks


def evaluate_chunk(
    key: xbeam.Key, chunk: xr.Dataset
) -> tuple[xbeam.Key, xr.Dataset]:
  new_chunk = _evaluate_chunk_core(chunk)
  new_key = key.with_offsets(
      **{k: None for k in key.offsets if k not in new_chunk.dims}
  )
  return new_key, new_chunk


def _evaluate_chunk_core(chunk: xr.Dataset) -> xr.Dataset:
  """Implementation of evaluate_chunk that doesn't use a key."""
  preserve_dims = _get_preserve_dims(chunk)
  if not preserve_dims.issubset(set(chunk.dims)):
    raise ValueError(
        f'User specified {DIM.value=}, which results in preserved dims'
        f' {preserve_dims} , not being a subset of {set(chunk.dims)=}'
    )

  quantiles = [float(q) for q in QUANTILES.value]
  if any(q < 0 or q > 1 for q in quantiles):
    raise ValueError(
        f'Expected all quantiles to be in [0, 1]. Found {quantiles=}'
    )
  values = chunk.quantile(quantiles, dim=DIM.value, skipna=SKIPNA.value)
  return values.rename_vars({v: v + NAME_SUFFIX.value for v in values})


def main(argv: list[str]) -> None:
  source_ds, source_chunks = _impose_data_selection(
      *xbeam.open_zarr(INPUT_PATH.value)
  )

  preserve_dims = _get_preserve_dims(source_ds)

  if not set(WORKING_CHUNKS.value).issubset(preserve_dims):
    raise flags.IllegalFlagValueError(
        f'{WORKING_CHUNKS.value.keys()=} was not a subset of preserved dims'
        f' {preserve_dims}'
    )

  working_chunks = WORKING_CHUNKS.value.copy()
  for k in set(source_chunks).difference(working_chunks):
    if k in preserve_dims:
      working_chunks[k] = 1
    else:
      working_chunks[k] = -1
  output_chunks = {
      k: OUTPUT_CHUNKS.value.get(k, source_chunks[k])
      for k in preserve_dims.intersection(source_chunks)
  }
  output_chunks.setdefault('quantile', -1)

  # Make the template by evaluation (which reduces to produce a dataset with
  # correct output dims).
  template = _evaluate_chunk_core(xbeam.make_template(source_ds))

  output_chunks = {
      # The template may be smaller than output_chunks.
      k: min(output_chunks[k], template.sizes[k])
      for k in output_chunks
  }

  itemsize = max(var.dtype.itemsize for var in template.values())

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(
            source_ds,
            source_chunks,
            split_vars=True,
            num_threads=NUM_THREADS.value,
        )
        # TODO(langmore) Write a xarray_beam quantile reducer to avoid this
        # rechunking.
        | 'RechunkToWorkingChunks'
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            source_ds.sizes,
            source_chunks,
            working_chunks,
            itemsize=itemsize,
        )
        | 'Compute_nan_fraction' >> beam.MapTuple(evaluate_chunk)
        | 'RechunkToOutputChunks'
        >> xbeam.Rechunk(  # pytype: disable=wrong-arg-types
            template.sizes,
            # Want to inject -1 for new dims
            {k: working_chunks.get(k, -1) for k in output_chunks},
            output_chunks,
            itemsize=itemsize,
        )
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template,
            output_chunks,
            num_threads=NUM_THREADS.value,
        )
    )


if __name__ == '__main__':
  flags.mark_flags_as_required(
      ['input_path', 'output_path', 'dim', 'quantiles']
  )
  app.run(main)
