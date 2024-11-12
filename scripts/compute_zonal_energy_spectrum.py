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
# pyformat: mode=pyink
r"""Compute ZonalEnergySpectrum derived variable.

Example Usage:
  ```
  export BUCKET=my-bucket
  export PROJECT=my-project
  export REGION=us-central1

  python scripts/compute_zonal_energy_spectrum.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/era5/$USER/1959-2022-6h-64x32_equiangular_with_poles_conservative_with_zonal_energy_spectrum.zarr \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --region=$REGION \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./dataflow-requirements.txt \
    --job_name=compute-zonal-energy-spectrum-$USER
  ```
"""
import typing as t

from absl import app
from absl import flags
import apache_beam as beam
from weatherbench2.derived_variables import ZonalEnergySpectrum
import xarray as xr
import xarray_beam as xbeam

_DEFAULT_BASE_VARIABLES = [
    'geopotential',
    'specific_humidity',
    'temperature',
]
_DEFAULT_LEVELS = ['500', '700', '850']
_DEFAULT_AVERAGING_DIMS = ['time']


INPUT_PATH = flags.DEFINE_string('input_path', None, help='Input Zarr path')
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path')
BASE_VARIABLES = flags.DEFINE_list(
    'base_variables',
    _DEFAULT_BASE_VARIABLES,
    help=(
        'Comma delimited list of variables in --input_path. Each variable VAR '
        'results in a VAR_zonal_power_spectrum entry in --output_path.'
    ),
)
TIME_DIM = flags.DEFINE_string(
    'time_dim', 'time', help='Name for the time dimension to slice data on.'
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
LEVELS = flags.DEFINE_list(
    'levels',
    _DEFAULT_LEVELS,
    help=(
        'Comma delimited list of pressure levels to compute spectra on. If'
        ' empty, compute on all levels of --input_path. Ignored if "level" is'
        ' not a dimension.'
    ),
)
AVERAGING_DIMS = flags.DEFINE_list(
    'averaging_dims',
    _DEFAULT_AVERAGING_DIMS,
    help=(
        'Comma delimited list of variables to average over. If empty, do not'
        ' average.'
    ),
)
FANOUT = flags.DEFINE_integer(
    'fanout',
    None,
    help='Beam CombineFn fanout. Might be required for large dataset.',
)
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)

RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


def _make_derived_variables_ds(
    source: xr.Dataset,
    derived_variables: t.Sequence[ZonalEnergySpectrum],
) -> xr.Dataset:
  """Dataset with power spectrum for BASE_VARIABLES before averaging."""
  arrays = []
  for dv in derived_variables:
    arrays.append({dv.variable_name: dv.compute(source[dv.base_variables])})
  return xr.merge(arrays).transpose(
      *_output_dims(source, include_averaging_dims=True)
  )


def _make_template(
    source: xr.Dataset,
    derived_variables: t.Sequence[ZonalEnergySpectrum],
) -> xr.Dataset:
  """Makes a template with shape equivalent to making derived ds on source."""
  # Shorten source along these dims.
  # Exclude some dims from shortening since they are necessary for proper
  # functioning of _make_derived_variables_ds.
  shortn_dims = {
      d: source[d]
      for d in set(source.dims).difference(['latitude', 'longitude', 'level'])
  }
  small_output = _make_derived_variables_ds(
      source.isel({k: 0 for k in shortn_dims}, drop=True),
      derived_variables,
  )
  # Not including AVERAGING_DIMS in the expansion (and the drop=True) above
  # effectively removes them from this template.
  return (
      xbeam.make_template(small_output)
      .expand_dims(
          {
              d: length
              for d, length in shortn_dims.items()
              if d not in AVERAGING_DIMS.value
          }
      )
      .transpose(*_output_dims(source, include_averaging_dims=False))
  )


def _output_dims(
    source: xr.Dataset,
    include_averaging_dims: bool,
) -> list[str]:
  """Dimensions in the output, in canonical order."""
  dims = []
  for d in source.dims:
    assert isinstance(d, str), f'{type(d)=} not a supported dimension type'
    if d == 'longitude':
      dims.append('zonal_wavenumber')
    elif include_averaging_dims or d not in AVERAGING_DIMS.value:
      dims.append(d)
  return dims


def _impose_data_selection(
    source: xr.Dataset,
    source_chunks: t.Mapping[str, int],
) -> tuple[xr.Dataset, t.Mapping[str, int]]:
  """Select subset of source data for this script."""
  source = source[BASE_VARIABLES.value]
  selection = {
      TIME_DIM.value: slice(TIME_START.value, TIME_STOP.value),
  }
  if 'level' in source.dims:
    selection['level'] = [int(level) for level in LEVELS.value]
  source = source.sel({k: v for k, v in selection.items() if k in source.dims})
  source_chunks = {  # Remove dims that disappeared after data selection
      k: v for k, v in source_chunks.items() if k in source.dims
  }
  source_chunks = {  # Truncate chunks that are shorter after data selection
      k: min(source.sizes[k], source_chunks[k]) for k in source_chunks
  }
  return source, source_chunks


def _strip_offsets(
    key: xbeam.Key, dataset: xr.Dataset
) -> tuple[xbeam.Key, xr.Dataset]:
  """Remove offsets without corresponding coordinate in dataset."""
  key = key.with_offsets(
      **{k: v if k in dataset.coords else None for k, v in key.offsets.items()}
  )
  return key, dataset


def main(argv: list[str]) -> None:
  derived_variables = [
      ZonalEnergySpectrum(varname) for varname in BASE_VARIABLES.value
  ]

  source_dataset, source_chunks = xbeam.open_zarr(INPUT_PATH.value)
  source_dataset, source_chunks = _impose_data_selection(
      source_dataset, source_chunks
  )
  output_chunks = {}
  for d in _output_dims(source_dataset, include_averaging_dims=False):
    if d == 'zonal_wavenumber':
      output_chunks[d] = source_chunks['longitude']
    else:
      output_chunks[d] = source_chunks[d]

  template = _make_template(source_dataset, derived_variables)

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    _ = (
        root
        | xbeam.DatasetToChunks(
            source_dataset,
            source_chunks,
            split_vars=False,
            num_threads=NUM_THREADS.value,
        )
        | beam.MapTuple(
            lambda k, v: (  # pylint: disable=g-long-lambda
                k,
                _make_derived_variables_ds(v, derived_variables),
            )
        )
        | xbeam.SplitVariables()
        | beam.MapTuple(_strip_offsets)
        | xbeam.Mean(AVERAGING_DIMS.value, fanout=FANOUT.value)
        | xbeam.ChunksToZarr(
            OUTPUT_PATH.value,
            template,
            output_chunks,
            num_threads=NUM_THREADS.value,
        )
    )


if __name__ == '__main__':
  app.run(main)
