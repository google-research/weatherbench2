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
r"""CLI to create probabilistic forecasts from historical ground truth.

These forecasts are sampled from contiguous FORECAST_DURATION length segments of
INPUT (typically an obsrvation). They are intended to be evaluated in the
Weatherbench2 pipeline, and serve as a baseline for evaluation.

For each initial time between INITIAL_TIME_START and INITIAL_TIME_END, spaced
by INITIAL_TIME_SPACING, this script creates forecasts that

* are comprised of ENSEMBLE_SIZE realizations.
* appear at lead times (dimension "prediction_timedelta") from 0 out to
  FORECAST_DURATION, spaced by TIMEDELTA_SPACING.

For each output initial time T, forecasts are created in two steps.
1. Sample, an ENSEMBLE_SIZE set of perturbed initial times {t1, ..., }.
2. For each ti, create a forecast (indexed by "prediction_timedelta") comprised
   of the historical weather starting at ti.

Each "ti" is selected to be a perturbation of the output init time T as

* t.minute = T.minute
* t.hour = T.hour
* t.year ~ Uniform({CLIMATOLOGY_START_YEAR,..., CLIMATOLOGY_END_YEAR})
* t.day = (T.day + δ) % [days in t.year], where the day offset δ is uniform:
  δ ~ Uniform(-DAY_WINDOW_SIZE // 2, DAY_WINDOW_SIZE // 2) + DAY_WINDOW_SIZE % 2

Example Usage:

  ```
  export BUCKET=my-bucket
  export PROJECT=my-project

  python scripts/compute_probabilistic_climatological_forecasts.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/era5/$USER/probabilistic-climatological-forecasts.zarr \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --initial_time_start=2020-01-01 \
    --initial_time_stop=2020-12-31 \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./scripts/dataflow-requirements.txt \
    --job_name=compute-vertical-profile-$USER
  ```
"""

import calendar
from collections import abc
import typing as t

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
import pandas as pd
from weatherbench2 import flag_utils
import xarray as xr
import xarray_beam as xbeam

REALIZATION = 'realization'


# Paths
INPUT_PATH = flags.DEFINE_string(
    'input_path',
    None,
    help=(
        'Input ground truth. Should contain weather for all possible sample'
        ' times. This means (i) time resolution should be at least daily, (ii)'
        ' it should include all days of year that could be seen by forecasts,'
        ' and (iii), for simplicity, every day should contain weather at the'
        ' same time of day.'
    ),
)
OUTPUT_PATH = flags.DEFINE_string('output_path', None, help='Output Zarr path.')

# To determine what to take from INPUT_PATH.
CLIMATOLOGY_START_YEAR = flags.DEFINE_integer(
    'climatology_start_year',
    1990,
    help='Inclusive start year of --input_path to include.',
)
CLIMATOLOGY_END_YEAR = flags.DEFINE_integer(
    'climatology_end_year',
    2020,
    help='Inclusive end year of --input_path to include.',
)
LEVELS = flags.DEFINE_list(
    'levels',
    None,
    help=(
        'Comma delimited list of pressure levels to compute spectra on. If'
        ' empty, compute on all levels of --input_path'
    ),
)
VARIABLES = flags.DEFINE_list(
    'variables',
    None,
    help=(
        'Comma delimited list of variables to select from weather. The default'
        ' will include all variables.'
    ),
)
TIME_DIM = flags.DEFINE_string(
    'time_dim', 'time', help='Name for the time dimension in input and output.'
)

# Determines what times will appear in the output.
INITIAL_TIME_START = flags.DEFINE_string(
    'initial_time_start',
    None,
    help='First initial time in output forecasts',
)
INITIAL_TIME_END = flags.DEFINE_string(
    'initial_time_end',
    None,
    help='Last initial time in output forecasts',
)
INITIAL_TIME_SPACING = flags.DEFINE_string(
    'initial_time_spacing',
    '6h',
    help=(
        'Spacing between initial forecast times. Must be a multiple of spacing'
        ' between times in INPUT. Must be a multiple or divisor of both one day'
        ' and TIMEDELTA_SPACING. Cannot specify resolution finer than one'
        ' hour.'
    ),
)
FORECAST_DURATION = flags.DEFINE_string(
    'forecast_duration', '15 days', help='Length of forecasts.'
)
TIMEDELTA_SPACING = flags.DEFINE_string(
    'timedelta_spacing',
    '12h',
    help=(
        'Distance between lead times in forecasts. Must be a multiple of'
        ' difference between times in INPUT. Must be a multiple or divisor of'
        ' both one day and INITIAL_TIME_SPACING. Cannot specify resolution'
        ' finer than one hour.'
    ),
)

# Determines how to form ensembles.
DAY_WINDOW_SIZE = flags.DEFINE_integer(
    'day_window_size',
    10,
    help='Width of window (in days) to take samples from.',
)
ENSEMBLE_SIZE = flags.DEFINE_integer(
    'ensemble_size',
    2,
    help=(
        'Size of output in the REALIZATION_NAME dimension. Setting to "-1" is'
        ' the same as ensemble_size = "number of possible day perturbations" x'
        ' "number of possible years." If WITH_REPLACEMENT=False as well, this'
        ' means every possible day and year combination will be used exactly'
        ' once.'
    ),
)
WITH_REPLACEMENT = flags.DEFINE_boolean(
    'with_replacement',
    True,
    help='Whether sampling is done with or without replacement.',
)
SEED = flags.DEFINE_integer(
    'seed', 802701, help='Seed for the random number generator.'
)

# Determines what the output will look like.
OUTPUT_CHUNKS = flag_utils.DEFINE_chunks(
    'output_chunks',
    '',
    help=(
        'Chunk sizes for output. Values here override input chunk values. This'
        ' could be useful if FORECAST_DURATION is very long or ENSEMBLE_SIZE is'
        ' large. The default is the input chunk sizes, and -1 for both the'
        ' DELTA and REALIZATION dims.'
    ),
)
REALIZATION_NAME = flags.DEFINE_string(
    'realization_name',
    REALIZATION,
    'Name of realization/member/number dimension.',
)

# Computing choices.
NUM_THREADS = flags.DEFINE_integer(
    'num_threads',
    None,
    help='Number of chunks to read/write in parallel per worker.',
)
RUNNER = flags.DEFINE_string(
    'runner',
    None,
    help='beam.runners.Runner',
)

# Names for coordinates on output dataset.
DELTA = 'prediction_timedelta'

ONE_DAY = pd.Timedelta('1d')


def _with_new_values_inserted(
    keys: list[str],
    values: list[t.Any],
    a_dict: dict[str, t.Any],
):
  """Insert new values into copy of dictionary and return the new dict."""
  a_dict = a_dict.copy()
  for key, value in zip(keys, values):
    if key in a_dict:
      raise ValueError(f'Expected {key=} was new, but was not: {a_dict=}')
    a_dict[key] = value
  return a_dict


def _independent_choice(x: np.ndarray, axis: int, n=None, seed=None):
  """Shuffle and choose n x along axis, independently for every batch axis."""
  # np.choice(x, axis) would make the same choice along every batch axis.
  rng = np.random.default_rng(seed=seed)
  indices = rng.random(x.shape).argsort(axis=axis)
  if n is not None:
    if n < 1 or n > x.shape[axis]:
      raise ValueError(
          f'n must be None or in [1, x.shape[axis]] = [1, {x.shape[axis]}],'
          f' found {n=}'
      )
    indices = np.take(indices, np.arange(n), axis=axis)
  return np.take_along_axis(x, indices, axis=axis)


def _get_possible_year_values(
    climatology_start_year: int, climatology_end_year: int
) -> np.ndarray:
  return np.arange(climatology_start_year, climatology_end_year + 1)


def _get_possible_day_perturbation_values(day_window_size: int) -> np.ndarray:
  day_perturbation_values = (
      np.arange(-day_window_size // 2, day_window_size // 2)
      + day_window_size % 2
  )
  assert len(day_perturbation_values) == day_window_size
  return day_perturbation_values


def _get_ensemble_size(
    ensemble_size_flag_value: int,
    climatology_start_year: int,
    climatology_end_year: int,
    day_window_size: int,
) -> int:
  """Computes the ensemble size from FLAGS."""
  if ensemble_size_flag_value == -1:
    return len(
        _get_possible_year_values(climatology_start_year, climatology_end_year)
    ) * len(_get_possible_day_perturbation_values(day_window_size))
  elif ensemble_size_flag_value > 0:
    return ensemble_size_flag_value
  else:
    raise flags.ValidationError(
        f'{ensemble_size_flag_value=} but should be "-1" or a positive integer'
    )


def _repeat_along_new_axis(
    x: np.ndarray, repeats: int, axis: int
) -> np.ndarray:
  """Repeat `x`, `repeat` times along new axis `axis`."""
  return np.repeat(
      np.expand_dims(x, axis=axis),
      repeats,
      axis=axis,
  )


def _get_sampled_init_times(
    output_times: pd.DatetimeIndex,
    climatology_start_year: int,
    climatology_end_year: int,
    day_window_size: int,
    ensemble_size: int,
    with_replacement: bool,
    seed: int,
) -> np.ndarray:
  """For each output time, get the times to sample from observations.

  Each initial time (in the output file) will be "represented" by a number of
  times sampled from historical observations. This returns those times as a
  shape [ensemble_size, len(output_times)] array T, such that

    T[i, j] is used as the ith realization of the forecast at output_times[j].

  Roughly speaking, with t = output_times[k] is an output time,

    T[i, j] = y + (t.day + δ) % DaysInYear(y) + t.hour

  where y is a random year, and δ is a day perturbation, both depending only on
  seed, ensemble_size, and len(output_times). Therefore, if we perturb
  output_times, this function returns an array perturbed by exactly the same
  amount.

  Args:
    output_times: Initial times for the output forecasts.
    climatology_start_year: First year to grab samples from.
    climatology_end_year: Last year to grab samples from.
    day_window_size: Size of window, in dayofyear, to grab samples.
    ensemble_size: Number of samples (per init time) to grab.
    with_replacement: Whether to sample with or without replacement.
    seed: Integer seed for the RNG.

  Returns:
    Shape [ensemble_size, len(output_times)] array of np.datetime64[ns].
  """
  rng = np.random.default_rng(seed)

  # The scheme below samples uniformly over initial day (ignoring leap years).
  # Conceptually, think of each climatology year as a circle. The days
  # [0, ..., 365] with 0 and 365 (or 366) connected. This sampler
  # (i) selects a random year (circle)
  # (ii) starts at a location output_time.dayofyear
  # (iii) adds a uniform perturbation (on the circle) to find a new day.

  # Get the range of possible values
  day_perturbation_values = _get_possible_day_perturbation_values(
      day_window_size
  )
  year_values = _get_possible_year_values(
      climatology_start_year, climatology_end_year
  )
  n_days = len(day_perturbation_values)
  n_years = len(year_values)
  n_times = len(output_times)
  if ensemble_size > 0:
    pass
  elif ensemble_size == -1:
    ensemble_size = n_days * n_years
  else:
    raise ValueError(f'{ensemble_size=} was not > 0 or -1.')

  sample_shape = (ensemble_size, len(output_times))

  # Get sampled years and day_perturbations.
  if with_replacement:
    # In this case, years and days are iid samples. Easy!
    years = rng.integers(
        year_values.min(),
        year_values.max() + 1,  # +1 because the interval is open on the right.
        size=sample_shape,
    )
    day_perturbations = rng.integers(
        day_perturbation_values.min(),
        day_perturbation_values.max() + 1,
        size=sample_shape,
    )
  else:
    if not isinstance(seed, int):
      raise AssertionError(
          f'{seed=} was not an integer. Seeding with None causes a nasty bug'
          ' whereby different choices will be used for day_perturbations and'
          ' years!'
      )
    tiled_day_window_values = _repeat_along_new_axis(
        # tiled_day_window_values.shape = [n_years, n_days, n_times].
        # tiled_day_window_values[i, :, j] = day_window_values for every i, j.
        _repeat_along_new_axis(
            day_perturbation_values, repeats=n_years, axis=0
        ),
        repeats=n_times,
        axis=-1,
    )
    day_perturbations = _independent_choice(
        tiled_day_window_values.reshape(-1, n_times),
        axis=0,
        n=ensemble_size,
        seed=seed,
    )

    tiled_year_values = _repeat_along_new_axis(
        # tiled_year_values.shape = [n_years, n_days, n_times].
        # tiled_year_values[:, i, j] = year_values, for every i, j.
        _repeat_along_new_axis(year_values, repeats=n_days, axis=-1),
        repeats=n_times,
        axis=-1,
    )
    years = _independent_choice(
        tiled_year_values.reshape(-1, n_times),
        axis=0,
        n=ensemble_size,
        seed=seed,
    )
  # End of get sampled years and day_perturbations.

  # If output_times is near the start or end of the year, we want the
  # perturbation to wrap around and find a date within the same year.
  dayofyears = output_times.dayofyear.values + day_perturbations
  for year in range(climatology_start_year, climatology_end_year + 1):
    mask = years == year
    dayofyears[mask] = (dayofyears[mask] - 1) % (
        365 + calendar.isleap(year)
    ) + 1

  return (
      # Years is always defined in years since the epoch.
      np.array(years - 1970, dtype='datetime64[Y]')
      # Add daysofyears - 1 to year, since e.g. if dayofyear = 1, then we will
      # add 0 to the year, which results in the first day of the year.
      + np.array(dayofyears - 1, dtype='timedelta64[D]')
      + np.array(output_times.hour, dtype='timedelta64[h]')
  ).astype('datetime64[ns]')


def _check_input_spacing_and_time_flags(input_ds: xr.Dataset) -> None:
  """Validates input spacing, TIMEDELTA_SPACING, and INITIAL_TIME_SPACING."""
  input_spacings = np.unique(np.diff(input_ds[TIME_DIM.value].data))
  if len(input_spacings) != 1:
    raise ValueError(f'Non-unique spacing in INPUT along dim {TIME_DIM.value}')

  # We want these three to be multiples or divisors of each other. Why?
  # It allows us to grab one set of chunks from INPUT and use in multiple places
  input_spacing = pd.to_timedelta(input_spacings[0])
  timedelta = pd.Timedelta(TIMEDELTA_SPACING.value)
  init_time_spacing = pd.Timedelta(INITIAL_TIME_SPACING.value)

  if timedelta % init_time_spacing and init_time_spacing % timedelta:
    raise ValueError(
        f'Neither one of {TIMEDELTA_SPACING.value=} and'
        f' {INITIAL_TIME_SPACING=} was a multiple of the other.'
    )

  for flag_name, value, delta in [
      ('TIMEDELTA_SPACING', TIMEDELTA_SPACING.value, timedelta),
      ('INITIAL_TIME_SPACING', INITIAL_TIME_SPACING.value, init_time_spacing),
  ]:
    if delta % input_spacing:
      raise ValueError(
          f'Requested {flag_name}={value} is not a multiple of input'
          f' spacing {input_spacing}'
      )
    if delta % ONE_DAY and ONE_DAY % delta:
      raise ValueError(
          f'Requested {flag_name}={value} was neither a multiple or divisor of'
          ' one day. This will result in different times of day being used at'
          ' different analysis points, which is inconvenient.'
      )
    if delta % pd.Timedelta('1h'):
      raise ValueError(
          f'Requested {flag_name}={value} specified sub-hour resolution.'
      )


def _emit_sampled_weather(
    unused_groupby_key: int,
    values: dict[
        str, t.Union[list[tuple[xbeam.Key, xr.Dataset]], list[dict[str, t.Any]]]
    ],
) -> t.Iterable[tuple[xbeam.Key, xr.Dataset]]:
  """Scatters one dataset to multiple init times and timedeltas.

  Args:
    unused_groupby_key: Key for grouping. Will be the hash of the time in the
      Dataset.
    values: Dictionary with keys "dataset_in_chunks" and
      "time_key_and_index_info"

  Yields:
    Tuples of keys and Dataset chunks to use in an xbeam pipeline.
  """
  # We should encounter one and only one (xbeam.key, Dataset) in values.
  if len(values['dataset_in_chunks']) != 1:
    raise AssertionError(
        'Expected exactly one (xbeam.Key, Dataset) pair in values. Found'
        f' {values=}'
    )
  xbeam_key, ds = values['dataset_in_chunks'][0]
  if not (isinstance(xbeam_key, xbeam.Key) and isinstance(ds, xr.Dataset)):
    raise ValueError(f'Unexpected values {xbeam_key=}, {ds=}')

  # If output_index_info is empty, it just means we have a Dataset chunk and no
  # output times to scatter it to. That's okay, we will Yield nothing.
  for info in values['time_key_and_index_info']:
    info = info.copy()
    del info['sampled_time_value']  #  Was only for ValueError printouts above.
    output_ds = (
        ds.expand_dims({DELTA: [info.pop('timedelta_value')]})
        .assign_coords({TIME_DIM.value: [info.pop('output_init_time_value')]})
        .expand_dims({REALIZATION_NAME.value: [info[REALIZATION_NAME.value]]})
    )
    assert isinstance(xbeam_key, xbeam.Key), xbeam_key  # To satisfy pytype.
    yield xbeam_key.with_offsets(**info), output_ds


def main(argv: abc.Sequence[str]) -> None:

  input_ds, input_chunks = xbeam.open_zarr(INPUT_PATH.value)

  if VARIABLES.value:
    input_ds = input_ds[VARIABLES.value]
  if LEVELS.value:
    input_ds = input_ds.sel(level=[int(l) for l in LEVELS.value])

  input_chunks = {k: v for k, v in input_chunks.items() if k in input_ds.dims}

  _check_input_spacing_and_time_flags(input_ds)
  ensemble_size = _get_ensemble_size(
      ENSEMBLE_SIZE.value,
      CLIMATOLOGY_START_YEAR.value,
      CLIMATOLOGY_END_YEAR.value,
      DAY_WINDOW_SIZE.value,
  )

  # Select all needed samples from INPUT.
  time_buffer = pd.to_timedelta(FORECAST_DURATION.value) + pd.to_timedelta(
      f'{DAY_WINDOW_SIZE.value}d'
  )
  sample_spacing = min(
      ONE_DAY,
      pd.Timedelta(TIMEDELTA_SPACING.value),
      pd.Timedelta(INITIAL_TIME_SPACING.value),
  )
  times_needed_for_sampling = pd.date_range(
      pd.to_datetime(f'{CLIMATOLOGY_START_YEAR.value}-01-01'),
      pd.to_datetime(f'{CLIMATOLOGY_END_YEAR.value}-12-31') + time_buffer,
      freq=sample_spacing,
  )
  missing_times = times_needed_for_sampling.difference(input_ds[TIME_DIM.value])
  if missing_times.size:
    raise flags.ValidationError(
        'Time flags (CLIMATOLOGY_START_YEAR, CLIMATOLOGY_END_YEAR,'
        ' TIMEDELTA_SPACING) asked for values in INPUT that are not available.'
        f' {missing_times=}.'
    )
  input_ds = input_ds.sel({TIME_DIM.value: times_needed_for_sampling})

  # Define output times and the template.
  output_init_times = pd.date_range(
      INITIAL_TIME_START.value,
      INITIAL_TIME_END.value,
      freq=INITIAL_TIME_SPACING.value,
  )
  timedeltas = pd.timedelta_range(
      '0 days', FORECAST_DURATION.value, freq=TIMEDELTA_SPACING.value
  )
  assert isinstance(input_ds, xr.Dataset)  # To satisfy pytype.
  if DELTA in input_ds.dims:
    raise ValueError(f'INPUT_PATH data already had {DELTA} as a dimension')
  template = (
      xbeam.make_template(input_ds)
      .isel({TIME_DIM.value: 0}, drop=True)
      .expand_dims({TIME_DIM.value: output_init_times})
      .expand_dims({DELTA: timedeltas})
      .expand_dims({REALIZATION_NAME.value: np.arange(ensemble_size)})
  )

  sampled_init_times = _get_sampled_init_times(
      # _get_sampled_init_times returns shape [ensemble_size, n_times] array of
      # np.datetime64. These are use as initial times for output samples.
      # Ravel it into shape [ensemble_size * n_times] set of times.
      output_init_times,
      CLIMATOLOGY_START_YEAR.value,
      CLIMATOLOGY_END_YEAR.value,
      DAY_WINDOW_SIZE.value,
      ensemble_size,
      WITH_REPLACEMENT.value,
      SEED.value,
  ).ravel()

  def sampled_times_for_timedelta(timedelta: pd.Timedelta) -> np.ndarray:
    """Times to grab from input for forecasts at this timedelta."""
    # Simply add the timedelta to the sampled_init_times, ensuring the forecasts
    # are continuous in time.
    return sampled_init_times + timedelta.to_numpy()

  # init_time_offsets[i] is the (init_time, realization) offset to use with
  # sampled_init_times[i].
  init_time_offsets = np.stack(
      np.meshgrid(
          np.arange(len(output_init_times)),
          np.arange(ensemble_size),
      ),
      axis=-1,
  ).reshape(-1, 2)

  def make_index_info(timedelta: np.ndarray, timedelta_offset: int):
    """Information about where a particular dataset should be scattered to."""
    return (
        {
            'timedelta_value': timedelta,
            TIME_DIM.value: int(offset[0]),
            REALIZATION_NAME.value: int(offset[1]),
            DELTA: timedelta_offset,
        }
        for offset in init_time_offsets
    )

  # Convenient for distributing during a Beam stage below.
  ensemble_of_output_init_times = _repeat_along_new_axis(
      output_init_times.values, repeats=ensemble_size, axis=0
  ).ravel()

  # We will scatter each chunk to various init times and timedeltas. It's by far
  # easiest to use chunksize of 1 in time.
  working_chunks = input_chunks.copy()
  working_chunks[TIME_DIM.value] = 1

  # After the scatter, we have new ensemble and delta dims. They are size 1,
  # since they were assembled with working chunks of size 1 in the time dim.
  done_working_chunks = working_chunks.copy()
  done_working_chunks.update(
      {
          REALIZATION_NAME.value: 1,
          DELTA: 1,
      }
  )  # fmt: skip

  output_chunks = input_chunks.copy()
  output_chunks.update(
      {
          REALIZATION_NAME.value: -1,
          DELTA: -1,
      }
  )  # fmt: skip
  output_chunks.update(OUTPUT_CHUNKS.value)

  itemsize = max(var.dtype.itemsize for var in template.values())

  # TODO(langmore) Consider writing this as a "gather" into destination chunks,
  # rather than "scatter" from source chunks. This is potentially much less
  # expensive, because you can avoid two intermediate shuffles.
  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    time_key_and_index_info = (
        root
        | 'DistributeTimedeltas' >> beam.Create(enumerate(timedeltas))
        | 'GetSampledTimeAndOutputInitTimeAndIndexInfo'
        >> beam.FlatMapTuple(
            lambda timedelta_offset, timedelta: zip(
                sampled_times_for_timedelta(timedelta),
                ensemble_of_output_init_times,
                make_index_info(timedelta, timedelta_offset),
            )
        )
        | 'KeyBySampledTimeAndAssembleIndexInfo'
        >> beam.MapTuple(
            lambda sampled_time, output_init_time, index_info: (
                str(sampled_time),
                _with_new_values_inserted(
                    keys=['output_init_time_value', 'sampled_time_value'],
                    values=[output_init_time, sampled_time],
                    a_dict=index_info,
                ),
            )
        )
    )

    dataset_in_chunks = (
        root
        | xbeam.DatasetToChunks(
            input_ds,
            input_chunks,
            # Keep variables together since (i) the chunks are tiny
            # (ii) the groupby (on time) will send them to the same place
            # anyways.
            split_vars=False,
            num_threads=NUM_THREADS.value,
        )
        | 'RechunkToWorkingChunks' >> xbeam.SplitChunks(working_chunks)
        | 'KeyByInputTime'
        >> beam.MapTuple(
            lambda xbeam_key, ds: (
                str(ds[TIME_DIM.value].data[0]),
                (xbeam_key, ds),
            )
        )
    )

    _ = (
        {
            'dataset_in_chunks': dataset_in_chunks,
            'time_key_and_index_info': time_key_and_index_info,
        }
        | beam.CoGroupByKey()
        | 'ScatterInputDataset' >> beam.FlatMapTuple(_emit_sampled_weather)
        | 'RechunkToOutputChunks'
        >> xbeam.Rechunk(
            # Intermediate rechunk necessary since input/output chunks are
            # different.
            template.sizes,
            done_working_chunks,
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
  flags.mark_flags_as_required(
      ['input_path', 'output_path', 'initial_time_start', 'initial_time_end'],
  )
  app.run(main)
