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

Each source time "ti" is a perturbation of the output init time T:

* t.minute = T.minute
* t.hour = T.hour
* t.year ~ Uniform({CLIMATOLOGY_START_YEAR,..., CLIMATOLOGY_END_YEAR})
* t.day = (T.day + δ) % [days in t.year], where the day offset δ is uniform:
  δ ~ Uniform(-DAY_WINDOW_SIZE // 2, DAY_WINDOW_SIZE // 2) + DAY_WINDOW_SIZE % 2

The (T.day + δ) % [days in t.year] step is the default behavior indicated by
INITIAL_TIME_EDGE_BEHAVIOR=WRAP_YEAR. This is needed to ensure every year and
dayofyear is sampled with/without replacement. If instead,
INITIAL_TIME_EDGE_BEHAVIOR=REFLECT_RANGE, then

  t.day = (T.day + δ) % [days in t.year]
  t.year = T.year + (T.day + δ) // [days in t.year]

except at the climatology start/end boundary, where T.day is reflected back into
bounds.

By default, every initial time has its day and year sampled independently. This
means every single forecast could come from an entirely different season.
SAMPLE_HOLD_DAYS provides the ability to alter this behavior, by making each
realization fix the number of days between the output time and source time,
(T - t).days, for SAMPLE_HOLD_DAYS days in a row. After SAMPLE_HOLD_DAYS days,
each realization selects (independently) a new (T - t).days. This option is most
useful when used with INITIAL_TIME_EDGE_BEHAVIOR=REFLECT_RANGE. In that case,

* SAMPLE_HOLD_DAYS=365 means realizations each come from a random season (year
  and time of year), and this random season is changed only once every 365 days.
  This emulates a forecast model that may or may not have the seasonal trends
  (e.g. ENSO) correct.
* SAMPLE_HOLD_DAYS=50 emulates forecast model that may or may not have the
  subseasonal trends correct.


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
SAMPLE_HOLD_DAYS = flags.DEFINE_integer(
    'sample_hold_days',
    0,
    help=(
        'Non-negative multiple of INITIAL_TIME_SPACING. 0 means no hold. If'
        ' nonzero, the total days perturbation is constant for this time.'
        ' Warning: If INITIAL_TIME_EDGE_BEHAVIOR=WRAP_YEAR, The "hold" means'
        ' observations may be needed an additional year before or after the'
        ' CLIMATOLOGY START and END.'
    ),
)
WRAP_YEAR = 'WRAP_YEAR'
NO_EDGE = 'NO_EDGE'
REFLECT_RANGE = 'REFLECT_RANGE'
INITIAL_TIME_EDGE_BEHAVIOR = flags.DEFINE_enum(
    'initial_time_edge_behavior',
    WRAP_YEAR,
    enum_values=[WRAP_YEAR, NO_EDGE, REFLECT_RANGE],
    help=(
        'What to do when the day perturbation would select a time before or'
        f' after the sampled year. "{WRAP_YEAR}" means e.g. YYYY-12-31 + 5 days'
        f' becomes YYYY-01-05, for ever year YYYY.  "{NO_EDGE}" means the '
        'climatological edge is ignored, and initial times come from beyond the'
        ' [CLIMATOLOGY_START_YEAR, CLIMATOLOGY_END_YEAR] interval. If INPUT'
        ' does not extend far enough, a noisy failure will be raised. '
        f'"{REFLECT_RANGE}" means we'
        ' reflect the perturbation, but only do this at climatology boundary'
        ' years. So, if start/end is 1990, 2000, then 2000-12-31 + 5 days ='
        ' 2000-12-31 - 5 days = 2000-12-26, but 1995-12-31 + 5 days ='
        ' 1996-01-05. This ensures IID sampling from each year.'
    ),
)
FORECAST_DURATION = flags.DEFINE_string(
    'forecast_duration', '15 days', help='Length of forecasts.'
)
TIMEDELTA_SPACING = flags.DEFINE_string(
    'timedelta_spacing',
    '6h',
    help=(
        'Distance between lead times in forecasts. Must be a multiple of'
        ' difference between times in INPUT. Must be a multiple or divisor of'
        ' both one day and INITIAL_TIME_SPACING. Cannot specify resolution'
        ' finer than one hour.'
    ),
)

SOURCE_TIME = 'source_time'
ADD_SOURCE_TIME = flags.DEFINE_boolean(
    'add_source_time',
    False,
    help=(
        f'Whether to add a "{SOURCE_TIME}" variable, indicating what time in'
        ' INPUT_PATH was used for each output sample'
    ),
)

# Determines how to form ensembles.
DAY_WINDOW_SIZE = flags.DEFINE_integer(
    'day_window_size',
    10,
    help=(
        'Width of window (in days) to take samples from. Must be in [0, 2*364].'
    ),
)
ENSEMBLE_SIZE = flags.DEFINE_integer(
    'ensemble_size',
    2,
    help=(
        'Size of output in the REALIZATION_NAME dimension. Setting to "-1" is'
        ' the same as ensemble_size = "number of possible day perturbations" x'
        ' "number of possible years." If WITH_REPLACEMENT=False as well, this'
        ' means every possible day and year combination will be used exactly'
        f' once (if INITIAL_TIME_EDGE_BEHAVIOR="{WRAP_YEAR}").'
    ),
)
WITH_REPLACEMENT = flags.DEFINE_boolean(
    'with_replacement',
    True,
    help=(
        'Whether sampling is done with or without replacement. Warning: If'
        f' INITIAL_TIME_EDGE_BEHAVIOR="{REFLECT_RANGE}", then some samples may'
        ' be repeated near the climatological boundary, even if'
        ' with_replacement=False.'
    ),
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
    sample_hold_days: int,
    initial_time_edge_behavior: str,
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
    sample_hold_days: How long consecutive initial times use the same
      perturbation.  0 means switch perturbations every consecutive init time.
    initial_time_edge_behavior: How to deal with perturbations that move the
      sampled day outside of sampled year.
    seed: Integer seed for the RNG.

  Returns:
    Shape [ensemble_size, len(output_times)] array of np.datetime64[ns].
  """
  rng = np.random.default_rng(seed)

  if day_window_size > 2 * 364:
    # This complicates the REFLECT_RANGE behavior, and no sensible human would
    # want this.
    raise ValueError(f'{day_window_size=} > 2 * 364, which is not allowed.')

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

  dayofyears = output_times.dayofyear.values + day_perturbations

  if initial_time_edge_behavior == WRAP_YEAR:
    for year in range(climatology_start_year, climatology_end_year + 1):
      mask = years == year
      days_in_this_year = 365 + calendar.isleap(year)
      dayofyears[mask] = (dayofyears[mask] - 1) % days_in_this_year + 1

  elif initial_time_edge_behavior == REFLECT_RANGE:
    for year in {climatology_start_year, climatology_end_year}:
      mask = years == year
      days_in_this_year = 365 + calendar.isleap(year)
      if year == climatology_start_year:
        # Transform e.g. 1 --> 1, 0 --> 2, -1 --> 3
        dayofyears[mask] = np.where(
            dayofyears[mask] >= 1,
            dayofyears[mask],
            np.abs(dayofyears[mask]) + 2,
        )
      elif year == climatology_end_year:
        dayofyears[mask] = np.where(
            dayofyears[mask] <= days_in_this_year,
            dayofyears[mask],
            # If d > 365, set to 2*365 - d = 365 - (d - 365)
            2 * days_in_this_year - dayofyears[mask],
        )
  elif initial_time_edge_behavior == NO_EDGE:
    pass
  else:
    raise ValueError(f'Unhandled {initial_time_edge_behavior=}')

  sampled_times = (
      # Years is always defined in years since the epoch.
      np.array(years - 1970, dtype='datetime64[Y]')
      # Add daysofyears - 1 to year, since e.g. if dayofyear = 1, then we will
      # add 0 to the year, which results in the first day of the year.
      + np.array(dayofyears - 1, dtype='timedelta64[D]')
      + np.array(output_times.hour, dtype='timedelta64[h]')
  ).astype('datetime64[ns]')

  if sample_hold_days:
    output_time_strides = set(output_times.diff()[1:])
    if len(output_time_strides) > 1:
      raise ValueError(
          f'Cannot sample hold with more than one {output_time_strides=}'
      )
    output_time_stride = output_time_strides.pop()
    hold_dt = pd.Timedelta(f'{sample_hold_days}d')
    hold_stride = hold_dt // output_time_stride
    if output_time_stride * hold_stride != hold_dt:
      raise ValueError(
          f'{sample_hold_days=} was not a multiple of {output_time_stride=}'
      )
    hold_idx = np.repeat(
        # E.g. hold_idx = [0, 0, ..., 0, 1, 1, ..., 1, 2, ...]
        np.arange(len(output_times) // hold_stride + 1)[:, np.newaxis],
        hold_stride,
        axis=1,
    ).ravel()[: len(output_times)]

    # Convert np datetimes into δ days, sample-hold, then add back to datetimes.
    delta_days = np.array(
        pd.to_timedelta((sampled_times - output_times.values).ravel()).days,
        dtype=np.int64,
    ).reshape(sampled_times.shape)

    delta_days = np.take(delta_days, hold_idx, axis=1)
    sampled_times = output_times.values + np.array(
        delta_days, dtype='timedelta64[D]'
    )

  return sampled_times


def _check_times_in_dataset(
    times: np.ndarray | pd.DatetimeIndex, ds: xr.Dataset
) -> None:
  """Checks that `times` are in `ds` and gives a nice error if not."""
  missing_times = pd.to_datetime(times).difference(ds[TIME_DIM.value])
  if not missing_times.size:
    return
  err_lines = []
  if INITIAL_TIME_EDGE_BEHAVIOR.value == WRAP_YEAR and SAMPLE_HOLD_DAYS.value:
    err_lines.append(
        f'{INITIAL_TIME_EDGE_BEHAVIOR.value=} and'
        f' {SAMPLE_HOLD_DAYS.value=} means observations may be needed up to 1'
        ' year before/after the [CLIMATOLOGY_START_YEAR, CLIMATOLOGY_END_YEAR]'
        ' interval.'
    )
  err_lines.append(
      'Time flags (INITIAL_TIME_EDGE_BEHAVIOR, CLIMATOLOGY_START_YEAR,'
      ' CLIMATOLOGY_END_YEAR, TIMEDELTA_SPACING, SAMPLE_HOLD_DAYS) asked for'
      f' values in INPUT that are not available. {missing_times=}.'
  )
  raise flags.ValidationError('\n'.join(err_lines))


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
    output_ds = ds.copy()
    sampled_time_value = info.pop('sampled_time_value')
    if ADD_SOURCE_TIME.value:
      output_ds[SOURCE_TIME] = xr.DataArray(
          # Insert as a DataArray, which lets us assign the proper dims.
          [sampled_time_value],
          dims=TIME_DIM.value,
          coords={TIME_DIM.value: ds[TIME_DIM.value]},
      )
    output_ds = (
        output_ds.expand_dims({DELTA: [info.pop('timedelta_value')]})
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

  sampled_init_times = _get_sampled_init_times(
      # _get_sampled_init_times returns shape [ensemble_size, n_times] array of
      # np.datetime64. These are use as initial times for output samples.
      # Ravel it into shape [ensemble_size * n_times] set of times.
      output_times=output_init_times,
      climatology_start_year=CLIMATOLOGY_START_YEAR.value,
      climatology_end_year=CLIMATOLOGY_END_YEAR.value,
      day_window_size=DAY_WINDOW_SIZE.value,
      ensemble_size=ensemble_size,
      with_replacement=WITH_REPLACEMENT.value,
      initial_time_edge_behavior=INITIAL_TIME_EDGE_BEHAVIOR.value,
      sample_hold_days=SAMPLE_HOLD_DAYS.value,
      seed=SEED.value,
  ).ravel()

  def sampled_times_for_timedelta(timedelta: pd.Timedelta) -> np.ndarray:
    """Times to grab from input for forecasts at this timedelta."""
    # Simply add the timedelta to the sampled_init_times, ensuring the forecasts
    # are continuous in time.
    times = sampled_init_times + timedelta.to_numpy()
    return times

  times_needed_for_sampling = np.unique(
      np.stack([sampled_times_for_timedelta(td) for td in timedeltas])
  )
  _check_times_in_dataset(times_needed_for_sampling, input_ds)
  input_ds = input_ds.sel({TIME_DIM.value: times_needed_for_sampling})

  if ADD_SOURCE_TIME.value:
    input_ds = input_ds.assign(
        # Assign SOURCE_TIME with an arbitrary DataArray of type datetime64[ns].
        # Using a DataArray with time index is important: It ensures it will be
        # stored as a data_var, and will get indices sliced/expanded below
        # correctly. It is also important to not directly use input_ds.time.
        {
            # TODO(langmore) Remove the "+1" once Xarray bug is fixed;
            # https://github.com/pydata/xarray/issues/9859
            # Until then, assigning to input_ds[TIME_DIM.value] without the "+1"
            # results in an error:
            # ValueError: Cannot assign to the .data attribute of dimension
            # coordinate a.k.a. IndexVariable 'time'. Instead, add 1 so it is a
            # new variable.
            SOURCE_TIME: (input_ds[TIME_DIM.value]
                          + np.array(1, dtype='timedelta64[ns]'))  # fmt: skip
        }
    )
  template = (
      xbeam.make_template(input_ds)
      .isel({TIME_DIM.value: 0}, drop=True)
      .expand_dims({TIME_DIM.value: output_init_times})
      .expand_dims({DELTA: timedeltas})
      .expand_dims({REALIZATION_NAME.value: np.arange(ensemble_size)})
  )

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
