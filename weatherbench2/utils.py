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
"""Utility function for WeatherBench2."""
import functools
from typing import Callable, Union

import fsspec
import numpy as np
import xarray as xr


def open_nc(filename: str) -> xr.Dataset:
  """Open NetCDF file from filesystem."""
  with fsspec.open(filename, 'rb') as f:
    ds = xr.open_dataset(f)
  return ds


def replace_time_with_doy(ds: xr.Dataset) -> xr.Dataset:
  """Replace time coordinate with days of year."""
  return ds.assign_coords({'time': ds.time.dt.dayofyear}).rename(
      {'time': 'dayofyear'}
  )


def select_hour(ds: xr.Dataset, hour: int) -> xr.Dataset:
  """Select given hour of day from Datset."""
  # Select hour
  ds = ds.isel(time=ds.time.dt.hour == hour)
  # Adjust time dimension
  ds = ds.assign_coords({'time': ds.time.astype('datetime64[D]')})
  return ds


def make_probabilistic_climatology(
    ds: xr.Dataset, start_year: int, end_year: int, hour_interval: int
) -> xr.Dataset:
  """Stack years as ensemble. Day 366 will only contain data for leap years."""
  hours = np.arange(0, 24, hour_interval)
  years = np.arange(start_year, end_year + 1)
  out = []
  for hour in hours:
    datasets = []
    for year in years:
      tmp = ds.isel(time=ds.time.dt.hour == hour).sel(time=str(year))
      tmp = tmp.assign_coords(dayofyear=tmp.time.dt.dayofyear).swap_dims(
          {'time': 'dayofyear'}
      )
      datasets.append(tmp)
    ds_per_hour = xr.concat(
        datasets,
        dim=xr.DataArray(
            np.arange(len(years)), coords={'number': np.arange(len(years))}
        ),
    )
    out.append(ds_per_hour)
  out = xr.concat(out, dim=xr.DataArray(hours, dims=['hour']))
  return out


def create_window_weights(window_size: int) -> xr.DataArray:
  """Create linearly decaying window weights."""
  assert window_size % 2 == 1, 'Window size must be odd.'
  half_window_size = window_size // 2
  window_weights = np.concatenate(
      [
          np.linspace(0, 1, half_window_size + 1),
          np.linspace(1, 0, half_window_size + 1)[1:],
      ]
  )
  window_weights = window_weights / window_weights.mean()
  window_weights = xr.DataArray(window_weights, dims=['window'])
  return window_weights


def compute_rolling_stat(
    ds: xr.Dataset,
    window_weights: xr.DataArray,
    stat_fn: Union[str, Callable[..., xr.Dataset]] = 'mean',
) -> xr.Dataset:
  """Compute rolling climatology."""
  window_size = len(window_weights)
  half_window_size = window_size // 2  # For padding
  # Stack years
  stacked = xr.concat(
      [
          replace_time_with_doy(ds.sel(time=str(y)))
          for y in np.unique(ds.time.dt.year)
      ],
      dim='year',
  )
  # Fill gap day (366) with values from previous day 365
  stacked = stacked.fillna(stacked.sel(dayofyear=365))
  # Pad edges for perioding window
  stacked = stacked.pad(pad_width={'dayofyear': half_window_size}, mode='wrap')
  # Weighted rolling mean
  stacked = stacked.rolling(dayofyear=window_size, center=True).construct(
      'window'
  )
  if stat_fn == 'mean':
    rolling_stat = stacked.weighted(window_weights).mean(dim=('window', 'year'))
  elif stat_fn == 'std':
    rolling_stat = stacked.weighted(window_weights).std(dim=('window', 'year'))
  else:
    rolling_stat = stat_fn(
        stacked, weights=window_weights, dim=('window', 'year')
    )
  # Remove edges
  rolling_stat = rolling_stat.isel(
      dayofyear=slice(half_window_size, -half_window_size)
  )
  return rolling_stat


def compute_daily_stat(
    obs: xr.Dataset,
    window_size: int,
    clim_years: slice,
    stat_fn: Union[str, Callable[..., xr.Dataset]] = 'mean',
) -> xr.Dataset:
  """Compute daily average climatology with running window."""
  # NOTE: Loading seems to be necessary, otherwise computation takes forever
  # Will be converted to xarray-beam pipeline anyway
  obs = obs.load()
  obs_daily = obs.sel(time=clim_years).resample(time='D').mean()
  window_weights = create_window_weights(window_size)
  daily_rolling_clim = compute_rolling_stat(obs_daily, window_weights, stat_fn)
  return daily_rolling_clim


def compute_hourly_stat(
    obs: xr.Dataset,
    window_size: int,
    clim_years: slice,
    hour_interval: int,
    stat_fn: Union[str, Callable[..., xr.Dataset]] = 'mean',
) -> xr.Dataset:
  """Compute climatology by day of year and hour of day."""
  obs = obs.compute()
  hours = xr.DataArray(range(0, 24, hour_interval), dims=['hour'])
  window_weights = create_window_weights(window_size)

  hourly_rolling_clim = xr.concat(
      [  # pylint: disable=g-complex-comprehension
          compute_rolling_stat(
              select_hour(obs.sel(time=clim_years), hour),
              window_weights,
              stat_fn,
          )
          for hour in hours
      ],
      dim=hours,
  )
  return hourly_rolling_clim


def smooth_dayofyear_variable_with_rolling_window(
    obs_dayofyear: xr.Dataset, window_size: int
) -> xr.Dataset:
  """Smoothens day of year values with rolling window.

  The rolling window runs on the loop of connecting the first and the last
  day of the year. This provides another way to calculate daily climatological
  mean and std by providing as input the raw mean/std of each 'dayofyear', and
  smoothing out the timeseries.

  Args:
    obs_dayofyear: dataset with dimension of dayofyear.
    window_size: the number of days of the rolling window.

  Returns:
    A dataset with dimension of dayofyear.
  """
  assert 'dayofyear' in obs_dayofyear.dims, 'dayofyear must be a dimension.'
  # The window_size is required to be odd in create_window_weights().
  window_weights = create_window_weights(window_size)
  half_window = window_size // 2
  stacked_rolling = xr.concat(
      [
          obs_dayofyear.roll(dayofyear=i) * window_weights[i + half_window]
          for i in np.arange(-half_window, window_size - half_window)
      ],
      dim='stack',
  )
  return stacked_rolling.mean(dim='stack')


def compute_daily_climatology_std(
    obs: xr.Dataset, window_size: int, clim_years: slice
) -> xr.Dataset:
  """Computes daily climatological std with rolling window."""
  obs_daily = obs.sel(time=clim_years).resample(time='D').mean()
  std_daily = obs_daily.groupby('time.dayofyear').std()
  return smooth_dayofyear_variable_with_rolling_window(std_daily, window_size)


def compute_daily_climatology_mean(
    obs: xr.Dataset, window_size: int, clim_years: slice
) -> xr.Dataset:
  """Computes daily climatological mean with rolling window."""
  obs_daily = obs.sel(time=clim_years).groupby('time.dayofyear').mean()
  return smooth_dayofyear_variable_with_rolling_window(obs_daily, window_size)


def compute_hourly_climatology_mean_fast(
    obs: xr.Dataset, window_size: int, clim_years: slice, hour_interval: int = 1
) -> xr.Dataset:
  """Compute climatology mean by day of year and hour of day."""
  obs = obs.sel(time=clim_years)
  hours = xr.DataArray(range(0, 24, hour_interval), dims=['hour'])
  hourly_rolling_clim = xr.concat(
      [
          smooth_dayofyear_variable_with_rolling_window(
              select_hour(obs, hour).groupby('time.dayofyear').mean(),
              window_size,
          )
          for hour in hours
      ],
      dim=hours,
  )
  return hourly_rolling_clim


def compute_hourly_climatology_std_fast(
    obs: xr.Dataset, window_size: int, clim_years: slice, hour_interval: int = 1
) -> xr.Dataset:
  """Compute climatology std by day of year and hour of day."""
  obs = obs.sel(time=clim_years)
  hours = xr.DataArray(range(0, 24, hour_interval), dims=['hour'])
  hourly_rolling_clim = xr.concat(
      [
          smooth_dayofyear_variable_with_rolling_window(
              select_hour(obs, hour).groupby('time.dayofyear').std(),
              window_size,
          )
          for hour in hours
      ],
      dim=hours,
  )
  return hourly_rolling_clim


def compute_hourly_stat_fast(
    obs: xr.Dataset,
    window_size: int,
    clim_years: slice,
    hour_interval: int,
    stat_fn: str = 'mean',
) -> xr.Dataset:
  """Compute climatology mean or std by day of year and hour of day."""
  if stat_fn == 'mean':
    return compute_hourly_climatology_mean_fast(
        obs, window_size, clim_years, hour_interval
    )
  elif stat_fn == 'std':
    return compute_hourly_climatology_std_fast(
        obs, window_size, clim_years, hour_interval
    )
  else:
    raise NotImplementedError(f'stat {stat_fn} not implemented.')


def compute_daily_stat_fast(
    obs: xr.Dataset,
    window_size: int,
    clim_years: slice,
    stat_fn: str = 'mean',
) -> xr.Dataset:
  """Compute climatology mean or std by day of year."""
  if stat_fn == 'mean':
    return compute_daily_climatology_mean(obs, window_size, clim_years)
  elif stat_fn == 'std':
    return compute_daily_climatology_std(obs, window_size, clim_years)
  else:
    raise NotImplementedError(f'stat {stat_fn} not implemented.')


def random_like(dataset: xr.Dataset, seed: int = 0) -> xr.Dataset:
  """Random normal dataset configured like `dataset`."""
  rs = np.random.RandomState(seed)
  return dataset.copy(
      data={k: rs.normal(size=v.shape) for k, v in dataset.items()}
  )


class _WrappedDataset:
  """Hashable wrapper for xarray.Datasets."""

  def __init__(self, value):
    if not isinstance(value, xr.Dataset):
      raise ValueError(f'_WrappedDataset cannot wrap type {type(value)}')
    self.value = value

  def __eq__(self, other):
    if not isinstance(other, _WrappedDataset):
      return False
    return self.value.equals(other.value)

  def __hash__(self):
    # Something that can be calculated quickly -- we won't have many collisions.
    # Hash collisions just mean that that __eq__ needs to be checked.
    # https://stackoverflow.com/questions/16589791/most-efficient-property-to-hash-for-numpy-array
    return hash(
        tuple(
            (k, repr(v.data.ravel())) for k, v in self.value.data_vars.items()
        )
    )


def dataset_safe_lru_cache(maxsize=128):
  """An xarray.Dataset compatible version of functools.lru_cache."""

  def decorator(func):  # pylint: disable=missing-docstring
    @functools.lru_cache(maxsize)
    def cached_func(*args, **kwargs):
      args = tuple(
          a.value if isinstance(a, _WrappedDataset) else a for a in args
      )
      kwargs = {
          k: v.value if isinstance(v, _WrappedDataset) else v
          for k, v in kwargs.items()
      }
      return func(*args, **kwargs)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # pylint: disable=missing-docstring
      args = tuple(
          _WrappedDataset(a) if isinstance(a, xr.Dataset) else a for a in args
      )
      kwargs = {
          k: _WrappedDataset(v) if isinstance(v, xr.Dataset) else v
          for k, v in kwargs.items()
      }
      return cached_func(*args, **kwargs)

    return wrapper

  return decorator
