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
"""Routines for enforcing and verifying schemas."""
from collections import abc
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray


def apply_time_conventions(
    forecast: xarray.Dataset, by_init: bool
) -> xarray.Dataset:
  """Apply WeatherBench2 time name conventions onto a forecast dataset."""
  forecast = forecast.copy()
  if 'prediction_timedelta' in forecast.coords:
    forecast = forecast.rename({'prediction_timedelta': 'lead_time'})
    if by_init:
      # Need to rename time dimension because different from time dimension in
      # truth dataset
      forecast = forecast.rename({'time': 'init_time'})
      valid_time = forecast.init_time + forecast.lead_time
      forecast.coords['valid_time'] = valid_time
      assert not hasattr(
          forecast, 'time'
      ), f'Forecast should not have time dimension at this point: {forecast}'
    else:
      init_time = forecast.time - forecast.lead_time
      forecast.coords['init_time'] = init_time
  return forecast


ALL_3D_VARIABLES = (
    'geopotential',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'specific_humidity',
)

ALL_2D_VARIABLES = ('2m_temperature',)


# Mean of equitorial and polar radius
EARTH_RADIUS_M = 1000 * (6357 + 6378) / 2


def mock_truth_data(
    *,
    variables_3d: abc.Sequence[str] = ALL_3D_VARIABLES,
    variables_2d: abc.Sequence[str] = ALL_2D_VARIABLES,
    levels: abc.Sequence[int] = (500, 700, 850),
    spatial_resolution_in_degrees: float = 10.0,
    time_start: str = '2020-01-01',
    time_stop: str = '2021-01-01',
    time_resolution: str = '1 day',
    dtype: npt.DTypeLike = np.float32,
) -> xarray.Dataset:
  """Create a mock truth dataset with all zeros for testing."""
  num_latitudes = round(180 / spatial_resolution_in_degrees) + 1
  num_longitudes = round(360 / spatial_resolution_in_degrees)
  freq = pd.Timedelta(time_resolution)
  coords = {
      'time': pd.date_range(time_start, time_stop, freq=freq, inclusive='left'),
      'latitude': np.linspace(-90, 90, num_latitudes),
      'longitude': np.linspace(0, 360, num_longitudes, endpoint=False),
      'level': np.array(levels),
  }
  dims_3d = ('time', 'level', 'longitude', 'latitude')
  shape_3d = tuple(coords[dim].size for dim in dims_3d)
  data_vars_3d = {k: (dims_3d, np.zeros(shape_3d, dtype)) for k in variables_3d}
  if not data_vars_3d:
    del coords['level']

  dims_2d = ('time', 'longitude', 'latitude')
  shape_2d = tuple(coords[dim].size for dim in dims_2d)
  data_vars_2d = {k: (dims_2d, np.zeros(shape_2d, dtype)) for k in variables_2d}

  data_vars = {**data_vars_3d, **data_vars_2d}
  return xarray.Dataset(data_vars, coords)


def mock_forecast_data(
    *,
    lead_start: str = '0 day',
    lead_stop: str = '10 day',
    lead_resolution: str = '1 day',
    ensemble_size: Optional[int] = None,
    **kwargs,
):
  """Create a mock forecast dataset with all zeros for testing."""
  lead_time = pd.timedelta_range(
      pd.Timedelta(lead_start),
      pd.Timedelta(lead_stop),
      freq=pd.Timedelta(lead_resolution),
  )
  ds = mock_truth_data(**kwargs)
  ds = ds.expand_dims(prediction_timedelta=lead_time)
  if ensemble_size is not None:
    ds = ds.expand_dims(realization=ensemble_size)
  return ds


def mock_hourly_climatology_data(
    *, hour_interval: int = 1, **kwargs
) -> xarray.Dataset:
  """Create a mock hourly climatology dataset with all zeros for testing."""
  hours = range(0, 24, hour_interval)
  ds = mock_truth_data(**kwargs)
  ds = ds.isel(time=0, drop=True)
  ds = ds.expand_dims(hour=hours, dayofyear=1 + np.arange(366))
  return ds
