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
"""Classes for computing derived variables dynamically for evaluation."""
import dataclasses

import numpy as np
import xarray as xr


_EARTH_RADIUS_KM = 6378


@dataclasses.dataclass
class DerivedVariable:
  """Derived variable base class."""

  variable_name: str

  @property
  def base_variables(self) -> list[str]:
    """Return a list of base variables."""
    return []

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    """Compute derived variable, returning it in a new DataArray."""
    raise NotImplementedError


@dataclasses.dataclass
class WindSpeed(DerivedVariable):
  """Compute wind speed."""

  u_name: str
  v_name: str

  @property
  def base_variables(self) -> list[str]:
    return [self.u_name, self.v_name]

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    ws = np.sqrt(dataset[self.u_name] ** 2 + dataset[self.v_name] ** 2)
    return ws


@dataclasses.dataclass
class PrecipitationAccumulation(DerivedVariable):
  """Compute precipitation accumulation.

  Accumulation is computed for the time period leading up to the lead_time.
  E.g. 24h accumulation at lead_time=24h indicates 0-24h accumulation.
  Caution: Small negative values sometimes appear in model output.
  Here, we set them to zero.
  """

  total_precipitation_name: str
  accumulation_hours: int
  lead_time_name: str = 'prediction_timedelta'
  set_negative_to_zero: bool = True

  @property
  def base_variables(self) -> list[str]:
    return [self.total_precipitation_name]

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    # Get timestep diff
    tp = dataset[self.total_precipitation_name]
    diff = tp.diff(self.lead_time_name)

    # Compute accumulation steps
    timestep = dataset[self.lead_time_name].diff(self.lead_time_name)
    assert np.all(timestep == timestep[0]), 'All time steps must be equal.'
    timestep = timestep.values[0]
    steps = float(np.timedelta64(self.accumulation_hours, 'h') / timestep)
    assert steps.is_integer(), 'Accumulation time must be multiple of timestep.'

    # Compute accumulation
    accumulation = diff.rolling({self.lead_time_name: int(steps)}).sum()
    if self.set_negative_to_zero:
      # Set negative values to 0
      accumulation = accumulation.where(
          np.logical_or(accumulation >= 0.0, np.isnan(accumulation)), 0.0
      )
    # Add 0th time step with NaNs
    accumulation = xr.concat(
        [tp.isel({self.lead_time_name: [0]}) * np.NaN, accumulation],
        self.lead_time_name,
    )
    return accumulation


@dataclasses.dataclass
class ZonalPowerSpectrum(DerivedVariable):
  """Power spectrum along zonal direction.

  Given dataset with longitude dimension, this class computes spectral power as
  a function of wavenumber and frequency. Only non-negative frequencies are
  included (with units of "1 / km").

  At latitude α, where the circle of latitude has radius
  R(α) = R₀ Cos(α π / 180), the kth wavenumber corresponds to frequency
    f(k, α) = 1 / (2π R(α) longitude[k] / 360)

  Here, the DFT of a signal x[n], n = 0,..., N-1 is computed as
    X[k] = (1 / N) Σₙ x[n] exp(-2πink/N)
  The power spectrum is then
    S[0] = |X[0]|²,
    S[k] = 2 |X[k]|², k > 0, to account for positive and negative frequencies.

  This choice of normalization ensures that spectrum viewed as a function of
  frequency f(k, α) (see above) is independent of discretization N (up to
  discretization error, so long as N is high enough that f(k, α) can be
  resolved). If power spectral *density* is desired, the user should divide S[k]
  by the difference f(k, α) - f(k - 1, α).

  Attributes:
    variable_name: Name to use as base and also store output in.
  """

  variable_name: str

  @property
  def base_variables(self) -> list[str]:
    return [self.variable_name]

  def _lon_spacing_km(self, dataset: xr.Dataset) -> xr.DataArray:
    """Spacing between longitudinal values in `dataset`."""
    circum_at_equator = 2 * np.pi * _EARTH_RADIUS_KM
    circum_at_lat = np.cos(dataset.latitude * np.pi / 180) * circum_at_equator
    diffs = dataset.longitude.diff('longitude')
    if np.max(np.abs(diffs - diffs[0])) > 1e-3:
      raise ValueError(
          f'Expected uniform longitude spacing. {dataset.longitude.values=}'
      )
    return circum_at_lat * diffs[0].data / 360

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    """Computes zonal power at wavenumber and frequency."""
    spacing = self._lon_spacing_km(dataset)

    def simple_power(f_x):
      f_k = np.fft.rfft(f_x, axis=-1, norm='forward')
      # freq > 0 should be counted twice in power since it accounts for both
      # positive and negative complex values.
      one_and_many_twos = np.concatenate(([1], [2] * (f_k.shape[-1] - 1)))
      return np.real(f_k * np.conj(f_k)) * one_and_many_twos

    spectrum = xr.apply_ufunc(
        simple_power,
        dataset,
        input_core_dims=[['longitude']],
        output_core_dims=[['longitude']],
        exclude_dims={'longitude'},
    ).rename_dims(
        {'longitude': 'wavenumber'}
    )[self.variable_name]
    spectrum = spectrum.assign_coords(
        wavenumber=('wavenumber', spectrum.wavenumber.data)
    )
    base_frequency = xr.DataArray(
        np.fft.rfftfreq(len(dataset.longitude)),
        dims='wavenumber',
        coords={'wavenumber': spectrum.wavenumber},
    )
    spectrum = spectrum.assign_coords(frequency=base_frequency / spacing)
    spectrum['frequency'] = spectrum.frequency.assign_attrs(units='1 / km')
    return spectrum


@dataclasses.dataclass
class AggregatePrecipitationAccumulation(DerivedVariable):
  """Compute longer aggregation periods from existing shorter accumulations.

  Note: This is designed specifically for GraphCast forecasts for now. Assumes a
  6h raw time step and prediction_timedelta starting at 6h.
  """

  accumulation_hours: int
  raw_accumulation_name: str = 'total_precipitation_6hr'
  lead_time_name: str = 'prediction_timedelta'

  @property
  def base_variables(self):
    return [self.raw_accumulation_name]

  def compute(self, dataset: xr.Dataset):
    tp6h = dataset[self.raw_accumulation_name]

    # Compute aggregation steps
    steps = float(
        np.timedelta64(self.accumulation_hours, 'h') / np.timedelta64(6, 'h')
    )
    assert steps.is_integer(), 'Accumulation time must be multiple of timestep.'
    # Compute accumulation
    accumulation = tp6h.rolling({self.lead_time_name: int(steps)}).sum()
    return accumulation


DERIVED_VARIABLE_DICT = {
    'wind_speed': WindSpeed(
        u_name='u_component_of_wind',
        v_name='v_component_of_wind',
        variable_name='wind_speed',
    ),
    '10m_wind_speed': WindSpeed(
        u_name='10m_u_component_of_wind',
        v_name='10m_v_component_of_wind',
        variable_name='10m_wind_speed',
    ),
    'total_precipitation_6hr': PrecipitationAccumulation(
        total_precipitation_name='total_precipitation',
        accumulation_hours=6,
        lead_time_name='prediction_timedelta',
        variable_name='total_precipitation_6hr',
    ),
    'total_precipitation_24hr': PrecipitationAccumulation(
        total_precipitation_name='total_precipitation',
        accumulation_hours=24,
        lead_time_name='prediction_timedelta',
        variable_name='total_precipitation_24hr',
    ),
    'total_precipitation_24hr_from_6hr': AggregatePrecipitationAccumulation(
        accumulation_hours=24,
        lead_time_name='prediction_timedelta',
        variable_name='total_precipitation_24hr',
    ),
}
