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
import typing as t

import numpy as np
import xarray as xr

from weatherbench2 import schema


@dataclasses.dataclass
class DerivedVariable:
  """Derived variable base class.

  Attributes:
    variable_name: Name of variable to compute.
  """

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
  """Compute wind speed.

  Attributes:
    u_name: Name of U component.
    v_name: Name of V component.
  """

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
  """Compute precipitation accumulation from hourly accumulations.

  Accumulation is computed for the time period leading up to the lead_time.
  E.g. 24h accumulation at lead_time=24h indicates 0-24h accumulation.
  Caution: Small negative values sometimes appear in model output.
  Here, we set them to zero.

  Attributes:
    total_precipitation_name: Name of hourly total_precipitation input.
    accumulation_hours: Hours to accumulate precipitation over
    lead_time_name: Name of lead_time dimension.
    set_negative_to_zero: Specify whether to set negative temporal differences
      to zero.
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
class ZonalEnergySpectrum(DerivedVariable):
  """Energy spectrum along the zonal direction.

  Given dataset with longitude dimension, this class computes spectral energy as
  a function of wavenumber (as a dim). wavelength and frequency are also present
  as coords with units "1 / m" and "m" respectively. Only non-negative
  frequencies are included.

  Let f[l], l = 0,..., L - 1, be dataset values along a zonal circle of constant
  latitude, with circumference C (m).  The DFT is
    F[k] = (1 / L) Σₗ f[l] exp(-i2πkl/L)
  The energy spectrum is then set to
    S[0] = C |F[0]|²,
    S[k] = 2 C |F[k]|², k > 0, to account for positive and negative frequencies.

  With C₀ the equatorial circumference, the ith zonal circle has circumference
    C(i) = C₀ Cos(π latitude[i] / 180).
  Since data points occur at longitudes longitude[l], l = 0, ..., L - 1, the DFT
  will measure spectra at zonal sampling frequencies
    f(k, i) = longitude[k] / (C(i) 360), k = 0, ..., L // 2,
  and corresponding wavelengths
    λ(k, i) = 1 / f(k, i).

  This choice of normalization ensures Parseval's relation for energy holds:
  Supposing f[l] are sampled values of f(ℓ), where 0 < ℓ < C (meters) is a
  coordinate on the circle. Then (C / L) is the spacing of longitudinal samples,
  whence
    ∫|f(ℓ)|² dℓ ≈ (C / L) Σₗ |f[l]|² = Σₖ S[k].

  If f has units β, then S has units of m β². For example, if f is
  `u_component_of_wind`, with units (m / s), then S has units (m³ / s²). In
  air with mass density ρ (kg / m³), this gives energy density at wavenumber k
    ρ S[k] ~ (kg / m³) (m³ / s²) = kg / s²,
  which is energy density (per unit area).
  """

  variable_name: str

  @property
  def base_variables(self) -> list[str]:
    return [self.variable_name]

  def _circumference(self, dataset: xr.Dataset) -> xr.DataArray:
    """Earth's circumference as a function of latitude."""
    circum_at_equator = 2 * np.pi * schema.EARTH_RADIUS_M
    return np.cos(dataset.latitude * np.pi / 180) * circum_at_equator

  def lon_spacing_m(self, dataset: xr.Dataset) -> xr.DataArray:
    """Spacing (meters) between longitudinal values in `dataset`."""
    diffs = dataset.longitude.diff('longitude')
    if np.max(np.abs(diffs - diffs[0])) > 1e-3:
      raise ValueError(
          f'Expected uniform longitude spacing. {dataset.longitude.values=}'
      )
    return self._circumference(dataset) * diffs[0].data / 360

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    """Computes zonal power at wavenumber and frequency."""
    spacing = self.lon_spacing_m(dataset)

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
    ).rename_dims({'longitude': 'zonal_wavenumber'})[self.variable_name]
    spectrum = spectrum.assign_coords(
        zonal_wavenumber=('zonal_wavenumber', spectrum.zonal_wavenumber.data)
    )
    base_frequency = xr.DataArray(
        np.fft.rfftfreq(len(dataset.longitude)),
        dims='zonal_wavenumber',
        coords={'zonal_wavenumber': spectrum.zonal_wavenumber},
    )
    spectrum = spectrum.assign_coords(frequency=base_frequency / spacing)
    spectrum['frequency'] = spectrum.frequency.assign_attrs(units='1 / m')

    spectrum = spectrum.assign_coords(wavelength=1 / spectrum.frequency)
    spectrum['wavelength'] = spectrum.wavelength.assign_attrs(units='m')

    # This last step ensures the sum of spectral components is equal to the
    # (discrete) integral of data around a line of latitude.
    return spectrum * self._circumference(spectrum)


def interpolate_spectral_frequencies(
    spectrum: xr.DataArray,
    wavenumber_dim: str,
    frequencies: t.Optional[t.Sequence[float]] = None,
    method: str = 'linear',
    **interp_kwargs: t.Optional[dict[str, t.Any]],
) -> xr.DataArray:
  """Interpolate frequencies in `spectrum` to common values.

  Args:
    spectrum: Data as produced by ZonalEnergySpectrum.compute.
    wavenumber_dim: Dimension that indexes wavenumber, e.g. 'zonal_wavenumber'
      if `spectrum` is produced by ZonalEnergySpectrum.
    frequencies: Optional 1-D sequence of frequencies to interpolate to. By
      default, use the most narrow range of frequencies in `spectrum`.
    method: Interpolation method passed on to DataArray.interp.
    **interp_kwargs: Additional kwargs passed on to DataArray.interp.

  Returns:
    New DataArray with dimension "frequency" replacing the "wavenumber" dim in
      `spectrum`.
  """

  if set(spectrum.frequency.dims) != set((wavenumber_dim, 'latitude')):
    raise ValueError(
        f'{spectrum.frequency.dims=} was not a permutation of '
        f'("{wavenumber_dim}", "latitude")'
    )

  if frequencies is None:
    freq_min = spectrum.frequency.max('latitude').min(wavenumber_dim).data
    freq_max = spectrum.frequency.min('latitude').max(wavenumber_dim).data
    frequencies = np.linspace(
        freq_min, freq_max, num=spectrum.sizes[wavenumber_dim]
    )
  frequencies = np.asarray(frequencies)
  if frequencies.ndim != 1:
    raise ValueError(f'Expected 1-D frequencies, found {frequencies.shape=}')

  def interp_at_one_lat(da: xr.DataArray) -> xr.DataArray:
    da = (
        da.swap_dims({wavenumber_dim: 'frequency'})
        .drop_vars(wavenumber_dim)
        .interp(frequency=frequencies, method=method, **interp_kwargs)
    )
    # Interp didn't deal well with the infinite wavelength, so just reset λ as..
    da['wavelength'] = 1 / da.frequency
    da['wavelength'] = da['wavelength'].assign_attrs(units='m')
    return da

  return spectrum.groupby('latitude').apply(interp_at_one_lat)


@dataclasses.dataclass
class AggregatePrecipitationAccumulation(DerivedVariable):
  """Compute longer aggregation periods from existing shorter accumulations.

  Note: This assumes a 6h raw time step and prediction_timedelta starting at 6h.

  Attributes:
    accumulation_hours: Hours to accumulate precipitaiton over
    raw_accumulation_name: Name of the 6hr accumulation
    lead_time_name: Name of lead_time dimension
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


# Specify dictionary of common derived variables
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
