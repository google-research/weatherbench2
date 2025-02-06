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
import scipy.integrate
from weatherbench2 import schema
import xarray as xr


# pylint: disable=invalid-name


@dataclasses.dataclass
class DerivedVariable:
  """Derived variable base class."""

  @property
  def base_variables(self) -> list[str]:
    """Return a list of base variables."""
    return []

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    """Return core dimensions needed for computing this variable.

    Returns a tuple, where the first element is a tuple of all core dimensions
    for input (base) variables, and the second element is a tuple of all core
    dimensions on the output. For more details on the concept of "core
    dimensions", see xarray.apply_ufunc.
    """
    raise NotImplementedError

  @property
  def all_input_core_dims(self) -> set[str]:
    """The set of all input core dimensions."""
    return set().union(*self.core_dims[0])

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    """Compute derived variable, returning it in a new DataArray."""
    raise NotImplementedError


@dataclasses.dataclass
class _WindVariable(DerivedVariable):
  """Compute a variable dervied from U and V wind components.

  Attributes:
    u_name: Name of U component.
    v_name: Name of V component.
  """

  u_name: str
  v_name: str

  @property
  def base_variables(self) -> list[str]:
    return [self.u_name, self.v_name]


@dataclasses.dataclass
class WindSpeed(_WindVariable):
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

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    return ([], []), []

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    u = dataset[self.u_name]
    v = dataset[self.v_name]
    return np.sqrt(u**2 + v**2)


def _zero_poles(field: xr.Dataset, epsilon: float = 1e-6):
  cos_theta = np.cos(np.deg2rad(field.coords['latitude']))
  return field.where(cos_theta > epsilon, 0.0)


_METERS_PER_DEGREE = 2 * np.pi * schema.EARTH_RADIUS_M / 360


def _d_dx(field: xr.DataArray) -> xr.DataArray:
  latitude = field.coords['latitude']
  cos_theta = np.cos(np.deg2rad(latitude))
  # TODO(shoyer): use a custom calculation with roll() instead of
  # differentiate() to calculate rolling over 360 to 0 degrees properly.
  return _zero_poles(  # pytype: disable=bad-return-type
      field.differentiate('longitude') / cos_theta / _METERS_PER_DEGREE
  )


def _d_dy(field: xr.DataArray) -> xr.DataArray:
  return field.differentiate('latitude') / _METERS_PER_DEGREE


def _divergence(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
  return _d_dx(u) + _d_dy(v)


def _curl(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
  return _d_dx(v) - _d_dy(u)


@dataclasses.dataclass
class _3DWindVariable(DerivedVariable):
  """Compute a variable dervied from 3D U and V wind components.

  Attributes:
    u_name: Name of U component.
    v_name: Name of V component.
  """

  u_name: str = 'u_component_of_wind'
  v_name: str = 'v_component_of_wind'

  @property
  def base_variables(self) -> list[str]:
    return [self.u_name, self.v_name]


@dataclasses.dataclass
class WindDivergence(_3DWindVariable):
  """Compute wind divergence."""

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    lon_lat = ['longitude', 'latitude']
    return (lon_lat, lon_lat), lon_lat

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    u = dataset[self.u_name]
    v = dataset[self.v_name]
    return _divergence(u, v)


@dataclasses.dataclass
class WindVorticity(_3DWindVariable):
  """Compute wind vorticity."""

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    lon_lat = ['longitude', 'latitude']
    return (lon_lat, lon_lat), lon_lat

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    u = dataset[self.u_name]
    v = dataset[self.v_name]
    return _curl(u, v)


@dataclasses.dataclass
class VerticalVelocity(_3DWindVariable):
  r"""Compute vertical wind velocity, assuming the hydrostatic approximation.

  To calculate ω, we integrate the continuity equation [1] in pressure
  coordinates:
    ∇_p · u + ∂ω/∂p = 0
    ω = -∫ dp ∇_p · u

  [1] See section 8.6.1 from Durran, D. R. Numerical Methods for Fluid Dynamics:
  With Applications to Geophysics. (Springer, New York, NY, 2010).
  """

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    zxy = ['level', 'longitude', 'latitude']
    return (zxy, zxy), zxy

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    u = dataset[self.u_name]
    v = dataset[self.v_name]
    divergence = _divergence(u, v)
    pascals_per_hpa = 100
    pressure = pascals_per_hpa * dataset.coords['level']
    axis = divergence.dims.index('level')
    # TODO(shoyer): consider masking out vertical wind in locations below the
    # surface of the Earth (geopotential < geopotential_at_surface).
    vertical_wind = scipy.integrate.cumulative_trapezoid(
        -divergence.values, x=pressure, axis=axis, initial=0
    )
    return divergence.copy(data=vertical_wind)


@dataclasses.dataclass
class EddyKineticEnergy(_3DWindVariable):
  """Compute eddy kinetic energy.

  Eddies are defined as the deviation from the instantaneous zonal mean.
  """

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    return (['level', 'longitude'], ['level', 'longitude']), ['longitude']

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    u_wind = dataset[self.u_name]
    v_wind = dataset[self.v_name]
    u_delta = u_wind - u_wind.mean('longitude')
    v_delta = v_wind - v_wind.mean('longitude')
    return (1 / 2) * (u_delta**2 + v_delta**2).integrate('level')


def _geostrophic_wind(
    geopotential: xr.DataArray,
) -> t.Tuple[xr.DataArray, xr.DataArray]:
  omega = 7.292e-5  # radians / second
  coriolis_parameter = (
      2 * omega * np.sin(np.deg2rad(geopotential.coords['latitude']))
  )
  # Geostrophic wind is inf on the equator. We don't clip it to ensure that the
  # user makes an intentional choice about how handle these invalid values
  # (e.g., by evaluating over a region).
  return (
      -_d_dy(geopotential) / coriolis_parameter,
      +_d_dx(geopotential) / coriolis_parameter,
  )


@dataclasses.dataclass
class _GeostrophicWindVariable(DerivedVariable):
  """Base class for geostrophic wind variables."""

  geopotential_name: str = 'geopotential'

  @property
  def base_variables(self) -> list[str]:
    return [self.geopotential_name]

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    lon_lat = ['longitude', 'latitude']
    return (lon_lat,), lon_lat


@dataclasses.dataclass
class GeostrophicWindSpeed(_GeostrophicWindVariable):
  """Calculate geostrophic wind speed.

  Geostrophic wind is only meaningful when the Coriolis force is significant
  (small Rossby number), i.e., away from the equator, where it is undefined.

  This was suggested as a diagnostic for physical balance in AI weather models
  by Massimo Bonavita in https://arxiv.org/abs/2309.08473
  """

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    u, v = _geostrophic_wind(dataset[self.geopotential_name])
    return np.sqrt(u**2 + v**2)


class UComponentOfGeostrophicWind(_GeostrophicWindVariable):
  """East-west component of geostrophic wind."""

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    u, _ = _geostrophic_wind(dataset[self.geopotential_name])
    return u


class VComponentOfGeostrophicWind(_GeostrophicWindVariable):
  """North-south component of geostrophic wind."""

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    _, v = _geostrophic_wind(dataset[self.geopotential_name])
    return v


@dataclasses.dataclass
class _AgeostrophicWindVariable(DerivedVariable):
  """Base class for ageostrophic wind variables."""

  u_name: str = 'u_component_of_wind'
  v_name: str = 'v_component_of_wind'
  geopotential_name: str = 'geopotential'

  @property
  def base_variables(self) -> list[str]:
    return [self.u_name, self.v_name, self.geopotential_name]

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    lon_lat = ['longitude', 'latitude']
    return (lon_lat, lon_lat, lon_lat), lon_lat


class AgeostrophicWindSpeed(_AgeostrophicWindVariable):
  """Calculate ageostrophic wind speed."""

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    u = dataset[self.u_name]
    v = dataset[self.v_name]
    u_geo, v_geo = _geostrophic_wind(dataset[self.geopotential_name])
    return np.sqrt((u - u_geo) ** 2 + (v - v_geo) ** 2)


class UComponentOfAgeostrophicWind(_AgeostrophicWindVariable):
  """East-west component of ageostrophic wind."""

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    u = dataset[self.u_name]
    u_geo, _ = _geostrophic_wind(dataset[self.geopotential_name])
    return u - u_geo


class VComponentOfAgeostrophicWind(_AgeostrophicWindVariable):
  """North-south component of ageostrophic wind."""

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    v = dataset[self.v_name]
    _, v_geo = _geostrophic_wind(dataset[self.geopotential_name])
    return v - v_geo


@dataclasses.dataclass
class LapseRate(DerivedVariable):
  """Compute lapse rate in temperature."""

  temperature_name: str = 'temperature'
  geopotential_name: str = 'geopotential'

  @property
  def base_variables(self) -> list[str]:
    return [self.temperature_name, self.geopotential_name]

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    return (['level'], ['level']), ['level']

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    g = 9.81
    temperature = dataset[self.temperature_name]
    geopotential = dataset[self.geopotential_name]
    dT_dp = temperature.differentiate('level')
    dz_dp = (1 / g) * geopotential.differentiate('level')
    return dT_dp / dz_dp


@dataclasses.dataclass
class TotalColumnWater(DerivedVariable):
  """Compute total column water.

  Attributes:
    water_species_name: Name of water species to vertically integrate.
  """

  water_species_name: str = 'specific_humidity'

  @property
  def base_variables(self) -> list[str]:
    return [self.water_species_name]

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    return (['level'],), []

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    g = 9.81
    return 1 / g * dataset[self.water_species_name].integrate('level')


@dataclasses.dataclass
class IntegratedWaterTransport(DerivedVariable):
  """Compute integrated horizontal water transport in a vertical column.

  Integrated vapor transport (IVT) is a useful diagnostic to include for
  understanding atmospheric rviers. Default pressure levels to include are taken
  from the GraphCast paper.

  Attributes:
    u_name: Name of wind U component.
    v_name: Name of wind V component.
    water_species_name: Name of water species to vertically integrate.
    level_min: Minimum pressure level to include.
    level_max: Maximum pressure level to include.
  """

  u_name: str = 'u_component_of_wind'
  v_name: str = 'v_component_of_wind'
  water_species_name: str = 'specific_humidity'
  level_min: t.Optional[float] = 300
  level_max: t.Optional[float] = 1000

  @property
  def base_variables(self) -> list[str]:
    return [self.u_name, self.v_name, self.water_species_name]

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    return (['level'], ['level']), []

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    g = 9.81
    u_integral = (
        (dataset[self.water_species_name] * dataset[self.u_name])
        .sel(level=slice(self.level_min, self.level_max))
        .integrate('level')
    )
    v_integral = (
        (dataset[self.water_species_name] * dataset[self.v_name])
        .sel(level=slice(self.level_min, self.level_max))
        .integrate('level')
    )
    return (1 / g) * np.sqrt(u_integral**2 + v_integral**2)


@dataclasses.dataclass
class RelativeHumidity(DerivedVariable):
  """Calculate relativity humidity from specific humidity."""

  temperature_name: str = 'temperature'
  specific_humidity_name: str = 'specific_humidity'
  pressure_name: str = 'level'

  @property
  def base_variables(self) -> list[str]:
    return [
        self.temperature_name,
        self.specific_humidity_name,
        self.pressure_name,
    ]

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    return ([], []), []

  def compute(self, dataset: xr.Dataset) -> xr.DataArray:
    # We use the same formula as MetPy's
    # relative_humidity_from_specific_humidity.
    #
    # For saturation vapor pressure, we use the formula for Bolton 1980
    # (https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2) for T
    # in degrees Celsius:
    #   6.112 e^\frac{17.67T}{T + 243.5}
    # We assume pressure has units of hPa and temperature has units of Kelvin.
    temperature = dataset[self.temperature_name]
    specific_humidity = dataset[self.specific_humidity_name]
    pressure = dataset.coords[self.pressure_name]
    svp = 6.112 * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))
    mixing_ratio = specific_humidity / (1 - specific_humidity)
    saturation_mixing_ratio = 0.622 * svp / (pressure - svp)
    return mixing_ratio / saturation_mixing_ratio


@dataclasses.dataclass
class PrecipitationAccumulation(DerivedVariable):
  """Compute precipitation accumulation from hourly accumulations.

  Accumulation is computed for the time period leading up to and including the
  lead_time.  E.g. 24h accumulation at lead_time=24h indicates accumulation
  from lead_time=0 to lead_time=24. This is equal to the values of
  `total_precipitation_name` at 24, minus the value at 0.

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

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    return ([self.lead_time_name],), [self.lead_time_name]

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
        [tp.isel({self.lead_time_name: [0]}) * np.nan, accumulation],
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

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    return (['longitude'],), ['zonal_wavenumber']

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
        da.squeeze('latitude')
        .swap_dims(
            {wavenumber_dim: 'frequency'}
        )  # pytype: disable=wrong-arg-types
        .drop_vars(wavenumber_dim)
        .interp(frequency=frequencies, method=method, **interp_kwargs)
    )
    # Interp didn't deal well with the infinite wavelength, so just reset λ as..
    da['wavelength'] = 1 / da.frequency
    da['wavelength'] = da['wavelength'].assign_attrs(units='m')
    return da

  return spectrum.groupby('latitude', squeeze=False).apply(interp_at_one_lat)


@dataclasses.dataclass
class AggregatePrecipitationAccumulation(DerivedVariable):
  """Compute longer aggregation periods from existing shorter accumulations.

  Attributes:
    accumulation_hours: Hours to accumulate precipitaiton over
    raw_accumulation_name: Name of the  accumulation
    raw_accumulation_hours: Hours of the raw accumulation
    lead_time_name: Name of lead_time dimension
  """

  accumulation_hours: int
  raw_accumulation_name: str = 'total_precipitation_6hr'
  raw_accumulation_hours: int = 6
  lead_time_name: str = 'prediction_timedelta'

  @property
  def base_variables(self):
    return [self.raw_accumulation_name]

  @property
  def core_dims(self) -> t.Tuple[t.Tuple[t.List[str], ...], t.List[str]]:
    return ([self.lead_time_name],), [self.lead_time_name]

  def compute(self, dataset: xr.Dataset):
    tp6h = dataset[self.raw_accumulation_name]

    # Compute aggregation steps
    steps = float(
        np.timedelta64(self.accumulation_hours, 'h')
        / np.timedelta64(self.raw_accumulation_hours, 'h')
    )
    assert steps.is_integer(), 'Accumulation time must be multiple of timestep.'
    # Compute accumulation
    accumulation = tp6h.rolling({self.lead_time_name: int(steps)}).sum()
    return accumulation


# Specify dictionary of common derived variables
DERIVED_VARIABLE_DICT = {
    'wind_speed': WindSpeed(
        u_name='u_component_of_wind', v_name='v_component_of_wind'
    ),
    '10m_wind_speed': WindSpeed(
        u_name='10m_u_component_of_wind', v_name='10m_v_component_of_wind'
    ),
    'divergence': WindDivergence(),
    'vorticity': WindVorticity(),
    'vertical_velocity': VerticalVelocity(),
    'eddy_kinetic_energy': EddyKineticEnergy(),
    'geostrophic_wind_speed': GeostrophicWindSpeed(),
    'u_component_of_geostrophic_wind': UComponentOfGeostrophicWind(),
    'v_component_of_geostrophic_wind': VComponentOfGeostrophicWind(),
    'ageostrophic_wind_speed': AgeostrophicWindSpeed(),
    'u_component_of_ageostrophic_wind': UComponentOfAgeostrophicWind(),
    'v_component_of_ageostrophic_wind': VComponentOfAgeostrophicWind(),
    'lapse_rate': LapseRate(),
    'total_column_vapor': TotalColumnWater(
        water_species_name='specific_humidity'
    ),
    'total_column_liquid': TotalColumnWater(
        water_species_name='specific_cloud_liquid_water_content'
    ),
    'total_column_ice': TotalColumnWater(
        water_species_name='specific_cloud_ice_water_content'
    ),
    'integrated_vapor_transport': IntegratedWaterTransport(),
    'relative_humidity': RelativeHumidity(),
    'total_precipitation_6hr': PrecipitationAccumulation(
        total_precipitation_name='total_precipitation',
        accumulation_hours=6,
        lead_time_name='prediction_timedelta',
    ),
    'total_precipitation_24hr': PrecipitationAccumulation(
        total_precipitation_name='total_precipitation',
        accumulation_hours=24,
        lead_time_name='prediction_timedelta',
    ),
    'total_precipitation_24hr_from_6hr': AggregatePrecipitationAccumulation(
        accumulation_hours=24,
        lead_time_name='prediction_timedelta',
    ),
    'total_precipitation_24hr_from_12hr': AggregatePrecipitationAccumulation(
        accumulation_hours=24,
        lead_time_name='prediction_timedelta',
        raw_accumulation_name='total_precipitation_12hr',
        raw_accumulation_hours=12,
    ),
}
