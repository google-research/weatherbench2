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
"""Routines for horizontal regridding.

This module supports three types of regridding:
- Nearest neighbor: suitable for interpolating non-continuous fields (e.g.,
  categrorical land-surface type).
- Bilinear interpolation: most suitable for regridding to finer grids.
- Linear conservative regridding: most suitable for regridding to coarser grids.

Only rectalinear grids (one dimensional lat/lon coordinates) are supported, but
irregular spacing is OK.

Conservative regridding schemes are adapted from:
https://gist.github.com/shoyer/c0f1ddf409667650a076c058f9a17276
"""
from __future__ import annotations

import dataclasses
import enum
import functools
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from sklearn import neighbors
import xarray as xr

Array = Union[np.ndarray, jax.Array]


class LongitudeScheme(enum.Enum):
  # [0, Δ, 2Δ, ..., 360 - Δ]
  START_AT_ZERO = enum.auto()

  # [-180 + Δ/2, ..., 180 - Δ/2]
  CENTER_AT_ZERO = enum.auto()


class LatitudeSpacing(enum.Enum):
  EQUIANGULAR_WITH_POLES = enum.auto()
  EQUIANGULAR_WITHOUT_POLES = enum.auto()
  CUSTOM = enum.auto()  # custom spacing, e.g., Gaussian grids


def latitude_values(latitude_spacing: LatitudeSpacing, num: int) -> np.ndarray:
  """Latitude node values given spacing and number of nodes."""
  if latitude_spacing == LatitudeSpacing.EQUIANGULAR_WITH_POLES:
    lat_start = -90
    lat_stop = 90
  elif latitude_spacing == LatitudeSpacing.EQUIANGULAR_WITHOUT_POLES:
    lat_start = -90 + 0.5 * 180 / num
    lat_stop = 90 - 0.5 * 180 / num
  else:
    raise ValueError(f'Unhandled {latitude_spacing=}')
  return np.linspace(lat_start, lat_stop, num=num)


def longitude_values(longitude_scheme: LongitudeScheme, num: int) -> np.ndarray:
  """Longitude node values given scheme and number of nodes."""
  lon_delta = 360 / num
  if longitude_scheme == LongitudeScheme.START_AT_ZERO:
    lon_start = 0
    lon_stop = 360 - lon_delta
  elif longitude_scheme == LongitudeScheme.CENTER_AT_ZERO:
    lon_start = -180 + lon_delta / 2
    lon_stop = 180 - lon_delta / 2
  else:
    raise ValueError(f'Unhandled {longitude_scheme=}')
  return np.linspace(lon_start, lon_stop, num=num)


# pylint: disable=g-missing-property-docstring


def _check_global_coverage(
    longitudes: np.ndarray, latitudes: np.ndarray, tolerance: float
):
  """Check that the grid covers the entire globe."""
  min_lat = float(latitudes.min())
  max_lat = float(latitudes.max())
  min_lon = float(longitudes.min())
  max_lon = float(longitudes.max())

  if not abs(min_lat + 90) < tolerance:
    raise ValueError(
        f'min latitude must be within ±{tolerance} of -90, found {min_lat}'
    )
  if not abs(max_lat - 90) < tolerance:
    raise ValueError(
        f'max latitude must be within ±{tolerance} of 90, found {max_lat}'
    )
  if not (abs(min_lon - 0) < tolerance or abs(min_lon + 180) < tolerance):
    raise ValueError(
        f'min longitude must be within ±{tolerance} of 0 or -180, found'
        f' {min_lon}'
    )
  if not (abs(max_lon - 360) < tolerance or abs(max_lon - 180) < tolerance):
    raise ValueError(
        f'max longitude must be within ±{tolerance} of 360 or +180, found'
        f' {max_lon}'
    )


@dataclasses.dataclass(frozen=True)
class Grid:
  """Representation of a rectalinear grid.

  Attributes:
    longitudes: 1D array of longitude coordinates in degrees, from roughly 0 to
      360, or -180 to 180.
    latitudes: 1D array of latitude coordinates in degrees, from roughly -90 to
      90, or 90 to -90.
    periodic: if True, longitude coordinates are assumed to be periodic, i.e.,
      the difference between the first and last longitudes is 360 degrees.
    includes_poles: if True, the grid is assumed to cover the North and South
      poles.
  """

  longitudes: np.ndarray = dataclasses.field(kw_only=True)
  latitudes: np.ndarray = dataclasses.field(kw_only=True)
  periodic: bool = dataclasses.field(kw_only=True)
  includes_poles: bool = dataclasses.field(kw_only=True)

  @property
  def lat(self):
    raise AttributeError(
        'lat/lon attributes (in radians) is no longer supported. '
        'Use latitude/longitude (in degrees) instead'
    )

  @property
  def lon(self):
    raise AttributeError(
        'lat/lon attributes (in radians) is no longer supported. '
        'Use latitude/longitude (in degrees) instead'
    )

  @classmethod
  def from_degrees(cls, lon: np.ndarray, lat: np.ndarray) -> Grid:
    """Legacy constructor."""
    return cls(
        longitudes=lon, latitudes=lat, periodic=True, includes_poles=True
    )

  @property
  def shape(self) -> tuple[int, int]:
    return (len(self.longitudes), len(self.latitudes))

  def _to_tuple(
      self,
  ) -> tuple[tuple[float, ...], tuple[float, ...], bool, bool]:
    return (
        tuple(self.longitudes.tolist()),
        tuple(self.latitudes.tolist()),
        self.periodic,
        self.includes_poles,
    )

  def __eq__(self, other):  # needed for hashability
    return isinstance(other, Grid) and self._to_tuple() == other._to_tuple()

  def __hash__(self):
    return hash(self._to_tuple())


@dataclasses.dataclass(frozen=True)
class Regridder:
  """Base class for regridding."""

  source: Grid
  target: Grid

  def regrid_array(self, field: Array) -> jax.Array:
    """Regrid an array with dimensions (..., lon, lat) from source to target."""
    raise NotImplementedError

  def regrid_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
    """Regrid an xr.Dataset from source to target."""
    if not (dataset['latitude'].diff('latitude') > 0).all():
      # ensure latitude is increasing
      dataset = dataset.isel(latitude=slice(None, None, -1))  # reverse
    assert (dataset['latitude'].diff('latitude') > 0).all()
    regridded = xr.apply_ufunc(
        self.regrid_array,
        dataset,
        input_core_dims=[['longitude', 'latitude']],
        output_core_dims=[['longitude', 'latitude']],
        exclude_dims={'longitude', 'latitude'},
        vectorize=True,  # loop over level & time, for lower memory usage
    )
    return regridded.assign_coords(
        latitude=self.target.latitudes, longitude=self.target.longitudes
    ).transpose(*(dataset.dims))


def nearest_neighbor_indices(
    source_grid: Grid, target_grid: Grid
) -> np.ndarray:
  """Returns Haversine nearest neighbor indices from source_grid to target_grid."""
  # construct a BallTree to find nearest neighbor on the surface of a sphere
  source_lat_rad = np.deg2rad(source_grid.latitudes)
  source_lon_rad = np.deg2rad(source_grid.longitudes)
  target_lat_rad = np.deg2rad(target_grid.latitudes)
  target_lon_rad = np.deg2rad(target_grid.longitudes)
  source_mesh = np.meshgrid(source_lat_rad, source_lon_rad)
  target_mesh = np.meshgrid(target_lat_rad, target_lon_rad)
  index_coords = np.stack([x.ravel() for x in source_mesh], axis=-1)
  query_coords = np.stack([x.ravel() for x in target_mesh], axis=-1)
  tree = neighbors.BallTree(index_coords, metric='haversine')
  indices = tree.query(query_coords, return_distance=False).squeeze(axis=-1)
  return indices


class NearestRegridder(Regridder):
  """Regrid with nearest neighbor interpolation."""

  @functools.cached_property
  def indices(self):
    """The interpolation indices associated with source_grid."""
    return nearest_neighbor_indices(self.source, self.target)

  def _nearest_neighbor_2d(self, array: Array) -> jax.Array:
    """2d nearest neighbor interpolation using BallTree."""
    if array.shape != self.source.shape:
      raise ValueError(f'expected {array.shape=} to match {self.source.shape=}')
    array = array.ravel().take(self.indices)
    return array.reshape(self.target.shape)

  @functools.partial(jax.jit, static_argnums=0)
  def regrid_array(self, field: Array) -> jax.Array:
    interp = jnp.vectorize(self._nearest_neighbor_2d, signature='(a,b)->(c,d)')
    return interp(field)


_interp_without_extrapolation = functools.partial(
    jnp.interp, left=jnp.nan, right=jnp.nan
)


class BilinearRegridder(Regridder):
  """Regrid with bilinear interpolation."""

  @functools.partial(jax.jit, static_argnums=0)
  def regrid_array(self, field: Array) -> jax.Array:

    # interpolate latitude
    lat_source = self.source.latitudes
    lat_target = self.target.latitudes
    lat_interp = (
        jnp.interp
        if self.source.includes_poles
        else _interp_without_extrapolation
    )
    vec_lat_interp = jnp.vectorize(
        jax.vmap(lat_interp, in_axes=(0, None, None)),
        signature='(a),(b),(b)->(a)',
    )
    field = vec_lat_interp(lat_target, lat_source, field)

    # interpolation longitude
    lon_source = self.source.longitudes
    lon_target = self.target.longitudes
    lon_interp = (
        functools.partial(jnp.interp, period=360)
        if self.source.periodic
        else _interp_without_extrapolation
    )
    vec_lon_interp = jnp.vectorize(
        jax.vmap(
            jax.vmap(lon_interp, in_axes=(0, None, None)),
            in_axes=(None, None, -1),
            out_axes=-1,
        ),
        signature='(a),(b),(b,y)->(a,y)',
    )
    field = vec_lon_interp(lon_target, lon_source, field)

    return field


def _assert_increasing(x: np.ndarray) -> None:
  if not (np.diff(x) > 0).all():
    raise ValueError(f'array is not increasing: {x}')


def _latitude_cell_bounds(x: Array, include_poles: bool = True) -> jax.Array:
  if include_poles:
    initial = jnp.array([-90])
    final = jnp.array([90])
  else:
    initial = x[:1] - (x[1] - x[0]) / 2
    final = x[-1:] + (x[-1] - x[-2]) / 2
  return jnp.concatenate([initial, (x[:-1] + x[1:]) / 2, final])


def _latitude_area_from_bounds(lower: Array, upper: Array) -> jax.Array:
  # normalized cell area: integral from lower to upper of cos(latitude)
  return jnp.sin(jnp.deg2rad(upper)) - jnp.sin(jnp.deg2rad(lower))


def _latitude_area(points: Array, include_poles: bool) -> jax.Array:
  """Calculate the relative area of square cells along latitude."""
  bounds = _latitude_cell_bounds(points, include_poles)
  return _latitude_area_from_bounds(bounds[:-1], bounds[1:])


def _latitude_overlap(
    source_points: Array,
    target_points: Array,
    source_includes_poles: bool,
    target_includes_poles: bool,
) -> jax.Array:
  """Calculate the area overlap as a function of latitude."""
  source_bounds = _latitude_cell_bounds(source_points, source_includes_poles)
  target_bounds = _latitude_cell_bounds(target_points, target_includes_poles)
  upper = jnp.minimum(
      target_bounds[1:, jnp.newaxis], source_bounds[jnp.newaxis, 1:]
  )
  lower = jnp.maximum(
      target_bounds[:-1, jnp.newaxis], source_bounds[jnp.newaxis, :-1]
  )
  return (upper > lower) * _latitude_area_from_bounds(lower, upper)


def _conservative_latitude_weights(
    source_points: Array,
    target_points: Array,
    source_includes_poles: bool,
    target_includes_poles: bool,
) -> jax.Array:
  """Create a weight matrix for conservative regridding along latitude.

  Args:
    source_points: 1D latitude coordinates in degrees for centers of source
      cells.
    target_points: 1D latitude coordinates in degrees for centers of target
      cells.
    source_includes_poles: if True, the source grid includes the poles.
    target_includes_poles: if True, the target grid includes the poles.

  Returns:
    NumPy array with shape (target, source). Rows sum to 1.
  """
  _assert_increasing(source_points)
  _assert_increasing(target_points)
  overlap = _latitude_overlap(
      source_points, target_points, source_includes_poles, target_includes_poles
  )
  coverage = jnp.sum(overlap, axis=1, keepdims=True)
  weights = overlap / coverage
  if not source_includes_poles:
    target_areas = _latitude_area(target_points, target_includes_poles)
    target_areas = target_areas[:, jnp.newaxis]
    is_covered = jnp.isclose(coverage, target_areas, rtol=1e-3)
    weights = jnp.where(is_covered, weights, jnp.nan)
  assert weights.shape == (target_points.size, source_points.size)
  return weights


def _align_phase_with(x, target, period):
  """Align the phase of a periodic number to match another.

  The returned number is equivalent to the original (modulo the period) with
  the smallest distance from the target, among the values
  `{x - period, x, x + period}`.

  Args:
    x: number to adjust.
    target: number with phase to match.
    period: periodicity.

  Returns:
    x possibly shifted up or down by `period`.
  """
  if period is None:
    return x
  shift_down = x > target + period / 2
  shift_up = x < target - period / 2
  return x + period * shift_up - period * shift_down


def _periodic_upper_bounds(x, period):
  if period is None:
    # x right shifted with extrapolation
    x_plus = jnp.concatenate([x[1:], x[-1:] + (x[-1] - x[-2])])
  else:
    # Midpoint of x and roll(x, -1), unique up to multiple of period
    x_plus = _align_phase_with(jnp.roll(x, -1), x, period)
  return (x + x_plus) / 2


def _periodic_lower_bounds(x, period):
  if period is None:
    # x left shifted with extrapolation
    x_minus = jnp.concatenate([x[:1] - (x[1] - x[0]), x[:-1]])
  else:
    # Midpoint of x and roll(x, +1), unique up to multiple of period
    x_minus = _align_phase_with(jnp.roll(x, +1), x, period)
  return (x_minus + x) / 2


def _periodic_upper_lower_bounds(x, period):
  if period is not None:
    x = x % period
  x_upper = _periodic_upper_bounds(x, period)
  x_lower = _periodic_lower_bounds(x, period)
  return x_upper, x_lower


def _longitude_length(points: Array, periodic: bool) -> jax.Array:
  """Calculate cell lengths in degrees."""
  upper, lower = _periodic_upper_lower_bounds(points, 360 if periodic else None)
  return upper - lower


def _periodic_overlap(x0, x1, y0, y1, period):
  # valid as long as no intervals are larger than period/2
  y0 = _align_phase_with(y0, x0, period)
  y1 = _align_phase_with(y1, x0, period)
  upper = jnp.minimum(x1, y1)
  lower = jnp.maximum(x0, y0)
  return jnp.maximum(upper - lower, 0)


def _longitude_overlap(
    first_points: Array,
    second_points: Array,
    first_periodic: bool,
    second_periodic: bool,
) -> jax.Array:
  """Calculate the area overlap as a function of latitude."""
  first_upper, first_lower = _periodic_upper_lower_bounds(
      first_points, 360 if first_periodic else None
  )
  second_upper, second_lower = _periodic_upper_lower_bounds(
      second_points, 360 if second_periodic else None
  )
  return jnp.vectorize(functools.partial(_periodic_overlap, period=360))(
      first_lower[:, jnp.newaxis],
      first_upper[:, jnp.newaxis],
      second_lower[jnp.newaxis, :],
      second_upper[jnp.newaxis, :],
  )


def _conservative_longitude_weights(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_periodic: bool,
    target_periodic: bool,
) -> jax.Array:
  """Create a weight matrix for conservative regridding along longitude.

  Args:
    source_points: 1D longitude coordinates in degrees for centers of source
      cells.
    target_points: 1D longitude coordinates in degrees for centers of target
      cells.
    source_periodic: if True, the source grid is periodic.
    target_periodic: if True, the target grid is periodic.

  Returns:
    NumPy array with shape (new_size, old_size). Rows sum to 1.
  """
  if len(target_points) < 3 and target_periodic:
    raise ValueError(
        'Need 3 or more target points else overlap is not well defined. Found'
        f' {len(target_points)}'
    )
  _assert_increasing(source_points)
  _assert_increasing(target_points)
  overlap = _longitude_overlap(
      target_points, source_points, target_periodic, source_periodic
  )
  coverage = jnp.sum(overlap, axis=1, keepdims=True)
  weights = overlap / coverage
  if not source_periodic:
    target_lengths = _longitude_length(target_points, target_periodic)
    target_lengths = target_lengths[:, jnp.newaxis]
    is_covered = jnp.isclose(coverage, target_lengths, rtol=1e-3)
    weights = jnp.where(is_covered, weights, jnp.nan)
  assert weights.shape == (target_points.size, source_points.size)
  return weights


class ConservativeRegridder(Regridder):
  """Regrid with linear conservative regridding."""

  @functools.partial(jax.jit, static_argnums=0)
  def _mean(self, field: Array) -> jax.Array:
    """Computes cell-averages of field on the target grid."""
    lon_weights = _conservative_longitude_weights(
        self.source.longitudes,
        self.target.longitudes,
        self.source.periodic,
        self.target.periodic,
    )
    lat_weights = _conservative_latitude_weights(
        self.source.latitudes,
        self.target.latitudes,
        self.source.includes_poles,
        self.target.includes_poles,
    )
    return jnp.einsum(
        'ab,cd,...bd->...ac',
        lon_weights,
        lat_weights,
        field,
        precision='highest',
    )

  @functools.partial(jax.jit, static_argnums=0)
  def _nanmean(self, field: Array) -> jax.Array:
    """Compute cell-averages skipping NaNs like np.nanmean."""
    nulls = jnp.isnan(field)
    total = self._mean(jnp.where(nulls, 0, field))
    count = self._mean(jnp.logical_not(nulls))
    return total / count  # intentionally NaN if count == 0

  regrid_array = _nanmean
