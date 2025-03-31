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


@dataclasses.dataclass(frozen=True)
class Grid:
  """Representation of a rectalinear grid."""

  lon: np.ndarray
  lat: np.ndarray

  @classmethod
  def from_degrees(cls, lon: np.ndarray, lat: np.ndarray) -> Grid:
    return cls(np.deg2rad(lon), np.deg2rad(lat))

  @property
  def shape(self) -> tuple[int, int]:
    return (len(self.lon), len(self.lat))

  @property
  def latitude_spacing(self) -> LatitudeSpacing:
    return _determine_latitude_spacing(self.lat)

  @property
  def longitude_scheme(self) -> LongitudeScheme:
    return _determine_longitude_scheme(self.lon)

  def _to_tuple(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return tuple(self.lon.tolist()), tuple(self.lat.tolist())

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
        latitude=latitude_values(
            self.target.latitude_spacing, len(self.target.lat)
        ),
        longitude=longitude_values(
            self.target.longitude_scheme, len(self.target.lon)
        ),
    ).transpose(*(dataset.dims))


def _determine_latitude_spacing(
    lat_in_radians: np.ndarray,
) -> LatitudeSpacing:
  """Determine latitude spacing."""
  _assert_increasing(lat_in_radians)
  _assert_equal_spacing(lat_in_radians)
  eq = lambda a, b: np.allclose(a, b, atol=1e-5)
  lat_in_degrees = np.rad2deg(lat_in_radians)
  if eq(lat_in_degrees[0], -90) and eq(lat_in_degrees[-1], 90):
    return LatitudeSpacing.EQUIANGULAR_WITH_POLES
  elif eq(90 + lat_in_degrees[0], 90 - lat_in_degrees[-1]):
    return LatitudeSpacing.EQUIANGULAR_WITHOUT_POLES
  else:
    raise ValueError(f'Unknown spacing for {lat_in_degrees=}')


def _determine_longitude_scheme(
    lon_in_radians: np.ndarray,
) -> LongitudeScheme:
  """Determine longitude scheme."""
  _assert_increasing(lon_in_radians)
  _assert_equal_spacing(lon_in_radians)
  lon_in_degrees = np.rad2deg(lon_in_radians)
  eq = lambda a, b: np.allclose(a, b, atol=1e-5)
  if eq(lon_in_degrees[0], 0) and lon_in_degrees[-1] < 360:
    return LongitudeScheme.START_AT_ZERO
  elif lon_in_degrees[0] < 0 and eq(-lon_in_degrees[0], lon_in_degrees[-1]):
    return LongitudeScheme.CENTER_AT_ZERO
  else:
    raise ValueError(f'Unknown longitude scheme for {lon_in_degrees=}')


def nearest_neighbor_indices(
    source_grid: Grid, target_grid: Grid
) -> np.ndarray:
  """Returns Haversine nearest neighbor indices from source_grid to target_grid."""
  # construct a BallTree to find nearest neighbor on the surface of a sphere
  source_mesh = np.meshgrid(source_grid.lat, source_grid.lon, indexing='ij')
  target_mesh = np.meshgrid(target_grid.lat, target_grid.lon, indexing='ij')
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


class BilinearRegridder(Regridder):
  """Regrid with bilinear interpolation."""

  @functools.partial(jax.jit, static_argnums=0)
  def regrid_array(self, field: Array) -> jax.Array:
    batch_interp = jax.vmap(jnp.interp, in_axes=(0, None, None))

    # interpolate latitude
    lat_source = self.source.lat
    lat_target = self.target.lat
    lat_interp = jnp.vectorize(batch_interp, signature='(a),(b),(b)->(a)')
    field = lat_interp(lat_target, lat_source, field)

    # interpolation longitude
    lon_source = self.source.lon
    lon_target = self.target.lon
    lon_interp = jnp.vectorize(
        jax.vmap(batch_interp, in_axes=(None, None, -1), out_axes=-1),
        signature='(a),(b),(b,y)->(a,y)',
    )
    field = lon_interp(lon_target, lon_source, field)

    return field


def _assert_increasing(x: np.ndarray) -> None:
  if not (np.diff(x) > 0).all():
    raise ValueError(f'array is not increasing: {x}')


def _assert_equal_spacing(x: np.ndarray) -> None:
  diffs = np.unique(np.diff(x))
  if (diffs.max() - diffs.min()) / diffs.max() > 1e-5:
    raise ValueError(f'array does not have equal spacing. {diffs=}, {x=}')


def _latitude_cell_bounds(x: Array) -> jax.Array:
  pi_over_2 = jnp.array([np.pi / 2], dtype=x.dtype)
  return jnp.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])


def _latitude_overlap(
    source_points: Array,
    target_points: Array,
) -> jax.Array:
  """Calculate the area overlap as a function of latitude."""
  source_bounds = _latitude_cell_bounds(source_points)
  target_bounds = _latitude_cell_bounds(target_points)
  upper = jnp.minimum(
      target_bounds[1:, jnp.newaxis], source_bounds[jnp.newaxis, 1:]
  )
  lower = jnp.maximum(
      target_bounds[:-1, jnp.newaxis], source_bounds[jnp.newaxis, :-1]
  )
  # normalized cell area: integral from lower to upper of cos(latitude)
  return (upper > lower) * (jnp.sin(upper) - jnp.sin(lower))


def _conservative_latitude_weights(
    source_points: Array, target_points: Array
) -> jax.Array:
  """Create a weight matrix for conservative regridding along latitude.

  Args:
    source_points: 1D latitude coordinates in units of radians for centers of
      source cells.
    target_points: 1D latitude coordinates in units of radians for centers of
      target cells.

  Returns:
    NumPy array with shape (target, source). Rows sum to 1.
  """
  _assert_increasing(source_points)
  _assert_increasing(target_points)
  weights = _latitude_overlap(source_points, target_points)
  weights /= jnp.sum(weights, axis=1, keepdims=True)
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
  shift_down = x > target + period / 2
  shift_up = x < target - period / 2
  return x + period * shift_up - period * shift_down


def _periodic_upper_bounds(x, period):
  # Midpoint of x and roll(x, -1), unique up to multiple of 2π
  x_plus = _align_phase_with(jnp.roll(x, -1), x, period)
  return (x + x_plus) / 2


def _periodic_lower_bounds(x, period):
  # Midpoint of x and roll(x, +1), unique up to multiple of 2π
  x_minus = _align_phase_with(jnp.roll(x, +1), x, period)
  return (x_minus + x) / 2


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
    period: float = 2 * np.pi,
) -> jax.Array:
  """Calculate the area overlap as a function of latitude."""
  first_points = first_points % period
  first_upper = _periodic_upper_bounds(first_points, period)
  first_lower = _periodic_lower_bounds(first_points, period)

  second_points = second_points % period
  second_upper = _periodic_upper_bounds(second_points, period)
  second_lower = _periodic_lower_bounds(second_points, period)

  return jnp.vectorize(functools.partial(_periodic_overlap, period=period))(
      first_lower[:, jnp.newaxis],
      first_upper[:, jnp.newaxis],
      second_lower[jnp.newaxis, :],
      second_upper[jnp.newaxis, :],
  )


def _conservative_longitude_weights(
    source_points: np.ndarray, target_points: np.ndarray
) -> jax.Array:
  """Create a weight matrix for conservative regridding along longitude.

  Args:
    source_points: 1D longitude coordinates in units of radians for centers of
      source cells.
    target_points: 1D longitude coordinates in units of radians for centers of
      target cells.

  Returns:
    NumPy array with shape (new_size, old_size). Rows sum to 1.
  """
  if len(target_points) < 3:
    raise ValueError(
        'Need 3 or more target points else overlap is not well defined. Found'
        f' {len(target_points)}'
    )
  _assert_increasing(source_points)
  _assert_increasing(target_points)
  weights = _longitude_overlap(target_points, source_points)
  weights /= jnp.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (target_points.size, source_points.size)
  return weights


class ConservativeRegridder(Regridder):
  """Regrid with linear conservative regridding."""

  @functools.partial(jax.jit, static_argnums=0)
  def _mean(self, field: Array) -> jax.Array:
    """Computes cell-averages of field on the target grid."""
    lon_weights = _conservative_longitude_weights(
        self.source.lon, self.target.lon
    )
    lat_weights = _conservative_latitude_weights(
        self.source.lat, self.target.lat
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
