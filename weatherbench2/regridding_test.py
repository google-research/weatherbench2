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
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from weatherbench2 import regridding
import xarray as xr

LatitudeSpacing = regridding.LatitudeSpacing
LongitudeScheme = regridding.LongitudeScheme


def _maybe_roll_longitude(
    ds: xr.Dataset, longitude_scheme: LongitudeScheme
) -> xr.Dataset:
  """Rolls longitude to approximate longitude_scheme."""
  ds = ds.copy()
  current = regridding._determine_longitude_scheme(
      np.deg2rad(ds.longitude.data),
  )
  if current == longitude_scheme:
    return ds

  if (
      current == LongitudeScheme.START_AT_ZERO
      and longitude_scheme == LongitudeScheme.CENTER_AT_ZERO
  ):
    mid = ds.sizes['longitude'] // 2
    ds['longitude'] = xr.where(
        ds.longitude < ds.longitude[mid], ds.longitude, -(360 - ds.longitude)
    )
    return ds.roll(longitude=-mid, roll_coords=True)
  elif (
      current == LongitudeScheme.CENTER_AT_ZERO
      and longitude_scheme == LongitudeScheme.START_AT_ZERO
  ):
    mid = ds.sizes['longitude'] // 2
    ds['longitude'] = xr.where(
        ds.longitude >= 0, ds.longitude, 180 - ds.longitude
    )
    return ds.roll(longitude=mid, roll_coords=True)
  else:
    raise ValueError(
        f'Unhandled combination {current=} and {longitude_scheme=}'
    )


class RegriddingTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          regridder_cls=regridding.ConservativeRegridder,
          source_lon_scheme=LongitudeScheme.CENTER_AT_ZERO,
          target_lon_scheme=LongitudeScheme.CENTER_AT_ZERO,
      ),
      dict(
          regridder_cls=regridding.ConservativeRegridder,
          source_lon_scheme=LongitudeScheme.START_AT_ZERO,
          target_lon_scheme=LongitudeScheme.START_AT_ZERO,
      ),
      dict(
          regridder_cls=regridding.ConservativeRegridder,
          source_lat_spacing=LatitudeSpacing.EQUIANGULAR_WITHOUT_POLES,
          target_lat_spacing=LatitudeSpacing.EQUIANGULAR_WITHOUT_POLES,
      ),
      dict(
          regridder_cls=regridding.ConservativeRegridder,
          source_lon_scheme=LongitudeScheme.START_AT_ZERO,
          target_lon_scheme=LongitudeScheme.CENTER_AT_ZERO,
      ),
      dict(
          regridder_cls=regridding.ConservativeRegridder,
          source_lon_scheme=LongitudeScheme.CENTER_AT_ZERO,
          target_lon_scheme=LongitudeScheme.START_AT_ZERO,
      ),
      dict(
          regridder_cls=regridding.ConservativeRegridder,
          source_lon_scheme=LongitudeScheme.CENTER_AT_ZERO,
          target_lon_scheme=LongitudeScheme.START_AT_ZERO,
          source_lat_spacing=LatitudeSpacing.EQUIANGULAR_WITHOUT_POLES,
          target_lat_spacing=LatitudeSpacing.EQUIANGULAR_WITH_POLES,
      ),
      dict(
          regridder_cls=regridding.BilinearRegridder,
          source_lon_scheme=LongitudeScheme.START_AT_ZERO,
          target_lon_scheme=LongitudeScheme.CENTER_AT_ZERO,
      ),
      dict(
          regridder_cls=regridding.BilinearRegridder,
          source_lon_scheme=LongitudeScheme.CENTER_AT_ZERO,
          target_lon_scheme=LongitudeScheme.START_AT_ZERO,
      ),
      dict(
          source_lon_scheme=LongitudeScheme.START_AT_ZERO,
          target_lon_scheme=LongitudeScheme.CENTER_AT_ZERO,
      ),
      dict(
          regridder_cls=regridding.NearestRegridder,
          source_lon_scheme=LongitudeScheme.CENTER_AT_ZERO,
          target_lon_scheme=LongitudeScheme.START_AT_ZERO,
          source_lat_spacing=LatitudeSpacing.EQUIANGULAR_WITHOUT_POLES,
          target_lat_spacing=LatitudeSpacing.EQUIANGULAR_WITH_POLES,
      ),
  )
  def test_coarse_grid_interpolates(
      self,
      regridder_cls: regridding.Regridder = regridding.ConservativeRegridder,
      source_lat_spacing: LatitudeSpacing = LatitudeSpacing.EQUIANGULAR_WITH_POLES,
      source_lon_scheme: LongitudeScheme = LongitudeScheme.START_AT_ZERO,
      target_lat_spacing: LatitudeSpacing = LatitudeSpacing.EQUIANGULAR_WITH_POLES,
      target_lon_scheme: LongitudeScheme = LongitudeScheme.START_AT_ZERO,
  ):
    # Make source...
    #  Continuous and periodic, so interpolation should work well.
    #  Not symmetric w.r.t. CW, CCW, to test orientation
    #  Have dims that are powers of 2 for easy lower/upper bound computation
    #  below.
    n_lats = 32
    n_lons = 64
    source = xr.Dataset(
        {
            'X': xr.DataArray(
                np.zeros((1, n_lats, n_lons)),
                dims=('time', 'latitude', 'longitude'),
                name='X',
            )
        },
        coords=dict(
            time=[np.datetime64('2000-01-01T00')],
            latitude=regridding.latitude_values(source_lat_spacing, n_lats),
            longitude=regridding.longitude_values(source_lon_scheme, n_lons),
        ),
    )
    theta = 2 * np.pi * source.longitude / 360
    phi = 2 * np.pi * (source.latitude - 90) / 180
    source += np.sin(phi) * (np.cos(theta) ** 2 + np.sin(theta))

    # Test the "determine" functions.
    self.assertEqual(
        regridding._determine_latitude_spacing(np.deg2rad(source.latitude)),
        source_lat_spacing,
    )
    self.assertEqual(
        regridding._determine_longitude_scheme(np.deg2rad(source.longitude)),
        source_lon_scheme,
    )

    reduce_factor = 2

    # Get approximate target values just by choosing nearest values in source.
    regridded_latitude = regridding.latitude_values(
        target_lat_spacing,
        source.sizes['latitude'] // reduce_factor,
    )
    regridded_longitude = regridding.longitude_values(
        target_lon_scheme,
        source.sizes['longitude'] // reduce_factor,
    )

    # E.g. if reduce_factor=2, [0, 1] -> [0, 0, 1, 1]
    repeat = lambda x: np.repeat(x[:, np.newaxis], reduce_factor)
    repeat_lon = xr.DataArray(
        repeat(regridded_longitude), dims=['longitude'], name='longitude'
    )
    repeat_lat = xr.DataArray(
        repeat(regridded_latitude), dims=['latitude'], name='latitude'
    )

    lower_bound = (
        _maybe_roll_longitude(source, target_lon_scheme)
        .groupby(repeat_lon, restore_coord_dims=True)
        .min()
        .groupby(repeat_lat, restore_coord_dims=True)
        .min()
    )
    upper_bound = (
        _maybe_roll_longitude(source, target_lon_scheme)
        .groupby(repeat_lon, restore_coord_dims=True)
        .max()
        .groupby(repeat_lat, restore_coord_dims=True)
        .max()
    )

    source_grid = regridding.Grid.from_degrees(
        lon=source.longitude.data,
        lat=source.latitude.data,
    )
    target_grid = regridding.Grid.from_degrees(
        lon=regridded_longitude,
        lat=regridded_latitude,
    )
    regridder = regridder_cls(source_grid, target_grid)

    regridded_ds = regridder.regrid_dataset(source)

    np.testing.assert_equal(regridded_latitude, regridded_ds.latitude)
    np.testing.assert_equal(regridded_longitude, regridded_ds.longitude)
    self.assertTrue(np.all(np.isfinite(regridded_ds.X)))

    if source_lon_scheme == target_lon_scheme:
      min_frac_obeying_bounds = 0.99
    else:
      min_frac_obeying_bounds = 0.5

    # The regridded is always between the lower and upper bounds.
    # This also checks that the indices align, and in particular that they are
    # as expected.
    self.assertGreaterEqual(
        (lower_bound <= regridded_ds).mean().X, min_frac_obeying_bounds
    )
    self.assertGreaterEqual(
        (regridded_ds <= upper_bound).mean().X, min_frac_obeying_bounds
    )

  def test_conservative_latitude_weights(self):
    source_lat = np.pi / 180 * np.array([-75, -45, -15, 15, 45, 75])
    target_lat = np.pi / 180 * np.array([-45, 45])
    # from Wolfram alpha:
    # integral of cos(x) from 0*pi/6 to 1*pi/6 -> 0.5
    # integral of cos(x) from 1*pi/6 to 2*pi/6 -> (sqrt(3) - 1) / 2
    # integral of cos(x) from 2*pi/6 to 3*pi/6 -> 1 - sqrt(3) / 2
    expected = np.array(
        [
            [1 - np.sqrt(3) / 2, (np.sqrt(3) - 1) / 2, 1 / 2, 0, 0, 0],
            [0, 0, 0, 1 / 2, (np.sqrt(3) - 1) / 2, 1 - np.sqrt(3) / 2],
        ]
    )  # fmt: skip
    actual = regridding._conservative_latitude_weights(source_lat, target_lat)
    np.testing.assert_almost_equal(expected, actual)

  @parameterized.parameters(
      (1, 0, 1),
      (-1, 0, -1),
      (5, 0, 5),
      (6, 0, -4),
      (1, 9, 11),
      (5, 9, 5),
  )
  def test_align_phase_with(self, x, y, expected):
    actual = regridding._align_phase_with(x, y, period=10)
    self.assertEqual(actual, expected)

  def test_conservative_longitude_weights_same_branch(self):
    source_lon = np.pi / 180 * np.array([0, 60, 120, 180, 240, 300])
    target_lon = np.pi / 180 * np.array([0, 90, 180, 270])
    expected = (
        np.array(
            [
                [4, 1, 0, 0, 0, 1],
                [0, 3, 3, 0, 0, 0],
                [0, 0, 1, 4, 1, 0],
                [0, 0, 0, 0, 3, 3],
            ]
        )
        / 6
    )  # fmt: skip
    actual = regridding._conservative_longitude_weights(source_lon, target_lon)
    np.testing.assert_allclose(expected, actual, atol=1e-5)

  def test_conservative_longitude_weights_different_branch(self):
    source_lon = np.pi / 180 * np.array([90, 180, 270, 360])
    target_lon = np.pi / 180 * np.array([-270, -180, -90, 0])
    expected = np.eye(4)
    actual = regridding._conservative_longitude_weights(source_lon, target_lon)
    np.testing.assert_allclose(expected, actual, atol=1e-5)

  @parameterized.named_parameters(
      {
          'testcase_name': 'bilinear',
          'regridder_cls': regridding.BilinearRegridder,
      },
      {
          'testcase_name': 'conservative',
          'regridder_cls': regridding.ConservativeRegridder,
      },
      {
          'testcase_name': 'nearest',
          'regridder_cls': regridding.NearestRegridder,
      },
  )
  def test_regridding_shape(self, regridder_cls):
    source_grid = regridding.Grid.from_degrees(
        lon=np.linspace(0, 360, num=128, endpoint=False),
        lat=np.linspace(-90, 90, num=65, endpoint=True),
    )
    target_grid = regridding.Grid.from_degrees(
        lon=np.linspace(0, 360, num=100, endpoint=False),
        lat=np.linspace(-90, 90, num=50, endpoint=True),
    )
    regridder = regridder_cls(source_grid, target_grid)

    inputs = np.zeros(source_grid.shape)
    outputs = regridder.regrid_array(inputs)
    self.assertEqual(outputs.shape, target_grid.shape)

    batch_inputs = np.zeros((2,) + source_grid.shape)
    batch_outputs = regridder.regrid_array(batch_inputs)
    self.assertEqual(batch_outputs.shape, (2,) + target_grid.shape)

  @parameterized.named_parameters(
      {
          'testcase_name': 'bilinear',
          'regridder_cls': regridding.BilinearRegridder,
      },
      {
          'testcase_name': 'conservative',
          'regridder_cls': regridding.ConservativeRegridder,
      },
      {
          'testcase_name': 'nearest',
          'regridder_cls': regridding.NearestRegridder,
      },
  )
  def test_regridding_nans(self, regridder_cls):
    source_grid = regridding.Grid.from_degrees(
        lon=np.linspace(0, 360, num=512, endpoint=False),
        lat=np.linspace(-90, 90, num=256, endpoint=True),
    )
    target_grid = regridding.Grid.from_degrees(
        lon=np.linspace(0, 360, num=360, endpoint=False),
        lat=np.linspace(-90, 90, num=181, endpoint=True),
    )
    regridder = regridder_cls(source_grid, target_grid)

    inputs = np.ones(source_grid.shape)
    in_valid = (
        source_grid.lat[np.newaxis, :] ** 2
        + (source_grid.lon[:, np.newaxis] - np.pi) ** 2
        < (np.pi / 2) ** 2
    )
    inputs = np.where(in_valid, inputs, np.nan)
    outputs = regridder.regrid_array(inputs)

    out_valid = ~np.isnan(outputs)
    np.testing.assert_allclose(out_valid.mean(), in_valid.mean(), atol=0.01)
    np.testing.assert_allclose(outputs[out_valid], 1.0)


if __name__ == '__main__':
  absltest.main()
