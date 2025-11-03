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


LongitudeScheme = regridding.LongitudeScheme
LatitudeSpacing = regridding.LatitudeSpacing


def infer_longitude_scheme(
    lon_in_degrees: np.ndarray,
) -> LongitudeScheme:
  """Determine longitude scheme."""
  eq = lambda a, b: np.allclose(a, b, atol=1e-5)
  if eq(lon_in_degrees[0], 0) and lon_in_degrees[-1] < 360:
    return LongitudeScheme.START_AT_ZERO
  elif lon_in_degrees[0] < 0 and eq(-lon_in_degrees[0], lon_in_degrees[-1]):
    return LongitudeScheme.CENTER_AT_ZERO
  else:
    raise ValueError(f'Unknown longitude scheme for {lon_in_degrees=}')


def _maybe_roll_longitude(
    ds: xr.Dataset, longitude_scheme: LongitudeScheme
) -> xr.Dataset:
  """Rolls longitude to approximate longitude_scheme."""
  ds = ds.copy()
  current = infer_longitude_scheme(ds.longitude.data)
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
      dict(
          regridder_cls=regridding.ConservativeRegridder,
          source_lat_spacing=LatitudeSpacing.CUSTOM,
          target_lat_spacing=LatitudeSpacing.CUSTOM,
      ),
      dict(
          regridder_cls=regridding.ConservativeRegridder,
          source_lat_spacing=LatitudeSpacing.EQUIANGULAR_WITHOUT_POLES,
          target_lat_spacing=LatitudeSpacing.CUSTOM,
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

    if source_lat_spacing == LatitudeSpacing.CUSTOM:
      source_latitude = np.linspace(-87.5, 87.5, n_lats)
    else:
      source_latitude = regridding.latitude_values(source_lat_spacing, n_lats)

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
            latitude=source_latitude,
            longitude=regridding.longitude_values(source_lon_scheme, n_lons),
        ),
    )
    theta = 2 * np.pi * source.longitude / 360
    phi = 2 * np.pi * (source.latitude - 90) / 180
    source += np.sin(phi) * (np.cos(theta) ** 2 + np.sin(theta))

    reduce_factor = 2

    # Get approximate target values just by choosing nearest values in source.
    if target_lat_spacing == LatitudeSpacing.CUSTOM:
      regridded_latitude = np.linspace(
          -87.5, 87.5, source.sizes['latitude'] // reduce_factor
      )
    else:
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

    source_grid = regridding.Grid(
        longitudes=source.longitude.data,
        latitudes=source.latitude.data,
        includes_poles=True,
        periodic=True,
    )
    target_grid = regridding.Grid(
        longitudes=regridded_longitude,
        latitudes=regridded_latitude,
        includes_poles=True,
        periodic=True,
    )
    regridder = regridder_cls(source_grid, target_grid)

    regridded_ds = regridder.regrid_dataset(source)
    self.assertTrue(np.all(np.isfinite(regridded_ds.X)))

    if target_lat_spacing == LatitudeSpacing.CUSTOM:
      min_frac_obeying_bounds = 0.97
    elif source_lon_scheme == target_lon_scheme:
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
    source_lat = np.array([-75, -45, -15, 15, 45, 75])
    target_lat = np.array([-45, 45])
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
    actual = regridding._conservative_latitude_weights(
        source_lat,
        target_lat,
        source_includes_poles=True,
        target_includes_poles=True,
    )
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
    source_lon = np.array([0, 60, 120, 180, 240, 300])
    target_lon = np.array([0, 90, 180, 270])
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
    actual = regridding._conservative_longitude_weights(
        source_lon, target_lon, source_periodic=True, target_periodic=True
    )
    np.testing.assert_allclose(expected, actual, atol=1e-5)

  def test_conservative_longitude_weights_different_branch(self):
    source_lon = np.array([90, 180, 270, 360])
    target_lon = np.array([-270, -180, -90, 0])
    expected = np.eye(4)
    actual = regridding._conservative_longitude_weights(
        source_lon, target_lon, source_periodic=True, target_periodic=True
    )
    np.testing.assert_allclose(expected, actual, atol=1e-5)

  def test_conservative_regridding_extrapolation(self):
    source_grid = regridding.Grid(
        longitudes=np.array([1, 3, 5]),
        latitudes=np.array([1, 3]),
        includes_poles=False,
        periodic=False,
    )
    target_grid = regridding.Grid(
        longitudes=np.array([0, 2, 4]),
        latitudes=np.array([0, 2]),
        includes_poles=False,
        periodic=False,
    )
    regridder = regridding.ConservativeRegridder(source_grid, target_grid)
    field = np.array([[1, 1], [2, 2], [3, 3]])
    actual = regridder.regrid_array(field)
    expected = np.array([[np.nan, np.nan], [np.nan, 1.5], [np.nan, 2.5]])
    np.testing.assert_allclose(actual, expected, atol=1e-6)

  @parameterized.named_parameters(
      dict(
          testcase_name='global',
          source_poles=True,
          target_poles=True,
          source_periodic=True,
          target_periodic=True,
          expect_nans=False,
      ),
      dict(
          testcase_name='no_poles',
          source_poles=False,
          target_poles=False,
          source_periodic=True,
          target_periodic=True,
          expect_nans=True,
      ),
      dict(
          testcase_name='no_poles_source',
          source_poles=False,
          target_poles=True,
          source_periodic=True,
          target_periodic=True,
          expect_nans=True,
      ),
      dict(
          testcase_name='no_poles_target',
          source_poles=True,
          target_poles=False,
          source_periodic=True,
          target_periodic=True,
          expect_nans=False,
      ),
      dict(
          testcase_name='non_periodic',
          source_poles=True,
          target_poles=True,
          source_periodic=False,
          target_periodic=False,
          expect_nans=True,
      ),
  )
  def test_conservative_regridder_has_expected_nans(
      self,
      source_poles,
      target_poles,
      source_periodic,
      target_periodic,
      expect_nans,
  ):
    def make_lats(includes_poles, num):
      return (
          np.linspace(-90, 90, num)
          if includes_poles
          else np.linspace(-80, 80, num)
      )

    def make_lons(periodic, num):
      return (
          np.linspace(0, 360, num, endpoint=False)
          if periodic
          else np.linspace(0, 180, num)
      )

    source_grid = regridding.Grid(
        longitudes=make_lons(source_periodic, 20),
        latitudes=make_lats(source_poles, 10),
        includes_poles=source_poles,
        periodic=source_periodic,
    )
    target_grid = regridding.Grid(
        longitudes=make_lons(target_periodic, 15),
        latitudes=make_lats(target_poles, 8),
        includes_poles=target_poles,
        periodic=target_periodic,
    )
    regridder = regridding.ConservativeRegridder(source_grid, target_grid)
    field = np.ones(source_grid.shape)
    actual = regridder.regrid_array(field)
    self.assertEqual(np.isnan(actual).any(), expect_nans)
    np.testing.assert_allclose(actual[~np.isnan(actual)], 1.0, atol=1e-6)

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
    source_grid = regridding.Grid(
        longitudes=np.linspace(0, 360, num=128, endpoint=False),
        latitudes=np.linspace(-90, 90, num=65, endpoint=True),
        includes_poles=True,
        periodic=True,
    )
    target_grid = regridding.Grid(
        longitudes=np.linspace(0, 360, num=100, endpoint=False),
        latitudes=np.linspace(-90, 90, num=50, endpoint=True),
        includes_poles=True,
        periodic=True,
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
    source_grid = regridding.Grid(
        longitudes=np.linspace(0, 360, num=512, endpoint=False),
        latitudes=np.linspace(-90, 90, num=256, endpoint=True),
        includes_poles=True,
        periodic=True,
    )
    target_grid = regridding.Grid(
        longitudes=np.linspace(0, 360, num=360, endpoint=False),
        latitudes=np.linspace(-90, 90, num=181, endpoint=True),
        includes_poles=True,
        periodic=True,
    )
    regridder = regridder_cls(source_grid, target_grid)

    inputs = np.ones(source_grid.shape)
    source_lat_rad = np.deg2rad(source_grid.latitudes)
    source_lon_rad = np.deg2rad(source_grid.longitudes)
    in_valid = (
        source_lat_rad[np.newaxis, :] ** 2
        + (source_lon_rad[:, np.newaxis] - np.pi) ** 2
        < (np.pi / 2) ** 2
    )
    inputs = np.where(in_valid, inputs, np.nan)
    outputs = regridder.regrid_array(inputs)

    out_valid = ~np.isnan(outputs)
    np.testing.assert_allclose(out_valid.mean(), in_valid.mean(), atol=0.01)
    np.testing.assert_allclose(outputs[out_valid], 1.0)

  @parameterized.parameters(
      dict(
          periodic=True,
          field=np.array([[0.0], [1.0], [2.0], [3.0]]),
          expected=np.array([[0.5], [1.5], [2.5], [1.5]]),
      ),
      dict(
          periodic=False,
          field=np.array([[0.0], [1.0], [2.0], [3.0]]),
          expected=np.array([[0.5], [1.5], [2.5], [np.nan]]),
      ),
  )
  def test_bilinear_regridder_longitude_periodicity(
      self, periodic, field, expected
  ):
    source_grid = regridding.Grid(
        longitudes=np.array([0.0, 90.0, 180.0, 270.0]),
        latitudes=np.array([0]),
        includes_poles=True,
        periodic=periodic,
    )
    target_grid = regridding.Grid(
        longitudes=np.array([45.0, 135.0, 225.0, 315.0]),
        latitudes=np.array([0]),
        includes_poles=True,
        periodic=periodic,
    )
    regridder = regridding.BilinearRegridder(source_grid, target_grid)
    actual = regridder.regrid_array(field)
    np.testing.assert_allclose(expected, actual, atol=1e-6)

  @parameterized.parameters(
      dict(
          include_poles=True,
          source_latitudes=np.array([-90.0, -30.0, 30.0, 90.0]),
          target_latitudes=np.array([-60.0, 0.0, 60.0]),
          field_values=np.array([0.0, 1.0, 2.0, 3.0]),
          expected=np.array([[0.5, 1.5, 2.5]]),
      ),
      dict(
          include_poles=True,
          source_latitudes=np.array([-60.0, 0.0, 60.0]),
          target_latitudes=np.array([-90.0, -30.0, 30.0, 90.0]),
          field_values=np.array([0.0, 1.0, 2.0]),
          expected=np.array([[0.0, 0.5, 1.5, 2.0]]),
      ),
      dict(
          include_poles=False,
          source_latitudes=np.array([-60.0, -20.0, 20.0, 60.0]),
          target_latitudes=np.array([-70.0, 0.0, 70.0]),
          field_values=np.array([0.0, 1.0, 2.0, 3.0]),
          expected=np.array([[np.nan, 1.5, np.nan]]),
      ),
  )
  def test_bilinear_regridder_latitude_poles(
      self,
      include_poles,
      source_latitudes,
      target_latitudes,
      field_values,
      expected,
  ):
    source_grid = regridding.Grid(
        longitudes=np.array([0.0]),
        latitudes=source_latitudes,
        includes_poles=include_poles,
        periodic=True,
    )
    target_grid = regridding.Grid(
        longitudes=np.array([0.0]),
        latitudes=target_latitudes,
        includes_poles=include_poles,
        periodic=True,
    )
    regridder = regridding.BilinearRegridder(source_grid, target_grid)
    field = field_values[np.newaxis, :]
    actual = regridder.regrid_array(field)
    np.testing.assert_allclose(expected, actual, atol=1e-6)

  def test_nearest_regridder_exact(self):
    source_grid = regridding.Grid(
        longitudes=np.array([0, 90, 180, 270]),
        latitudes=np.array([-30, 0, 30]),
        includes_poles=True,
        periodic=True,
    )
    target_grid = regridding.Grid(
        longitudes=np.array([0, 180]),
        latitudes=np.array([-30, 0, 30]),
        includes_poles=True,
        periodic=True,
    )
    regridder = regridding.NearestRegridder(source_grid, target_grid)
    field = np.array([[0, 1, 2], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    expected = np.array([[0, 1, 2], [7, 8, 9]])
    actual = regridder.regrid_array(field)
    np.testing.assert_allclose(expected, actual, atol=1e-6)

  def test_quarter_degree_grid_doesnt_lead_to_nan(self):
    # This 1/4 deg grid region caused issues whereby the
    # is_covered = np.isclose(coverage, target_lengths, rtol=1e-5)
    # statement in _conservative_latitude_weights determined the last point
    # was not covered. This only happened when _conservative_latitude_weights
    # was jit-compiled (due to being inside of ConservativeRegridder._mean).
    # The 1e-5 rtol was too sensitive for jitted operations.
    lats = np.array([31., 31.25, 31.5])
    lons = np.array([0.0, 1.0])

    source_grid = regridding.Grid(
      longitudes=lons,
      latitudes=lats,
      includes_poles=False,
      periodic=False,
    )
    target_grid = regridding.Grid(
      longitudes=lons,
      latitudes=lats,
      includes_poles=False,
      periodic=False,
    )
    regridder = regridding.ConservativeRegridder(source_grid, target_grid)
    field = np.ones((source_grid.longitudes.size, source_grid.latitudes.size))
    actual = regridder.regrid_array(field)
    np.testing.assert_array_equal(True, np.isfinite(actual))


if __name__ == '__main__':
  absltest.main()
