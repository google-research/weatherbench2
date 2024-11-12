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
import typing as t

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from weatherbench2 import derived_variables as dvs
from weatherbench2 import schema
from weatherbench2 import test_utils
from weatherbench2 import utils
import xarray as xr


def get_random_weather(
    variables=('geopotential',), ensemble_size=None, seed=802701, **data_kwargs
):
  data_kwargs_to_use = dict(
      variables_3d=variables,
      variables_2d=[],
      time_start='2019-12-01',
      time_stop='2019-12-02',
      spatial_resolution_in_degrees=30,
  )
  data_kwargs_to_use.update(data_kwargs)

  return utils.random_like(
      schema.mock_forecast_data(
          ensemble_size=ensemble_size, **data_kwargs_to_use
      ),
      seed=seed + 1,
  )


def make_multispectral_dataset(
    spatial_resolution_in_degrees: int = 5,
    latitude: t.Optional[t.Union[int, t.Sequence[int]]] = None,
    min_wavelength_lon: int = 50,
    max_wavelength_lon: int = 100,
    constant_to_add: float = 0,
    seed: int = 802701,
):
  """Makes a dataset with smooth (in spectral space) data."""
  # Initialize dataset without random noise, so we can insert smooth
  # functions of lat/lon consistently for different resolutions.
  dataset = 0 * get_random_weather(
      variables=['geopotential'],
      ensemble_size=None,
      spatial_resolution_in_degrees=spatial_resolution_in_degrees,
      seed=seed,
  )
  if latitude is not None:
    dataset = dataset.sel(latitude=latitude)
  # Add signal at the many wavelengths to create a smooth spectra
  # Ensure all are long enough for this spatial resolution to resolve
  assert spatial_resolution_in_degrees < min_wavelength_lon / 2
  n_signals = 100
  for wavelength_lon in np.linspace(
      min_wavelength_lon, max_wavelength_lon, num=n_signals
  ):
    dataset['geopotential'] += (
        np.cos(2 * np.pi * dataset.longitude / wavelength_lon)
        * np.exp(-wavelength_lon / max_wavelength_lon)
        * np.sin(dataset['level'] / 500)
        * np.cos(dataset['latitude'] / 100)
    ) / n_signals
  dataset += constant_to_add * np.abs(dataset).mean()
  return dataset


class DerivedVariablesTest(absltest.TestCase):

  def testWindSpeed(self):
    dataset = xr.Dataset(
        {
            'u_component_of_wind': xr.DataArray([0, 3, np.nan]),
            'v_component_of_wind': xr.DataArray([0, -4, 1]),
        }
    )

    derived_variable = dvs.WindSpeed(
        u_name='u_component_of_wind',
        v_name='v_component_of_wind',
    )

    result = derived_variable.compute(dataset)

    expected = xr.DataArray([0, 5, np.nan])
    xr.testing.assert_allclose(result, expected)

  def testRelativeHumidity(self):
    dataset = xr.Dataset(
        {
            'temperature': ('level', np.array([240, 280, 295, 310])),
            'specific_humidity': ('level', np.array([1e-3, 1e-2, 2e-2, 4e-2])),
        },
        coords={'level': np.array([50, 200, 500, 850])},
    )
    derived_variable = dvs.RelativeHumidity()
    result = derived_variable.compute(dataset)
    expected = xr.DataArray(
        # from metpy.calc.relative_humidity_from_specific_humidity
        np.array([0.2116, 0.3115, 0.5937, 0.8462]),
        dims=['level'],
        coords=dataset.coords,
    )
    xr.testing.assert_allclose(result, expected, atol=1e-4)

  def _create_precip_dataset(self):
    lead_time = np.arange(0, 36 + 1, 6, dtype='timedelta64[h]')
    dataset = xr.Dataset(
        {
            'total_precipitation': xr.DataArray(
                [0, 5, 15, 14, 20, 30, 30],
                dims=['prediction_timedelta'],
                coords={'prediction_timedelta': lead_time},
            )
        }
    )
    return dataset

  def testPrecipitationAccumulation6hr(self):
    dataset = self._create_precip_dataset()

    derived_variable = dvs.PrecipitationAccumulation(
        total_precipitation_name='total_precipitation',
        accumulation_hours=6,
    )
    result = derived_variable.compute(dataset)

    # Test a few specific times for example's sake.
    # We want to verify that
    #   PrecipAccum6hr[t] = ReLu(TotalPrecip[t] - TotalPrecip[t - 6])
    sel = lambda ds, hr: ds.sel(prediction_timedelta=f'{hr}hr')
    relu = lambda ds: np.maximum(0, ds)
    np.testing.assert_array_equal(
        relu(sel(dataset, 24) - sel(dataset, 24 - 6)).total_precipitation.data,
        sel(result, 24),
    )
    np.testing.assert_array_equal(
        relu(sel(dataset, 18) - sel(dataset, 18 - 6)).total_precipitation.data,
        sel(result, 18),
    )

    # Test every timedelta.
    expected = xr.DataArray(
        [np.nan, 5, 10, 0, 6, 10, 0],
        dims=['prediction_timedelta'],
        coords={'prediction_timedelta': dataset.prediction_timedelta},
    )
    xr.testing.assert_allclose(result, expected)

  def testPrecipitationAccumulation24hr(self):
    dataset = self._create_precip_dataset()

    derived_variable = dvs.PrecipitationAccumulation(
        total_precipitation_name='total_precipitation',
        accumulation_hours=24,
    )
    result = derived_variable.compute(dataset)

    # Test a few specific times for example's sake.
    # We want to verify that
    #   PrecipAccum24hr[t] = ReLu(TotalPrecip[t] - TotalPrecip[t - 24])
    sel = lambda ds, hr: ds.sel(prediction_timedelta=f'{hr}hr')
    relu = lambda ds: np.maximum(0, ds)
    np.testing.assert_array_equal(
        relu(sel(dataset, 36) - sel(dataset, 36 - 24)).total_precipitation.data,
        sel(result, 36),
    )
    np.testing.assert_array_equal(
        relu(sel(dataset, 30) - sel(dataset, 30 - 24)).total_precipitation.data,
        sel(result, 30),
    )

    expected = xr.DataArray(
        [np.nan, np.nan, np.nan, np.nan, 20, 25, 15],
        dims=['prediction_timedelta'],
        coords={'prediction_timedelta': dataset.prediction_timedelta},
    )
    xr.testing.assert_allclose(result, expected)

  def testAggregatePrecipitationAccumulation(self):
    lead_time = np.arange(6, 36 + 1, 6, dtype='timedelta64[h]')
    dataset = xr.Dataset(
        {
            'total_precipitation_6hr': xr.DataArray(
                [5, 0, 2, 1, 0, 10],
                dims=['prediction_timedelta'],
                coords={'prediction_timedelta': lead_time},
            )
        }
    )

    derived_variable = dvs.AggregatePrecipitationAccumulation(
        accumulation_hours=24,
    )
    result = derived_variable.compute(dataset)
    expected = xr.DataArray(
        [np.nan, np.nan, np.nan, 8, 3, 13],
        dims=['prediction_timedelta'],
        coords={'prediction_timedelta': dataset.prediction_timedelta},
    )
    xr.testing.assert_allclose(result, expected)


class ZonalEnergySpectrumTest(parameterized.TestCase):

  def _wavelength_m(self, wavelength_lon: float, latitude: float) -> float:
    """Wavelength in m from wavelength in longitude units."""
    return (
        (wavelength_lon / 360)
        * (2 * np.pi * schema.EARTH_RADIUS_M)
        * np.cos(np.pi * latitude / 180)
    )

  def test_lon_spacing_m_correct_at_equator(self):
    spatial_resolution_in_degrees = 30
    dataset = get_random_weather(
        variables=['geopotential'],
        spatial_resolution_in_degrees=spatial_resolution_in_degrees,
    )
    derived_variable = dvs.ZonalEnergySpectrum(variable_name='geopotential')
    circum_at_equator = schema.EARTH_RADIUS_M * 2 * np.pi
    np.testing.assert_allclose(
        derived_variable._circumference(dataset).sel(latitude=0).data,
        circum_at_equator,
    )
    np.testing.assert_allclose(
        derived_variable.lon_spacing_m(dataset).sel(latitude=0).data,
        circum_at_equator * spatial_resolution_in_degrees / 360,
    )

  def test_data_has_right_shape_and_dims(self):
    dataset = get_random_weather(variables=['geopotential'], ensemble_size=None)
    spectrum = dvs.ZonalEnergySpectrum(
        variable_name='geopotential',
    ).compute(dataset)

    # 'longitude' gets changed to 'wavenumber', whose length is shorter (as we
    # store only the positive frequencies).
    expected_dims = dict(dataset.sizes)
    expected_dims['zonal_wavenumber'] = dataset.sizes['longitude'] // 2 + 1
    del expected_dims['longitude']
    spectrum_dims = dict(spectrum.sizes)
    self.assertEqual(expected_dims, spectrum_dims)

    # A new coordinate is 'frequency'
    self.assertEqual(('zonal_wavenumber', 'latitude'), spectrum.frequency.dims)
    self.assertEqual('1 / m', spectrum.frequency.units)

    # Frequency is increasing along wavenumber direction.
    test_utils.assert_positive(spectrum.frequency.diff('zonal_wavenumber'))
    np.testing.assert_array_equal(
        0, spectrum.frequency.isel(zonal_wavenumber=0)
    )

    # Wavelength is equal to 1 / frequency
    np.testing.assert_array_equal(spectrum.wavelength, 1 / spectrum.frequency)
    self.assertEqual('m', spectrum.wavelength.units)

    # Along the latitude direction, the smaller slice radius means frequency is
    # increasing, except of course the starting frequency which is always 0.
    lat_mid_idx = spectrum_dims['latitude'] // 2
    test_utils.assert_positive(
        spectrum.frequency.isel(
            zonal_wavenumber=slice(1, None), latitude=slice(lat_mid_idx, None)
        ).diff('latitude'),
        err_msg='Implies frequency not increasing along latitude',
    )
    test_utils.assert_negative(
        spectrum.frequency.isel(
            zonal_wavenumber=slice(1, None), latitude=slice(0, lat_mid_idx)
        ).diff('latitude'),
        err_msg='Implies frequency not decreasing along latitude',
    )

  @parameterized.named_parameters(
      dict(testcase_name='Equitorial', latitude=0),
      dict(testcase_name='30deg', latitude=30),
      dict(testcase_name='60deg', latitude=60),
  )
  def test_longitudinal_wave_detected(self, latitude):
    dataset = get_random_weather(
        variables=['geopotential'],
        ensemble_size=None,
        spatial_resolution_in_degrees=10,
    ).sel(
        # Confine the wave to latitude.
        latitude=latitude,
    )
    wavelength_lon = 100  # >> spatial_resolution_in_degrees to prevent aliasing
    dataset['geopotential'] += 10 * np.cos(
        2 * np.pi * dataset.longitude / wavelength_lon
    )
    spectrum = dvs.ZonalEnergySpectrum(variable_name='geopotential').compute(
        dataset
    )

    wavelength_m = self._wavelength_m(wavelength_lon, latitude)

    # Assert there is a spectral peak at the expected frequency, which is
    # = 1 / wavelength_m.
    np.testing.assert_array_equal(
        spectrum.argmax('zonal_wavenumber'),
        np.abs(spectrum.frequency - 1 / wavelength_m).argmin(
            'zonal_wavenumber'
        ),
    )

  @parameterized.named_parameters(
      # Since frequency 0 is treated special (we double its value in the energy
      # spectrum), test adding a constant to the values specially.
      dict(testcase_name='NoAddConstant', add_constant=False),
      dict(testcase_name='YesAddConstant', add_constant=True),
  )
  def test_resolved_frequencies_are_mostly_independent_of_discretization(
      self, add_constant
  ):
    # Confine the wave to a single latitude, so we can use frequency as a
    # dimension to select with.
    latitude = 30

    # Min/max wavelengths that we will add spectral content for.
    min_wavelength_lon = 50
    max_wavelength_lon = 100

    dataset_5 = make_multispectral_dataset(
        spatial_resolution_in_degrees=5,
        latitude=latitude,
        min_wavelength_lon=min_wavelength_lon,
        max_wavelength_lon=max_wavelength_lon,
        constant_to_add=50 if add_constant else 0,
    )
    dataset_20 = make_multispectral_dataset(
        spatial_resolution_in_degrees=20,
        latitude=latitude,
        min_wavelength_lon=min_wavelength_lon,
        max_wavelength_lon=max_wavelength_lon,
        constant_to_add=50 if add_constant else 0,
    )

    derived_variable = dvs.ZonalEnergySpectrum(variable_name='geopotential')

    spectrum_5 = derived_variable.compute(dataset_5).swap_dims(
        {'zonal_wavenumber': 'frequency'}
    )
    spectrum_20 = derived_variable.compute(dataset_20).swap_dims(
        {'zonal_wavenumber': 'frequency'}
    )

    # Test around the wavelengths we've added signal at, but not at the
    # boundary, since we expect boundary effects.
    test_frequencies = sorted(
        1
        / self._wavelength_m(
            np.random.uniform(
                low=1.1 * min_wavelength_lon,
                high=0.9 * max_wavelength_lon,
                size=30,
            ),
            latitude,
        )
    )

    # frequency=0 is well defined (and not a boundary case) at both resolutions.
    test_frequencies.append(0)

    # Assert spectrum_5 and spectrum_20 are close at all frequencies.
    # This fails if a different normalization of the DFT is used.
    for i, f in enumerate(test_frequencies):
      err = np.abs(
          spectrum_5.sel(frequency=f, method='nearest')
          - spectrum_20.sel(frequency=f, method='nearest')
      )

      # The difference in the low resolution spectra should provide an error
      # bound.
      def _select(da, where, f_):
        """Select `da` at frequency value above or below `f_`."""
        idx = int(np.argmin(np.abs(da.frequency.data - f_)))
        if where == 'below':
          idx = max(0, idx - 1)
        elif where == 'above':
          idx = min(da.frequency.size - 1, idx + 1)
        return da.isel(frequency=idx)

      expected_err_bound = np.abs(
          _select(spectrum_20, 'below', f) - _select(spectrum_20, 'above', f)
      )
      np.testing.assert_array_less(
          err.data,
          expected_err_bound.data,
          err_msg=f'Failed at {i=} {f=}',
      )

  @parameterized.named_parameters(
      # Since frequency 0 is treated special (we double its value in the energy
      # spectrum), test adding a constant to the values specially.
      dict(testcase_name='NoAddConstant', add_constant=False),
      dict(testcase_name='YesAddConstant', add_constant=True),
  )
  def test_parsevals_relation(self, add_constant):
    dataset = make_multispectral_dataset(
        spatial_resolution_in_degrees=5,
        latitude=slice(-30, 30),
        constant_to_add=50 if add_constant else 0,
    )

    derived_variable = dvs.ZonalEnergySpectrum(variable_name='geopotential')

    energy = (
        (derived_variable.lon_spacing_m(dataset) * dataset**2)
        .sum('longitude')
        .geopotential
    )

    spectrum = derived_variable.compute(dataset)
    spectral_energy = spectrum.sum('zonal_wavenumber')

    xr.testing.assert_allclose(
        spectral_energy.transpose(*energy.dims), energy, rtol=2e-3
    )

  def test_interpolate_frequencies_default_args(self):
    dataset = make_multispectral_dataset(
        spatial_resolution_in_degrees=5,
        latitude=slice(-30, 30),
    )
    derived_variable = dvs.ZonalEnergySpectrum(variable_name='geopotential')
    spectrum = derived_variable.compute(dataset)

    interpolated = dvs.interpolate_spectral_frequencies(
        spectrum, wavenumber_dim='zonal_wavenumber'
    )

    self.assertEqual(
        {'frequency'} | set(spectrum.dims) - {'zonal_wavenumber'},
        set(interpolated.dims),
    )

    # Since we had a latitude=0 point, and all frequencies started at 0, this
    # point must have the most narrow frequency range, which will be used as the
    # default.  In fact, all data at this range should be unchanged.
    xr.testing.assert_allclose(
        spectrum.sel(latitude=0)
        .swap_dims({'zonal_wavenumber': 'frequency'})
        .drop_vars('zonal_wavenumber'),
        interpolated.sel(latitude=0),
    )

    # Results at latitude = +- 5 deg should be barely changed.
    xr.testing.assert_allclose(
        spectrum.sel(latitude=5)
        .swap_dims({'zonal_wavenumber': 'frequency'})
        .drop_vars('zonal_wavenumber'),
        interpolated.sel(latitude=5),
        rtol=0.15,
    )

    # Check wavelength
    np.testing.assert_allclose(
        interpolated.wavelength, 1 / interpolated.frequency
    )
    self.assertEqual(interpolated.wavelength.units, 'm')

  def test_interpolate_frequencies_use_5_degree_values(self):
    dataset = make_multispectral_dataset(
        spatial_resolution_in_degrees=1.0,
        latitude=slice(-30, 30),
    )
    derived_variable = dvs.ZonalEnergySpectrum(variable_name='geopotential')
    spectrum = derived_variable.compute(dataset)

    reference_lat = 5
    reference_wavenumbers = slice(3, 8)

    interpolated = dvs.interpolate_spectral_frequencies(
        spectrum,
        wavenumber_dim='zonal_wavenumber',
        frequencies=spectrum.frequency.sel(
            latitude=reference_lat, zonal_wavenumber=reference_wavenumbers
        ),
    )

    self.assertEqual(
        {'frequency'} | set(spectrum.dims) - {'zonal_wavenumber'},
        set(interpolated.dims),
    )

    # At the reference latitude point, data should be unchanged.
    xr.testing.assert_allclose(
        spectrum.sel(
            latitude=reference_lat, zonal_wavenumber=reference_wavenumbers
        )
        .swap_dims({'zonal_wavenumber': 'frequency'})
        .drop_vars('zonal_wavenumber'),
        interpolated.sel(latitude=reference_lat),
    )

    # Results at reference_latitude = +- 1 deg should be barely changed.
    xr.testing.assert_allclose(
        spectrum.sel(
            latitude=reference_lat + 1, zonal_wavenumber=reference_wavenumbers
        )
        .swap_dims({'zonal_wavenumber': 'frequency'})
        .drop_vars('zonal_wavenumber'),
        interpolated.sel(latitude=reference_lat + 1),
        rtol=0.1,
    )

    # Check wavelength
    np.testing.assert_allclose(
        interpolated.wavelength, 1 / interpolated.frequency
    )
    self.assertEqual(interpolated.wavelength.units, 'm')


if __name__ == '__main__':
  absltest.main()
