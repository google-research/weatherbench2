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
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from weatherbench2 import derived_variables
from weatherbench2 import schema
from weatherbench2 import test_utils
from weatherbench2 import utils
from weatherbench2.derived_variables import WindSpeed, PrecipitationAccumulation, AggregatePrecipitationAccumulation, ZonalPowerSpectrum  # pylint: disable=g-multiple-import
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


class DerivedVariablesTest(absltest.TestCase):

  def testWindSpeed(self):
    dataset = xr.Dataset({
        'u_component_of_wind': xr.DataArray([0, 3, np.NaN]),
        'v_component_of_wind': xr.DataArray([0, -4, 1]),
    })

    derived_variable = WindSpeed(
        u_name='u_component_of_wind',
        v_name='v_component_of_wind',
        variable_name='wind_speed',
    )

    result = derived_variable.compute(dataset)

    expected = xr.DataArray([0, 5, np.NaN])
    xr.testing.assert_allclose(result, expected)

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

    derived_variable = PrecipitationAccumulation(
        variable_name='total_precipitation_6hr',
        total_precipitation_name='total_precipitation',
        accumulation_hours=6,
    )
    result = derived_variable.compute(dataset)
    expected = xr.DataArray(
        [np.NaN, 5, 10, 0, 6, 10, 0],
        dims=['prediction_timedelta'],
        coords={'prediction_timedelta': dataset.prediction_timedelta},
    )
    xr.testing.assert_allclose(result, expected)

  def testPrecipitationAccumulation24hr(self):
    dataset = self._create_precip_dataset()

    derived_variable = PrecipitationAccumulation(
        variable_name='total_precipitation_24hr',
        total_precipitation_name='total_precipitation',
        accumulation_hours=24,
    )
    result = derived_variable.compute(dataset)
    expected = xr.DataArray(
        [np.NaN, np.NaN, np.NaN, np.NaN, 20, 25, 15],
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

    derived_variable = AggregatePrecipitationAccumulation(
        variable_name='total_precipitation_24hr',
        accumulation_hours=24,
    )
    result = derived_variable.compute(dataset)
    expected = xr.DataArray(
        [np.NaN, np.NaN, np.NaN, 8, 3, 13],
        dims=['prediction_timedelta'],
        coords={'prediction_timedelta': dataset.prediction_timedelta},
    )
    xr.testing.assert_allclose(result, expected)


class ZonalPowerSpectrumTest(parameterized.TestCase):

  def test_data_has_right_shape_and_dims(self):
    dataset = get_random_weather(variables=['geopotential'], ensemble_size=None)
    spectrum = ZonalPowerSpectrum(
        variable_name='geopotential',
    ).compute(dataset)

    # 'longitude' gets changed to 'wavenumber', whose length is shorter (as we
    # store only the positive frequencies).
    expected_dims = dict(dataset.dims)
    expected_dims['wavenumber'] = dataset.dims['longitude'] // 2 + 1
    del expected_dims['longitude']
    spectrum_dims = dict(zip(spectrum.dims, spectrum.shape))
    self.assertEqual(expected_dims, spectrum_dims)

    # A new coordinate is 'frequency'
    self.assertEqual(('wavenumber', 'latitude'), spectrum.frequency.dims)
    self.assertEqual('1 / km', spectrum.frequency.units)

    # Frequency is increasing along wavenumber direction.
    test_utils.assert_positive(spectrum.frequency.diff('wavenumber'))

    np.testing.assert_array_equal(0, spectrum.frequency.isel(wavenumber=0))

    # Along the latitude direction, the smaller slice radius means frequency is
    # increasing, except of course the starting frequency which is always 0.
    lat_mid_idx = spectrum_dims['latitude'] // 2
    test_utils.assert_positive(
        spectrum.frequency.isel(
            wavenumber=slice(1, None), latitude=slice(lat_mid_idx, None)
        ).diff('latitude'),
        err_msg='Implies frequency not increasing along latitude',
    )
    test_utils.assert_negative(
        spectrum.frequency.isel(
            wavenumber=slice(1, None), latitude=slice(0, lat_mid_idx)
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
    spectrum = ZonalPowerSpectrum(variable_name='geopotential').compute(dataset)

    wavelength_km = (wavelength_lon / 360) * (
        2 * np.pi * derived_variables._EARTH_RADIUS_KM
    ) * np.cos(np.pi * latitude / 180)

    # Assert there is a spectral peak at the expected frequency.
    np.testing.assert_array_equal(
        spectrum.argmax('wavenumber'),
        np.abs(spectrum.frequency - 1 / wavelength_km).argmin('wavenumber'),
    )


if __name__ == '__main__':
  absltest.main()
