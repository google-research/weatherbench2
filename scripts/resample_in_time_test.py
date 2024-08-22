# Copyright 2024 Google LLC
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
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import pandas as pd
from weatherbench2 import schema
from weatherbench2 import utils
import xarray as xr
import xarray_beam

from . import resample_in_time


class ResampleInTimeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='NoNaN', insert_nan=False),
      dict(testcase_name='YesNaN', insert_nan=True),
  )
  def test_demonstrating_resample_and_rolling_are_aligned(self, insert_nan):
    # times = 10 days, starting at Jan 1
    times = pd.DatetimeIndex(
        [
            '2023-01-01',
            '2023-01-02',
            '2023-01-03',
            '2023-01-04',
            '2023-01-05',
            '2023-01-06',
            '2023-01-07',
            '2023-01-08',
            '2023-01-09',
            '2023-01-10',
        ]
    )
    temperatures = np.arange(len(times)).astype(float)

    if insert_nan:
      # NaN inserted to (i) verify skipna=False, and (ii) verify correct setting
      # for min_periods. If e.g. min_periods=1, then NaN values get skipped so
      # long as there is at least one non-NaN value!
      temperatures[0] = np.nan

    input_ds = xr.Dataset(
        {
            'temperature': xr.DataArray(
                temperatures, coords=[times], dims=['time']
            )
        }
    )

    input_path = self.create_tempdir('source').full_path
    input_ds.to_zarr(input_path)

    # Get resampled output
    resample_output_path = self.create_tempdir('resample').full_path
    with flagsaver.as_parsed(
        input_path=input_path,
        output_path=resample_output_path,
        method='resample',
        period='3d',
        mean_vars='ALL',
        runner='DirectRunner',
    ):
      resample_in_time.main([])
    resample, unused_output_chunks = xarray_beam.open_zarr(resample_output_path)

    # Show that the output at time T uses data from the window [T, T + period]
    np.testing.assert_array_equal(
        pd.to_datetime(resample.time),
        pd.DatetimeIndex(
            ['2023-01-01', '2023-01-04', '2023-01-07', '2023-01-10']
        ),
    )

    np.testing.assert_array_equal(
        resample.temperature.data,
        [
            np.mean(temperatures[:3]),  # Will be NaN if `insert_nan`
            np.mean(temperatures[3:6]),
            np.mean(temperatures[6:9]),
            np.mean(temperatures[9:12]),
        ],
    )

    # Get rolled output
    rolling_output_path = self.create_tempdir('rolling').full_path
    with flagsaver.as_parsed(
        input_path=input_path,
        output_path=rolling_output_path,
        method='rolling',
        period='3d',
        mean_vars='ALL',
        runner='DirectRunner',
    ):
      resample_in_time.main([])
    rolling, unused_output_chunks = xarray_beam.open_zarr(rolling_output_path)

    common_times = pd.DatetimeIndex(['2023-01-01', '2023-01-04', '2023-01-07'])
    xr.testing.assert_equal(
        resample.sel(time=common_times),
        rolling.sel(time=common_times),
    )

  @parameterized.parameters(
      (20, '3d', None),
      (21, '3d', None),
      (21, '8d', None),
      (5, '1d', None),
      (20, '3d', [0, 4, 8]),
      (21, '3d', [20]),
      (21, '8d', [15]),
  )
  def test_demonstrating_resample_and_rolling_are_aligned_many_combinations(
      self,
      n_times,
      period,
      nan_locations,
  ):
    # Less readable than test_demonstrating_resample_and_rolling_are_aligned,
    # but these sorts of automated checks ensure we didn't miss an edge case
    # (there are many!!!!)
    times = pd.date_range('2010', periods=n_times)
    temperatures = np.random.RandomState(802701).rand(n_times)

    for i in nan_locations or []:
      temperatures[i] = np.nan

    input_ds = xr.Dataset(
        {
            'temperature': xr.DataArray(
                temperatures, coords=[times], dims=['time']
            )
        }
    )

    input_path = self.create_tempdir('source').full_path
    input_ds.to_zarr(input_path)

    # Get resampled output
    resample_output_path = self.create_tempdir('resample').full_path
    with flagsaver.as_parsed(
        input_path=input_path,
        output_path=resample_output_path,
        method='resample',
        period=period,
        mean_vars='ALL',
        runner='DirectRunner',
    ):
      resample_in_time.main([])
    resample, unused_output_chunks = xarray_beam.open_zarr(resample_output_path)

    # Get rolled output
    rolling_output_path = self.create_tempdir('rolling').full_path
    with flagsaver.as_parsed(
        input_path=input_path,
        output_path=rolling_output_path,
        method='rolling',
        period=period,
        mean_vars='ALL',
        runner='DirectRunner',
    ):
      resample_in_time.main([])
    rolling, unused_output_chunks = xarray_beam.open_zarr(rolling_output_path)

    common_times = pd.to_datetime(resample.time.data).intersection(
        rolling.time.data
    )

    # At most, one time is lost if the period doesn't evenly divide n_times.
    self.assertGreaterEqual(len(common_times), len(resample.time) - 1)
    xr.testing.assert_equal(
        resample.sel(time=common_times),
        rolling.sel(time=common_times),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='Resample_NoSuffix_5d',
          method='resample',
          add_mean_suffix=False,
          period='5d',
      ),
      dict(
          testcase_name='Resample_YesSuffix_1w',
          method='resample',
          add_mean_suffix=True,
          period='1w',
      ),
      dict(
          testcase_name='Resample_YesSuffix_1d',
          method='resample',
          add_mean_suffix=True,
          period='1d',
      ),
      dict(
          testcase_name='Roll_YesSuffix_1w',
          method='rolling',
          add_mean_suffix=True,
          period='1w',
      ),
      dict(
          testcase_name='Roll_NoSuffix_30d',
          method='rolling',
          add_mean_suffix=False,
          period='30d',
      ),
      dict(
          testcase_name='Roll_YesSuffix_1d',
          method='rolling',
          add_mean_suffix=True,
          period='1d',
      ),
  )
  def test_resample_time(self, method, add_mean_suffix, period):
    # Make sure slice(start, stop, period) doesn't give you a singleton, since
    # then, for this singleton, the resampled mean/min/max will all be equal,
    # and the test will fail.
    time_start = '2021-02-01'
    time_stop = '2021-04-01'
    mean_vars = ['temperature', 'geopotential']
    min_vars = ['temperature', 'geopotential']
    max_vars = ['temperature']
    input_time_resolution = '1d'

    input_ds = utils.random_like(
        schema.mock_truth_data(
            variables_3d=['temperature', 'geopotential', 'should_drop'],
            # time_start/stop for the raw data is wider than the times for
            # resampled data.
            time_start='2021-01-01',
            time_stop='2022-01-01',
            spatial_resolution_in_degrees=30.0,
            time_resolution=input_time_resolution,
        )
    )

    # Make variables different so we test that variables are handled
    # individually.
    input_ds = input_ds.assign({'geopotential': input_ds.geopotential + 10})

    # Add a variable that uses some new dimension. We expect this dimension to
    # be dropped from the output (since we won't put this variable in the
    # output).
    input_ds['var_to_drop'] = xr.DataArray(np.ones((5,)), dims=['dim_to_drop'])

    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_chunks = {
        'time': 40,
        'longitude': 6,
        'latitude': 5,
        'level': 3,
        'dim_to_drop': 5,
    }
    input_ds.chunk(input_chunks).to_zarr(input_path)

    with flagsaver.as_parsed(
        input_path=input_path,
        output_path=output_path,
        method=method,
        period=period,
        mean_vars=','.join(mean_vars),
        min_vars=','.join(min_vars),
        max_vars=','.join(max_vars),
        add_mean_suffix=str(add_mean_suffix),
        time_start=time_start,
        time_stop=time_stop,
        working_chunks='level=1',
        runner='DirectRunner',
    ):
      resample_in_time.main([])

    output_ds, output_chunks = xarray_beam.open_zarr(output_path)

    if method == 'resample':
      expected_mean = (
          input_ds.sel(time=slice(time_start, time_stop))
          .resample(time=pd.to_timedelta(period))
          .mean()
      )
    elif method == 'rolling':
      expected_mean = (
          input_ds.sel(time=slice(time_start, time_stop))
          # input_ds timedelta is 1 day.
          .rolling(
              time=pd.to_timedelta(period)
              // pd.to_timedelta(input_time_resolution)
          ).mean()
      )
      # Enact the time offsetting needed to align resample and rolling.
      expected_mean = expected_mean.assign_coords(
          time=expected_mean.time
          - pd.to_timedelta(period)
          + pd.to_timedelta(input_time_resolution)
      )
    else:
      raise ValueError(f'Unhandled {method=}')

    expected_chunks = input_chunks.copy()
    del expected_chunks['dim_to_drop']
    if method == 'resample':
      expected_chunks['time'] = min(
          len(expected_mean.time), expected_chunks['time']
      )
    self.assertEqual(expected_chunks, output_chunks)

    expected_varnames = []

    for k in mean_vars:
      expected_varnames.append(k + '_mean' if add_mean_suffix else k)
      xr.testing.assert_allclose(
          expected_mean[k],
          output_ds[k + '_mean' if add_mean_suffix else k],
      )

    for k in min_vars:
      expected_varnames.append(k + '_min')
      if period != input_time_resolution:
        np.testing.assert_array_less(output_ds[k + '_min'], expected_mean[k])

    for k in max_vars:
      expected_varnames.append(k + '_max')
      if period != input_time_resolution:
        np.testing.assert_array_less(expected_mean[k], output_ds[k + '_max'])

    self.assertCountEqual(expected_varnames, output_ds.data_vars)

  @parameterized.named_parameters(
      dict(
          testcase_name='Resample_NoSuffix_5d',
          method='resample',
          add_mean_suffix=False,
          period='5d',
      ),
      dict(
          testcase_name='Resample_YesSuffix_1w',
          method='resample',
          add_mean_suffix=True,
          period='1w',
      ),
      dict(
          testcase_name='Resample_YesSuffix_1d',
          method='resample',
          add_mean_suffix=True,
          period='1d',
      ),
      dict(
          testcase_name='Roll_YesSuffix_1w',
          method='rolling',
          add_mean_suffix=True,
          period='1w',
      ),
      dict(
          testcase_name='Roll_NoSuffix_30d',
          method='rolling',
          add_mean_suffix=False,
          period='30d',
      ),
      dict(
          testcase_name='Roll_YesSuffix_1d',
          method='rolling',
          add_mean_suffix=True,
          period='1d',
      ),
  )
  def test_resample_prediction_timedelta(self, method, add_mean_suffix, period):
    # Make sure slice(start, stop, period) doesn't give you a singleton, since
    # then, for this singleton, the resampled mean/min/max will all be equal,
    # and the test will fail.
    timedelta_start = '0 day'
    timedelta_stop = '9 days'
    mean_vars = ['temperature', 'geopotential']
    min_vars = ['temperature', 'geopotential']
    max_vars = ['temperature']
    input_time_resolution = '1d'

    input_ds = utils.random_like(
        schema.mock_forecast_data(
            lead_start='0 day',
            lead_stop='15 days',
            lead_resolution='1 day',
            variables_3d=['temperature', 'geopotential', 'should_drop'],
            time_start='2021-01-01',
            time_stop='2021-01-10',
            spatial_resolution_in_degrees=30.0,
            time_resolution=input_time_resolution,
        )
    )

    # Make variables different so we test that variables are handled
    # individually.
    input_ds = input_ds.assign({'geopotential': input_ds.geopotential + 10})

    # Add a variable that uses some new dimension. We expect this dimension to
    # be dropped from the output (since we won't put this variable in the
    # output).
    input_ds['var_to_drop'] = xr.DataArray(np.ones((5,)), dims=['dim_to_drop'])

    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_chunks = {
        'time': 9,
        'prediction_timedelta': 5,
        'longitude': 6,
        'latitude': 5,
        'level': 3,
        'dim_to_drop': 5,
    }
    input_ds.chunk(input_chunks).to_zarr(input_path)

    with flagsaver.as_parsed(
        input_path=input_path,
        output_path=output_path,
        method=method,
        period=period,
        mean_vars=','.join(mean_vars),
        min_vars=','.join(min_vars),
        max_vars=','.join(max_vars),
        add_mean_suffix=str(add_mean_suffix),
        time_start=timedelta_start,
        time_stop=timedelta_stop,
        working_chunks='level=1',
        time_dim='prediction_timedelta',
        runner='DirectRunner',
    ):
      resample_in_time.main([])

    output_ds, output_chunks = xarray_beam.open_zarr(output_path)

    if method == 'resample':
      expected_mean = (
          input_ds.sel(
              prediction_timedelta=slice(timedelta_start, timedelta_stop)
          )
          .resample(prediction_timedelta=pd.to_timedelta(period))
          .mean()
      )
    elif method == 'rolling':
      expected_mean = (
          input_ds.sel(
              prediction_timedelta=slice(timedelta_start, timedelta_stop)
          )
          # input_ds timedelta is 1 day.
          .rolling(
              prediction_timedelta=pd.to_timedelta(period)
              // pd.to_timedelta(input_time_resolution)
          ).mean()
      )
      # Enact the time offsetting needed to align resample and rolling.
      expected_mean = expected_mean.assign_coords(
          prediction_timedelta=expected_mean.prediction_timedelta
          - pd.to_timedelta(period)
          + pd.to_timedelta(input_time_resolution)
      )
    else:
      raise ValueError(f'Unhandled {method=}')

    expected_chunks = input_chunks.copy()
    del expected_chunks['dim_to_drop']
    if method == 'resample':
      expected_chunks['prediction_timedelta'] = min(
          len(expected_mean.prediction_timedelta),
          expected_chunks['prediction_timedelta'],
      )
    self.assertEqual(expected_chunks, output_chunks)

    expected_varnames = []

    for k in mean_vars:
      expected_varnames.append(k + '_mean' if add_mean_suffix else k)
      xr.testing.assert_allclose(
          expected_mean[k],
          output_ds[k + '_mean' if add_mean_suffix else k],
      )

    for k in min_vars:
      expected_varnames.append(k + '_min')
      if period != input_time_resolution:
        np.testing.assert_array_less(output_ds[k + '_min'], expected_mean[k])

    for k in max_vars:
      expected_varnames.append(k + '_max')
      if period != input_time_resolution:
        np.testing.assert_array_less(expected_mean[k], output_ds[k + '_max'])

    self.assertCountEqual(expected_varnames, output_ds.data_vars)


if __name__ == '__main__':
  absltest.main()
