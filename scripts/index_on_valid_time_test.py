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
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import pandas as pd
import xarray
import xarray_beam

from . import index_on_valid_time


class InitToValidTimeTest(parameterized.TestCase):

  @parameterized.parameters(
      {'chunksize': 1},
      {'chunksize': 2},
  )
  def test_no_timedelta_offset_index_on_timedelta(self, chunksize):
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds = xarray.Dataset(
        {
            'foo': (
                ('time', 'prediction_timedelta'),
                np.array(
                    [
                        [1, 2, 3, 4, 5],
                        [6, 7, 8, 9, 10],
                        [11, 12, 13, 14, 15],
                        [16, 17, 18, 19, 20],
                    ]
                ),
            ),
        },  # fmt: skip
        coords={
            'time': pd.date_range('2000-01-01T00', periods=4, freq='12h'),
            'prediction_timedelta': pd.timedelta_range('0h', '24h', freq='6h'),
        },  # fmt: skip
    )
    input_ds.chunk(chunksize).to_zarr(input_path)

    expected_ds = xarray.Dataset(
        {
            'foo': (
                ('time', 'prediction_timedelta'),
                np.array(
                    [
                        [1, np.nan, np.nan],
                        [6, 3, np.nan],
                        [11, 8, 5],
                        [16, 13, 10],
                        [np.nan, 18, 15],
                        [np.nan, np.nan, 20],
                    ]
                ),
            ),
        },  # fmt: skip
        coords={
            'time': pd.date_range('2000-01-01T00', periods=6, freq='12h'),
            'prediction_timedelta': pd.timedelta_range('0h', '24h', freq='12h'),
        },  # fmt: skip
    )
    expected_chunks = {'time': chunksize, 'prediction_timedelta': chunksize}

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
        runner='DirectRunner',
    ):
      index_on_valid_time.main([])

    actual_ds, actual_chunks = xarray_beam.open_zarr(output_path)
    xarray.testing.assert_allclose(actual_ds, expected_ds)
    self.assertEqual(actual_chunks, expected_chunks)

  @parameterized.parameters(
      {'chunksize': 1},
      {'chunksize': 2},
  )
  def test_no_timedelta_offset_index_on_init(self, chunksize):
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds = xarray.Dataset(
        {
            'foo': (
                ('time', 'prediction_timedelta'),
                np.array([
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                ]),
            ),
        },  # fmt: skip
        coords={
            # Different init times will have different valid times, due to
            # timedelta = 6 < 12 = init time frequency.
            'time': pd.date_range('2000-01-01T00', periods=4, freq='12h'),
            'prediction_timedelta': pd.timedelta_range('0h', '24h', freq='6h'),
        },  # fmt: skip
    )
    input_ds.chunk(chunksize).to_zarr(input_path)

    nan = np.nan
    expected_ds = xarray.Dataset(
        {
            'foo': (
                ('init', 'time'),
                np.array([
                    [1, 2, 3, 4, 5, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 6, 7, 8, 9, 10, nan, nan, nan, nan],
                    [nan, nan, nan, nan, 11, 12, 13, 14, 15, nan, nan],
                    [nan, nan, nan, nan, nan, nan, 16, 17, 18, 19, 20],
                ]),
            ),
        },  # fmt: skip
        coords={
            'init': pd.date_range('2000-01-01T00', periods=4, freq='12h'),
            'time': pd.date_range('2000-01-01T00', periods=11, freq='6h'),
        },  # fmt: skip
    )
    expected_chunks = {'time': chunksize, 'init': chunksize}

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
        desired_time_dims='valid_and_init',
        runner='DirectRunner',
    ):
      index_on_valid_time.main([])

    actual_ds, actual_chunks = xarray_beam.open_zarr(output_path)
    xarray.testing.assert_allclose(actual_ds, expected_ds)
    self.assertEqual(actual_chunks, expected_chunks)

  @parameterized.parameters(
      {'chunksize': 1},
      {'chunksize': 2},
  )
  def test_with_timedelta_offset_index_on_timedelta(self, chunksize):
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds = xarray.Dataset(
        {
            'foo': (
                ('time', 'prediction_timedelta'),
                np.array(
                    [
                        [2, 3, 4, 5],
                        [7, 8, 9, 10],
                        [12, 13, 14, 15],
                        [17, 18, 19, 20],
                    ]
                ),
            ),
        },  # fmt: skip
        coords={
            'time': pd.date_range('2000-01-01T00', periods=4, freq='12h'),
            'prediction_timedelta': pd.timedelta_range('6h', '24h', freq='6h'),
        },  # fmt: skip
    )
    input_ds.chunk(chunksize).to_zarr(input_path)

    expected_ds = xarray.Dataset(
        {
            'foo': (
                ('time', 'prediction_timedelta'),
                np.array(
                    [
                        [3, np.nan],
                        [8, 5],
                        [13, 10],
                        [18, 15],
                        [np.nan, 20],
                    ]
                ),
            ),
        },  # fmt: skip
        coords={
            'time': pd.date_range('2000-01-01T12', periods=5, freq='12h'),
            'prediction_timedelta': pd.timedelta_range(
                '12h', '24h', freq='12h'
            ),
        },  # fmt: skip
    )
    expected_chunks = {'time': chunksize, 'prediction_timedelta': chunksize}

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
        runner='DirectRunner',
    ):
      index_on_valid_time.main([])

    actual_ds, actual_chunks = xarray_beam.open_zarr(output_path)
    xarray.testing.assert_allclose(actual_ds, expected_ds)
    self.assertEqual(actual_chunks, expected_chunks)

  @parameterized.parameters(
      {'chunksize': 1},
      {'chunksize': 2},
  )
  def test_with_timedelta_offset_index_on_init(self, chunksize):
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds = xarray.Dataset(
        {
            'foo': (
                ('time', 'prediction_timedelta'),
                np.array(
                    [
                        [2, 3, 4, 5],
                        [7, 8, 9, 10],
                        [12, 13, 14, 15],
                        [17, 18, 19, 20],
                    ]
                ),
            ),
        },  # fmt: skip
        coords={
            'time': pd.date_range('2000-01-01T00', periods=4, freq='12h'),
            'prediction_timedelta': pd.timedelta_range('6h', '24h', freq='6h'),
        },  # fmt: skip
    )
    input_ds.chunk(chunksize).to_zarr(input_path)

    nan = np.nan
    expected_ds = xarray.Dataset(
        {
            'foo': (
                ('init', 'time'),
                np.array([
                    [2, 3, 4, 5, nan, nan, nan, nan, nan, nan],
                    [nan, nan, 7, 8, 9, 10, nan, nan, nan, nan],
                    [nan, nan, nan, nan, 12, 13, 14, 15, nan, nan],
                    [nan, nan, nan, nan, nan, nan, 17, 18, 19, 20],
                ]),
            ),
        },  # fmt: skip
        coords={
            'init': pd.date_range('2000-01-01T00', periods=4, freq='12h'),
            'time': pd.date_range('2000-01-01T06', periods=10, freq='6h'),
        },  # fmt: skip
    )
    expected_chunks = {'time': chunksize, 'init': chunksize}

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
        desired_time_dims='valid_and_init',
        runner='DirectRunner',
    ):
      index_on_valid_time.main([])

    actual_ds, actual_chunks = xarray_beam.open_zarr(output_path)
    xarray.testing.assert_allclose(actual_ds, expected_ds)
    self.assertEqual(actual_chunks, expected_chunks)


if __name__ == '__main__':
  absltest.main()
