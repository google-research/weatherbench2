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
from weatherbench2 import schema
from weatherbench2 import utils
import xarray as xr
import xarray_beam

from . import slice_dataset


class GetSelectionsTest(parameterized.TestCase):

  def test_valid_selections(self):
    sel = slice_dataset._get_selections(
        flag_values={
            'A_start': '1 day',
            'A_stop': '10 days',
            'A_step': 2,
            'B_stop': 2.2,
            'C_step': 3,
            'D_list': 'planes+trains+automobiles',
        },
        force_string=False,
    )
    expected_sel = [
        {'A': slice('1 day', '10 days', 2)},
        {'B': slice(None, 2.2, None)},
        {'C': slice(None, None, 3)},
        {'D': ['planes', 'trains', 'automobiles']},
    ]
    self.assertCountEqual(expected_sel, sel)

  def test_valid_selections_is_sel_or_dropsel(self):
    sel = slice_dataset._get_selections(
        flag_values={
            'A_start': '1 day',
            'A_stop': '10 days',
            'A_step': 2,
            'B_stop': 2020,  # As in the year 2020 for a date
            'D_list': 'planes+trains+automobiles',
        },
        force_string=True,
    )
    expected_sel = [
        {'A': slice('1 day', '10 days', 2)},
        {'B': slice(None, '2020', None)},
        {'D': ['planes', 'trains', 'automobiles']},
    ]
    self.assertCountEqual(expected_sel, sel)

  def test_valid_index_selections(self):
    isel = slice_dataset._get_selections(
        flag_values={
            'A_list': '9+-1+0',
            'B_list': '-1+-2+02',
            'X_start': 0,
            'X_stop': 10,
            'X_step': 2,
            'Y_stop': 4,
            'Z_start': 1,
            'W_step': 2,
        },
        force_string=False,
    )
    expected_isel = [
        {'A': [9, -1, 0]},
        {'B': [-1, -2, 2]},
        {'X': slice(0, 10, 2)},
        {'Y': slice(None, 4, None)},
        {'Z': slice(1, None, None)},
        {'W': slice(None, None, 2)},
    ]
    self.assertCountEqual(expected_isel, isel)

  def test_invalid_placement_raises(self):
    with self.subTest('Not ending in (start|stop|step|list) raises'):
      with self.assertRaisesRegex(ValueError, 'did not end in'):
        slice_dataset._get_selections(
            flag_values={
                'X_start': 0,
                'X_stop': 10,
                'X_bad': 2,
            },
            force_string=False,
        )

    with self.subTest('Not ending in (start|stop|step|list) raises 2'):
      with self.assertRaisesRegex(ValueError, 'did not end in'):
        slice_dataset._get_selections(
            flag_values={
                'X_start': 0,
                'X_stop': 10,
                'X_step_and_more': 2,
            },
            force_string=False,
        )

    with self.subTest('Not ending in (start|stop|step|list) raises 2'):
      with self.assertRaisesRegex(ValueError, 'did not end in'):
        slice_dataset._get_selections(
            flag_values={
                'X_start': 0,
                'X_stop': 10,
                'X_step_': 2,
            },
            force_string=False,
        )


class SliceDatasetTest(parameterized.TestCase):

  def test_simple_slicing(self):
    input_ds = utils.random_like(
        schema.mock_truth_data(
            variables_2d=[],
            variables_3d=['temperature', 'geopotential', 'should_drop'],
            # time_start/stop for the raw data is wider than the times for
            # resampled data.
            time_start='2021-01-01',
            time_stop='2022-01-01',
            spatial_resolution_in_degrees=30.0,
            time_resolution='1d',
        )
    )

    # Make variables different so we test that variables are handled
    # individually.
    input_ds = input_ds.assign({'geopotential': input_ds.geopotential + 10})

    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_chunks = {'time': 40, 'longitude': 6, 'latitude': 5, 'level': 3}

    # Reverse it so we can make it right in the script.
    input_ds.sel(latitude=input_ds.latitude.data[::-1]).chunk(
        input_chunks
    ).to_zarr(input_path)

    with flagsaver.as_parsed(
        input_path=input_path,
        output_path=output_path,
        output_chunks='level=1',
        sel=(
            # Note that time_step is an integer, since pandas requires this.
            'time_start=2021-02-01,time_stop=2021-04-01,time_step=5,'
            'longitude_step=60'
        ),
        isel='latitude_stop=5',
        drop_variables='should_drop',
        make_dims_increasing='latitude',
        runner='DirectRunner',
    ):
      slice_dataset.main([])

    output_ds, output_chunks = xarray_beam.open_zarr(output_path)
    expected_output_ds = input_ds.sel(
        time=slice('2021-02-01', '2021-04-01', 5),
        longitude=slice(None, None, 60),
    ).isel(latitude=slice(5))[['temperature', 'geopotential']]
    xr.testing.assert_equal(output_ds, expected_output_ds)

    expected_output_chunks = {
        'time': min(input_chunks['time'], output_ds.sizes['time']),
        'longitude': min(
            input_chunks['longitude'], output_ds.sizes['longitude']
        ),
        'latitude': min(input_chunks['latitude'], output_ds.sizes['latitude']),
        'level': 1,  # level was explicitly specified
    }
    self.assertEqual(expected_output_chunks, output_chunks)

  def test_slicing_with_lists_and_dropping(self):
    input_ds = utils.random_like(
        schema.mock_truth_data(
            variables_2d=[],
            variables_3d=['temperature', 'geopotential', 'should_drop'],
            # time_start/stop for the raw data is wider than the times for
            # resampled data.
            time_start='2021-01-01',
            time_stop='2022-01-01',
            spatial_resolution_in_degrees=30.0,
            time_resolution='1d',
        )
    )

    # Make variables different so we test that variables are handled
    # individually.
    input_ds = input_ds.assign({'geopotential': input_ds.geopotential + 10})

    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_chunks = {'time': 40, 'longitude': 6, 'latitude': 5, 'level': 3}
    input_ds.chunk(input_chunks).to_zarr(input_path)

    with flagsaver.as_parsed(
        input_path=input_path,
        output_path=output_path,
        output_chunks='level=1',
        sel='longitude_list=60+150',
        drop_isel='latitude_list=-1',
        drop_variables='should_drop',
        runner='DirectRunner',
    ):
      slice_dataset.main([])

    output_ds, output_chunks = xarray_beam.open_zarr(output_path)
    expected_output_ds = input_ds.sel(
        longitude=[60, 150],
    ).drop_isel(
        latitude=[-1]
    )[['temperature', 'geopotential']]
    xr.testing.assert_equal(output_ds, expected_output_ds)

    expected_output_chunks = {
        'time': min(input_chunks['time'], output_ds.sizes['time']),
        'longitude': min(
            input_chunks['longitude'], output_ds.sizes['longitude']
        ),
        'latitude': min(input_chunks['latitude'], output_ds.sizes['latitude']),
        'level': 1,  # level was explicitly specified
    }
    self.assertEqual(expected_output_chunks, output_chunks)


if __name__ == '__main__':
  absltest.main()
