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


class RegriddingTest(parameterized.TestCase):

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
    )
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

  def test_conservative_longitude_weights(self):
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
    )
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
