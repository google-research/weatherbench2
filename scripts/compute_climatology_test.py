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
import xarray as xr

from . import compute_climatology as cc


class SEEPSThresholdTest(absltest.TestCase):

  def testSEEPSThreshold(self):
    dataset = xr.Dataset(
        {
            'total_precipitation_6hr': xr.DataArray([0, 0.25, 2, 4, 6]) / 1000,
        }
    )

    result = cc.SEEPSThreshold(0.25, 'total_precipitation_6hr').compute(
        dataset,
        dim=...,
    )

    expected = xr.Dataset(
        {
            'total_precipitation_6hr_seeps_dry_fraction': xr.DataArray(1 / 5),
            'total_precipitation_6hr_seeps_threshold': xr.DataArray(4 / 1000),
        }
    )
    xr.testing.assert_allclose(result, expected)


if __name__ == '__main__':
  absltest.main()
