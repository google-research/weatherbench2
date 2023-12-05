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
# pyformat: mode=pyink
from absl.testing import absltest
import numpy as np
from weatherbench2 import metrics
from weatherbench2 import regions
from weatherbench2 import schema
import xarray as xr


class RegionsTest(absltest.TestCase):

  def testLandRegion(self):
    # Test that non-land regions are not considered in metric computation
    forecast = schema.mock_forecast_data(
        variables_3d=[],
        variables_2d=['2m_temperature'],
        time_start='2022-01-01',
        time_stop='2022-01-02',
        lead_stop='0 day',
    )
    truth = schema.mock_truth_data(
        variables_3d=[],
        variables_2d=['2m_temperature'],
        time_start='2022-01-01',
        time_stop='2022-01-02',
    )
    forecast = forecast.where(forecast.latitude > 0, 1)
    lsm = xr.zeros_like(forecast['2m_temperature'].squeeze())
    lsm = lsm.where(lsm.latitude < 1.0, 1)
    land_region = regions.LandRegion(lsm)

    rmse = metrics.RMSESqrtBeforeTimeAvg()

    results = rmse.compute(forecast, truth, region=land_region)
    np.testing.assert_allclose(results['2m_temperature'].values, 0.0)


if __name__ == '__main__':
  absltest.main()
