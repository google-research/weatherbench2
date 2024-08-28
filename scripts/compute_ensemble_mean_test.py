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
"""Tests for ensemble_mean."""

from absl.testing import absltest
from absl.testing import flagsaver
from weatherbench2 import schema
from weatherbench2 import utils
import xarray
from xarray_beam._src import test_util

from . import compute_ensemble_mean


class EnsembleMeanTest(test_util.TestCase):

  def test_keeping_all_variables(self):
    # By default, all variables are kept.
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds = utils.random_like(schema.mock_forecast_data(ensemble_size=3))

    input_ds.chunk({'time': 31}).to_zarr(input_path)

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
    ):
      compute_ensemble_mean.main([])

    output_ds = xarray.open_zarr(output_path)

    xarray.testing.assert_allclose(
        output_ds, input_ds.mean(compute_ensemble_mean.REALIZATION)
    )

  def test_keep_some_variables(self):
    input_path = self.create_tempdir('source').full_path
    output_path = self.create_tempdir('destination').full_path

    input_ds = utils.random_like(schema.mock_forecast_data(ensemble_size=3))

    all_vars = list(input_ds.data_vars)
    self.assertGreaterEqual(len(all_vars), 2)  # Else test is trivial
    variables_to_get_mean_of = all_vars[:-1]

    input_ds.chunk({'time': 31}).to_zarr(input_path)

    with flagsaver.flagsaver(
        input_path=input_path,
        output_path=output_path,
        variables=variables_to_get_mean_of,
    ):
      compute_ensemble_mean.main([])

    output_ds = xarray.open_zarr(output_path)

    xarray.testing.assert_allclose(
        output_ds,
        input_ds[variables_to_get_mean_of].mean(
            compute_ensemble_mean.REALIZATION
        ),
    )


if __name__ == '__main__':
  absltest.main()
