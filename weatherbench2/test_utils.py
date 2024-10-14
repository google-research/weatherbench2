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
"""Testing utilities."""
from typing import Any

import numpy as np
import xarray as xr


def assert_strictly_decreasing(x: Any, axis=-1, err_msg: str = '') -> None:
  assert_negative(
      np.diff(x, axis=axis),
      err_msg=f'Was not strictly decreasing. {err_msg}',
  )


def assert_strictly_increasing(x: Any, axis=-1, err_msg: str = '') -> None:
  assert_positive(
      np.diff(x, axis=axis),
      err_msg=f'Was not strictly increasing. {err_msg}',
  )


def assert_positive(x: Any, err_msg: str = '') -> None:
  np.testing.assert_array_less(
      0,
      x,
      err_msg=f'Was not positive. {err_msg}',
  )


def assert_negative(x: Any, err_msg: str = '') -> None:
  np.testing.assert_array_less(
      x,
      0,
      err_msg=f'Was not negative. {err_msg}',
  )


def insert_nan(
    ds: xr.Dataset, frac_nan: float = 0.1, seed=802701
) -> xr.Dataset:
  """Copy ds with NaN inserted in every variable."""
  ds = ds.copy()
  rng = np.random.RandomState(seed)
  for name in ds:
    data = ds[name].data
    mask = rng.rand(*data.shape) < frac_nan
    data = np.where(mask, np.nan, data)
    ds[name] = ds[name].copy(data=data)
  return ds
