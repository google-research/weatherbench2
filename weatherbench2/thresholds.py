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
"""Thresholding classes for discrete probabilistic metrics."""

from collections import abc
import dataclasses
import typing

from scipy import stats
import xarray as xr


def _get_climatology_mean(
    climatology: xr.Dataset, variables: abc.Sequence[str]
) -> xr.Dataset:
  """Returns the climatological mean of the given variables."""
  try:
    climatology_mean = climatology[variables]
  except KeyError as e:
    not_found = set(variables).difference(climatology.data_vars)
    clim_var_dict = {
        var + "_mean": var for var in variables
    }  # pytype: disable=unsupported-operands
    not_found_means = set(clim_var_dict).difference(climatology.data_vars)
    if not_found and not_found_means:
      raise KeyError(
          f"Did not find {not_found} keys in climatology. Appending "
          "'mean' did not help."
      ) from e
    climatology_mean = climatology[list(clim_var_dict.keys())].rename(
        clim_var_dict
    )
  return typing.cast(xr.Dataset, climatology_mean)


def _get_climatology_std(
    climatology: xr.Dataset, variables: abc.Sequence[str]
) -> xr.Dataset:
  """Returns the climatological standard deviation of the given variables."""
  clim_std_dict = {
      key + "_std": key for key in variables
  }  # pytype: disable=unsupported-operands
  try:
    climatology_std = climatology[list(clim_std_dict.keys())].rename(
        clim_std_dict
    )
    return typing.cast(xr.Dataset, climatology_std)
  except KeyError as e:
    not_found_stds = set(clim_std_dict).difference(climatology.data_vars)
    raise KeyError(f"Did not find {not_found_stds} keys in climatology.") from e


def _get_climatology_quantile(
    climatology: xr.Dataset,
    variables: abc.Sequence[str],
    quantile: typing.Union[abc.Sequence[float], float],
    atol: float = 0.01,
) -> xr.Dataset:
  """Returns climatological quantiles of the given variables."""
  clim_q_dict = {key + "_quantile": key for key in variables}
  try:
    climatology_q = climatology[list(clim_q_dict.keys())].rename(clim_q_dict)
  except KeyError as e:
    not_found_qs = set(clim_q_dict).difference(climatology.data_vars)
    raise KeyError(f"Did not find {not_found_qs} keys in climatology.") from e

  try:
    climatology_q = climatology_q.sel(
        quantile=quantile, tolerance=atol, method="nearest"
    )
  except KeyError as e:
    raise KeyError(
        f"Did not find quantiles {quantile}+-{atol} in climatology."
        " Consider increasing the tolerance or recomputing the climatology."
    ) from e
  return typing.cast(xr.Dataset, climatology_q)


@dataclasses.dataclass
class Threshold:
  """Threshold for discrete probabilistic metric evaluation.

  Attributes:
    climatology: Dataset containing information about the climatological
      distribution.
    quantile: The quantile to be evaluated.
  """

  climatology: xr.Dataset
  quantile: float

  def compute(self, truth: xr.Dataset) -> xr.Dataset:
    """Computes the threshold for each true label variable.

    Args:
      truth: A dataset of label variables.

    Returns:
      A dataset containing thresholds for all label variables.
    """
    raise NotImplementedError


@dataclasses.dataclass
class QuantileThreshold(Threshold):
  """Quantile threshold for discrete probabilistic metrics.

  Attributes:
    climatology: Dataset containing quantiles of the climatological
      distribution.
    quantile: The climatological quantile to be evaluated.
  """

  climatology: xr.Dataset
  quantile: float

  def compute(self, truth: xr.Dataset) -> xr.Dataset:
    """Computes the threshold for each label variable."""
    if "time" in truth.dims:
      time_dim = "time"
    else:
      time_dim = "valid_time"

    climatology_chunk = self.climatology
    if "level" in truth.dims:
      climatology_chunk = climatology_chunk.sel(level=truth.level)

    time_selection = dict(dayofyear=truth["time"].dt.dayofyear)
    if "hour" in climatology_chunk.dims:
      time_selection["hour"] = truth[time_dim].dt.hour
    climatology_chunk = climatology_chunk.sel(time_selection).compute()

    variables = [str(key) for key in truth.keys()]
    return _get_climatology_quantile(
        climatology_chunk, variables, self.quantile
    )


@dataclasses.dataclass
class GaussianQuantileThreshold(Threshold):
  """Gaussian quantile threshold for discrete probabilistic metrics.

  Attributes:
    climatology: Dataset containing the mean and standard deviation of the
      climatological distribution.
    quantile: The quantile to be evaluated given a Gaussian approximation of the
      climatological distribution.
  """

  climatology: xr.Dataset
  quantile: float

  def compute(self, truth: xr.Dataset) -> xr.Dataset:
    """Computes the threshold for each label variable."""
    if "time" in truth.dims:
      time_dim = "time"
    else:
      time_dim = "valid_time"

    climatology_chunk = self.climatology
    if "level" in truth.dims:
      climatology_chunk = climatology_chunk.sel(level=truth.level)

    time_selection = dict(dayofyear=truth["time"].dt.dayofyear)
    if "hour" in climatology_chunk.dims:
      time_selection["hour"] = truth[time_dim].dt.hour
    climatology_chunk = climatology_chunk.sel(time_selection).compute()

    variables = [str(key) for key in truth.keys()]
    climatology_mean = _get_climatology_mean(climatology_chunk, variables)
    climatology_std = _get_climatology_std(climatology_chunk, variables)
    threshold = (
        climatology_mean + stats.norm.ppf(self.quantile) * climatology_std
    )
    return threshold


def get_threshold_cls(threshold_method: str) -> type[Threshold]:
  """Returns the threshold class for the given threshold method."""
  if threshold_method == "quantile":
    return QuantileThreshold
  elif threshold_method == "gaussian_quantile":
    return GaussianQuantileThreshold
  else:
    raise NotImplementedError(f"Unknown threshold method: {threshold_method}")
