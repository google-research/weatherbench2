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
"""Definition of evaluation metric classes.

Contains classes for all evaluation metrics used for WB2.
"""
from collections.abc import Sequence
import dataclasses
import functools
import logging
import typing as t

import numpy as np
from scipy import stats
from weatherbench2 import thresholds
from weatherbench2 import utils
from weatherbench2.regions import Region
import xarray as xr

REALIZATION = "realization"


def _assert_increasing(x: np.ndarray):
  if not (np.diff(x) > 0).all():
    raise ValueError(f"array is not increasing: {x}")


def _latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
  pi_over_2 = np.array([np.pi / 2], dtype=x.dtype)
  return np.concatenate([-pi_over_2, (x[:-1] + x[1:]) / 2, pi_over_2])


def _cell_area_from_latitude(points: np.ndarray) -> np.ndarray:
  """Calculate the area overlap as a function of latitude."""
  bounds = _latitude_cell_bounds(points)
  _assert_increasing(bounds)
  upper = bounds[1:]
  lower = bounds[:-1]
  # normalized cell area: integral from lower to upper of cos(latitude)
  return np.sin(upper) - np.sin(lower)


def get_lat_weights(ds: xr.Dataset) -> xr.DataArray:
  """Computes latitude/area weights from latitude coordinate of dataset."""
  weights = _cell_area_from_latitude(np.deg2rad(ds.latitude.data))
  weights /= np.mean(weights)
  weights = ds.latitude.copy(data=weights)
  return weights


def _get_climatology_chunk(
    climatology: xr.Dataset, truth: xr.Dataset
) -> xr.Dataset:
  """Returns the climatological mean of the observed true variables."""
  try:
    climatology_chunk = climatology[list(truth.keys())]
  except KeyError as e:
    not_found = set(truth.keys()).difference(climatology.data_vars)
    clim_var_dict = {str(key) + "_mean": key for key in truth.keys()}
    not_found_means = set(clim_var_dict).difference(climatology.data_vars)
    if not_found and not_found_means:
      raise KeyError(
          f"Did not find {not_found} keys in climatology. Appending "
          "'mean' did not help."
      ) from e
    climatology_chunk = climatology[list(clim_var_dict.keys())].rename(
        clim_var_dict
    )
  return climatology_chunk


@dataclasses.dataclass
class Metric:
  """Base class for metrics."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """Evaluate this metric on a temporal chunk of data.

    The metric should be evaluated independently for each time (averaging over
    time is performed later, on multiple chunks). Thus `forecast` and `truth`
    chunks should cover the full spatial extent of the data, but not necessarily
    all times.

    Args:
      forecast: dataset of forecasts to evaluate.
      truth: dataset of ground truth. Should have the same variables as
        forecast.
      region: Region class. .apply() method is called inside before spatial
        averaging.
      skipna: Whether to skip NaN values in both forecasts and observations
        during evaluation.

    Returns:
      Dataset with metric results for each variable in forecasts/truth, without
      spatial dimensions (latitude/longitude).
    """
    raise NotImplementedError

  def compute(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """Evaluate this metric on datasets with full temporal coverages."""
    if "time" in forecast.dims:
      avg_dim = "time"
    elif "init_time" in forecast.dims:
      avg_dim = "init_time"
    else:
      raise ValueError(
          f"Forecast has neither valid_time or init_time dimension {forecast}"
      )
    return self.compute_chunk(
        forecast, truth, region=region, skipna=skipna
    ).mean(
        avg_dim,
        skipna=skipna,
    )


def _spatial_average(
    dataset: xr.Dataset,
    region: t.Optional[Region],
    skipna: bool,
) -> xr.Dataset:
  """Compute spatial average after applying region mask.

  Args:
    dataset: Metric dataset as a function of latitude/longitude.
    region: Region object (optional).
    skipna: Skip NaNs in spatial mean.

  Returns:
    dataset: Spatially averaged metric.
  """
  weights = get_lat_weights(dataset)
  if region is not None:
    dataset, weights = region.apply(dataset, weights)
    # ignore NaN/Inf values in regions with zero weight
    dataset = dataset.where(weights > 0, 0)
  return dataset.weighted(weights).mean(
      ["latitude", "longitude"], skipna=skipna
  )


def _spatial_average_l2_norm(
    dataset: xr.Dataset,
    region: t.Optional[Region],
    skipna: bool,
) -> xr.Dataset:
  """Helper function to compute sqrt(spatial_average(ds**2))."""
  return np.sqrt(_spatial_average(dataset**2, region=region, skipna=skipna))


@dataclasses.dataclass
class WindVectorMSE(Metric):
  """Compute wind vector mean square error. See WB2 paper for definition.

  Attributes:
    u_name: Name of U component.
    v_name: Name of V component.
    vector_name: Name of wind vector to be computed.
  """

  u_name: str
  v_name: str
  vector_name: str

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    diff = forecast - truth
    result = _spatial_average(
        diff[self.u_name] ** 2 + diff[self.v_name] ** 2,
        region=region,
        skipna=skipna,
    )
    return result


@dataclasses.dataclass
class WindVectorRMSESqrtBeforeTimeAvg(Metric):
  """Compute wind vector RMSE. See WB2 paper for definition.

  This SqrtBeforeTimeAvg metric takes a square root before any time averaging.
  Most users will prefer to use WindVectorMSE and then take a square root in
  user code after running the evaluate script.

  Attributes:
    u_name: Name of U component.
    v_name: Name of V component.
    vector_name: Name of wind vector to be computed.
  """

  u_name: str
  v_name: str
  vector_name: str

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    mse = WindVectorMSE(
        u_name=self.u_name, v_name=self.v_name, vector_name=self.vector_name
    ).compute_chunk(forecast, truth, region=region, skipna=skipna)
    return np.sqrt(mse)


@dataclasses.dataclass
class RMSESqrtBeforeTimeAvg(Metric):
  """Root mean squared error.

  This SqrtBeforeTimeAvg metric takes a square root before any time averaging.
  Most users will prefer to use MSE and then take a square root in user
  code after running the evaluate script.

  Attributes:
    wind_vector_rmse: Optionally provide list of WindVectorRMSESqrtBeforeTimeAvg
      instances to compute.
  """

  wind_vector_rmse: t.Optional[list[WindVectorRMSESqrtBeforeTimeAvg]] = None

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    results = _spatial_average_l2_norm(
        forecast - truth, region=region, skipna=skipna
    )
    if self.wind_vector_rmse is not None:
      for wv in self.wind_vector_rmse:
        results[wv.vector_name] = wv.compute_chunk(
            forecast,
            truth,
            region=region,
            skipna=skipna,
        )
    return results


@dataclasses.dataclass
class MSE(Metric):
  """Mean squared error.

  Attributes:
    wind_vector_mse: Optionally provide list of WindVectorMSE instances to
      compute.
  """

  wind_vector_mse: t.Optional[list[WindVectorMSE]] = None

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    results = _spatial_average(
        (forecast - truth) ** 2, region=region, skipna=skipna
    )
    if self.wind_vector_mse is not None:
      for wv in self.wind_vector_mse:
        results[wv.vector_name] = wv.compute_chunk(
            forecast,
            truth,
            region=region,
            skipna=skipna,
        )
    return results


@dataclasses.dataclass
class SpatialMSE(Metric):
  """MSE without spatial averaging."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    del skipna  # Ignored
    return (forecast - truth) ** 2


@dataclasses.dataclass
class MAE(Metric):
  """Mean absolute error."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    return _spatial_average(abs(forecast - truth), region=region, skipna=skipna)


@dataclasses.dataclass
class SpatialMAE(Metric):
  """Mean absolute error without spatial averaging."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    del skipna  # Ignored
    return abs(forecast - truth)


@dataclasses.dataclass
class Bias(Metric):
  """Bias."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    return _spatial_average(forecast - truth, region=region, skipna=skipna)


@dataclasses.dataclass
class SpatialBias(Metric):
  """Bias without spatial averaging."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    del skipna  # Ignored
    return forecast - truth


@dataclasses.dataclass
class ACC(Metric):
  """Anomaly correlation coefficient.

  Attribute:
    climatology: Climatology for computing anomalies.
  """

  climatology: xr.Dataset

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    if "init_time" in forecast.dims:
      time_dim = "valid_time"
    else:
      time_dim = "time"
    climatology_chunk = _get_climatology_chunk(self.climatology, truth)
    if hasattr(forecast, "level"):
      climatology_chunk = climatology_chunk.sel(level=forecast.level)
    time_selection = dict(dayofyear=forecast[time_dim].dt.dayofyear)
    if "hour" in set(climatology_chunk.coords):
      time_selection["hour"] = forecast[time_dim].dt.hour
    climatology_chunk = climatology_chunk.sel(time_selection).compute()
    forecast_anom = forecast - climatology_chunk
    truth_anom = truth - climatology_chunk
    return _spatial_average(
        forecast_anom * truth_anom,
        region=region,
        skipna=skipna,
    ) / np.sqrt(
        _spatial_average(forecast_anom**2, region=region, skipna=skipna)
        * _spatial_average(truth_anom**2, region=region, skipna=skipna)
    )


@dataclasses.dataclass
class SpatialSEEPS(Metric):
  """Computes Stable Equitable Error in Probability Space.

  Definition in Rodwell et al. (2010):
  https://www.ecmwf.int/en/elibrary/76205-new-equitable-score-suitable-verifying-precipitation-nwp

  Attributes:
    climatology: climatology dataset containing seeps_threshold [meters] and
      seeps_dry_fraction [0-1] for given precip_name.
    dry_threshold_mm: Dry threhsold in mm, same as used to compute
      climatological values.
    precip_name: Name of precipitation variable, e.g. total_precipitation_24hr.
    min_p1: Mask out values with smaller average dry fraction.
    max_p1: Mask out values with larger average dry fraction.
    p1: Average dry fraction.
  """

  climatology: xr.Dataset
  dry_threshold_mm: float = 0.25
  precip_name: str = "total_precipitation_24hr"
  min_p1: float = 0.1
  max_p1: float = 0.85

  @functools.cached_property
  def p1(self) -> xr.DataArray:
    dry_fraction = self.climatology[f"{self.precip_name}_seeps_dry_fraction"]
    return dry_fraction.mean(("hour", "dayofyear")).compute()

  def _convert_precip_to_seeps_cat(self, ds):
    """Helper function for SEEPS computation. Converts values to categories."""
    wet_threshold = self.climatology[f"{self.precip_name}_seeps_threshold"]
    # Convert to SI units [meters]
    dry_threshold = self.dry_threshold_mm / 1000.0
    da = ds[self.precip_name]
    wet_threshold_for_valid_time = wet_threshold.sel(
        dayofyear=da.valid_time.dt.dayofyear, hour=da.valid_time.dt.hour
    ).load()

    dry = da < dry_threshold
    light = np.logical_and(
        da > dry_threshold, da < wet_threshold_for_valid_time
    )
    heavy = da >= wet_threshold_for_valid_time
    result = xr.concat(
        [dry, light, heavy],
        dim=xr.DataArray(["dry", "light", "heavy"], dims=["seeps_cat"]),
    )
    # Convert NaNs back to NaNs
    result = result.astype("int").where(da.notnull())
    return result

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    del skipna  # Ignored, must be effectively True because of p1 mask.
    forecast_cat = self._convert_precip_to_seeps_cat(forecast)
    truth_cat = self._convert_precip_to_seeps_cat(truth)

    # Compute contingency table
    out = (
        forecast_cat.rename({"seeps_cat": "forecast_cat"})
        * truth_cat.rename({"seeps_cat": "truth_cat"})
    ).compute()

    # Compute scoring matrix
    scoring_matrix = [
        [xr.zeros_like(self.p1), 1 / (1 - self.p1), 4 / (1 - self.p1)],
        [1 / self.p1, xr.zeros_like(self.p1), 3 / (1 - self.p1)],
        [
            1 / self.p1 + 3 / (2 + self.p1),
            3 / (2 + self.p1),
            xr.zeros_like(self.p1),
        ],
    ]
    das = []
    for mat in scoring_matrix:
      das.append(xr.concat(mat, dim=out.truth_cat))
    scoring_matrix = 0.5 * xr.concat(das, dim=out.forecast_cat)
    scoring_matrix = scoring_matrix.compute()

    # Take dot product
    result = xr.dot(out, scoring_matrix, dims=("forecast_cat", "truth_cat"))

    # Mask out p1 thresholds
    result = result.where(self.p1 < self.max_p1, np.nan)
    result = result.where(self.p1 > self.min_p1, np.nan)
    return xr.Dataset({f"{self.precip_name}": result})


@dataclasses.dataclass
class SEEPS(SpatialSEEPS):
  """Spatially averaged SEEPS."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    del skipna  # Ignored, must be effectively True because of p1 mask.
    result = super().compute_chunk(forecast, truth, region)
    return _spatial_average(result, region=region, skipna=True)


################################################################################
# Probabilistic metrics.
################################################################################


def _debiased_ensemble_mean_mse(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    ensemble_dim: str,
    skipna: bool,
) -> xr.Dataset:
  """Debiased estimate of E(forecast.mean() - truth)².

  Suppose we have n iid {Xₖ}, each with mean μ and variance σ², and ground
  truth Y. We wish to estimate M = (Y - μ)² in an unbiased fashion.
  Define
    μ(n)  = (1/n) Sum(Xₖ)
    σ²(n) = (1/(n-1)) Sum((Xₖ - μ(n))²)
    M(n)  = (μ(n) - Y)²
    M̃(n)  = M(n) - (1/n) σ²(n)
  One can check that
    E[M̃(n)] = E[M(n)] - σ²/n
            = E[M] + σ²/n - σ²/n
            = E[M].

  Args:
    forecast: A forecast dataset.
    truth: A ground truth dataset.
    ensemble_dim: Dimension indexing ensembles in the forecast.
    skipna: Whether to skip NaN values in both forecasts and observations during
      evaluation.

  Returns:
    Dataset with debiased (forecast - truth)².
  """
  forecast_mean = forecast.mean(ensemble_dim, skipna=skipna)
  forecast_var = forecast.var(ensemble_dim, skipna=skipna, ddof=1)
  biased_mse = (truth - forecast_mean) ** 2
  return biased_mse - forecast_var / _get_n_ensemble(forecast, ensemble_dim)


def _get_n_ensemble(
    ds: xr.Dataset,
    ensemble_dim: str,
    expect_n_ensemble_at_least: int = 1,
) -> int:
  """Returns the size of `ensemble_dim`, optionally asserting size at least."""
  if ensemble_dim not in ds.dims:
    raise ValueError(f"{ensemble_dim=} not found in {ds.dims=}")
  n_ensemble = ds.sizes[ensemble_dim]
  if n_ensemble < expect_n_ensemble_at_least:
    raise ValueError(
        f"{n_ensemble=} is less than expected size of "
        f"{expect_n_ensemble_at_least}"
    )
  return n_ensemble


@dataclasses.dataclass
class EnsembleMetric(Metric):
  """Ensemble metric base class."""

  ensemble_dim: str = REALIZATION

  def _ensemble_slice(self, ds: xr.Dataset, slice_obj: slice) -> xr.Dataset:
    """Slice `ds` and assign coords to [0, ..., len(slice)]."""
    ds = ds.isel({self.ensemble_dim: slice_obj})
    return ds.assign_coords(
        {self.ensemble_dim: np.arange(ds.sizes[self.ensemble_dim])}
    )

  def compute(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """Evaluate this metric on datasets with full temporal coverages."""
    result = super().compute(forecast, truth, region=region, skipna=skipna)
    return result.assign_attrs(ensemble_size=forecast[self.ensemble_dim].size)


@dataclasses.dataclass
class CRPS(EnsembleMetric):
  """Continuous Ranked Probability Score, averaged over space and time.

  Given ground truth scalar random variable Y, and two iid predictions X, X',
  the Continuously Ranked Probability Score is defined as
    CRPS = E|X - Y| - 0.5 * E|X - X'|
  where `E` is mathematical expectation, and |⋅| is absolute value. CRPS has a
  unique minimum when X is distributed the same as Y.

  The associated spread/skill ratio is
    SS(CRPS) = E|X - X'| / E|X - Y|.
  Assuming Y is non-constant, SS(CRPS) = 0 only when X is constant. Since X, X'
  are independent, |X - Y| < |X - Y| + |X - Y|, and thus 0 ≤ SS(CRPS) < 2.
  If X has the same distribution as Y, SS(CRPS) = 1. Caution, it is possible for
  SS(CRPS) = 1 even when X and Y have different distributions.

  CRPS for multi-dimensional random variables is computed as a weighted average
  over components. The minimum is achieved by any prediction X with the correct
  marginals.

  In our case, each prediction is conditioned on the start time t. Given T
  different start times, this class estimates time and ensemble averaged
  quantities for each tendency "V", producing entries
    V_spread := (1 / T) Σₜ ‖Xₜ - Xₜ'‖
    V_skill  := (1 / T) Σₜ ‖Xₜ - Yₜ‖
    V_score  := V_skill - 0.5 * V_spread
  ‖⋅‖ is the area-averaged L1 norm. Estimation is done separately for each
  tendency, level, and lag time.

  If N ensemble members are available, the ensemble mean is taken using the PWM
  method from [Zamo & Naveau, 2018].

  So long as 2 or more ensemble members are used, the estimates of spread, skill
  and CRPS are unbiased at each time. Therefore, assuming some ergodicity, one
  can average over many time points and obtain highly accurate estimates.

  NaN values propagate through and result in NaN in the corresponding output
  position.

  References:
  [Gneiting & Raftery, 2012], Strictly Proper Scoring Rules, Prediction, and
    Estimation
  [Zamo & Naveau, 2018], Estimation of the Continuous Ranked Probability Score
    with Limited Information and Applications to Ensemble Weather Forecasts.
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """CRPS, averaged over space, for a time chunk of data."""
    return CRPSSkill(self.ensemble_dim).compute_chunk(
        forecast,
        truth,
        region=region,
        skipna=skipna,
    ) - 0.5 * CRPSSpread(self.ensemble_dim).compute_chunk(
        forecast,
        truth,
        region=region,
        skipna=skipna,
    )


@dataclasses.dataclass
class CRPSSpread(EnsembleMetric):
  """The spread measure associated with CRPS, E|X - X'|."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """CRPSSpread, averaged over space, for a time chunk of data."""
    return _spatial_average(
        _pointwise_crps_spread(forecast, self.ensemble_dim, skipna=skipna),
        region=region,
        skipna=skipna,
    )


@dataclasses.dataclass
class CRPSSkill(EnsembleMetric):
  """The skill measure associated with CRPS, E|X - Y|."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """CRPSSkill, averaged over space, for a time chunk of data."""
    return _spatial_average(
        _pointwise_crps_skill(
            forecast, truth, self.ensemble_dim, skipna=skipna
        ),
        region=region,
        skipna=skipna,
    )


@dataclasses.dataclass
class SpatialCRPS(EnsembleMetric):
  """CRPS without spatial averaging."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """CRPS, averaged over space, for a time chunk of data."""
    return SpatialCRPSSkill(self.ensemble_dim).compute_chunk(
        forecast,
        truth,
        region=region,
        skipna=skipna,
    ) - 0.5 * SpatialCRPSSpread(self.ensemble_dim).compute_chunk(
        forecast,
        truth,
        region=region,
        skipna=skipna,
    )


@dataclasses.dataclass
class SpatialCRPSSpread(EnsembleMetric):
  """CRPSSpread without spatial averaging."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """CRPSSpread, averaged over space, for a time chunk of data."""
    return _pointwise_crps_spread(forecast, self.ensemble_dim, skipna=skipna)


@dataclasses.dataclass
class SpatialCRPSSkill(EnsembleMetric):
  """CRPSSkill without spatial averaging."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """CRPSSkill, averaged over space, for a time chunk of data."""
    return _pointwise_crps_skill(
        forecast, truth, self.ensemble_dim, skipna=skipna
    )


@utils.dataset_safe_lru_cache(
    # This is used in _metric_and_region_loop. The same dataset is used
    # repeatedly for different metrics/regions, then the loop returns.
    # Therefore, maxsize=1 is sufficient.
    maxsize=1,
)
def _pointwise_crps_spread(
    forecast: xr.Dataset,
    ensemble_dim: str,
    skipna: bool,
) -> xr.Dataset:
  """CRPS spread at each point in truth, averaged over ensemble only."""
  n_ensemble = _get_n_ensemble(forecast, ensemble_dim)
  if n_ensemble < 2:
    return xr.zeros_like(forecast.isel({ensemble_dim: 0}))

  # one_half_spread is ̂̂λ₂ from Zamo. That is, with n_ensemble = M,
  #   λ₂ = 1 / (2 M (M - 1)) Σ_{i,j=1}^M |Xi - Xj|
  # See the definition of eFAIR and then
  # eqn 3 (appendix B), which shows that this double summation of absolute
  # differences can be written as a sum involving sorted elements multiplied
  # by their index. That is, if X1 < X2 < ... < XM,
  #   λ₂ = 1 / (M(M-1)) Σ_{i,j=1}^M (2*i - M - 1) Xi.
  # The term (2*i - M - 1) is +1 times the number of terms Xi is greater than,
  # and -1 times the number of terms Xi is less than.
  # Here we do not sort, but instead compute the rank of each element, multiply
  # appropriately, then sum. We prefer this second form, since it involves an
  # O(M Log[M]) compute and O(M) memory usage, whereas the first is O(M²) in
  # compute and memory.
  rank = _rank_ds(forecast, ensemble_dim)
  return (
      2
      * (
          ((2 * rank - n_ensemble - 1) * forecast).mean(
              ensemble_dim, skipna=skipna
          )
      )
      / (n_ensemble - 1)
  )


def _pointwise_crps_skill(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    ensemble_dim: str,
    skipna: bool,
) -> xr.Dataset:
  """CRPS skill at each point in truth, averaged over ensemble only."""
  _get_n_ensemble(forecast, ensemble_dim)  # Will raise if no ensembles.
  return abs(truth - forecast).mean(ensemble_dim, skipna=skipna)


def _rank_ds(ds: xr.Dataset, dim: str) -> xr.Dataset:
  """The ranking of `ds` along `dim`, with 1 being the smallest entry."""

  def _rank_da(da: xr.DataArray) -> np.ndarray:
    return _rankdata(da.values, axis=da.dims.index(dim))

  return ds.copy(data={k: _rank_da(v) for k, v in ds.items()})


def _rankdata(x: np.ndarray, axis: int) -> np.ndarray:
  """Version of (ordinal) scipy.rankdata from V13."""
  x = np.asarray(x)
  x = np.swapaxes(x, axis, -1)
  j = np.argsort(x, axis=-1)
  ordinal_ranks = np.broadcast_to(
      np.arange(1, x.shape[-1] + 1, dtype=int), x.shape
  )
  ordered_ranks = np.empty(j.shape, dtype=ordinal_ranks.dtype)
  np.put_along_axis(ordered_ranks, j, ordinal_ranks, axis=-1)
  return np.swapaxes(ordered_ranks, axis, -1)


@dataclasses.dataclass
class GaussianCRPS(Metric):
  """The analytical formulation of CRPS for a Gaussian."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """GaussianCRPS, averaged over space, for a time chunk of data."""
    return _spatial_average(
        _pointwise_gaussian_crps(forecast, truth),
        region=region,
        skipna=skipna,
    )


def _pointwise_gaussian_crps(
    forecast: xr.Dataset, truth: xr.Dataset
) -> xr.Dataset:
  r"""Returns pointwise CRPS of a Gaussian distribution with mean and std values.

  CRPS of a Gaussian distribution with mean value m and standard deviation s
  can be computed as

  CRPS(F_(m,s), y) = s * {(y-m)/s * [2G((y-m)/s) - 1] + 2g((y-m)/s) -
  1/\sqrt(\pi))}

  where G and g denote the CDF and PDF of a standard Gaussian distribution,
  respectively.
  References:
  [Gneiting, Raftery, Westveld III, Goldman, 2005], Calibrated Probabilistic
  Forecasting Using Ensemble Model Output Statistics and Minimum CRPS Estimation
  DOI: https://doi.org/10.1175/MWR2904.1

  Args:
    forecast: A forecast dataset.
    truth: A ground truth dataset.

  Returns:
    xr.Dataset: Pointwise calculated crps for a Gaussian distribution.
  """
  var_list = []
  dataset = {}
  for var in forecast.keys():
    if f"{var}_std" in forecast.keys():
      var_list.append(var)
  for var_name in var_list:
    norm_diff = (forecast[var_name] - truth[var_name]) / forecast[
        f"{var_name}_std"
    ]
    value = forecast[f"{var_name}_std"] * (
        norm_diff * (2 * xr.apply_ufunc(stats.norm.cdf, norm_diff.load()) - 1)
        + 2 * xr.apply_ufunc(stats.norm.pdf, norm_diff.load())
        - 1 / np.sqrt(np.pi)
    )
    dataset[var_name] = value
  return xr.Dataset(dataset, coords=forecast.coords)


@dataclasses.dataclass
class GaussianVariance(Metric):
  """The variance of a Gaussian forecast."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """GaussianVariance, averaged over space, for a time chunk of data."""
    del truth  # unused
    var_list = []
    dataset = {}
    for var in forecast.keys():
      if f"{var}_std" in forecast.keys():
        var_list.append(var)
    for var_name in var_list:
      variance = forecast[f"{var_name}_std"] * forecast[f"{var_name}_std"]
      dataset[var_name] = variance

    return _spatial_average(
        xr.Dataset(dataset, coords=forecast.coords),
        region=region,
        skipna=skipna,
    )


@dataclasses.dataclass
class ThresholdMetric(Metric):
  """Base class for metrics based on thresholds."""

  thresholds: Sequence[thresholds.Threshold]

  def _map_over_thresholds(
      self,
      calculate_score: t.Callable[
          [xr.Dataset, xr.Dataset, xr.Dataset], xr.Dataset
      ],
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region],
      skipna: bool,
      spatial_agg: bool,
  ) -> xr.Dataset:
    """Map a function over all thresholds."""
    scores = []
    for threshold in self.thresholds:
      threshold_ds = threshold.compute(truth)
      score = calculate_score(forecast, truth, threshold_ds)
      if spatial_agg:
        score = _spatial_average(score, region=region, skipna=skipna)
      scores.append(score.expand_dims(dim={"quantile": [threshold.quantile]}))
    threshold_method = type(self.thresholds[0]).__name__
    return xr.concat(scores, dim="quantile").assign_attrs(
        threshold_method=threshold_method
    )


def _compute_gaussian_brier_score(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    threshold: xr.Dataset,
) -> xr.Dataset:
  """Computes the Brier score for a Gaussian distribution."""
  truth_probability = xr.where(truth > threshold, 1.0, 0.0)

  var_list = []
  exceedance_probability = {}
  for var in forecast.keys():
    if f"{var}_std" in forecast.keys():
      var_list.append(var)

  for var_name in var_list:
    std = forecast[f"{var_name}_std"]
    norm_threshold = (threshold[var_name] - forecast[var_name]) / std
    exceedance_probability[var_name] = 1 - xr.apply_ufunc(
        stats.norm.cdf, norm_threshold.load()
    )

  forecast_probability = xr.Dataset(
      exceedance_probability, coords=forecast.coords
  )
  return (forecast_probability - truth_probability) ** 2


@dataclasses.dataclass
class GaussianBrierScore(ThresholdMetric):
  """Brier score of a Gaussian forecast for a given binary threshold.

  The Brier score is computed based on the forecast probability of exceedance of
  a given climatological quantile. The true probability is binarized to 0 or 1.

  The Brier score for the binarized event of exceedance of a given threshold is
  equal to the Brier score for the opposite event, i.e., the forecast remaining
  below the threshold.

  References:
  [Ferro, 2007], Comparing Probabilistic Forecasting Systems with the Brier
  Score, DOI: https://doi.org/10.1175/WAF1034.1

  Attribute:
    climatology: Climatology for computing threshold.
    threshold: Climatological quantile used to binarize predictions and targets.

  Returns:
    Spatially averaged Brier score for a Gaussian distribution.
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    return self._map_over_thresholds(
        _compute_gaussian_brier_score,
        forecast,
        truth,
        region=region,
        skipna=skipna,
        spatial_agg=True,
    )


def _compute_gaussian_ignorance_score(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    threshold: xr.Dataset,
) -> xr.Dataset:
  """Computes the Ignorance score for a Gaussian distribution."""
  truth_probability = xr.where(truth > threshold, 1.0, 0.0)

  log_realized_probability = {}
  var_list = [var for var in forecast.keys() if f"{var}_std" in forecast.keys()]

  for var_name in var_list:
    norm_threshold = (threshold[var_name] - forecast[var_name]) / forecast[
        f"{var_name}_std"
    ]
    cdf_value = xr.apply_ufunc(stats.norm.cdf, norm_threshold.load())
    log_realized_probability[var_name] = -xr.where(
        truth_probability[var_name],
        xr.apply_ufunc(np.log, 1 - cdf_value),
        xr.apply_ufunc(np.log, cdf_value),
    )

  ignorance_score = xr.Dataset(log_realized_probability, coords=forecast.coords)
  return ignorance_score


@dataclasses.dataclass
class GaussianIgnoranceScore(ThresholdMetric):
  """Ignorance score of a Gaussian forecast for a given binary threshold.

  The ignorance or logarithmic score is computed based on the forecast
  probability of exceedance of a given climatological quantile. The true
  probability is binarized to 0 or 1.

  References:
  [Benedetti, 2010], Scoring Rules for Forecast Verification,
  DOI: https://doi.org/10.1175/2009MWR2945.1

  Attribute:
    climatology: Climatology for computing threshold.
    threshold: Climatological quantile used to binarize predictions and targets.

  Returns:
    Spatially averaged ignorance score for a Gaussian distribution.
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:

    return self._map_over_thresholds(
        _compute_gaussian_ignorance_score,
        forecast,
        truth,
        region=region,
        skipna=skipna,
        spatial_agg=True,
    )


def _compute_gaussian_rps_part(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    threshold: xr.Dataset,
) -> xr.Dataset:
  """Computes the Ranked Probability Score for a Gaussian distribution."""
  truth_ecdf = xr.where(truth < threshold, 1.0, 0.0)

  var_list = [var for var in forecast.keys() if f"{var}_std" in forecast.keys()]

  cdf_values = {}
  for var_name in var_list:
    std = forecast[f"{var_name}_std"]
    norm_threshold = (threshold[var_name] - forecast[var_name]) / std
    cdf_values[var_name] = xr.apply_ufunc(stats.norm.cdf, norm_threshold.load())

  forecast_cdf = xr.Dataset(cdf_values, coords=forecast.coords)
  return (forecast_cdf - truth_ecdf) ** 2


@dataclasses.dataclass
class GaussianRPS(ThresholdMetric):
  """Ranked probability score of a Gaussian forecast for a given quantization.

  The ranked probability score (RPS) is computed based on the forecast and
  observed cumulative distribution functions. See `EnsembleRPS` for a discussion
  of this metric.

  References:
  [Epstein, 1969] A Scoring System for Probability Forecasts of Ranked
  Categories,
  DOI: https://doi.org/10.1175/1520-0450(1969)008<0985:ASSFPF>2.0.CO;2

  Attribute:
    threshold: A sequence of threshold used to divide predictions and targets
    categorically.

  Returns:
    Spatially averaged RPS for a Gaussian distribution.
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:

    return self._map_over_thresholds(
        _compute_gaussian_rps_part,
        forecast,
        truth,
        region=region,
        skipna=skipna,
        spatial_agg=True,
    ).sum(dim="quantile")


@dataclasses.dataclass
class EnsembleStddevSqrtBeforeTimeAvg(EnsembleMetric):
  """The standard deviation of an ensemble of forecasts.

  This SqrtBeforeTimeAvg metric takes a square root before any time averaging.
  Most users will prefer to use EnsembleVariance and then take a square root in
  user code after running the evaluate script.

  Given predictive ensemble Xₜ at times t = (1,..., T),
    EnsembleStddevSqrtBeforeTimeAvg := (1 / T) Σₜ ‖σ(Xₜ)‖
  Above σ(Xₜ) is element-wise standard deviation, and ‖⋅‖ is an area-weighted
  L2 norm.

  Estimation is done separately for each tendency, level, and lag time.
  Ensembles of size 1 lead SPREAD=0, otherwise the unbiased sample standard
  deviation is used.

  This estimator has a slight amount of bias, since ‖σ(Xₜ)‖ has bias
  proportional to 1 / (ensemble_size * grid_size).

  NaN values propagate through and result in NaN in the corresponding output
  position.
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """Ensemble Stddev, averaged over space, for a time chunk of data."""
    del truth  # unused
    n_ensemble = _get_n_ensemble(forecast, self.ensemble_dim)

    if n_ensemble == 1:
      return xr.zeros_like(
          # Compute the average, even though we return zeros_like. Why? Because,
          # this will preserve the scalar values of lat/lon coords correctly.
          _spatial_average(forecast, region=region, skipna=skipna).mean(
              self.ensemble_dim,
              skipna=skipna,
          )
      )
    else:
      return _spatial_average_l2_norm(
          forecast.std(self.ensemble_dim, ddof=1, skipna=skipna),
          region=region,
          skipna=skipna,
      )


@dataclasses.dataclass
class EnsembleVariance(EnsembleMetric):
  """The variance of an ensemble of forecasts."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """EnsembleVariance, averaged over space, for a time chunk of data."""
    del truth  # unused
    n_ensemble = _get_n_ensemble(forecast, self.ensemble_dim)

    if n_ensemble == 1:
      return xr.zeros_like(
          # Compute the average, even though we return zeros_like. Why? Because,
          # this will preserve the scalar values of lat/lon coords correctly.
          _spatial_average(forecast, region=region, skipna=skipna).mean(
              self.ensemble_dim, skipna=skipna
          )
      )
    else:
      return _spatial_average(
          forecast.var(self.ensemble_dim, ddof=1, skipna=skipna),
          region=region,
          skipna=skipna,
      )


@dataclasses.dataclass
class SpatialEnsembleVariance(EnsembleMetric):
  """Ensemble variance without spatial averaging."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """Ensemble variance, for a time chunk of data."""
    del truth  # unused
    n_ensemble = _get_n_ensemble(forecast, self.ensemble_dim)

    if n_ensemble == 1:
      return xr.zeros_like(
          # Compute the average, even though we return zeros_like. Why? Because,
          # this will preserve the scalar values of lat/lon coords correctly.
          forecast
      ).mean(self.ensemble_dim, skipna=skipna)
    else:
      return forecast.var(self.ensemble_dim, ddof=1, skipna=skipna)


@dataclasses.dataclass
class EnsembleMeanRMSESqrtBeforeTimeAvg(EnsembleMetric):
  """RMSE between the ensemble mean and ground truth.

  This SqrtBeforeTimeAvg metric takes a square root before any time averaging.
  Most users will prefer to use EnsembleMeanMSE and then take a square root in
  user code after running the evaluate script.

  Given ground truth Yₜ, and predictive ensemble Xₜ, both at times
  t = (1,..., T),
    EnsembleMeanRMSESqrtBeforeTimeAvg := (1 / T) Σₜ ‖Y - E(Xₜ)‖.
  Above, `E` is ensemble average, and ‖⋅‖ is an area-weighted L2 norm.

  Estimation is done separately for each tendency, level, and lag time.
  Ensembles of size 1 lead SPREAD=0, otherwise the unbiased sample standard
  deviation is used.

  Unfortunately, this estimator is biased since ‖Y - E(Xₜ)‖ has bias
  proportional to 1 / ensemble_size.

  NaN values propagate through and result in NaN in the corresponding output
  position.
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """EnsembleMeanRMSE, averaged over space, for a time chunk of data."""
    _get_n_ensemble(forecast, self.ensemble_dim)  # Will raise if no ensembles.

    return _spatial_average_l2_norm(
        truth - forecast.mean(self.ensemble_dim, skipna=skipna),
        region=region,
        skipna=skipna,
    )


@dataclasses.dataclass
class EnsembleMeanMSE(EnsembleMetric):
  """Mean square error between the ensemble mean and ground truth.

  Suppose we have a size n ensemble, {Xₖ}, each an iid copy of X having with
  mean μ and variance σ². Let Y be the ground truth.
  This class estimates E(X - Y)² with a bias equal to σ² / n.
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """EnsembleMeanRMSE, averaged over space, for a time chunk of data."""
    _get_n_ensemble(forecast, self.ensemble_dim)  # Will raise if no ensembles.

    return _spatial_average(
        (truth - forecast.mean(self.ensemble_dim, skipna=skipna)) ** 2,
        region=region,
        skipna=skipna,
    )


@dataclasses.dataclass
class DebiasedEnsembleMeanMSE(EnsembleMetric):
  """Debiased mean square error between the ensemble mean and ground truth.

  Suppose we have a size n ensemble, {Xₖ}, each an iid copy of X having with
  mean μ and variance σ². Let Y be the ground truth.
  This class estimates E(X - Y)² with no bias. This is done by subtracting the
  sample variance divided by n. As such, you must have n > 1 or the result will
  be NaN.
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """DebiasedEnsembleMeanMSE, averaged over space, for one time chunk."""
    _get_n_ensemble(forecast, self.ensemble_dim)  # Will raise if no ensembles.

    return _spatial_average(
        _debiased_ensemble_mean_mse(
            forecast, truth, self.ensemble_dim, skipna=skipna
        ),
        region=region,
        skipna=skipna,
    )


@dataclasses.dataclass
class SpatialEnsembleMeanMSE(EnsembleMetric):
  """EnsembleMeanMSE (MSE, not RMSE), without spatial averaging."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """Squared error in the ensemble mean, for a time chunk of data."""
    _get_n_ensemble(forecast, self.ensemble_dim)  # Will raise if no ensembles.

    return (truth - forecast.mean(self.ensemble_dim, skipna=skipna)) ** 2


@dataclasses.dataclass
class DebiasedSpatialEnsembleMeanMSE(EnsembleMetric):
  """DebiasedEnsembleMeanMSE (MSE, not RMSE), without spatial averaging."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """Squared error in the ensemble mean, for a time chunk of data."""
    _get_n_ensemble(forecast, self.ensemble_dim)  # Will raise if no ensembles.

    return _debiased_ensemble_mean_mse(
        forecast, truth, self.ensemble_dim, skipna=skipna
    )


@dataclasses.dataclass
class EnergyScore(EnsembleMetric):
  """The Energy Score along with spread and skill parts.

  Given ground truth random vector Y, and two iid predictions X, X', the
  Energy Score is defined as
    ES = E‖X - Y‖ - 0.5 * E‖X - X'‖
  where `E` is mathematical expectation, and ‖⋅‖ is a weighted L2 norm. ES has a
  unique minimum when X is distributed the same as Y.

  The associated spread/skill ratio is
    SS(ES) = E‖X - X'‖ / E‖X - Y‖.
  Assuming Y is non-constant, SS(ES) = 0 only when X is constant. Since X, X'
  are independent, ‖X - Y‖ < ‖X - Y‖ + ‖X - Y‖, and thus 0 ≤ SS(ES) < 2.
  If X has the same distribution as Y, SS(ES) = 1. Caution, it is possible for
  SS(CRPS) = 1 even when X and Y have different distributions.

  In our case, each prediction is conditioned on the start time t. Given T
  different start times, this class estimates time and ensemble averaged
  quantities for each tendency "V", producing entries
    V_spread := (1 / T) Σₜ ‖Xₜ - Xₜ'‖
    V_skill  := (1 / T) Σₜ ‖Xₜ - Yₜ‖
    V_score  := V_skill - 0.5 * V_spread
  ‖⋅‖ is the area-averaged L2 norm. Estimation is done separately for each
  tendency, level, and lag time. So correlations between tendency/level/lag are
  ignored.

  If N ensemble members are available, we estimate the spread with N-1
  adjacent differences. This strikes a balance between memory usage and variance
  reduction.
    E‖Xₜ - Xₜ'‖ ≈ (1 / (N-1)) Σₙ ‖Xₜ[n] - Xₜ[n+1]‖

  So long as 2 or more ensemble members are used, the estimates of spread, skill
  and ES are unbiased at each time. Therefore, assuming some ergodicity, one can
  average over many time points and obtain highly accurate estimates.

  NaN values propagate through and result in NaN in the corresponding output
  position.

  References:
  [Gneiting & Raftery, 2012], Strictly Proper Scoring Rules, Prediction, and
    Estimation
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """Energy score, averaged over space, for a time chunk of data."""
    return EnergyScoreSkill(self.ensemble_dim).compute_chunk(
        forecast,
        truth,
        region=region,
        skipna=skipna,
    ) - 0.5 * EnergyScoreSpread(self.ensemble_dim).compute_chunk(
        forecast,
        truth,
        region=region,
        skipna=skipna,
    )


@dataclasses.dataclass
class EnergyScoreSpread(EnsembleMetric):
  """The spread measure associated with EnergyScore, E‖X - X'‖."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """Energy score spread, averaged over space, for a time chunk of data."""
    n_ensemble = _get_n_ensemble(forecast, self.ensemble_dim)

    if n_ensemble == 1:
      return xr.zeros_like(
          # Compute the average, even though we return zeros_like. Why? Because,
          # this will preserve the scalar values of lat/lon coords correctly.
          _spatial_average(forecast, region=region, skipna=skipna).mean(
              self.ensemble_dim,
              skipna=skipna,
          )
      )
    else:
      return _spatial_average_l2_norm(
          self._ensemble_slice(forecast, slice(None, -1))
          - self._ensemble_slice(forecast, slice(1, None)),
          region=region,
          skipna=skipna,
      ).mean(self.ensemble_dim, skipna=skipna)


@dataclasses.dataclass
class EnergyScoreSkill(EnsembleMetric):
  """The skill measure associated with EnergyScore, E‖X - Y‖."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """Energy score skill, averaged over space, for a time chunk of data."""
    _get_n_ensemble(forecast, self.ensemble_dim)  # Will raise if no ensembles.
    return _spatial_average_l2_norm(
        forecast - truth, region=region, skipna=skipna
    ).mean(
        self.ensemble_dim,
        skipna=skipna,
    )


# TODO(shoyer): Consider adding WindVectorEnergyScore based on a pair of wind
# components, as a sort of probabilistic variant of WindVectorMSE.


def _compute_brier_score(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    threshold: xr.Dataset,
    ensemble_dim: str,
    debias: bool,
    skipna: bool,
) -> xr.Dataset:
  """Compute the Brier score for a single threshold."""
  # Notice we allow NaN in truth/forecast probabilities, then skipna during
  # computation of BrierScore (which is really just an MSE over the
  # probabilities).
  truth_probability = xr.where(
      truth.isnull(),
      np.nan,
      xr.where(truth > threshold, 1.0, 0.0),
  )
  forecast_probability = xr.where(
      forecast.isnull(),
      np.nan,
      xr.where(forecast > threshold, 1.0, 0.0),
  )
  if debias:
    mse_of_probabilities = _debiased_ensemble_mean_mse(
        forecast_probability,
        truth_probability,
        ensemble_dim,
        skipna=skipna,
    )
  else:
    mse_of_probabilities = (
        forecast_probability.mean(ensemble_dim, skipna=skipna)
        - truth_probability
    ) ** 2
  return mse_of_probabilities


@dataclasses.dataclass
class EnsembleBrierScore(EnsembleMetric, ThresholdMetric):
  """Brier score of an ensemble forecast for a given binary threshold.

  The Brier score is computed based on the forecast probability of exceedance of
  a given climatological quantile. The true probability is binarized to 0 or 1.
  The forecast probability is equal to the proportion of members that exceed
  the quantile.

  The Brier score for the binarized event of exceedance of a given threshold is
  equal to the Brier score for the opposite event, i.e., the forecast remaining
  below the threshold.

  Given threshold σ, consider the Brier score of forecasts at fixed time and
  space. Let Xₖ be the kth ensemble member (out of n total). Let 1{Y > σ} and
  1{Xₖ > σ} be the indicators equal to 1 just when Y > σ and Xₖ > σ. Then,
    EnsembleBrierScore = Bn = ( 1{Y > σ} - (1/n)Σₖ 1{Xₖ > σ} )²

  As the ensemble size n → ∞
    (1/n)Σₖ 1{Xₖ > σ} → Prob[X > σ]
    Bn → B := ( 1{Y > σ} - Prob[X > σ] )²

  For finite ensemble size, the bias is
    E[Bn - B] = Prob[X > σ] (1 - Prob[X > σ]) / n,
  which is just the sample variance divided by n.

  References:
  [Ferro, 2007], Comparing Probabilistic Forecasting Systems with the Brier
  Score, DOI: https://doi.org/10.1175/WAF1034.1
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    return self._map_over_thresholds(
        functools.partial(
            _compute_brier_score,
            ensemble_dim=self.ensemble_dim,
            debias=False,
            skipna=skipna,
        ),
        forecast,
        truth,
        region=region,
        skipna=skipna,
        spatial_agg=True,
    )


@dataclasses.dataclass
class SpatialEnsembleBrierScore(EnsembleMetric, ThresholdMetric):
  """Spatial map of ensemble Brier score."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    return self._map_over_thresholds(
        functools.partial(
            _compute_brier_score,
            ensemble_dim=self.ensemble_dim,
            debias=False,
            skipna=skipna,
        ),
        forecast,
        truth,
        region=region,
        skipna=skipna,
        spatial_agg=False,
    )


@dataclasses.dataclass
class DebiasedEnsembleBrierScore(EnsembleMetric, ThresholdMetric):
  """Debiased Brier score of an ensemble forecast for a given binary threshold.

  The Brier score is computed based on the forecast probability of exceedance of
  a given climatological quantile. The true probability is binarized to 0 or 1.
  The forecast probability is equal to the proportion of members that exceed
  the quantile.

  The Brier score for the binarized event of exceedance of a given threshold is
  equal to the Brier score for the opposite event, i.e., the forecast remaining
  below the threshold.

  Given threshold σ, consider the Brier score of forecasts at fixed time and
  space. Let Xₖ be the kth ensemble member (out of n total). Let 1{Y > σ} and
  1{Xₖ > σ} be the indicators equal to 1 just when Y > σ and Xₖ > σ. Then,
    EnsembleBrierScore = Bn = ( 1{Y > σ} - (1/n)Σₖ 1{Xₖ > σ} )²

  As the ensemble size n → ∞
    (1/n)Σₖ 1{Xₖ > σ} → Prob[X > σ]
    Bn → B := ( 1{Y > σ} - Prob[X > σ] )²

  For finite ensemble size, the bias is
    E[Bn - B] = Prob[X > σ] (1 - Prob[X > σ]) / n,
  which is just the sample variance divided by n.

  For finite ensemble size, we debias the result by subtracting the sample
  variance divided by n. As such, you must have n > 1 or the result will be NaN.

  References:
  [Ferro, 2007], Comparing Probabilistic Forecasting Systems with the Brier
  Score, DOI: https://doi.org/10.1175/WAF1034.1
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    return self._map_over_thresholds(
        functools.partial(
            _compute_brier_score,
            ensemble_dim=self.ensemble_dim,
            debias=True,
            skipna=skipna,
        ),
        forecast,
        truth,
        region=region,
        skipna=skipna,
        spatial_agg=True,
    )


@dataclasses.dataclass
class SpatialDebiasedEnsembleBrierScore(EnsembleMetric, ThresholdMetric):
  """Spatial map of ensemble debiased Brier score."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    return self._map_over_thresholds(
        functools.partial(
            _compute_brier_score,
            ensemble_dim=self.ensemble_dim,
            debias=True,
            skipna=skipna,
        ),
        forecast,
        truth,
        region=region,
        skipna=skipna,
        spatial_agg=False,
    )


def _compute_ignorance_score(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    threshold: xr.Dataset,
    ensemble_dim: str,
    skipna: bool,
) -> xr.Dataset:
  """Compute the Ignorance score for a single threshold."""
  truth_probability = xr.where(truth > threshold, 1.0, 0.0)
  forecast_probability = xr.where(forecast > threshold, 1.0, 0.0)
  ensemble_forecast_probability = forecast_probability.mean(
      ensemble_dim, skipna=skipna
  )
  ignorance_score = -xr.where(
      truth_probability,
      xr.apply_ufunc(np.log, ensemble_forecast_probability),
      xr.apply_ufunc(np.log, 1 - ensemble_forecast_probability),
  )
  return ignorance_score


@dataclasses.dataclass
class EnsembleIgnoranceScore(EnsembleMetric, ThresholdMetric):
  """Ignorance score of an ensemble forecast for a given binary threshold.

  The ignorance or logarithmic score is computed based on the forecast
  probability of exceedance of a given climatological quantile. The true
  probability is binarized to 0 or 1. The forecast probability is equal to the
  proportion of members that exceed the quantile.

  References:
  [Benedetti, 2010], Scoring Rules for Forecast Verification,
  DOI: https://doi.org/10.1175/2009MWR2945.1
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:

    return self._map_over_thresholds(
        functools.partial(
            _compute_ignorance_score,
            ensemble_dim=self.ensemble_dim,
            skipna=skipna,
        ),
        forecast,
        truth,
        region=region,
        skipna=skipna,
        spatial_agg=True,
    )


@dataclasses.dataclass
class SpatialEnsembleIgnoranceScore(EnsembleMetric, ThresholdMetric):
  """Spatial map of ensemble ignorance score."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:

    return self._map_over_thresholds(
        functools.partial(
            _compute_ignorance_score,
            ensemble_dim=self.ensemble_dim,
            skipna=skipna,
        ),
        forecast,
        truth,
        region=region,
        skipna=skipna,
        spatial_agg=False,
    )


def _compute_rps_part(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    threshold: xr.Dataset,
    ensemble_dim: str,
    skipna: bool,
) -> xr.Dataset:
  """Compute the contribution to RPS for a single threshold."""
  truth_ecdf = xr.where(truth < threshold, 1.0, 0.0)
  forecast_ecdf = xr.where(forecast < threshold, 1.0, 0.0)
  ensemble_forecast_ecdf = forecast_ecdf.mean(ensemble_dim, skipna=skipna)
  return (ensemble_forecast_ecdf - truth_ecdf) ** 2


@dataclasses.dataclass
class EnsembleRPS(EnsembleMetric, ThresholdMetric):
  """Ranked probability score of an ensemble forecast for a given quantization.

  The ranked probability score (RPS) is computed based on the forecast and
  observed cumulative distribution functions, coarsened to the level of
  the input thresholds.

  The thresholds are taken to define the limits of each considered interval,
  except the first (resp. last) interval, which includes all values
  lower (resp. higher) than the first (resp. last) threshold. Three thresholds
  would define the following four intervals:

                        <-- | --- | --- | -->

  As an example, if the thresholds are climatological terciles [0.33, 0.66],
  the observed event was at the climatological quantile 0.5, and the ensemble
  forecasts were at climatological quantiles [0.1, 0.5, 0.6, 0.8], then the
  observed CDF would be [0, 1, 1], and the forecast CDF would be
  [0.25, 0.75, 1]. Note that the score over the last interval need not
  be computed, since the quantized CDFs are always one for both forecasts and
  observations there.

  References:
  [Epstein, 1969] A Scoring System for Probability Forecasts of Ranked
  Categories,
  DOI: https://doi.org/10.1175/1520-0450(1969)008<0985:ASSFPF>2.0.CO;2
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:

    result = self._map_over_thresholds(
        functools.partial(
            _compute_rps_part, ensemble_dim=self.ensemble_dim, skipna=skipna
        ),
        forecast,
        truth,
        region=region,
        skipna=skipna,
        spatial_agg=True,
    )
    return result.sum("quantile")


@dataclasses.dataclass
class SpatialEnsembleRPS(EnsembleMetric, ThresholdMetric):
  """Spatial map of ensemble RPS."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:

    result = self._map_over_thresholds(
        functools.partial(
            _compute_rps_part, ensemble_dim=self.ensemble_dim, skipna=skipna
        ),
        forecast,
        truth,
        region=region,
        skipna=skipna,
        spatial_agg=False,
    )
    return result.sum("quantile")


@dataclasses.dataclass
class RankHistogram(EnsembleMetric):
  """Histogram of truth's rank with respect to forecast ensemble members.

  Given a K member ensemble {Xᵢ}, and ground truth Y, the rank of Y is the count
  of ensemble members less than Y. This class expresses that rank with one-hot
  encoding, which facilitates averaging/summation (typically over time) to form
  rank histograms. The histograms will have K+1 bins, and are indexed by 'bin'.

  This class also allows aggregation of the bins into num_bins ≤ K + 1 provided
  num_bins evently divides K + 1. This reduces the size of output files, but is
  equivalent to averaging default results along the 'bin' dimension.

  If these one-hot encodings are averaged over N times, a well calibrated
  forecast should contain roughly equal values in the bins. The bin variance
  will be (num_bins - 1) / (N num_bins²). Since the expected value is
  1 / num_bins, the relative error is
    Sqrt(variance) / expected = Sqrt((num_bins - 1) / N).

  NaN values are treated as larger than any other. The skipna kwarg is ignored.
  """

  def __init__(
      self,
      ensemble_dim: str = REALIZATION,
      num_bins: t.Optional[int] = None,
      break_ties_randomly: bool = True,
      seed: t.Optional[int] = None,
  ):
    """Initializes a RankHistogram.

    Args:
      ensemble_dim: Dimension indexing ensemble member.
      num_bins: Number of bins in histogram. If None, the number of bins will be
        `ensemble_size + 1`. If provided, `num_bins` must evenly divide into
        `ensemble_size + 1`.
      break_ties_randomly: If True, break ties with the following behavior.
        If a subset of bins are identical (due to identical ensemble members),
        and truth falls within the corresponding bins, a random choice (within
        the tied bins) is made. If truth is exactly equal to some ensemble
        members, it is randomly assigned a bin within the tied bins.
      seed: Seed for RNG used to break ties.
    """
    super().__init__(ensemble_dim=ensemble_dim)
    self.num_bins = num_bins
    self._break_ties_randomly = break_ties_randomly
    self._seed = seed

  def _num_bins_actual(self, ensemble_size: int) -> int:
    default_n_bins = ensemble_size + 1
    if self.num_bins is None:
      return default_n_bins
    if default_n_bins % self.num_bins:
      raise ValueError(
          f"Cannot bin data with {ensemble_size=} into {self.num_bins} bins"
      )
    return self.num_bins

  def _bin_ranks(self, ensemble_size: int, ranks: xr.DataArray):
    """Transforms ensemble rank into bin membership."""
    default_n_bins = ensemble_size + 1
    num_bins = self._num_bins_actual(ensemble_size)
    reduction_factor = default_n_bins // num_bins
    if reduction_factor == 1:
      return ranks
    else:
      return ranks // reduction_factor

  def _perturb_by_min_ensemble_diff(self, da: xr.DataArray) -> xr.DataArray:
    """Perturbs da values by the minimum diff along ensemble_dim / 2."""
    if da.sizes[self.ensemble_dim] < 2:
      # The purpose of the perturbation is to break ties. No ties if only 1.
      return da
    idx = da.dims.index(self.ensemble_dim)
    diffs = np.diff(np.sort(da, axis=idx), axis=idx)

    # diff = 0 is a problem. We don't want to perturb by 0, since that does
    # nothing. We want to perturb by the smallest diff that is > 0.
    diffs_zero_replaced_by_inf = np.where(
        diffs == 0,
        np.inf,
        diffs,
    )
    min_diff = diffs_zero_replaced_by_inf.min(axis=idx, keepdims=True)
    perturbation_size = np.where(
        # If all diffs were zero, then the minimum will be Inf, and in this case
        # perturb by 1.
        min_diff < np.inf,
        min_diff / 2,
        1,
    )
    perturbation = np.random.default_rng(self._seed).uniform(
        size=da.shape, low=-perturbation_size / 2, high=perturbation_size / 2
    )
    return da + perturbation

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
      skipna: bool = False,
  ) -> xr.Dataset:
    """Computes one-hot encoding of rank on a chunk of forecast/truth."""

    if skipna and (
        any(truth[v].isnull().any() for v in truth)
        or any(forecast[v].isnull().any() for v in forecast)
    ):
      logging.warning(
          "NaN values detected in truth or forecast. skipna=True but it will be"
          " ignored."
      )
    # Create a fake ensemble member for truth. This is for concatenation.
    truth_realization = forecast[self.ensemble_dim].data.min() - 1
    truth = truth.assign_coords({self.ensemble_dim: truth_realization})
    # Forecast typically already has an ensemble coord...but sometimes it will
    # have an ensemble *dim* but not a coord, and this will mess up xr.concat.
    forecast = forecast.assign_coords(
        {self.ensemble_dim: forecast[self.ensemble_dim]}
    )
    combined = xr.concat([truth, forecast], dim=self.ensemble_dim)
    if self._break_ties_randomly:
      # It is not enough to just perturb forecast. Consider e.g. the case when
      # forecast = [0, 0, 0, 0, 0], every time, and so is truth. Then, if we
      # only perturb the forecast, truth will end up in the center bin more than
      # the others.
      combined = combined.map(
          self._perturb_by_min_ensemble_diff, keep_attrs=True
      )

    def array_rank_one_hot(da: xr.DataArray) -> xr.DataArray:
      ensemble_size = forecast.sizes[self.ensemble_dim]
      num_bins = self._num_bins_actual(ensemble_size)
      # order has the same dims as da, which is
      #   concat([truth, forecast], ensemble_dim)
      order = da.argsort(axis=da.dims.index(self.ensemble_dim))
      # Since we *prepended* truth to forecast, argmin will give the location of
      # truth in the results of argsort. Therefore, ranks is the positional
      # order of the truth realization.
      # ranks has the same dims as forecast, with ensemble_dim dropped.
      ranks = order.argmin(self.ensemble_dim)
      ranks = self._bin_ranks(ensemble_size, ranks)
      return ranks.expand_dims(bins=np.arange(num_bins), axis=-1).copy(
          # data will have shape ranks.shape + [num_bins].
          data=np.eye(num_bins)[ranks],
      )

    return combined.map(array_rank_one_hot, keep_attrs=False)


def central_reliability(hist: xr.Dataset) -> xr.Dataset:
  """Computes reliability diagram for central histogram probabilities.

  Roughtly speaking, a forecast X with median μ is considered to have
  "central reliability" with respect to truth Y, if for δ > 0,
     P[μ - δ < X < μ - δ] = P[μ - δ < Y < μ - δ].
  Since there is only one truth value, the right hand side cannot be computed
  directly. Instead, one often uses a rank histogram.

  For a rank histogram hist with N bins [0, ..., N-1], the probability that
  truth had value less extreme than the 2k + N % 2 central ensemble members is
     prob[k] = hist.sel(bins=central_slice[k]).sum('bins')
  where for k ∈ {0, ..., N // 2 - 1 + N % 2},
     central_slice[k] = slice(N // 2 - (k + 1) + N % 2, N // 2 + (k + 1))
  defines the length 2(k + 1) - N % 2 slice of central bins.

  The result is indexed by the desired probabilities, obtained for a perfectly
  calibrated forecast. These are, for even N,
     desired_prob[k] = 2(k + 1) / N,
  and for odd N,
     desired_prob[k] = (2k + 1) / N.

  Args:
    hist: Dataset with 'bins' dimension of length >= 3, indexing probabilities.
      For example, the returned values of `RankHistogram`, possibly after
      averaging over time or other non-bin dimensions.

  Returns:
    Dataset of central reliability values, indexed by 'desired_prob'.
  """

  n_bins = len(hist.bins)
  if n_bins < 3:
    raise ValueError(f"Too few bins. {n_bins=} but should be >= 3")

  # To efficiently compute the reliability, we use cumsum, rather than directly
  # following the docstring.
  left_hist = hist.sel(bins=slice(None, n_bins // 2 - 1))
  right_hist = hist.sel(bins=slice(n_bins // 2 + n_bins % 2, None))
  linear_bins = left_hist.bins.data
  probs = (
      (
          # Must reverse left_hist, so that we are doing a cumsum from the
          # inside out.
          left_hist.reindex(bins=left_hist.bins[::-1]).assign_coords(
              bins=linear_bins
          )
          # Must assign right_hist with bins [0, 1, ...]
          + right_hist.assign_coords(bins=linear_bins)
      )
      .cumsum("bins")
      .rename(bins="prob_index")
  )

  desired_prob_unnormalized = np.ones((len(probs.prob_index),))

  if n_bins % 2:
    # If odd bins, there is a midpoint probability that doesn't get lumped into
    # the cumsum.
    probs = probs.assign_coords(prob_index=linear_bins + 1)
    center_prob = hist.sel(bins=n_bins // 2, drop=True)
    probs = xr.concat(
        [center_prob.expand_dims(prob_index=[0]), center_prob + probs],
        dim="prob_index",
    )
    desired_prob_unnormalized = np.concatenate(
        # The midpoint probability is made from 1 bin, whereas the others are
        # from two.
        ([0.5], desired_prob_unnormalized)
    )
  else:
    # Ensure the dim is also a coord.
    probs = probs.assign_coords(prob_index=probs.prob_index)

  desired_prob_unnormalized = np.cumsum(desired_prob_unnormalized)
  probs = probs.assign_coords(
      desired_prob=(
          "prob_index",  # Corresponding dimension.
          desired_prob_unnormalized / desired_prob_unnormalized[-1],
      )
  )
  return probs.swap_dims({"prob_index": "desired_prob"})
