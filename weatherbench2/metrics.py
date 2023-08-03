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
import dataclasses
import functools
import typing as t

import numpy as np
from scipy import stats
from weatherbench2.regions import Region
import xarray as xr

REALIZATION = 'realization'


def _assert_increasing(x: np.ndarray):
  if not (np.diff(x) > 0).all():
    raise ValueError(f'array is not increasing: {x}')


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


@dataclasses.dataclass
class Metric:
  """Base class for metrics."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
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
  ) -> xr.Dataset:
    """Evaluate this metric on datasets with full temporal coverages."""
    if 'time' in forecast.dims:
      avg_dim = 'time'
    elif 'init_time' in forecast.dims:
      avg_dim = 'init_time'
    else:
      raise ValueError(
          f'Forecast has neither valid_time or init_time dimension {forecast}'
      )
    return self.compute_chunk(forecast, truth, region=region).mean(avg_dim)


def _spatial_average(
    dataset: xr.Dataset, region: t.Optional[Region] = None, skipna: bool = False
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
  return dataset.weighted(weights).mean(
      ['latitude', 'longitude'], skipna=skipna
  )


def _spatial_average_l2_norm(
    dataset: xr.Dataset, region: t.Optional[Region] = None, skipna: bool = False
) -> xr.Dataset:
  """Helper function to compute sqrt(spatial_average(ds**2))."""
  return np.sqrt(_spatial_average(dataset**2, region=region, skipna=skipna))


@dataclasses.dataclass
class WindVectorRMSE(Metric):
  """Compute wind vector RMSE. See WB2 paper for definition.

  Attributes:
    u_name: Name of U component.
    v_name: Name of v component.
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
  ) -> xr.Dataset:
    diff = forecast - truth
    result = np.sqrt(
        _spatial_average(
            diff[self.u_name] ** 2 + diff[self.v_name] ** 2,
            region=region,
        )
    )
    return result


@dataclasses.dataclass
class RMSE(Metric):
  """Root mean squared error.

  Attributes:
    wind_vector_rmse: Optionally provide list of WindVectorRMSE instances to
      compute.
  """

  wind_vector_rmse: t.Optional[list[WindVectorRMSE]] = None

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    results = _spatial_average_l2_norm(forecast - truth, region=region)
    if self.wind_vector_rmse is not None:
      for wv in self.wind_vector_rmse:
        results[wv.vector_name] = wv.compute_chunk(
            forecast, truth, region=region
        )
    return results


@dataclasses.dataclass
class MSE(Metric):
  """Mean squared error."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    return _spatial_average((forecast - truth) ** 2, region=region)


@dataclasses.dataclass
class SpatialMSE(Metric):
  """MSE without spatial averaging."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    return (forecast - truth) ** 2


@dataclasses.dataclass
class MAE(Metric):
  """Mean absolute error."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    return _spatial_average(abs(forecast - truth), region=region)


@dataclasses.dataclass
class SpatialMAE(Metric):
  """Mean absolute error without spatial averaging."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    return abs(forecast - truth)


@dataclasses.dataclass
class Bias(Metric):
  """Bias."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    return _spatial_average(forecast - truth, region=region)


@dataclasses.dataclass
class SpatialBias(Metric):
  """Bias without spatial averaging."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
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
  ) -> xr.Dataset:
    if 'init_time' in forecast.dims:
      time_dim = 'valid_time'
    else:
      time_dim = 'time'
    climatology_chunk = self.climatology[list(forecast.keys())]
    if hasattr(forecast, 'level'):
      climatology_chunk = climatology_chunk.sel(level=forecast.level)
    climatology_chunk = climatology_chunk.sel(
        dayofyear=forecast[time_dim].dt.dayofyear,
        hour=forecast[time_dim].dt.hour,
    ).compute()
    forecast_anom = forecast - climatology_chunk
    truth_anom = truth - climatology_chunk
    return _spatial_average(
        forecast_anom * truth_anom, region=region
    ) / np.sqrt(
        _spatial_average(forecast_anom**2, region=region)
        * _spatial_average(truth_anom**2, region=region)
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
  precip_name: str = 'total_precipitation_24hr'
  min_p1: float = 0.01
  max_p1: float = 0.85

  @functools.cached_property
  def p1(self) -> xr.DataArray:
    dry_fraction = self.climatology[f'{self.precip_name}_seeps_dry_fraction']
    return dry_fraction.mean(('hour', 'dayofyear')).compute()

  def _convert_precip_to_seeps_cat(self, ds):
    """Helper function for SEEPS computation. Converts values to categories."""
    wet_threshold = self.climatology[f'{self.precip_name}_seeps_threshold']
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
        dim=xr.DataArray(['dry', 'light', 'heavy'], dims=['seeps_cat']),
    )
    # Convert NaNs back to NaNs
    result = result.astype('int').where(da.notnull())
    return result

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    forecast_cat = self._convert_precip_to_seeps_cat(forecast)
    truth_cat = self._convert_precip_to_seeps_cat(truth)

    # Compute contingency table
    out = (
        forecast_cat.rename({'seeps_cat': 'forecast_cat'})
        * truth_cat.rename({'seeps_cat': 'truth_cat'})
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
    result = xr.dot(out, scoring_matrix, dims=('forecast_cat', 'truth_cat'))

    # Mask out p1 thresholds
    result = result.where(self.p1 < self.max_p1, np.NaN)
    result = result.where(self.p1 > self.min_p1, np.NaN)
    return xr.Dataset({f'{self.precip_name}': result})


@dataclasses.dataclass
class SEEPS(SpatialSEEPS):
  """Spatially averaged SEEPS."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    result = super().compute_chunk(forecast, truth, region)
    # Need skipna = True because of p1 mask
    return _spatial_average(result, region=region, skipna=True)


################################################################################
# Probabilistic metrics.
################################################################################


def _get_n_ensemble(
    ds: xr.Dataset,
    ensemble_dim: str,
    expect_n_ensemble_at_least: int = 1,
) -> int:
  """Returns the size of `ensemble_dim`, optionally asserting size at least."""
  if ensemble_dim not in ds.dims:
    raise ValueError(f'{ensemble_dim=} not found in {ds.dims=}')
  n_ensemble = ds.dims[ensemble_dim]
  if n_ensemble < expect_n_ensemble_at_least:
    raise ValueError(
        f'{n_ensemble=} is less than expected size of '
        f'{expect_n_ensemble_at_least}'
    )
  return n_ensemble


@dataclasses.dataclass
class EnsembleMetric(Metric):
  ensemble_dim: str = REALIZATION

  def _ensemble_slice(self, ds: xr.Dataset, slice_obj: slice) -> xr.Dataset:
    """Slice `ds` and assign coords to [0, ..., len(slice)]."""
    ds = ds.isel({self.ensemble_dim: slice_obj})
    return ds.assign_coords(
        {self.ensemble_dim: np.arange(ds.dims[self.ensemble_dim])}
    )


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
  ) -> xr.Dataset:
    """CRPS, averaged over space, for a time chunk of data."""
    return CRPSSkill(self.ensemble_dim).compute_chunk(
        forecast, truth, region=region
    ) - 0.5 * CRPSSpread(self.ensemble_dim).compute_chunk(
        forecast, truth, region=region
    )


@dataclasses.dataclass
class CRPSSpread(EnsembleMetric):
  """The spread measure associated with CRPS, E|X - X'|."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    """CRPSSpread, averaged over space, for a time chunk of data."""
    return _spatial_average(
        _pointwise_crps_spread(forecast, truth, self.ensemble_dim),
        region=region,
    )


@dataclasses.dataclass
class CRPSSkill(EnsembleMetric):
  """The skill measure associated with CRPS, E|X - Y|."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    """CRPSSkill, averaged over space, for a time chunk of data."""
    return _spatial_average(
        _pointwise_crps_skill(forecast, truth, self.ensemble_dim),
        region=region,
    )


def _pointwise_crps_spread(
    forecast: xr.Dataset, truth: xr.Dataset, ensemble_dim: str
) -> xr.Dataset:
  """CRPS spread at each point in truth, averaged over ensemble only."""
  del truth  # unused
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
              ensemble_dim, skipna=False
          )
      )
      / (n_ensemble - 1)
  )


def _pointwise_crps_skill(
    forecast: xr.Dataset, truth: xr.Dataset, ensemble_dim: str
) -> xr.Dataset:
  """CRPS skill at each point in truth, averaged over ensemble only."""
  _get_n_ensemble(forecast, ensemble_dim)  # Will raise if no ensembles.
  return abs(truth - forecast).mean(ensemble_dim, skipna=False)


def _rank_ds(ds: xr.Dataset, dim: str) -> xr.Dataset:
  """The ranking of `ds` along `dim`, with 1 being the smallest entry."""

  def _rank_da(da: xr.DataArray) -> np.ndarray:
    return stats.rankdata(da.values, method='ordinal', axis=da.dims.index(dim))

  return ds.copy(data={k: _rank_da(v) for k, v in ds.items()})


@dataclasses.dataclass
class EnsembleStddev(EnsembleMetric):
  """The standard deviation of an ensemble of forecasts.

  This forms the SPREAD component of the traditional spread-skill-ratio. See
  [Garg & Rasp & Thuerey, 2022].

  Given predictive ensemble Xₜ at times t = (1,..., T),
    EnsembleStddev := (1 / T) Σₜ ‖σ(Xₜ)‖
  Above σ(Xₜ) is element-wise standard deviation, and ‖⋅‖ is an area-weighted
  L2 norm.

  Estimation is done separately for each tendency, level, and lag time.
  Ensembles of size 1 lead SPREAD=0, otherwise the unbiased sample standard
  deviation is used.

  This estimator has a slight amount of bias, since ‖σ(Xₜ)‖ has bias
  proportional to 1 / (ensemble_size * grid_size).

  NaN values propagate through and result in NaN in the corresponding output
  position.

  We use the unbiased estimator of σ(Xₜ) (dividing by n_ensemble - 1). If
  n_ensemble = 1, we return zero for the stddev. This choice allows
  EnsembleStddev to behave in the spread-skill-ratio as expected.

  References:
  [Garg & Rasp & Thuerey, 2022], WeatherBench Probability: A benchmark dataset
    for probabilistic medium-range weather forecasting along with deep learning
    baseline models.
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    """EnsembleStddev, averaged over space, for a time chunk of data."""
    del truth  # unused
    n_ensemble = _get_n_ensemble(forecast, self.ensemble_dim)

    if n_ensemble == 1:
      return xr.zeros_like(
          # Compute the average, even though we return zeros_like. Why? Because,
          # this will preserve the scalar values of lat/lon coords correctly.
          _spatial_average(forecast, region=region).mean(self.ensemble_dim)
      )
    else:
      return _spatial_average_l2_norm(
          forecast.std(self.ensemble_dim, ddof=1), region=region
      )


@dataclasses.dataclass
class EnsembleMeanRMSE(EnsembleMetric):
  """RMSE between the ensemble mean and ground truth.

  This forms the SKILL component of the traditional spread-skill-ratio. See
  [Garg & Rasp & Thuerey, 2022].

  Given ground truth Yₜ, and predictive ensemble Xₜ, both at times
  t = (1,..., T),
    EnsembleMeanRMSE := (1 / T) Σₜ ‖Y - E(Xₜ)‖.
  Above, `E` is ensemble average, and ‖⋅‖ is an area-weighted L2 norm.

  Estimation is done separately for each tendency, level, and lag time.
  Ensembles of size 1 lead SPREAD=0, otherwise the unbiased sample standard
  deviation is used.

  Unfortunately, this estimator is biased since ‖Y - E(Xₜ)‖ has bias
  proportional to 1 / ensemble_size.

  NaN values propagate through and result in NaN in the corresponding output
  position.

  References:
  [Garg & Rasp & Thuerey, 2022], WeatherBench Probability: A benchmark dataset
    for probabilistic medium-range weather forecasting along with deep learning
    baseline models.
  """

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    """EnsembleMeanRMSE, averaged over space, for a time chunk of data."""
    _get_n_ensemble(forecast, self.ensemble_dim)  # Will raise if no ensembles.

    return _spatial_average_l2_norm(
        truth - forecast.mean(self.ensemble_dim), region=region
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
  ) -> xr.Dataset:
    """Energy score, averaged over space, for a time chunk of data."""
    return EnergyScoreSkill(self.ensemble_dim).compute_chunk(
        forecast, truth, region=region
    ) - 0.5 * EnergyScoreSpread(self.ensemble_dim).compute_chunk(
        forecast, truth, region=region
    )


@dataclasses.dataclass
class EnergyScoreSpread(EnsembleMetric):
  """The spread measure associated with EnergyScore, E‖X - X'‖."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    """Energy score spread, averaged over space, for a time chunk of data."""
    n_ensemble = _get_n_ensemble(forecast, self.ensemble_dim)

    if n_ensemble == 1:
      return xr.zeros_like(
          # Compute the average, even though we return zeros_like. Why? Because,
          # this will preserve the scalar values of lat/lon coords correctly.
          _spatial_average(forecast, region=region).mean(self.ensemble_dim)
      )
    else:
      return _spatial_average_l2_norm(
          self._ensemble_slice(forecast, slice(None, -1))
          - self._ensemble_slice(forecast, slice(1, None)),
      ).mean(self.ensemble_dim)


@dataclasses.dataclass
class EnergyScoreSkill(EnsembleMetric):
  """The skill measure associated with EnergyScore, E‖X - Y‖."""

  def compute_chunk(
      self,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      region: t.Optional[Region] = None,
  ) -> xr.Dataset:
    """Energy score skill, averaged over space, for a time chunk of data."""
    _get_n_ensemble(forecast, self.ensemble_dim)  # Will raise if no ensembles.
    return _spatial_average_l2_norm(forecast - truth).mean(self.ensemble_dim)
