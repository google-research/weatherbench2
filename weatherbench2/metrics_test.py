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
from absl.testing import parameterized
import numpy as np
from weatherbench2 import metrics
from weatherbench2 import regions
from weatherbench2 import schema
from weatherbench2 import test_utils
from weatherbench2 import utils
import xarray as xr


def get_random_truth_and_forecast(
    variables=('geopotential',), ensemble_size=None, seed=802701, **data_kwargs
):
  """Makes the tuple (truth, forecast) from kwargs."""
  data_kwargs_to_use = dict(
      variables_3d=variables,
      variables_2d=[],
      time_start='2019-12-01',
      time_stop='2019-12-02',
      spatial_resolution_in_degrees=30,
      time_resolution='3 hours',
  )
  data_kwargs_to_use.update(data_kwargs)
  truth = utils.random_like(
      schema.mock_truth_data(**data_kwargs_to_use), seed=seed
  )
  forecast = utils.random_like(
      schema.mock_forecast_data(
          ensemble_size=ensemble_size, **data_kwargs_to_use
      ),
      seed=seed + 1,
  )
  return truth, forecast


class MetricsTest(parameterized.TestCase):

  def test_get_lat_weights(self):
    ds = xr.Dataset(coords={'latitude': np.array([-75, -45, -15, 15, 45, 75])})
    weights = metrics.get_lat_weights(ds)
    self.assertAlmostEqual(float(weights.mean(skipna=False)), 1.0)
    # from Wolfram alpha:
    # integral of cos(x) from 0*pi/6 to 1*pi/6 -> 0.5
    # integral of cos(x) from 1*pi/6 to 2*pi/6 -> (sqrt(3) - 1) / 2
    # integral of cos(x) from 2*pi/6 to 3*pi/6 -> 1 - sqrt(3) / 2
    expected_data = 3 * np.array([
        1 - np.sqrt(3) / 2,
        (np.sqrt(3) - 1) / 2,
        1 / 2,
        1 / 2,
        (np.sqrt(3) - 1) / 2,
        1 - np.sqrt(3) / 2,
    ])
    expected = xr.DataArray(expected_data, coords=ds.coords, dims=['latitude'])
    xr.testing.assert_allclose(expected, weights)

  def test_wind_vector_rmse(self):
    wv = metrics.WindVectorRMSESqrtBeforeTimeAvg(
        u_name='u_component_of_wind',
        v_name='v_component_of_wind',
        vector_name='wind_vector',
    )
    forecast = schema.mock_forecast_data(
        variables_3d=['u_component_of_wind', 'v_component_of_wind'],
        variables_2d=[],
        time_start='2022-01-01',
        time_stop='2022-01-02',
        lead_stop='0 day',
    )
    truth = schema.mock_truth_data(
        variables_3d=['u_component_of_wind', 'v_component_of_wind'],
        variables_2d=[],
        time_start='2022-01-01',
        time_stop='2022-01-02',
    )

    forecast_modifier = xr.Dataset({
        'u_component_of_wind': xr.DataArray(
            [0, 3, np.NaN], coords={'level': forecast.level}
        ),
        'v_component_of_wind': xr.DataArray(
            [0, -4, 1], coords={'level': forecast.level}
        ),
    })
    truth_modifier = xr.Dataset({
        'u_component_of_wind': xr.DataArray(
            [0, -3, np.NaN], coords={'level': forecast.level}
        ),
        'v_component_of_wind': xr.DataArray(
            [0, 4, 1], coords={'level': forecast.level}
        ),
    })

    forecast = forecast + forecast_modifier
    truth = truth + truth_modifier

    result = wv.compute(forecast, truth).values.squeeze()

    expected = np.array([0, 10, np.NaN])
    np.testing.assert_allclose(result, expected)

  @parameterized.named_parameters(
      dict(testcase_name='inf', invalid_value=np.inf),
      dict(testcase_name='nan', invalid_value=np.nan),
  )
  def test_rmse_over_invalid_region(self, invalid_value):
    rmse = metrics.RMSESqrtBeforeTimeAvg()
    truth = xr.Dataset(
        {'wind_speed': ('latitude', [0.0, invalid_value, 0.0])},
        coords={'latitude': [-45, 0, 45]},
    ).expand_dims(['time', 'longitude'])
    forecast = truth + 1

    actual = rmse.compute(forecast, truth)
    expected = xr.Dataset({'wind_speed': np.nan})
    xr.testing.assert_allclose(actual, expected)

    region = regions.ExtraTropicalRegion()
    actual = rmse.compute(forecast, truth, region=region)
    expected = xr.Dataset({'wind_speed': 1.0})
    xr.testing.assert_allclose(actual, expected)

  def test_daily_avg_acc(self):
    kwargs = dict(time_resolution='1 day')
    truth, forecast = get_random_truth_and_forecast(**kwargs)
    climatology = truth.isel(time=0, drop=True).expand_dims(
        dayofyear=366,
    )
    climatology_mean = (
        truth.isel(time=0, drop=True)
        .expand_dims(
            dayofyear=366,
        )
        .rename({'geopotential': 'geopotential_mean'})
    )
    # Check compatibility with climatological data from wb2 scripts
    acc1 = metrics.ACC(climatology).compute_chunk(forecast, truth)
    acc2 = metrics.ACC(climatology_mean).compute_chunk(forecast, truth)
    xr.testing.assert_allclose(acc1, acc2)


class CRPSTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='EnsembleSize2', ensemble_size=2),
      dict(testcase_name='EnsembleSize3', ensemble_size=3),
      dict(testcase_name='EnsembleSize5', ensemble_size=5),
  )
  def test_vs_brute_force(self, ensemble_size):
    truth, forecast = get_random_truth_and_forecast(ensemble_size=ensemble_size)
    expected_crps = _crps_brute_force(forecast, truth)

    xr.testing.assert_allclose(
        expected_crps['score'],
        metrics.CRPS().compute_chunk(forecast, truth),
    )

    self.assertEqual(
        ensemble_size, metrics.CRPS().compute(forecast, truth).ensemble_size
    )

  def test_ensemble_size_1_gives_mae(self):
    truth, forecast = get_random_truth_and_forecast(ensemble_size=1)

    expected_skill = metrics._spatial_average(
        abs(truth - forecast.isel({metrics.REALIZATION: 0}))
    )

    xr.testing.assert_allclose(
        metrics.CRPSSkill().compute_chunk(forecast, truth),
        expected_skill,
    )
    xr.testing.assert_allclose(
        metrics.CRPSSpread().compute_chunk(forecast, truth),
        xr.zeros_like(metrics.CRPSSpread().compute_chunk(forecast, truth)),
    )
    xr.testing.assert_allclose(
        metrics.CRPS().compute_chunk(forecast, truth),
        expected_skill,  # Spread = 0
    )

  def test_nan_forecasts_result_in_nan_crps(self):
    truth, forecast = get_random_truth_and_forecast(
        variables=['geopotential', 'temperature'], ensemble_size=7
    )

    # Make realization 0 have a NaN in the very first place.
    new_values = forecast.geopotential.values.copy()
    np.put(new_values, [0] * new_values.ndim, np.nan)
    forecast = forecast.copy(
        data={'geopotential': new_values, 'temperature': forecast.temperature}
    )

    crps = metrics.CRPS().compute_chunk(forecast, truth)

    # The only NaN geopotential is in the very first place.
    score_values = crps.geopotential.values.copy()
    self.assertTrue(np.isnan(score_values[0, 0, 0]))
    score_values[0, 0, 0] = 0  # Replace the NaN
    self.assertTrue(np.all(np.isfinite(score_values)))

    # Temperature is not NaN at all (NaNs didn't propagate).
    self.assertTrue(np.all(np.isfinite(crps.temperature.values)))

    xr.testing.assert_allclose(
        crps, _crps_brute_force(forecast, truth)['score']
    )

  def test_repeated_forecasts_are_okay(self):
    truth, forecast = get_random_truth_and_forecast(ensemble_size=7)

    # Make realizations 0 and 1 be the same.
    assert forecast.geopotential.dims.index(metrics.REALIZATION) == 0
    new_values = forecast.geopotential.values.copy()
    new_values[0] = new_values[1]
    forecast = forecast.copy(data={'geopotential': new_values})

    crps = metrics.CRPS().compute_chunk(forecast, truth)
    xr.testing.assert_allclose(
        crps, _crps_brute_force(forecast, truth)['score']
    )


class GaussianCRPSTest(parameterized.TestCase):

  def test_gaussian_crps(self):
    forecast = schema.mock_forecast_data(
        variables_3d=[],
        variables_2d=['2m_temperature', '2m_temperature_std'],
        time_start='2022-01-01',
        time_stop='2022-01-02',
        lead_stop='1 day',
    )
    truth = schema.mock_truth_data(
        variables_3d=[],
        variables_2d=['2m_temperature'],
        time_start='2022-01-01',
        time_stop='2022-01-20',
    )
    forecast = forecast + 1.0
    truth = truth + 1.02
    result = metrics.GaussianCRPS().compute(forecast, truth)
    expected = np.array([0.23385455, 0.23385455])
    np.testing.assert_allclose(result['2m_temperature'].values, expected)

  def test_convergence_gaussian_crps(self):
    """Tests that the ensemble CRPS converges to analytical formula."""
    forecast = schema.mock_forecast_data(
        variables_3d=[],
        variables_2d=['2m_temperature', '2m_temperature_std'],
        time_start='2022-01-01',
        time_stop='2022-01-02',
        lead_stop='1 day',
    )
    ens_forecast = schema.mock_forecast_data(
        variables_3d=[],
        variables_2d=['2m_temperature'],
        time_start='2022-01-01',
        time_stop='2022-01-02',
        lead_stop='1 day',
        ensemble_size=5000,
    )
    truth = schema.mock_truth_data(
        variables_3d=[],
        variables_2d=['2m_temperature'],
        time_start='2022-01-01',
        time_stop='2022-01-20',
    )
    forecast['2m_temperature'] = forecast['2m_temperature'] + 0.1
    forecast['2m_temperature_std'] = forecast['2m_temperature_std'] + 1.0
    ens_forecast['2m_temperature'] = (
        ens_forecast['2m_temperature']
        + np.random.randn(*ens_forecast['2m_temperature'].shape)
        + 0.1
    )

    result = metrics.GaussianCRPS().compute(forecast, truth)
    result2 = metrics.CRPS().compute(ens_forecast, truth)
    np.testing.assert_allclose(
        result['2m_temperature'].values,
        result2['2m_temperature'].values,
        rtol=2e-2,
    )


class GaussianVarianceTest(parameterized.TestCase):

  def test_gaussian_variance(self):
    forecast = schema.mock_forecast_data(
        variables_3d=[],
        variables_2d=['2m_temperature', '2m_temperature_std'],
        time_start='2022-01-01',
        time_stop='2022-01-02',
        lead_stop='1 day',
    )
    truth = schema.mock_truth_data(
        variables_3d=[],
        variables_2d=['2m_temperature'],
        time_start='2022-01-01',
        time_stop='2022-01-20',
    )
    forecast['2m_temperature_std'] = forecast['2m_temperature_std'] + 1.0
    result = metrics.GaussianVariance().compute(forecast, truth)
    expected = np.array([1.0, 1.0])
    np.testing.assert_allclose(result['2m_temperature'].values, expected)


class RankHistogramTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='EnsembleSize1', ensemble_size=1),
      dict(testcase_name='EnsembleSize10', ensemble_size=10),
      dict(testcase_name='EnsembleSize2', ensemble_size=2),
      dict(testcase_name='EnsembleSize9_NumBins5', ensemble_size=9, num_bins=5),
  )
  def test_well_and_mis_calibrated(self, ensemble_size, num_bins=None):
    num_bins = ensemble_size + 1 if num_bins is None else num_bins
    # Forecast and truth come from same distribution
    truth, forecast = get_random_truth_and_forecast(
        ensemble_size=ensemble_size,
        # Get enough days so our sample size is large.
        time_start='2019-12-01',
        time_stop='2019-12-10',
        levels=(0, 1, 2, 3, 4),
    )

    # level=0 is well calibrated
    # level=1,2 are under/over dispersed
    forecast.loc[{'level': 1}] *= 0.1
    forecast.loc[{'level': 2}] *= 10
    # level=3,4 are skew left/right
    forecast.loc[{'level': 3}] -= 1
    forecast.loc[{'level': 4}] += 1

    one_hot_ranks = metrics.RankHistogram(
        ensemble_dim='realization', num_bins=num_bins
    ).compute_chunk(forecast, truth)

    expected_sizes = {
        d: s for d, s in forecast.sizes.items() if d != 'realization'
    } | {'bins': num_bins}
    self.assertEqual(expected_sizes, one_hot_ranks.sizes)

    # Average over dimensions where forecast is iid. I.e., not 'bins' or 'level'
    averaging_dims = ['prediction_timedelta', 'time', 'latitude', 'longitude']
    sample_size = np.prod([one_hot_ranks.sizes[d] for d in averaging_dims])
    rtol = 5 * np.sqrt((num_bins - 1) / sample_size)  # 5 standard errors.

    hist = one_hot_ranks.mean(averaging_dims).geopotential

    # Recall level=0 is well calibrated.
    np.testing.assert_allclose(1 / num_bins, hist.sel(level=0), rtol=rtol)

    # num_bins=2 does not have enough resolution to detect under/over dispersed.
    if num_bins > 2:
      convex = hist.sel(level=1).data  # Under dispersed ==> convex.
      test_utils.assert_strictly_decreasing(convex[: len(convex) // 2 + 1])
      test_utils.assert_strictly_increasing(convex[len(convex) // 2 :])
      concave = hist.sel(level=2).data  # Over dispersed ==> concave.
      test_utils.assert_strictly_increasing(concave[: len(concave) // 2 + 1])
      test_utils.assert_strictly_decreasing(concave[len(concave) // 2 :])

    # level=3,4 are skew left/right
    test_utils.assert_strictly_increasing(hist.sel(level=3))
    test_utils.assert_strictly_decreasing(hist.sel(level=4))


class CentralReliabilityTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='NBins1', n_bins=1),
      dict(testcase_name='NBins2', n_bins=2),
  )
  def test_n_bins_too_small_raises(self, n_bins):
    hist = xr.Dataset(
        {'temperature': ('bins', np.ones((n_bins,)) / n_bins)},
        coords={'bins': np.arange(n_bins)},
    )
    with self.assertRaisesRegex(ValueError, 'Too few bins'):
      metrics.central_reliability(hist)

  @parameterized.named_parameters(
      dict(testcase_name='NBins3', n_bins=3),
      dict(testcase_name='NBins4', n_bins=4),
      dict(testcase_name='NBins10', n_bins=10),
      dict(testcase_name='NBins11', n_bins=11),
  )
  def test_perfectly_calibrated_histogram(self, n_bins):
    hist = xr.Dataset(
        {'temperature': ('bins', np.ones((n_bins,)) / n_bins)},
        coords={'bins': np.arange(n_bins)},
    )
    reliability = metrics.central_reliability(hist)
    self.assertLen(reliability.desired_prob, n_bins // 2 + n_bins % 2)

    expected_prob_unnormalized = np.ones((n_bins // 2,))
    if n_bins % 2:
      expected_prob_unnormalized = np.concatenate(
          ([0.5], expected_prob_unnormalized)
      )
    expected_prob = np.cumsum(expected_prob_unnormalized) / np.sum(
        expected_prob_unnormalized
    )

    # Since perfectly calibrated, the expected and desired probs are equal.
    expected_ds = xr.Dataset(
        {'temperature': ('desired_prob', expected_prob)},
        coords={
            # desired_prob is the dimension.
            # prob_index is a range, whose dimension is desired_prob
            'desired_prob': expected_prob,
            'prob_index': ('desired_prob', np.arange(len(expected_prob))),
        },
    )
    xr.testing.assert_allclose(expected_ds, reliability)

  def test_a_particular_length_3_histogram(self):
    hist = xr.Dataset(
        {'temperature': ('bins', [0.2, 0.1, 0.7])},
        coords={'bins': np.arange(3)},
    )
    reliability = metrics.central_reliability(hist)

    expected_prob = [0.1, 1.0]
    desired_prob = [1 / 3, 1.0]
    expected_ds = xr.Dataset(
        {'temperature': ('desired_prob', expected_prob)},
        coords={
            # desired_prob is the dimension.
            # prob_index is a range, whose dimension is desired_prob
            'desired_prob': desired_prob,
            'prob_index': ('desired_prob', np.arange(len(expected_prob))),
        },
    )
    xr.testing.assert_allclose(expected_ds, reliability)

  def test_a_particular_length_5_histogram(self):
    hist = xr.Dataset(
        {'temperature': ('bins', [0.2, 0.0, 0.1, 0.1, 0.6])},
        coords={'bins': np.arange(5)},
    )
    reliability = metrics.central_reliability(hist)

    expected_prob = [0.1, 0.2, 1.0]
    desired_prob = [1 / 5, 2 / 5 + 1 / 5, 1]
    expected_ds = xr.Dataset(
        {'temperature': ('desired_prob', expected_prob)},
        coords={
            # desired_prob is the dimension.
            # prob_index is a range, whose dimension is desired_prob
            'desired_prob': desired_prob,
            'prob_index': ('desired_prob', np.arange(len(expected_prob))),
        },
    )
    xr.testing.assert_allclose(expected_ds, reliability)

  def test_a_particular_length_4_histogram(self):
    hist = xr.Dataset(
        {'temperature': ('bins', [0.1, 0.1, 0.5, 0.3])},
        coords={'bins': np.arange(4)},
    )
    reliability = metrics.central_reliability(hist)

    expected_prob = [0.6, 1.0]
    desired_prob = [1 / 2, 1.0]
    expected_ds = xr.Dataset(
        {'temperature': ('desired_prob', expected_prob)},
        coords={
            # desired_prob is the dimension.
            # prob_index is a range, whose dimension is desired_prob
            'desired_prob': desired_prob,
            'prob_index': ('desired_prob', np.arange(len(expected_prob))),
        },
    )
    xr.testing.assert_allclose(expected_ds, reliability)

  def test_a_particular_length_6_histogram(self):
    hist = xr.Dataset(
        {'temperature': ('bins', [0.1, 0.1, 0.3, 0.2, 0.0, 0.3])},
        coords={'bins': np.arange(6)},
    )
    reliability = metrics.central_reliability(hist)

    expected_prob = [0.5, 0.6, 1.0]
    desired_prob = [1 / 3, 2 / 3, 1]
    expected_ds = xr.Dataset(
        {'temperature': ('desired_prob', expected_prob)},
        coords={
            # desired_prob is the dimension.
            # prob_index is a range, whose dimension is desired_prob
            'desired_prob': desired_prob,
            'prob_index': ('desired_prob', np.arange(len(expected_prob))),
        },
    )
    xr.testing.assert_allclose(expected_ds, reliability)


class EnsembleMeanRMSEAndStddevTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='EnsembleSize1', ensemble_size=1),
      dict(testcase_name='EnsembleSize2', ensemble_size=2),
      dict(testcase_name='EnsembleSize3', ensemble_size=3),
      dict(testcase_name='EnsembleSize10', ensemble_size=100),
  )
  def test_on_random_dataset(self, ensemble_size):
    truth, forecast = get_random_truth_and_forecast(ensemble_size=ensemble_size)

    rmse = metrics.EnsembleMeanRMSESqrtBeforeTimeAvg().compute_chunk(
        forecast, truth
    )
    ensemble_stddev = metrics.EnsembleStddevSqrtBeforeTimeAvg().compute_chunk(
        forecast, truth
    )

    for dataset in [rmse, ensemble_stddev]:
      self.assertEqual(
          dict(dataset.dims),
          {
              k: v
              for k, v in forecast.dims.items()
              if k not in ['realization', 'latitude', 'longitude']
          },
      )
      self.assertCountEqual(['geopotential'], dataset.data_vars)

    if ensemble_size == 1:
      xr.testing.assert_equal(
          xr.zeros_like(ensemble_stddev.geopotential),
          ensemble_stddev.geopotential,
      )
      return

    # Subsequent asserts assume ensemble_size > 1.

    # Since truth and forecast both came from the same random distribution,
    # we expect spread and skill to be equal.
    n_independent_samples = np.prod(list(rmse.dims.values()))

    # At each time point, the estimator is biased ~ 1 / ensemble_size.
    atol = 4 * (1 / np.sqrt(n_independent_samples) + 1 / ensemble_size)

    xr.testing.assert_allclose(rmse.mean(), ensemble_stddev.mean(), atol=atol)

  def test_effect_of_large_bias_on_rmse(self):
    truth, forecast = get_random_truth_and_forecast(ensemble_size=10)
    truth += 1000

    mean_rmse = (
        metrics.EnsembleMeanRMSESqrtBeforeTimeAvg()
        .compute_chunk(forecast, truth)
        .mean()
    )

    # Dominated by bias of 1000
    np.testing.assert_allclose(1000, mean_rmse.geopotential.values, rtol=1e-3)

  def test_perfect_prediction_zero_rmse(self):
    truth, unused_forecast = get_random_truth_and_forecast(ensemble_size=10)
    forecast = truth.expand_dims({metrics.REALIZATION: 1})
    mean_rmse = (
        metrics.EnsembleMeanRMSESqrtBeforeTimeAvg()
        .compute_chunk(forecast, truth)
        .mean()
    )

    xr.testing.assert_allclose(xr.zeros_like(mean_rmse), mean_rmse)


def _crps_brute_force(forecast: xr.Dataset, truth: xr.Dataset) -> xr.Dataset:
  """The eFAIR version of CRPS from Zamo & Naveau over a chunk of data."""

  # This version is simple enough that we can use it as a reference.
  def _l1_norm(x):
    return metrics._spatial_average(abs(x))

  n_ensemble = forecast.dims[metrics.REALIZATION]
  skill = _l1_norm(truth - forecast).mean(metrics.REALIZATION, skipna=False)
  if n_ensemble == 1:
    spread = xr.zeros_like(skill)
  else:
    spread = _l1_norm(
        forecast - forecast.rename({metrics.REALIZATION: 'dummy'})
    ).mean(dim=(metrics.REALIZATION, 'dummy'), skipna=False) * (
        n_ensemble / (n_ensemble - 1)
    )

  return {
      'score': skill - 0.5 * spread,  # CRPS
      'spread': spread,
      'skill': skill,
  }


class EnergyScoreTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='EnsembleSize1', ensemble_size=1),
      dict(testcase_name='EnsembleSize2', ensemble_size=2),
      dict(testcase_name='EnsembleSize3', ensemble_size=3),
  )
  def test_on_random_dataset(self, ensemble_size):
    truth, forecast = get_random_truth_and_forecast(ensemble_size=ensemble_size)

    score = metrics.EnergyScore().compute_chunk(forecast, truth)
    spread = metrics.EnergyScoreSpread().compute_chunk(forecast, truth)
    skill = metrics.EnergyScoreSkill().compute_chunk(forecast, truth)

    for dataset in [score, spread, skill]:
      self.assertEqual(
          dict(dataset.dims),
          {
              k: v
              for k, v in forecast.dims.items()
              if k not in ['realization', 'latitude', 'longitude']
          },
      )
      self.assertCountEqual(['geopotential'], dataset.data_vars)

    if ensemble_size == 1:
      xr.testing.assert_equal(
          xr.zeros_like(spread.geopotential), spread.geopotential
      )
      xr.testing.assert_allclose(score, skill)
      return

    # Subsequent asserts assume ensemble_size > 1.

    # Since truth and forecast both came from the same random distribution,
    # we expect spread and skill to be equal.
    n_independent_samples = np.prod(list(score.dims.values()))
    xr.testing.assert_allclose(
        spread.mean(),
        skill.mean(),
        atol=4  # 4 standard errors.
        * score.geopotential.std().values
        / np.sqrt(n_independent_samples),
    )

    # And of course the final energy score should be computed correctly.
    xr.testing.assert_allclose(score, skill - 0.5 * spread)

  def test_effect_of_bias_on_skill(self):
    truth, forecast = get_random_truth_and_forecast(ensemble_size=10)
    truth += 1000

    score = metrics.EnergyScore().compute_chunk(forecast, truth).mean()
    spread = metrics.EnergyScoreSpread().compute_chunk(forecast, truth).mean()

    # Dominated by bias of 1000
    np.testing.assert_allclose(1000, score.geopotential.values, rtol=1e-3)

    # Spread is basically sqrt(2), but the area weighting makes things different
    np.testing.assert_allclose(
        spread.geopotential.values, np.sqrt(2), rtol=0.05
    )


class SEEPSTest(absltest.TestCase):

  def testExpectedValues(self):
    forecast = schema.mock_forecast_data(
        variables_3d=[],
        variables_2d=['total_precipitation_24hr'],
        time_start='2022-01-01',
        time_stop='2022-01-11',
        lead_stop='0 day',
    )
    forecast = forecast.rename({'time': 'init_time'})
    forecast.coords['valid_time'] = (
        forecast.init_time + forecast.prediction_timedelta
    )
    truth = schema.mock_truth_data(
        variables_3d=[],
        variables_2d=['total_precipitation_24hr'],
        time_start='2022-01-01',
        time_stop='2022-01-11',
    )
    truth_like_forecast = truth.sel(time=forecast.valid_time)
    climatology = truth.isel(time=0, drop=True).expand_dims(
        dayofyear=366, hour=4
    )
    climatology['total_precipitation_24hr_seeps_dry_fraction'] = (
        climatology['total_precipitation_24hr'] + 0.4
    )
    climatology['total_precipitation_24hr_seeps_threshold'] = (
        climatology['total_precipitation_24hr'] + 1.0
    )

    seeps = metrics.SEEPS(climatology=climatology)

    # Test that perfect forecast results in SEEPS = 0
    result1 = seeps.compute(forecast, truth_like_forecast)
    np.testing.assert_allclose(
        result1['total_precipitation_24hr'].values, 0, atol=1e-4
    )

    # Test that obs_cat = dry and fc_cat = light = 1/p1 = 0.5 * 1 / 0.4 = 1.25
    # This means the scoring matrix is correctly oriented
    forecast = forecast + 0.5
    result2 = seeps.compute(forecast, truth_like_forecast)
    np.testing.assert_allclose(
        result2['total_precipitation_24hr'].values, 1.25, atol=1e-4
    )


if __name__ == '__main__':
  absltest.main()
