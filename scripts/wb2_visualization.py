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
"""Run full WB2 visualization."""

import typing as t

from absl import app
from absl import flags
from weatherbench2.config import PanelConfig, VizConfig  # pylint: disable=g-multiple-import
from weatherbench2.visualization import visualize_timeseries, visualize_scorecard, long2short, units  # pylint: disable=g-multiple-import


_DEFAULT_MODELS = [
    'hres_vs_analysis',
    'ens_mean_vs_analysis',
    'clim_vs_era',
    'persistence_vs_era',
]

RESULTS_DIR = flags.DEFINE_string(
    'results_dir',
    None,
    help='path to results',
)

FIGURE_DIR = flags.DEFINE_string(
    'figure_dir',
    None,
    help='path to save figures',
)

MODELS = flags.DEFINE_list(
    'models',
    _DEFAULT_MODELS,
    help='Models to visualize',
)


def main(_: t.Sequence[str]) -> None:
  model_paths = {
      'hres_vs_analysis': (
          f'{RESULTS_DIR.value}/hres_vs_analysis_deterministic.nc'
      ),
      'hres_vs_era': f'{RESULTS_DIR.value}/hres_vs_era_deterministic.nc',
      'ens_mean_vs_analysis': (
          f'{RESULTS_DIR.value}/ens_vs_analysis_deterministic.nc'
      ),
      'ens_mean_vs_era': f'{RESULTS_DIR.value}/ens_vs_era_deterministic.nc',
      'climatology_vs_era': (
          f'{RESULTS_DIR.value}/climatology_vs_era_deterministic.nc'
      ),
      'persistence_vs_era': (
          f'{RESULTS_DIR.value}/persistence_vs_era_deterministic.nc'
      ),
      'neuralgcm_vs_era': (
          f'{RESULTS_DIR.value}/neuralgcm_vs_era_deterministic.nc'
      ),
      'graphcast_vs_era': (
          f'{RESULTS_DIR.value}/graphcast_vs_era_deterministic.nc'
      ),
      'graphcast_0618_vs_era': (
          f'{RESULTS_DIR.value}/graphcast_0618_vs_era_deterministic.nc'
      ),
  }

  regions = ['global', 'tropics', 'extra-tropics']

  viz_config = VizConfig(
      results={model: model_paths[model] for model in MODELS.value},
      colors={
          'hres_vs_analysis': 'blue',
          'hres_vs_era': 'blue',
          'ens_mean_vs_analysis': 'green',
          'ens_mean_vs_era': 'green',
          # 'hres_pp_vs_era': 'lightgreen',
          # 'hres_pp_vs_analysis': 'green',
          'neuralgcm_vs_era': 'red',
          'graphcast_vs_era': 'orange',
          'graphcast_0618_vs_era': 'purple',
          # 'neuralgcm_vs_analysis': 'lightred',
          'climatology_vs_era': '0.3',
          'persistence_vs_era': '0.7',
      },
      layout=(2, 2),
      figsize=(12, 8),
      labels={
          'hres_vs_era': 'IFS HRES vs ERA5',
          'hres_vs_analysis': 'IFS HRES vs AN',
          'ens_mean_vs_analysis': 'IFS ENS (mean) vs AN',
          'ens_mean_vs_era': 'IFS ENS (mean) vs ERA5',
          'hres_pp_vs_era': 'PP HRES vs ERA5',
          'hres_pp_vs_analysis': 'PP HRES vs AN',
          'neuralgcm_vs_era': 'NeuralGCM vs ERA5',
          'graphcast_vs_era': 'GraphCast vs ERA5',
          'graphcast_0618_vs_era': 'GraphCast 06/18 vs ERA5',
          'neuralgcm_vs_analysis': 'NeuralGCM vs AN',
          'climatology_vs_era': 'Climatology vs ERA5',
          'persistence_vs_era': 'Persistence vs ERA5',
      },
  )

  def plot_headline_scores(
      metric: str, relative: bool, regions: t.Sequence[str], kind: str
  ):
    if kind == 'pressure':
      variables = [
          ('geopotential', 500),
          ('temperature', 850),
          ('specific_humidity', 700),
          ('wind_vector' if metric == 'rmse' else 'u_component_of_wind', 850),
      ]
    elif kind == 'single':
      variables = [
          ('2m_temperature', None),
          ('mean_sea_level_pressure', None),
          ('10m_wind_speed', None),
          ('total_precipitation_6hr', None),
      ]
    for region in regions:
      plot_configs = []
      for variable, level in variables:
        if relative:
          ylabel = '% relative to IFS HRES'
        elif metric == 'acc':
          ylabel = 'ACC'
        else:
          ylabel = f'{metric.upper()} [{units[variable]}]'

        plot_configs.append(
            PanelConfig(
                metric=metric,
                variable=variable,
                level=level,
                region=region,
                relative='hres_vs_analysis' if relative else None,
                ylabel=ylabel,
                xlabel='Lead time [days]',
                title=(
                    f'{long2short[variable]}{level if level else ""} - {region}'
                ),
            )
        )

      visualize_timeseries(
          viz_config,
          plot_configs,
          save_path=(
              f'{FIGURE_DIR.value}/timeseries_{metric}_{kind}_{"rel" if relative else "abs"}_{region}.png'
          ),
      )

  ### Pressure
  # Plot 1: RMSE headline pressure levels, absolute
  plot_headline_scores('rmse', relative=False, regions=regions, kind='pressure')

  # Plot 2: RMSE headline pressure levels, relative
  plot_headline_scores('rmse', relative=True, regions=regions, kind='pressure')

  # Plot 3: ACC headline pressure levels, absolute
  plot_headline_scores('acc', relative=False, regions=regions, kind='pressure')

  # Plot 4: ACC headline pressure levels, relative
  plot_headline_scores('acc', relative=True, regions=regions, kind='pressure')

  ### Single
  # Plot 1: RMSE headline single levels, absolute
  plot_headline_scores('rmse', relative=False, regions=regions, kind='single')

  # Plot 2: RMSE headline single levels, relative
  plot_headline_scores('rmse', relative=True, regions=regions, kind='single')

  # Plot 3: ACC headline single levels, absolute
  plot_headline_scores('acc', relative=False, regions=regions, kind='single')

  # Plot 4: ACC headline single levels, relative
  plot_headline_scores('acc', relative=True, regions=regions, kind='single')

  ### Scorecards
  # Plot 1: RMSE
  metric = 'rmse'
  baseline = 'hres_vs_analysis'
  forecasts = ['ens_mean_vs_analysis', 'neuralgcm_vs_era', 'graphcast_vs_era']

  for forecast in forecasts:
    for region in regions:
      visualize_scorecard(
          viz_config,
          baseline=baseline,
          forecast=forecast,
          metric=metric,
          region=region,
          save_path=f'{FIGURE_DIR.value}/scorecard_{metric}_{region}_{forecast}_vs_{baseline}.png',
          cmap_scale=30,
      )

  metric = 'rmse'
  baseline = 'ens_mean_vs_analysis'
  forecasts = ['neuralgcm_vs_era', 'graphcast_vs_era']

  for forecast in forecasts:
    for region in regions:
      visualize_scorecard(
          viz_config,
          baseline=baseline,
          forecast=forecast,
          metric=metric,
          region=region,
          save_path=f'{FIGURE_DIR.value}/scorecard_{metric}_{region}_{forecast}_vs_{baseline}.png',
          cmap_scale=30,
      )

  metric = 'rmse'
  baseline = 'graphcast_vs_era'
  forecasts = ['neuralgcm_vs_era']

  for forecast in forecasts:
    for region in regions:
      visualize_scorecard(
          viz_config,
          baseline=baseline,
          forecast=forecast,
          metric=metric,
          region=region,
          save_path=f'{FIGURE_DIR.value}/scorecard_{metric}_{region}_{forecast}_vs_{baseline}.png',
          cmap_scale=30,
      )


if __name__ == '__main__':
  app.run(main)
