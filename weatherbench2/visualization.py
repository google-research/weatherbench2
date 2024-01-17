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
"""Code to visualize saved results."""

import typing as t

import fsspec
import matplotlib
from matplotlib import patches
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from weatherbench2 import config
from weatherbench2.utils import open_nc
import xarray as xr

long2short = {
    'geopotential': 'Z',
    'temperature': 'T',
    'specific_humidity': 'Q',
    'u_component_of_wind': 'U',
    'v_component_of_wind': 'V',
    '10m_u_component_of_wind': 'U10',
    '10m_v_component_of_wind': 'V10',
    'mean_sea_level_pressure': 'MSLP',
    '2m_temperature': 'T2M',
    'total_precipitation_6hr': 'TP6h',
    'total_precipitation_24hr': 'TP24h',
    'wind_speed': 'WS',
    '10m_wind_speed': 'WS10',
    'wind_vector': 'WV',
    '10m_wind_vector': 'WV10',
}

units = {
    'geopotential': 'm$^2$/s$^{2}$',
    'temperature': 'K',
    'specific_humidity': 'g/kg',
    'u_component_of_wind': 'm/s',
    'v_component_of_wind': 'm/s',
    '10m_u_component_of_wind': 'm/s',
    '10m_v_component_of_wind': 'm/s',
    '2m_temperature': 'K',
    'mean_sea_level_pressure': 'Pa',
    'total_precipitation_6hr': 'mm',
    'total_precipitation_24hr': 'mm',
    'wind_speed': 'm/s',
    '10m_wind_speed': 'm/s',
    'wind_vector': 'm/s',
    '10m_wind_vector': 'm/s',
}


def set_wb2_style() -> None:
  """Changes MPL defaults to WB2 style."""
  plt.rcParams['axes.grid'] = True
  plt.rcParams['lines.linewidth'] = 2
  plt.rcParams['figure.facecolor'] = 'None'
  plt.rcParams['axes.facecolor'] = '0.95'
  plt.rcParams['grid.color'] = 'white'
  plt.rcParams['axes.spines.right'] = False
  plt.rcParams['axes.spines.top'] = False


def load_results(results_dict: t.Dict[str, str]) -> t.Dict[str, xr.Dataset]:
  """Load results."""
  results = {}
  for name, path_or_ds in results_dict.items():
    if isinstance(path_or_ds, xr.Dataset):
      results[name] = path_or_ds
    else:
      if path_or_ds.endswith('.zarr'):
        r = xr.open_zarr(path_or_ds)
      else:
        r = open_nc(path_or_ds)
      # Add perfect scores at t=0
      if r.lead_time[0] > np.timedelta64(0):
        lt0 = r.isel(lead_time=0).assign_coords(
            lead_time=[np.timedelta64(0, 'h')]
        )
        lt0 = lt0.where(lt0.metric != 'acc', 1)
        lt0 = lt0.where(lt0.metric != 'rmse', 0)
        lt0 = lt0.where(lt0.metric != 'mse', 0)
        lt0 = lt0.where(lt0.metric != 'bias', 0)
        r = xr.concat([lt0, r], 'lead_time')
      results[name] = r
  return results


def datetime_to_xticks(
    lead_time: xr.DataArray, ax: matplotlib.axes.Axes, xlim: t.Sequence[float]
) -> None:
  if xlim is not None:
    mx = np.max([np.max(xlim), lead_time.values.max()])
  else:
    mx = lead_time.values.max()
  ns = np.arange(lead_time.values.min(), mx + 1, np.timedelta64(1, 'D'))
  days = ns.astype('timedelta64[D]')
  ax.set_xticks(ns.astype(int))
  ax.set_xticklabels(days.astype(int))
  ax.set_xlim(lead_time.values.min(), lead_time.values.max())


def compute_relative_metrics(
    results: t.Dict[str, xr.Dataset], reference: str, metric: str
) -> t.Dict[str, xr.Dataset]:
  """Compute relative metrics."""

  def relative_percent(fc, baseline, metric):
    fc = fc.where(fc.lead_time > np.timedelta64(0))
    if metric in ['rmse', 'seeps']:
      return (fc - baseline) / baseline * 100
    elif metric == 'acc':
      return (fc - baseline) / (1 - baseline) * 100

  baseline = results[reference]
  others = {k: v for k, v in results.items() if k != reference}
  relative = {
      k: relative_percent(v, baseline, metric) for k, v in others.items()
  }
  return relative


def compute_spread_skill_ratio(da: xr.DataArray) -> xr.DataArray:
  spread = da.sel(metric='ensemble_stddev')
  skill = da.sel(metric='ensemble_mean_rmse')
  ratio = spread / skill
  ratio = ratio.where(ratio.lead_time > np.timedelta64(0))
  return ratio


def plot_timeseries(
    results: t.Dict[str, xr.Dataset],
    metric: str,
    variable: str,
    level: t.Optional[int] = None,
    region: t.Optional[str] = None,
    colors: t.Optional[t.Dict[str, str]] = None,
    linestyles: t.Optional[t.Dict[str, int]] = None,
    marker: t.Optional[str] = None,
    markersize: t.Optional[int] = None,
    ax: t.Optional[matplotlib.axes.Axes] = None,
    add_legend: t.Optional[bool] = True,
    relative: t.Optional[str] = None,
    title: t.Optional[str] = None,
    xlabel: t.Optional[str] = None,
    ylabel: t.Optional[str] = None,
    ylim: t.Optional[t.Sequence[int]] = None,
    xlim: t.Optional[t.Sequence[int]] = None,
    labels: t.Optional[t.Dict[str, str]] = None,
    average_climatology: t.Optional[bool] = True,
    legend_position: t.Optional[int] = 2,
) -> matplotlib.axes.Axes:
  """Plot a time series panel."""
  if not ax:
    _, ax = plt.subplots()

  if relative is not None:
    results = compute_relative_metrics(
        results=results, reference=relative, metric=metric
    )
    ax.axhline(0, color='grey', zorder=0.1)

  for name, r in results.items():
    # Hack to get rid of climatology for relative and ACC plots
    if (relative is not None or metric in ['acc', 'spread/skill']) and (
        'climatology_' in name or 'persistence_' in name
    ):
      continue
    # Don't plot if variable not in results
    if not hasattr(r, variable):
      continue
    if metric == 'spread&skill':
      da = r[variable].sel(metric=['ensemble_mean_rmse', 'ensemble_stddev'])
    elif metric == 'spread/skill':
      da = compute_spread_skill_ratio(r[variable])
      ax.axhline(1, color='k')
    elif metric == '1-seeps':
      da = 1 - r[variable].sel(metric='seeps')
    else:
      da = r[variable].sel(metric=metric)
    if (
        not relative
        and metric in ['crps', 'rmse', 'spread&skill', 'rms_bias']
        and variable
        in [
            'specific_humidity',
            'total_precipitation_6hr',
            'total_precipitation_24hr',
        ]
    ):
      da = da * 1000.0
    label = name if labels is None else labels[name]
    if level is not None:
      da = da.sel(level=level)
    if region is not None and hasattr(da, 'region'):
      da = da.sel(region=region)
    if 'climatology_' in name and average_climatology:
      da = da.mean()

    if 'lead_time' in da.coords:
      if metric == 'spread&skill':
        da.sel(metric='ensemble_mean_rmse').plot(
            label=label + ' (Skill)',
            ax=ax,
            color=colors[name] if colors else None,
            ls='-',
        )
        da.sel(metric='ensemble_stddev').plot(
            label=label + ' (Spread)',
            ax=ax,
            color=colors[name] if colors else None,
            ls='--',
        )
      else:
        da.plot(
            label=label,
            ax=ax,
            color=colors[name] if colors else None,
            ls=linestyles[name] if linestyles else None,
            marker=marker,
            markersize=markersize,
        )
      datetime_to_xticks(da.lead_time, ax, xlim=xlim)
    else:
      ax.axhline(da, label=label, color=colors[name] if colors else None)
  if add_legend:
    ax.legend(loc=legend_position, fontsize=8)
  if title:
    ax.set_title(title, fontsize=12)
  if xlabel:
    ax.set_xlabel(xlabel)
  if ylabel:
    ax.set_ylabel(ylabel)
  if ylim:
    ax.set_ylim(ylim)
  if xlim:
    ax.set_xlim(xlim)
  return ax


def visualize_timeseries(
    viz_config: config.Viz,
    panel_configs: t.Sequence[config.Panel],
    save_path: t.Optional[str] = None,
    subplots_adjust_kwargs: t.Optional[t.Dict[str, float]] = None,
    legend_position: t.Optional[int] = 2,
) -> None:
  """Top-level visualization function."""
  set_wb2_style()

  results = load_results(viz_config.results)

  nrows, ncols = viz_config.layout or (1, len(panel_configs))
  fig, axs = plt.subplots(nrows, ncols, figsize=viz_config.figsize)

  for iax, (ax, panel_config) in enumerate(zip(axs.flat, panel_configs)):
    ax = plot_timeseries(
        results=results,
        metric=panel_config.metric,
        variable=panel_config.variable,
        level=panel_config.level,
        region=panel_config.region,
        colors=viz_config.colors,
        linestyles=viz_config.linestyles,
        marker=viz_config.marker,
        markersize=viz_config.markersize,
        ax=ax,
        add_legend=iax == 0,
        relative=panel_config.relative,
        title=panel_config.title,
        xlabel=panel_config.xlabel,
        ylabel=panel_config.ylabel,
        ylim=panel_config.ylim,
        xlim=panel_config.xlim,
        labels=viz_config.labels,
        legend_position=legend_position,
    )
  if viz_config.tight_layout:
    plt.tight_layout()
  if subplots_adjust_kwargs:
    plt.subplots_adjust(**subplots_adjust_kwargs)
  for ax in axs[:-1, :].flat:
    # ax.set_xticklabels([])
    ax.set_xlabel('')
  if save_path is not None:
    with fsspec.open(save_path, 'wb', auto_mkdir=True) as f:
      fig.savefig(f, **viz_config.save_kwargs)
      plt.close(fig)


def visualize_scorecard(
    viz_config: config.Viz,
    baseline: str,
    forecast: str,
    metric: str,
    region: t.Optional[str] = None,
    vars_3d: t.Optional[t.Sequence[str]] = None,
    vars_2d: t.Optional[t.Sequence[str]] = None,
    save_path: t.Optional[str] = None,
    cmap: t.Optional[str] = 'RdBu_r',
    cmap_scale: float = 100,
) -> None:
  """Plot relative scorecard."""
  matplotlib.rcParams.update(matplotlib.rcParamsDefault)

  # Compute relative
  results = load_results(viz_config.results)
  relative = (results[forecast] - results[baseline]) / results[baseline] * 100
  relative = relative.sel(metric=metric)
  if region is not None:
    relative = relative.sel(region=region)

  if vars_3d is None:
    vars_3d = [var for var in relative if hasattr(relative[var], 'level')]
  if vars_2d is None:
    vars_2d = [var for var in relative if not hasattr(relative[var], 'level')]

  def set_x_labels(ax, dataset):
    lead_time_h = int(dataset.lead_time[1] / np.timedelta64(1, 'h'))
    factor_24h = 24 // lead_time_h
    xticks = np.arange(0, len(dataset.lead_time), factor_24h)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks // factor_24h)
    ax.spines['top'].set_color('0.7')
    ax.spines['right'].set_color('0.7')
    ax.spines['bottom'].set_color('0.7')
    ax.spines['left'].set_color('0.7')

  def set_y_labels(ax, dataset, levels=True):
    ax.set_xticks([])
    if levels:
      ax.set_yticks(np.arange(len(dataset.level)))
      ax.set_yticklabels(dataset.level.values)
      # ax.tick_params(axis='y', which='major', pad=10)
    else:
      ax.set_yticks([0])
      ax.tick_params(axis='y', color='None')
      ax.set_yticklabels(['000'], color='None')

    ax.spines['top'].set_color('0.7')
    ax.spines['right'].set_color('0.7')
    ax.spines['bottom'].set_color('0.7')
    ax.spines['left'].set_color('0.7')

  def add_white_lines(ax, dataset):
    img = dataset.values
    for i in range(img.shape[0]):
      for j in range(img.shape[1]):
        rect = patches.Rectangle(
            (j - 0.5, i - 0.5),
            1,
            1,
            linewidth=2,
            edgecolor='white',
            facecolor='None',
        )
        ax.add_patch(rect)

  nvar_3d = len(vars_3d)
  nvar_2d = len(vars_2d)
  nlev = len(relative.level)

  ratio = (nvar_3d * nlev + nvar_2d) / len(relative.lead_time)
  fig_width = 12

  fig = plt.figure(figsize=(fig_width, fig_width * ratio))
  gs = GridSpec(
      nvar_3d * nlev + nvar_2d,
      len(relative.lead_time) + 1,
      figure=fig,
      hspace=0,
      left=0.1,
      right=0.9,
      top=0.9,
      bottom=0.1,
  )
  row_counter = 0
  for var in vars_3d:
    data = relative[var].transpose('level', 'lead_time')
    ax = fig.add_subplot(gs[row_counter : row_counter + nlev, :-1])
    if row_counter == 0:
      ax0 = ax
    img = ax.imshow(data, vmin=-cmap_scale, vmax=cmap_scale, cmap=cmap)
    add_white_lines(ax, data)
    ax.set_ylabel(long2short[var], rotation='horizontal', labelpad=20)
    set_y_labels(ax, relative, levels=True)
    row_counter += nlev

  for var in vars_2d:
    data = relative[var].expand_dims({'dummy': 1})
    ax = fig.add_subplot(gs[row_counter, :-1])
    img = ax.imshow(data, vmin=-cmap_scale, vmax=cmap_scale, cmap=cmap)
    add_white_lines(ax, data)
    set_y_labels(ax, relative, levels=False)
    ax.set_ylabel(long2short[var], rotation='horizontal', labelpad=20)
    row_counter += 1
  set_x_labels(ax, relative)
  ax.set_xlabel('Lead time (days)')

  ax0.set_title(
      f'{viz_config.labels[forecast]} RMSE relative to'
      f' {viz_config.labels[baseline]}'
  )

  cax = fig.add_subplot(gs[:, -1])
  fig.colorbar(img, cax=cax, orientation='vertical')

  if save_path is not None:
    with fsspec.open(save_path, 'wb', auto_mkdir=True) as f:
      fig.savefig(f, **viz_config.save_kwargs)
      plt.close(fig)
