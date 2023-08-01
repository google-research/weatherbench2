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
"""Evaluation routines for WB2.

Evaluation functions, including reading data, computing metrics and
saving results for a given config file.
"""
from __future__ import annotations

from collections import abc
import copy
import dataclasses
import logging
import os.path
from typing import Any, Optional

import apache_beam as beam
import fsspec
import numpy as np
import xarray as xr
import xarray_beam as xbeam

from weatherbench2 import config
from weatherbench2 import schema
from weatherbench2.utils import make_probabilistic_climatology

# pylint: disable=logging-fstring-interpolation


def make_latitude_increasing(dataset) -> xr.Dataset:
  """Make sure latitude values are increasing. Flip dataset if necessary."""
  lat = dataset.latitude.values
  if (np.diff(lat) < 0).all():
    reverse_lat = lat[::-1]
    dataset = dataset.sel(latitude=reverse_lat)
  return dataset


def _ensure_aligned_grid(
    dataset: xr.Dataset,
    target: xr.Dataset,
    atol: float = 1e-3,
) -> xr.Dataset:
  """Ensure that the horizontal coordinates on dataset exactly match target."""
  for coord_name in ['latitude', 'longitude']:
    np.testing.assert_allclose(
        dataset[coord_name].data, target[coord_name].data, atol=atol
    )
  return dataset.assign_coords(
      latitude=target.latitude, longitude=target.longitude
  )


def _ensure_nonempty(dataset: xr.Dataset, message: str = '') -> None:
  """Make sure dataset is nonempty."""
  if not min(dataset.dims.values()):
    raise ValueError(f'`dataset` was empty: {dataset.dims=}.  {message}')


def _decode_pressure_level_suffixes(forecast: xr.Dataset) -> xr.Dataset:
  """Decode forecast dataset with pressure level as suffix into level dim."""
  das = []
  if hasattr(forecast, 'channel'):
    forecast = forecast['forecast'].to_dataset('channel')

  for var in forecast:
    # TODO(srasp): Consider writing this using regular expressions instead,
    # via something like re.fullmatch(r'(\w+)_(\d+)', channel)
    da = forecast[var]
    if var.split('_')[-1].isdigit():  # Check for pressure level suffix
      da = da.assign_coords(level=int(var.split('_')[-1]))
      da = da.rename('_'.join(var.split('_')[:-1])).expand_dims({'level': 1})
    das.append(da)

  ds = xr.merge(das)
  logging.info(f'Merged forecast: {ds}')
  return ds


def open_source_files(
    forecast_path: str,
    obs_path: str,
    by_init: bool = False,
    use_dask: bool = False,
    rename_variables: Optional[dict[str, str]] = None,
    pressure_level_suffixes: bool = False,
) -> tuple[xr.Dataset, xr.Dataset]:
  """Open forecast and ground obs Zarr files and standardize them.

  Args:
    forecast_path: Path to forecast files.
    obs_path: Path to groud-truth file.
    by_init: Specifies whether forecast is in by-init or by-valid convention.
    use_dask: Specifies whether to use dask to open Zarr store. Otherwise load
      lazy numpy array.
    rename_variables: Rename dimensions and variables according to given
      dictionary.
    pressure_level_suffixes: Whether to decide variables with pressure levels as
      suffixes.

  Returns:
    (forecast, obs): Tuple containing forecast and ground-truth datasets.
  """
  obs = xr.open_zarr(obs_path, chunks='auto' if (use_dask or by_init) else None)
  forecast = xr.open_zarr(
      forecast_path,
      # Use dask to decode pressure levels since xr's expand_dims is not lazy
      chunks='auto' if (use_dask or pressure_level_suffixes) else None,
  )

  if pressure_level_suffixes:
    forecast = _decode_pressure_level_suffixes(forecast)
  if rename_variables is not None:
    forecast = forecast.rename(rename_variables)

  obs = make_latitude_increasing(obs)
  forecast = make_latitude_increasing(forecast)
  forecast = _ensure_aligned_grid(forecast, obs)
  forecast = schema.apply_time_conventions(forecast, by_init=by_init)

  _ensure_nonempty(obs)
  _ensure_nonempty(forecast)

  return forecast, obs


def _impose_data_selection(
    dataset: xr.Dataset,
    selection: config.Selection,
    select_time: bool = True,
    time_dim: Optional[str] = None,
) -> xr.Dataset:
  """Returns selection of dataset specified in Selection instance."""
  dataset = dataset[selection.variables].sel(
      latitude=selection.lat_slice,
      longitude=selection.lon_slice,
  )
  if selection.levels is not None and hasattr(dataset, 'level'):
    dataset = dataset.sel(level=selection.levels)
  if select_time:
    dataset = dataset.sel({time_dim: selection.time_slice})
  _ensure_nonempty(dataset, message='Selection created empty dataset')
  return dataset


def create_persistence_forecast(
    forecast: xr.Dataset,
    obs: xr.Dataset,
) -> xr.Dataset:
  """Create persistence forecast from observation with same shape as forecast.

  Warning: For by-valid this is not 100% correct. Truth has already been sliced
  in time, same as forecast.time. However, init_time will go back further. For
  now, select only available times and raise warning.

  Args:
    forecast: Forecast dataset with dimensions init_time and lead_time.
    obs: Ground-truth dataset with time dimensions.

  Returns:
    persistence_forecast: Ground-truth dataset at forecast initialization time
    with same dimensions as forecast.
  """
  logging.warning('by-valid with evaluate_persistence is not 100% correct.')
  init_time = forecast.init_time
  init_time = init_time.sel(
      time=slice(init_time.time[0] + init_time.lead_time.max(), None)
  )
  persistence_forecast = (
      obs.sel(time=init_time.rename({'time': 'valid_time'}))
      .drop_vars('time')
      .rename({'valid_time': 'time'})
  )
  return persistence_forecast


def _unique_step_size(data: np.ndarray) -> Any:
  """Ensure all lead time steps are the same."""
  if data.ndim != 1:
    raise ValueError(f'array has wrong number of dimensions: {data.ndim}')
  if len(data) < 2:
    raise ValueError(f'{len(data)=}, which is too small to determine step size')
  uniques = np.unique(np.diff(data))
  if uniques.size != 1:
    raise ValueError(f'too many unique values: {uniques}')
  return uniques[0]


def _ensure_consistent_time_step_sizes(
    truth: xr.Dataset, forecast: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
  """Ensure consistent time-step sizes between truth and forecasts."""
  truth_time_step = _unique_step_size(truth['time'].data)
  forecast_time_step = _unique_step_size(forecast['time'].data)
  if truth_time_step > forecast_time_step:
    multiple, remainder = divmod(truth_time_step, forecast_time_step)
    if remainder:
      raise ValueError(
          'truth time step not a multiple of forecast time step: '
          f'{truth_time_step} vs {forecast_time_step}'
      )
    forecast = forecast.thin(time=int(multiple))
  elif truth_time_step < forecast_time_step:
    multiple, remainder = divmod(forecast_time_step, truth_time_step)
    if remainder:
      raise ValueError(
          'forecast time step not a multiple of truth time step: '
          f'{forecast_time_step} vs {truth_time_step}'
      )
    truth = truth.thin(time=int(multiple))
  return truth, forecast


def _add_base_variables(
    data_config: config.DataConfig, eval_config: config.EvalConfig
) -> config.DataConfig:
  """Add required base variables for computing derived variables.

  Args:
    data_config: Raw data config.
    eval_config: Eval config that contains derived variable objects.

  Returns:
    data_config: Deepcopied data_config with base variables added and derived
      variables removed from data_config.selection.variables.
  """
  data_config = copy.deepcopy(data_config)

  for derived_variable in eval_config.derived_variables:
    # Add base variables
    data_config.selection.variables = list(
        set(data_config.selection.variables).union(
            derived_variable.base_variables
        )
    )

  return data_config


def _select_analysis_init_time(
    forecast: xr.Dataset, forecast_all_times: xr.Dataset
) -> xr.Dataset:
  """Selects appropriate forecast/analysis pairings for init-time convention."""
  analysis = forecast_all_times.sel(lead_time=np.timedelta64(0), drop=True)
  analysis = analysis.rename({'init_time': 'time'})

  init_interval = analysis.time.diff('time')
  if not (init_interval == init_interval[0]).all():
    raise ValueError(f'Not all init_time intervals are equal: {init_interval}')

  init_interval = init_interval[0]

  lead_interval = forecast.lead_time.diff('lead_time')
  assert np.all(
      lead_interval == lead_interval[0]
  ), 'Not all lead_time intervals are equal.'
  lead_interval = lead_interval[0]

  assert (
      init_interval >= lead_interval
  ), 'Initialization interval cannot be less that lead_time interval.'

  lead_per_init = float((init_interval / lead_interval).values)
  assert lead_per_init.is_integer(), 'Init must be multiple of lead.'
  lead_per_init = int(lead_per_init)

  assert (
      analysis.time.max() >= forecast.valid_time.max()
  ), 'Analysis does not extend to latest forecast init+lead'

  # Need to select appropriate lead_times from forecasts
  # Corresponding to initialization interval
  forecast = forecast.isel(lead_time=slice(None, None, lead_per_init))
  return forecast, analysis


def open_forecast_and_truth_datasets(
    data_config: config.DataConfig,
    eval_config: config.EvalConfig,
    use_dask: bool = False,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset | None]:
  """Open datasets and select desired slices.

  Args:
    data_config: DataConfig instance.
    eval_config: EvalConfig instance.
    use_dask: Specifies whether to open datasets using dask.

  Returns:
    (forecast, truth, climatology): Tuple containing datasets. Climatology is
      None if not in data_config.
  """
  data_config = _add_base_variables(data_config, eval_config)

  logging.info('Loading data')
  forecast, obs = open_source_files(
      forecast_path=data_config.paths.forecast,
      obs_path=data_config.paths.obs,
      by_init=data_config.by_init,
      use_dask=use_dask,
      rename_variables=data_config.rename_variables,
      pressure_level_suffixes=data_config.pressure_level_suffixes,
  )

  obs_all_times = _impose_data_selection(
      obs, data_config.selection, select_time=False
  )
  forecast_all_times = _impose_data_selection(
      forecast, data_config.selection, select_time=False
  )

  if data_config.by_init:  # Will select appropriate chunks later
    obs = obs_all_times
  else:
    obs = _impose_data_selection(obs, data_config.selection, time_dim='time')
  forecast = _impose_data_selection(
      forecast,
      data_config.selection,
      time_dim='init_time' if data_config.by_init else 'time',
  )

  # Determine ground truth dataset
  if eval_config.against_analysis:
    eval_truth = forecast.sel(lead_time=np.timedelta64(0), drop=True)
    if data_config.by_init:
      forecast, eval_truth = _select_analysis_init_time(
          forecast, forecast_all_times
      )
  else:
    eval_truth = obs

  if not data_config.by_init:
    eval_truth, forecast = _ensure_consistent_time_step_sizes(
        eval_truth, forecast
    )

  if eval_config.evaluate_climatology:
    climatology = xr.open_zarr(data_config.paths.climatology)
    climatology = make_latitude_increasing(climatology)
  else:
    climatology = None

  return (forecast, eval_truth, climatology)


def _get_output_path(
    data_config: config.DataConfig, eval_name: str, output_format: str
) -> str:
  if output_format == 'netcdf':
    suffix = 'nc'
  elif output_format == 'zarr':
    suffix = 'zarr'
  else:
    raise ValueError(f'unrecogonized data format: {output_format}')
  return os.path.join(
      data_config.paths.output_dir,
      f'{data_config.paths.output_file_prefix}{eval_name}.{suffix}',
  )


def _to_netcdf(dataset: xr.Dataset, filename: str) -> None:
  with fsspec.open(filename, 'wb', auto_mkdir=True) as f:
    f.write(dataset.to_netcdf())


def _metric_and_region_loop(
    forecast: xr.Dataset,
    truth: xr.Dataset,
    eval_config: config.EvalConfig,
    compute_chunk: bool = False,
) -> xr.Dataset:
  """Compute metric results looping over metrics and regions in eval config."""
  # Compute derived variables
  for derived_variable in eval_config.derived_variables:
    logging.info(f'Logging: derived_variable: {derived_variable}')
    forecast[derived_variable.variable_name] = derived_variable.compute(
        forecast
    )
    truth[derived_variable.variable_name] = derived_variable.compute(truth)

  results = []
  for name, metric in eval_config.metrics.items():
    logging.info(f'Logging metric: {name}')
    # Add a metric dimension, to be concatenated later
    metric_dim = xr.DataArray([name], coords={'metric': [name]})
    if compute_chunk or not eval_config.temporal_mean:
      eval_fn = metric.compute_chunk
    else:
      eval_fn = metric.compute
    if eval_config.regions is not None:
      tmp_results = []  # For storing different regions
      for region_name, region in eval_config.regions.items():
        logging.info(f'Logging region: {region_name}')
        region_dim = xr.DataArray(
            [region_name], coords={'region': [region_name]}
        )
        tmp_result = eval_fn(forecast=forecast, truth=truth, region=region)
        tmp_results.append(
            tmp_result.expand_dims({'metric': metric_dim, 'region': region_dim})
        )
        logging.info(f'Logging region done: {region_name}')
      result = xr.concat(tmp_results, 'region')
    else:
      result = eval_fn(forecast=forecast, truth=truth).expand_dims(
          {'metric': metric_dim}
      )
    results.append(result)
    logging.info(f'Logging metric done: {name}')
  results = xr.merge(results)
  return results


def _evaluate_all_metrics(
    eval_name: str,
    eval_config: config.EvalConfig,
    data_config: config.DataConfig,
) -> None:
  """Evaluate a set of eval metrics in memory."""
  forecast, truth, climatology = open_forecast_and_truth_datasets(
      data_config, eval_config, use_dask=True
  )

  if eval_config.evaluate_climatology:
    time_dim = 'valid_time' if data_config.by_init else 'time'
    forecast = climatology[list(forecast.keys())].sel(
        dayofyear=forecast[time_dim].dt.dayofyear,
        hour=forecast[time_dim].dt.hour,
    )
  if eval_config.evaluate_probabilistic_climatology:
    probabilistic_climatology = make_probabilistic_climatology(
        truth,
        eval_config.probabilistic_climatology_start_year,
        eval_config.probabilistic_climatology_end_year,
        eval_config.probabilistic_climatology_hour_interval,
    )
    time_dim = 'valid_time' if data_config.by_init else 'time'
    forecast = probabilistic_climatology[list(forecast.keys())].sel(
        dayofyear=forecast[time_dim].dt.dayofyear,
        hour=forecast[time_dim].dt.hour,
    )

  if eval_config.evaluate_persistence:
    forecast = create_persistence_forecast(forecast, truth)

  if data_config.by_init:
    truth = truth.sel(time=forecast.valid_time)

  results = _metric_and_region_loop(forecast, truth, eval_config)

  logging.info(f'Logging Evaluation complete:\n{results}')

  output_path = _get_output_path(data_config, eval_name, 'netcdf')
  _to_netcdf(results, output_path)
  logging.info(f'Logging Saved results to {output_path}')


def evaluate_in_memory(
    data_config: config.DataConfig,
    eval_configs: dict[str, config.EvalConfig],
) -> None:
  """Run evaluation in memory.

  Will save a separate results NetCDF file for each EvalConfig.
  An example for a results dataset with the respective dimensions is given
  below. Note that region and level are optional.

  ```
  <xarray.Dataset>
  Dimensions:              (lead_time: 21, region: 3, level: 3, metric: 2)
  Coordinates:
    * lead_time            (lead_time) timedelta64[ns] 0 days 00:00:00 ...
    * region               (region) object 'global' 'tropics' 'extra-tropics'
    * level                (level) int32 500 700 850
    * metric               (metric) object 'rmse' 'acc'
  Data variables:
      geopotential         (metric, region, lead_time, level) float64 ...
      2m_temperature       (metric, region, lead_time) float64 0.6337 ...
  ```

  Args:
    data_config: DataConfig instance.
    eval_configs: Dictionary of EvalConfig instances.
  """
  for eval_name, eval_config in eval_configs.items():
    _evaluate_all_metrics(eval_name, eval_config, data_config)


@dataclasses.dataclass
class _SaveOutputs(beam.PTransform):
  """Save outputs to Zarr or netCDF."""

  eval_name: str
  data_config: config.DataConfig
  output_format: str

  def _write_netcdf(self, datasets: list[xr.Dataset]) -> xr.Dataset:
    combined = xr.combine_by_coords(datasets)
    output_path = _get_output_path(
        self.data_config, self.eval_name, self.output_format
    )
    _to_netcdf(combined, output_path)

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    if self.output_format == 'netcdf':
      return (
          pcoll
          | 'DropKey' >> beam.MapTuple(lambda k, v: v)
          | beam.combiners.ToList()
          | beam.Map(self._write_netcdf)
      )
    elif self.output_format == 'zarr':
      output_path = _get_output_path(
          self.data_config, self.eval_name, self.output_format
      )
      return pcoll | xbeam.ChunksToZarr(output_path)
    else:
      raise ValueError(f'unrecogonized data format: {self.output_format}')


@dataclasses.dataclass
class _EvaluateAllMetrics(beam.PTransform):
  """Evaluate a set of eval metrics using a Beam pipeline.

  Attributes:
    eval_name: Name of evaluation.
    eval_config: EvalCongig instance.
    data_config: DataConfig instance.
    input_chunks: Chunks to use for input files.
    fanout: Fanout parameter for Beam combiners.
  """

  eval_name: str
  eval_config: config.EvalConfig
  data_config: config.DataConfig
  input_chunks: abc.Mapping[str, int]
  fanout: Optional[int] = None

  def _evaluate_chunk(
      self,
      key: xbeam.Key,
      forecast_and_truth: list[xr.Dataset],
  ) -> tuple[xbeam.Key, xr.Dataset]:
    forecast, truth = forecast_and_truth
    logging.info(f'Logging _evaluate_chunk Key: {key}')
    results = _metric_and_region_loop(
        forecast, truth, self.eval_config, compute_chunk=True
    )
    dropped_dims = [dim for dim in key.offsets if dim not in results.dims]
    result_key = key.with_offsets(**{dim: None for dim in dropped_dims})
    return result_key, results

  def _sel_corresponding_truth_chunk(
      self, key: xbeam.Key, forecast_chunk: xr.Dataset, truth: xr.Dataset
  ) -> tuple[xr.Dataset, xr.Dataset]:
    truth_chunk = truth.sel(time=forecast_chunk.valid_time).compute()
    return (forecast_chunk, truth_chunk)

  def _climatology_like_forecast_chunk(
      self,
      key: xbeam.Key,
      chunks: tuple[xr.Dataset, xr.Dataset],
      climatology: xr.Dataset,
      variables: list[str],
      by_init: bool,
  ) -> tuple[xbeam.Key, xr.Dataset]:
    forecast_chunk, truth_chunk = chunks
    # Load the data, using a separate thread for each variable
    num_threads = len(variables)
    time_dim = 'valid_time' if by_init else 'time'
    climatology_chunk = (
        climatology[variables]
        .sel(
            dayofyear=forecast_chunk[time_dim].dt.dayofyear,
            hour=forecast_chunk[time_dim].dt.hour,
        )
        .chunk()
        .compute(num_workers=num_threads)
    )
    return key, (climatology_chunk, truth_chunk)

  def _persistence_like_forecast_chunk(
      self,
      key: xbeam.Key,
      chunks: tuple[xr.Dataset, xr.Dataset],
      truth: xr.Dataset,
      variables: list[str],
      by_init: bool,
  ) -> tuple[xbeam.Key, xr.Dataset]:
    forecast_chunk, truth_chunk = chunks
    num_threads = len(variables)
    if by_init:
      persistence_chunk = truth.sel(time=forecast_chunk.init_time).compute(
          num_workers=num_threads
      )
      persistence_chunk = persistence_chunk.expand_dims(
          lead_time=forecast_chunk.lead_time
      ).assign_coords({'valid_time': forecast_chunk.valid_time})
    else:
      raise NotImplementedError(
          'Persistence not compatible with by-valid format.'
      )
    return key, (persistence_chunk, truth_chunk)

  def _pipeline(
      self,
      pcoll: beam.PCollection,
      forecast: xr.Dataset,
      truth: xr.Dataset,
      climatology: xr.Dataset,
      by_init: bool,
  ) -> beam.PCollection:
    if (
        self.eval_config.evaluate_climatology
        or self.eval_config.evaluate_probabilistic_climatology
        or self.eval_config.evaluate_persistence
    ):
      variables = list(forecast.keys())
      forecast = forecast.drop(variables)

    if by_init:
      forecast_pipeline = xbeam.DatasetToChunks(
          forecast,
          self.input_chunks,
          split_vars=False,
      ) | beam.MapTuple(
          lambda k, v: (k, self._sel_corresponding_truth_chunk(k, v, truth))
      )
    else:
      forecast_pipeline = xbeam.DatasetToChunks(
          [forecast, truth],
          self.input_chunks,
          split_vars=False,
      )

    if self.eval_config.evaluate_climatology:
      forecast_pipeline = forecast_pipeline | beam.MapTuple(
          lambda k, v: self._climatology_like_forecast_chunk(  # pylint: disable=g-long-lambda
              k, v, climatology, variables, by_init
          ),
      )
    if self.eval_config.evaluate_probabilistic_climatology:
      probabilistic_climatology = make_probabilistic_climatology(
          truth,
          self.eval_config.probabilistic_climatology_start_year,
          self.eval_config.probabilistic_climatology_end_year,
          self.eval_config.probabilistic_climatology_hour_interval,
      )
      forecast_pipeline = forecast_pipeline | beam.MapTuple(
          lambda k, v: self._climatology_like_forecast_chunk(  # pylint: disable=g-long-lambda
              k, v, probabilistic_climatology, variables, by_init
          ),
      )
    elif self.eval_config.evaluate_persistence:
      forecast_pipeline = forecast_pipeline | beam.MapTuple(
          lambda k, v: self._persistence_like_forecast_chunk(  # pylint: disable=g-long-lambda
              k, v, truth, variables, by_init
          ),
      )

    forecast_pipeline = forecast_pipeline | 'EvaluateChunk' >> beam.MapTuple(
        self._evaluate_chunk
    )
    if self.eval_config.temporal_mean:
      forecast_pipeline = forecast_pipeline | 'TemporalMean' >> xbeam.Mean(
          dim='init_time' if by_init else 'time', fanout=self.fanout
      )

    forecast_pipeline = forecast_pipeline | 'SaveOutputs' >> _SaveOutputs(
        self.eval_name, self.data_config, self.eval_config.output_format
    )
    return pcoll | forecast_pipeline

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    forecast, truth, climatology = open_forecast_and_truth_datasets(
        self.data_config, self.eval_config
    )
    logging.info(f'Logging 1 forecast {forecast}')
    logging.info(f'Logging 1 truth {truth}')
    return self._pipeline(
        pcoll, forecast, truth, climatology, by_init=self.data_config.by_init
    )


def evaluate_with_beam(
    data_config: config.DataConfig,
    eval_configs: dict[str, config.EvalConfig],
    *,
    input_chunks: abc.Mapping[str, int],
    runner: str,
    fanout: Optional[int] = None,
    argv: Optional[list[str]] = None,
) -> None:
  """Run evaluation with a Beam pipeline.

  Will save a separate results NetCDF file for each EvalConfig.
  An example for a results dataset with the respective dimensions is given
  below. Note that region and level are optional.

  ```
  <xarray.Dataset>
  Dimensions:              (lead_time: 21, region: 3, level: 3, metric: 2)
  Coordinates:
    * lead_time            (lead_time) timedelta64[ns] 0 days 00:00:00 ...
    * region               (region) object 'global' 'tropics' 'extra-tropics'
    * level                (level) int32 500 700 850
    * metric               (metric) object 'rmse' 'acc'
  Data variables:
      geopotential         (metric, region, lead_time, level) float64 ...
      2m_temperature       (metric, region, lead_time) float64 0.6337 ...
  ```

  Args:
    data_config: DataConfig instance.
    eval_configs: Dictionary of EvalConfig instances.
    input_chunks: Chunking of input datasets.
    runner: Beam runner.
    fanout: Beam CombineFn fanout.
    argv: Other arguments to pass into the Beam pipeline.
  """

  with beam.Pipeline(runner=runner, argv=argv) as root:
    for eval_name, eval_config in eval_configs.items():
      logging.info(f'Logging Eval config: {eval_config}')
      _ = root | f'evaluate_{eval_name}' >> _EvaluateAllMetrics(
          eval_name, eval_config, data_config, input_chunks, fanout=fanout
      )
