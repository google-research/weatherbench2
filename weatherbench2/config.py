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
"""Configuration classes."""

import dataclasses
import typing as t
import numpy as np
from weatherbench2 import data_readers
from weatherbench2.derived_variables import DerivedVariable
from weatherbench2.metrics import Metric
# from weatherbench2.regions import ExtraTropicalRegion
# from weatherbench2.regions import LandRegion
# from weatherbench2.regions import Region
# from weatherbench2.regions import SliceRegion
from weatherbench2.aggregations import Aggregation


@dataclasses.dataclass
class Selection:
  """Select a sub-set of forecast and truth data.

  Attributes:
    variables: List of variables to evaluate.
    time_slice: Range of time/init_time to use from forecast.
    levels: List of pressure levels.
    lat_slice: Latitude range in degrees.
    lon_slice: Longitude range in degrees.
  """

  variables: t.Sequence[str]
  time_slice: t.Optional[slice] = None
  time_start: t.Optional[str] = None
  time_end: t.Optional[str] = None
  init_frequency: t.Optional[np.timedelta64] = None
  lead_times: t.Optional[np.ndarray] = None
  levels: t.Optional[t.Sequence[int]] = None
  lat_slice: t.Optional[slice] = dataclasses.field(
      default_factory=lambda: slice(None, None)
  )
  lon_slice: t.Optional[slice] = dataclasses.field(
      default_factory=lambda: slice(None, None)
  )
  chunks: t.Optional[t.Mapping[str, int]] = None


@dataclasses.dataclass
class Paths:
  """Configuration for input and output paths.

  Attributes:
    forecast: Path to forecast file.
    obs: Path to ground-truth file.
    output_dir: Path to output directory.
    output_file_prefix: Prefix for output file name.
    climatology: Path to optional climatology file.
  """

  output_dir: str
  forecast: t.Optional[str] = None
  obs: t.Optional[str] = None
  forecast_data_loader: t.Optional[data_readers.DataReader] = None
  # TODO(srasp): Naming mismatch, obs here, truth in evaluation.py
  obs_data_loader: t.Optional[data_readers.DataReader] = None
  output_file_prefix: t.Optional[str] = ''
  climatology: t.Optional[str] = None


@dataclasses.dataclass
class Data:
  """Data configuration class combining Selection and Paths.

  Attributes:
    selection: Selection instance.
    paths: Paths instance.
    by_init: Specifies whether forecast file follows by-init or by-valid
      convention (see official documentation).
    rename_variables: Optional dictionary to convert forecast dimension an
      variable names to WB2 convention.
    pressure_level_suffixes: Specifies whether forecast variables are stored
      with pressure level suffixes instead of a level dimension, e.g.
      "geopotential_500".
  """

  selection: Selection
  paths: Paths
  by_init: t.Optional[bool] = True
  rename_variables: t.Optional[t.Dict[str, str]] = None
  pressure_level_suffixes: t.Optional[bool] = False
  grid2sparse: t.Optional[bool] = False
  grid2sparse_method: t.Optional[str] = 'nearest'

  def __post_init__(self):
    if self.paths.forecast_data_loader is None:
      self.paths.forecast_data_loader = data_readers.GriddedForecastFromZarr(
          path=self.paths.forecast,
          variables=self.selection.variables,
          rename_variables=self.rename_variables,
          levels=self.selection.levels,
          pressure_level_suffixes=self.pressure_level_suffixes,
      )
    if self.paths.obs_data_loader is None:
      self.paths.obs_data_loader = data_readers.GriddedGroundTruthFromZarr(
          path=self.paths.obs,
          variables=self.selection.variables,
          rename_variables=self.rename_variables,
          levels=self.selection.levels,
      )
    if self.selection.time_slice is not None:
      # Take template from fc file
      template = self.paths.forecast_data_loader.ds.sel(
          init_time=self.selection.time_slice
      )
      self.selection.template = template
    else:
      self.selection.template = None


@dataclasses.dataclass
class Eval:
  """Evaluation configuration class.

  Attributes:
    metrics: Dictionary of Metric instances.
    regions: Optional dictionary of Region instances.
    evaluate_persistence: Evaluate persistence forecast, i.e. forecast at t=0.
    evaluate_climatology: Evaluate climatology forecast.
    evaluate_probabilistic_climatology: Evaluate probabilistic climatology,
      derived from using each year of the ground-truth dataset as a member.
    probabilistic_climatology_start_year: First year of ground-truth to use for
      probabilistic climatology.
    probabilistic_climatology_end_year: Last year of ground-truth to use for
      probabilistic climatology.
    probabilistic_climatology_hour_interval: Hour interval to compute
      probabilistic climatology.
    against_analysis: Use forecast at t=0 as ground-truth. Warning: only for
      by-valid convention. For by-init, specify analysis dataset as obs.
    derived_variables: dict of DerivedVariable instances to compute on the fly.
    temporal_mean: Compute temporal mean (over time/init_time) for metrics.
    output_format: Wether to save to 'netcdf' or 'zarr'.
  """

  metrics: t.Dict[str, Metric]
  # regions: t.Optional[
  #     t.Dict[str, t.Union[Region, ExtraTropicalRegion, SliceRegion, LandRegion]]
  # ] = None
  aggregations: t.Optional[t.Dict[str, Aggregation]] = None
  evaluate_persistence: t.Optional[bool] = False
  evaluate_climatology: t.Optional[bool] = False
  evaluate_probabilistic_climatology: t.Optional[bool] = False
  probabilistic_climatology_start_year: t.Optional[int] = None
  probabilistic_climatology_end_year: t.Optional[int] = None
  probabilistic_climatology_hour_interval: t.Optional[int] = None
  against_analysis: t.Optional[bool] = False
  derived_variables: t.Dict[str, DerivedVariable] = dataclasses.field(
      default_factory=dict
  )
  temporal_mean: t.Optional[bool] = True
  # output_format='zarr' is also supported, but may be buggy due to
  # https://github.com/google/xarray-beam/issues/85
  output_format: str = 'netcdf'


@dataclasses.dataclass
class Viz:
  """Visualization configuration class."""

  results: t.Dict[str, str]
  save_kwargs: t.Dict[str, t.Any] = dataclasses.field(default_factory=dict)
  colors: t.Optional[t.Dict[str, str]] = None
  layout: t.Optional[t.Tuple[int, int]] = None
  figsize: t.Optional[t.Tuple[int, int]] = None
  tight_layout: t.Optional[bool] = True
  labels: t.Optional[t.Dict[str, str]] = None
  linestyles: t.Optional[t.Dict[str, str]] = None
  marker: t.Optional[str] = None
  markersize: t.Optional[int] = None


@dataclasses.dataclass
class Panel:
  """Config for each panel."""

  metric: str
  variable: str
  level: t.Optional[int] = None
  region: t.Optional[str] = None
  relative: t.Optional[str] = None
  title: t.Optional[str] = None
  xlabel: t.Optional[str] = None
  ylabel: t.Optional[str] = None
  ylim: t.Optional[tuple[str]] = None
  xlim: t.Optional[tuple[str]] = None