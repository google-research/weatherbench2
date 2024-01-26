import dataclasses
import typing as t
from typing import Optional

import numpy as np
from weatherbench2 import metrics
import xarray as xr

@dataclasses.dataclass
class Aggregation:
  skipna: bool = False

  def aggregate_in_space(self, statistic):
    raise NotImplementedError

@dataclasses.dataclass
class NoAggregation(Aggregation):

  def aggregate_in_space(self, statistic):
    return statistic

@dataclasses.dataclass
class LatLonAverage(Aggregation):

  def aggregate_in_space(self, statistic):
    weights = metrics.get_lat_weights(statistic)
    return statistic.weighted(weights).mean(
        ('latitude', 'longitude'), skipna=self.skipna
    )


# TODO: Implement regions, also for Stations
@dataclasses.dataclass
class UnweightedAverage(Aggregation):
  dims: list[str] = dataclasses.field(default_factory=list)

  def aggregate_in_space(self, statistic):
    return statistic.mean(self.dims, skipna=self.skipna)

@dataclasses.dataclass
class WeightedStationAverage(Aggregation):
  station_dim: str = 'stationName'
  weights: Optional[xr.DataArray] = None
  alpha_0: float = 0.75
  min_weight: float = 1
  max_weight: float = 10

  def aggregate_in_space(self, statistic):

    return statistic.weighted(self.weights).mean(
        self.station_dim, skipna=self.skipna
    )