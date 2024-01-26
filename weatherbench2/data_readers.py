"""
"""

from collections.abc import Iterable, Sequence
import dataclasses
import logging
from typing import Optional

import fsspec
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from weatherbench2 import schema
import xarray as xr


def _decode_pressure_level_suffixes(forecast: xr.Dataset) -> xr.Dataset:
  """Decode forecast dataset with pressure level as suffix into level dim."""
  das = []
  if hasattr(forecast, 'channel'):
    forecast = forecast['forecast'].to_dataset('channel')

  for var in forecast:
    # TODO(srasp): Consider writing this using regular expressions instead,
    # via something like re.fullmatch(r'(\w+)_(\d+)', channel)
    da = forecast[var]
    if var.split('_')[-1].isdigit():  # Check for pressure level suffix  # pytype: disable=attribute-error
      da = da.assign_coords(level=int(var.split('_')[-1]))  # pytype: disable=attribute-error
      da = da.rename('_'.join(var.split('_')[:-1])).expand_dims({'level': 1})  # pytype: disable=attribute-error
    das.append(da)

  ds = xr.merge(das)
  logging.info(f'Merged forecast: {ds}')
  return ds


def make_latitude_increasing(dataset: xr.Dataset) -> xr.Dataset:
  """Make sure latitude values are increasing. Flip dataset if necessary."""
  lat = dataset.latitude.values
  if (np.diff(lat) < 0).all():
    reverse_lat = lat[::-1]
    dataset = dataset.sel(latitude=reverse_lat)
  return dataset


def _ensure_nonempty(dataset: xr.Dataset, message: str = '') -> None:
  """Make sure dataset is nonempty."""
  if not min(dataset.dims.values()):
    raise ValueError(f'`dataset` was empty: {dataset.dims=}.  {message}')


@dataclasses.dataclass()
class DataReader:
  path: str
  variables: Optional[Sequence[str]] = None
  rename_variables: Optional[str] = None

  def get_chunk(self, time_chunk: xr.Dataset) -> xr.Dataset:
    """Return chunk of data for given time_chunk.

    Args:
      time_chunk: xr.Dataset with coordinates: init_time, lead_time and their
        combination valid_time

    Returns:
      chunk: xr.Dataset with corresponding data chunk
    """
    raise NotImplementedError()


@dataclasses.dataclass()
class GriddedFromZarr(DataReader):
  path: str
  levels: Optional[Sequence[int]] = None
  pressure_level_suffixes: bool = False
  ds: xr.Dataset = dataclasses.field(init=False)

  def __post_init__(self):
    # Open dataset using dask = lazy
    self.ds = xr.open_zarr(self.path)

    # Select variables and levels
    if self.variables is not None:
      self.ds = self.ds[self.variables]
    if self.levels is not None and hasattr(self.ds, 'level'):
      self.ds = self.ds.sel(level=self.levels)

    # Standardize dataset
    if self.pressure_level_suffixes:
      self.ds = _decode_pressure_level_suffixes(self.ds)
    if self.rename_variables is not None:
      self.ds = self.ds.rename(self.rename_variables)
    self.ds = make_latitude_increasing(self.ds)
    self.ds = schema.apply_time_conventions(self.ds, by_init=True)
    _ensure_nonempty(self.ds)


@dataclasses.dataclass()
class GriddedForecastFromZarr(GriddedFromZarr):

  def get_chunk(self, time_chunk: xr.Dataset) -> xr.Dataset:
    chunk = self.ds.sel(
        init_time=time_chunk.init_time, lead_time=time_chunk.lead_time
    )
    return chunk


@dataclasses.dataclass()
class GriddedGroundTruthFromZarr(GriddedFromZarr):

  def get_chunk(self, time_chunk: xr.Dataset) -> xr.Dataset:
    chunk = self.ds.sel(time=time_chunk.valid_time)
    return chunk


@dataclasses.dataclass()
class SparseGroundTruthFromParquet(DataReader):
  tolerance: Optional[np.timedelta64] = None
  time_dim: Optional[str] = 'timeObs'
  bad_quality_flags: Optional[Iterable[str]] = ('Z', 'B', 'X', 'Q', 'K', 'k')

  def _pick_closest_from_duplicates(
      self, df: pd.DataFrame, valid_time: np.datetime64
  ):
    df['timeDiff'] = np.abs(df[self.time_dim] - valid_time)
    df = df.sort_values('timeDiff', ascending=True)
    non_duplicated = df[~df['stationName'].duplicated(keep='first')]
    return non_duplicated

  # def _select_rows(
  #     self,
  #     texact: Optional[np.datetime64] = None,
  #     tmin: np.datetime64 = None,
  #     tmax: np.datetime64 = None
  #     ):
  #   if texact is not None:
  #     assert tmin is None and tmax is None, 'Either texact or tmin/tmax'
  #     df = pq.ParquetDataset(
  #       self.path,
  #       filters=[
  #           (f'{self.time_dim}Unix', '==', pd.to_datetime(texact).timestamp()),
  #       ],
  #       filesystem=fsspec.filesystem('gfile')
  #     ).read().to_pandas()
  #   else:
  #     df = pq.ParquetDataset(
  #       self.path,
  #       filters=[
  #           (f'{self.time_dim}Unix', '>=', pd.to_datetime(tmin).timestamp()),
  #           (f'{self.time_dim}Unix', '<=', pd.to_datetime(tmax).timestamp()),
  #       ],
  #       filesystem=fsspec.filesystem('gfile')
  #     ).read().to_pandas()
  #   return df

  def _get_all_valid_times(self, valid_times: np.ndarray) -> xr.Dataset:
    valid_times = np.unique(valid_times)

    # Load dataset for entire range of valid times
    # TODO(srasp): More fine grained filtering to avoid loading a bunch of unnecessary data
    global_tmin = valid_times.min()
    global_tmax = valid_times.max()
    if self.tolerance is not None:
      global_tmin = global_tmin - self.tolerance
      global_tmax = global_tmax + self.tolerance
    df = (
        pq.ParquetDataset(
            self.path,
            filters=[
                (
                    f'{self.time_dim}Unix',
                    '>=',
                    pd.to_datetime(global_tmin).timestamp(),
                ),
                (
                    f'{self.time_dim}Unix',
                    '<=',
                    pd.to_datetime(global_tmax).timestamp(),
                ),
            ],
            filesystem=fsspec.filesystem('gfile'),
        )
        .read()
        .to_pandas()
    )

    obs_dss = []
    for valid_time in valid_times:
      if self.tolerance is not None:
        tmin = valid_time - self.tolerance
        tmax = valid_time + self.tolerance
        obs_selection = df[
            np.logical_and(
                df[f'{self.time_dim}'] >= tmin, df[f'{self.time_dim}'] <= tmax
            )
        ]
      else:
        obs_selection = df[df[f'{self.time_dim}'] == valid_time]
      obs_selection = self._pick_closest_from_duplicates(
          obs_selection.copy(), valid_time
      )
      obs_ds = (
          obs_selection.set_index('stationName')
          .to_xarray()
          .assign_coords(time=valid_time)
      )
      obs_dss.append(obs_ds)
    return xr.concat(obs_dss, dim='time')

  def _set_bad_quality_to_nan(self, ds: xr.Dataset):
    for variable in self.variables:
      for r in self.bad_quality_flags:
        ds[variable] = ds[variable].where(ds[variable + 'DD'] != r)
    return ds

  def get_chunk(self, time_chunk: xr.Dataset) -> xr.Dataset:
    obs_chunk = self._get_all_valid_times(time_chunk.valid_time.values)
    if self.bad_quality_flags is not None:
      obs_chunk = self._set_bad_quality_to_nan(obs_chunk)
    # Convert to 0-360 longitude
    obs_chunk['longitude'] = obs_chunk.longitude.where(
        obs_chunk.longitude >= 0, 360 + obs_chunk.longitude
    )
    # Ignore lat/lon NaNs and make coord
    obs_chunk['longitude'] = obs_chunk['longitude'].mean('time')
    obs_chunk['latitude'] = obs_chunk['latitude'].mean('time')
    obs_chunk = obs_chunk.set_coords(['latitude', 'longitude'])
    obs_chunk = obs_chunk[self.variables]
    return obs_chunk.sel(time=time_chunk.valid_time)  # pytype: disable=bad-return-type
