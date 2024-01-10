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
"""Configuration files for evaluation and visualization."""

import dataclasses
import typing as t

import numpy as np
import xarray as xr


@dataclasses.dataclass
class Region:
  """Region selector for spatially averaged metrics.

  .apply() method is called before spatial averaging in the Metrics classes.
  Region selection can be either applied as an operation on the dataset itself
  or a weights dataset, typically the latitude weights. The latter option is
  required to implement non-box regions without the use of .where() which would
  clash with skipna=False used as default in the metrics. The way this is
  implemented is by multiplying the input weights with a boolean weight dataset.

  Since sometimes the dataset and sometimes the weights are modified, these must
  be used together, most likely insice the _spatial_average function defined in
  metrics.py.
  """

  def apply(
      self, dataset: xr.Dataset, weights: xr.DataArray, **kwargs
  ) -> tuple[xr.Dataset, xr.DataArray]:
    """Apply region selection to dataset and/or weights.

    Args:
      dataset: Spatial metric, i.e. RMSE
      weights: Weights dataset, i.e. latitude weights
      **kwargs: Arguments for specific regions

    Returns:
      dataset: Potentially modified (sliced) dataset.
      weights: Potentially modified weights data array, to be used in
      combination with dataset, e.g. in _spatial_average().
    """
    raise NotImplementedError


@dataclasses.dataclass
class SliceRegion(Region):
  """Latitude-longitude box selection.

  Attributes:
    lat_slice: One or more latitude slices to be included in evaluation.
    lon_slice: One or more longitude slices to be included in evaluation.
    above_ground_climatology: (Optional) Mask for above ground regions with
      level dimension.
  """

  lat_slice: t.Optional[t.Union[slice, list[slice]]] = dataclasses.field(
      default_factory=lambda: slice(None, None)
  )
  lon_slice: t.Optional[t.Union[slice, list[slice]]] = dataclasses.field(
      default_factory=lambda: slice(None, None)
  )
  above_ground_climatology: t.Optional[xr.DataArray] = None

  def apply(  # pytype: disable=signature-mismatch
      self, dataset: xr.Dataset, weights: xr.DataArray
  ) -> tuple[xr.Dataset, xr.DataArray]:
    """Returns dataset sliced according to lat/lon_sliceparameters."""
    lats = (
        self.lat_slice if isinstance(self.lat_slice, list) else [self.lat_slice]
    )
    lons = (
        self.lon_slice if isinstance(self.lon_slice, list) else [self.lon_slice]
    )

    lats = xr.concat(
        [dataset.latitude.sel(latitude=s) for s in lats], dim='latitude'
    )
    lons = xr.concat(
        [dataset.longitude.sel(longitude=s) for s in lons], dim='longitude'
    )

    if self.above_ground_climatology is not None:
      time_selection = dict(dayofyear=dataset['valid_time'].dt.dayofyear)
      if 'hour' in set(self.above_ground_climatology.coords):
        time_selection['hour'] = dataset['valid_time'].dt.hour
      above_ground_weights = self.above_ground_climatology.sel(
          time_selection
      ).compute()

      # Above ground weights depend on level
      # Need to explicitly add weights only for 3D variables
      if isinstance(dataset, xr.Dataset):
        above_ground_weights_per_var = xr.Dataset()
        for v in dataset:
          if 'level' in dataset[v].dims:
            above_ground_weights_per_var[v] = above_ground_weights
          else:  # No weights necessary for 2D variables
            above_ground_weights_per_var[v] = xr.ones_like(dataset[v])
      else:
        if 'level' in dataset.dims:
          above_ground_weights_per_var = above_ground_weights
        else:  # No weights necessary for 2D variables
          above_ground_weights_per_var = xr.ones_like(dataset)
      weights = weights * above_ground_weights_per_var

    weight_indexers = {}
    if 'latitude' in weights.dims:
      weight_indexers['latitude'] = lats
    if 'longitude' in weights.dims:
      weight_indexers['longitude'] = lons
    return (
        dataset.sel(latitude=lats, longitude=lons),
        weights.sel(weight_indexers),
    )


@dataclasses.dataclass
class ExtraTropicalRegion(Region):
  """Latitude-longitude box selection."""

  threshold_lat: t.Optional[float] = 20

  def apply(  # pytype: disable=signature-mismatch
      self, dataset: xr.Dataset, weights: xr.DataArray, **kwargs
  ) -> tuple[xr.Dataset, xr.DataArray]:
    """Returns weights multiplied with a boolean mask to exclude tropics."""
    region_weights = (np.abs(dataset.latitude) >= 20).astype(float)
    return dataset, weights * region_weights


@dataclasses.dataclass
class LandRegion(Region):
  """Selects land grid point.

  Attributes:
    land_sea_mask: DataArray containing land sea mask in corresponding
      resolution.
    threshold: If given (between 0 and 1), threshold the land sea mask and
      convert to a boolean mask.
  """

  land_sea_mask: xr.DataArray
  threshold: t.Optional[float] = None

  def apply(  # pytype: disable=signature-mismatch
      self, dataset: xr.Dataset, weights: xr.DataArray, **kwargs
  ) -> tuple[xr.Dataset, xr.DataArray]:
    """Returns weights multiplied with a boolean land mask."""
    land_weights = self.land_sea_mask
    # Make sure lsm has same dtype for lat/lon
    land_weights = land_weights.assign_coords(
        latitude=land_weights.latitude.astype(dataset.latitude.dtype),
        longitude=land_weights.longitude.astype(dataset.longitude.dtype),
    )
    if self.threshold is not None:
      land_weights = (land_weights > self.threshold).astype(float)
    return dataset, weights * land_weights


@dataclasses.dataclass
class CombinedRegion(Region):
  """Sequentially applies regions selections.

  Allows for combination of e.g. SliceRegion and LandRegion.

  Attributes:
    regions: List of Region instances
  """

  regions: list[Region] = dataclasses.field(default_factory=list)

  def apply(  # pytype: disable=signature-mismatch
      self, dataset: xr.Dataset, weights: xr.DataArray, **kwargs
  ) -> tuple[xr.Dataset, xr.DataArray]:
    for region in self.regions:
      dataset, weights = region.apply(dataset, weights, **kwargs)
    return dataset, weights
