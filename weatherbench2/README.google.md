# WeatherBench2

WARNING: WeatherBench 2 and this documentation are currently under active development.

## Data guide

The exact commands to compile all datasets below are stored as bash scripts here: `google3/experimental/users/srasp/wb2_superscripts`

### Ground truth datasets

#### ERA5
Location: `/cns/od-d/home/attractor/weatherbench/datasets/era5/`

The raw ERA5 files were copied from DeepMind (their v2).

Raw file: 

`1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2`

Downsampled (6hr adn 13 levels): 

`1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2`

With derived variables (e.g. wind speed, precipitation accumulation): 

`1959-2022-6h-1440x721.zarr`

Regridded:

`1959-2022-6h-512x256_equiangular_conservative.zarr`

`1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr`

`1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr`

`1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr`


#### IFS forecast at t=0

This is not equivalent to the analysis dataset but we will use it here as analysis.

Location: `/cns/od-d/home/attractor/weatherbench/datasets/hres_t0/`

Raw data (copied from DeepMind):

`forecasts_t0_analysis_2016-2022-6h-0p25deg-chunk-1.zarr`


With derived variables: 

`2016-2022-6h-1440x721.zarr/`

Regridded:

`2016-2022-6h-512x256_equiangular_conservative.zarr`   
`2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr`   
`2016-2022-6h-128x64_equiangular_with_poles_conservative.zarr`   
`2016-2022-6h-64x32_equiangular_with_poles_conservative.zarr`


### Climatology

There are climatology versions computed from 1990 to 2017 and from 1990 to 2019. 

Location: `/cns/od-d/home/attractor/weatherbench/datasets/era5-hourly-climatology/`

All files:

`1990-2017_6h_1440x721.zarr`   
`1990-2017_6h_512x256_equiangular_conservative.zarr`   
`1990-2017_6h_240x121_equiangular_with_poles_conservative.zarr`   
`1990-2017_6h_128x64_equiangular_with_poles_conservative.zarr`   
`1990-2017_6h_64x32_equiangular_with_poles_conservative.zarr`   
`1990-2019_6h_1440x721.zarr`   
`1990-2019_6h_512x256_equiangular_conservative.zarr`   
`1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr`   
`1990-2019_6h_128x64_equiangular_with_poles_conservative.zarr`   
`1990-2019_6h_64x32_equiangular_with_poles_conservative.zarr`   


### Forecast datasets

#### IFS HRES

Location: `/cns/od-d/home/attractor/weatherbench/datasets/hres/`

Raw data:

`2016-2022-12h-6h-0p25deg-chunk-1.zarr`

With derived variables:

`2016-2022-0012-1440x721.zarr`

Regridded:

`2016-2022-0012-512x256_equiangular_conservative.zarr`    
`2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr`   
`2016-2022-0012-128x64_equiangular_with_poles_conservative.zarr`   
`2016-2022-0012-64x32_equiangular_with_poles_conservative.zarr`   


#### IFS ENS

Location: `/cns/oz-d/home/attractor/weatherbench/datasets/ens/`

Data for `2018` is also available. Just replace `2020`.

Raw data: `2020-0p25.zarr`

With derived variables:

`2018-1440x721.zarr`

Regridded:

`2020-512x256_equiangular_conservative.zarr`   
`2020-240x121_equiangular_with_poles_conservative.zarr`   
`2020-128x64_equiangular_with_poles_conservative.zarr`    
`2020-64x32_equiangular_with_poles_conservative.zarr`   

#### IFS ENS Mean

Location: `/cns/oz-d/home/attractor/weatherbench/datasets/ens/`

Same paths as above with `_mean`.


#### GraphCast

Location: `/cns/oa-d/home/attractor/weatherbench/datasets/graphcast/<year>`

GraphCast has models trained up to different years, from <2016 to <2021. The `<year>` corresponds to the first test year, i.e. `2016` corresponds to the <2016 model. Note: Only 2020 is fully processed right now.

Raw data: 

`date_range_2019-11-16_2021-02-01_12_hours.zarr/`

With derived variables:

`date_range_2019-11-16_2021-02-01_12_hours_derived.zarr`

Regridded:

`date_range_2019-11-16_2021-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr`   
`date_range_2019-11-16_2021-02-01_12_hours-128x64_equiangular_with_poles_conservative.zarr`   
`date_range_2019-11-16_2021-02-01_12_hours-64x32_equiangular_with_poles_conservative.zarr`   

For 2018 a 0.7 degree file also exists:

`date_range_2017-11-16_2019-02-01_12_hours-512x256_equiangular_conservative.zarr`

#### Pangu

Location: `/cns/oz-d/home/attractor/weatherbench/datasets/pangu/`

Raw file:

`pangu_2020.zarr`

With derived variables:

`2020-1440x721.zarr`

Regridded:

`2020-512x256_equiangular_conservative.zarr`   
`2020-240x121_equiangular_with_poles_conservative.zarr`   
`2020-128x64_equiangular_with_poles_conservative.zarr`   
`2020-64x32_equiangular_with_poles_conservative.zarr`   


### Evaluation results

Evaluation output produced by `wb2_evaluation.py` is stored here for different resolutions and years: 

`/cns/od-d/home/attractor/weatherbench/results/`




## Evaluation pipeline

The WB2 evaluation pipeline works by taking a forecast file, a truth file and, optionally for computing climatological baselines or the ACC, a pre-computed climatology file. It will then compute the desired scores and save them following some configuration arguments. Below is a detailed guide on how to use the evaluation pipeline.

Example pipelines can be found in `scripts/wb2_evaluation.py`. 

### 1. Input files

#### Forecast file

The forecast file should be a Zarr file with the following coordinates:

- `time` [`np.datetime64`]: Time at forecast is valid.
- `lead_time` [`np.timedelta64`]: Lead time
- `latitude` [float]: For WB2: 0.25 or 1.5 degree grid spacing including the poles
- `longitude` [float]: For WB2: 0.25 or 1.5 degree grid spacing starting at 0
- `level` [hPa]: Pressure levels (optional)

Here are some example datasets: /cns/od-d/home/attractor/weatherbench/datasets/hres/by-valid-time/

Note 1: Here we follow the valid time convention. This means that the `time` dimension indicates the time at which the forecast is valid. This is opposed to the init time convention that is used, e.g. in raw ECMWF forecast files. The reason for this is that it simplifies downstream computation of the scores. To convert from init to valid time convention, see `scripts/wb2_init_to_valid_time.py`

Note 2: `latitude` and `longitude` dimensions can be freely chosen as long as they correspond to the truth/reanalysis file. For regridding, see google3/research/simulation/whirl/gcm/pipelines/ecmwf_regrid.py

#### Obs/reanalysis file

This is the corresponding observation/reanalysis file, e.g. from ERA5, also in Zarr format. See for example /namespace/gas/primary/whirl/datasets/era5/shoyer/ for a range of regridded ERA5 files. 

#### Climatology file

For computing climatological baselines or scores that require climatology like the ACC, the evaluation pipeline requires a pre-computed climatology file. See `scripts/compute_climatology.py` or `scripts/compute_climatology_beam.py` for scripts to compute these climatology files. Here are some that we already computed: /cns/od-d/home/attractor/weatherbench/datasets/era5-hourly-climatology/

#### Data config

The file paths are defined in a Paths config object, alongside an output directory:

```
from weatherbench2.config import Selection, Paths, DataConfig, EvalConfig
paths = Paths(
    forecast=FORECAST_PATH.value,
    obs=OBS_PATH.value,
    output_dir=OUTPUT_PATH.value,
)
```

In addition, we specify a Selection object that selects the variables and time period to be evaluated.

```

selection = Selection(
    variables=[
        'geopotential',
        'temperature',
        'u_component_of_wind',
        'v_component_of_wind',
        'specific_humidity',
        '2m_temperature
    ],
    levels=[500, 700, 850],
    time_slice=slice(TIME_START.value, TIME_STOP.value),
)
```

Together they make up the DataConfig:

```
data_config = DataConfig(selection=selection, paths=paths)
```

### 2. Evaluation Config

Next, we can defined which evaluation we want to run. To do so, we can define a dictionary of EvalConfigs, each of which will be evaluated separately and saved to a different file.

EvalConfig contains the metrics objects, defined in `metrics.py`. A special not one ACC, which requires a climatology. Here the climatology has to be opened explicitly and passed as an init parameter to the ACC class (see example below).

```
climatology = xr.open_zarr(climatology_path, chunks=None)
eval_configs = {
  'deterministic': EvalConfig(
      metrics={'rmse': RMSE(time_mean_last=False), 'acc': ACC(climatology=climatology)},
      against_analysis=False,
  ),
  'deterministic_vs_analysis': EvalConfig(
      metrics={'rmse': RMSE(time_mean_last=False), 'acc': ACC(climatology=climatology)},
      against_analysis=True
}
```

The evaluation configs also have an option to evaluate particular regions, such as a geographical lat-lon box. These are defined as region objects defined in `regions.py`. A simple box region can be defined by a `SliceRegion`. Currently, the only other implemented option is `ExtraTropicalRegion` but a land-sea mask will be added. See this example for an example of how to use regions. 

```
regions = {
    'global': SliceRegion(),
    'tropics': SliceRegion(lat_slice=slice(-20, 20)),
    'extra-tropics': ExtraTropicalRegion(),
}

eval_configs = {
  'deterministic': EvalConfig(
      metrics={'rmse': RMSE(time_mean_last=False), 'acc': ACC(climatology=climatology)},
      against_analysis=False,
      regions=regions
  )
}
```

All given regions will be evaluated separately and saved as an additional dimension in the results file.

### 3. Run pipeline

With the configs defined, we can run the pipeline. There are two ways to do this: a) in memory and b) as an xarray-beam pipeline for larger datasets.

In-memory evaluation is suitable for smaller files, e.g. 64x32 resolution.

```
from weatherbench2 import evaluation
evaluation.evaluate_in_memory(data_config, eval_configs)
```

For larger files, computations can be split across chunks using beam. The chunks can be specified alongside the beam runner.

```
evaluation.evaluate_with_beam(
    data_config,
    eval_configs,
    runner=BEAM_RUNNER.value,
    input_chunks=input_chunks,
)
```

<!-- TODO(srasp): Some instructions on chunk size -->


### 4. Output

The evaluation function will compute the metrics and save them in an output NetCDF file. Here is an example.

```
<xarray.Dataset>
Dimensions:              (lead_time: 21, region: 3, level: 3, metric: 2)
Coordinates:
  * lead_time            (lead_time) timedelta64[ns] 0 days 00:00:00 ... 10 d...
  * region               (region) object 'global' 'tropics' 'extra-tropics'
  * level                (level) int32 500 700 850
  * metric               (metric) object 'rmse' 'acc'
Data variables:
    geopotential         (metric, region, lead_time, level) float64 22.76 ......
    specific_humidity    (metric, region, lead_time, level) float64 0.0001047...
    temperature          (metric, region, lead_time, level) float64 0.224 ......
    u_component_of_wind  (metric, region, lead_time, level) float64 0.6683 .....
    2m_temperature       (metric, region, lead_time) float64 0.6337 ... 0.4817
```

## Adding new metrics

To implement a new metric, you can simply add a new Metric class in `metrics.py`. 

As an example, see the MSE class below. Each new class must define a `compute_chunk` method, which computes the metric for a given chunk (but without the time average which is defined in the `compute` method of the parent class). `_spatial_average` is a function that computes a spatial average with a latitude weight and also applies the region selection.

```
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
```


# Visualization pipeline

See `scripts/wb2_visualization.py` for a working example. This is likely to change still.


