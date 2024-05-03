(cli)=
# Command line scripts

The `scripts/` directory contains a number of command line scripts for processing data and evaluating forecasts. All scripts are based on [Xarray-Beam](https://xarray-beam.readthedocs.io/en/latest/) to ensure scalability for large input files. The scripts can be run locally or on the cloud, e.g. using DataFlow (see [this guide](dataflow)).

(evaluation-cli)=
## Evaluation
Main evaluation script. To reproduce the official WeatherBench 2 evaluation, follow [these commands](official-evaluation). The results files for the baseline models can be found [here](https://console.cloud.google.com/storage/browser/weatherbench2/results).

```
usage: evaluate.py [-h] 
                         [--forecast_path FORECAST_PATH] 
                         [--obs_path OBS_PATH]
                         [--climatology_path CLIMATOLOGY_PATH] 
                         [--by_init] 
                         [--evaluate_persistence] 
                         [--evaluate_climatology] 
                         [--evaluate_probabilistic_climatology] 
                         [--probabilistic_climatology_start_year PROBABILISTIC_CLIMATOLOGY_START_YEAR] 
                         [--probabilistic_climatology_end_year PROBABILISTIC_CLIMATOLOGY_END_YEAR] 
                         [--probabilistic_climatology_hour_interval PROBABILISTIC_CLIMATOLOGY_HOUR_INTERVAL]
                         [--regions REGIONS]
                         [--lsm_dataset LSM_DATASET]
                         [--compute_seeps]
                         [--eval_configs EVAL_CONFIGS]
                         [--ensemble_dim ENSEMBLE_DIM]
                         [--rename_variables RENAME_VARIABLES]
                         [--pressure_level_suffixes]
                         [--levels LEVELS]
                         [--variables VARIABLES]
                         [--derived_variables DERIVED_VARIABLES]
                         [--time_start TIME_START]
                         [--time_stop TIME_STOP]
                         [--output_dir OUTPUT_DIR]
                         [--output_file_prefix OUTPUT_FILE_PREFIX]
                         [--input_chunks INPUT_CHUNKS]
                         [--use_beam]
                         [--beam_runner BEAM_RUNNER]
                         [--fanout FANOUT]
```

_Command options_:

* `--forecast_path`: (required) Path to forecasts to evaluate in Zarr format.
* `--obs_path`: (required) Path to ground-truth to evaluate in Zarr format.
* `--climatology_path`: Path to climatology. Used to compute e.g. ACC.
* `--by_init`: Specifies whether forecasts are in by-init or by-valid format.
* `--evaluate_persistence`: Evaluate persistence forecast, i.e. forecast at t=0
* `--evaluate_climatology`: Evaluate climatology forecast specified in climatology path
* `--evaluate_probabilistic_climatology`: Evaluate probabilistic climatology,
      derived from using each year of the ground-truth dataset as a member
* `--probabilistic_climatology_start_year`: First year of ground-truth to use
      for probabilistic climatology
* `--probabilistic_climatology_end_year`: Last year of ground-truth to use
      for probabilistic climatology
* `--probabilistic_climatology_hour_interval`: Hour interval to compute 
      probabilistic climatology. Default: 6
* `--regions`: Comma delimited list of predefined regions to evaluate. "all" for all predefined regions.
* `--lsm_dataset`: Dataset containing land-sea-mask at same resolution of datasets to be evaluated. Required if region with land-sea-mask is picked. If None, defaults to observation dataset.
* `--compute_seeps`: Compute SEEPS for total_precipitation.
* `--eval_configs`: Comma-separated list of evaluation configs to run. See details below. Default: `deterministic`
* `--ensemble_dim`: Ensemble dimension name for ensemble metrics. Default: `number`.
* `--rename_variables`: Dictionary of variable to rename to standard names. E.g. {"2t": "2m_temperature"}
* `--pressure_level_suffixes`: 'Decode pressure levels as suffixes in forecast file. E.g. temperature_850'
* `--levels`: Comma delimited list of pressure levels to select for evaluation. Default: `500,700,850`
* `--variables`: Comma delimited list of variables to select from weather. Default: `geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,mean_sea_level_pressure`
* `--derived_variables`: Comma delimited list of derived variables to dynamically compute during evaluation. Default: `None`
* `--time_start`: ISO 8601 timestamp (inclusive) at which to start evaluation. Default: `2020-01-01`
* `--time_stop`: ISO 8601 timestamp (inclusive) at which to stop evaluation. Default: `2020-12-31`
* `--output_dir`: Directory in which to save evaluation results in netCDF format.
* `--output_file_prefix`: Prefix of results filename. If "_" or "-" is desired, please add specifically here.
* `--input_chunks`: chunk sizes overriding input chunks to use for loading forecast and observation data. By default, omitted dimension names are loaded in one chunk. Default: `time=1`
* `--use_beam`: Run evaluation pipeline as beam pipeline. If False, run in memory.
* `--beam_runner`: Beam runner
* `--fanout`: Beam CombineFn fanout. Might be required for large dataset. Default: `None`

[Predefined evaluation configs](https://github.com/google-research/weatherbench2/blob/main/scripts/evaluate.py#L389)


*Example*

```bash
python evaluate.py \
  --forecast_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_with_poles_conservative.zarr \
  --obs_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
  --climatology_path=gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_with_poles_conservative.zarr \
  --output_dir=./ \
  --output_file_prefix=hres_vs_era_2020_ \
  --input_chunks=init_time=1 \
  --eval_configs=deterministic,deterministic_spatial,deterministic_temporal \
  --time_start=2020-01-01 \
  --time_stop=2020-12-31 \
  --variables=geopotential,temperature,u_component_of_wind,v_component_of_wind,specific_humidity,2m_temperature,10m_u_component_of_wind,10m_v_component_of_wind,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr,10m_wind_speed,wind_speed
```


## Compute climatology

This scripts computes a day-of-year, hour-of-day climatology with optional smoothing following the methodology proposed [here](https://www.ecmwf.int/en/elibrary/75101-scale-dependent-verification-ensemble-forecasts). 

```
usage: compute_climatology.py [-h] 
                                  [--input_path INPUT_PATH] 
                                  [--output_path OUTPUT_PATH]
                                  [--frequency FREQUENCY]
                                  [--hour_interval HOUR_INTERVAL] 
                                  [--window_size WINDOW_SIZE] 
                                  [--start_year START_YEAR] 
                                  [--end_year END_YEAR] 
                                  [--working_chunks WORKING_CHUNKS] 
                                  [--rechunk_itemsize RECHUNK_ITEMSIZE] 
                                  [--output_chunks OUTPUT_CHUNKS]
                                  [--statistics STATISTICS]
                                  [--add_statistic_suffix]
                                  [--method METHOD]
                                  [--seeps_dry_threshold_mm SEEPS_DRY_THRESHOLD_MM]
                                  [--runner RUNNER]

```

_Command options_:

* `--input_path`: (required) Input Zarr path
* `--output_path`: (required) Output Zarr path
* `--frequency`: Frequency of the computed climatology. "hourly": Compute the climatology per day of year and hour of day. "daily": Compute the climatology per day of year.
* `--hour_interval`: Which intervals to compute hourly climatology for. Default: `1`
* `--window_size`: Window size in days to average over. Default: `61` 
* `--start_year`: Inclusive start year of climatology. Default: `1990`
* `--end_year`: Inclusive end year of climatology. Default: `2019`
* `--working_chunks`: Chunk sizes overriding input chunks to use for computing climatology, e.g., "longitude=10,latitude=10".
* `--rechunk_itemsize`: Itemsize for rechunking.. Default: `4`
* `--output_chunks`: Chunk sizes overriding input chunks to use for storing climatology.
* `--statistics`: List of statistics to compute from "mean", "std", "seeps". Default: `["mean"]`
* `--add_statistic_suffix`: Add suffix of statistic to variable name. Required for >1 statistic.
* `--method`: Computation method to use. "explicit": Stack years first, apply rolling and then compute weighted statistic over (year, rolling_window). "fast": Compute statistic over day-of-year first and then apply weighted smoothing. Mathematically equivalent for mean but different for nonlinear statistics. Default: `explicit`
* `--seeps_dry_threshold_mm`: Dict defining dry threshold for SEEPS quantile computation for each precipitation variable. In mm. Default: `"{'total_precipitation_24hr':0.25, 'total_precipitation_6hr':0.1}"`
* `--runner`: Beam runner. Use `DirectRunner` for local execution.

*Example*

```bash
python compute_climatology.py \
  --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
  --output_path=PATH \
  --working_chunks=level=13,longitude=4,latitude=4 \
  --output_chunks=level=13,hour=4 \
  --start_year=1990 \
  --end_year=2019 \
  --hour_interval=6 \
  --statistics=mean,seeps
```

## Compute derived variables

Computes derived variables, adds them to the original dataset and saves it as a new file. See derived_variables.py for a list of available derived variables.

```
usage: compute_derived_variables.py [-h] 
                                        [--input_path INPUT_PATH] 
                                        [--output_path OUTPUT_PATH]
                                        [--derived_variables DERIVED_VARIABLES]
                                        [--preexisting_variables_to_remove PREEXISTING_VARIABLES_TO_REMOVE]
                                        [--raw_tp_name RAW_TP_NAME] 
                                        [--rename_raw_tp_name] 
                                        [--rename_variables RENAME_VARIABLES]
                                        [--working_chunks WORKING_CHUNKS] 
                                        [--rechunk_itemsize RECHUNK_ITEMSIZE] 
                                        [--max_mem_gb MAX_MEM_GB] 
                                        [--runner RUNNER]
```

_Command options_:

* `--input_path`: (required) Input Zarr path
* `--output_path`: (required) Output Zarr path
* `--derived_variables`: Comma delimited list of derived variables to compute. By default, tries to compute all derived variables.
* `--preexisting_variables_to_remove`: Comma delimited list of variables to remove from the source data, if they exist. This is useful to allow for overriding source dataset variables with derived variables of the same name.
* `--raw_tp_name`: Raw name of total precipitation variables. Use "total_precipitation_6hr" for backwards compatibility. 
* `--rename_raw_tp_name`: Rename raw tp name to "total_precipitation".
* `--rename_variables`: Dictionary of variable to rename to standard names. E.g. {"2t":"2m_temperature"}
* `--working_chunks`: Chunk sizes overriding input chunks to use for computing aggregations e.g., "longitude=10,latitude=10". No need to add prediction_timedelta=-1, this is automatically added for aggregation variables. Default: `None`, i.e. input chunks
* `--rechunk_itemsize`: Itemsize for rechunking. Default: `4`
* `--max_mem_gb`: Max memory for rechunking in GB. Default: `1`
* `--runner`: Beam runner. Use `DirectRunner` for local execution.

*Example*

```bash
python compute_derived_variables.py \
  --input_path=gs://weatherbench2/datasets/hres/2016-2022-12h-6h-0p25deg-chunk-1.zarr/ \
  --output_path=PATH \
  --working_chunks=prediction_timedelta=-1 \
  --rename_raw_tp_name=True \
  --raw_tp_name=total_precipitation_6hr \
  --rechunk_itemsize=1 \
```

## Compute Spectra

Computes zonal energy spectra.

```
usage: compute_zonal_energy_spectrum.py [-h] 
                                            [--input_path INPUT_PATH] 
                                            [--output_path OUTPUT_PATH]
                                            [--base_variables BASE_VARIABLES] 
                                            [--time_dim TIME_DIM] 
                                            [--time_start TIME_START] 
                                            [--time_stop TIME_STOP] 
                                            [--levels LEVELS] 
                                            [--averaging_dims AVERAGING_DIMS]
                                            [--fanout FANOUT]
                                            [--runner RUNNER] 
```

_Command options_:

* `--input_path`: (required) Input Zarr path
* `--output_path`: (required) Output Zarr path
* `--base_variables`: Comma delimited list of variables in --input_path. Each variable VAR results in a VAR_zonal_power_spectrum entry in --output_path. Default: `geopotential,specific_humidity,temperature`
* `--time_dim`: Name for the time dimension to slice data on. Default: `time`
* `--time_start`: ISO 8601 timestamp (inclusive) at which to start evaluation. Default: `2020-01-01'`
* `--time_stop`: ISO 8601 timestamp (inclusive) at which to stop evaluation. Default: `2020-12-31`
* `--levels`: Comma delimited list of pressure levels to compute spectra on. If empty, compute on all levels of --input_path. Default: `500,700,850`
* `--averaging_dims`: Comma delimited list of variables to average over. If empty, do not average. Default: `time`
* `--fanout`: Beam CombineFn fanout. Might be required for large dataset.
* `--runner`: Beam runner. Use `DirectRunner` for local execution.

*Example*

```bash
python compute_zonal_energy_spectrum.py \
  --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr  \
  --output_path=PATH \
  --time_start=2020 \
  --time_stop=2020 \
  --base_variables=geopotential,specific_humidity,temperature,u_component_of_wind,v_component_of_wind,wind_speed,10m_u_component_of_wind,10m_v_component_of_wind,10m_wind_speed,2m_temperature,mean_sea_level_pressure,total_precipitation_6hr,total_precipitation_24hr
```

## Compute ensemble mean

To use the ensemble mean in deterministic evaluation, we first must compute the mean and save it in a separate file.

```
usage: compute_ensemble_mean.py [-h] 
                                    [--input_path INPUT_PATH] 
                                    [--output_path OUTPUT_PATH]
                                    [--time_dim TIME_DIM] 
                                    [--time_start TIME_START] 
                                    [--time_stop TIME_STOP] 
                                    [--realization_name REALIZATION_NAME]
                                    [--runner RUNNER] 
```

_Command options_:

* `--input_path`: (required) Input Zarr path
* `--output_path`: (required) Output Zarr path
* `--time_dim`: Name for the time dimension to slice data on. Default: `time`
* `--time_start`: ISO 8601 timestamp (inclusive) at which to start evaluation. Default: `2020-01-01'`
* `--time_stop`: ISO 8601 timestamp (inclusive) at which to stop evaluation. Default: `2020-12-31`
* `--realization_name`: Name of realization/member/number dimension. Default: `realization`
* `--runner`: Beam runner. Use `DirectRunner` for local execution.

*Example*

```bash
python compute_ensemble_mean.py -- \
  --input_path=gs://weatherbench2/datasets/ens/2018-64x32_equiangular_with_poles_conservative.zarr \
  --output_path=PATH \
  --realization_name=number
```

## Compute statistical moments

This script computes statistical moments of every variable in the input
dataset per vertical level. Expectations are taken with respect to longitude,
latitude, and time. 

```
usage: compute_statistical_moments.py [-h] 
                                  [--input_path INPUT_PATH] 
                                  [--output_path OUTPUT_PATH]
                                  [--start_year START_YEAR] 
                                  [--end_year END_YEAR] 
                                  [--rechunk_itemsize RECHUNK_ITEMSIZE] 
                                  [--runner BEAM_RUNNER]

```

_Command options_:

* `--input_path`: (required) Input Zarr path
* `--output_path`: (required) Output Zarr path
* `--start_year`: Inclusive start year of climatology. Default: `1990`
* `--end_year`: Inclusive end year of climatology. Default: `2020`
* `--rechunk_itemsize`: Itemsize for rechunking.. Default: `4`
* `--runner`: Beam runner. Use `DirectRunner` for local execution.

*Example*

```bash
python compute_statistical_moments.py \
  --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
  --output_path=PATH \
  --start_year=1990 \
  --end_year=2019
```

(regridding)=
## Regrid
Only rectilinear grids (one dimensional lat/lon coordinates) on the input Zarr file are supported, but irregular spacing is OK.

```
usage: regrid.py [-h] 
                 [--input_path INPUT_PATH]
                 [--output_path OUTPUT_PATH]
                 [--output_chunks OUTPUT_CHUNKS]
                 [--latitude_nodes LATITUDE_NODES]
                 [--longitude_nodes LONGITUDE_NODES]
                 [--latitude_spacing LATITUDE_SPACING]
                 [--regridding_method REGRIDDING_METHOD]
                 [--latitude_name LATITUDE_NAME]
                 [--longitude_name LONGITUDE_NAME]
                 [--runner RUNNER]
```

_Command options_:

* `--input_path`: (required) Input Zarr path
* `--output_path`: (required) Output Zarr path
* `--output_chunks`: Desired chunking of output zarr
* `--latitude_nodes`: Number of desired latitude nodes
* `--longitude_nodes`: Number of desired longitude nodes
* `--latitude_spacing`: Desired latitude spacing from `equiangular_with_poles` or `equiangular_without_poles`. Default: `equiangular_with_poles`
* `--regridding_method`: Regridding method from `nearest`, `bilinear` or `conservative`. Default: `conservative`
* `--latitude_name`: Name of latitude dimension in dataset. Default: `latitude`
* `--longitude_name`: Name of longitude dimension in dataset. Default: `longitude`
* `--runner`: Beam runner. Use `DirectRunner` for local execution.

*Example*

```bash
python regrid.py \
  --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr \
  --output_path=PATH \
  --output_chunks="time=100" \
  --latitude_nodes=64 \
  --longitude_nodes=33 \
  --latitude_spacing=equiangular_with_poles \
  --regridding_method=conservative
```

(compute_averages)=
## Compute averages
Computes average over dimensions of a forecast dataset.

```
usage: compute_averages.py [-h] 
                 [--input_path INPUT_PATH]
                 [--output_path OUTPUT_PATH]
                 [--output_chunks OUTPUT_CHUNKS]
                 [--time_dim TIME_DIM] 
                 [--time_start TIME_START] 
                 [--time_stop TIME_STOP] 
                 [--variables VARIABLES]
                 [--fanout FANOUT]
                 [--runner RUNNER]
```

_Command options_:

* `--input_path`: (required) Input Zarr path
* `--output_path`: (required) Output Zarr path
* `--time_dim`: Name for the time dimension to slice data on. Default: `time`
* `--time_start`: ISO 8601 timestamp (inclusive) at which to start evaluation. Default: `2020-01-01'`
* `--time_stop`: ISO 8601 timestamp (inclusive) at which to stop evaluation. Default: `2020-12-31`
* `--variables`: Comma delimited list of data variables to include in output. If empty, compute on all data_vars of --input_path.
* `--fanout`: Beam CombineFn fanout. Might be required for large dataset.
* `--runner`: Beam runner. Use `DirectRunner` for local execution.

*Example*

```bash
python compute_averages.py \
    --input_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr \
    --output_path=gs://$BUCKET/datasets/era5/$USER/temperature-vertical-profile.zarr \
    --runner=DataflowRunner \
    -- \
    --project=$PROJECT \
    --averaging_dims=time,longitude \
    --variables=temperature \
    --temp_location=gs://$BUCKET/tmp/ \
    --setup_file=./setup.py \
    --requirements_file=./scripts/dataflow-requirements.txt \
    --job_name=compute-vertical-profile-$USER
```

(resample_daily)=
## Resample daily
Computes average over dimensions of a forecast dataset.

```
usage: resample_daily.py [-h] 
                 [--input_path INPUT_PATH]
                 [--output_path OUTPUT_PATH]
                 [--method METHOD]
                 [--period PERIOD]
                 [--statistics STATISTICS]
                 [--add_statistic_suffix]
                 [--num_threads NUM_THREADS]
                 [--start_year START_YEAR]
                 [--end_year END_YEAR]
                 [--working_chunks WORKING_CHUNKS]
                 [--beam_runner BEAM_RUNNER]
```

_Command options_:

* `--input_path`: (required) Input Zarr path
* `--output_path`: (required) Output Zarr path
* `--method`: resample or roll
* `--period`: int + d or w
* `--statistics`: Output resampled time statistics, from "mean", "min", or "max".
* `--add_statistic_suffix`: Add suffix of statistic to variable name. Required for >1 statistic.
* `--num_threads`: Number of chunks to load in parallel per worker.
* `--start_year`: Start year (inclusive).
* `--end_year`: End year (inclusive).
* `--working_chunks`: Spatial chunk sizes to use during time downsampling, e.g., "longitude=10,latitude=10". They may not include "time".
* `--beam_runner`: Beam runner. Use `DirectRunner` for local execution.

## Slice dataset
Slices a Zarr file containing an xarray Dataset, using `.sel` and `.isel`.

```
usage: slice_dataset.py [-h]
                 [--input_path INPUT_PATH]
                 [--output_path OUTPUT_PATH]
                 [--sel SEL]
                 [--isel ISEL]
                 [--drop_variables DROP_VARIABLES]
                 [--keep_variables KEEP_VARIABLES]
                 [--output_chunks OUTPUT_CHUNKS]
                 [--runner RUNNER]

```

_Command options_:

* `--input_path`: (required) Input Zarr path
* `--output_path`: (required) Output Zarr path
* `--sel`: Selection criteria, to pass to `xarray.Dataset.sel`. Passed as
           key=value pairs, with key = `VARNAME_{start,stop,step}`
* `--isel`: Selection criteria, to pass to `xarray.Dataset.isel`. Passed as
            key=value pairs, with key = `VARNAME_{start,stop,step}`
* `--drop_variables`: Comma delimited list of variables to drop. If empty, drop
                      no variables.
* `--keep_variables`: Comma delimited list of variables to keep. If empty, use
                      `--drop_variables` to determine which variables to keep.
* `--output_chunks`: Chunk sizes overriding input chunks.
* `--runner`: Beam runner. Use `DirectRunner` for local execution.

*Example*

```bash
python slice_dataset.py -- \
  --input_path=gs://weatherbench2/datasets/ens/2018-64x32_equiangular_with_poles_conservative.zarr \
  --output_path=PATH \
  --sel="prediction_timedelta_stop=15 days,latitude_start=-33.33,latitude_stop=33.33" \
  --isel="longitude_start=0,longitude_stop=180,longitude_step=40" \
  --keep_variables=geopotential,temperature
```

## Expand climatology

`expand_climatology.py` takes a climatology dataset and expands it into a forecast-like format (`init_time` + `lead_time`). This is not currently used as `evaluation.py` is able to do this on-the-fly, reducing the number of intermediate steps. We still included the script here in case others find it useful. 

## Init to valid time conversion

`compute_init_to_valid_time.py` converts a forecasts in init-time convention to valid-time convention. Since currently, we do all evaluation in the init-time format, this script is not used. 
